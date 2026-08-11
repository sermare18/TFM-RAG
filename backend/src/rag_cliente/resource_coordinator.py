"""Coordinador FIFO de los recursos de modelos residentes en memoria."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Literal

ResourceName = Literal["parser_bundle", "embeddings", "chat"]
WorkloadName = Literal["index", "query"]


@dataclass(slots=True)
class _Waiter:
    ticket: int
    resource: ResourceName
    workload: WorkloadName


class ResourceLease:
    """Lease idempotente que libera el recurso también ante excepciones."""

    def __init__(
        self,
        coordinator: "ResourceCoordinator",
        resource: ResourceName,
        lease_id: str,
    ) -> None:
        self._coordinator = coordinator
        self.resource = resource
        self.lease_id = lease_id
        self._released = False

    def __enter__(self) -> "ResourceLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._coordinator._release_resource(self.resource, self.lease_id)
        self._released = True


class IndexingLease:
    """Reserva un único trabajo de indexación de principio a fin."""

    def __init__(self, coordinator: "ResourceCoordinator", lease_id: str) -> None:
        self._coordinator = coordinator
        self.lease_id = lease_id
        self._released = False

    def __enter__(self) -> "IndexingLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._coordinator._release_indexing(self.lease_id)
        self._released = True


class ResourceCoordinator:
    """Serializa parser, embeddings y chat sin starvation.

    Los tres recursos comparten memoria y, por tanto, solo uno puede estar
    activo. La cola conserva FIFO. Durante una indexación se posponen trabajos
    de consulta; se permite que las solicitudes del propio indexado avancen para
    evitar un bloqueo entre el parser y embeddings.
    """

    _VALID_RESOURCES = {"parser_bundle", "embeddings", "chat"}

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._waiters: deque[_Waiter] = deque()
        self._next_ticket = 0
        self._active_resource: ResourceName | None = None
        self._active_resource_lease: str | None = None

        self._index_waiters: deque[tuple[int, str]] = deque()
        self._active_index_lease: str | None = None

    def acquire(
        self,
        resource: ResourceName,
        *,
        workload: WorkloadName = "query",
        timeout: float | None = None,
    ) -> ResourceLease:
        if resource not in self._VALID_RESOURCES:
            raise ValueError(f"Recurso desconocido: {resource}")
        if workload not in {"index", "query"}:
            raise ValueError(f"Tipo de trabajo desconocido: {workload}")

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            waiter = _Waiter(self._next_ticket, resource, workload)
            self._next_ticket += 1
            self._waiters.append(waiter)

            try:
                while not self._can_grant(waiter):
                    remaining = self._remaining(deadline)
                    if remaining == 0:
                        raise TimeoutError(
                            f"Timeout esperando el recurso FIFO '{resource}'"
                        )
                    self._condition.wait(remaining)

                self._waiters.remove(waiter)
                lease_id = uuid.uuid4().hex
                self._active_resource = resource
                self._active_resource_lease = lease_id
                return ResourceLease(self, resource, lease_id)
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                raise

    def acquire_indexing(self, *, timeout: float | None = None) -> IndexingLease:
        deadline = None if timeout is None else time.monotonic() + timeout
        lease_id = uuid.uuid4().hex
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            waiter = (ticket, lease_id)
            self._index_waiters.append(waiter)
            try:
                while not self._can_grant_indexing(waiter):
                    remaining = self._remaining(deadline)
                    if remaining == 0:
                        raise TimeoutError("Timeout esperando el slot único de indexación")
                    self._condition.wait(remaining)

                self._index_waiters.popleft()
                self._active_index_lease = lease_id
                self._condition.notify_all()
                return IndexingLease(self, lease_id)
            except BaseException:
                if waiter in self._index_waiters:
                    self._index_waiters.remove(waiter)
                    self._condition.notify_all()
                raise

    def _can_grant(self, waiter: _Waiter) -> bool:
        if self._active_resource is not None:
            return False

        # Ninguna consulta adelanta a una indexación ya admitida. Las peticiones
        # de indexado sí pueden saltar temporalmente consultas incompatibles;
        # esas consultas conservan su orden y se reanudan al finalizar el job.
        if self._active_index_lease is not None:
            if waiter.workload != "index":
                return False
            eligible = next(
                (item for item in self._waiters if item.workload == "index"),
                None,
            )
            return eligible is waiter

        if not self._waiters or self._waiters[0] is not waiter:
            return False
        oldest_index_ticket = self._index_waiters[0][0] if self._index_waiters else None
        return oldest_index_ticket is None or waiter.ticket < oldest_index_ticket

    def _can_grant_indexing(self, waiter: tuple[int, str]) -> bool:
        if self._active_index_lease is not None or self._active_resource is not None:
            return False
        if not self._index_waiters or self._index_waiters[0] != waiter:
            return False
        oldest_resource_ticket = self._waiters[0].ticket if self._waiters else None
        return oldest_resource_ticket is None or waiter[0] < oldest_resource_ticket

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _release_resource(self, resource: ResourceName, lease_id: str) -> None:
        with self._condition:
            if self._active_resource_lease != lease_id or self._active_resource != resource:
                raise RuntimeError("Intento de liberar un lease que no está activo")
            self._active_resource = None
            self._active_resource_lease = None
            self._condition.notify_all()

    def _release_indexing(self, lease_id: str) -> None:
        with self._condition:
            if self._active_index_lease != lease_id:
                raise RuntimeError("Intento de liberar un job de indexación ajeno")
            self._active_index_lease = None
            self._condition.notify_all()

    def snapshot(self) -> dict:
        """Estado observable para diagnóstico y tests, sin mutaciones."""
        with self._condition:
            return {
                "active_resource": self._active_resource,
                "indexing_active": self._active_index_lease is not None,
                "queue": [
                    {
                        "ticket": waiter.ticket,
                        "resource": waiter.resource,
                        "workload": waiter.workload,
                    }
                    for waiter in self._waiters
                ],
            }


_GLOBAL_COORDINATOR = ResourceCoordinator()


def get_resource_coordinator() -> ResourceCoordinator:
    return _GLOBAL_COORDINATOR
