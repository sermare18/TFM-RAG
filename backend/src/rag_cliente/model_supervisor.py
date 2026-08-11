"""Supervisor seguro de procesos llama.cpp administrados por la aplicación."""

from __future__ import annotations

import os
import atexit
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlparse

import httpx

from rag_cliente.config import Settings, resolve_local_model_profile
from rag_cliente.local_endpoints import is_local_model_endpoint
from rag_cliente.model_manifest import resolve_runtime_model_path

ServerMode = Literal["managed", "external"]


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    cpu_threads: int
    nvidia_available: bool
    nvidia_gpus: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelServerSpec:
    role: str
    mode: ServerMode
    endpoint: str
    model_path: Path
    alias: str
    use_gpu: bool = False
    gpu_layers: int = 0
    context_size: int = 16384
    embeddings: bool = False
    extra_args: tuple[str, ...] = ()


@dataclass(slots=True)
class OwnedProcess:
    role: str
    process: subprocess.Popen
    log_handle: object
    log_path: Path
    command: tuple[str, ...]
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def pid(self) -> int:
        return int(self.process.pid)


def detect_hardware() -> HardwareInfo:
    """Detecta CPU y NVIDIA sin importar ni cargar frameworks de modelos."""
    cpu_threads = os.cpu_count() or 1
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return HardwareInfo(cpu_threads=cpu_threads, nvidia_available=False)

    try:
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return HardwareInfo(cpu_threads=cpu_threads, nvidia_available=False)

    gpu_names = tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    return HardwareInfo(
        cpu_threads=cpu_threads,
        nvidia_available=completed.returncode == 0 and bool(gpu_names),
        nvidia_gpus=gpu_names,
    )


def _endpoint_host_port(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Endpoint HTTP inválido: {endpoint}")
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, parsed.port or default_port


def _health_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def build_server_specs(settings: Settings) -> dict[str, ModelServerSpec]:
    """Resuelve cada rol a rutas GGUF locales explícitas."""
    profile = resolve_local_model_profile(settings)
    use_gpu = profile == "gpu"
    embeddings_role = "embeddings_gpu" if use_gpu else "embeddings_cpu"
    chat_role = "chat_gpu" if use_gpu else "chat_cpu"

    embeddings_model = resolve_runtime_model_path(settings, embeddings_role)
    chat_model = resolve_runtime_model_path(settings, chat_role)

    gpu_layers = settings.model_gpu_layers if use_gpu else 0
    return {
        "embeddings": ModelServerSpec(
            role="embeddings",
            mode=settings.model_embeddings_mode,
            endpoint=settings.llama_cpp_embedding_base_url,
            model_path=embeddings_model,
            alias=settings.default_endpoint_model,
            use_gpu=use_gpu,
            gpu_layers=gpu_layers,
            context_size=settings.model_context_size,
            embeddings=True,
        ),
        "chat": ModelServerSpec(
            role="chat",
            mode=settings.model_chat_mode,
            endpoint=settings.llama_cpp_chat_base_url,
            model_path=chat_model,
            alias=settings.default_endpoint_model,
            use_gpu=use_gpu,
            gpu_layers=gpu_layers,
            context_size=settings.model_context_size,
        ),
    }


class ModelSupervisor:
    """Arranca, verifica y detiene exclusivamente procesos propios."""

    def __init__(
        self,
        settings: Settings,
        *,
        specs: dict[str, ModelServerSpec] | None = None,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        health_probe: Callable[[str, float, float], bool] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.specs = specs or build_server_specs(settings)
        self._process_factory = process_factory
        self._health_probe = health_probe or self._probe_health
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.RLock()
        self._owned: dict[str, OwnedProcess] = {}
        self._idle_timers: dict[str, threading.Timer] = {}
        atexit.register(self.close)

    def validate_binary(self) -> Path:
        binary = Path(self.settings.llama_cpp_binary).expanduser().resolve()
        if not binary.is_file():
            raise FileNotFoundError(
                f"LLAMA_CPP_BINARY no existe o no es un archivo: {binary}"
            )
        return binary

    def build_command(self, spec: ModelServerSpec) -> list[str]:
        binary = self.validate_binary()
        model_path = spec.model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"GGUF local no encontrado para {spec.role}: {model_path}")
        host, port = _endpoint_host_port(spec.endpoint)
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                f"Un servidor managed solo puede escuchar en loopback, no en {host}"
            )

        command = [
            str(binary),
            "-m",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "--ctx-size",
            str(spec.context_size),
            "--parallel",
            "1",
            "--alias",
            spec.alias,
            "-ngl",
            str(spec.gpu_layers if spec.use_gpu else 0),
        ]
        if spec.embeddings:
            command.extend(("--embedding", "--pooling", "last"))
        command.extend(spec.extra_args)
        if any(argument in {"-hf", "--hf-repo"} for argument in command):
            raise ValueError("La ejecución normal no permite -hf/--hf-repo")
        return command

    @staticmethod
    def _probe_health(url: str, connect_timeout: float, read_timeout: float) -> bool:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=connect_timeout,
        )
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.get(url)
            return response.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def _wait_until_healthy(
        self,
        spec: ModelServerSpec,
        process: subprocess.Popen | None,
    ) -> None:
        deadline = self._monotonic() + self.settings.model_start_timeout
        health_url = _health_url(spec.endpoint)
        while self._monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"llama-server de {spec.role} terminó antes de responder /health"
                )
            remaining = max(0.0, deadline - self._monotonic())
            if self._health_probe(
                health_url,
                min(self.settings.model_health_connect_timeout, remaining),
                min(self.settings.model_health_read_timeout, remaining),
            ):
                return
            remaining = max(0.0, deadline - self._monotonic())
            if remaining:
                self._sleep(min(0.2, remaining))
        raise TimeoutError(
            f"Timeout esperando /health de {spec.role} tras "
            f"{self.settings.model_start_timeout:g}s"
        )

    def ensure_started(self, role: str) -> None:
        spec = self.specs[role]
        if not is_local_model_endpoint(
            spec.endpoint,
            self.settings.allowed_local_model_hosts,
        ):
            raise ValueError(
                f"El endpoint de {role} debe ser local; recibido: {spec.endpoint}"
            )
        self.cancel_idle_stop(role)
        with self._lock:
            owned = self._owned.get(role)
            if owned is not None and owned.process.poll() is None:
                return

        if spec.mode == "external":
            # Nunca se registra ni detiene un PID external.
            self._wait_until_healthy(spec, None)
            return

        attempts = self.settings.model_max_retries + 1
        last_error: BaseException | None = None
        for _attempt in range(attempts):
            try:
                owned = self._spawn_owned(spec)
                self._wait_until_healthy(spec, owned.process)
                return
            except BaseException as exc:
                last_error = exc
                self.stop(role)
        assert last_error is not None
        raise last_error

    def _spawn_owned(self, spec: ModelServerSpec) -> OwnedProcess:
        command = self.build_command(spec)
        self.settings.model_logs_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_path = (self.settings.model_logs_path / f"{spec.role}-{timestamp}.log").resolve()
        log_handle = log_path.open("ab")
        try:
            process = self._process_factory(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            log_handle.close()
            raise

        owned = OwnedProcess(
            role=spec.role,
            process=process,
            log_handle=log_handle,
            log_path=log_path,
            command=tuple(command),
        )
        with self._lock:
            self._owned[spec.role] = owned
        return owned

    def stop(self, role: str) -> bool:
        """Detiene solo el ``Popen`` registrado por este supervisor."""
        self.cancel_idle_stop(role)
        spec = self.specs[role]
        if spec.mode == "external":
            return False

        with self._lock:
            owned = self._owned.pop(role, None)
        if owned is None:
            return False

        try:
            if owned.process.poll() is None:
                owned.process.terminate()
                try:
                    owned.process.wait(timeout=self.settings.model_stop_timeout)
                except subprocess.TimeoutExpired:
                    # Sigue dirigido al mismo Popen/PID creado y registrado.
                    owned.process.kill()
                    owned.process.wait(timeout=self.settings.model_stop_timeout)
        finally:
            owned.log_handle.close()
        return True

    def stop_bundle(self, roles: tuple[str, ...] | list[str]) -> None:
        for role in roles:
            self.stop(role)

    def schedule_idle_stop(self, role: str, timeout: float | None = None) -> None:
        spec = self.specs[role]
        if spec.mode == "external":
            return
        idle_timeout = self.settings.model_chat_idle_timeout if timeout is None else timeout
        self.cancel_idle_stop(role)
        if idle_timeout <= 0:
            self.stop(role)
            return
        timer = threading.Timer(idle_timeout, self.stop, args=(role,))
        timer.daemon = True
        with self._lock:
            self._idle_timers[role] = timer
        timer.start()

    def cancel_idle_stop(self, role: str) -> None:
        with self._lock:
            timer = self._idle_timers.pop(role, None)
        if timer is not None:
            timer.cancel()

    def owned_pids(self) -> dict[str, int]:
        with self._lock:
            return {role: process.pid for role, process in self._owned.items()}

    def close(self) -> None:
        for role in tuple(self.specs):
            self.stop(role)
