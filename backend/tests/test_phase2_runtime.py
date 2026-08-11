from __future__ import annotations

import io
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from rag_cliente.cli import main as cli_main
from rag_cliente.config import Settings
from rag_cliente.marker_llm import (
    BudgetedMarkerOpenAIService,
    LLMBudgetExceededError,
    MarkerLLMError,
)
from rag_cliente.model_manifest import check_artifact
from rag_cliente.model_supervisor import ModelServerSpec, ModelSupervisor
from rag_cliente.pdf_loader import create_marker_converter
from rag_cliente.pipeline import RagPipeline
from rag_cliente.resource_coordinator import ResourceCoordinator


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class SupervisorTests(unittest.TestCase):
    def _fixture(self, tmp_dir: str, **settings_overrides):
        root = Path(tmp_dir)
        binary = root / "llama-server.exe"
        model = root / "model.gguf"
        mmproj = root / "mmproj.gguf"
        for path in (binary, model, mmproj):
            path.write_bytes(b"fake")

        settings_values = {
            "llama_cpp_binary": str(binary),
            "model_logs_dir": str(root / "logs"),
            "model_start_timeout": 0.02,
            "model_stop_timeout": 0.02,
            "model_max_retries": 0,
            **settings_overrides,
        }
        settings = Settings(**settings_values)
        spec = ModelServerSpec(
            role="vlm",
            mode="managed",
            endpoint="http://127.0.0.1:18083/v1",
            model_path=model,
            mmproj_path=mmproj,
            alias="marker-vlm",
            use_gpu=False,
        )
        return settings, spec

    def test_build_command_uses_local_paths_mmproj_one_slot_and_cpu_ngl_zero(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            settings, spec = self._fixture(tmp_dir)
            supervisor = ModelSupervisor(settings, specs={"vlm": spec})

            command = supervisor.build_command(spec)

        self.assertIn("--mmproj", command)
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertEqual(command[command.index("-ngl") + 1], "0")
        self.assertNotIn("-hf", command)
        self.assertNotIn("--hf-repo", command)

    def test_health_success_registers_only_spawned_pid_and_passes_http_timeouts(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            settings, spec = self._fixture(tmp_dir)
            process = FakeProcess(41001)
            probes = []
            supervisor = ModelSupervisor(
                settings,
                specs={"vlm": spec},
                process_factory=lambda *args, **kwargs: process,
                health_probe=lambda url, connect, read: probes.append((url, connect, read)) or True,
            )

            supervisor.ensure_started("vlm")
            owned = supervisor.owned_pids()
            stopped = supervisor.stop("vlm")

        self.assertEqual(owned, {"vlm": 41001})
        self.assertTrue(stopped)
        self.assertTrue(process.terminated)
        self.assertEqual(probes[0][0], "http://127.0.0.1:18083/health")
        self.assertGreater(probes[0][1], 0)
        self.assertLessEqual(probes[0][1], settings.model_health_connect_timeout)
        self.assertGreater(probes[0][2], 0)
        self.assertLessEqual(probes[0][2], settings.model_health_read_timeout)

    def test_start_timeout_stops_each_owned_attempt_and_retries_at_most_once(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            settings, spec = self._fixture(tmp_dir, model_max_retries=1)
            processes = []

            def factory(*args, **kwargs):
                process = FakeProcess(42000 + len(processes))
                processes.append(process)
                return process

            supervisor = ModelSupervisor(
                settings,
                specs={"vlm": spec},
                process_factory=factory,
                health_probe=lambda *args: False,
                sleep=lambda _seconds: None,
            )

            with self.assertRaises(TimeoutError):
                supervisor.ensure_started("vlm")

        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.terminated for process in processes))
        self.assertEqual(supervisor.owned_pids(), {})

    def test_external_server_is_never_spawned_registered_or_stopped(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            settings, managed_spec = self._fixture(tmp_dir)
            external_spec = ModelServerSpec(
                role="chat",
                mode="external",
                endpoint="http://127.0.0.1:18081/v1",
                model_path=managed_spec.model_path,
                alias="default",
            )
            factory = Mock(side_effect=AssertionError("no debe crear procesos"))
            supervisor = ModelSupervisor(
                settings,
                specs={"chat": external_spec},
                process_factory=factory,
                health_probe=lambda *args: True,
            )

            supervisor.ensure_started("chat")
            stopped = supervisor.stop("chat")

        factory.assert_not_called()
        self.assertFalse(stopped)
        self.assertEqual(supervisor.owned_pids(), {})


class ResourceCoordinatorTests(unittest.TestCase):
    @staticmethod
    def _wait_for_queue(coordinator: ResourceCoordinator, size: int) -> None:
        deadline = time.monotonic() + 1
        while len(coordinator.snapshot()["queue"]) < size:
            if time.monotonic() >= deadline:
                raise AssertionError("la cola no alcanzó el tamaño esperado")
            time.sleep(0.005)

    def test_fifo_and_resource_exclusion(self) -> None:
        coordinator = ResourceCoordinator()
        first = coordinator.acquire("parser_bundle")
        order = []
        active = 0
        max_active = 0
        lock = threading.Lock()

        def worker(resource):
            nonlocal active, max_active
            with coordinator.acquire(resource, timeout=1):
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    order.append(resource)
                time.sleep(0.01)
                with lock:
                    active -= 1

        embeddings_thread = threading.Thread(target=worker, args=("embeddings",))
        chat_thread = threading.Thread(target=worker, args=("chat",))
        embeddings_thread.start()
        self._wait_for_queue(coordinator, 1)
        chat_thread.start()
        self._wait_for_queue(coordinator, 2)
        first.release()
        embeddings_thread.join(1)
        chat_thread.join(1)

        self.assertEqual(order, ["embeddings", "chat"])
        self.assertEqual(max_active, 1)

    def test_lease_is_released_after_exception(self) -> None:
        coordinator = ResourceCoordinator()
        with self.assertRaisesRegex(RuntimeError, "fallo"):
            with coordinator.acquire("parser_bundle"):
                raise RuntimeError("fallo")

        with coordinator.acquire("embeddings", timeout=0.1):
            self.assertEqual(coordinator.snapshot()["active_resource"], "embeddings")

    def test_chat_waits_until_index_job_finishes(self) -> None:
        coordinator = ResourceCoordinator()
        acquired = threading.Event()

        def acquire_chat():
            with coordinator.acquire("chat", workload="query", timeout=1):
                acquired.set()

        with coordinator.acquire_indexing():
            thread = threading.Thread(target=acquire_chat)
            thread.start()
            self._wait_for_queue(coordinator, 1)
            self.assertFalse(acquired.is_set())
        # El test principal de FIFO cubre la liberación normal; aquí no dejamos
        # que un chat pueda coexistir con el job de indexación.
        thread.join(0.2)
        self.assertTrue(acquired.is_set())

    def test_older_query_is_not_starved_by_a_later_index_job(self) -> None:
        coordinator = ResourceCoordinator()
        blocker = coordinator.acquire("parser_bundle")
        order = []

        def query():
            with coordinator.acquire("chat", timeout=1):
                order.append("query")

        def index_job():
            with coordinator.acquire_indexing(timeout=1):
                order.append("index")

        query_thread = threading.Thread(target=query)
        index_thread = threading.Thread(target=index_job)
        query_thread.start()
        self._wait_for_queue(coordinator, 1)
        index_thread.start()
        blocker.release()
        query_thread.join(1)
        index_thread.join(1)

        self.assertEqual(order, ["query", "index"])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def fake_response(completion_tokens: int, payload: str = '{"ok": true}'):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=payload))],
        usage=types.SimpleNamespace(
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens + 10,
        ),
    )


class MarkerBudgetTests(unittest.TestCase):
    @staticmethod
    def _service(settings: Settings, responses, monotonic=time.monotonic):
        completions = FakeCompletions(responses)
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        service = BudgetedMarkerOpenAIService(
            settings,
            client_factory=lambda: client,
            monotonic=monotonic,
        )
        return service, completions

    def test_request_budget_and_per_request_token_cap(self) -> None:
        settings = Settings(marker_llm_max_requests=1)
        service, completions = self._service(settings, [fake_response(1)])

        with service.document_budget("doc"):
            self.assertEqual(service("prompt", None, None, dict), {"ok": True})
            with self.assertRaises(LLMBudgetExceededError) as raised:
                service("prompt 2", None, None, dict)

        self.assertEqual(raised.exception.code, "llm_budget_exceeded")
        self.assertEqual(
            completions.calls[0]["max_tokens"],
            settings.marker_llm_max_tokens_per_request,
        )

    def test_completion_token_budget_is_accumulated(self) -> None:
        settings = Settings(marker_llm_max_generated_tokens_per_document=3)
        service, _ = self._service(settings, [fake_response(4)])

        with self.assertRaises(LLMBudgetExceededError) as raised:
            with service.document_budget("doc"):
                service("prompt", None, None, dict)

        self.assertEqual(raised.exception.details["budget"], "generated_tokens_per_document")

    def test_job_timeout_aborts_before_request(self) -> None:
        values = iter((0.0, 2.0, 2.0))
        settings = Settings(marker_llm_job_timeout=1)
        service, completions = self._service(
            settings,
            [fake_response(1)],
            monotonic=lambda: next(values),
        )

        with self.assertRaises(LLMBudgetExceededError) as raised:
            with service.document_budget("doc"):
                service("prompt", None, None, dict)

        self.assertEqual(raised.exception.details["budget"], "job_timeout_seconds")
        self.assertEqual(completions.calls, [])

    def test_external_endpoint_is_rejected(self) -> None:
        settings = Settings(marker_openai_base_url="https://api.openai.com/v1")
        with self.assertRaises(MarkerLLMError) as raised:
            BudgetedMarkerOpenAIService(settings)
        self.assertEqual(raised.exception.code, "external_llm_endpoint_rejected")

        public_ip_settings = Settings(marker_openai_base_url="http://8.8.8.8/v1")
        with self.assertRaises(MarkerLLMError):
            BudgetedMarkerOpenAIService(public_ip_settings)

    def test_marker_has_no_gemini_fallback(self) -> None:
        settings = Settings(
            marker_profile="cpu-quality",
            marker_llm_fallback_to_base=True,
        )
        with patch("rag_cliente.pdf_loader._require_marker_2") as require_marker:
            with self.assertRaises(MarkerLLMError) as raised:
                create_marker_converter(settings)
        require_marker.assert_not_called()
        self.assertEqual(raised.exception.code, "external_llm_fallback_rejected")


class PipelineResourceSequenceTests(unittest.TestCase):
    class RecordingSupervisor:
        def __init__(self, events):
            self.events = events

        def ensure_started(self, role):
            self.events.append(f"start:{role}")

        def stop_bundle(self, roles):
            self.events.append(f"stop:{','.join(roles)}")

        def schedule_idle_stop(self, role):
            self.events.append(f"idle:{role}")

    def test_index_sequence_stops_parser_then_embeddings_before_writes(self) -> None:
        events = []
        settings = Settings(
            marker_profile="cpu-quality",
            hybrid_search_enabled=True,
        )
        pipeline = RagPipeline(settings)
        pipeline.coordinator = ResourceCoordinator()
        pipeline.supervisor = self.RecordingSupervisor(events)
        pipeline.chunker = Mock()
        chunk = types.SimpleNamespace(text="contenido")
        pipeline.chunker.chunk_pages.return_value = [chunk]
        pipeline.client = Mock()
        pipeline.client.embed_texts.side_effect = lambda *args, **kwargs: (
            events.append("embed") or [[0.1, 0.2]]
        )
        pipeline.store = Mock()
        pipeline.store.replace_chunks.side_effect = lambda *args: events.append("lancedb")
        pipeline.bm25_store = Mock()
        pipeline.bm25_store.replace_chunks.side_effect = lambda *args: events.append("bm25")

        with patch(
            "rag_cliente.pipeline.load_documents_from_directory",
            side_effect=lambda *args, **kwargs: events.append("marker") or [object()],
        ):
            pipeline.index_documents(Path("docs"))

        self.assertEqual(
            events,
            [
                "start:surya",
                "start:vlm",
                "marker",
                "stop:surya,vlm",
                "start:embeddings",
                "embed",
                "stop:embeddings",
                "lancedb",
                "bm25",
            ],
        )

    def test_query_sequence_unloads_embeddings_before_single_chat_load(self) -> None:
        events = []
        settings = Settings(hybrid_search_enabled=False)
        pipeline = RagPipeline(settings)
        pipeline.coordinator = ResourceCoordinator()
        pipeline.supervisor = self.RecordingSupervisor(events)
        match = {
            "document_id": "doc",
            "source": "doc.pdf",
            "source_path": "doc.pdf",
            "source_type": "pdf",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "text": "contenido",
        }
        pipeline.client = Mock()
        pipeline.client.rewrite_question_for_retrieval.side_effect = lambda *args, **kwargs: (
            events.append("rewrite:deterministic") or "pregunta"
        )
        pipeline.client.embed_texts.side_effect = lambda *args, **kwargs: (
            events.append("embed") or [[0.1, 0.2]]
        )
        pipeline.client.generate_answer.side_effect = lambda *args, **kwargs: (
            events.append("answer") or {"answer": "respuesta", "reasoning": ""}
        )
        pipeline.client.select_used_source_ids.side_effect = lambda *args, **kwargs: (
            events.append("audit") or ["S1"]
        )
        pipeline.store = Mock()
        pipeline.store.search.side_effect = lambda *args, **kwargs: (
            events.append("retrieve") or [match]
        )

        pipeline.ask("pregunta")

        self.assertLess(events.index("stop:embeddings"), events.index("start:chat"))
        self.assertEqual(events.count("start:chat"), 1)
        self.assertLess(events.index("answer"), events.index("audit"))
        self.assertEqual(events[-1], "idle:chat")


class ModelCommandTests(unittest.TestCase):
    def test_mmproj_check_requires_visual_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            path = Path(tmp_dir) / "mmproj.gguf"
            path.write_bytes(b"GGUF" + b"\0" * 32 + b"clip.projector_type")
            self.assertEqual(check_artifact(path, "mmproj"), (True, "GGUF válido"))

    def test_plan_and_check_never_call_download(self) -> None:
        valid_report = [
            {
                "role": "test",
                "label": "Test",
                "repository": None,
                "quantization": "Q4",
                "valid": True,
                "artifacts": [
                    {
                        "kind": "model",
                        "path": "model.gguf",
                        "expected_size": "1 GiB",
                        "valid": True,
                        "message": "GGUF válido",
                    }
                ],
            }
        ]
        download = Mock(side_effect=AssertionError("no debe descargar"))
        with (
            patch("rag_cliente.cli.get_settings", return_value=Settings()),
            patch("rag_cliente.cli.plan_models", return_value=valid_report),
            patch("rag_cliente.cli.check_models", return_value=valid_report),
            patch("rag_cliente.cli.download_models", download),
            redirect_stdout(io.StringIO()),
        ):
            with patch.object(sys, "argv", ["rag-cli", "models", "plan", "cpu"]):
                cli_main()
            with patch.object(sys, "argv", ["rag-cli", "models", "check"]):
                cli_main()

        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
