from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from marker.services.openai import OpenAIService

from rag_cliente.cli import main as cli_main
from rag_cliente.config import Settings, resolve_marker_profile
from rag_cliente.diagnostics import run_doctor
from rag_cliente.marker_capabilities import marker_capabilities
from rag_cliente.marker_llm import BudgetedMarkerOpenAIService
from rag_cliente.pdf_loader import (
    MARKER_OFFICIAL_TABLE_LLM_PROCESSORS,
    PageDocument,
    _build_marker_config,
    _extract_marker_structured_chunks,
    _official_marker_processor_paths,
)
from rag_cliente.smoke_parser import run_smoke_parser


class OfficialMarkerTablePipelineTests(unittest.TestCase):
    def test_quality_profiles_reach_official_table_merge_in_order(self) -> None:
        for profile_name in ("cpu-quality", "gpu-quality"):
            with self.subTest(profile=profile_name):
                profile = resolve_marker_profile(
                    Settings(marker_profile=profile_name),
                    cuda_available=profile_name == "gpu-quality",
                )
                processors = _official_marker_processor_paths(profile)
                table_index = processors.index(
                    "marker.processors.table.TableProcessor"
                )
                self.assertEqual(
                    processors[table_index + 1 : table_index + 3],
                    MARKER_OFFICIAL_TABLE_LLM_PROCESSORS,
                )
                llm_processors = [item for item in processors if ".llm." in item]
                self.assertEqual(llm_processors, list(MARKER_OFFICIAL_TABLE_LLM_PROCESSORS))
                self.assertNotIn("LLMPageCorrectionProcessor", ",".join(processors))

    def test_cpu_digital_has_no_llm_processor_or_service(self) -> None:
        profile = resolve_marker_profile(Settings(marker_profile="cpu-digital"))
        processors = _official_marker_processor_paths(profile)
        config = _build_marker_config(Settings(marker_profile="cpu-digital"), profile)

        self.assertFalse(profile.use_llm)
        self.assertFalse(config["use_llm"])
        self.assertFalse(any(".llm." in item for item in processors))

    def test_only_official_openai_service_is_extended(self) -> None:
        self.assertTrue(issubclass(BudgetedMarkerOpenAIService, OpenAIService))

    def test_no_adapter_page_correction_or_continuation_heuristic_exists(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / "rag_cliente"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in source_root.glob("*.py")
        )
        forbidden = (
            "TableContinuation" + "Resolver",
            "Table" + "Stitcher",
            "LLMPage" + "CorrectionProcessor",
            "block_" + "correction_prompt",
        )
        for name in forbidden:
            self.assertNotIn(name, source)


class StructuredProvenanceTests(unittest.TestCase):
    def test_merged_preclassified_tables_span_two_pages(self) -> None:
        rendered = {
            "block_type": "Document",
            "children": [
                {
                    "block_type": "Page",
                    "id": "/page/0/Page/0",
                    "children": [
                        {
                            "block_type": "Table",
                            "id": "/page/0/Table/0",
                            "html": "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>",
                            "children": [
                                {
                                    "block_type": "TableCell",
                                    "id": "/page/0/TableCell/0",
                                    "text": "A",
                                    "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                                },
                                {
                                    "block_type": "TableCell",
                                    "id": "/page/1/TableCell/0",
                                    "text": "B",
                                    "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                                },
                            ],
                        }
                    ],
                }
            ],
        }

        page = _extract_marker_structured_chunks(rendered)[0]
        table = page["children"][0]

        self.assertEqual(table["kind"], "Table")
        self.assertEqual((table["page_start"], table["page_end"]), (1, 2))
        self.assertEqual(table["source_pages"], [1, 2])
        self.assertEqual(page["source_pages"], [1, 2])
        self.assertIn("/page/0/Table/0", table["source_block_ids"])
        self.assertIn("/page/1/TableCell/0", table["source_block_ids"])

    def test_table_plus_text_remains_separate_and_text_is_not_reclassified(self) -> None:
        rendered = {
            "block_type": "Document",
            "children": [
                {
                    "block_type": "Page",
                    "id": "/page/0/Page/0",
                    "children": [
                        {
                            "block_type": "Table",
                            "id": "/page/0/Table/0",
                            "html": "<table><tr><td>A</td></tr></table>",
                        }
                    ],
                },
                {
                    "block_type": "Page",
                    "id": "/page/1/Page/1",
                    "children": [
                        {
                            "block_type": "Text",
                            "id": "/page/1/Text/0",
                            "html": "<p>B</p>",
                        }
                    ],
                },
            ],
        }

        pages = _extract_marker_structured_chunks(rendered)

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["children"][0]["kind"], "Table")
        self.assertEqual(pages[1]["children"][0]["kind"], "Text")
        self.assertEqual(pages[0]["source_pages"], [1])
        self.assertEqual(pages[1]["source_pages"], [2])


class CapabilitiesAndPackagingTests(unittest.TestCase):
    def test_declared_capabilities_match_accepted_limit(self) -> None:
        self.assertEqual(
            marker_capabilities(),
            {
                "marker_version": "2.0.0",
                "multipage_table_merge": "preclassified_tables_only",
                "text_to_table_reclassification": False,
                "complete_multipage_table_support": False,
            },
        )

    def test_marker_dependency_is_exactly_pinned(self) -> None:
        backend = Path(__file__).parents[1]
        for filename in ("requirements.txt", "pyproject.toml"):
            content = (backend / filename).read_text(encoding="utf-8")
            self.assertIn("marker-pdf[full]==2.0.0", content)
            self.assertNotIn("marker-pdf[full]>=", content)

    def test_doctor_exposes_capabilities_without_starting_models(self) -> None:
        hardware = type(
            "Hardware",
            (),
            {"cpu_threads": 4, "nvidia_available": False, "nvidia_gpus": ()},
        )()
        model_report = {
            "role": "test",
            "label": "Test",
            "repository": None,
            "quantization": "Q4",
            "valid": True,
            "artifacts": [],
        }
        with (
            patch("rag_cliente.diagnostics.detect_hardware", return_value=hardware),
            patch("rag_cliente.diagnostics.check_models", return_value=[model_report]),
        ):
            report = run_doctor(Settings(marker_profile="cpu-digital"))

        self.assertEqual(
            report["capabilities"]["multipage_table_merge"],
            "preclassified_tables_only",
        )
        self.assertFalse(report["capabilities"]["text_to_table_reclassification"])


class SmokeParserTests(unittest.TestCase):
    def test_smoke_parser_returns_structured_json_without_real_marker(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            pdf = Path(tmp_dir) / "sample.pdf"
            pdf.write_bytes(b"simulated")
            settings = Settings(
                marker_profile="cpu-digital",
                marker_page_range="0-1",
                model_supervision_enabled=False,
            )
            fake_page = PageDocument(
                document_id="sample",
                source="sample.pdf",
                source_path=str(pdf.resolve()),
                source_type="pdf",
                page_number=1,
                text="contenido",
                block_type="Page",
                id="/page/0/Page/0",
            )
            with patch(
                "rag_cliente.smoke_parser.load_pdf_pages",
                return_value=[fake_page],
            ) as loader:
                report = run_smoke_parser(pdf, settings)

        loader.assert_called_once()
        self.assertEqual(report["elements"][0]["kind"], "Page")
        self.assertEqual(
            report["metadata"]["capabilities"]["multipage_table_merge"],
            "preclassified_tables_only",
        )
        self.assertEqual(report["metadata"]["diagnostic"]["requested_page_range"], "0-1")

    def test_cli_forwards_profile_and_page_range(self) -> None:
        report = {"id": "sample", "elements": [], "metadata": {}}
        with (
            patch("rag_cliente.cli.get_settings", return_value=Settings()),
            patch("rag_cliente.cli.run_smoke_parser", return_value=report) as smoke,
            patch.object(
                sys,
                "argv",
                [
                    "rag-cli",
                    "smoke-parser",
                    "sample.pdf",
                    "--profile",
                    "cpu-quality",
                    "--pages",
                    "2-4",
                ],
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            cli_main()

        smoke_settings = smoke.call_args.args[1]
        self.assertEqual(smoke_settings.marker_profile, "cpu-quality")
        self.assertEqual(smoke_settings.marker_page_range, "2-4")
        self.assertEqual(json.loads(output.getvalue())["id"], "sample")


if __name__ == "__main__":
    unittest.main()
