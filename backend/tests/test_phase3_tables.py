from __future__ import annotations

import io
import importlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from marker.services.openai import OpenAIService

# test_chat_history puede instalar dobles globales durante el discovery. Esta
# fase los retira solo si siguen siendo dobles; no recarga módulos reales que
# otros tests ya hayan importado y parcheen mediante su identidad de módulo.
_loaded_indexer = sys.modules.get("rag_cliente.indexer")
if _loaded_indexer is not None and not hasattr(_loaded_indexer, "ChunkRecord"):
    sys.modules.pop("rag_cliente.indexer", None)
_loaded_pdf_loader = sys.modules.get("rag_cliente.pdf_loader")
if _loaded_pdf_loader is not None and not hasattr(_loaded_pdf_loader, "DocumentElement"):
    sys.modules.pop("rag_cliente.pdf_loader", None)
_loaded_vector_store = sys.modules.get("rag_cliente.vector_store")
if _loaded_vector_store is not None and not hasattr(_loaded_vector_store, "build_schema"):
    sys.modules.pop("rag_cliente.vector_store", None)
importlib.invalidate_caches()

from rag_cliente.cli import main as cli_main
from rag_cliente.bm25_store import BM25Store
from rag_cliente.config import Settings, resolve_marker_profile
from rag_cliente.diagnostics import run_doctor
from rag_cliente.marker_capabilities import marker_capabilities
from rag_cliente.marker_llm import BudgetedMarkerOpenAIService
from rag_cliente.indexer import PdfChunker
from rag_cliente.pipeline import RagPipeline
from rag_cliente.pdf_loader import (
    DocumentElement,
    MARKER_OFFICIAL_TABLE_LLM_PROCESSORS,
    PageDocument,
    ParsedDocument,
    _build_marker_config,
    _extract_marker_structured_chunks,
    _official_marker_processor_paths,
    parsed_document_from_pages,
)
from rag_cliente.smoke_parser import run_smoke_parser
from rag_cliente.vector_store import LanceDBStore


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


class StructuredDocumentChunkingTests(unittest.TestCase):
    @staticmethod
    def _document(elements: list[DocumentElement]) -> ParsedDocument:
        return ParsedDocument(
            id="doc",
            source="doc.pdf",
            source_path="C:/docs/doc.pdf",
            source_type="pdf",
            elements=elements,
            metadata={
                "parser_profile": "gpu-quality",
                "ocr_used_by_page": {"1": False, "2": False, "3": False},
                "capabilities": marker_capabilities(),
            },
        )

    @staticmethod
    def _text_element(
        element_id: str,
        text: str,
        page: int,
        section: str,
        kind: str = "Text",
    ) -> DocumentElement:
        return DocumentElement(
            id=element_id,
            kind=kind,
            html=f"<p>{text}</p>",
            text=text,
            page_start=page,
            page_end=page,
            source_pages=[page],
            source_block_ids=[element_id],
            source_spans=[{"page": page, "block_id": element_id}],
            section_path=[section],
            provenance="marker",
            document_id="doc",
        )

    @staticmethod
    def _table_element(
        html: str,
        *,
        pages: list[int] | None = None,
    ) -> DocumentElement:
        source_pages = pages or [1]
        return DocumentElement(
            id="/page/0/Table/0",
            kind="Table",
            html=html,
            text="texto aplanado que no debe gobernar el chunking",
            page_start=min(source_pages),
            page_end=max(source_pages),
            source_pages=source_pages,
            source_block_ids=["/page/0/Table/0", "/page/1/Table/0"],
            source_spans=[
                {"page": page, "block_id": f"/page/{page - 1}/Table/0"}
                for page in source_pages
            ],
            section_path=["Incidencias"],
            provenance="marker",
            document_id="doc",
            table_id="/page/0/Table/0",
        )

    def test_consecutive_text_in_same_section_forms_multipage_chunk(self) -> None:
        document = self._document(
            [
                self._text_element("t1", "Texto de la primera pagina", 1, "A"),
                self._text_element("t2", "Continua en la segunda pagina", 2, "A"),
            ]
        )

        chunks = PdfChunker(Settings()).chunk_documents([document])

        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].page_start, chunks[0].page_end), (1, 2))
        self.assertEqual(chunks[0].source_pages, [1, 2])

    def test_section_header_prevents_incorrect_merge(self) -> None:
        document = self._document(
            [
                self._text_element("t1", "Seccion anterior", 1, "A"),
                self._text_element("h2", "Nueva seccion", 2, "B", "SectionHeader"),
                self._text_element("t2", "Contenido nuevo", 2, "B"),
            ]
        )

        chunks = PdfChunker(Settings()).chunk_documents([document])

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section_path, ["A"])
        self.assertEqual(chunks[1].section_path, ["B"])

    def test_preclassified_merged_table_keeps_pages_rows_and_header(self) -> None:
        table = self._table_element(
            "<table><thead><tr><th>ID</th><th>Estado</th></tr></thead>"
            "<tbody><tr><td>INC-001</td><td>Cerrada</td></tr>"
            "<tr><td>INC-002</td><td>Abierta</td></tr></tbody></table>",
            pages=[1, 2],
        )

        chunks = PdfChunker(Settings()).chunk_documents([self._document([table])])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].kind, "table")
        self.assertEqual((chunks[0].page_start, chunks[0].page_end), (1, 2))
        self.assertEqual(chunks[0].source_pages, [1, 2])
        self.assertIn("| ID | Estado |", chunks[0].text)
        self.assertIn("| INC-001 | Cerrada |", chunks[0].text)
        self.assertIn("<tr><th>ID</th><th>Estado</th></tr>", chunks[0].html)

    def test_table_plus_text_remains_separate(self) -> None:
        table = self._table_element(
            "<table><tr><th>ID</th></tr><tr><td>INC-001</td></tr></table>"
        )
        text = self._text_element("t2", "No soy continuacion de tabla", 2, "A")

        chunks = PdfChunker(Settings()).chunk_documents([self._document([table, text])])

        self.assertEqual([chunk.kind for chunk in chunks], ["table", "text"])
        self.assertIsNone(chunks[1].table_id)

    def test_table_rows_are_not_split_and_structural_header_repeats(self) -> None:
        rows = "".join(
            f"<tr><td>INC-{index:03d}</td><td>{'dato ' * 8}</td></tr>"
            for index in range(1, 7)
        )
        table = self._table_element(
            "<table><thead><tr><th>ID</th><th>Detalle</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        settings = Settings(table_chunk_max_tokens=45)

        chunks = PdfChunker(settings).chunk_documents([self._document([table])])

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.text.startswith("| ID | Detalle |"))
        for index in range(1, 7):
            row_id = f"INC-{index:03d}"
            self.assertEqual(sum(row_id in chunk.text for chunk in chunks), 1)

    def test_oversize_table_row_stays_complete(self) -> None:
        long_value = " ".join(f"valor{index}" for index in range(50))
        table = self._table_element(
            "<table><tr><th>ID</th><th>Detalle</th></tr>"
            f"<tr><td>INC-001</td><td>{long_value}</td></tr></table>"
        )

        chunks = PdfChunker(
            Settings(table_chunk_max_tokens=20)
        ).chunk_documents([self._document([table])])

        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].oversize)
        self.assertIn(long_value, chunks[0].text)

    def test_oversize_first_row_remains_marked_when_more_rows_follow(self) -> None:
        long_value = " ".join(f"valor{index}" for index in range(50))
        table = self._table_element(
            "<table><tr><th>ID</th><th>Detalle</th></tr>"
            f"<tr><td>INC-001</td><td>{long_value}</td></tr>"
            "<tr><td>INC-002</td><td>corto</td></tr></table>"
        )

        chunks = PdfChunker(
            Settings(table_chunk_max_tokens=20)
        ).chunk_documents([self._document([table])])

        self.assertGreaterEqual(len(chunks), 2)
        first = next(chunk for chunk in chunks if "INC-001" in chunk.text)
        self.assertTrue(first.oversize)
        self.assertIn(long_value, first.text)

    def test_chunk_ids_are_deterministic(self) -> None:
        document = self._document(
            [self._text_element("t1", "Contenido estable", 1, "A")]
        )
        chunker = PdfChunker(Settings())

        first = chunker.chunk_documents([document])
        second = chunker.chunk_documents([document])

        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])

    def test_marker_json_table_regression_is_not_flattened(self) -> None:
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
                            "html": (
                                "<table><tr><th>ID</th><th>Fecha</th></tr>"
                                "<tr><td>INC-001</td><td>05/01/2026</td></tr></table>"
                            ),
                            "children": [
                                {
                                    "block_type": "TableCell",
                                    "id": "/page/0/TableCell/0",
                                    "text": "INC-001",
                                },
                                {
                                    "block_type": "TableCell",
                                    "id": "/page/2/TableCell/0",
                                    "text": "05/01/2026",
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        structured = _extract_marker_structured_chunks(rendered)[0]
        page = PageDocument(
            document_id="doc",
            source="doc.pdf",
            source_path="C:/docs/doc.pdf",
            source_type="pdf",
            page_number=1,
            text=structured["text"],
            block_type=structured["block_type"],
            id=structured["id"],
            html=structured["html"],
            children=structured["children"],
            source_pages=structured["source_pages"],
            source_block_ids=structured["source_block_ids"],
            source_spans=structured["source_spans"],
            page_start=structured["page_start"],
            page_end=structured["page_end"],
            parser_profile="gpu-quality",
        )
        document = parsed_document_from_pages(Path("C:/docs/doc.pdf"), [page])

        chunks = PdfChunker(Settings()).chunk_documents([document])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].kind, "table")
        self.assertIn("| ID | Fecha |", chunks[0].text)
        self.assertEqual(chunks[0].source_pages, [1, 3])


class RetrievalAndSchemaV2Tests(unittest.TestCase):
    @staticmethod
    def _match(chunk_id: str) -> dict:
        return {
            "id": chunk_id,
            "chunk_id": chunk_id,
            "document_id": "doc",
            "source": "doc.pdf",
            "source_path": "C:/docs/doc.pdf",
            "source_type": "pdf",
            "page_start": 1,
            "page_end": 1,
            "source_pages": [1],
            "chunk_index": ord(chunk_id) - ord("A"),
            "text": chunk_id,
        }

    def test_rrf_is_deterministic_and_uses_one_based_positions(self) -> None:
        a, b, c = (self._match(value) for value in "ABC")
        ranked = RagPipeline._merge_hybrid_matches(
            vector_match_groups=[[a, b]],
            bm25_match_groups=[[{**b, "_bm25_score": 4.0}, {**c, "_bm25_score": 3.0}]],
            top_k=3,
            rrf_k=60,
        )

        self.assertEqual([item["chunk_id"] for item in ranked], ["B", "A", "C"])
        self.assertAlmostEqual(ranked[0]["_rrf_score"], (1 / 62) + (1 / 61))
        self.assertEqual((ranked[0]["_vector_rank"], ranked[0]["_bm25_rank"]), (2, 1))

    def test_citations_preserve_single_page_and_range(self) -> None:
        single = self._match("A")
        ranged = {**self._match("B"), "page_end": 3, "source_pages": [1, 2, 3]}

        single_citation = RagPipeline._citation_from_match(single, "S1")
        range_citation = RagPipeline._citation_from_match(ranged, "S2")

        self.assertEqual(RagPipeline._page_label(1, 1), "p.1")
        self.assertEqual(RagPipeline._page_label(1, 3), "pp.1-3")
        self.assertEqual(single_citation["source_pages"], [1])
        self.assertEqual(range_citation["source_pages"], [1, 2, 3])

    def test_incompatible_lancedb_schema_requires_reindex(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            store = LanceDBStore(Path(tmp_dir) / "lancedb", "chunks")
            store.db.create_table("chunks", data=[{"id": "old", "text": "legacy"}])

            with self.assertRaisesRegex(RuntimeError, "reindexar"):
                store.list_chunks()

    def test_lancedb_v2_roundtrip_preserves_structured_metadata(self) -> None:
        element = StructuredDocumentChunkingTests._text_element(
            "t1",
            "Contenido persistente",
            1,
            "Seccion",
        )
        document = StructuredDocumentChunkingTests._document([element])
        chunk = PdfChunker(Settings()).chunk_documents([document])[0]

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            store = LanceDBStore(Path(tmp_dir) / "lancedb", "chunks")
            store.replace_chunks([chunk], [[1.0, 0.0]])
            rows = store.list_chunks()
            matches = store.search([1.0, 0.0], top_k=1, document_id="doc")

        self.assertEqual(rows[0]["chunk_id"], chunk.chunk_id)
        self.assertEqual(rows[0]["source_spans"], chunk.source_spans)
        self.assertEqual(rows[0]["metadata"], chunk.metadata)
        self.assertEqual(matches[0]["document_id"], "doc")

    def test_incompatible_bm25_schema_requires_reindex(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp_dir:
            index_path = Path(tmp_dir) / "bm25.json"
            index_path.write_text(
                json.dumps({"rows": [{"text": "legacy"}]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "reindexar"):
                BM25Store(index_path).search("legacy", top_k=1)


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
