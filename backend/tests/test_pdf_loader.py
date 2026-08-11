from __future__ import annotations

import tempfile
import types
import unittest
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

# El otro módulo de pruebas instala dobles ligeros para FastAPI. Yo retiro solo
# el doble del loader para comprobar aquí la implementación real de Marker.
sys.modules.pop("rag_cliente.pdf_loader", None)
importlib.invalidate_caches()

from rag_cliente.config import Settings, resolve_marker_profile
import rag_cliente.pdf_loader as pdf_loader
from rag_cliente.pdf_loader import (
    PageDocument,
    _build_marker_config,
    _collect_marker_page_extraction_details,
    _extract_marker_structured_chunks,
    _require_marker_2,
    _split_marker_markdown_by_page,
    create_marker_converter,
    load_documents_from_directory,
)


class Marker2ConfigurationTests(unittest.TestCase):
    def test_explicit_profiles_have_exact_marker_values(self) -> None:
        expected = {
            "cpu-digital": ("fast", True, False, "cpu", None),
            "cpu-quality": ("fast", False, True, "cpu", "llamacpp"),
            "gpu-quality": ("balanced", False, True, "cuda", "llamacpp"),
        }

        for profile_name, values in expected.items():
            with self.subTest(profile=profile_name):
                profile = resolve_marker_profile(
                    Settings(marker_profile=profile_name),
                    cuda_available=profile_name == "gpu-quality",
                )
                config = _build_marker_config(
                    Settings(marker_profile=profile_name),
                    profile,
                )

                self.assertEqual(
                    (
                        config["mode"],
                        config["disable_ocr"],
                        config["use_llm"],
                        profile.torch_device,
                        profile.inference_backend,
                    ),
                    values,
                )

    def test_auto_selects_cpu_without_cuda_and_gpu_with_cuda(self) -> None:
        settings = Settings(marker_profile="auto")

        self.assertEqual(
            resolve_marker_profile(settings, cuda_available=False).name,
            "cpu-quality",
        )
        self.assertEqual(
            resolve_marker_profile(settings, cuda_available=True).name,
            "gpu-quality",
        )

    def test_explicit_profile_always_precedes_hardware_detection(self) -> None:
        cpu_profile = resolve_marker_profile(
            Settings(marker_profile="cpu-digital"),
            cuda_available=True,
        )
        gpu_profile = resolve_marker_profile(
            Settings(marker_profile="gpu-quality"),
            cuda_available=False,
        )

        self.assertEqual(cpu_profile.name, "cpu-digital")
        self.assertEqual(gpu_profile.name, "gpu-quality")

    def test_configuration_loads_without_torch_or_cuda(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            settings = Settings(marker_profile="cpu-digital")
            config = _build_marker_config(settings)
            auto_profile = resolve_marker_profile(Settings(marker_profile="auto"))

        self.assertEqual(config["mode"], "fast")
        self.assertTrue(config["disable_ocr"])
        self.assertEqual(auto_profile.name, "cpu-quality")

    def test_json_is_primary_and_markdown_requires_compatibility_flag(self) -> None:
        json_config = _build_marker_config(Settings(marker_profile="cpu-digital"))
        markdown_config = _build_marker_config(
            Settings(
                marker_profile="cpu-digital",
                marker_markdown_compatibility=True,
            )
        )

        self.assertEqual(json_config["output_format"], "json")
        self.assertFalse(json_config["paginate_output"])
        self.assertEqual(markdown_config["output_format"], "markdown")
        self.assertTrue(markdown_config["paginate_output"])

    def test_no_custom_marker_line_builder_remains(self) -> None:
        source = Path(pdf_loader.__file__).read_text(encoding="utf-8")
        forbidden_builder = "LayoutAware" + "LineBuilder"

        self.assertNotIn(forbidden_builder, source)
        self.assertNotIn("from marker.builders.line import", source)
        self.assertNotIn("line_builder_class", source)

    def test_marker_2_version_is_required(self) -> None:
        with patch("rag_cliente.pdf_loader.version", return_value="1.10.2"):
            with self.assertRaisesRegex(RuntimeError, "requiere marker-pdf 2.x"):
                _require_marker_2()

    def test_quality_profile_fails_before_marker_without_local_vlm_endpoint(self) -> None:
        settings = Settings(
            marker_profile="cpu-quality",
            marker_openai_base_url="",
        )

        with patch("rag_cliente.pdf_loader._require_marker_2") as marker_version:
            with self.assertRaisesRegex(RuntimeError, "local_llm_endpoint_required"):
                create_marker_converter(settings)

        marker_version.assert_not_called()

    def test_setup_accepts_auto_cpu_cuda_and_keeps_editable_install_dependency_free(self) -> None:
        setup_source = (Path(__file__).parents[1] / "setup.ps1").read_text(encoding="utf-8")

        self.assertIn('[ValidateSet("auto", "cpu", "cuda")]', setup_source)
        self.assertIn("pip install -e $ProjectRoot --no-deps", setup_source)


class Marker2MetadataTests(unittest.TestCase):
    def test_json_page_content_refs_fall_back_to_child_html(self) -> None:
        rendered = {
            "block_type": "Document",
            "metadata": {},
            "children": [
                {
                    "block_type": "Page",
                    "id": "/page/0/Page/0",
                    "html": '<content-ref src="/page/0/Text/0"></content-ref>',
                    "children": [
                        {
                            "block_type": "Text",
                            "id": "/page/0/Text/0",
                            "html": "<p>Texto real del PDF</p>",
                            "children": None,
                        }
                    ],
                }
            ],
        }

        chunks = _extract_marker_structured_chunks(rendered)

        self.assertEqual(chunks[0]["text"], "Texto real del PDF")

    def test_table_visual_fallback_marks_page_as_ocr_used(self) -> None:
        page = types.SimpleNamespace(
            text_extraction_method="pdftext",
            ocr_errors_detected=False,
            contained_blocks=lambda document: [
                types.SimpleNamespace(text_extraction_method="pdftext"),
                types.SimpleNamespace(text_extraction_method="surya"),
            ],
        )

        details = _collect_marker_page_extraction_details(page, object())

        self.assertTrue(details["ocr_used"])
        self.assertEqual(details["text_extraction_methods"], ["pdftext", "surya"])
        self.assertIn("visual_block_or_table_fallback", details["ocr_reasons"])

    def test_paginated_subset_keeps_human_page_number(self) -> None:
        markdown = "{17}" + "-" * 48 + "\n\ncontenido"

        self.assertEqual(_split_marker_markdown_by_page(markdown), [(18, "contenido")])

    def test_json_output_keeps_required_structured_fields(self) -> None:
        rendered = {
            "block_type": "Document",
            "metadata": {
                "document_type": "pdf",
                "page_stats": [{"page_id": 0, "ocr_used": True}],
            },
            "children": [
                {
                    "block_type": "Page",
                    "id": "/page/0/Page/0",
                    "html": "<p>Texto <strong>útil</strong></p>",
                    "polygon": [[0, 0], [100, 0], [100, 200], [0, 200]],
                    "children": [
                        {
                            "block_type": "Text",
                            "id": "/page/0/Text/0",
                            "text": "Texto útil",
                        }
                    ],
                    "section_hierarchy": {"1": "Introducción"},
                }
            ],
        }

        chunks = _extract_marker_structured_chunks(rendered)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            set(chunks[0]),
            {
                "block_type",
                "id",
                "html",
                "text",
                "page",
                "polygon",
                "children",
                "section_hierarchy",
                "extraction_metadata",
            },
        )
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["text"], "Texto útil")
        self.assertTrue(chunks[0]["extraction_metadata"]["page_stats"][0]["ocr_used"])


class GenericMarkerDocumentTests(unittest.TestCase):
    def test_marker_full_formats_share_the_generic_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name in ("manual.pdf", "informe.docx", "datos.xlsx", "slides.pptx", "libro.epub", "web.html"):
                (root / name).write_bytes(b"test")

            settings = Settings(marker_enabled=True)

            def fake_load(path, settings, marker_converter):
                return [
                    PageDocument(
                        document_id=path.stem,
                        source=path.name,
                        source_path=str(path.resolve()),
                        source_type=path.suffix.lstrip("."),
                        page_number=1,
                        text=path.name,
                    )
                ]

            with (
                patch("rag_cliente.pdf_loader.create_marker_converter", return_value=object()),
                patch("rag_cliente.pdf_loader.load_marker_document_pages", side_effect=fake_load) as loader,
            ):
                pages = load_documents_from_directory(root, settings=settings)

        self.assertEqual(len(pages), 6)
        self.assertEqual(loader.call_count, 6)
        self.assertEqual(
            {page.source_type for page in pages},
            {"pdf", "docx", "xlsx", "pptx", "epub", "html"},
        )


if __name__ == "__main__":
    unittest.main()
