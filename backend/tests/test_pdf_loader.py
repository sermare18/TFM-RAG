from __future__ import annotations

import tempfile
import types
import unittest
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

# El otro módulo de pruebas instala dobles ligeros para FastAPI. Yo retiro solo
# el doble del loader para comprobar aquí la implementación real de Marker.
sys.modules.pop("rag_cliente.pdf_loader", None)
importlib.invalidate_caches()

from rag_cliente.config import Settings
from rag_cliente.pdf_loader import (
    PageDocument,
    _build_marker_config,
    _collect_marker_page_extraction_details,
    _install_surya_windows_cleanup_workaround,
    _require_marker_2,
    _split_marker_markdown_by_page,
    create_marker_converter,
    load_documents_from_directory,
)


class Marker2ConfigurationTests(unittest.TestCase):
    def test_balanced_mode_keeps_ocr_adaptive_by_default(self) -> None:
        config = _build_marker_config(Settings())

        self.assertEqual(config["mode"], "balanced")
        self.assertFalse(config["disable_ocr"])
        self.assertEqual(config["min_recon_score"], 0.75)
        self.assertTrue(config["force_ocr_complex_layout"])
        self.assertNotIn("force_ocr", config)

    def test_single_ocr_mode_maps_to_marker_flags(self) -> None:
        adaptive = _build_marker_config(Settings(marker_ocr_mode="adaptive"))
        forced = _build_marker_config(Settings(marker_ocr_mode="force"))
        disabled = _build_marker_config(Settings(marker_ocr_mode="disabled"))

        self.assertFalse(adaptive["disable_ocr"])
        self.assertNotIn("force_ocr", adaptive)
        self.assertTrue(forced["force_ocr"])
        self.assertFalse(forced["disable_ocr"])
        self.assertTrue(disabled["disable_ocr"])
        self.assertNotIn("force_ocr", disabled)

    def test_marker_2_version_is_required(self) -> None:
        with patch("rag_cliente.pdf_loader.version", return_value="1.10.2"):
            with self.assertRaisesRegex(RuntimeError, "requiere marker-pdf 2.x"):
                _require_marker_2()

    def test_missing_llama_cpp_binary_is_reported_before_model_start(self) -> None:
        settings = Settings(
            marker_inference_backend="llamacpp",
            marker_llama_cpp_binary="Z:/missing/llama-server.exe",
        )

        with patch("rag_cliente.pdf_loader._require_marker_2", return_value="2.0.0"):
            with self.assertRaisesRegex(RuntimeError, "no existe"):
                create_marker_converter(settings)

    @unittest.skipUnless(os.name == "nt", "Este ajuste solo se aplica en Windows")
    def test_windows_cleanup_targets_only_the_spawned_server_pid(self) -> None:
        from surya.inference.backends import spawn as surya_spawn

        original_stop_process = surya_spawn._stop_process
        completed = types.SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
        try:
            with patch("rag_cliente.pdf_loader.subprocess.run", return_value=completed) as taskkill:
                _install_surya_windows_cleanup_workaround()
                surya_spawn._stop_process(12345, "llamacpp")

            # Compruebo que cierro únicamente el árbol del PID creado por Surya.
            self.assertEqual(taskkill.call_args.args[0][:4], ["taskkill", "/PID", "12345", "/T"])
            self.assertIn("/F", taskkill.call_args.args[0])
        finally:
            surya_spawn._stop_process = original_stop_process


class Marker2MetadataTests(unittest.TestCase):
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
