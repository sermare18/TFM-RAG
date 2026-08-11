from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_cliente.bedrock_parser import (
    BEDROCK_SYSTEM_PROMPT,
    BedrockMarkdownParser,
    MarkdownDocument,
    MarkdownPage,
)
from rag_cliente.config import Settings
from rag_cliente.indexer import MarkdownChunker


class FakeBedrockClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        content = kwargs["messages"][0]["content"]
        pages = []
        for item in content:
            text = item.get("text", "") if isinstance(item, dict) else ""
            if text.startswith("PAGE ") and " IMAGE" in text:
                page = int(text.split()[1])
                pages.append({"page": page, "markdown": f"# Pagina {page}\n\n| A | B |\n|---|---|\n| {page} | dato |"})
        return {
            "output": {
                "message": {
                    "content": [{"text": json.dumps({"pages": pages})}]
                }
            }
        }


class FailAfterOneBatchClient(FakeBedrockClient):
    def converse(self, **kwargs):
        if self.calls:
            self.calls.append(kwargs)
            raise RuntimeError("synthetic second batch failure")
        return super().converse(**kwargs)


class DailyQuotaError(RuntimeError):
    response = {
        "Error": {
            "Code": "ThrottlingException",
            "Message": "Too many tokens per day, please wait before trying again.",
        }
    }


class DailyQuotaClient:
    def converse(self, **_kwargs):
        raise DailyQuotaError()


class BedrockParserTests(unittest.TestCase):
    def make_settings(self, cache: Path, *, enabled: bool = True) -> Settings:
        return Settings(
            bedrock_enabled=enabled,
            aws_region="eu-west-1",
            bedrock_model_id="test-model",
            bedrock_cache_dir=str(cache),
            bedrock_pages_per_batch=4,
            model_supervision_enabled=False,
        )

    def test_five_pages_are_sent_as_four_plus_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf-for-mocked-render")
            client = FakeBedrockClient()
            parser = BedrockMarkdownParser(self.make_settings(root / "cache"), client)
            parser._render_pdf = lambda _source: [
                (page, f"png-{page}".encode(), f"reference {page}")
                for page in range(1, 6)
            ]

            document = parser.parse_pdf(source)

            self.assertEqual([page.page_number for page in document.pages], [1, 2, 3, 4, 5])
            self.assertEqual(len(client.calls), 2)
            image_counts = [
                sum("image" in block for block in call["messages"][0]["content"])
                for call in client.calls
            ]
            self.assertEqual(image_counts, [4, 1])
            self.assertIn("primary and authoritative", BEDROCK_SYSTEM_PROMPT)
            first_content = client.calls[0]["messages"][0]["content"]
            self.assertTrue(any("<reference_text" in block.get("text", "") for block in first_content))

    def test_valid_cache_is_reused_while_bedrock_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"same-source")
            settings = self.make_settings(root / "cache")
            first_client = FakeBedrockClient()
            first = BedrockMarkdownParser(settings, first_client)
            first._render_pdf = lambda _source: [(1, b"png", "reference")]
            first.parse_pdf(source)

            disabled = self.make_settings(root / "cache", enabled=False)
            second_client = FakeBedrockClient()
            cached = BedrockMarkdownParser(disabled, second_client).parse_pdf(source)

            self.assertTrue(cached.metadata["cache_hit"])
            self.assertEqual(second_client.calls, [])

    def test_partial_cache_resumes_after_last_completed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            settings = self.make_settings(root / "cache")
            rendered = [
                (page, f"png-{page}".encode(), f"reference {page}")
                for page in range(1, 6)
            ]

            failing = BedrockMarkdownParser(settings, FailAfterOneBatchClient())
            failing._render_pdf = lambda _source: rendered
            with self.assertRaisesRegex(RuntimeError, "second batch failure"):
                failing.parse_pdf(source)

            _markdown_path, manifest_path = failing._cache_paths(source)
            partial = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["page_numbers"], [1, 2, 3, 4])

            resumed_client = FakeBedrockClient()
            resumed = BedrockMarkdownParser(settings, resumed_client)
            resumed._render_pdf = lambda _source: rendered
            document = resumed.parse_pdf(source)

            self.assertEqual([page.page_number for page in document.pages], [1, 2, 3, 4, 5])
            self.assertEqual(len(resumed_client.calls), 1)
            sent_text = "\n".join(
                block.get("text", "")
                for block in resumed_client.calls[0]["messages"][0]["content"]
            )
            self.assertIn("PAGE 5 IMAGE", sent_text)
            self.assertNotIn("PAGE 4 IMAGE", sent_text)
            complete = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(complete["complete"])

    def test_daily_quota_error_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            parser = BedrockMarkdownParser(
                self.make_settings(root / "cache"),
                DailyQuotaClient(),
            )
            parser._render_pdf = lambda _source: [(1, b"png", "reference")]

            with self.assertRaisesRegex(RuntimeError, "cuota diaria"):
                parser.parse_pdf(source)

    def test_pdf_without_cache_fails_before_any_call_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"pdf")
            client = FakeBedrockClient()
            parser = BedrockMarkdownParser(
                self.make_settings(root / "cache", enabled=False), client
            )
            with self.assertRaisesRegex(RuntimeError, "BEDROCK_ENABLED=false"):
                parser.parse_pdf(source)
            self.assertEqual(client.calls, [])

    def test_page_budget_fails_before_a_paid_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"pdf")
            client = FakeBedrockClient()
            settings = self.make_settings(root / "cache").model_copy(
                update={"bedrock_max_pages_per_document": 4}
            )
            parser = BedrockMarkdownParser(settings, client)
            parser._render_pdf = lambda _source: [
                (page, b"png", "") for page in range(1, 6)
            ]
            with self.assertRaisesRegex(RuntimeError, "MAX_PAGES_PER_DOCUMENT"):
                parser.parse_pdf(source)
            self.assertEqual(client.calls, [])

    def test_markdown_page_separators_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "notes.md"
            source.write_text(
                "<!-- PAGE 3 -->\n# Tres\n\n<!-- PAGE 4 -->\n# Cuatro\n",
                encoding="utf-8",
            )
            document = BedrockMarkdownParser(Settings()).load_markdown(source)
            self.assertEqual([page.page_number for page in document.pages], [3, 4])
            self.assertEqual(document.parser_model, "direct-markdown")

    def test_directory_supports_only_pdf_and_markdown_and_derives_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tagged = root / "guias"
            tagged.mkdir()
            (tagged / "guide.md").write_text("contenido", encoding="utf-8")
            (tagged / "ignored.txt").write_text("ignorado", encoding="utf-8")
            documents = BedrockMarkdownParser(Settings()).load_directory(root)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].tag, "guias")


class MarkdownChunkerTests(unittest.TestCase):
    @staticmethod
    def document(markdown_pages: list[str]) -> MarkdownDocument:
        return MarkdownDocument(
            document_id="doc-1",
            source="doc.pdf",
            source_path="C:/docs/doc.pdf",
            source_type="pdf",
            pages=[MarkdownPage(index, text) for index, text in enumerate(markdown_pages, 1)],
            source_sha256="abc",
            parser_model="test-model",
            prompt_version="v1",
        )

    def test_chunks_never_cross_page_boundaries(self) -> None:
        settings = Settings(
            chunk_target_tokens=5,
            chunk_max_tokens=8,
            chunk_overlap_tokens=2,
        )
        chunks = MarkdownChunker(settings).chunk_documents(
            [self.document(["uno dos tres cuatro cinco seis\n\nsiete ocho", "pagina dos"])]
        )
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.page_start == chunk.page_end for chunk in chunks))
        self.assertEqual({tuple(chunk.source_pages) for chunk in chunks}, {(1,), (2,)})

    def test_overlap_alone_does_not_create_a_duplicate_final_chunk(self) -> None:
        settings = Settings(
            chunk_target_tokens=3,
            chunk_max_tokens=20,
            chunk_overlap_tokens=2,
        )
        page = self.document(["uno dos tres\n"])
        chunks = MarkdownChunker(settings).chunk_documents([page])
        self.assertEqual(len(chunks), 1)

    def test_single_oversized_line_is_preserved_and_flagged(self) -> None:
        settings = Settings(chunk_target_tokens=3, chunk_max_tokens=5, chunk_overlap_tokens=1)
        long_line = " ".join(f"token{index}" for index in range(12))
        chunks = MarkdownChunker(settings).chunk_documents([self.document([long_line])])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, long_line)
        self.assertTrue(chunks[0].oversize)

    def test_chunk_ids_are_deterministic(self) -> None:
        chunker = MarkdownChunker(Settings())
        document = self.document(["# Titulo\n\nContenido"])
        first = chunker.chunk_documents([document])
        second = chunker.chunk_documents([document])
        self.assertEqual([item.chunk_id for item in first], [item.chunk_id for item in second])


if __name__ == "__main__":
    unittest.main()
