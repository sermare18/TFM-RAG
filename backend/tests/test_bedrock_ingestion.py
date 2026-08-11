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
from rag_cliente.config import CLAUDE_SONNET_4_6_GLOBAL_MODEL_ID, Settings
from rag_cliente.indexer import MarkdownChunker


class FakeBedrockClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        content = kwargs["messages"][0]["content"]
        target_page = None
        for item in content:
            text = item.get("text", "") if isinstance(item, dict) else ""
            if text.startswith("TARGET PAGE ") and " IMAGE" in text:
                target_page = int(text.split()[2])
        if target_page is None:
            raise AssertionError("La llamada simulada no identifica la pagina objetivo")
        markdown = (
            f"# Pagina {target_page}\n\n"
            f"| A | B |\n|---|---|\n| {target_page} | dato |"
        )
        return {
            "stopReason": "end_turn",
            "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
            "output": {
                "message": {
                    "content": [{"text": json.dumps({"markdown": markdown})}]
                }
            }
        }


class FailAfterOnePageClient(FakeBedrockClient):
    def converse(self, **kwargs):
        if self.calls:
            self.calls.append(kwargs)
            raise RuntimeError("synthetic second page failure")
        return super().converse(**kwargs)


class IncompleteThenCompleteClient(FakeBedrockClient):
    def converse(self, **kwargs):
        if not self.calls:
            self.calls.append(kwargs)
            return {
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 5,
                    "totalTokens": 105,
                },
                "output": {
                    "message": {
                        "content": [
                            {"text": json.dumps({"markdown": "Texto cortado **"})}
                        ]
                    }
                },
            }
        return super().converse(**kwargs)


class AlwaysIncompleteClient(FakeBedrockClient):
    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 100, "outputTokens": 5, "totalTokens": 105},
            "output": {
                "message": {
                    "content": [
                        {"text": json.dumps({"markdown": "Texto cortado"})}
                    ]
                }
            },
        }


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


class MinuteQuotaError(RuntimeError):
    response = {
        "Error": {
            "Code": "ThrottlingException",
            "Message": "Too many tokens per minute, please wait before trying again.",
        }
    }


class MinuteQuotaClient:
    def converse(self, **_kwargs):
        raise MinuteQuotaError()


class ServiceUnavailableError(RuntimeError):
    response = {
        "Error": {
            "Code": "ServiceUnavailableException",
            "Message": "Bedrock is unable to process your request.",
        }
    }


class TransientThenCompleteClient(FakeBedrockClient):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def converse(self, **kwargs):
        if self.failures:
            self.calls.append(kwargs)
            self.failures -= 1
            raise ServiceUnavailableError()
        return super().converse(**kwargs)


class InferenceProfileError(RuntimeError):
    response = {
        "Error": {
            "Code": "ValidationException",
            "Message": "Use of this model requires an inference profile.",
        }
    }


class InferenceProfileClient:
    def converse(self, **_kwargs):
        raise InferenceProfileError()


class BedrockParserTests(unittest.TestCase):
    def make_settings(self, cache: Path, *, enabled: bool = True) -> Settings:
        return Settings(
            bedrock_enabled=enabled,
            aws_region="eu-west-1",
            bedrock_model_id=CLAUDE_SONNET_4_6_GLOBAL_MODEL_ID,
            bedrock_cache_dir=str(cache),
            bedrock_context_pages=4,
            bedrock_max_output_tokens=16384,
            model_supervision_enabled=False,
        )

    def test_quote_entities_are_restored_after_json_parsing(self) -> None:
        response = json.dumps(
            {"markdown": "Título **&quot;Organización del trabajo&quot;**"},
            ensure_ascii=False,
        )

        page = BedrockMarkdownParser._parse_target_response(response, 16)

        self.assertEqual(
            page.markdown,
            'Título **"Organización del trabajo"**',
        )

    def test_each_target_page_gets_one_call_with_a_four_page_context(self) -> None:
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
            self.assertEqual(len(client.calls), 5)
            image_counts = [
                sum("image" in block for block in call["messages"][0]["content"])
                for call in client.calls
            ]
            self.assertEqual(image_counts, [4, 4, 4, 4, 4])
            self.assertEqual(
                client.calls[0]["modelId"],
                CLAUDE_SONNET_4_6_GLOBAL_MODEL_ID,
            )
            self.assertEqual(
                client.calls[0]["inferenceConfig"]["maxTokens"],
                16384,
            )
            output_schema = json.loads(
                client.calls[0]["outputConfig"]["textFormat"]["structure"]
                ["jsonSchema"]["schema"]
            )
            self.assertEqual(output_schema["required"], ["markdown"])
            self.assertFalse(output_schema["additionalProperties"])
            self.assertEqual(
                output_schema["properties"]["markdown"],
                {"type": "string"},
            )
            self.assertIn("TARGET PAGE image is the primary", BEDROCK_SYSTEM_PROMPT)
            self.assertIn("HTML entity &quot;", BEDROCK_SYSTEM_PROMPT)
            self.assertIn(
                "Never emit literal straight or typographic double quotation marks",
                BEDROCK_SYSTEM_PROMPT,
            )

            target_labels = []
            for call in client.calls:
                labels = [
                    block.get("text", "")
                    for block in call["messages"][0]["content"]
                    if block.get("text", "").startswith("TARGET PAGE ")
                ]
                self.assertEqual(len(labels), 1)
                target_labels.append(int(labels[0].split()[2]))
                references = [
                    block.get("text", "")
                    for block in call["messages"][0]["content"]
                    if "<target_reference_text" in block.get("text", "")
                ]
                self.assertEqual(len(references), 1)
                self.assertIn(f'page="{target_labels[-1]}"', references[0])
            self.assertEqual(target_labels, [1, 2, 3, 4, 5])

    def test_sliding_window_crosses_old_batch_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            client = FakeBedrockClient()
            parser = BedrockMarkdownParser(self.make_settings(root / "cache"), client)
            parser._render_pdf = lambda _source: [
                (page, f"png-{page}".encode(), f"reference {page}")
                for page in range(1, 7)
            ]

            parser.parse_pdf(source)

            fourth_call_text = "\n".join(
                block.get("text", "")
                for block in client.calls[3]["messages"][0]["content"]
            )
            self.assertIn("CONTEXT ONLY PAGE 2 IMAGE", fourth_call_text)
            self.assertIn("CONTEXT ONLY PAGE 3 IMAGE", fourth_call_text)
            self.assertIn("TARGET PAGE 4 IMAGE", fourth_call_text)
            self.assertIn("CONTEXT ONLY PAGE 5 IMAGE", fourth_call_text)
            self.assertNotIn('target_reference_text page="3"', fourth_call_text)
            self.assertIn('target_reference_text page="4"', fourth_call_text)

    def test_preview_calls_only_selected_target_and_does_not_write_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            client = FakeBedrockClient()
            parser = BedrockMarkdownParser(self.make_settings(root / "cache"), client)
            parser._render_pdf = lambda _source: [
                (page, f"png-{page}".encode(), f"reference {page}")
                for page in range(1, 7)
            ]

            pages = parser.preview_pdf_pages(source, [4])

            self.assertEqual([page.page_number for page in pages], [4])
            self.assertEqual(len(client.calls), 1)
            call_text = "\n".join(
                block.get("text", "")
                for block in client.calls[0]["messages"][0]["content"]
            )
            self.assertIn("TARGET PAGE 4 IMAGE", call_text)
            self.assertIn("CONTEXT ONLY PAGE 2 IMAGE", call_text)
            self.assertIn("CONTEXT ONLY PAGE 3 IMAGE", call_text)
            self.assertIn("CONTEXT ONLY PAGE 5 IMAGE", call_text)
            markdown_path, manifest_path = parser._cache_paths(source)
            self.assertFalse(markdown_path.exists())
            self.assertFalse(manifest_path.exists())

    def test_incomplete_markdown_is_retried_once_then_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            client = IncompleteThenCompleteClient()
            parser = BedrockMarkdownParser(self.make_settings(root / "cache"), client)
            parser._render_pdf = lambda _source: [(1, b"png", "referencia corta")]

            document = parser.parse_pdf(source)

            self.assertEqual(len(client.calls), 2)
            self.assertIn("# Pagina 1", document.pages[0].markdown)

    def test_retry_uses_only_the_target_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = IncompleteThenCompleteClient()
            parser = BedrockMarkdownParser(self.make_settings(root / "cache"), client)
            window = [
                (page, f"png-{page}".encode(), f"referencia {page}")
                for page in range(1, 5)
            ]

            page = parser._invoke_target(window, 3)

            self.assertEqual(page.page_number, 3)
            image_counts = [
                sum("image" in block for block in call["messages"][0]["content"])
                for call in client.calls
            ]
            self.assertEqual(image_counts, [4, 1])
            retry_text = "\n".join(
                block.get("text", "")
                for block in client.calls[1]["messages"][0]["content"]
            )
            self.assertNotIn("CONTEXT ONLY", retry_text)
            self.assertIn("TARGET PAGE 3 IMAGE", retry_text)

    def test_two_incomplete_responses_skip_page_and_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            client = AlwaysIncompleteClient()
            parser = BedrockMarkdownParser(self.make_settings(root / "cache"), client)
            parser._render_pdf = lambda _source: [(1, b"png", "referencia corta")]

            document = parser.parse_pdf(source)

            self.assertEqual(len(client.calls), 2)
            self.assertEqual(len(document.pages), 1)
            self.assertEqual(document.pages[0].markdown, "")
            self.assertIn(1, document.metadata["failed_pages"])
            markdown_path, manifest_path = parser._cache_paths(source)
            self.assertTrue(markdown_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertIn("1", manifest["failed_pages"])

    def test_low_reference_coverage_is_rejected(self) -> None:
        reference = " ".join(f"palabra{index}" for index in range(60))
        markdown = " ".join(f"palabra{index}" for index in range(10))

        issue = BedrockMarkdownParser._reference_coverage_issue(markdown, reference)

        self.assertIsNotNone(issue)
        self.assertIn("cobertura", issue or "")

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

    def test_partial_cache_resumes_after_last_completed_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            settings = self.make_settings(root / "cache")
            rendered = [
                (page, f"png-{page}".encode(), f"reference {page}")
                for page in range(1, 6)
            ]

            failing = BedrockMarkdownParser(settings, FailAfterOnePageClient())
            failing._render_pdf = lambda _source: rendered
            with self.assertRaisesRegex(RuntimeError, "second page failure"):
                failing.parse_pdf(source)

            _markdown_path, manifest_path = failing._cache_paths(source)
            partial = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["page_numbers"], [1])

            resumed_client = FakeBedrockClient()
            resumed = BedrockMarkdownParser(settings, resumed_client)
            resumed._render_pdf = lambda _source: rendered
            document = resumed.parse_pdf(source)

            self.assertEqual([page.page_number for page in document.pages], [1, 2, 3, 4, 5])
            self.assertEqual(len(resumed_client.calls), 4)
            sent_text = "\n".join(
                block.get("text", "")
                for block in resumed_client.calls[0]["messages"][0]["content"]
            )
            self.assertIn("TARGET PAGE 2 IMAGE", sent_text)
            self.assertIn("CONTEXT ONLY PAGE 1 IMAGE", sent_text)
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

    def test_minute_quota_error_preserves_completed_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            parser = BedrockMarkdownParser(
                self.make_settings(root / "cache"),
                MinuteQuotaClient(),
            )
            parser._render_pdf = lambda _source: [(1, b"png", "reference")]

            with self.assertRaisesRegex(RuntimeError, "tokens por minuto"):
                parser.parse_pdf(source)

    def test_service_unavailable_retries_with_exponential_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            client = TransientThenCompleteClient(failures=2)
            delays: list[float] = []
            progress: list[str] = []
            parser = BedrockMarkdownParser(
                self.make_settings(root / "cache"),
                client,
                sleep=delays.append,
                jitter=lambda: 0.0,
            )
            parser._render_pdf = lambda _source: [(1, b"png", "reference")]

            document = parser.parse_pdf(source, progress_callback=progress.append)

            self.assertEqual(document.pages[0].page_number, 1)
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(delays, [0.5, 1.0])
            self.assertTrue(
                any("reintento 1/5" in message for message in progress),
                progress,
            )
            self.assertTrue(
                any("reintento 2/5" in message for message in progress),
                progress,
            )

    def test_inference_profile_error_shows_the_required_global_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "document.pdf"
            source.write_bytes(b"synthetic-pdf")
            parser = BedrockMarkdownParser(
                self.make_settings(root / "cache"),
                InferenceProfileClient(),
            )
            parser._render_pdf = lambda _source: [(1, b"png", "reference")]

            with self.assertRaisesRegex(
                RuntimeError,
                CLAUDE_SONNET_4_6_GLOBAL_MODEL_ID,
            ):
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
