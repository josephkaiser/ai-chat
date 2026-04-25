import pathlib
import tempfile
import unittest
import zipfile

from src.python.ai_chat.prompts import DEFAULT_SYSTEM_PROMPT, TOOL_USE_SYSTEM_PROMPT, DEEP_BUILD_SYSTEM_PROMPT, DEEP_INSPECT_SYSTEM_PROMPT
import src.python.ai_chat.workspace_reader as workspace_reader


class AppBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = pathlib.Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_text(self, rel_path: str, content: str) -> pathlib.Path:
        target = self.workspace / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def _write_bytes(self, rel_path: str, content: bytes) -> pathlib.Path:
        target = self.workspace / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def test_text_like_files_return_raw_text_for_http_and_tool_reads(self):
        fixtures = {
            "DESIGN.md": ("# Design\nhello\n", "markdown"),
            "config.json": ('{"ok": true}\n', "text"),
            "main.py": ("print('hi')\n", "text"),
            "notes.txt": ("plain text\n", "text"),
        }
        for rel_path, (content, content_kind) in fixtures.items():
            target = self._write_text(rel_path, content)
            payload = workspace_reader.build_workspace_file_result(
                target,
                rel_path=rel_path,
                max_bytes=1024 * 1024,
                document_preview_builder=lambda *_args, **_kwargs: {},
                text_limit=None,
                truncate_output_func=lambda text, _limit: text,
            )
            self.assertEqual(payload["content"], content)
            self.assertEqual(payload["content_kind"], content_kind)
            self.assertTrue(payload["editable"])
            self.assertEqual(payload["default_view"], "preview")
            self.assertFalse(payload["truncated"])

    def test_pdf_reader_keeps_document_preview_behavior(self):
        target = self._write_bytes("report.pdf", b"%PDF-1.4\n")
        preview_payload = {
            "path": "report.pdf",
            "content": "Preview excerpt",
            "size": 15,
            "lines": 1,
            "file_type": "pdf",
            "extractor": "stub",
            "page_count": 1,
            "title": "report.pdf",
            "metadata": {},
        }
        payload = workspace_reader.build_workspace_file_result(
            target,
            rel_path="report.pdf",
            max_bytes=1024 * 1024,
            document_preview_builder=lambda *_args, **_kwargs: dict(preview_payload),
            text_limit=40000,
            truncate_output_func=lambda text, _limit: text,
        )
        self.assertEqual(payload["content"], "Preview excerpt")
        self.assertEqual(payload["content_kind"], "pdf")
        self.assertFalse(payload["editable"])
        self.assertEqual(payload["default_view"], "preview")

    def test_pdf_reader_falls_back_to_inline_preview_when_extraction_is_unavailable(self):
        target = self._write_bytes("report.pdf", b"%PDF-1.4\n")
        payload = workspace_reader.build_workspace_file_result(
            target,
            rel_path="report.pdf",
            max_bytes=1024 * 1024,
            document_preview_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("pdftotext unavailable")),
            text_limit=40000,
            truncate_output_func=lambda text, _limit: text,
        )
        self.assertEqual(payload["path"], "report.pdf")
        self.assertEqual(payload["content"], "")
        self.assertEqual(payload["content_kind"], "pdf")
        self.assertFalse(payload["editable"])
        self.assertEqual(payload["default_view"], "preview")
        self.assertEqual(payload["metadata"].get("preview_error"), "pdftotext unavailable")

    def test_rtf_reader_uses_document_preview_mode_and_stays_read_only(self):
        target = self._write_text("notes.rtf", "{\\rtf1\\ansi Hello world}")
        preview_payload = {
            "path": "notes.rtf",
            "content": "Hello world",
            "size": 11,
            "lines": 1,
            "file_type": "rtf",
            "extractor": "rtf-parser",
            "page_count": None,
            "title": "notes.rtf",
            "metadata": {},
        }
        payload = workspace_reader.build_workspace_file_result(
            target,
            rel_path="notes.rtf",
            max_bytes=1024 * 1024,
            document_preview_builder=lambda *_args, **_kwargs: dict(preview_payload),
            text_limit=40000,
            truncate_output_func=lambda text, _limit: text,
        )
        self.assertEqual(payload["content"], "Hello world")
        self.assertEqual(payload["content_kind"], "text")
        self.assertFalse(payload["editable"])
        self.assertEqual(payload["default_view"], "preview")

    def test_zip_reader_returns_archive_preview_entries(self):
        target = self.workspace / "bundle.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("src/main.py", "print('hi')\n")
            archive.writestr("README.md", "# Hello\n")
        payload = workspace_reader.build_workspace_file_result(
            target,
            rel_path="bundle.zip",
            max_bytes=1024 * 1024,
            document_preview_builder=lambda *_args, **_kwargs: {},
            text_limit=None,
            truncate_output_func=lambda text, _limit: text,
        )
        self.assertEqual(payload["content_kind"], "archive")
        self.assertFalse(payload["editable"])
        self.assertEqual(payload["file_type"], "zip")
        self.assertGreaterEqual(payload["entry_count"], 2)
        self.assertIn("src/main.py", payload["content"])

    def test_image_files_open_as_non_editable_binary_previews(self):
        target = self._write_bytes("plots/chart.png", b"\x89PNG\r\n\x1a\n")
        payload = workspace_reader.build_workspace_file_result(
            target,
            rel_path="plots/chart.png",
            max_bytes=1024 * 1024,
            document_preview_builder=lambda *_args, **_kwargs: {},
            text_limit=None,
            truncate_output_func=lambda text, _limit: text,
        )
        self.assertEqual(payload["path"], "plots/chart.png")
        self.assertEqual(payload["content"], "")
        self.assertEqual(payload["content_kind"], "image")
        self.assertEqual(payload["file_type"], "binary")
        self.assertEqual(payload["media_type"], "image/png")
        self.assertFalse(payload["editable"])
        self.assertEqual(payload["default_view"], "preview")
        self.assertFalse(payload["truncated"])

    def test_oversized_text_file_returns_clear_reader_error(self):
        target = self._write_text("big.txt", "a" * ((1024 * 1024) + 1))
        with self.assertRaises(ValueError) as exc_info:
            workspace_reader.build_workspace_file_result(
                target,
                rel_path="big.txt",
                max_bytes=1024 * 1024,
                document_preview_builder=lambda *_args, **_kwargs: {},
                text_limit=None,
                truncate_output_func=lambda text, _limit: text,
            )
        self.assertIn("File too large", str(exc_info.exception))

    def test_hard_limit_message_drops_step_confirmation_language(self):
        message = workspace_reader.build_tool_loop_hard_limit_message("Updated `plan.json` and inspected `DESIGN.md`.")
        self.assertIn("Paused after reaching the current tool budget.", message)
        self.assertIn("Say continue", message)
        self.assertNotIn("Would you like me to continue", message)

    def test_default_prompt_no_longer_instructs_step_by_step_confirmation(self):
        self.assertNotIn("ask a short yes-or-no question", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Keep the visible answer brief", DEFAULT_SYSTEM_PROMPT)

    def test_prompts_favor_context_window_aware_progress_over_tiny_changes(self):
        combined = "\n".join((DEFAULT_SYSTEM_PROMPT, TOOL_USE_SYSTEM_PROMPT, DEEP_BUILD_SYSTEM_PROMPT))
        self.assertNotIn("smallest useful", combined)
        self.assertNotIn("shortest useful tool sequence", combined)
        self.assertIn("context window is limited", combined)
        self.assertIn("durable progress", combined)
        self.assertIn("highest-leverage next tool call", TOOL_USE_SYSTEM_PROMPT)
        self.assertIn("make measurable progress or finish", TOOL_USE_SYSTEM_PROMPT)
        self.assertIn("use it instead of claiming you cannot run code", TOOL_USE_SYSTEM_PROMPT)
        self.assertIn("instead of giving local setup or run instructions back to the user", TOOL_USE_SYSTEM_PROMPT)
        self.assertIn("specific output shape", DEEP_BUILD_SYSTEM_PROMPT)
        self.assertIn("more illustrative output artifact", DEEP_BUILD_SYSTEM_PROMPT)
        self.assertIn("short sequence, table, chart", TOOL_USE_SYSTEM_PROMPT)
        self.assertIn("short sequence or quick chart", DEEP_BUILD_SYSTEM_PROMPT)
        self.assertIn("ask one short clarifying question", DEEP_INSPECT_SYSTEM_PROMPT)
        self.assertIn("Do not hand execution back to the user", DEEP_BUILD_SYSTEM_PROMPT)
        self.assertIn("Match the scale of the change to the current step", DEEP_BUILD_SYSTEM_PROMPT)
        self.assertIn("managed chat-scoped Python environment", TOOL_USE_SYSTEM_PROMPT)
        self.assertIn("save it as a workspace file", TOOL_USE_SYSTEM_PROMPT)
        self.assertIn("responsive layouts that fit narrow panes and phones", TOOL_USE_SYSTEM_PROMPT)
        self.assertIn("workspace viewer, normal desktop browsers, and phones", DEEP_BUILD_SYSTEM_PROMPT)
        self.assertNotIn(".venv", TOOL_USE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
