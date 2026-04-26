import asyncio
import os
import pathlib
import tempfile
import unittest
import zipfile

os.environ.setdefault("MODEL_NAME", "test-model")

import app
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

    def test_html_preview_injects_responsive_shell(self):
        html = "<!DOCTYPE html><html><head><title>Demo</title></head><body><h1>Hello</h1></body></html>"
        rendered = app.build_responsive_html_preview(html)
        self.assertIn('id="codex-responsive-preview"', rendered)
        self.assertIn('name="viewport"', rendered)
        self.assertIn("<title>Demo</title>", rendered)
        self.assertIn("<h1>Hello</h1>", rendered)

    def test_render_workspace_file_preview_response_wraps_html_without_touching_other_files(self):
        self._write_text("site/index.html", "<html><body><a href=\"about.html\">About</a></body></html>")
        css_target = self._write_text("site/style.css", "body { color: red; }\n")

        html_response = app.render_workspace_file_preview_response(self.workspace, "site/index.html")
        self.assertIsInstance(html_response, app.HTMLResponse)

        css_response = app.render_workspace_file_preview_response(self.workspace, "site/style.css")
        self.assertIsInstance(css_response, app.FileResponse)
        response_path = getattr(css_response, "path", None) or getattr(css_response, "content", None)
        self.assertEqual(pathlib.Path(response_path).resolve(), css_target.resolve())

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

    @unittest.skipIf(app.pd is None, "pandas is not installed")
    def test_mislabeled_xlsx_falls_back_to_delimited_preview(self):
        target = self._write_text("table.xlsx", "Name,Age,City\nAlice,30,Paris\nBob,25,Berlin\n")
        payload = app.load_spreadsheet_summary(target)
        self.assertEqual(payload["file_type"], "csv")
        self.assertEqual(payload["row_count"], 2)
        self.assertEqual(payload["column_count"], 3)
        self.assertEqual(payload["preview_rows"][0]["Name"], "Alice")
        self.assertIn("format_warning", payload)

    def test_hard_limit_message_drops_step_confirmation_language(self):
        message = workspace_reader.build_tool_loop_hard_limit_message("Updated `plan.json` and inspected `DESIGN.md`.")
        self.assertIn("Paused after reaching the current tool budget.", message)
        self.assertIn("Say continue", message)
        self.assertNotIn("Would you like me to continue", message)

    def test_patch_mismatch_followup_message_tells_model_to_reread_file(self):
        message = app.build_tool_result_followup_message({
            "id": "call_patch_1",
            "ok": False,
            "error": "edit 1 expected 1 match(es) for old_text, found 0",
            "details": {
                "type": "patch_mismatch",
                "path": "src/web/index.html",
            },
        })

        self.assertIn("<tool_result>", message)
        self.assertIn("Read the live file again before attempting another patch.", message)
        self.assertIn("Do not reuse stale old_text.", message)
        self.assertIn("src/web/index.html", message)

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
        self.assertIn("workspace.inspect_document", TOOL_USE_SYSTEM_PROMPT)
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

    def test_continue_prepared_turn_in_workspace_execution_uses_build_flow(self):
        original_send_activity_event = app.send_activity_event
        original_orchestrated_chat = app.orchestrated_chat

        class DummyWebSocket:
            async def send_json(self, _payload):
                return None

        recorded_events = []

        async def fake_send_activity_event(websocket, phase, title, content, **kwargs):
            recorded_events.append((phase, title, content))

        async def fake_orchestrated_chat(
            websocket,
            conversation_id,
            message,
            history,
            system_prompt,
            max_tokens,
            features,
            auto_execute=False,
            plan_override_builder_steps=None,
            workflow_execution=None,
        ):
            self.assertEqual(conversation_id, "conv-escalate")
            self.assertEqual(message, "show me a one-file toy operating system example")
            self.assertTrue(auto_execute)
            self.assertEqual(plan_override_builder_steps, [])
            return "Created `toy_kernel.c` in the workspace."

        prepared = app.PreparedTurnRequest(
            conversation_id="conv-escalate",
            active_file_path="",
            turn_kind=app.TURN_KIND_VISIBLE_CHAT,
            user_message_id=1,
            saved_user_message="show me a one-file toy operating system example",
            effective_message="show me a one-file toy operating system example",
            model_message="show me a one-file toy operating system example",
            history=[],
            model_history=[],
            system_prompt="system",
            requested_mode="auto",
            resolved_mode="chat",
            features=app.FeatureFlags(agent_tools=True, workspace_write=True),
            slash_command=None,
            max_tokens=1024,
            workspace_intent="none",
            tool_policy_trace={},
            enabled_tools=["conversation.search_history"],
            auto_execute_workspace=False,
            resume_saved_workspace=False,
            plan_override_builder_steps=[],
            promoted_to_planning=False,
            repo_bootstrapped=False,
            repo_bootstrap_summary="",
            assessment=app.TurnAssessment(
                execution_style="direct_answer",
                workspace_intent="none",
                enabled_tools=["conversation.search_history"],
            ),
            attachment_paths=[],
        )

        try:
            app.send_activity_event = fake_send_activity_event
            app.orchestrated_chat = fake_orchestrated_chat
            response = asyncio.run(
                app.continue_prepared_turn_in_workspace_execution(
                    DummyWebSocket(),
                    prepared,
                )
            )
        finally:
            app.send_activity_event = original_send_activity_event
            app.orchestrated_chat = original_orchestrated_chat

        self.assertEqual(response, "Created `toy_kernel.c` in the workspace.")
        self.assertEqual(
            recorded_events,
            [(
                "evaluate",
                "Escalate",
                "Switching from answer mode into workspace execution because the next step needs file changes.",
            )],
        )

    def test_process_chat_turn_escalates_workspace_upgrade_instead_of_leaking_internal_message(self):
        original_prepare_turn_request = app.prepare_turn_request
        original_run_resumable_tool_loop = app.run_resumable_tool_loop
        original_orchestrated_chat = app.orchestrated_chat
        original_create_workflow_execution = app.create_workflow_execution
        original_complete_successful_turn = app.complete_successful_turn

        class DummyWebSocket:
            def __init__(self):
                self.sent = []

            async def send_json(self, payload):
                self.sent.append(payload)

        prepared = app.PreparedTurnRequest(
            conversation_id="conv-upgrade",
            active_file_path="",
            turn_kind=app.TURN_KIND_VISIBLE_CHAT,
            user_message_id=1,
            saved_user_message="like just a 1 file example operating system toy example?",
            effective_message="like just a 1 file example operating system toy example?",
            model_message="like just a 1 file example operating system toy example?",
            history=[],
            model_history=[],
            system_prompt="system",
            requested_mode="auto",
            resolved_mode="chat",
            features=app.FeatureFlags(agent_tools=True, workspace_write=True),
            slash_command=None,
            max_tokens=1024,
            workspace_intent="none",
            tool_policy_trace={},
            enabled_tools=["conversation.search_history"],
            auto_execute_workspace=False,
            resume_saved_workspace=False,
            plan_override_builder_steps=[],
            promoted_to_planning=False,
            repo_bootstrapped=False,
            repo_bootstrap_summary="",
            assessment=app.TurnAssessment(
                execution_style="direct_answer",
                workspace_intent="none",
                enabled_tools=["conversation.search_history"],
            ),
            attachment_paths=[],
        )

        async def fake_prepare_turn_request(_data):
            return prepared

        async def fake_run_resumable_tool_loop(*args, **kwargs):
            return app.ToolLoopOutcome(
                final_text=(
                    "The next step needs workspace execution because it requested workspace.patch_file. "
                    "Switch into the build flow and continue from the current task."
                ),
                requested_phase_upgrade="workspace_execution",
                requested_tool_name="workspace.patch_file",
            )

        async def fake_orchestrated_chat(
            websocket,
            conversation_id,
            message,
            history,
            system_prompt,
            max_tokens,
            features,
            auto_execute=False,
            plan_override_builder_steps=None,
            workflow_execution=None,
        ):
            self.assertEqual(conversation_id, "conv-upgrade")
            self.assertTrue(auto_execute)
            return "Created `toy_kernel.c` in the workspace."

        def fake_create_workflow_execution(*args, **kwargs):
            return None

        async def fake_complete_successful_turn(*args, **kwargs):
            return None

        websocket = DummyWebSocket()
        try:
            app.prepare_turn_request = fake_prepare_turn_request
            app.run_resumable_tool_loop = fake_run_resumable_tool_loop
            app.orchestrated_chat = fake_orchestrated_chat
            app.create_workflow_execution = fake_create_workflow_execution
            app.complete_successful_turn = fake_complete_successful_turn
            asyncio.run(
                app.process_chat_turn(
                    websocket,
                    {
                        "conversation_id": "conv-upgrade",
                        "message": "like just a 1 file example operating system toy example?",
                    },
                )
            )
        finally:
            app.prepare_turn_request = original_prepare_turn_request
            app.run_resumable_tool_loop = original_run_resumable_tool_loop
            app.orchestrated_chat = original_orchestrated_chat
            app.create_workflow_execution = original_create_workflow_execution
            app.complete_successful_turn = original_complete_successful_turn

        final_replacements = [payload for payload in websocket.sent if payload.get("type") == "final_replace"]
        self.assertEqual(len(final_replacements), 1)
        self.assertEqual(final_replacements[0]["content"], "Created `toy_kernel.c` in the workspace.")
        joined_payloads = "\n".join(str(payload) for payload in websocket.sent)
        self.assertNotIn("The next step needs workspace execution", joined_payloads)

    def test_resolve_route_intake_for_turn_uses_structured_llm_pass(self):
        original_infer = app.infer_structured_route_intake_via_llm
        try:
            async def fake_infer(**kwargs):
                self.assertEqual(kwargs["message"], "can you summarize the changes to nvim 0.12")
                return app.StructuredRouteIntake(
                    needs_fresh_info=True,
                    is_versioned_release_query=True,
                    entity="nvim",
                    time_sensitivity="versioned",
                    answer_shape="summary",
                    needs_search_citations=True,
                    web_search_requested=True,
                    reasoning="versioned release summary",
                    confidence=0.94,
                )

            app.infer_structured_route_intake_via_llm = fake_infer
            route = asyncio.run(
                app.resolve_route_intake_for_turn(
                    "conv-route",
                    "can you summarize the changes to nvim 0.12",
                    app.FeatureFlags(agent_tools=True, web_search=True),
                    history=[],
                )
            )
        finally:
            app.infer_structured_route_intake_via_llm = original_infer

        self.assertTrue(route.web_search_requested)
        self.assertTrue(route.needs_fresh_info)
        self.assertTrue(route.is_versioned_release_query)
        self.assertEqual(route.entity, "nvim")
        self.assertEqual(route.answer_shape, "summary")

    def test_resolve_route_intake_for_turn_preserves_explicit_web_search_rail(self):
        original_infer = app.infer_structured_route_intake_via_llm
        try:
            async def fake_infer(**kwargs):
                return app.StructuredRouteIntake(
                    answer_shape="summary",
                    reasoning="bad miss",
                    confidence=0.71,
                )

            app.infer_structured_route_intake_via_llm = fake_infer
            route = asyncio.run(
                app.resolve_route_intake_for_turn(
                    "conv-route",
                    "search the web for current sources on nvim 0.12 changes",
                    app.FeatureFlags(agent_tools=True, web_search=True),
                    history=[],
                )
            )
        finally:
            app.infer_structured_route_intake_via_llm = original_infer

        self.assertTrue(route.web_search_requested)
        self.assertTrue(route.needs_fresh_info)

    def test_route_intake_allows_workspace_execution_only_for_workspace_work(self):
        self.assertFalse(
            app.route_intake_allows_workspace_execution(
                app.StructuredRouteIntake(
                    needs_fresh_info=True,
                    web_search_requested=True,
                    answer_shape="summary",
                )
            )
        )
        self.assertTrue(
            app.route_intake_allows_workspace_execution(
                app.StructuredRouteIntake(
                    needs_workspace=True,
                    workspace_intent_hint="focused_write",
                    needs_artifact=True,
                )
            )
        )

    def test_route_intake_prefers_direct_web_summary_for_versioned_release_query(self):
        self.assertTrue(
            app.route_intake_prefers_direct_web_summary(
                app.StructuredRouteIntake(
                    needs_fresh_info=True,
                    is_versioned_release_query=True,
                    entity="nvim",
                    answer_shape="summary",
                    web_search_requested=True,
                ),
                "what changed in update to nvim 0.12?",
                features=app.FeatureFlags(web_search=True),
                history=[],
            )
        )
        self.assertFalse(
            app.route_intake_prefers_direct_web_summary(
                app.StructuredRouteIntake(
                    needs_fresh_info=True,
                    answer_shape="summary",
                    web_search_requested=True,
                    needs_artifact=True,
                    needs_workspace=True,
                    workspace_intent_hint="focused_write",
                ),
                "create a downloadable markdown summary of nvim 0.12 changes",
                features=app.FeatureFlags(web_search=True, agent_tools=True, workspace_write=True),
                history=[],
            )
        )

    def test_force_direct_web_summary_route_clears_workspace_hints(self):
        route = app.force_direct_web_summary_route(
            app.StructuredRouteIntake(
                needs_fresh_info=True,
                is_versioned_release_query=True,
                entity="nvim",
                answer_shape="summary",
                web_search_requested=True,
                needs_workspace=True,
                workspace_intent_hint="focused_write",
            ),
            "what changed in update to nvim 0.12?",
            features=app.FeatureFlags(web_search=True, agent_tools=True, workspace_write=True),
            history=[],
        )

        self.assertTrue(route.web_search_requested)
        self.assertFalse(route.needs_workspace)
        self.assertFalse(route.needs_artifact)
        self.assertEqual(route.workspace_intent_hint, "none")

    def test_prepare_turn_request_ignores_stale_workspace_resume_for_pure_web_summary(self):
        original_resolve_route_intake_for_turn = app.resolve_route_intake_for_turn
        original_should_resume_saved_workspace_task = app.should_resume_saved_workspace_task
        original_should_auto_execute_workspace_task = app.should_auto_execute_workspace_task
        original_maybe_bootstrap_workspace_from_current_repo = app.maybe_bootstrap_workspace_from_current_repo
        try:
            async def fake_resolve_route_intake_for_turn(*args, **kwargs):
                return app.StructuredRouteIntake(
                    needs_fresh_info=True,
                    is_versioned_release_query=True,
                    entity="nvim",
                    time_sensitivity="versioned",
                    answer_shape="summary",
                    web_search_requested=True,
                    needs_workspace=True,
                    workspace_intent_hint="focused_write",
                    reasoning="stale route contamination",
                    confidence=0.91,
                )

            app.resolve_route_intake_for_turn = fake_resolve_route_intake_for_turn
            app.should_resume_saved_workspace_task = lambda *args, **kwargs: True
            app.should_auto_execute_workspace_task = lambda *args, **kwargs: True
            app.maybe_bootstrap_workspace_from_current_repo = lambda *args, **kwargs: None

            prepared = asyncio.run(
                app.prepare_turn_request({
                    "conversation_id": "conv-nvim-direct",
                    "message": "what changed in update to nvim 0.12?",
                    "feature_flags": {
                        "agent_tools": True,
                        "workspace_write": True,
                        "web_search": True,
                    },
                })
            )
        finally:
            app.resolve_route_intake_for_turn = original_resolve_route_intake_for_turn
            app.should_resume_saved_workspace_task = original_should_resume_saved_workspace_task
            app.should_auto_execute_workspace_task = original_should_auto_execute_workspace_task
            app.maybe_bootstrap_workspace_from_current_repo = original_maybe_bootstrap_workspace_from_current_repo

        self.assertFalse(prepared.resume_saved_workspace)
        self.assertFalse(prepared.auto_execute_workspace)
        self.assertEqual(prepared.workspace_intent, "none")
        self.assertEqual(prepared.route_intake.workspace_intent_hint, "none")
        self.assertTrue(app.should_route_prepared_turn_via_direct_search(prepared))
        self.assertIn("web.search", prepared.enabled_tools)
        self.assertIn("web.fetch_page", prepared.enabled_tools)
        self.assertFalse(any(tool.startswith("workspace.") for tool in prepared.enabled_tools))

    def test_format_build_substep_progress_message_uses_focus_text(self):
        rendered = app.format_build_substep_progress_message(
            "Inspect the official release notes and pull out the main breaking changes",
            substep_index=0,
            substep_count=3,
            step_index=0,
            step_count=3,
        )

        self.assertIn("Inspect the official release notes", rendered)
        self.assertIn("(1/3 in step 1/3)", rendered)

    def test_format_turn_route_activity_prefers_direct_search_copy(self):
        rendered = app.format_turn_route_activity(
            app.TurnAssessment(
                primary_skill="search",
                requires_search=True,
                execution_style="direct_answer",
                workspace_intent="none",
                needs_fresh_info=True,
                is_versioned_release_query=True,
                entity="nvim",
            ),
            mode="chat",
            promoted_to_planning=False,
        )

        self.assertEqual(
            rendered,
            "Routing directly to current web sources for a concise summary of nvim.",
        )

    def test_extract_workspace_path_references_ignores_version_numbers(self):
        self.assertEqual(
            app.extract_workspace_path_references("what changed in update to nvim 0.12?"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
