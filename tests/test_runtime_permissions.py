import tempfile
import unittest
import pathlib
import asyncio

try:
    import app
except Exception as exc:
    app = None
    APP_IMPORT_ERROR = exc
else:
    APP_IMPORT_ERROR = None


class RuntimePermissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if app is None:
            raise unittest.SkipTest(f"app.py dependencies are unavailable: {APP_IMPORT_ERROR}")

    def test_parse_feature_flags_keeps_allowed_tool_permissions(self):
        features = app.parse_feature_flags({
            "allowed_tool_permissions": ["tool:web.search", " TOOL:workspace.write "],
            "auto_approve_tool_permissions": True,
        })

        self.assertTrue(features.web_search)
        self.assertTrue(features.auto_approve_tool_permissions)
        self.assertEqual(
            features.allowed_tool_permissions,
            ["tool:web.search", "tool:workspace.write"],
        )

    def test_workspace_grep_permission_request_is_granular(self):
        request = app.build_tool_permission_request(
            "conv-grep",
            {
                "id": "call1",
                "name": "workspace.grep",
                "arguments": {"query": "FeatureFlags", "path": "."},
            },
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.key, "tool:workspace.grep")
        self.assertEqual(request.approval_target, "tool")
        self.assertIn("search", request.content.lower())

    def test_command_permission_request_uses_executable_key(self):
        previous_root = app.WORKSPACE_ROOT
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                app.WORKSPACE_ROOT = tempdir
                app.get_workspace_path("conv-command")
                request = app.build_tool_permission_request(
                    "conv-command",
                    {
                        "id": "call2",
                        "name": "workspace.run_command",
                        "arguments": {"command": ["git", "status"], "cwd": "."},
                    },
                )
        finally:
            app.WORKSPACE_ROOT = previous_root

        self.assertIsNotNone(request)
        self.assertEqual(request.key, "exec:git")
        self.assertEqual(request.approval_target, "command")
        self.assertIn("git", request.title.lower())

    def test_python_m_pip_install_permission_is_scoped_to_pip(self):
        request = app.build_tool_permission_request(
            "conv-pip",
            {
                "id": "call-pip",
                "name": "workspace.run_command",
                "arguments": {"command": ["python3", "-m", "pip", "install", "pandas"], "cwd": "."},
            },
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.key, "exec:pip.install")
        self.assertEqual(request.approval_target, "command")
        self.assertEqual(request.title, "Allow pip install?")
        self.assertIn("pandas", request.content.lower())
        self.assertIn("managed python environment", request.content.lower())

    def test_python_venv_setup_permission_is_scoped(self):
        previous_root = app.WORKSPACE_ROOT
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                app.WORKSPACE_ROOT = tempdir
                app.get_workspace_path("conv-venv")
                request = app.build_tool_permission_request(
                    "conv-venv",
                    {
                        "id": "call-venv",
                        "name": "workspace.run_command",
                        "arguments": {"command": ["python3", "-m", "venv", ".venv"], "cwd": "."},
                    },
                )
        finally:
            app.WORKSPACE_ROOT = previous_root

        self.assertIsNotNone(request)
        self.assertEqual(request.key, "exec:python.venv")
        self.assertEqual(request.title, "Allow Python venv setup?")
        self.assertIn(".venv", request.content)

    def test_normalize_direct_slash_command_accepts_pip(self):
        self.assertEqual(app.normalize_direct_slash_command("pip"), "pip")

    def test_install_like_python_setup_commands_skip_short_timeout(self):
        self.assertIsNone(
            app.command_runtime_timeout_seconds(["python3", "-m", "pip", "install", "pandas"])
        )
        self.assertIsNone(
            app.command_runtime_timeout_seconds(["python3", "-m", "venv", ".venv"])
        )
        self.assertEqual(
            app.command_runtime_timeout_seconds(["git", "status"]),
            app.COMMAND_TIMEOUT_SECONDS,
        )

    def test_workspace_list_files_hides_dotfiles_until_explicitly_targeted(self):
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = pathlib.Path(tempdir)
            (workspace / "app.py").write_text("print('hi')\n", encoding="utf-8")
            (workspace / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
            (workspace / ".venv").mkdir(parents=True, exist_ok=True)
            (workspace / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
            (workspace / "__pycache__").mkdir(parents=True, exist_ok=True)
            (workspace / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"pyc")
            original = app.get_workspace_path
            try:
                app.get_workspace_path = lambda _conversation_id, create=True: workspace
                root_listing = app.workspace_list_files_result("conv-hidden")
                hidden_listing = app.workspace_list_files_result("conv-hidden", ".venv")
                cache_listing = app.workspace_list_files_result("conv-hidden", "__pycache__")
            finally:
                app.get_workspace_path = original

        self.assertEqual([item["name"] for item in root_listing["items"]], ["app.py"])
        self.assertEqual(hidden_listing["path"], ".venv")
        self.assertEqual([item["name"] for item in hidden_listing["items"]], ["pyvenv.cfg"])
        self.assertEqual(cache_listing["path"], "__pycache__")
        self.assertEqual([item["name"] for item in cache_listing["items"]], ["app.cpython-311.pyc"])

    def test_validate_workspace_command_allows_managed_python_env_paths(self):
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = pathlib.Path(tempdir) / "workspace"
            env_root = pathlib.Path(tempdir) / "python-env"
            workspace.mkdir(parents=True, exist_ok=True)
            env_bin = env_root / ("Scripts" if app.os.name == "nt" else "bin")
            env_bin.mkdir(parents=True, exist_ok=True)
            pip_name = "pip.exe" if app.os.name == "nt" else "pip"
            pip_path = env_bin / pip_name
            pip_path.write_text("", encoding="utf-8")

            original_workspace = app.get_workspace_path
            original_env = app.get_managed_python_env_path
            try:
                app.get_workspace_path = lambda _conversation_id, create=True: workspace
                app.get_managed_python_env_path = lambda _conversation_id, create=False: env_root
                app.validate_workspace_command(
                    "conv-managed",
                    [str(pip_path), "install", "pandas"],
                    workspace,
                )
            finally:
                app.get_workspace_path = original_workspace
                app.get_managed_python_env_path = original_env

    def test_managed_python_env_prefers_server_root_and_can_migrate_legacy_run_env(self):
        original_managed_root = app.MANAGED_PYTHON_ENVS_ROOT_PATH
        original_runs_root = app.RUNS_ROOT_PATH
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                app.MANAGED_PYTHON_ENVS_ROOT_PATH = pathlib.Path(tempdir) / "managed-python-envs"
                app.RUNS_ROOT_PATH = pathlib.Path(tempdir) / "runs"
                app.MANAGED_PYTHON_ENVS_ROOT_PATH.mkdir(parents=True, exist_ok=True)
                app.RUNS_ROOT_PATH.mkdir(parents=True, exist_ok=True)

                preferred = app.get_managed_python_env_path("conv-env")
                legacy = app.get_legacy_managed_python_env_path("conv-env", create=True)
                legacy.mkdir(parents=True, exist_ok=True)
                (legacy / "marker.txt").write_text("legacy", encoding="utf-8")

                self.assertNotIn(app.RUNS_ROOT_PATH, preferred.parents)
                self.assertEqual(app.resolve_existing_managed_python_env_path("conv-env"), legacy)

                migrated = app.migrate_legacy_managed_python_env("conv-env")

                self.assertEqual(migrated, preferred)
                self.assertTrue(preferred.exists())
                self.assertFalse(legacy.exists())
                self.assertEqual((preferred / "marker.txt").read_text(encoding="utf-8"), "legacy")
        finally:
            app.MANAGED_PYTHON_ENVS_ROOT_PATH = original_managed_root
            app.RUNS_ROOT_PATH = original_runs_root

    def test_capability_recovery_detects_false_tool_limitation_language(self):
        self.assertTrue(
            app.should_attempt_capability_recovery(
                "The tools available in the workspace do not support direct PDF processing or generation."
            )
        )
        self.assertFalse(
            app.should_attempt_capability_recovery(
                "The task is paused here while waiting for approval. Approve it for this chat and then say continue."
            )
        )

    def test_capability_recovery_detects_execution_handoff_when_request_needed_real_output(self):
        self.assertTrue(
            app.should_attempt_capability_recovery(
                "The FastAPI app is built and ready to run. To run the app locally:\n1. Install dependencies:\n2. Start the server:\nVisit http://localhost:8000",
                request_text="Build a small FastAPI app in the workspace, install anything needed, run it, and render the homepage in the viewer.",
                allowed_tools=["workspace.patch_file", "workspace.run_command", "workspace.render"],
            )
        )

    def test_capability_recovery_ignores_legitimate_local_instructions_requests(self):
        self.assertFalse(
            app.should_attempt_capability_recovery(
                "To run it locally:\npython3 main.py",
                request_text="How do I run this locally?",
                allowed_tools=["workspace.run_command"],
            )
        )

    def test_tool_loop_step_limit_gives_focused_write_more_room(self):
        steps = app.tool_loop_step_limit_for_request(
            "Build a tiny terminal game in Python in the workspace and run it.",
            ["workspace.patch_file", "workspace.run_command"],
        )

        self.assertGreaterEqual(steps, 5)

    def test_tool_permission_allowlist_round_trip(self):
        features = app.FeatureFlags()

        self.assertFalse(app.is_tool_permission_allowlisted(features, "tool:web.search"))
        app.remember_approved_tool_permission(features, "tool:web.search")
        self.assertTrue(app.is_tool_permission_allowlisted(features, "tool:web.search"))

    def test_auto_approve_tool_permissions_skips_runtime_prompt(self):
        features = app.FeatureFlags(auto_approve_tool_permissions=True)

        approved, request = asyncio.run(
            app.ensure_tool_permission(
                None,
                "conv-auto-approve",
                {
                    "id": "call-auto",
                    "name": "web.search",
                    "arguments": {"query": "NVDA price"},
                },
                features,
            )
        )

        self.assertTrue(approved)
        self.assertIsNotNone(request)
        self.assertEqual(request.key, "tool:web.search")
        self.assertIn("tool:web.search", features.allowed_tool_permissions)

    def test_permission_blocked_message_explains_pause_and_resume(self):
        request = app.PermissionApprovalRequest(
            key="exec:git",
            approval_target="command",
            title="Allow git status?",
            content="The assistant wants to run git status in the workspace",
            preview="git status",
        )

        message = app.render_permission_blocked_message(request)

        self.assertIn("paused here", message)
        self.assertIn("Needed command approval: Allow git status.", message)
        self.assertIn("Request details: git status", message)
        self.assertIn("Approve it for this chat and then say continue", message)

    def test_permission_denied_result_includes_resume_message(self):
        request = app.PermissionApprovalRequest(
            key="tool:web.search",
            approval_target="tool",
            title="Use web search?",
            content="The assistant wants to search the web",
        )
        denied = app.build_permission_denied_result(
            {"id": "call-denied", "name": "web.search"},
            request,
        )

        self.assertFalse(denied["ok"])
        self.assertEqual(denied["error_code"], "permission_denied")
        self.assertIn("paused here", denied["message_to_user"])

    def test_render_deep_plan_preview_points_to_approval_panel(self):
        preview = app.render_deep_plan_preview({"builder_steps": ["Inspect code", "Apply fix"]})

        self.assertIn("approval panel", preview)
        self.assertIn("Approve And Run", preview)

    def test_natural_python_build_request_auto_executes_workspace_flow(self):
        message = "Start making a python model to keep track of AVGO daily and build a DCF tracker."
        features = app.FeatureFlags(agent_tools=True, workspace_write=True)

        self.assertEqual(app.classify_workspace_intent(message), "focused_write")
        self.assertTrue(app.should_use_workspace_tools("conv-avgo", message, features))
        self.assertTrue(app.should_auto_execute_workspace_task("conv-avgo", message, features))

    def test_patch_request_can_upgrade_from_respond_phase_into_execution(self):
        features = app.FeatureFlags(agent_tools=True, workspace_write=True)

        self.assertTrue(
            app.should_upgrade_to_workspace_execution(
                {"name": "workspace.patch_file"},
                features,
                "respond",
            )
        )
        self.assertFalse(
            app.should_upgrade_to_workspace_execution(
                {"name": "workspace.patch_file"},
                features,
                "verify",
            )
        )

    def test_workspace_download_uses_clean_filename(self):
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = pathlib.Path(tempdir)
            (workspace / "notes.txt").write_text("hello", encoding="utf-8")
            original = app.get_workspace_path
            try:
                app.get_workspace_path = lambda _conversation_id, create=True: workspace
                response = asyncio.run(app.download_workspace("conv-download"))
            finally:
                app.get_workspace_path = original

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Content-Disposition"), 'attachment; filename="workspace.zip"')
        self.assertEqual(response.media_type, "application/zip")

    def test_workspace_download_skips_internal_only_workspaces(self):
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = pathlib.Path(tempdir)
            (workspace / ".ai").mkdir(parents=True, exist_ok=True)
            (workspace / ".ai" / "task-state.json").write_text("{}", encoding="utf-8")
            original = app.get_workspace_path
            try:
                app.get_workspace_path = lambda _conversation_id, create=True: workspace
                response = asyncio.run(app.download_workspace("conv-download-empty"))
            finally:
                app.get_workspace_path = original

        self.assertEqual(response.status_code, 204)

    def test_workspace_run_command_result_surfaces_image_artifacts_for_preview(self):
        python_cmd = "python.exe" if app.os.name == "nt" else "python3"
        features = app.FeatureFlags(allowed_commands=[f"exec:{pathlib.Path(python_cmd).name.lower()}"])

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = pathlib.Path(tempdir)
            original_get_workspace_path = app.get_workspace_path
            try:
                app.get_workspace_path = lambda _conversation_id, create=True: workspace
                result = asyncio.run(
                    app.workspace_run_command_result(
                        "conv-artifacts",
                        [
                            python_cmd,
                            "-c",
                            "from pathlib import Path; "
                            "Path('notes.txt').write_text('ok\\n', encoding='utf-8'); "
                            "Path('plot.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')",
                        ],
                        ".",
                        features,
                    )
                )
            finally:
                app.get_workspace_path = original_get_workspace_path

        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["path"], "plot.png")
        self.assertEqual(result["open_path"], "plot.png")
        self.assertGreaterEqual(result["artifacts_detected"], 2)
        self.assertEqual(result["items"][0]["path"], "plot.png")
        self.assertEqual(result["items"][0]["content_kind"], "image")
        self.assertEqual(result["items"][1]["path"], "notes.txt")

    def test_successful_workspace_write_paths_include_command_and_render_outputs(self):
        tool_results = [
            {
                "call": {"name": "workspace.run_command"},
                "result": {
                    "ok": True,
                    "result": {
                        "path": "reports/attention_report.pdf",
                        "items": [
                            {"path": "reports/attention_report.pdf"},
                            {"path": "reports/attention_report.tex"},
                        ],
                    },
                },
            },
            {
                "call": {"name": "workspace.render"},
                "result": {"ok": True, "result": {"path": "preview/dashboard.html"}},
            },
        ]

        written = app.successful_workspace_write_paths(tool_results)

        self.assertIn("reports/attention_report.pdf", written)
        self.assertIn("reports/attention_report.tex", written)
        self.assertIn("preview/dashboard.html", written)

    def test_truthfulness_filter_allows_claims_backed_by_command_artifacts(self):
        message = "Created `reports/attention_report.pdf` and saved it in the workspace."
        tool_results = [
            {
                "call": {"name": "workspace.run_command"},
                "result": {"ok": True, "result": {"path": "reports/attention_report.pdf", "items": []}},
            }
        ]

        cleaned = app.strip_unverified_workspace_write_claims(message, tool_results)

        self.assertEqual(cleaned, message)

    def test_workspace_command_env_disables_python_bytecode_clutter(self):
        env = app.build_workspace_command_env("conv-env-no-pyc")

        self.assertEqual(env.get("PYTHONDONTWRITEBYTECODE"), "1")


if __name__ == "__main__":
    unittest.main()
