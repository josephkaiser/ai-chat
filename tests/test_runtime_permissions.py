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
        })

        self.assertTrue(features.web_search)
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

    def test_tool_permission_allowlist_round_trip(self):
        features = app.FeatureFlags()

        self.assertFalse(app.is_tool_permission_allowlisted(features, "tool:web.search"))
        app.remember_approved_tool_permission(features, "tool:web.search")
        self.assertTrue(app.is_tool_permission_allowlisted(features, "tool:web.search"))

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


if __name__ == "__main__":
    unittest.main()
