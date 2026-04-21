import tempfile
import unittest
import pathlib
import asyncio
import sqlite3

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

    def test_normalize_attachment_paths_for_workspace_deduplicates_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = pathlib.Path(tempdir)
            (workspace / "pasted").mkdir(parents=True, exist_ok=True)
            (workspace / "pasted" / "note.txt").write_text("hello", encoding="utf-8")

            cleaned = app.normalize_attachment_paths_for_workspace(
                workspace,
                ["pasted/note.txt", "./pasted/note.txt", "pasted/note.txt"],
            )

        self.assertEqual(cleaned, ["pasted/note.txt"])

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
            original = app.get_workspace_path
            try:
                app.get_workspace_path = lambda _conversation_id, create=True: workspace
                root_listing = app.workspace_list_files_result("conv-hidden")
                hidden_listing = app.workspace_list_files_result("conv-hidden", ".venv")
            finally:
                app.get_workspace_path = original

        self.assertEqual([item["name"] for item in root_listing["items"]], ["app.py"])
        self.assertEqual(hidden_listing["path"], ".venv")
        self.assertEqual([item["name"] for item in hidden_listing["items"]], ["pyvenv.cfg"])

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

    def test_detect_implicit_failure_feedback_classifies_corrective_reply(self):
        signal = app.detect_implicit_failure_feedback(
            "the artifact isn't interactive and you can't pick one and it's not a real artifact"
        )

        self.assertEqual(signal.get("label"), "negative")
        self.assertEqual(signal.get("category"), "non_interactive_artifact")

    def test_save_message_marks_previous_assistant_negative_after_corrective_user_reply(self):
        original_db_path = app.DB_PATH
        original_runs_root = app.RUNS_ROOT_PATH
        original_workspace_root = app.WORKSPACE_ROOT
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                temp_root = pathlib.Path(tempdir)
                db_path = temp_root / "chat.db"
                app.DB_PATH = str(db_path)
                app.RUNS_ROOT_PATH = temp_root / "runs"
                app.WORKSPACE_ROOT = str(temp_root / "workspaces")
                app.RUNS_ROOT_PATH.mkdir(parents=True, exist_ok=True)

                conn = sqlite3.connect(db_path)
                conn.execute(
                    '''CREATE TABLE conversations
                       (id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT, run_id TEXT)'''
                )
                conn.execute(
                    '''CREATE TABLE messages
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, role TEXT, content TEXT, timestamp TEXT, feedback TEXT)'''
                )
                conn.execute(
                    '''CREATE TABLE runs
                       (id TEXT PRIMARY KEY, conversation_id TEXT UNIQUE NOT NULL, title TEXT, status TEXT NOT NULL DEFAULT 'active',
                        sandbox_path TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT, summary TEXT NOT NULL DEFAULT '',
                        promoted_count INTEGER NOT NULL DEFAULT 0)'''
                )
                conn.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at, run_id) VALUES (?, ?, ?, ?, ?)",
                    ("conv-feedback", "Feedback", "2026-04-10T00:00:00", "2026-04-10T00:00:00", "run-conv-feedback"),
                )
                conn.execute(
                    "INSERT INTO runs (id, conversation_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("run-conv-feedback", "conv-feedback", "Feedback", "active", str(app.RUNS_ROOT_PATH / "run-conv-feedback"), "2026-04-10T00:00:00", None, "", 0),
                )
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, feedback) VALUES (?, ?, ?, ?, ?)",
                    ("conv-feedback", "assistant", "Here is the generated matplotlib plot from the sample data.", "2026-04-10T00:00:01", "neutral"),
                )
                conn.commit()
                conn.close()

                app.save_message("conv-feedback", "user", "the plot doesn't show!")
                assistant = app.get_message_by_id(1)
                capture_dir = pathlib.Path(app.get_workspace_path("conv-feedback")) / ".ai" / "context-evals"
                capture_files = sorted(capture_dir.glob("*.json"))
        finally:
            app.DB_PATH = original_db_path
            app.RUNS_ROOT_PATH = original_runs_root
            app.WORKSPACE_ROOT = original_workspace_root

        self.assertIsNotNone(assistant)
        self.assertEqual(assistant["feedback"], "negative")
        self.assertEqual(len(capture_files), 1)
        payload = __import__("json").loads(capture_files[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["capture"]["trigger"], "implicit_feedback")
        self.assertIn("retrieved_memory", payload["expectation"]["required_selected_keys"])

    def test_collect_recent_product_feedback_entries_returns_recent_corrective_replies(self):
        original_db_path = app.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                db_path = pathlib.Path(tempdir) / "chat.db"
                app.DB_PATH = str(db_path)
                conn = sqlite3.connect(db_path)
                conn.execute(
                    '''CREATE TABLE conversations
                       (id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT, run_id TEXT)'''
                )
                conn.execute(
                    '''CREATE TABLE messages
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, role TEXT, content TEXT, timestamp TEXT, feedback TEXT)'''
                )
                conn.executemany(
                    "INSERT INTO conversations (id, title, created_at, updated_at, run_id) VALUES (?, ?, ?, ?, ?)",
                    [
                        ("conv-a", "Plot", "2026-04-10T00:00:00", "2026-04-10T07:34:32", None),
                        ("conv-b", "DoorDash", "2026-04-10T00:00:00", "2026-04-10T07:13:14", None),
                    ],
                )
                conn.executemany(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, feedback) VALUES (?, ?, ?, ?, ?)",
                    [
                        ("conv-a", "assistant", "Here is the generated matplotlib plot from the sample data.", "2026-04-10T07:34:31", "neutral"),
                        ("conv-a", "user", "the plot doesn't show!", "2026-04-10T07:34:33", None),
                        ("conv-b", "assistant", "I've created a door_dash_menu.json artifact with 10 options.", "2026-04-10T07:10:00", "neutral"),
                        ("conv-b", "user", "the artifact isn't interactive and you can't pick one and it's not a real artifact :(", "2026-04-10T07:10:10", None),
                    ],
                )
                conn.commit()
                conn.close()

                entries = app.collect_recent_product_feedback_entries(limit=4)
        finally:
            app.DB_PATH = original_db_path

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["category"], "artifact_visibility")
        self.assertEqual(entries[1]["category"], "non_interactive_artifact")

    def test_refresh_conversation_title_updates_short_llm_title(self):
        original_db_path = app.DB_PATH
        original_vllm = app.vllm_chat_complete
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                db_path = pathlib.Path(tempdir) / "chat.db"
                app.DB_PATH = str(db_path)
                app.init_db()

                conn = sqlite3.connect(app.DB_PATH)
                conn.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    ("conv-title", "Can you help me learn SQL by creating a new db in sqlite3 from CLI", "2026-04-10T00:00:00", "2026-04-10T00:00:00"),
                )
                conn.executemany(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, kind, feedback) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        ("conv-title", "user", "Can you help me learn SQL from the CLI?", "2026-04-10T00:00:00", "visible_chat", None),
                        ("conv-title", "assistant", "Yes, let's build a SQLite guide.", "2026-04-10T00:00:01", "visible_chat", "neutral"),
                        ("conv-title", "user", "Please include table creation and inserts too.", "2026-04-10T00:00:02", "visible_chat", None),
                        ("conv-title", "assistant", "I'll include create table, insert, and query steps.", "2026-04-10T00:00:03", "visible_chat", "neutral"),
                    ],
                )
                conn.commit()
                conn.close()

                async def fake_vllm_chat_complete(messages, max_tokens=None, temperature=None):
                    self.assertEqual(messages[0]["content"], app.CONVERSATION_TITLE_SYSTEM_PROMPT)
                    return "SQLite CLI CRUD quickstart guide today"

                app.vllm_chat_complete = fake_vllm_chat_complete
                asyncio.run(app.refresh_conversation_title("conv-title"))

                conn = sqlite3.connect(app.DB_PATH)
                updated_title = conn.execute(
                    "SELECT title FROM conversations WHERE id = ?",
                    ("conv-title",),
                ).fetchone()[0]
                conn.close()
        finally:
            app.DB_PATH = original_db_path
            app.vllm_chat_complete = original_vllm

        self.assertEqual(updated_title, "SQLite CLI CRUD quickstart guide")

    def test_request_wants_recent_product_feedback_for_repo_improvement_prompts(self):
        self.assertTrue(
            app.request_wants_recent_product_feedback(
                "This chat and some others in here contain feedback for the developer of this ai-chat software. Interpret the user's feedback as failure and improve the app overall."
            )
        )
        self.assertFalse(
            app.request_wants_recent_product_feedback(
                "Review this repo for the top 3 bugs and patch the top 2."
            )
        )

    def test_capture_context_eval_case_for_assistant_message_writes_replay_case(self):
        original_db_path = app.DB_PATH
        original_workspace_root = app.WORKSPACE_ROOT
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                temp_root = pathlib.Path(tempdir)
                db_path = temp_root / "chat.db"
                app.DB_PATH = str(db_path)
                app.WORKSPACE_ROOT = str(temp_root / "workspaces")

                conn = sqlite3.connect(db_path)
                conn.execute(
                    '''CREATE TABLE conversations
                       (id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT, run_id TEXT, workspace_id TEXT)'''
                )
                conn.execute(
                    '''CREATE TABLE messages
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, role TEXT, content TEXT, timestamp TEXT, kind TEXT, feedback TEXT)'''
                )
                conn.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at, run_id, workspace_id) VALUES (?, ?, ?, ?, ?, ?)",
                    ("conv-capture", "Capture", "2026-04-10T00:00:00", "2026-04-10T00:00:00", None, ""),
                )
                conn.executemany(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, kind, feedback) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        ("conv-capture", "user", "please verify the app.js change", "2026-04-10T00:00:01", "visible_chat", None),
                        ("conv-capture", "assistant", "I updated [[artifact:src/web/app.js]]", "2026-04-10T00:00:02", "visible_chat", "negative"),
                    ],
                )
                conn.commit()
                conn.close()

                workspace = pathlib.Path(app.get_workspace_path("conv-capture"))
                (workspace / "src" / "web").mkdir(parents=True, exist_ok=True)
                (workspace / "src" / "web" / "app.js").write_text("function sendMessage() { return true; }\n", encoding="utf-8")

                rel_path = app.capture_context_eval_case_for_assistant_message(
                    "conv-capture",
                    2,
                    trigger="explicit_feedback",
                )
                payload = __import__("json").loads((workspace / rel_path).read_text(encoding="utf-8"))
        finally:
            app.DB_PATH = original_db_path
            app.WORKSPACE_ROOT = original_workspace_root

        self.assertTrue(rel_path.endswith(".json"))
        self.assertEqual(payload["capture"]["trigger"], "explicit_feedback")
        self.assertEqual(payload["policy_inputs"]["phase"], "verify")
        self.assertIn("workspace_excerpts", [item["key"] for item in payload["selection_candidates"]])

    def test_tool_loop_step_limit_gives_focused_write_more_room(self):
        steps = app.tool_loop_step_limit_for_request(
            "Build a tiny terminal game in Python in the workspace and run it.",
            ["workspace.patch_file", "workspace.run_command"],
        )

        self.assertGreaterEqual(steps, 5)

    def test_tool_loop_step_limit_gives_repo_scale_requests_more_room(self):
        generic_steps = app.tool_loop_step_limit_for_request(
            "Improve the workspace UX and verify it.",
            ["workspace.grep", "workspace.read_file", "workspace.patch_file", "workspace.run_command"],
        )
        repo_steps = app.tool_loop_step_limit_for_request(
            "Inspect this repository and improve the workspace UX for pair programming.",
            ["workspace.grep", "workspace.read_file", "workspace.patch_file", "workspace.run_command"],
        )

        self.assertGreaterEqual(repo_steps, generic_steps)
        self.assertGreaterEqual(repo_steps, 10)

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

    def test_scraper_request_prefers_code_tools_over_implicit_web_search(self):
        message = "Write a Python script to scrape a website and save the results to CSV."
        features = app.FeatureFlags(
            agent_tools=True,
            workspace_write=True,
            workspace_run_commands=True,
            web_search=True,
        )

        allowed = app.select_enabled_tools("conv-scrape", message, features)
        direct_allowed = app.select_direct_answer_tools(message, allowed)

        self.assertEqual(app.classify_workspace_intent(message), "focused_write")
        self.assertFalse(app.should_offer_web_search(message, features))
        self.assertIn("workspace.patch_file", allowed)
        self.assertIn("workspace.run_command", allowed)
        self.assertIn("workspace.patch_file", direct_allowed)
        self.assertIn("workspace.run_command", direct_allowed)

    def test_explicit_search_request_still_enables_web_search(self):
        message = "Search the web for the latest FastAPI release and cite sources."
        features = app.FeatureFlags(agent_tools=True, web_search=True)

        self.assertTrue(app.should_offer_web_search(message, features))

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

    def test_render_request_does_not_upgrade_respond_phase_into_execution(self):
        features = app.FeatureFlags(agent_tools=True, workspace_write=True)

        self.assertFalse(
            app.should_upgrade_to_workspace_execution(
                {"name": "workspace.render"},
                features,
                "respond",
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

    def test_request_targets_current_repo_detects_existing_repo_language(self):
        self.assertTrue(app.request_targets_current_repo("Review this repo for the top 3 issues."))
        self.assertTrue(app.request_targets_current_repo("Find the approval logic in this repository and patch it."))
        self.assertFalse(app.request_targets_current_repo("Create a repo for this project from scratch."))
        self.assertTrue(app.request_targets_current_repo("Create a tiny demo app in this repo that proves it works."))

    def test_request_is_repo_scaffold_does_not_capture_create_app_in_this_repo(self):
        self.assertFalse(app.request_is_repo_scaffold("Create a tiny demo app in this repo that proves it works."))
        self.assertTrue(app.request_is_repo_scaffold("Create a repo structure for this project from scratch."))

    def test_request_prefers_illustrative_output_for_demo_and_chart_language(self):
        self.assertTrue(
            app.request_prefers_illustrative_output(
                "Create a tiny demo app in this repo that proves it works and keep going until the artifact is real."
            )
        )
        self.assertTrue(
            app.request_prefers_illustrative_output(
                "Show the graph or chart so the result is visible."
            )
        )
        self.assertFalse(
            app.request_prefers_illustrative_output(
                "Patch the approval logic in this repo and run the narrowest useful tests."
            )
        )

    def test_workspace_hidden_paths_include_runtime_cache_directories(self):
        self.assertTrue(app.workspace_rel_path_is_hidden("__pycache__/app.cpython-311.pyc"))
        self.assertTrue(app.workspace_rel_path_is_hidden(".venv/bin/python"))
        self.assertFalse(app.workspace_rel_path_is_hidden("src/demo.py"))

    def test_extract_context_clarification_from_workspace_facts_detects_blocking_question(self):
        facts = (
            "The workspace currently contains no files or directories. "
            "There is no existing repository to inspect or modify. "
            "Would you like to proceed with creating the necessary files, or is there a different issue to address?\n\n"
            "Grounded workspace snapshot:\n- Workspace root: /tmp/workspace"
        )
        self.assertIn(
            "Would you like to proceed",
            app.extract_context_clarification_from_workspace_facts(facts),
        )

    def test_should_pause_for_workspace_clarification_only_when_workspace_is_still_missing_context(self):
        facts = (
            "The workspace currently contains no files or directories. "
            "There is no existing repository to inspect or modify. "
            "Would you like to proceed with creating the necessary files, or is there a different issue to address?"
        )
        self.assertIn(
            "Would you like to proceed",
            app.should_pause_for_workspace_clarification(
                "Inspect this repository and improve the workspace UX.",
                facts,
                {"user_file_count": 0},
            ),
        )
        self.assertEqual(
            app.should_pause_for_workspace_clarification(
                "Inspect this repository and improve the workspace UX.",
                facts,
                {"user_file_count": 12},
            ),
            "",
        )

    def test_saved_progress_fallback_response_explains_repo_snapshot_context(self):
        original_load_task_state = app.load_task_state
        try:
            app.load_task_state = lambda _conversation_id, rel_path=".ai/task-state.json": {
                "request": "Read this codebase and find one place where command-generated outputs are still treated as second-class compared with edited files.",
                "workspace_snapshot": {"user_file_count": 13},
                "plan": {"builder_steps": ["Inspect the codebase", "Patch the bug"]},
                "build_step_summaries": [],
                "changed_files": [],
                "task_board_path": ".ai/task-board.md",
                "pause_reason": "",
                "verification_summary": "",
            }
            response = app.build_saved_progress_fallback_response("conv-repo")
        finally:
            app.load_task_state = original_load_task_state

        self.assertIn("repo snapshot for context", response)
        self.assertIn("[[artifact:.ai/task-board.md]]", response)
        self.assertNotIn("Open `[[artifact:.ai/task-board.md]]`", response)

    def test_ensure_nonempty_turn_response_prefers_saved_progress_summary(self):
        original_builder = app.build_saved_progress_fallback_response
        try:
            app.build_saved_progress_fallback_response = lambda *args, **kwargs: "Saved progress summary."
            response = app.ensure_nonempty_turn_response("", "conv-fallback", "Review this repo")
        finally:
            app.build_saved_progress_fallback_response = original_builder

        self.assertEqual(response, "Saved progress summary.")

    def test_ensure_nonempty_turn_response_uses_cleaner_resume_copy_on_error(self):
        response = app.ensure_nonempty_turn_response("", "conv-fallback", "Review this repo", error_text="boom")

        self.assertIn("wasn't able to finish", response)
        self.assertIn("Say continue", response)
        self.assertNotIn("unexpected internal error", response)

    def test_bootstrap_workspace_from_current_repo_seeds_filtered_snapshot(self):
        original_base_dir = app._base_dir
        original_get_workspace_path = app.get_workspace_path
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                source_root = pathlib.Path(tempdir) / "repo"
                workspace = pathlib.Path(tempdir) / "workspace"
                source_root.mkdir(parents=True, exist_ok=True)
                workspace.mkdir(parents=True, exist_ok=True)

                (source_root / "app.py").write_text("print('repo')\n", encoding="utf-8")
                (source_root / "README.md").write_text("# Repo\n", encoding="utf-8")
                (source_root / "static").mkdir(parents=True, exist_ok=True)
                (source_root / "static" / "app.js").write_text("console.log('ok');\n", encoding="utf-8")
                (source_root / "runs").mkdir(parents=True, exist_ok=True)
                (source_root / "runs" / "junk.txt").write_text("skip me\n", encoding="utf-8")
                (source_root / "data").mkdir(parents=True, exist_ok=True)
                (source_root / "data" / "chat.db").write_text("skip me\n", encoding="utf-8")
                (source_root / "__pycache__").mkdir(parents=True, exist_ok=True)
                (source_root / "__pycache__" / "app.pyc").write_bytes(b"pyc")
                (source_root / ".env").write_text("SECRET=1\n", encoding="utf-8")

                app._base_dir = source_root
                app.get_workspace_path = lambda _conversation_id, create=True: workspace

                result = app.maybe_bootstrap_workspace_from_current_repo(
                    "conv-repo",
                    "Review this repo and patch the approval flow.",
                )

                self.assertTrue(result)
                self.assertTrue((workspace / "app.py").exists())
                self.assertTrue((workspace / "README.md").exists())
                self.assertTrue((workspace / "static" / "app.js").exists())
                self.assertFalse((workspace / "runs").exists())
                self.assertFalse((workspace / "data").exists())
                self.assertFalse((workspace / "__pycache__").exists())
                self.assertFalse((workspace / ".env").exists())
        finally:
            app._base_dir = original_base_dir
            app.get_workspace_path = original_get_workspace_path

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
