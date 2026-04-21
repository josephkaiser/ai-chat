import tempfile
import unittest
import pathlib
import asyncio
import sqlite3
import json

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

    def test_missing_optional_dependency_only_accepts_missing_target_package(self):
        nested = ModuleNotFoundError("missing nested")
        nested.name = "fastapi.routing"
        self.assertTrue(app.missing_optional_dependency(nested, "fastapi"))

        other = ModuleNotFoundError("other")
        other.name = "starlette"
        self.assertFalse(app.missing_optional_dependency(other, "fastapi"))

    def test_normalize_requested_mode_defaults_to_auto(self):
        self.assertEqual(app.normalize_requested_mode(None), "auto")
        self.assertEqual(app.normalize_requested_mode(""), "auto")
        self.assertEqual(app.normalize_requested_mode("chat"), "auto")
        self.assertEqual(app.normalize_requested_mode("deep"), "deep")

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

    def test_normalize_conversation_title_rejects_chatty_leadins(self):
        self.assertEqual(
            app.normalize_conversation_title("Okay, let's debug weather search", fallback="Weather in San Francisco"),
            "Weather in San Francisco",
        )
        self.assertEqual(
            app.normalize_conversation_title("Can you help me?", fallback="SQL CLI guide"),
            "SQL CLI guide",
        )

    def test_derive_conversation_title_seed_uses_first_user_topic(self):
        rows = [
            ("user", "what's the weather in san francisco?", "2026-04-10T00:00:00"),
            ("assistant", "Let me check.", "2026-04-10T00:00:01"),
            ("user", "what is 11 C in F?", "2026-04-10T00:00:02"),
        ]

        self.assertEqual(
            app.derive_conversation_title_seed(rows),
            "The weather in san francisco",
        )

    def test_should_refresh_conversation_title_for_count_throttles_followups(self):
        self.assertFalse(app.should_refresh_conversation_title_for_count("New chat", 3))
        self.assertTrue(app.should_refresh_conversation_title_for_count("New chat", 4))
        self.assertFalse(app.should_refresh_conversation_title_for_count("Useful title", 6))
        self.assertTrue(app.should_refresh_conversation_title_for_count("Useful title", 8))

    def test_fallback_conversation_memory_payload_uses_summary_lines_and_recent_requests(self):
        rows = [
            ("user", "Can you make the app feel more chat-first?", "2026-04-10T00:00:00"),
            ("assistant", "Yes, we should keep workspace artifacts durable.", "2026-04-10T00:00:01"),
            ("user", "Also improve follow-up handling.", "2026-04-10T00:00:02"),
        ]

        payload = app.fallback_conversation_memory_payload(
            rows,
            summary_text=(
                "Goals: Make the app chat-first; Improve follow-up handling\n"
                "Decisions: Keep durable workspace artifacts\n"
                "Files: src/web/app.ts; src/python/harness.py\n"
                "Open questions: How much memory should be retrieved?"
            ),
        )

        self.assertEqual(payload["goals"], ["Make the app chat-first", "Improve follow-up handling"])
        self.assertEqual(payload["decisions"], ["Keep durable workspace artifacts"])
        self.assertEqual(payload["active_files"], ["src/web/app.ts", "src/python/harness.py"])
        self.assertEqual(payload["open_questions"], ["How much memory should be retrieved?"])
        self.assertEqual(payload["recent_requests"], ["Can you make the app feel more chat-first?", "Also improve follow-up handling"])

    def test_normalize_conversation_memory_payload_falls_back_cleanly(self):
        rows = [
            ("user", "Write a linked list example in C", "2026-04-10T00:00:00"),
            ("assistant", "I can create the file.", "2026-04-10T00:00:01"),
        ]

        payload = app.normalize_conversation_memory_payload(
            {
                "summary": " ",
                "goals": [],
                "active_files": [" linked_list.c ", "linked_list.c"],
                "recent_requests": [],
            },
            summary_fallback="Goals: Linked list example in C\nFiles: linked_list.c",
            rows=rows,
        )

        self.assertEqual(payload["summary"], "Goals: Linked list example in C Files: linked_list.c")
        self.assertEqual(payload["goals"], ["Linked list example in C"])
        self.assertEqual(payload["active_files"], ["linked_list.c"])
        self.assertEqual(payload["recent_requests"], ["Write a linked list example in C"])

    def test_refresh_conversation_title_skips_non_trigger_message_counts(self):
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
                    ("conv-title-skip", "Already titled", "2026-04-10T00:00:00", "2026-04-10T00:00:00"),
                )
                conn.executemany(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, kind, feedback) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        ("conv-title-skip", "user", "one", "2026-04-10T00:00:00", "visible_chat", None),
                        ("conv-title-skip", "assistant", "two", "2026-04-10T00:00:01", "visible_chat", "neutral"),
                        ("conv-title-skip", "user", "three", "2026-04-10T00:00:02", "visible_chat", None),
                        ("conv-title-skip", "assistant", "four", "2026-04-10T00:00:03", "visible_chat", "neutral"),
                        ("conv-title-skip", "user", "five", "2026-04-10T00:00:04", "visible_chat", None),
                        ("conv-title-skip", "assistant", "six", "2026-04-10T00:00:05", "visible_chat", "neutral"),
                    ],
                )
                conn.commit()
                conn.close()

                async def fail_if_called(messages, max_tokens=None, temperature=None):
                    raise AssertionError("title refresh should have been skipped")

                app.vllm_chat_complete = fail_if_called
                asyncio.run(app.refresh_conversation_title("conv-title-skip"))
        finally:
            app.DB_PATH = original_db_path
            app.vllm_chat_complete = original_vllm

    def test_refresh_conversation_title_falls_back_from_chatty_llm_title(self):
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
                    ("conv-title-fallback", "New chat", "2026-04-10T00:00:00", "2026-04-10T00:00:00"),
                )
                conn.executemany(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, kind, feedback) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        ("conv-title-fallback", "user", "what's the weather in san francisco?", "2026-04-10T00:00:00", "visible_chat", None),
                        ("conv-title-fallback", "assistant", "I can check that.", "2026-04-10T00:00:01", "visible_chat", "neutral"),
                        ("conv-title-fallback", "user", "what is 11 C in F?", "2026-04-10T00:00:02", "visible_chat", None),
                        ("conv-title-fallback", "assistant", "11 C is 51.8 F.", "2026-04-10T00:00:03", "visible_chat", "neutral"),
                    ],
                )
                conn.commit()
                conn.close()

                async def fake_vllm_chat_complete(messages, max_tokens=None, temperature=None):
                    return "Okay, let's figure this out"

                app.vllm_chat_complete = fake_vllm_chat_complete
                asyncio.run(app.refresh_conversation_title("conv-title-fallback"))

                conn = sqlite3.connect(app.DB_PATH)
                updated_title = conn.execute(
                    "SELECT title FROM conversations WHERE id = ?",
                    ("conv-title-fallback",),
                ).fetchone()[0]
                conn.close()
        finally:
            app.DB_PATH = original_db_path
            app.vllm_chat_complete = original_vllm

        self.assertEqual(updated_title, "The weather in san francisco")

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

    def test_typoed_search_the_web_request_still_enables_web_search(self):
        message = "Can you search teh web for the current FastAPI docs?"
        features = app.FeatureFlags(agent_tools=True, web_search=True)

        self.assertTrue(app.should_offer_web_search(message, features))

    def test_recommendation_research_request_enables_web_search(self):
        message = "What are the best current note-taking apps? Please compare top options."
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

    def test_truthfulness_filter_strips_unverified_artifact_reference(self):
        cleaned = app.strip_unverified_workspace_write_claims("[[artifact:workspace/simple_script.py]]")

        self.assertEqual(cleaned, "")

    def test_truthfulness_filter_canonicalizes_workspace_prefixed_artifact_reference(self):
        tool_results = [
            {
                "call": {"name": "workspace.patch_file"},
                "result": {"ok": True, "result": {"path": "simple_script.py"}},
            }
        ]

        cleaned = app.strip_unverified_workspace_write_claims(
            "[[artifact:workspace/simple_script.py]]",
            tool_results,
        )

        self.assertEqual(cleaned, "[[artifact:simple_script.py]]")

    def test_inline_code_response_gets_artifact_reference_when_file_like(self):
        response = (
            "I'll modify the example to include proper headers. Here's the updated version:\n\n"
            "```c\n"
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n\n"
            "int main(void) {\n"
            "    printf(\"hello\\n\");\n"
            "    return 0;\n"
            "}\n"
            "```"
        )
        history = [
            {"role": "assistant", "content": "```c\nint main(){return 0;}\n```"},
        ]

        wrapped = app.maybe_attach_inline_code_artifact_reference(
            response,
            "Ok i am looking at your linked list. I have only used stdio.h, not stdlib.h",
            history=history,
            tool_results=[],
        )

        self.assertIn("[[artifact:", wrapped)
        self.assertIn("linked_list.c", wrapped)
        self.assertIn(".c]]", wrapped)

    def test_inline_code_response_does_not_get_artifact_reference_after_real_write(self):
        response = "```c\nint main(void) { return 0; }\n```"
        tool_results = [
            {
                "call": {"name": "workspace.patch_file"},
                "result": {"ok": True, "result": {"path": "linked_list.c"}},
            }
        ]

        wrapped = app.maybe_attach_inline_code_artifact_reference(
            response,
            "Update linked_list.c",
            history=[],
            tool_results=tool_results,
        )

        self.assertEqual(wrapped, response)

    def test_select_enabled_tools_treats_yes_after_save_offer_as_write_intent(self):
        features = app.FeatureFlags(
            agent_tools=True,
            workspace_write=True,
            workspace_run_commands=True,
        )

        allowed = app.select_enabled_tools(
            "conv-save-offer",
            "yes",
            features,
            history=[
                {
                    "role": "assistant",
                    "content": "Would you like me to save this script to your workspace?",
                }
            ],
        )

        self.assertIn("workspace.patch_file", allowed)
        self.assertIn("workspace.run_command", allowed)

    def test_select_direct_answer_tools_keeps_write_tools_for_yes_after_save_offer(self):
        filtered = app.select_direct_answer_tools(
            "yes",
            ["workspace.patch_file", "workspace.run_command", "conversation.search_history"],
            history=[
                {
                    "role": "assistant",
                    "content": "If you want, I can save the result to your workspace.",
                }
            ],
        )

        self.assertIn("workspace.patch_file", filtered)
        self.assertIn("workspace.run_command", filtered)
        self.assertIn("conversation.search_history", filtered)

    def test_classify_workspace_intent_inherits_edit_offer_for_short_followup(self):
        intent = app.classify_workspace_intent(
            "do that",
            history=[
                {
                    "role": "assistant",
                    "content": "If you want, I can patch that file directly in the workspace.",
                }
            ],
        )

        self.assertEqual(intent, "focused_write")

    def test_should_offer_web_search_inherits_offer_for_short_followup(self):
        features = app.FeatureFlags(agent_tools=True, web_search=True)

        self.assertTrue(
            app.should_offer_web_search(
                "ok",
                features,
                history=[
                    {
                        "role": "assistant",
                        "content": "If you want, I can search the web for current sources and cite them.",
                    }
                ],
            )
        )

    def test_resolve_contextual_followup_request_rewrites_google_followup_from_memory(self):
        original_loader = app.load_conversation_memory
        try:
            app.load_conversation_memory = lambda _conversation_id, rel_path=app.CONVERSATION_MEMORY_ARTIFACT_PATH: {
                "recent_requests": ["what is the weather in san francisco currently?"],
            }
            rewritten = app.resolve_contextual_followup_request(
                "conv-google-followup",
                "can't you see on google?",
                history=[
                    {"role": "user", "content": "what is the weather in san francisco currently?"},
                    {"role": "assistant", "content": "Here are some weather sites I found."},
                ],
            )
        finally:
            app.load_conversation_memory = original_loader

        self.assertEqual(
            rewritten,
            "Use web search or current online sources to answer this request directly: what is the weather in san francisco currently?",
        )

    def test_resolve_contextual_followup_request_rewrites_save_followup_with_active_file(self):
        original_loader = app.load_conversation_memory
        try:
            app.load_conversation_memory = lambda _conversation_id, rel_path=app.CONVERSATION_MEMORY_ARTIFACT_PATH: {
                "recent_requests": ["write a linked list example in C"],
                "active_files": ["linked_list.c"],
            }
            rewritten = app.resolve_contextual_followup_request(
                "conv-save-followup",
                "yes",
                history=[
                    {"role": "assistant", "content": "Would you like me to save this script to your workspace?"},
                ],
            )
        finally:
            app.load_conversation_memory = original_loader

        self.assertEqual(
            rewritten,
            "Save or update the workspace file linked_list.c for this request: write a linked list example in C",
        )

    def test_should_use_workspace_tools_inherits_render_offer_for_show_me(self):
        features = app.FeatureFlags(agent_tools=True)

        self.assertTrue(
            app.should_use_workspace_tools(
                "conv-render-followup",
                "show me",
                features,
                history=[
                    {
                        "role": "assistant",
                        "content": "If you want, I can open it in the viewer and preview the page.",
                    }
                ],
            )
        )

    def test_request_demands_agent_execution_inherits_run_offer_for_yes(self):
        self.assertTrue(
            app.request_demands_agent_execution(
                "yes",
                history=[
                    {
                        "role": "assistant",
                        "content": "If you want, I can run that and show you the real output here.",
                    }
                ],
            )
        )

    def test_select_enabled_tools_includes_web_tools_for_short_followup_after_offer(self):
        features = app.FeatureFlags(agent_tools=True, web_search=True)

        allowed = app.select_enabled_tools(
            "conv-web-followup",
            "look it up",
            features,
            history=[
                {
                    "role": "assistant",
                    "content": "If you want, I can search the web for current sources and compare the top options.",
                }
            ],
        )

        self.assertIn("web.search", allowed)
        self.assertIn("web.fetch_page", allowed)

    def test_conversation_search_history_result_does_not_crash_on_fts_path(self):
        original_db_path = app.DB_PATH
        original_semantic = app.fetch_semantic_message_candidates
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                db_path = pathlib.Path(tempdir) / "chat.db"
                app.DB_PATH = str(db_path)
                app.init_db()

                conn = sqlite3.connect(app.DB_PATH)
                conn.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    ("conv-search", "Search chat", "2026-04-10T00:00:00", "2026-04-10T00:00:00"),
                )
                conn.executemany(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, kind, feedback) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        ("conv-search", "user", "Can you remember anything from earlier in this chat?", "2026-04-10T00:00:00", "visible_chat", None),
                        ("conv-search", "assistant", "Yes, I can search the current conversation history when needed.", "2026-04-10T00:00:01", "visible_chat", "neutral"),
                        ("conv-search", "user", "Please tell me about prior messages.", "2026-04-10T00:00:02", "visible_chat", None),
                    ],
                )
                conn.commit()
                conn.close()

                app.fetch_semantic_message_candidates = lambda conversation_id, query, limit=0: []
                result = app.conversation_search_history_result("conv-search", "prior messages", limit=3)
        finally:
            app.DB_PATH = original_db_path
            app.fetch_semantic_message_candidates = original_semantic

        self.assertEqual(result["query"], "prior messages")
        self.assertGreaterEqual(result["count"], 1)
        self.assertTrue(any("prior" in match["content"].lower() for match in result["matches"]))

    def test_capture_context_eval_case_includes_tool_policy_trace(self):
        original_db_path = app.DB_PATH
        original_get_workspace_path = app.get_workspace_path
        original_cache = dict(app.MESSAGE_WORKFLOW_METADATA_CACHE)
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                db_path = pathlib.Path(tempdir) / "chat.db"
                workspace_root = pathlib.Path(tempdir) / "workspace"
                workspace_root.mkdir(parents=True, exist_ok=True)
                app.DB_PATH = str(db_path)
                app.init_db()
                resolved_workspace_root = workspace_root.resolve()
                app.get_workspace_path = lambda _conversation_id, create=True: resolved_workspace_root
                app.MESSAGE_WORKFLOW_METADATA_CACHE.clear()

                conn = sqlite3.connect(app.DB_PATH)
                conn.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    ("conv-capture", "Capture test", "2026-04-10T00:00:00", "2026-04-10T00:00:00"),
                )
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, kind, feedback) VALUES (?, ?, ?, ?, ?, ?)",
                    ("conv-capture", "user", "what is the weather in san francisco?", "2026-04-10T00:00:00", "visible_chat", None),
                )
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, kind, feedback) VALUES (?, ?, ?, ?, ?, ?)",
                    ("conv-capture", "assistant", "Here are some weather sites.", "2026-04-10T00:00:01", "visible_chat", "neutral"),
                )
                assistant_message_id = int(conn.execute("SELECT MAX(id) FROM messages").fetchone()[0])
                conn.commit()
                conn.close()

                execution = app.create_workflow_execution(
                    "conv-capture",
                    1,
                    "chat_turn",
                    {
                        "tool_policy": {
                            "web_search_requested": True,
                            "enabled_tools": ["web.search", "web.fetch_page"],
                            "has_web_tools": True,
                            "workspace_requested": False,
                        }
                    },
                )
                app.record_workflow_step(
                    execution,
                    step_name="respond",
                    call={"name": "web.search", "arguments": {"query": "weather san francisco"}},
                    result={"ok": True, "result": {}},
                )
                app.finalize_workflow_execution(
                    execution,
                    assistant_message_id=assistant_message_id,
                    final_outcome="completed_with_tools",
                )

                capture_path = app.capture_context_eval_case_for_assistant_message(
                    "conv-capture",
                    assistant_message_id,
                    trigger="retry",
                )
                payload = json.loads((resolved_workspace_root / capture_path).read_text(encoding="utf-8"))
        finally:
            app.DB_PATH = original_db_path
            app.get_workspace_path = original_get_workspace_path
            app.MESSAGE_WORKFLOW_METADATA_CACHE.clear()
            app.MESSAGE_WORKFLOW_METADATA_CACHE.update(original_cache)

        self.assertEqual(payload["capture"]["tool_policy"]["web_search_requested"], True)
        self.assertEqual(payload["capture"]["tool_policy"]["enabled_tools"], ["web.search", "web.fetch_page"])
        self.assertEqual(payload["capture"]["tool_names"], ["web.search"])
        self.assertEqual(payload["capture"]["final_outcome"], "completed_with_tools")

    def test_direct_search_route_handles_pure_web_facts_turns(self):
        prepared = app.PreparedTurnRequest(
            conversation_id="conv-weather",
            active_file_path="",
            turn_kind="visible_chat",
            user_message_id=1,
            saved_user_message="what is the weather in san francisco right now?",
            effective_message="what is the weather in san francisco right now?",
            model_message="what is the weather in san francisco right now?",
            history=[],
            model_history=[],
            system_prompt="system",
            requested_mode="deep",
            resolved_mode="deep",
            features=app.FeatureFlags(agent_tools=True, web_search=True),
            slash_command=None,
            max_tokens=1024,
            workspace_intent="none",
            tool_policy_trace={},
            enabled_tools=["web.search", "web.fetch_page"],
            auto_execute_workspace=False,
            resume_saved_workspace=False,
            plan_override_builder_steps=[],
            promoted_to_planning=False,
            repo_bootstrapped=False,
            repo_bootstrap_summary="",
            assessment=app.TurnAssessment(
                requires_search=True,
                primary_skill="search",
                execution_style="plan_preview",
                enabled_tools=["web.search", "web.fetch_page"],
            ),
        )

        self.assertTrue(app.should_route_prepared_turn_via_direct_search(prepared))

    def test_direct_search_route_does_not_hijack_workspace_search_turns(self):
        prepared = app.PreparedTurnRequest(
            conversation_id="conv-mixed",
            active_file_path="",
            turn_kind="visible_chat",
            user_message_id=2,
            saved_user_message="search docs and patch the file",
            effective_message="search docs and patch the file",
            model_message="search docs and patch the file",
            history=[],
            model_history=[],
            system_prompt="system",
            requested_mode="deep",
            resolved_mode="deep",
            features=app.FeatureFlags(agent_tools=True, web_search=True, workspace_write=True),
            slash_command=None,
            max_tokens=1024,
            workspace_intent="focused_write",
            tool_policy_trace={},
            enabled_tools=["web.search", "web.fetch_page", "workspace.patch_file"],
            auto_execute_workspace=False,
            resume_saved_workspace=False,
            plan_override_builder_steps=[],
            promoted_to_planning=False,
            repo_bootstrapped=False,
            repo_bootstrap_summary="",
            assessment=app.TurnAssessment(
                requires_search=True,
                primary_skill="plan_and_code",
                execution_style="plan_execution",
                workspace_intent="focused_write",
                enabled_tools=["web.search", "web.fetch_page", "workspace.patch_file"],
            ),
        )

        self.assertFalse(app.should_route_prepared_turn_via_direct_search(prepared))

    def test_handle_direct_search_prefetches_fact_pages_and_answers_directly(self):
        original_emit_direct_tool_call = app.emit_direct_tool_call
        original_vllm_chat_complete = app.vllm_chat_complete
        original_run_resumable_tool_loop = app.run_resumable_tool_loop

        class DummyWebSocket:
            async def send_json(self, _payload):
                return None

        recorded_calls = []

        async def fake_emit_direct_tool_call(
            websocket,
            conversation_id,
            call,
            *,
            features=None,
            status_prefix="",
            activity_phase="respond",
            activity_step_label=None,
            workflow_execution=None,
        ):
            recorded_calls.append(call["name"])
            if call["name"] == "web.search":
                return {
                    "id": call["id"],
                    "ok": True,
                    "result": {
                        "query": "what is the weather in san francisco",
                        "count": 2,
                        "results": [
                            {
                                "title": "San Francisco Weather",
                                "url": "https://weather.example/sf",
                                "domain": "weather.example",
                                "snippet": "Current conditions for San Francisco.",
                            },
                            {
                                "title": "San Francisco Hourly Forecast",
                                "url": "https://forecast.example/sf-hourly",
                                "domain": "forecast.example",
                                "snippet": "Hourly weather forecast for San Francisco.",
                            },
                        ],
                    },
                }
            if call["name"] == "web.fetch_page":
                return {
                    "id": call["id"],
                    "ok": True,
                    "result": {
                        "url": call["arguments"]["url"],
                        "final_url": call["arguments"]["url"],
                        "title": "San Francisco Weather",
                        "domain": "weather.example",
                        "content": "Current weather in San Francisco is 61 F with light wind and clear skies.",
                    },
                }
            raise AssertionError(f"Unexpected tool call: {call['name']}")

        async def fake_vllm_chat_complete(messages, max_tokens=None, temperature=None):
            self.assertIn("Current weather in San Francisco is 61 F", messages[-1]["content"])
            self.assertIn("Answer the user's question directly.", messages[-1]["content"])
            return "San Francisco is **61 F** with light wind and clear skies. [Weather](https://weather.example/sf)"

        async def fail_run_resumable_tool_loop(*args, **kwargs):
            raise AssertionError("Current-fact direct search should not fall back to the resumable tool loop")

        try:
            app.emit_direct_tool_call = fake_emit_direct_tool_call
            app.vllm_chat_complete = fake_vllm_chat_complete
            app.run_resumable_tool_loop = fail_run_resumable_tool_loop

            response = asyncio.run(
                app.handle_direct_search_command(
                    DummyWebSocket(),
                    "conv-weather-direct",
                    [],
                    "system",
                    1024,
                    app.FeatureFlags(agent_tools=True, web_search=True),
                    "what is the weather in san francisco",
                )
            )
        finally:
            app.emit_direct_tool_call = original_emit_direct_tool_call
            app.vllm_chat_complete = original_vllm_chat_complete
            app.run_resumable_tool_loop = original_run_resumable_tool_loop

        self.assertEqual(
            recorded_calls,
            ["web.search", "web.fetch_page", "web.fetch_page"],
        )
        self.assertIn("61 F", response)

    def test_workspace_command_env_disables_python_bytecode_clutter(self):
        env = app.build_workspace_command_env("conv-env-no-pyc")

        self.assertEqual(env.get("PYTHONDONTWRITEBYTECODE"), "1")


if __name__ == "__main__":
    unittest.main()
