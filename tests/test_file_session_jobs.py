import asyncio
import pathlib
import tempfile
import unittest
import zipfile
from unittest import mock

try:
    import app
except Exception as exc:
    app = None
    APP_IMPORT_ERROR = exc
else:
    APP_IMPORT_ERROR = None


class FileSessionJobSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if app is None:
            raise unittest.SkipTest(f"app.py dependencies are unavailable: {APP_IMPORT_ERROR}")

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tempdir.name)
        self.workspace_root = self.root / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.catalog_root = self.root / "catalog"
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.python_env_root = self.root / "python-envs"
        self.python_env_root.mkdir(parents=True, exist_ok=True)
        self.voice_root = self.root / "voice"
        self.voice_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "chat.db"

        self.originals = {
            "DB_PATH": app.DB_PATH,
            "WORKSPACE_ROOT": app.WORKSPACE_ROOT,
            "WORKSPACE_ROOT_PATH": app.WORKSPACE_ROOT_PATH,
            "RUNS_ROOT": app.RUNS_ROOT,
            "RUNS_ROOT_PATH": app.RUNS_ROOT_PATH,
            "MANAGED_PYTHON_ENVS_ROOT": app.MANAGED_PYTHON_ENVS_ROOT,
            "MANAGED_PYTHON_ENVS_ROOT_PATH": app.MANAGED_PYTHON_ENVS_ROOT_PATH,
            "VOICE_ROOT": app.VOICE_ROOT,
            "VOICE_ROOT_PATH": app.VOICE_ROOT_PATH,
        }

        app.DB_PATH = str(self.db_path)
        app.WORKSPACE_ROOT = str(self.catalog_root)
        app.WORKSPACE_ROOT_PATH = self.catalog_root
        app.RUNS_ROOT = str(self.runs_root)
        app.RUNS_ROOT_PATH = self.runs_root
        app.MANAGED_PYTHON_ENVS_ROOT = str(self.python_env_root)
        app.MANAGED_PYTHON_ENVS_ROOT_PATH = self.python_env_root
        app.VOICE_ROOT = str(self.voice_root)
        app.VOICE_ROOT_PATH = self.voice_root

        app.init_db()
        self.workspace = app.create_workspace_record(
            display_name="Test Workspace",
            root_path=str(self.workspace_root),
        )

    def tearDown(self):
        app.DB_PATH = self.originals["DB_PATH"]
        app.WORKSPACE_ROOT = self.originals["WORKSPACE_ROOT"]
        app.WORKSPACE_ROOT_PATH = self.originals["WORKSPACE_ROOT_PATH"]
        app.RUNS_ROOT = self.originals["RUNS_ROOT"]
        app.RUNS_ROOT_PATH = self.originals["RUNS_ROOT_PATH"]
        app.MANAGED_PYTHON_ENVS_ROOT = self.originals["MANAGED_PYTHON_ENVS_ROOT"]
        app.MANAGED_PYTHON_ENVS_ROOT_PATH = self.originals["MANAGED_PYTHON_ENVS_ROOT_PATH"]
        app.VOICE_ROOT = self.originals["VOICE_ROOT"]
        app.VOICE_ROOT_PATH = self.originals["VOICE_ROOT_PATH"]
        self.tempdir.cleanup()

    def test_background_job_updates_persist_into_session_record(self):
        path = "drafts/example.md"
        app.ensure_file_session_record(self.workspace["id"], path)

        job = app.create_file_session_job_record(
            self.workspace["id"],
            path,
            lane="background",
            job_kind="optimize_draft",
            title="Optimize example.md",
            status="queued",
        )

        queued_session = app.get_file_session_record(self.workspace["id"], path)
        queued_summary = queued_session["job_summary"]
        self.assertEqual(queued_summary["current_lane"], "background")
        self.assertEqual(queued_summary["current_status"], "queued")
        self.assertEqual(queued_summary["current_job"]["id"], job["id"])
        self.assertIsNone(queued_summary["last_result"])

        app.update_file_session_job_record(self.workspace["id"], job["id"], status="running")
        running_session = app.get_file_session_record(self.workspace["id"], path)
        self.assertEqual(running_session["job_summary"]["current_status"], "running")

        app.update_file_session_job_record(
            self.workspace["id"],
            job["id"],
            status="failed",
            error_text="Evaluation failed",
        )
        finished_session = app.get_file_session_record(self.workspace["id"], path)
        finished_summary = finished_session["job_summary"]
        self.assertEqual(finished_summary["current_status"], "idle")
        self.assertIsNone(finished_summary["current_job"])
        self.assertEqual(finished_summary["last_result"]["lane"], "background")
        self.assertEqual(finished_summary["last_result"]["status"], "failed")
        self.assertEqual(finished_summary["last_result"]["error_text"], "Evaluation failed")
        self.assertEqual(finished_session["latest_job"]["id"], job["id"])
        self.assertIsNone(finished_session["active_job"])

    def test_background_focus_moves_to_new_file_and_supersedes_old_jobs(self):
        first_path = "drafts/first.html"
        second_path = "drafts/second.html"
        first_session = app.ensure_file_session_record(self.workspace["id"], first_path)
        second_session = app.ensure_file_session_record(self.workspace["id"], second_path)

        first_job = app.create_file_session_job_record(
            self.workspace["id"],
            first_path,
            lane="background",
            job_kind="optimize_output",
            status="queued",
        )

        app.activate_background_focus_for_file(self.workspace["id"], first_path)
        moved_workspace = app.activate_background_focus_for_file(self.workspace["id"], second_path)

        refreshed_first_job = app.get_file_session_job_record(self.workspace["id"], first_job["id"])
        self.assertEqual(refreshed_first_job["status"], "superseded")
        self.assertIn("moved to another file", refreshed_first_job["error_text"])
        self.assertEqual(moved_workspace["background_focus_path"], second_path)
        self.assertTrue(moved_workspace["background_focus_enabled"])

        enriched = app.enrich_file_session_records(
            self.workspace["id"],
            [first_session, second_session],
        )
        by_path = {item["path"]: item for item in enriched}
        self.assertFalse(by_path[first_path]["background_focus_active"])
        self.assertTrue(by_path[second_path]["background_focus_active"])

    def test_process_chat_turn_marks_foreground_job_completed_server_side(self):
        path = "drafts/active.md"
        session = app.ensure_file_session_record(self.workspace["id"], path)
        foreground_job = app.create_file_session_job_record(
            self.workspace["id"],
            path,
            lane="foreground",
            job_kind="realize_draft",
            title="Realize active.md",
            status="queued",
            source_conversation_id=session["conversation_id"],
        )

        async def fake_orchestrated_chat(*_args, **_kwargs):
            return "Updated the draft."

        with (
            mock.patch.object(app, "orchestrated_chat", side_effect=fake_orchestrated_chat),
            mock.patch.object(app, "maybe_bootstrap_workspace_from_current_repo", return_value=None),
            mock.patch.object(app, "schedule_conversation_summary_refresh", return_value=None),
            mock.patch.object(app, "queue_background_optimize_job_for_file_session", return_value={}),
        ):
            transport = app.BufferedChatTransport()
            asyncio.run(
                app.process_chat_turn(
                    transport,
                    {
                        "message": "Update the active draft.",
                        "conversation_id": session["conversation_id"],
                        "workspace_id": self.workspace["id"],
                        "file_path": path,
                        "attachments": [],
                        "mode": "deep",
                        "features": {},
                        "slash_command": None,
                        "plan_override_steps": [],
                    },
                )
            )

        updated_session = app.get_file_session_record(self.workspace["id"], path)
        summary = updated_session["job_summary"]
        self.assertEqual(summary["current_status"], "idle")
        self.assertIsNone(summary["current_job"])
        self.assertEqual(summary["last_result"]["lane"], "foreground")
        self.assertEqual(summary["last_result"]["status"], "completed")
        self.assertEqual(summary["last_result"]["id"], foreground_job["id"])
        self.assertEqual(updated_session["latest_job"]["id"], foreground_job["id"])
        self.assertIsNone(updated_session["active_job"])
        self.assertEqual([event.get("type") for event in transport.events if isinstance(event, dict)][-2:], ["message_id", "done"])

    def test_process_chat_turn_bootstraps_missing_visible_output_before_main_generation(self):
        path = "drafts/site.html"
        session = app.ensure_file_session_record(self.workspace["id"], path)
        app.write_workspace_text_for_session(
            self.workspace["id"],
            app.file_session_spec_path(path),
            (
                'Create a website for Joe with a "Welcome to Joe\'s World" splash page, '
                "a CTA button, fun gradients, math, science, ai / ml, statistics, hockey, and basketball."
            ),
        )

        async def fake_orchestrated_chat(*_args, **_kwargs):
            return "Refined the draft."

        with (
            mock.patch.object(app, "orchestrated_chat", side_effect=fake_orchestrated_chat),
            mock.patch.object(app, "maybe_bootstrap_workspace_from_current_repo", return_value=None),
            mock.patch.object(app, "schedule_conversation_summary_refresh", return_value=None),
            mock.patch.object(app, "queue_background_optimize_job_for_file_session", return_value={}),
        ):
            transport = app.BufferedChatTransport()
            asyncio.run(
                app.process_chat_turn(
                    transport,
                    {
                        "message": "Build the active draft.",
                        "conversation_id": session["conversation_id"],
                        "workspace_id": self.workspace["id"],
                        "file_path": path,
                        "attachments": [],
                        "mode": "deep",
                        "features": {},
                        "slash_command": None,
                        "plan_override_steps": [],
                    },
                )
            )

        output = app.read_workspace_text_for_session(self.workspace["id"], path)
        self.assertIn("Welcome to Joe&#x27;s World", output)
        self.assertIn("Enter the build", output)
        event_types = [event.get("type") for event in transport.events if isinstance(event, dict)]
        self.assertIn("draft_bootstrap", event_types)
        self.assertLess(event_types.index("draft_bootstrap"), event_types.index("start"))

    def test_bootstrap_content_for_typescript_file_looks_like_code(self):
        content = app.build_file_session_bootstrap_content(
            "drafts/app.ts",
            "Create a typescript app that compiles a simple static html webpage.",
        )

        self.assertIn("function renderApp", content)
        self.assertIn("console.log(renderApp());", content)
        self.assertNotIn("\nApp\n\n", content)
        self.assertNotIn("\n- Create a typescript app", content)

    def test_bootstrap_content_for_text_file_reads_like_prose(self):
        content = app.build_file_session_bootstrap_content(
            "drafts/note.txt",
            "Write a note about how good I am at programming.",
        )

        self.assertIn("A note about how good I am at programming.", content)
        self.assertNotIn("\nNote\n\n", content)
        self.assertNotIn("\n- Write a note", content)

    def test_bootstrap_refreshes_legacy_typescript_placeholder_output(self):
        path = "drafts/app.ts"
        app.ensure_file_session_record(self.workspace["id"], path)
        app.write_workspace_text_for_session(
            self.workspace["id"],
            app.file_session_spec_path(path),
            "Create a typescript app that compiles a simple static html webpage.",
        )
        app.write_workspace_text_for_session(
            self.workspace["id"],
            path,
            (
                "App\n\n"
                "a typescript app that compiles a simple static html webpage.\n\n"
                "- Create a typescript app that compiles a simple static html webpage.\n"
            ),
        )

        payload = app.maybe_bootstrap_visible_file_session_output(self.workspace["id"], path)
        refreshed = app.read_workspace_text_for_session(self.workspace["id"], path)

        self.assertIsNotNone(payload)
        self.assertIn("function renderApp", refreshed)
        self.assertNotIn("\nApp\n\n", refreshed)

    def test_extract_workspace_archive_safely_expands_zip_into_workspace(self):
        archive_path = self.workspace_root / "uploads" / "bundle.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("src/main.py", "print('hi')\n")
            archive.writestr("../ignored.txt", "nope")
            archive.writestr("docs/readme.md", "# Hello\n")

        payload = app.extract_workspace_archive_for_workspace(self.workspace["id"], "uploads/bundle.zip")

        self.assertEqual(payload["archive_path"], "uploads/bundle.zip")
        self.assertEqual(payload["count"], 2)
        self.assertTrue((self.workspace_root / payload["destination_path"] / "src" / "main.py").exists())
        self.assertTrue((self.workspace_root / payload["destination_path"] / "docs" / "readme.md").exists())
        self.assertFalse((self.workspace_root / "ignored.txt").exists())

    def test_infer_auto_command_plan_for_typescript_project_bootstraps_npm_then_typechecks(self):
        path = "web/src/app.ts"
        session = app.ensure_file_session_record(self.workspace["id"], path)
        app_file = self.workspace_root / path
        app_file.parent.mkdir(parents=True, exist_ok=True)
        app_file.write_text("export const answer: number = 42;\n", encoding="utf-8")
        (self.workspace_root / "web" / "package.json").write_text(
            (
                "{\n"
                '  "name": "demo",\n'
                '  "scripts": {\n'
                '    "typecheck": "tsc --noEmit -p tsconfig.json"\n'
                "  }\n"
                "}\n"
            ),
            encoding="utf-8",
        )
        (self.workspace_root / "web" / "package-lock.json").write_text("{\n}\n", encoding="utf-8")
        (self.workspace_root / "web" / "tsconfig.json").write_text(
            '{ "compilerOptions": { "target": "ES2020" }, "include": ["src/**/*.ts"] }\n',
            encoding="utf-8",
        )

        plan = app.infer_auto_command_plan(session["conversation_id"], path)

        self.assertEqual(
            plan,
            [
                {
                    "command": ["npm", "ci", "--no-audit", "--no-fund"],
                    "cwd": "web",
                    "label": "Prepare Node workspace with npm ci",
                    "phase": "setup",
                },
                {
                    "command": ["npm", "run", "typecheck"],
                    "cwd": "web",
                    "label": "Auto-verify with npm run typecheck",
                    "phase": "verify",
                },
            ],
        )

    def test_infer_auto_command_plan_for_plain_typescript_uses_tsc_without_package_json(self):
        path = "standalone/app.ts"
        session = app.ensure_file_session_record(self.workspace["id"], path)
        app_file = self.workspace_root / path
        app_file.parent.mkdir(parents=True, exist_ok=True)
        app_file.write_text("const answer: number = 42;\n", encoding="utf-8")
        (self.workspace_root / "standalone" / "tsconfig.json").write_text(
            '{ "compilerOptions": { "target": "ES2020" }, "include": ["app.ts"] }\n',
            encoding="utf-8",
        )

        plan = app.infer_auto_command_plan(session["conversation_id"], path)

        self.assertEqual(
            plan,
            [
                {
                    "command": ["tsc", "--noEmit", "-p", "tsconfig.json"],
                    "cwd": "standalone",
                    "label": "Auto-verify with tsc --noEmit",
                    "phase": "verify",
                },
            ],
        )

    def test_infer_auto_command_plan_for_rust_project_uses_cargo_fetch_then_check(self):
        path = "rust/src/lib.rs"
        session = app.ensure_file_session_record(self.workspace["id"], path)
        rust_file = self.workspace_root / path
        rust_file.parent.mkdir(parents=True, exist_ok=True)
        rust_file.write_text("pub fn meaning() -> i32 { 42 }\n", encoding="utf-8")
        (self.workspace_root / "rust" / "Cargo.toml").write_text(
            (
                "[package]\n"
                'name = "demo"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
            ),
            encoding="utf-8",
        )

        plan = app.infer_auto_command_plan(session["conversation_id"], path)

        self.assertEqual(
            plan,
            [
                {
                    "command": ["cargo", "fetch"],
                    "cwd": "rust",
                    "label": "Prepare Rust workspace with cargo fetch",
                    "phase": "setup",
                },
                {
                    "command": ["cargo", "check"],
                    "cwd": "rust",
                    "label": "Auto-verify with cargo check",
                    "phase": "verify",
                },
            ],
        )

    def test_delete_file_session_removes_hidden_artifacts_and_empty_conversation(self):
        path = "drafts/cleanup.md"
        session = app.ensure_file_session_record(self.workspace["id"], path)
        spec_path = app.file_session_spec_path(path)
        version_path = app.file_session_version_path(path)
        candidate_path = app.file_session_candidate_path(path)
        evaluation_path = app.file_session_evaluation_path(path)

        app.write_workspace_text_for_session(self.workspace["id"], spec_path, "temporary draft spec")
        app.write_workspace_text_for_session(self.workspace["id"], version_path, "old snapshot")
        app.write_workspace_text_for_session(self.workspace["id"], candidate_path, "candidate")
        app.write_workspace_text_for_session(self.workspace["id"], evaluation_path, '{"decision":"keep_current"}')

        result = app.delete_file_session_record(self.workspace["id"], session["id"])
        spec_file = self.workspace_root / spec_path
        version_file = self.workspace_root / version_path
        candidate_file = self.workspace_root / candidate_path
        evaluation_file = self.workspace_root / evaluation_path

        self.assertTrue(result["deleted"])
        self.assertIsNone(app.get_file_session_record(self.workspace["id"], path))
        self.assertFalse(spec_file.exists())
        self.assertFalse(version_file.exists())
        self.assertFalse(candidate_file.exists())
        self.assertFalse(evaluation_file.exists())
        self.assertIsNone(app.get_conversation_record(session["conversation_id"]))

    def test_background_job_queues_follow_up_pass_after_promoting_incomplete_candidate(self):
        path = "drafts/site.html"
        session = app.ensure_file_session_record(self.workspace["id"], path)
        spec_path = app.file_session_spec_path(path)
        app.write_workspace_text_for_session(
            self.workspace["id"],
            spec_path,
            "Build a slick HTML dev site with a hero CTA that opens the main content.",
        )
        app.write_workspace_text_for_session(
            self.workspace["id"],
            path,
            "<html><body><h1>Hello</h1></body></html>",
        )

        job = app.queue_background_optimize_job_for_file_session(
            self.workspace["id"],
            path,
            source_conversation_id=session["conversation_id"],
            attempt=1,
        )
        payload = dict(job["payload"])
        candidate_path = payload["candidate_output_path"]
        evaluation_path = payload["evaluation_output_path"]

        async def fake_hidden_turn(_workspace_id, _conversation_id, _file_path, prompt, **_kwargs):
            if "Write an improved candidate" in prompt:
                app.write_workspace_text_for_session(
                    self.workspace["id"],
                    candidate_path,
                    "<html><body><button>Enter Joe's World</button><main hidden>Real site</main></body></html>",
                )
            else:
                app.write_workspace_text_for_session(
                    self.workspace["id"],
                    evaluation_path,
                    '{"decision":"promote","summary":"Much better but can still be polished.","current_score":3,"candidate_score":7.5,"should_promote":true,"follow_up_needed":true}',
                )
            return []

        with mock.patch.object(app, "run_hidden_file_session_turn", side_effect=fake_hidden_turn):
            asyncio.run(app.process_background_file_session_job(job))

        jobs = app.list_file_session_job_records(self.workspace["id"], session["id"], lane="background", limit=5)
        self.assertEqual(jobs[0]["status"], "queued")
        self.assertEqual(jobs[0]["payload"].get("attempt"), 2)
        self.assertEqual(jobs[1]["id"], job["id"])
        self.assertEqual(jobs[1]["status"], "completed")
        self.assertIn("Enter Joe's World", app.read_workspace_text_for_session(self.workspace["id"], path))


if __name__ == "__main__":
    unittest.main()
