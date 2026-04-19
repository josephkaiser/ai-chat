import asyncio
import pathlib
import tempfile
import unittest
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
