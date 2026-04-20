import pathlib
import tempfile
import unittest
import json

from src.python.ai_chat import context_eval
from src.python.ai_chat.context_eval import (
    CapturedContextEvalResult,
    DEFAULT_CONTEXT_EVAL_CASES,
    DEFAULT_CONTEXT_EVAL_CASES_PATH,
    list_promoted_context_eval_fixtures,
    load_promoted_context_eval_fixture_detail,
    find_context_eval_capture_files,
    load_context_eval_cases,
    normalize_promoted_context_eval_case_payload,
    promoted_context_eval_case_path,
    replay_captured_context_eval_files,
    replay_captured_context_eval_payload,
    evaluate_context_case,
    serialize_context_eval_case,
    update_promoted_context_eval_fixture_review_state,
    write_promoted_context_eval_case,
    summarize_captured_context_eval_results,
    summarize_context_eval_results,
)


class ContextEvalTests(unittest.TestCase):
    def test_default_context_eval_cases_load_from_fixture(self):
        loaded = load_context_eval_cases(DEFAULT_CONTEXT_EVAL_CASES_PATH)

        self.assertEqual([case.name for case in loaded], [case.name for case in DEFAULT_CONTEXT_EVAL_CASES])
        self.assertTrue(pathlib.Path(DEFAULT_CONTEXT_EVAL_CASES_PATH).exists())

    def test_default_context_eval_cases_pass(self):
        results = [evaluate_context_case(case) for case in DEFAULT_CONTEXT_EVAL_CASES]
        summary = summarize_context_eval_results(results)

        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(summary["failed_cases"], 0)
        self.assertEqual(summary["passed_cases"], len(DEFAULT_CONTEXT_EVAL_CASES))
        self.assertEqual(summary["average_score"], 1.0)

    def test_failed_case_reports_failed_checks(self):
        case = DEFAULT_CONTEXT_EVAL_CASES[0]
        broken_case = type(case)(
            name="broken",
            policy_inputs=case.policy_inputs,
            selection_candidates=case.selection_candidates,
            selection_max_sections=case.selection_max_sections,
            expectation=type(case.expectation)(
                retrieve_memory=case.expectation.retrieve_memory,
                retrieve_workspace_previews=case.expectation.retrieve_workspace_previews,
                min_memory_limit=case.expectation.min_memory_limit,
                min_workspace_preview_limit=case.expectation.min_workspace_preview_limit,
                required_selected_keys=("missing_key_that_will_never_be_selected",),
            ),
        )

        result = evaluate_context_case(broken_case)

        self.assertFalse(result.passed)
        self.assertTrue(result.failed_checks)
        self.assertLess(result.score, 1.0)

    def test_context_eval_case_round_trips_through_serialization(self):
        case = DEFAULT_CONTEXT_EVAL_CASES[0]

        payload = serialize_context_eval_case(case)
        loaded = load_context_eval_cases(DEFAULT_CONTEXT_EVAL_CASES_PATH)[0]

        self.assertEqual(payload["name"], case.name)
        self.assertEqual(loaded.name, case.name)
        self.assertEqual(loaded.selection_max_sections, case.selection_max_sections)

    def test_load_context_eval_cases_supports_promoted_fixture_directory(self):
        case = DEFAULT_CONTEXT_EVAL_CASES[0]
        payload = serialize_context_eval_case(case)

        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            base_path = root / "context_eval_cases.json"
            base_path.write_text(__import__("json").dumps([payload]), encoding="utf-8")
            promoted_dir = root / "context_eval_cases.d"
            promoted_dir.mkdir(parents=True, exist_ok=True)
            (promoted_dir / "captured.json").write_text(
                json.dumps(normalize_promoted_context_eval_case_payload(payload, fixture_name="captured_verify_case")),
                encoding="utf-8",
            )
            (promoted_dir / "superseded.json").write_text(
                json.dumps(
                    normalize_promoted_context_eval_case_payload(
                        payload,
                        fixture_name="superseded_verify_case",
                        review_status="superseded",
                    )
                ),
                encoding="utf-8",
            )

            loaded_base = load_context_eval_cases(base_path)
            loaded_promoted = load_context_eval_cases(promoted_dir)

        self.assertEqual(len(loaded_base), 1)
        self.assertEqual(loaded_base[0].name, case.name)
        self.assertEqual(len(loaded_promoted), 1)
        self.assertEqual(loaded_promoted[0].name, "captured_verify_case")

    def test_write_promoted_context_eval_case_persists_normalized_fixture(self):
        payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[0])
        payload["capture"] = {"trigger": "retry", "phase": "plan"}

        with tempfile.TemporaryDirectory() as tempdir:
            target = write_promoted_context_eval_case(
                payload,
                fixture_name="Plan Missing Workspace Evidence",
                review_status="accepted",
                fixtures_dir=tempdir,
            )
            written = json.loads(pathlib.Path(target).read_text(encoding="utf-8"))

        self.assertEqual(target.name, "plan_missing_workspace_evidence.json")
        self.assertEqual(written["name"], "Plan Missing Workspace Evidence")
        self.assertNotIn("capture", written)
        self.assertIn("policy_inputs", written)
        self.assertIn("expectation", written)
        self.assertEqual(written["fixture_metadata"]["review_status"], "accepted")

    def test_update_promoted_fixture_review_state_and_list_metadata(self):
        payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[0])

        with tempfile.TemporaryDirectory() as tempdir:
            target = write_promoted_context_eval_case(
                payload,
                fixture_name="Review Queue Case",
                fixtures_dir=tempdir,
            )
            updated = update_promoted_context_eval_fixture_review_state(
                target,
                "superseded",
                fixtures_dir=tempdir,
                updated_at="2026-04-19T12:00:00+00:00",
            )
            listed = list_promoted_context_eval_fixtures(tempdir)
            written = json.loads(pathlib.Path(updated).read_text(encoding="utf-8"))

        self.assertEqual(pathlib.Path(updated).resolve(), pathlib.Path(target).resolve())
        self.assertEqual(written["fixture_metadata"]["review_status"], "superseded")
        self.assertEqual(written["fixture_metadata"]["updated_at"], "2026-04-19T12:00:00+00:00")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["review_status"], "superseded")

    def test_load_promoted_fixture_detail_returns_payload_and_metadata(self):
        payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[0])
        payload["capture"] = {"trigger": "retry", "phase": "plan", "conversation_id": "conv-1"}

        with tempfile.TemporaryDirectory() as tempdir:
            target = write_promoted_context_eval_case(
                payload,
                fixture_name="Detail Case",
                fixtures_dir=tempdir,
            )
            detail = load_promoted_context_eval_fixture_detail(target, fixtures_dir=tempdir)

        self.assertEqual(detail["name"], "Detail Case")
        self.assertEqual(detail["review_status"], "candidate")
        self.assertEqual(detail["source_trigger"], "retry")
        self.assertEqual(detail["source_conversation_id"], "conv-1")
        self.assertIn("payload", detail)
        self.assertEqual(detail["payload"]["name"], "Detail Case")

    def test_replay_captured_context_eval_files_from_workspace_dir(self):
        case = DEFAULT_CONTEXT_EVAL_CASES[0]
        payload = serialize_context_eval_case(case)
        payload["capture"] = {"trigger": "retry", "phase": case.policy_inputs.phase}

        with tempfile.TemporaryDirectory() as tempdir:
            workspace = pathlib.Path(tempdir)
            capture_dir = workspace / ".ai" / "context-evals"
            capture_dir.mkdir(parents=True, exist_ok=True)
            target = capture_dir / "sample.json"
            target.write_text(__import__("json").dumps(payload), encoding="utf-8")

            files = find_context_eval_capture_files(workspace)
            results = replay_captured_context_eval_files(files)

        self.assertEqual(len(files), 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].result.passed)
        self.assertEqual(results[0].capture["trigger"], "retry")

    def test_summarize_captured_context_eval_results_counts_failures(self):
        good_payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[0])
        good_payload["capture"] = {"trigger": "retry", "phase": "plan"}
        bad_payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[1])
        bad_payload["capture"] = {"trigger": "explicit_feedback", "phase": "verify"}
        bad_payload["expectation"]["forbidden_selected_keys"] = ["conversation_context", "workspace_excerpts"]

        results = [
            replay_captured_context_eval_payload(good_payload, source_path="/tmp/good.json"),
            replay_captured_context_eval_payload(bad_payload, source_path="/tmp/bad.json"),
        ]
        summary = summarize_captured_context_eval_results(results)

        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(summary["failed_cases"], 1)
        self.assertIn("trigger_counts", summary)
        self.assertIn("explicit_feedback", summary["trigger_counts"])
        self.assertTrue(summary["recent_failures"])
        self.assertTrue(summary["top_triage_buckets"])
        self.assertEqual(summary["recommended_fix"]["key"], "forbidden_selected:workspace_excerpts")
        self.assertEqual(summary["top_triage_buckets"][0]["severity"], "high")
        self.assertEqual(summary["top_triage_buckets"][0]["case_count"], 1)

    def test_summarize_captured_context_eval_results_groups_similar_failures(self):
        missing_workspace_payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[0])
        missing_workspace_payload["capture"] = {"trigger": "retry", "phase": "plan"}
        missing_workspace_payload["expectation"]["required_selected_keys"] = ["workspace_excerpts", "missing_key"]

        another_missing_workspace_payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[0])
        another_missing_workspace_payload["capture"] = {"trigger": "implicit_feedback", "phase": "plan"}
        another_missing_workspace_payload["expectation"]["required_selected_keys"] = ["workspace_excerpts", "missing_key"]

        results = [
            replay_captured_context_eval_payload(missing_workspace_payload, source_path="/tmp/one.json"),
            replay_captured_context_eval_payload(another_missing_workspace_payload, source_path="/tmp/two.json"),
        ]

        summary = summarize_captured_context_eval_results(results)
        top_bucket = summary["top_triage_buckets"][0]

        self.assertEqual(top_bucket["key"], "missing_selected:missing_key")
        self.assertEqual(top_bucket["failure_count"], 2)
        self.assertEqual(top_bucket["case_count"], 2)
        self.assertIn("retry", top_bucket["trigger_counts"])
        self.assertIn("plan", top_bucket["phase_counts"])
        self.assertTrue(top_bucket["example_cases"])

    def test_summarize_captured_context_eval_results_includes_fixture_coverage(self):
        missing_workspace_payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[0])
        missing_workspace_payload["capture"] = {"trigger": "retry", "phase": "plan"}
        missing_workspace_payload["expectation"]["required_selected_keys"] = ["missing_key"]

        fixture_payload = normalize_promoted_context_eval_case_payload(
            missing_workspace_payload,
            fixture_name="Accepted Missing Key Fixture",
            review_status="accepted",
        )

        with tempfile.TemporaryDirectory() as tempdir:
            promoted_dir = pathlib.Path(tempdir)
            (promoted_dir / "accepted_missing_key_fixture.json").write_text(
                json.dumps(fixture_payload),
                encoding="utf-8",
            )
            original_dir = context_eval.DEFAULT_CONTEXT_EVAL_CASES_DIR
            context_eval.DEFAULT_CONTEXT_EVAL_CASES_DIR = promoted_dir
            try:
                results = [
                    replay_captured_context_eval_payload(missing_workspace_payload, source_path="/tmp/one.json"),
                ]
                summary = summarize_captured_context_eval_results(results)
            finally:
                context_eval.DEFAULT_CONTEXT_EVAL_CASES_DIR = original_dir

        top_bucket = summary["top_triage_buckets"][0]
        self.assertEqual(top_bucket["key"], "missing_selected:missing_key")
        self.assertEqual(top_bucket["fixture_coverage"]["accepted_count"], 1)
        self.assertEqual(top_bucket["fixture_coverage"]["total_fixtures"], 1)

    def test_summarize_captured_context_eval_results_adds_promotion_suggestion_for_repeated_uncovered_bucket(self):
        repeated_payload = serialize_context_eval_case(DEFAULT_CONTEXT_EVAL_CASES[0])
        repeated_payload["capture"] = {"trigger": "thumbs_down", "phase": "verify"}
        repeated_payload["expectation"]["required_selected_keys"] = ["verification_log"]

        another_payload = json.loads(json.dumps(repeated_payload))
        another_payload["name"] = f"{repeated_payload['name']}_retry"

        results = [
            replay_captured_context_eval_payload(repeated_payload, source_path="/tmp/verify-one.json"),
            replay_captured_context_eval_payload(another_payload, source_path="/tmp/verify-two.json"),
        ]
        summary = summarize_captured_context_eval_results(results)

        top_bucket = summary["top_triage_buckets"][0]
        self.assertEqual(top_bucket["key"], "missing_selected:verification_log")
        self.assertTrue(top_bucket["promotion_suggestion"]["should_suggest"])
        self.assertEqual(top_bucket["promotion_suggestion"]["suggested_review_status"], "candidate")
        self.assertIn("Repeated across 2 captured cases", top_bucket["promotion_suggestion"]["reason"])


if __name__ == "__main__":
    unittest.main()
