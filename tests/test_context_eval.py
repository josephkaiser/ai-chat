import pathlib
import tempfile
import unittest

from src.python.ai_chat.context_eval import (
    CapturedContextEvalResult,
    DEFAULT_CONTEXT_EVAL_CASES,
    DEFAULT_CONTEXT_EVAL_CASES_PATH,
    find_context_eval_capture_files,
    load_context_eval_cases,
    replay_captured_context_eval_files,
    replay_captured_context_eval_payload,
    evaluate_context_case,
    serialize_context_eval_case,
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


if __name__ == "__main__":
    unittest.main()
