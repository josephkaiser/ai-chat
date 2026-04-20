import unittest

from src.python.ai_chat.context_policy_program import (
    DEFAULT_CONTEXT_POLICY_PROGRAM,
    ContextPolicyInputs,
)


class ContextPolicyProgramTests(unittest.TestCase):
    def test_plan_phase_prefers_memory_and_workspace_samples(self):
        decision = DEFAULT_CONTEXT_POLICY_PROGRAM.run(
            ContextPolicyInputs(
                phase="plan",
                request_text="improve the chat flow using app.js context",
                changed_files=("src/web/app.js",),
                workspace_sample_paths=("src/web/app.js", "src/python/harness.py"),
                has_task_board=True,
                has_recent_feedback=True,
            )
        )

        self.assertTrue(decision.retrieve_memory)
        self.assertTrue(decision.retrieve_workspace_previews)
        self.assertGreaterEqual(decision.memory_limit, 1)
        self.assertIn(".ai/task-board.md", decision.preferred_paths)
        self.assertIn("src/web/app.js", decision.preferred_paths)

    def test_verify_phase_enables_scope_gaps_and_step_evidence(self):
        decision = DEFAULT_CONTEXT_POLICY_PROGRAM.run(
            ContextPolicyInputs(
                phase="verify",
                request_text="verify the file changes",
                changed_files=("src/web/app.js", "src/python/harness.py"),
                has_scope_gaps=True,
                has_step_reports=True,
            )
        )

        self.assertFalse(decision.retrieve_memory)
        self.assertTrue(decision.retrieve_workspace_previews)
        self.assertTrue(decision.include_scope_gaps)
        self.assertTrue(decision.include_step_evidence)
        self.assertGreaterEqual(decision.workspace_preview_limit, 1)


if __name__ == "__main__":
    unittest.main()
