import unittest

from src.python.ai_chat.deep_flow import DeepRouteRequest, decide_deep_route


class DeepFlowTests(unittest.TestCase):
    def test_preview_route_wins_when_requested(self):
        decision = decide_deep_route(
            DeepRouteRequest(
                should_preview_plan=True,
                workspace_write=False,
            )
        )
        self.assertEqual(decision.action, "preview_plan")
        self.assertTrue(decision.requires_plan)

    def test_missing_saved_plan_is_reported(self):
        decision = decide_deep_route(
            DeepRouteRequest(
                requires_existing_plan=True,
                has_plan=False,
                execution_requested=True,
                workspace_write=True,
            )
        )
        self.assertEqual(decision.action, "missing_plan")

    def test_execution_without_write_permission_is_blocked(self):
        decision = decide_deep_route(
            DeepRouteRequest(
                execution_requested=True,
                workspace_write=False,
            )
        )
        self.assertEqual(decision.action, "blocked_write")
        self.assertTrue(decision.requires_write_permission)

    def test_execution_with_write_permission_runs_review_loop(self):
        decision = decide_deep_route(
            DeepRouteRequest(
                auto_execute=True,
                workspace_write=True,
            )
        )
        self.assertEqual(decision.action, "execute_plan")
        self.assertTrue(decision.requires_review)

    def test_non_execution_turn_answers_directly(self):
        decision = decide_deep_route(DeepRouteRequest())
        self.assertEqual(decision.action, "direct_answer")


if __name__ == "__main__":
    unittest.main()
