import unittest

from src.python.ai_chat.route_intake import StructuredRouteIntake
from src.python.ai_chat.turn_strategy import build_turn_assessment, format_turn_assessment_summary, infer_explicit_planning_request


class TurnStrategyTests(unittest.TestCase):
    def test_broad_refactor_routes_into_planned_coding_loop(self):
        assessment = build_turn_assessment(
            message="Refactor this app, create the missing files, and run tests after each step.",
            requested_mode="normal",
            resolved_mode="deep",
            workspace_intent="broad_write",
            enabled_tools=[
                "workspace.list_files",
                "workspace.grep",
                "workspace.read_file",
                "workspace.patch_file",
                "workspace.run_command",
            ],
            workspace_requested=True,
            has_attachment_context=True,
            auto_execute_workspace=True,
            workspace_run_commands_enabled=True,
        )

        self.assertTrue(assessment.compare_to_rag)
        self.assertTrue(assessment.requires_search)
        self.assertTrue(assessment.requires_file_creation)
        self.assertTrue(assessment.requires_coding_loop)
        self.assertTrue(assessment.requires_planning)
        self.assertTrue(assessment.requires_step_review)
        self.assertEqual(assessment.primary_skill, "plan_and_code")
        self.assertEqual(assessment.execution_style, "plan_execution")

    def test_web_research_request_prefers_one_shot_search_skill(self):
        assessment = build_turn_assessment(
            message="Look up the latest FastAPI release and cite sources.",
            requested_mode="normal",
            resolved_mode="normal",
            workspace_intent="none",
            enabled_tools=["web.search", "web.fetch_page"],
            workspace_requested=False,
            web_search_requested=True,
        )

        self.assertFalse(assessment.compare_to_rag)
        self.assertTrue(assessment.requires_search)
        self.assertFalse(assessment.requires_planning)
        self.assertEqual(assessment.primary_skill, "search")
        self.assertEqual(assessment.execution_style, "one_shot_skill")

    def test_slash_plan_stays_in_plan_preview_until_execution_is_requested(self):
        assessment = build_turn_assessment(
            message="Refactor the authentication flow.",
            requested_mode="normal",
            resolved_mode="normal",
            workspace_intent="broad_write",
            enabled_tools=["workspace.list_files", "workspace.read_file"],
            workspace_requested=True,
            slash_command_name="plan",
        )

        self.assertTrue(assessment.requires_planning)
        self.assertEqual(assessment.primary_skill, "planning")
        self.assertEqual(assessment.execution_style, "plan_preview")

    def test_explicit_planning_language_is_detected(self):
        self.assertTrue(infer_explicit_planning_request("Plan how to refactor this app step by step."))
        self.assertFalse(infer_explicit_planning_request("Refactor this app now."))

    def test_direct_answer_stays_simple(self):
        assessment = build_turn_assessment(
            message="Explain JavaScript closures in one paragraph.",
            requested_mode="normal",
            resolved_mode="normal",
            workspace_intent="none",
            enabled_tools=[],
            workspace_requested=False,
        )

        self.assertFalse(assessment.compare_to_rag)
        self.assertFalse(assessment.requires_search)
        self.assertFalse(assessment.requires_file_creation)
        self.assertFalse(assessment.requires_coding_loop)
        self.assertFalse(assessment.requires_planning)
        self.assertEqual(assessment.primary_skill, "direct_answer")
        self.assertEqual(assessment.execution_style, "direct_answer")

    def test_summary_mentions_loop_stages(self):
        assessment = build_turn_assessment(
            message="Search the web and create a starter repo.",
            requested_mode="normal",
            resolved_mode="normal",
            workspace_intent="broad_write",
            enabled_tools=["web.search", "web.fetch_page", "workspace.patch_file"],
            workspace_requested=True,
            web_search_requested=True,
        )

        summary = format_turn_assessment_summary(assessment)
        self.assertIn("Skill loop:", summary)
        self.assertIn("search yes", summary)
        self.assertIn("files yes", summary)
        self.assertIn("review yes", summary)

    def test_structured_route_intake_overrides_search_and_fresh_info_signals(self):
        assessment = build_turn_assessment(
            message="can you summarize the changes to nvim 0.12",
            requested_mode="normal",
            resolved_mode="normal",
            workspace_intent="none",
            enabled_tools=["web.search", "web.fetch_page"],
            workspace_requested=False,
            route_intake=StructuredRouteIntake(
                needs_fresh_info=True,
                is_versioned_release_query=True,
                entity="nvim",
                time_sensitivity="versioned",
                answer_shape="summary",
                needs_search_citations=True,
                web_search_requested=True,
                reasoning="versioned release summary",
                confidence=0.92,
            ),
        )

        self.assertTrue(assessment.requires_search)
        self.assertTrue(assessment.needs_fresh_info)
        self.assertTrue(assessment.needs_search_citations)
        self.assertTrue(assessment.is_versioned_release_query)
        self.assertEqual(assessment.entity, "nvim")
        self.assertEqual(assessment.time_sensitivity, "versioned")
        self.assertEqual(assessment.answer_shape, "summary")


if __name__ == "__main__":
    unittest.main()
