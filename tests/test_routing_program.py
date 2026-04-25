import unittest

from src.python.ai_chat.route_intake import StructuredRouteIntake
from src.python.ai_chat.routing_program import DEFAULT_ROUTE_PROGRAM, RouteProgramInputs


class RoutingProgramTests(unittest.TestCase):
    def test_default_route_program_keeps_plan_and_code_signal(self):
        output = DEFAULT_ROUTE_PROGRAM.run(RouteProgramInputs(
            message="Refactor the workspace app and run tests after each step.",
            requested_mode="normal",
            resolved_mode="deep",
            workspace_intent="broad_write",
            enabled_tools=[
                "workspace.list_files",
                "workspace.read_file",
                "workspace.patch_file",
                "workspace.run_command",
            ],
            workspace_requested=True,
            auto_execute_workspace=True,
            workspace_run_commands_enabled=True,
        ))

        self.assertTrue(output.assessment.requires_planning)
        self.assertTrue(output.assessment.requires_coding_loop)
        self.assertEqual(output.assessment.primary_skill, "plan_and_code")
        self.assertEqual(output.assessment.execution_style, "plan_execution")

    def test_default_route_program_carries_structured_route_intake_into_assessment(self):
        output = DEFAULT_ROUTE_PROGRAM.run(RouteProgramInputs(
            message="can you summarize the changes to nvim 0.12",
            requested_mode="normal",
            resolved_mode="normal",
            workspace_intent="none",
            enabled_tools=["web.search", "web.fetch_page"],
            workspace_requested=False,
            web_search_requested=True,
            route_intake=StructuredRouteIntake(
                needs_fresh_info=True,
                is_versioned_release_query=True,
                entity="nvim",
                time_sensitivity="versioned",
                answer_shape="summary",
                needs_search_citations=True,
                web_search_requested=True,
                reasoning="versioned release summary",
                confidence=0.88,
            ),
        ))

        self.assertTrue(output.assessment.requires_search)
        self.assertTrue(output.assessment.needs_fresh_info)
        self.assertEqual(output.assessment.entity, "nvim")
        self.assertEqual(output.assessment.answer_shape, "summary")


if __name__ == "__main__":
    unittest.main()
