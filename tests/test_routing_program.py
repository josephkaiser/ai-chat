import unittest

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


if __name__ == "__main__":
    unittest.main()
