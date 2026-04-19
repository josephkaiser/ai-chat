import unittest

from src.python.ai_chat.task_engine import run_task
from src.python.ai_chat.task_planner import build_heuristic_task_plan, supports_structured_task_tools
from src.python.ai_chat.task_types import TaskContext, TaskPathTarget, TaskResult, TaskState, TaskStep


class StructuredTaskPlannerTests(unittest.TestCase):
    def test_supports_only_expected_workspace_tools(self):
        self.assertTrue(
            supports_structured_task_tools(["workspace.read_file", "workspace.grep"], "focused_read")
        )
        self.assertFalse(
            supports_structured_task_tools(["workspace.read_file", "web.search"], "focused_read")
        )
        self.assertTrue(
            supports_structured_task_tools(
                ["workspace.read_file", "workspace.patch_file", "workspace.run_command"],
                "focused_write",
            )
        )
        self.assertFalse(
            supports_structured_task_tools(
                ["workspace.read_file", "workspace.patch_file", "web.search"],
                "focused_write",
            )
        )

    def test_file_grounded_plan_reads_file_then_finishes(self):
        context = TaskContext(
            conversation_id="conv",
            user_message="Explain src/web/app.js",
            history=[],
            workspace_intent="focused_read",
            allowed_tools=["workspace.read_file"],
            path_targets=[TaskPathTarget(path="src/web/app.js", kind="file")],
            request_focus="Explain src/web/app.js",
        )

        plan = build_heuristic_task_plan(context)

        self.assertEqual([step.kind for step in plan], ["read", "finish"])
        self.assertEqual(plan[0].args["path"], "src/web/app.js")

    def test_directory_grounded_plan_lists_then_searches(self):
        context = TaskContext(
            conversation_id="conv",
            user_message="Inspect docs for workspace behavior",
            history=[],
            workspace_intent="broad_read",
            allowed_tools=["workspace.list_files", "workspace.grep"],
            path_targets=[TaskPathTarget(path="docs", kind="dir")],
            request_focus="Inspect docs for workspace behavior",
        )

        plan = build_heuristic_task_plan(context)

        self.assertEqual([step.kind for step in plan], ["list", "search", "finish"])
        self.assertEqual(plan[0].args["path"], "docs")
        self.assertTrue(plan[1].args["query"])

    def test_single_file_write_plan_reads_then_patches_then_finishes(self):
        context = TaskContext(
            conversation_id="conv",
            user_message="Fix the close handler in src/web/app.js",
            history=[],
            workspace_intent="focused_write",
            allowed_tools=["workspace.read_file", "workspace.patch_file", "workspace.run_command"],
            path_targets=[TaskPathTarget(path="src/web/app.js", kind="file")],
            request_focus="Fix the close handler in src/web/app.js",
        )

        plan = build_heuristic_task_plan(context)

        self.assertEqual([step.kind for step in plan], ["read", "patch", "finish"])
        self.assertEqual(plan[1].args["path"], "src/web/app.js")


class StructuredTaskEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_engine_completes_after_successful_step_and_finish(self):
        task = TaskState(
            id="task",
            status="queued",
            context=TaskContext(
                conversation_id="conv",
                user_message="Explain src/web/app.js",
                history=[],
                workspace_intent="focused_read",
            ),
            plan=[
                TaskStep(id="s1", kind="read", title="Read file", args={"path": "src/web/app.js"}),
                TaskStep(id="s2", kind="finish", title="Answer", args={"summary_hint": "done"}),
            ],
        )

        async def execute_step(_task: TaskState, step: TaskStep) -> TaskResult:
            return TaskResult(step_id=step.id, ok=True, output={"path": step.args["path"]})

        final_task = await run_task(task, execute_step)

        self.assertEqual(final_task.status, "done")
        self.assertEqual(len(final_task.results), 1)
        self.assertEqual(final_task.final_text, "done")

    async def test_engine_stops_when_step_blocks(self):
        task = TaskState(
            id="task",
            status="queued",
            context=TaskContext(
                conversation_id="conv",
                user_message="Read src/web/app.js",
                history=[],
                workspace_intent="focused_read",
            ),
            plan=[
                TaskStep(id="s1", kind="read", title="Read file", args={"path": "src/web/app.js"}),
                TaskStep(id="s2", kind="finish", title="Answer"),
            ],
        )

        async def execute_step(_task: TaskState, step: TaskStep) -> TaskResult:
            return TaskResult(
                step_id=step.id,
                ok=False,
                output={"message_to_user": "approval needed"},
                error="denied",
                block_reason="permission_denied",
            )

        final_task = await run_task(task, execute_step)

        self.assertEqual(final_task.status, "blocked")
        self.assertEqual(final_task.final_text, "approval needed")
        self.assertEqual(len(final_task.results), 1)


if __name__ == "__main__":
    unittest.main()
