import unittest

from src.python.ai_chat.deep_runtime import (
    DeepRuntimeCallbacks,
    create_deep_session,
    orchestrate_deep_session,
)


class DeepRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_orchestrate_session_can_preview_plan(self):
        events = []

        async def send_activity_event(_websocket, phase, label, content):
            events.append((phase, label, content))

        async def deep_confirm_understanding(_session):
            return ""

        async def maybe_resume_task_state(_session):
            return False

        async def collect_recent_product_feedback_for_session(_session):
            return ""

        async def apply_deep_session_plan_override(_session):
            return False

        async def deep_inspect_workspace(session):
            session.workspace_facts = "Observed workspace facts."
            return session.workspace_facts

        async def deep_decompose(session, preview_only=False):
            session.plan = {
                "builder_steps": ["Inspect", "Patch"],
                "deliverable": "A working change",
            }
            return session.plan

        async def send_plan_ready(_websocket, plan, execute_prompt):
            events.append(("plan_ready", plan.get("deliverable", ""), execute_prompt))

        async def deep_build_workspace(_session):
            raise AssertionError("build should not run during preview")

        async def deep_parallel_solve(_session):
            raise AssertionError("parallel solve should not run during preview")

        async def deep_verify(_session):
            raise AssertionError("verify should not run during preview")

        async def deep_review(_session):
            raise AssertionError("review should not run during preview")

        async def persist_task_state(_session):
            return None

        async def deep_answer_directly(_session):
            raise AssertionError("direct answer should not run during preview")

        async def maybe_refine_deep_response(_session, draft_response):
            return draft_response

        callbacks = DeepRuntimeCallbacks(
            send_activity_event=send_activity_event,
            deep_confirm_understanding=deep_confirm_understanding,
            maybe_resume_task_state=maybe_resume_task_state,
            collect_recent_product_feedback_for_session=collect_recent_product_feedback_for_session,
            apply_deep_session_plan_override=apply_deep_session_plan_override,
            deep_inspect_workspace=deep_inspect_workspace,
            should_pause_for_workspace_clarification=lambda *_args, **_kwargs: "",
            should_preview_deep_plan=lambda _session: True,
            deep_decompose=deep_decompose,
            send_plan_ready=send_plan_ready,
            format_deep_execution_prompt=lambda _plan: "Execute approved plan",
            render_deep_plan_preview=lambda _plan: "Preview plan",
            render_saved_plan_write_access_message=lambda _plan: "Blocked",
            deep_build_workspace=deep_build_workspace,
            deep_parallel_solve=deep_parallel_solve,
            deep_verify=deep_verify,
            deep_review=deep_review,
            persist_task_state=persist_task_state,
            deep_answer_directly=deep_answer_directly,
            maybe_refine_deep_response=maybe_refine_deep_response,
        )

        session = create_deep_session(
            websocket=object(),
            conversation_id="conv",
            message="Plan the change",
            history=[{"role": "user", "content": "Plan the change"}],
            system_prompt="System",
            max_tokens=1024,
            features=type("Features", (), {"workspace_write": True})(),
            workspace_enabled=True,
            context_builder=lambda history: f"context:{len(history)}",
            execution_requested=False,
        )

        result = await orchestrate_deep_session(session, callbacks)

        self.assertEqual(result, "Preview plan")
        self.assertEqual(session.context, "context:1")
        self.assertIn(("evaluate", "Evaluate", "Workspace path selected."), events)
        self.assertTrue(any(item[0] == "plan_ready" for item in events))


if __name__ == "__main__":
    unittest.main()
