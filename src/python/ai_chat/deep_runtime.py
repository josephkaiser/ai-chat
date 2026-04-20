"""Deep-runtime state and orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.python.ai_chat.deep_flow import DeepRouteRequest, decide_deep_route


@dataclass
class DeepSession:
    """Shared state for a deep-mode request."""

    websocket: Any
    conversation_id: str
    message: str
    history: List[Dict[str, str]]
    system_prompt: str
    max_tokens: int
    features: Any
    task_request: str = ""
    context: str = ""
    workspace_enabled: bool = False
    workspace_facts: str = ""
    workspace_snapshot: Dict[str, Any] = field(default_factory=dict)
    recent_product_feedback_entries: List[Dict[str, Any]] = field(default_factory=list)
    recent_product_feedback_summary: str = ""
    recent_product_feedback_artifact_path: str = ".ai/recent-feedback.md"
    plan: Dict[str, Any] = field(default_factory=dict)
    plan_preview_pending: bool = False
    task_board_path: str = ".ai/task-board.md"
    task_state_path: str = ".ai/task-state.json"
    build_step_summaries: List[str] = field(default_factory=list)
    step_subplans: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step_substep_summaries: Dict[str, List[str]] = field(default_factory=dict)
    step_substep_reports: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    step_reports: List[Dict[str, Any]] = field(default_factory=list)
    build_summary: str = ""
    changed_files: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    verification_summary: str = ""
    scope_audit: Dict[str, Any] = field(default_factory=dict)
    scope_audit_summary: str = ""
    draft_response: str = ""
    final_response: str = ""
    execution_requested: bool = False
    auto_execute: bool = False
    plan_override_builder_steps: List[str] = field(default_factory=list)
    resumed: bool = False
    pause_reason: str = ""
    workflow_execution: Optional[Any] = None

    def history_messages(self) -> List[Dict[str, str]]:
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.history]


AsyncSessionFn = Callable[[DeepSession], Awaitable[Any]]
AsyncSessionStrFn = Callable[[DeepSession], Awaitable[str]]
AsyncSessionBoolFn = Callable[[DeepSession], Awaitable[bool]]
AsyncSessionPlanFn = Callable[..., Awaitable[Dict[str, Any]]]
AsyncSendPlanFn = Callable[[Any, Dict[str, Any], str], Awaitable[None]]
AsyncActivityFn = Callable[[Any, str, str, str], Awaitable[None]]
SyncSessionBoolFn = Callable[[DeepSession], bool]
SyncClarificationFn = Callable[..., str]
SyncTextFn = Callable[[Dict[str, Any]], str]
SyncPlanPromptFn = Callable[[Dict[str, Any]], str]


@dataclass(frozen=True)
class DeepRuntimeCallbacks:
    """Injected behavior for the deep-runtime lifecycle."""

    send_activity_event: AsyncActivityFn
    deep_confirm_understanding: AsyncSessionStrFn
    maybe_resume_task_state: AsyncSessionBoolFn
    collect_recent_product_feedback_for_session: AsyncSessionStrFn
    apply_deep_session_plan_override: AsyncSessionBoolFn
    deep_inspect_workspace: AsyncSessionStrFn
    should_pause_for_workspace_clarification: SyncClarificationFn
    should_preview_deep_plan: SyncSessionBoolFn
    deep_decompose: AsyncSessionPlanFn
    send_plan_ready: AsyncSendPlanFn
    format_deep_execution_prompt: SyncPlanPromptFn
    render_deep_plan_preview: SyncTextFn
    render_saved_plan_write_access_message: SyncTextFn
    deep_build_workspace: AsyncSessionFn
    deep_parallel_solve: AsyncSessionFn
    deep_verify: AsyncSessionStrFn
    deep_review: AsyncSessionStrFn
    persist_task_state: AsyncSessionFn
    deep_answer_directly: AsyncSessionStrFn
    maybe_refine_deep_response: Callable[[DeepSession, str], Awaitable[str]]


def create_deep_session(
    *,
    websocket: Any,
    conversation_id: str,
    message: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: Any,
    workspace_enabled: bool,
    context_builder: Callable[[List[Dict[str, str]]], str],
    task_request: Optional[str] = None,
    execution_requested: bool = False,
    auto_execute: bool = False,
    plan_override_builder_steps: Optional[List[str]] = None,
    plan_override_normalizer: Optional[Callable[[Optional[List[str]]], List[str]]] = None,
    workflow_execution: Optional[Any] = None,
) -> DeepSession:
    """Create a deep-session object with the shared defaults used across entry points."""
    normalized_override = (
        plan_override_normalizer(plan_override_builder_steps)
        if plan_override_normalizer is not None
        else list(plan_override_builder_steps or [])
    )
    return DeepSession(
        websocket=websocket,
        conversation_id=conversation_id,
        message=message,
        task_request=task_request or message,
        history=history,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        features=features,
        context=context_builder(history),
        workspace_enabled=workspace_enabled,
        execution_requested=execution_requested,
        auto_execute=auto_execute,
        plan_override_builder_steps=normalized_override,
        workflow_execution=workflow_execution,
    )


async def bootstrap_deep_session(
    session: DeepSession,
    callbacks: DeepRuntimeCallbacks,
    *,
    resume_task_state_allowed: bool = True,
) -> bool:
    """Run the shared confirmation/resume/inspect bootstrap for deep sessions."""
    try:
        await callbacks.deep_confirm_understanding(session)
    except Exception:
        pass

    if resume_task_state_allowed:
        try:
            await callbacks.maybe_resume_task_state(session)
        except Exception:
            pass

    try:
        await callbacks.collect_recent_product_feedback_for_session(session)
    except Exception:
        pass

    await callbacks.apply_deep_session_plan_override(session)
    await callbacks.deep_inspect_workspace(session)
    return not (session.pause_reason == "command_approval" and bool(session.draft_response))


async def run_deep_plan_preview_flow(session: DeepSession, callbacks: DeepRuntimeCallbacks) -> str:
    """Build and publish the current deep plan preview."""
    await callbacks.deep_decompose(session, preview_only=True)
    await callbacks.send_plan_ready(
        session.websocket,
        session.plan,
        callbacks.format_deep_execution_prompt(session.plan),
    )
    return callbacks.render_deep_plan_preview(session.plan)


async def run_deep_execution_flow(
    session: DeepSession,
    callbacks: DeepRuntimeCallbacks,
    *,
    requires_existing_plan: bool = False,
    missing_plan_message: str = "I couldn't find a saved plan to execute in this chat. Ask me to generate the plan again first.",
    blocked_write_renderer: Optional[Callable[[DeepSession], str]] = None,
) -> str:
    """Run the shared deep plan/build/verify/review pipeline."""
    decision = decide_deep_route(
        DeepRouteRequest(
            requires_existing_plan=requires_existing_plan,
            has_plan=bool(session.plan),
            should_preview_plan=False,
            execution_requested=session.execution_requested,
            auto_execute=session.auto_execute,
            workspace_write=session.features.workspace_write,
        )
    )
    if decision.action == "missing_plan":
        return missing_plan_message

    await callbacks.deep_decompose(session)

    decision = decide_deep_route(
        DeepRouteRequest(
            requires_existing_plan=False,
            has_plan=bool(session.plan),
            should_preview_plan=False,
            execution_requested=session.execution_requested,
            auto_execute=session.auto_execute,
            workspace_write=session.features.workspace_write,
        )
    )
    if decision.action == "blocked_write":
        session.pause_reason = "write_blocked"
        await callbacks.persist_task_state(session)
        await callbacks.send_plan_ready(
            session.websocket,
            session.plan,
            callbacks.format_deep_execution_prompt(session.plan),
        )
        renderer = blocked_write_renderer or (lambda current_session: callbacks.render_saved_plan_write_access_message(current_session.plan))
        return renderer(session)

    build_result = await callbacks.deep_build_workspace(session)
    if getattr(build_result, "needs_user_confirmation", False):
        session.pause_reason = str(getattr(build_result, "pause_reason", "") or session.pause_reason)
        return str(getattr(build_result, "summary", "") or "")
    await callbacks.deep_parallel_solve(session)
    await callbacks.deep_verify(session)
    if session.pause_reason == "hard_limit":
        return session.verification_summary
    final_review = await callbacks.deep_review(session)
    if session.pause_reason == "hard_limit":
        return session.draft_response
    return final_review


async def orchestrate_deep_session(session: DeepSession, callbacks: DeepRuntimeCallbacks) -> str:
    """Run the deep bootstrap, route decision, execution, and refinement lifecycle."""
    await callbacks.send_activity_event(
        session.websocket,
        "evaluate",
        "Evaluate",
        "Workspace path selected." if session.workspace_enabled else "Text path selected.",
    )

    bootstrap_ready = await bootstrap_deep_session(session, callbacks, resume_task_state_allowed=True)
    if not bootstrap_ready and session.draft_response:
        return session.draft_response

    clarification = callbacks.should_pause_for_workspace_clarification(
        session.task_request or session.message,
        session.workspace_facts,
        session.workspace_snapshot,
        has_plan=bool(session.plan),
        execution_requested=session.execution_requested,
    )
    if clarification:
        session.draft_response = clarification
        await callbacks.persist_task_state(session)
        return clarification

    decision = decide_deep_route(
        DeepRouteRequest(
            requires_existing_plan=session.execution_requested,
            has_plan=bool(session.plan),
            should_preview_plan=callbacks.should_preview_deep_plan(session),
            execution_requested=session.execution_requested,
            auto_execute=session.auto_execute,
            workspace_write=session.features.workspace_write,
        )
    )

    if decision.action == "preview_plan":
        draft_response = await run_deep_plan_preview_flow(session, callbacks)
    elif decision.action in {"execute_plan", "blocked_write", "missing_plan"}:
        draft_response = await run_deep_execution_flow(
            session,
            callbacks,
            requires_existing_plan=session.execution_requested,
        )
    else:
        draft_response = await callbacks.deep_answer_directly(session)

    return await callbacks.maybe_refine_deep_response(session, draft_response)
