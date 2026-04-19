"""Generic task executor for structured orchestration."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from src.python.ai_chat.task_types import TaskResult, TaskState, TaskStep

TaskEventHandler = Callable[[str, TaskState, Optional[TaskStep], Optional[TaskResult]], Awaitable[None]]
TaskStepExecutor = Callable[[TaskState, TaskStep], Awaitable[TaskResult]]


async def run_task(
    task: TaskState,
    execute_step: TaskStepExecutor,
    emit_event: Optional[TaskEventHandler] = None,
) -> TaskState:
    """Run a typed task plan until completion, failure, or a block."""
    task.status = "planning"
    if emit_event is not None:
        await emit_event("plan_ready", task, None, None)

    task.status = "executing"
    if emit_event is not None:
        await emit_event("task_started", task, None, None)

    for step in task.plan:
        task.current_step_id = step.id

        if step.kind == "finish":
            task.final_text = str(step.args.get("summary_hint", "")).strip()
            task.status = "done"
            if emit_event is not None:
                await emit_event("task_done", task, step, None)
            return task

        if emit_event is not None:
            await emit_event("step_started", task, step, None)

        result = await execute_step(task, step)
        task.results.append(result)

        if emit_event is not None:
            await emit_event("step_finished", task, step, result)

        if result.block_reason:
            task.status = "blocked"
            task.block_reason = result.block_reason
            task.final_text = str(result.output.get("message_to_user", "") or result.error).strip()
            if emit_event is not None:
                await emit_event("task_blocked", task, step, result)
            return task

        if not result.ok:
            task.status = "failed"
            task.final_text = str(result.error or "Structured task execution failed.").strip()
            if emit_event is not None:
                await emit_event("task_failed", task, step, result)
            return task

    task.status = "done"
    if emit_event is not None:
        await emit_event("task_done", task, None, None)
    return task
