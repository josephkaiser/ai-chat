"""Typed task objects for the structured orchestration path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

TaskStatus = Literal["queued", "planning", "executing", "blocked", "done", "failed"]
TaskStepKind = Literal["list", "search", "read", "patch", "run", "finish"]


@dataclass(frozen=True)
class TaskPathTarget:
    """One resolved workspace target surfaced from the user's request."""

    path: str
    kind: Literal["file", "dir"]


@dataclass
class TaskContext:
    """Stable inputs for one structured task run."""

    conversation_id: str
    user_message: str
    history: List[Dict[str, str]]
    workspace_intent: str
    allowed_tools: List[str] = field(default_factory=list)
    path_targets: List[TaskPathTarget] = field(default_factory=list)
    request_focus: str = ""


@dataclass
class TaskStep:
    """A single deterministic execution step."""

    id: str
    kind: TaskStepKind
    title: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Normalized result for one executed step."""

    step_id: str
    ok: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    block_reason: str = ""


@dataclass
class TaskState:
    """Runtime state for a structured task."""

    id: str
    status: TaskStatus
    context: TaskContext
    plan: List[TaskStep] = field(default_factory=list)
    results: List[TaskResult] = field(default_factory=list)
    current_step_id: Optional[str] = None
    block_reason: str = ""
    final_text: str = ""
