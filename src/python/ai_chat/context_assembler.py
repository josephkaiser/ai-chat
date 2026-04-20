"""Structured context assembly helpers for deep-runtime phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from src.python.ai_chat.deep_runtime import DeepSession


@dataclass(frozen=True)
class ContextSection:
    """One ranked slice of context for a runtime phase."""

    key: str
    title: str
    content: str
    priority: int = 100
    metadata: Dict[str, str] = field(default_factory=dict)

    def normalized_content(self) -> str:
        return str(self.content or "").strip()

    def is_present(self) -> bool:
        return bool(self.normalized_content())

    def render(self) -> str:
        cleaned = self.normalized_content()
        if not cleaned:
            return ""
        return f"{self.title}:\n{cleaned}"


@dataclass(frozen=True)
class ContextBundle:
    """A ranked set of prompt context sections for one phase."""

    phase: str
    sections: List[ContextSection] = field(default_factory=list)

    def present_sections(self) -> List[ContextSection]:
        ranked = [
            (idx, section)
            for idx, section in enumerate(self.sections)
            if isinstance(section, ContextSection) and section.is_present()
        ]
        ranked.sort(key=lambda item: (item[1].priority, item[0]))
        return [section for _, section in ranked]

    def render(self) -> str:
        blocks = [section.render() for section in self.present_sections()]
        return "\n\n".join(block for block in blocks if block)


def build_recent_context(history: List[Dict[str, str]], limit: int = 6) -> str:
    """Format recent chat history into a compact context block."""
    if not history:
        return ""
    recent = history[-limit:]
    return "\n".join(
        f"{str(message.get('role', '')).strip()}: {str(message.get('content', '')).strip()[:500]}"
        for message in recent
    )


def format_numbered_items(items: List[str], *, empty: str = "(none)") -> str:
    """Render a numbered list for prompt context."""
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return empty
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(cleaned, start=1))


def format_artifact_refs(paths: List[str], *, empty: str = "(none)") -> str:
    """Render workspace artifact references as one-per-line list items."""
    cleaned = [str(path).strip() for path in paths if str(path).strip()]
    if not cleaned:
        return empty
    return "\n".join(f"[[artifact:{path}]]" for path in cleaned)


def _feedback_section(session: DeepSession, *, include_heading: bool = False) -> str:
    summary = str(session.recent_product_feedback_summary or "").strip()
    if not summary:
        return ""
    if include_heading:
        return (
            f"{summary}\n\n"
            f"Feedback digest artifact:\n[[artifact:{session.recent_product_feedback_artifact_path}]]"
        )
    return (
        f"{summary}\n\n"
        f"[[artifact:{session.recent_product_feedback_artifact_path}]]"
    )


def assemble_inspect_context(
    session: DeepSession,
    *,
    workspace_snapshot_formatter: Callable[[Dict[str, object]], str],
) -> ContextBundle:
    """Build the inspect-phase context bundle."""
    return ContextBundle(
        phase="inspect",
        sections=[
            ContextSection("conversation_context", "Conversation context", session.context or "(none)", priority=10),
            ContextSection(
                "recent_feedback",
                "Recent product feedback to treat as failure signals for this pass",
                _feedback_section(session, include_heading=True),
                priority=20,
            ),
            ContextSection(
                "workspace_snapshot",
                "Deterministic workspace snapshot",
                workspace_snapshot_formatter(session.workspace_snapshot),
                priority=30,
            ),
            ContextSection("user_request", "User request", session.task_request or session.message, priority=40),
            ContextSection(
                "inspect_instruction",
                "Inspect goal",
                (
                    "Inspect the workspace and gather the most relevant facts before planning. "
                    "Use tools when needed, then return a concise fact summary."
                ),
                priority=90,
            ),
        ],
    )


def assemble_step_subtask_context(
    session: DeepSession,
    *,
    step: str,
    step_index: int,
    total_steps: int,
    workspace_snapshot_formatter: Callable[[Dict[str, object]], str],
) -> ContextBundle:
    """Build context for nested step decomposition."""
    return ContextBundle(
        phase="step_subplan",
        sections=[
            ContextSection("user_request", "User request", session.task_request or session.message, priority=10),
            ContextSection("workspace_facts", "Observed workspace facts", session.workspace_facts or "(none)", priority=20),
            ContextSection(
                "workspace_snapshot",
                "Workspace snapshot",
                workspace_snapshot_formatter(session.workspace_snapshot),
                priority=30,
            ),
            ContextSection(
                "plan_strategy",
                "Top-level plan strategy",
                str(session.plan.get("strategy", "(none)")).strip() or "(none)",
                priority=40,
            ),
            ContextSection(
                "completed_build_notes",
                "Completed top-level build notes",
                format_numbered_items(session.build_step_summaries),
                priority=50,
            ),
            ContextSection(
                "current_build_step",
                "Current build step",
                f"({step_index + 1}/{total_steps})\n{step}",
                priority=60,
            ),
        ],
    )


def assemble_plan_context(session: DeepSession) -> ContextBundle:
    """Build context for top-level plan generation."""
    return ContextBundle(
        phase="plan",
        sections=[
            ContextSection("conversation_context", "Conversation context", session.context or "(none)", priority=10),
            ContextSection("workspace_facts", "Observed workspace facts", session.workspace_facts or "(none)", priority=20),
            ContextSection("user_query", "User's new query", session.task_request or session.message, priority=30),
        ],
    )


def assemble_direct_answer_context(session: DeepSession) -> ContextBundle:
    """Build context for direct deep answers."""
    return ContextBundle(
        phase="direct_answer",
        sections=[
            ContextSection("conversation_context", "Conversation context", session.context or "(none)", priority=10),
            ContextSection("workspace_facts", "Observed workspace facts", session.workspace_facts or "(none)", priority=20),
            ContextSection("user_request", "User request", session.task_request or session.message, priority=30),
            ContextSection(
                "answer_instruction",
                "Answer mode",
                (
                    "Answer directly unless execution is clearly required. "
                    "Use read-only tools if they materially improve the answer."
                ),
                priority=90,
            ),
        ],
    )


def assemble_build_substep_context(
    session: DeepSession,
    *,
    step: str,
    step_index: int,
    total_steps: int,
    step_subplan: Dict[str, object],
    completed_substeps: List[str],
    current_substep: str,
    substep_index: int,
    substep_count: int,
    workspace_snapshot_formatter: Callable[[Dict[str, object]], str],
    subplan_progress_formatter: Callable[[Dict[str, object], List[str]], str],
) -> ContextBundle:
    """Build context for one executable build substep."""
    return ContextBundle(
        phase="build_substep",
        sections=[
            ContextSection("conversation_context", "Conversation context", session.context or "(none)", priority=10),
            ContextSection("workspace_facts", "Observed workspace facts", session.workspace_facts or "(none)", priority=20),
            ContextSection(
                "workspace_snapshot",
                "Current workspace snapshot",
                workspace_snapshot_formatter(session.workspace_snapshot),
                priority=30,
            ),
            ContextSection("user_request", "User request", session.task_request or session.message, priority=40),
            ContextSection(
                "deliverable",
                "Planned deliverable",
                str(session.plan.get("deliverable", "(none)")).strip() or "(none)",
                priority=50,
            ),
            ContextSection(
                "task_board",
                "Task board artifact",
                f"[[artifact:{session.task_board_path}]]",
                priority=60,
            ),
            ContextSection(
                "completed_build_notes",
                "Completed build step notes",
                format_numbered_items(session.build_step_summaries),
                priority=70,
            ),
            ContextSection(
                "current_build_step",
                "Current build step",
                f"({step_index + 1}/{total_steps})\n{step}",
                priority=80,
            ),
            ContextSection(
                "nested_step_plan",
                "Nested step plan",
                subplan_progress_formatter(step_subplan, completed_substeps),
                priority=90,
            ),
            ContextSection(
                "current_substep",
                "Current substep",
                f"({substep_index + 1}/{substep_count})\n{current_substep}",
                priority=100,
            ),
            ContextSection(
                "build_role",
                "Build role",
                str(session.plan.get("agent_a", {}).get("role", "builder")).strip() or "builder",
                priority=110,
            ),
            ContextSection(
                "build_task",
                "Build task",
                str(session.plan.get("agent_a", {}).get("prompt", "")).strip(),
                priority=120,
            ),
            ContextSection(
                "execution_instruction",
                "Execution focus",
                (
                    "Focus on the current substep only. Treat any later substeps as future work inside this same "
                    "top-level build step. Use the task board as external memory instead of trying to carry the "
                    "whole plan in-context. On each loop, either gather missing evidence, update a durable artifact, "
                    "or verify the current substep. Use tools to inspect, patch, and validate as needed for this "
                    "substep. When done, return a concise substep result covering what changed, any artifact paths, "
                    "and caveats that matter for later verification."
                ),
                priority=200,
            ),
        ],
    )


def assemble_verify_context(session: DeepSession) -> ContextBundle:
    """Build context for the verification phase."""
    return ContextBundle(
        phase="verify",
        sections=[
            ContextSection("user_request", "User request", session.task_request or session.message, priority=10),
            ContextSection("workspace_facts", "Observed workspace facts", session.workspace_facts or "(none)", priority=20),
            ContextSection(
                "recent_feedback",
                "Recent product feedback to verify against",
                _feedback_section(session),
                priority=30,
            ),
            ContextSection(
                "task_board",
                "Task board artifact",
                f"[[artifact:{session.task_board_path}]]",
                priority=40,
            ),
            ContextSection(
                "deliverable",
                "Planned deliverable",
                str(session.plan.get("deliverable", "(none)")).strip() or "(none)",
                priority=50,
            ),
            ContextSection(
                "verifier_checks",
                "Verifier checks",
                format_numbered_items(list(session.plan.get("verifier_checks", []))),
                priority=60,
            ),
            ContextSection("build_summary", "Build summary", session.build_summary or "(none)", priority=70),
            ContextSection(
                "changed_files",
                "Files changed",
                ", ".join(path for path in session.changed_files if path) or "(none)",
                priority=80,
            ),
            ContextSection(
                "implementation_pass",
                f"Proposed solution A ({session.agent_outputs.get('agent_a_role', 'builder')})",
                session.agent_outputs.get("output_a", ""),
                priority=90,
            ),
            ContextSection(
                "review_pass",
                f"Proposed solution B ({session.agent_outputs.get('agent_b_role', 'verifier')})",
                session.agent_outputs.get("output_b", ""),
                priority=100,
            ),
            ContextSection(
                "verification_instruction",
                "Verification focus",
                (
                    "Use read-only tools and commands to verify likely assumptions. "
                    "Return a concise verification summary."
                ),
                priority=200,
            ),
        ],
    )


def assemble_synthesis_context(session: DeepSession) -> ContextBundle:
    """Build context for the final synthesis phase."""
    artifact_refs = [session.task_board_path]
    if session.recent_product_feedback_summary:
        artifact_refs.append(session.recent_product_feedback_artifact_path)
    artifact_refs.extend(session.changed_files[:8])

    return ContextBundle(
        phase="synthesize",
        sections=[
            ContextSection("user_question", "User's question", session.task_request or session.message, priority=10),
            ContextSection("workspace_facts", "Observed workspace facts", session.workspace_facts or "(none)", priority=20),
            ContextSection(
                "recent_feedback",
                "Recent product feedback",
                _feedback_section(session),
                priority=30,
            ),
            ContextSection(
                "deliverable",
                "Planned deliverable",
                str(session.plan.get("deliverable", "(none)")).strip() or "(none)",
                priority=40,
            ),
            ContextSection(
                "artifacts",
                "Artifacts to inspect",
                format_artifact_refs(artifact_refs),
                priority=50,
            ),
            ContextSection("build_summary", "Build summary", session.build_summary or "(none)", priority=60),
            ContextSection(
                "verification_summary",
                "Verification summary",
                session.verification_summary or "(none)",
                priority=70,
            ),
            ContextSection("scope_audit", "Scope audit", session.scope_audit_summary or "(none)", priority=80),
            ContextSection(
                "implementation_pass",
                f"Implementation pass ({session.agent_outputs.get('agent_a_role', 'builder')})",
                session.agent_outputs.get("output_a", ""),
                priority=90,
            ),
            ContextSection(
                "review_pass",
                f"Review pass ({session.agent_outputs.get('agent_b_role', 'verifier')})",
                session.agent_outputs.get("output_b", ""),
                priority=100,
            ),
            ContextSection(
                "synthesis_instruction",
                "Synthesis focus",
                (
                    "Read the task board and the most relevant built artifacts before answering. "
                    "Base the final answer on what the workspace actually contains, what verification established, "
                    "and which scope gaps remain."
                ),
                priority=200,
            ),
        ],
    )
