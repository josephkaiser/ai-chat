"""Structured context assembly helpers for deep-runtime phases."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

from src.python.ai_chat.context_selection_program import (
    DEFAULT_CONTEXT_SELECTION_PROGRAM,
    ContextCandidate,
    ContextSelectionInputs,
)
from src.python.ai_chat.context_policy_program import (
    DEFAULT_CONTEXT_POLICY_PROGRAM,
    ContextPolicyInputs,
)

from src.python.ai_chat.deep_runtime import DeepSession


@dataclass(frozen=True)
class ContextSection:
    """One ranked slice of context for a runtime phase."""

    key: str
    title: str
    content: str
    priority: int = 100
    required: bool = False
    tags: Sequence[str] = field(default_factory=tuple)
    phase_hints: Sequence[str] = field(default_factory=tuple)
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
    candidate_count: int = 0
    omitted_keys: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def present_sections(self) -> List[ContextSection]:
        ranked = [
            (idx, section)
            for idx, section in enumerate(self.sections)
            if isinstance(section, ContextSection) and section.is_present()
        ]
        ranked.sort(key=lambda item: (
            int(item[1].metadata.get("selection_rank", "9999")),
            item[1].priority,
            item[0],
        ))
        return [section for _, section in ranked]

    def render(self) -> str:
        blocks = [section.render() for section in self.present_sections()]
        return "\n\n".join(block for block in blocks if block)


@dataclass(frozen=True)
class ContextRetrievalAdapters:
    """Runtime adapters for retrieval-backed context candidates."""

    conversation_search: Callable[[str, str, int], Dict[str, object]]
    read_workspace_text: Callable[[str, str], Optional[str]]
    truncate_output: Callable[[str, int], str]


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


def format_changed_files(paths: List[str], *, empty: str = "(none)") -> str:
    """Render changed files as a compact list."""
    cleaned = [str(path).strip() for path in paths if str(path).strip()]
    if not cleaned:
        return empty
    return "\n".join(f"- {path}" for path in cleaned[:8])


def format_step_report_digest(step_reports: List[Dict[str, object]], *, limit: int = 2, empty: str = "(none)") -> str:
    """Render recent build evidence as compact step digests."""
    cleaned_reports = [report for report in step_reports if isinstance(report, dict)]
    if not cleaned_reports:
        return empty
    lines: List[str] = []
    for report in cleaned_reports[-limit:]:
        step_index = int(report.get("step_index", 0)) + 1
        step = str(report.get("step", "")).strip() or f"Step {step_index}"
        summary = str(report.get("summary", "")).strip() or "(no summary)"
        paths = [str(path).strip() for path in report.get("paths", []) if str(path).strip()]
        path_suffix = f" Paths: {', '.join(paths[:4])}." if paths else ""
        lines.append(f"{step_index}. {step}: {summary}{path_suffix}")
    return "\n".join(lines) if lines else empty


def format_scope_audit_gaps(scope_audit: Dict[str, object], *, empty: str = "(none)") -> str:
    """Render current scope-audit gaps for verification and synthesis."""
    gaps = scope_audit.get("gaps", []) if isinstance(scope_audit, dict) else []
    cleaned = [str(item).strip() for item in gaps if str(item).strip()]
    if not cleaned:
        return empty
    return "\n".join(f"- {item}" for item in cleaned[:6])


def format_retrieved_history_matches(
    payload: Dict[str, object],
    *,
    truncate_output_fn: Callable[[str, int], str],
    empty: str = "(none)",
) -> str:
    """Render retrieved chat-memory matches into a compact evidence block."""
    matches = payload.get("matches", []) if isinstance(payload, dict) else []
    if not isinstance(matches, list) or not matches:
        return empty
    lines: List[str] = []
    for index, match in enumerate(matches[:3], start=1):
        if not isinstance(match, dict):
            continue
        role = str(match.get("role", "")).strip() or "message"
        snippet = str(match.get("snippet") or match.get("content") or "").strip()
        snippet = truncate_output_fn(snippet, 220) if snippet else "(no snippet)"
        context_before = str(match.get("context_before", "")).strip()
        context_after = str(match.get("context_after", "")).strip()
        lines.append(f"{index}. {role}: {snippet}")
        if context_before:
            lines.append(f"   before: {truncate_output_fn(context_before, 140)}")
        if context_after:
            lines.append(f"   after: {truncate_output_fn(context_after, 140)}")
    return "\n".join(lines) if lines else empty


def format_workspace_excerpt_blocks(
    previews: List[tuple[str, str]],
    *,
    truncate_output_fn: Callable[[str, int], str],
    empty: str = "(none)",
) -> str:
    """Render a few workspace file previews into one bounded block."""
    if not previews:
        return empty
    blocks: List[str] = []
    for path, content in previews[:3]:
        cleaned_path = str(path or "").strip()
        cleaned_content = str(content or "").strip()
        if not cleaned_path or not cleaned_content:
            continue
        blocks.append(f"[{cleaned_path}]\n{truncate_output_fn(cleaned_content, 700)}")
    return "\n\n".join(blocks) if blocks else empty


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


def _merge_dynamic_sections(
    bundle: ContextBundle,
    dynamic_sections: List[ContextSection],
    *,
    request_text: str,
    max_sections: int,
) -> ContextBundle:
    return _select_context_bundle(
        bundle.phase,
        request_text,
        list(bundle.sections) + [section for section in dynamic_sections if section.is_present()],
        max_sections=max_sections,
    )


def _phase_dynamic_paths(session: DeepSession, preferred_paths: Sequence[str]) -> List[str]:
    paths: List[str] = [str(path).strip() for path in preferred_paths if str(path).strip()]
    deduped: List[str] = []
    for path in paths:
        cleaned = str(path or "").strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


async def _read_workspace_preview(
    adapters: ContextRetrievalAdapters,
    conversation_id: str,
    rel_path: str,
) -> Optional[str]:
    try:
        return await asyncio.to_thread(adapters.read_workspace_text, conversation_id, rel_path)
    except Exception:
        return None


async def build_dynamic_context_sections(
    session: DeepSession,
    *,
    phase: str,
    request_text: str,
    adapters: ContextRetrievalAdapters,
) -> List[ContextSection]:
    """Retrieve dynamic context candidates from chat memory and workspace artifacts."""
    sections: List[ContextSection] = []
    cleaned_request = str(request_text or "").strip()
    sample_paths = session.workspace_snapshot.get("sample_paths", []) if isinstance(session.workspace_snapshot, dict) else []
    policy = DEFAULT_CONTEXT_POLICY_PROGRAM.run(
        ContextPolicyInputs(
            phase=phase,
            request_text=cleaned_request,
            changed_files=tuple(session.changed_files),
            workspace_sample_paths=tuple(
                str(path).strip() for path in sample_paths if str(path).strip()
            ) if isinstance(sample_paths, list) else tuple(),
            has_task_board=bool(session.task_board_path),
            has_recent_feedback=bool(session.recent_product_feedback_summary),
            has_scope_gaps=bool(format_scope_audit_gaps(session.scope_audit).strip() and format_scope_audit_gaps(session.scope_audit).strip() != "(none)"),
            has_step_reports=bool(session.step_reports),
        )
    )

    if policy.retrieve_memory and len(cleaned_request) >= 2:
        try:
            memory_payload = await asyncio.to_thread(
                adapters.conversation_search,
                session.conversation_id,
                cleaned_request,
                max(1, policy.memory_limit),
            )
        except Exception:
            memory_payload = {}
        memory_text = format_retrieved_history_matches(
            memory_payload if isinstance(memory_payload, dict) else {},
            truncate_output_fn=adapters.truncate_output,
        )
        if memory_text and memory_text != "(none)":
            sections.append(
                ContextSection(
                    "retrieved_memory",
                    "Retrieved chat memory",
                    memory_text,
                    priority=45,
                    tags=("history", "retrieval", "conversation"),
                    phase_hints=(phase,),
                    metadata={
                        "sticky": "true",
                        "policy_reason": policy.reasons.get("retrieve_memory", ""),
                    },
                )
            )

    if policy.retrieve_workspace_previews:
        preview_paths = _phase_dynamic_paths(session, policy.preferred_paths)
        preview_results = await asyncio.gather(
            *[
                _read_workspace_preview(adapters, session.conversation_id, path)
                for path in preview_paths[:max(1, policy.workspace_preview_limit)]
            ],
            return_exceptions=True,
        )
        previews: List[tuple[str, str]] = []
        for path, preview in zip(preview_paths[:max(1, policy.workspace_preview_limit)], preview_results):
            if isinstance(preview, Exception):
                continue
            cleaned_preview = str(preview or "").strip()
            if cleaned_preview:
                previews.append((path, cleaned_preview))

        preview_text = format_workspace_excerpt_blocks(
            previews,
            truncate_output_fn=adapters.truncate_output,
        )
        if preview_text and preview_text != "(none)":
            sections.append(
                ContextSection(
                    "workspace_excerpts",
                    "Relevant workspace excerpts",
                    preview_text,
                    priority=55,
                    required=phase in {"plan", "verify", "synthesize"},
                    tags=("workspace", "files", "artifacts"),
                    phase_hints=(phase,),
                    metadata={
                        "sticky": "true",
                        "policy_reason": policy.reasons.get("retrieve_workspace_previews", ""),
                    },
                )
            )
    return sections


def _select_context_bundle(
    phase: str,
    request_text: str,
    sections: List[ContextSection],
    *,
    max_sections: int,
) -> ContextBundle:
    candidates = [
        ContextCandidate(
            key=section.key,
            title=section.title,
            content=section.normalized_content(),
            priority=section.priority,
            required=section.required,
            tags=tuple(section.tags),
            phase_hints=tuple(section.phase_hints),
            metadata=dict(section.metadata),
        )
        for section in sections
        if isinstance(section, ContextSection) and section.is_present()
    ]
    decision = DEFAULT_CONTEXT_SELECTION_PROGRAM.run(
        ContextSelectionInputs(
            phase=phase,
            request_text=request_text,
            candidates=candidates,
            max_sections=max_sections,
        )
    )
    by_key = {section.key: section for section in sections if section.is_present()}
    selected_sections: List[ContextSection] = []
    for rank, key in enumerate(decision.selected_keys):
        section = by_key.get(key)
        if section is None:
            continue
        metadata = dict(section.metadata)
        metadata["selection_rank"] = str(rank)
        metadata["selection_reason"] = decision.reasons.get(key, "")
        metadata["selection_score"] = str(decision.scores.get(key, 0.0))
        selected_sections.append(replace(section, metadata=metadata))
    return ContextBundle(
        phase=phase,
        sections=selected_sections,
        candidate_count=len(candidates),
        omitted_keys=list(decision.omitted_keys),
        scores=dict(decision.scores),
    )


def assemble_inspect_context(
    session: DeepSession,
    *,
    workspace_snapshot_formatter: Callable[[Dict[str, object]], str],
) -> ContextBundle:
    """Build the inspect-phase context bundle."""
    return _select_context_bundle(
        "inspect",
        session.task_request or session.message,
        [
            ContextSection(
                "user_request",
                "User request",
                session.task_request or session.message,
                priority=5,
                required=True,
                tags=("request",),
                phase_hints=("inspect",),
            ),
            ContextSection(
                "recent_turns",
                "Recent conversation turns",
                build_recent_context(session.history, limit=4),
                priority=10,
                tags=("history", "conversation"),
                phase_hints=("inspect", "plan", "direct_answer"),
                metadata={"recency_rank": "0"},
            ),
            ContextSection(
                "conversation_context",
                "Conversation context",
                session.context or "(none)",
                priority=15,
                tags=("conversation", "history"),
                phase_hints=("inspect", "plan"),
            ),
            ContextSection(
                "recent_feedback",
                "Recent product feedback to treat as failure signals for this pass",
                _feedback_section(session, include_heading=True),
                priority=20,
                tags=("feedback", "artifacts"),
                phase_hints=("inspect", "verify", "synthesize"),
            ),
            ContextSection(
                "workspace_snapshot",
                "Deterministic workspace snapshot",
                workspace_snapshot_formatter(session.workspace_snapshot),
                priority=30,
                required=True,
                tags=("workspace", "snapshot", "files"),
                phase_hints=("inspect",),
            ),
            ContextSection(
                "workspace_facts",
                "Current workspace facts",
                session.workspace_facts or "(none)",
                priority=40,
                tags=("workspace", "facts"),
                phase_hints=("inspect", "plan"),
            ),
            ContextSection(
                "inspect_instruction",
                "Inspect goal",
                (
                    "Inspect the workspace and gather the most relevant facts before planning. "
                    "Use tools when needed, then return a concise fact summary."
                ),
                priority=90,
                required=True,
                tags=("instruction",),
                phase_hints=("inspect",),
            ),
        ],
        max_sections=6,
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
    return _select_context_bundle(
        "step_subplan",
        session.task_request or session.message,
        [
            ContextSection(
                "user_request",
                "User request",
                session.task_request or session.message,
                priority=10,
                required=True,
                tags=("request",),
                phase_hints=("step_subplan",),
            ),
            ContextSection(
                "workspace_facts",
                "Observed workspace facts",
                session.workspace_facts or "(none)",
                priority=20,
                required=True,
                tags=("workspace", "facts"),
                phase_hints=("step_subplan",),
            ),
            ContextSection(
                "workspace_snapshot",
                "Workspace snapshot",
                workspace_snapshot_formatter(session.workspace_snapshot),
                priority=30,
                tags=("workspace", "snapshot", "files"),
                phase_hints=("step_subplan",),
            ),
            ContextSection(
                "plan_strategy",
                "Top-level plan strategy",
                str(session.plan.get("strategy", "(none)")).strip() or "(none)",
                priority=40,
                tags=("plan", "strategy"),
                phase_hints=("step_subplan",),
            ),
            ContextSection(
                "completed_build_notes",
                "Completed top-level build notes",
                format_numbered_items(session.build_step_summaries),
                priority=50,
                tags=("plan", "progress"),
                phase_hints=("step_subplan",),
            ),
            ContextSection(
                "current_build_step",
                "Current build step",
                f"({step_index + 1}/{total_steps})\n{step}",
                priority=60,
                required=True,
                tags=("plan", "step"),
                phase_hints=("step_subplan",),
            ),
        ],
        max_sections=6,
    )


def assemble_plan_context(session: DeepSession) -> ContextBundle:
    """Build context for top-level plan generation."""
    return _select_context_bundle(
        "plan",
        session.task_request or session.message,
        [
            ContextSection(
                "user_query",
                "User's new query",
                session.task_request or session.message,
                priority=10,
                required=True,
                tags=("request",),
                phase_hints=("plan",),
            ),
            ContextSection(
                "recent_turns",
                "Recent conversation turns",
                build_recent_context(session.history, limit=4),
                priority=15,
                tags=("history", "conversation"),
                phase_hints=("plan",),
                metadata={"recency_rank": "0"},
            ),
            ContextSection(
                "conversation_context",
                "Conversation context",
                session.context or "(none)",
                priority=20,
                tags=("conversation", "history"),
                phase_hints=("plan",),
            ),
            ContextSection(
                "workspace_facts",
                "Observed workspace facts",
                session.workspace_facts or "(none)",
                priority=30,
                required=True,
                tags=("workspace", "facts"),
                phase_hints=("plan",),
            ),
            ContextSection(
                "recent_feedback",
                "Recent product feedback",
                _feedback_section(session),
                priority=40,
                tags=("feedback", "artifacts"),
                phase_hints=("plan", "verify", "synthesize"),
            ),
            ContextSection(
                "existing_progress",
                "Existing build progress",
                format_step_report_digest(session.step_reports, limit=2, empty=""),
                priority=50,
                tags=("progress", "workspace"),
                phase_hints=("plan",),
            ),
        ],
        max_sections=6,
    )


def assemble_direct_answer_context(session: DeepSession) -> ContextBundle:
    """Build context for direct deep answers."""
    return _select_context_bundle(
        "direct_answer",
        session.task_request or session.message,
        [
            ContextSection(
                "user_request",
                "User request",
                session.task_request or session.message,
                priority=10,
                required=True,
                tags=("request",),
                phase_hints=("direct_answer",),
            ),
            ContextSection(
                "recent_turns",
                "Recent conversation turns",
                build_recent_context(session.history, limit=4),
                priority=15,
                tags=("history", "conversation"),
                phase_hints=("direct_answer",),
                metadata={"recency_rank": "0"},
            ),
            ContextSection(
                "conversation_context",
                "Conversation context",
                session.context or "(none)",
                priority=20,
                tags=("conversation", "history"),
                phase_hints=("direct_answer",),
            ),
            ContextSection(
                "workspace_facts",
                "Observed workspace facts",
                session.workspace_facts or "(none)",
                priority=30,
                tags=("workspace", "facts"),
                phase_hints=("direct_answer",),
            ),
            ContextSection(
                "changed_files",
                "Files touched so far",
                format_changed_files(session.changed_files, empty=""),
                priority=40,
                tags=("files", "workspace"),
                phase_hints=("direct_answer", "verify", "synthesize"),
            ),
            ContextSection(
                "answer_instruction",
                "Answer mode",
                (
                    "Answer directly unless execution is clearly required. "
                    "Use read-only tools if they materially improve the answer."
                ),
                priority=90,
                required=True,
                tags=("instruction",),
                phase_hints=("direct_answer",),
            ),
        ],
        max_sections=5,
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
    return _select_context_bundle(
        "build_substep",
        current_substep or session.task_request or session.message,
        [
            ContextSection(
                "user_request",
                "User request",
                session.task_request or session.message,
                priority=10,
                required=True,
                tags=("request",),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "conversation_context",
                "Conversation context",
                session.context or "(none)",
                priority=15,
                tags=("conversation", "history"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "workspace_facts",
                "Observed workspace facts",
                session.workspace_facts or "(none)",
                priority=20,
                required=True,
                tags=("workspace", "facts"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "workspace_snapshot",
                "Current workspace snapshot",
                workspace_snapshot_formatter(session.workspace_snapshot),
                priority=30,
                tags=("workspace", "snapshot", "files"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "deliverable",
                "Planned deliverable",
                str(session.plan.get("deliverable", "(none)")).strip() or "(none)",
                priority=50,
                tags=("plan", "deliverable"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "task_board",
                "Task board artifact",
                f"[[artifact:{session.task_board_path}]]",
                priority=60,
                tags=("artifacts", "workspace"),
                phase_hints=("build_substep", "verify", "synthesize"),
            ),
            ContextSection(
                "completed_build_notes",
                "Completed build step notes",
                format_numbered_items(session.build_step_summaries),
                priority=70,
                tags=("progress", "plan"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "current_build_step",
                "Current build step",
                f"({step_index + 1}/{total_steps})\n{step}",
                priority=80,
                required=True,
                tags=("plan", "step"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "nested_step_plan",
                "Nested step plan",
                subplan_progress_formatter(step_subplan, completed_substeps),
                priority=90,
                required=True,
                tags=("plan", "substeps"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "current_substep",
                "Current substep",
                f"({substep_index + 1}/{substep_count})\n{current_substep}",
                priority=100,
                required=True,
                tags=("plan", "substep"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "build_role",
                "Build role",
                str(session.plan.get("agent_a", {}).get("role", "builder")).strip() or "builder",
                priority=110,
                tags=("plan", "role"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "build_task",
                "Build task",
                str(session.plan.get("agent_a", {}).get("prompt", "")).strip(),
                priority=120,
                tags=("plan", "task"),
                phase_hints=("build_substep",),
            ),
            ContextSection(
                "recent_step_evidence",
                "Recent execution evidence",
                format_step_report_digest(session.step_reports, limit=2, empty=""),
                priority=130,
                tags=("progress", "workspace", "files"),
                phase_hints=("build_substep", "verify"),
            ),
            ContextSection(
                "changed_files",
                "Changed files so far",
                format_changed_files(session.changed_files, empty=""),
                priority=140,
                tags=("files", "workspace"),
                phase_hints=("build_substep", "verify", "synthesize"),
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
                required=True,
                tags=("instruction",),
                phase_hints=("build_substep",),
            ),
        ],
        max_sections=10,
    )


def assemble_verify_context(session: DeepSession) -> ContextBundle:
    """Build context for the verification phase."""
    return _select_context_bundle(
        "verify",
        session.task_request or session.message,
        [
            ContextSection(
                "user_request",
                "User request",
                session.task_request or session.message,
                priority=10,
                required=True,
                tags=("request",),
                phase_hints=("verify",),
            ),
            ContextSection(
                "workspace_facts",
                "Observed workspace facts",
                session.workspace_facts or "(none)",
                priority=20,
                required=True,
                tags=("workspace", "facts"),
                phase_hints=("verify",),
            ),
            ContextSection(
                "recent_feedback",
                "Recent product feedback to verify against",
                _feedback_section(session),
                priority=30,
                tags=("feedback", "artifacts"),
                phase_hints=("verify",),
            ),
            ContextSection(
                "task_board",
                "Task board artifact",
                f"[[artifact:{session.task_board_path}]]",
                priority=40,
                tags=("artifacts", "workspace"),
                phase_hints=("verify",),
            ),
            ContextSection(
                "deliverable",
                "Planned deliverable",
                str(session.plan.get("deliverable", "(none)")).strip() or "(none)",
                priority=50,
                tags=("plan", "deliverable"),
                phase_hints=("verify",),
            ),
            ContextSection(
                "verifier_checks",
                "Verifier checks",
                format_numbered_items(list(session.plan.get("verifier_checks", []))),
                priority=60,
                required=True,
                tags=("verification", "plan"),
                phase_hints=("verify",),
            ),
            ContextSection(
                "build_summary",
                "Build summary",
                session.build_summary or "(none)",
                priority=70,
                tags=("progress", "workspace"),
                phase_hints=("verify",),
            ),
            ContextSection(
                "changed_files",
                "Files changed",
                format_changed_files(session.changed_files, empty=""),
                priority=80,
                tags=("files", "workspace"),
                phase_hints=("verify", "synthesize"),
            ),
            ContextSection(
                "step_evidence",
                "Recent execution evidence",
                format_step_report_digest(session.step_reports, limit=3, empty=""),
                priority=85,
                tags=("progress", "verification", "workspace"),
                phase_hints=("verify",),
            ),
            ContextSection(
                "implementation_pass",
                f"Proposed solution A ({session.agent_outputs.get('agent_a_role', 'builder')})",
                session.agent_outputs.get("output_a", ""),
                priority=90,
                tags=("implementation",),
                phase_hints=("verify", "synthesize"),
            ),
            ContextSection(
                "review_pass",
                f"Proposed solution B ({session.agent_outputs.get('agent_b_role', 'verifier')})",
                session.agent_outputs.get("output_b", ""),
                priority=100,
                tags=("verification",),
                phase_hints=("verify", "synthesize"),
            ),
            ContextSection(
                "scope_gaps",
                "Known scope gaps",
                format_scope_audit_gaps(session.scope_audit, empty=""),
                priority=110,
                tags=("audit", "verification"),
                phase_hints=("verify", "synthesize"),
            ),
            ContextSection(
                "verification_instruction",
                "Verification focus",
                (
                    "Use read-only tools and commands to verify likely assumptions. "
                    "Return a concise verification summary."
                ),
                priority=200,
                required=True,
                tags=("instruction", "verification"),
                phase_hints=("verify",),
            ),
        ],
        max_sections=9,
    )


def assemble_synthesis_context(session: DeepSession) -> ContextBundle:
    """Build context for the final synthesis phase."""
    artifact_refs = [session.task_board_path]
    if session.recent_product_feedback_summary:
        artifact_refs.append(session.recent_product_feedback_artifact_path)
    artifact_refs.extend(session.changed_files[:8])

    return _select_context_bundle(
        "synthesize",
        session.task_request or session.message,
        [
            ContextSection(
                "user_question",
                "User's question",
                session.task_request or session.message,
                priority=10,
                required=True,
                tags=("request",),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "workspace_facts",
                "Observed workspace facts",
                session.workspace_facts or "(none)",
                priority=20,
                required=True,
                tags=("workspace", "facts"),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "recent_feedback",
                "Recent product feedback",
                _feedback_section(session),
                priority=30,
                tags=("feedback", "artifacts"),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "deliverable",
                "Planned deliverable",
                str(session.plan.get("deliverable", "(none)")).strip() or "(none)",
                priority=40,
                tags=("plan", "deliverable"),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "artifacts",
                "Artifacts to inspect",
                format_artifact_refs(artifact_refs),
                priority=50,
                required=True,
                tags=("artifacts", "files"),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "build_summary",
                "Build summary",
                session.build_summary or "(none)",
                priority=60,
                tags=("progress",),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "verification_summary",
                "Verification summary",
                session.verification_summary or "(none)",
                priority=70,
                tags=("verification",),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "scope_audit",
                "Scope audit",
                session.scope_audit_summary or "(none)",
                priority=80,
                tags=("audit", "verification"),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "scope_gaps",
                "Scope gaps still open",
                format_scope_audit_gaps(session.scope_audit, empty=""),
                priority=85,
                tags=("audit", "verification"),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "changed_files",
                "Changed files",
                format_changed_files(session.changed_files, empty=""),
                priority=90,
                tags=("files", "workspace"),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "implementation_pass",
                f"Implementation pass ({session.agent_outputs.get('agent_a_role', 'builder')})",
                session.agent_outputs.get("output_a", ""),
                priority=100,
                tags=("implementation",),
                phase_hints=("synthesize",),
            ),
            ContextSection(
                "review_pass",
                f"Review pass ({session.agent_outputs.get('agent_b_role', 'verifier')})",
                session.agent_outputs.get("output_b", ""),
                priority=110,
                tags=("verification",),
                phase_hints=("synthesize",),
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
                required=True,
                tags=("instruction",),
                phase_hints=("synthesize",),
            ),
        ],
        max_sections=9,
    )


async def assemble_inspect_context_async(
    session: DeepSession,
    *,
    workspace_snapshot_formatter: Callable[[Dict[str, object]], str],
    retrieval_adapters: ContextRetrievalAdapters,
) -> ContextBundle:
    """Build inspect context with retrieval-backed dynamic candidates."""
    base = assemble_inspect_context(
        session,
        workspace_snapshot_formatter=workspace_snapshot_formatter,
    )
    dynamic_sections = await build_dynamic_context_sections(
        session,
        phase="inspect",
        request_text=session.task_request or session.message,
        adapters=retrieval_adapters,
    )
    return _merge_dynamic_sections(
        base,
        dynamic_sections,
        request_text=session.task_request or session.message,
        max_sections=6,
    )


async def assemble_step_subtask_context_async(
    session: DeepSession,
    *,
    step: str,
    step_index: int,
    total_steps: int,
    workspace_snapshot_formatter: Callable[[Dict[str, object]], str],
    retrieval_adapters: ContextRetrievalAdapters,
) -> ContextBundle:
    """Build nested step-planning context with retrieval-backed candidates."""
    base = assemble_step_subtask_context(
        session,
        step=step,
        step_index=step_index,
        total_steps=total_steps,
        workspace_snapshot_formatter=workspace_snapshot_formatter,
    )
    dynamic_sections = await build_dynamic_context_sections(
        session,
        phase="step_subplan",
        request_text=step or session.task_request or session.message,
        adapters=retrieval_adapters,
    )
    return _merge_dynamic_sections(
        base,
        dynamic_sections,
        request_text=step or session.task_request or session.message,
        max_sections=6,
    )


async def assemble_plan_context_async(
    session: DeepSession,
    *,
    retrieval_adapters: ContextRetrievalAdapters,
) -> ContextBundle:
    """Build top-level plan context with retrieval-backed candidates."""
    base = assemble_plan_context(session)
    dynamic_sections = await build_dynamic_context_sections(
        session,
        phase="plan",
        request_text=session.task_request or session.message,
        adapters=retrieval_adapters,
    )
    return _merge_dynamic_sections(
        base,
        dynamic_sections,
        request_text=session.task_request or session.message,
        max_sections=6,
    )


async def assemble_direct_answer_context_async(
    session: DeepSession,
    *,
    retrieval_adapters: ContextRetrievalAdapters,
) -> ContextBundle:
    """Build direct-answer context with retrieval-backed candidates."""
    base = assemble_direct_answer_context(session)
    dynamic_sections = await build_dynamic_context_sections(
        session,
        phase="direct_answer",
        request_text=session.task_request or session.message,
        adapters=retrieval_adapters,
    )
    return _merge_dynamic_sections(
        base,
        dynamic_sections,
        request_text=session.task_request or session.message,
        max_sections=5,
    )


async def assemble_build_substep_context_async(
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
    retrieval_adapters: ContextRetrievalAdapters,
) -> ContextBundle:
    """Build executable substep context with retrieval-backed candidates."""
    base = assemble_build_substep_context(
        session,
        step=step,
        step_index=step_index,
        total_steps=total_steps,
        step_subplan=step_subplan,
        completed_substeps=completed_substeps,
        current_substep=current_substep,
        substep_index=substep_index,
        substep_count=substep_count,
        workspace_snapshot_formatter=workspace_snapshot_formatter,
        subplan_progress_formatter=subplan_progress_formatter,
    )
    dynamic_sections = await build_dynamic_context_sections(
        session,
        phase="build_substep",
        request_text=current_substep or step or session.task_request or session.message,
        adapters=retrieval_adapters,
    )
    return _merge_dynamic_sections(
        base,
        dynamic_sections,
        request_text=current_substep or step or session.task_request or session.message,
        max_sections=10,
    )


async def assemble_verify_context_async(
    session: DeepSession,
    *,
    retrieval_adapters: ContextRetrievalAdapters,
) -> ContextBundle:
    """Build verification context with retrieval-backed candidates."""
    base = assemble_verify_context(session)
    dynamic_sections = await build_dynamic_context_sections(
        session,
        phase="verify",
        request_text=session.task_request or session.message,
        adapters=retrieval_adapters,
    )
    return _merge_dynamic_sections(
        base,
        dynamic_sections,
        request_text=session.task_request or session.message,
        max_sections=9,
    )


async def assemble_synthesis_context_async(
    session: DeepSession,
    *,
    retrieval_adapters: ContextRetrievalAdapters,
) -> ContextBundle:
    """Build synthesis context with retrieval-backed candidates."""
    base = assemble_synthesis_context(session)
    dynamic_sections = await build_dynamic_context_sections(
        session,
        phase="synthesize",
        request_text=session.task_request or session.message,
        adapters=retrieval_adapters,
    )
    return _merge_dynamic_sections(
        base,
        dynamic_sections,
        request_text=session.task_request or session.message,
        max_sections=9,
    )
