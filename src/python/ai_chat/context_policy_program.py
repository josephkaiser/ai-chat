"""Policy layer for deciding which retrieval actions a context pass should take."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Sequence, Set


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9_./-]+", str(text or "").lower()))


@dataclass(frozen=True)
class ContextPolicyInputs:
    """Stable inputs for retrieval-policy decisions."""

    phase: str
    request_text: str
    changed_files: Sequence[str] = field(default_factory=tuple)
    workspace_sample_paths: Sequence[str] = field(default_factory=tuple)
    has_task_board: bool = False
    has_recent_feedback: bool = False
    has_scope_gaps: bool = False
    has_step_reports: bool = False


@dataclass(frozen=True)
class ContextPolicyDecision:
    """Retrieval actions to take for one phase."""

    retrieve_memory: bool
    retrieve_workspace_previews: bool
    memory_limit: int
    workspace_preview_limit: int
    preferred_paths: List[str]
    include_scope_gaps: bool
    include_step_evidence: bool
    reasons: Dict[str, str]


class HeuristicContextPolicyProgram:
    """Heuristic policy today, DSPy-compatible policy seam later."""

    def run(self, inputs: ContextPolicyInputs) -> ContextPolicyDecision:
        request_tokens = _tokenize(inputs.request_text)
        phase = str(inputs.phase or "").strip()
        reasons: Dict[str, str] = {}

        retrieve_memory = False
        retrieve_workspace_previews = False
        memory_limit = 0
        workspace_preview_limit = 0

        conversational_phases = {"inspect", "plan", "direct_answer"}
        evidence_phases = {"build_substep", "verify", "synthesize", "step_subplan"}

        if phase in conversational_phases or request_tokens & {"before", "earlier", "previous", "history", "chat", "conversation"}:
            retrieve_memory = True
            memory_limit = 3 if phase in {"inspect", "plan"} else 2
            reasons["retrieve_memory"] = "conversation continuity or planning context"

        if phase == "plan":
            retrieve_workspace_previews = True
            workspace_preview_limit = max(workspace_preview_limit, 3)
            reasons["retrieve_workspace_previews"] = "planning should stay grounded in real workspace slices"

        if phase in evidence_phases:
            retrieve_workspace_previews = True
            workspace_preview_limit = 3 if phase in {"verify", "synthesize", "build_substep"} else 2
            reasons["retrieve_workspace_previews"] = "phase needs file-grounded evidence"

        if request_tokens & {"file", "files", "path", "paths", "artifact", "artifacts", "code", "app.js", "harness.py"}:
            retrieve_workspace_previews = True
            workspace_preview_limit = max(workspace_preview_limit, 3)
            reasons["retrieve_workspace_previews"] = "request explicitly targets file or artifact evidence"

        include_scope_gaps = bool(inputs.has_scope_gaps and phase in {"verify", "synthesize"})
        if include_scope_gaps:
            reasons["include_scope_gaps"] = "verification or synthesis should expose remaining gaps"

        include_step_evidence = bool(inputs.has_step_reports and phase in {"build_substep", "verify", "synthesize"})
        if include_step_evidence:
            reasons["include_step_evidence"] = "phase benefits from recent execution evidence"

        preferred_paths: List[str] = []
        if inputs.has_task_board and phase in {"plan", "build_substep", "verify", "synthesize"}:
            preferred_paths.append(".ai/task-board.md")
        if inputs.has_recent_feedback and phase in {"plan", "verify", "synthesize"}:
            preferred_paths.append(".ai/recent-feedback.md")
        for path in inputs.changed_files[:workspace_preview_limit or 3]:
            cleaned = str(path or "").strip()
            if cleaned and cleaned not in preferred_paths:
                preferred_paths.append(cleaned)
        for path in inputs.workspace_sample_paths[:workspace_preview_limit or 2]:
            cleaned = str(path or "").strip()
            if cleaned and cleaned not in preferred_paths:
                preferred_paths.append(cleaned)

        return ContextPolicyDecision(
            retrieve_memory=retrieve_memory,
            retrieve_workspace_previews=retrieve_workspace_previews,
            memory_limit=memory_limit,
            workspace_preview_limit=workspace_preview_limit,
            preferred_paths=preferred_paths,
            include_scope_gaps=include_scope_gaps,
            include_step_evidence=include_step_evidence,
            reasons=reasons,
        )


DEFAULT_CONTEXT_POLICY_PROGRAM = HeuristicContextPolicyProgram()
