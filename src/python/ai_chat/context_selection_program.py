"""Heuristic context selection seam for retrieval-style runtime bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Sequence, Set


@dataclass(frozen=True)
class ContextCandidate:
    """One candidate context slice before phase-specific selection."""

    key: str
    title: str
    content: str
    priority: int = 100
    required: bool = False
    tags: Sequence[str] = field(default_factory=tuple)
    phase_hints: Sequence[str] = field(default_factory=tuple)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextSelectionInputs:
    """Stable inputs for selecting the best context slices for one phase."""

    phase: str
    request_text: str
    candidates: Sequence[ContextCandidate]
    max_sections: int = 6


@dataclass(frozen=True)
class ContextSelectionOutput:
    """Selection result for one retrieval-style context assembly pass."""

    selected_keys: List[str]
    omitted_keys: List[str]
    scores: Dict[str, float]
    reasons: Dict[str, str]


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9_./-]+", str(text or "").lower()))


class HeuristicContextSelectionProgram:
    """Heuristic scorer today, DSPy-compatible seam later."""

    def run(self, inputs: ContextSelectionInputs) -> ContextSelectionOutput:
        request_tokens = _tokenize(inputs.request_text)
        selected_required: List[str] = []
        optional_scores: List[tuple[float, int, str]] = []
        scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        omitted: List[str] = []

        for index, candidate in enumerate(inputs.candidates):
            content = str(candidate.content or "").strip()
            if not content:
                continue

            score, reason = self._score_candidate(candidate, request_tokens, inputs.phase)
            scores[candidate.key] = round(score, 4)
            reasons[candidate.key] = reason

            if candidate.required:
                selected_required.append(candidate.key)
            else:
                optional_scores.append((score, index, candidate.key))

        optional_scores.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        optional_budget = max(0, int(inputs.max_sections or 0) - len(selected_required))
        selected_optional = [key for _, _, key in optional_scores[:optional_budget]]
        omitted.extend(key for _, _, key in optional_scores[optional_budget:])

        selected_keys = selected_required + selected_optional
        return ContextSelectionOutput(
            selected_keys=selected_keys,
            omitted_keys=omitted,
            scores=scores,
            reasons=reasons,
        )

    def _score_candidate(
        self,
        candidate: ContextCandidate,
        request_tokens: Set[str],
        phase: str,
    ) -> tuple[float, str]:
        candidate_tokens = _tokenize(" ".join([
            candidate.title,
            candidate.content,
            " ".join(candidate.tags),
        ]))
        overlap = len(request_tokens & candidate_tokens)

        score = 0.0
        reasons: List[str] = []

        score += max(0.0, 220.0 - float(candidate.priority))
        if candidate.required:
            score += 1000.0
            reasons.append("required")

        if phase and phase in set(candidate.phase_hints):
            score += 80.0
            reasons.append("phase-match")

        if overlap:
            score += float(overlap * 24)
            reasons.append(f"request-overlap:{overlap}")

        recency_rank = str(candidate.metadata.get("recency_rank", "")).strip()
        if recency_rank.isdigit():
            score += max(0.0, 30.0 - float(int(recency_rank) * 4))
            reasons.append("recent")

        sticky = str(candidate.metadata.get("sticky", "")).strip().lower()
        if sticky in {"1", "true", "yes"}:
            score += 90.0
            reasons.append("sticky")

        if request_tokens & {"artifact", "artifacts", "file", "files", "path", "paths"} and set(candidate.tags) & {"artifacts", "files"}:
            score += 35.0
            reasons.append("artifact-file-request")

        if request_tokens & {"verify", "verification", "check", "test"} and set(candidate.tags) & {"verification", "audit"}:
            score += 40.0
            reasons.append("verification-request")

        if request_tokens & {"edit", "change", "patch", "update", "refactor", "file"} and set(candidate.tags) & {"files", "workspace"}:
            score += 28.0
            reasons.append("workspace-edit-request")

        if phase in {"verify", "synthesize"} and set(candidate.tags) & {"files", "artifacts"}:
            score += 55.0
            reasons.append("phase-evidence")

        if phase == "verify" and set(candidate.tags) & {"conversation", "history"}:
            score -= 35.0
            reasons.append("verify-deprioritize-chat")

        if phase == "plan" and set(candidate.tags) & {"files", "artifacts", "workspace"}:
            score += 48.0
            reasons.append("planning-evidence")

        if phase == "build_substep" and set(candidate.tags) & {"progress", "files", "workspace"}:
            score += 30.0
            reasons.append("execution-evidence")

        if not reasons:
            reasons.append("base-priority")
        return score, ", ".join(reasons)


DEFAULT_CONTEXT_SELECTION_PROGRAM = HeuristicContextSelectionProgram()
