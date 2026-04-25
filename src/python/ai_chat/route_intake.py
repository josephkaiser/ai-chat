"""Structured route-intake primitives for the top-level turn router."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


ROUTE_TIME_SENSITIVITY_VALUES = {
    "stable",
    "current",
    "recent",
    "versioned",
    "historical",
}
ROUTE_ANSWER_SHAPE_VALUES = {
    "answer",
    "summary",
    "comparison",
    "explanation",
    "instructions",
    "edit",
    "build",
    "review",
    "search",
}
ROUTE_WORKSPACE_INTENT_VALUES = {
    "none",
    "focused_read",
    "broad_read",
    "focused_write",
    "broad_write",
}


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


@dataclass(frozen=True)
class StructuredRouteIntake:
    """Compact structured reading of one user turn before execution begins."""

    needs_fresh_info: bool = False
    is_versioned_release_query: bool = False
    entity: str = ""
    time_sensitivity: str = "stable"
    answer_shape: str = "answer"
    needs_workspace: bool = False
    needs_artifact: bool = False
    needs_search_citations: bool = False
    workspace_intent_hint: str = "none"
    render_requested: bool = False
    local_rag_requested: bool = False
    web_search_requested: bool = False
    reasoning: str = ""
    confidence: float = 0.0

    def as_metadata(self) -> Dict[str, Any]:
        """Return a JSON-safe representation for workflow metadata."""
        return asdict(self)


def normalize_structured_route_intake(value: Any) -> StructuredRouteIntake:
    """Coerce a dict-like payload into a validated route-intake object."""
    if isinstance(value, StructuredRouteIntake):
        return value
    if not isinstance(value, dict):
        return StructuredRouteIntake()

    raw_confidence = value.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    return StructuredRouteIntake(
        needs_fresh_info=bool(value.get("needs_fresh_info")),
        is_versioned_release_query=bool(value.get("is_versioned_release_query")),
        entity=str(value.get("entity", "") or "").strip(),
        time_sensitivity=_normalize_choice(
            value.get("time_sensitivity"),
            ROUTE_TIME_SENSITIVITY_VALUES,
            "stable",
        ),
        answer_shape=_normalize_choice(
            value.get("answer_shape"),
            ROUTE_ANSWER_SHAPE_VALUES,
            "answer",
        ),
        needs_workspace=bool(value.get("needs_workspace")),
        needs_artifact=bool(value.get("needs_artifact")),
        needs_search_citations=bool(value.get("needs_search_citations")),
        workspace_intent_hint=_normalize_choice(
            value.get("workspace_intent_hint") or value.get("workspace_intent"),
            ROUTE_WORKSPACE_INTENT_VALUES,
            "none",
        ),
        render_requested=bool(value.get("render_requested")),
        local_rag_requested=bool(value.get("local_rag_requested")),
        web_search_requested=bool(value.get("web_search_requested")),
        reasoning=str(value.get("reasoning", "") or "").strip(),
        confidence=confidence,
    )
