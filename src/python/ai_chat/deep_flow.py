"""Pure decision helpers for the deep inspect/plan/execute/verify pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepRouteRequest:
    """Inputs that decide how a deep-mode turn should proceed."""

    requires_existing_plan: bool = False
    has_plan: bool = False
    should_preview_plan: bool = False
    execution_requested: bool = False
    auto_execute: bool = False
    workspace_write: bool = False


@dataclass(frozen=True)
class DeepRouteDecision:
    """Normalized deep-mode action."""

    action: str
    requires_plan: bool = False
    requires_write_permission: bool = False
    requires_review: bool = False


def decide_deep_route(request: DeepRouteRequest) -> DeepRouteDecision:
    """Map deep-mode state to one explicit next action."""
    if request.requires_existing_plan and not request.has_plan:
        return DeepRouteDecision(
            action="missing_plan",
            requires_plan=True,
        )
    if request.should_preview_plan:
        return DeepRouteDecision(
            action="preview_plan",
            requires_plan=True,
        )
    if request.execution_requested or request.auto_execute:
        if not request.workspace_write:
            return DeepRouteDecision(
                action="blocked_write",
                requires_plan=True,
                requires_write_permission=True,
            )
        return DeepRouteDecision(
            action="execute_plan",
            requires_plan=True,
            requires_write_permission=True,
            requires_review=True,
        )
    return DeepRouteDecision(action="direct_answer")
