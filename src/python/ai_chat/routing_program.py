"""Routing-program seam for the structured top-level turn router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.python.ai_chat.route_intake import StructuredRouteIntake
from src.python.ai_chat.turn_strategy import TurnAssessment, build_turn_assessment


@dataclass(frozen=True)
class RouteProgramInputs:
    """Stable inputs for the top-level turn router."""

    message: str
    requested_mode: str
    resolved_mode: str
    workspace_intent: str
    enabled_tools: List[str]
    workspace_requested: bool
    has_attachment_context: bool = False
    slash_command_name: str = ""
    local_rag_requested: bool = False
    web_search_requested: bool = False
    auto_execute_workspace: bool = False
    resume_saved_workspace: bool = False
    execution_requested: bool = False
    workspace_run_commands_enabled: bool = False
    route_intake: Optional[StructuredRouteIntake] = None


@dataclass(frozen=True)
class RouteProgramOutput:
    """Normalized result for one routing-program invocation."""

    assessment: TurnAssessment


class HeuristicRouteProgram:
    """Route assessment implementation fed by structured intake plus strategy helpers."""

    def run(self, inputs: RouteProgramInputs) -> RouteProgramOutput:
        return RouteProgramOutput(
            assessment=build_turn_assessment(
                message=inputs.message,
                requested_mode=inputs.requested_mode,
                resolved_mode=inputs.resolved_mode,
                workspace_intent=inputs.workspace_intent,
                enabled_tools=inputs.enabled_tools,
                workspace_requested=inputs.workspace_requested,
                has_attachment_context=inputs.has_attachment_context,
                slash_command_name=inputs.slash_command_name,
                local_rag_requested=inputs.local_rag_requested,
                web_search_requested=inputs.web_search_requested,
                auto_execute_workspace=inputs.auto_execute_workspace,
                resume_saved_workspace=inputs.resume_saved_workspace,
                execution_requested=inputs.execution_requested,
                workspace_run_commands_enabled=inputs.workspace_run_commands_enabled,
                route_intake=inputs.route_intake,
            )
        )


DEFAULT_ROUTE_PROGRAM = HeuristicRouteProgram()
