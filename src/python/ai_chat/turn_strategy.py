"""Pure turn-routing helpers for the app's skill loop.

The goal is to make the main chat turn easy to reason about:
1. compare against local RAG context when it exists
2. detect whether search is needed
3. detect whether file creation is needed
4. detect whether a coding loop is needed
5. detect whether planning mode is needed
6. choose a one-shot skill or planned execution path
7. flag whether the route should end with review/verification
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List

from src.python.ai_chat.route_intake import StructuredRouteIntake


FILE_CREATION_PHRASES = (
    "create a file",
    "create files",
    "new file",
    "new files",
    "write a file",
    "write files",
    "scaffold a repo",
    "scaffold the repo",
    "generate a repo",
    "generate the repo",
    "create a repo",
    "create the repo",
    "starter project",
    "starter app",
)
PLANNING_PHRASES = (
    "make a plan",
    "give me a plan",
    "plan this",
    "plan it",
    "plan how to",
    "execution plan",
    "planning mode",
    "plan mode",
    "step by step plan",
    "step-by-step plan",
    "roadmap",
)
CODING_HINTS = {
    "bug", "build", "code", "coding", "compile", "debug", "developer", "edit",
    "feature", "fix", "implementation", "implement", "patch", "programmer",
    "programming", "refactor", "test",
}
FILE_HINTS = {
    "app", "boilerplate", "directory", "file", "files", "folder", "folders",
    "project", "repo", "repository", "scaffold", "starter", "template",
}
PLANNING_HINTS = {
    "architecture", "breakdown", "checklist", "milestone", "milestones", "plan",
    "planner", "planning", "roadmap", "steps", "strategy",
}
TEST_REVIEW_HINTS = {
    "check", "review", "smoke", "test", "tests", "validate", "verification", "verify",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_+-]+", str(text or "").lower()))


def _normalize_slash_name(value: str) -> str:
    return str(value or "").strip().lower()


def infer_explicit_planning_request(message: str) -> bool:
    """Return whether the user directly asked for a plan or planning mode."""
    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return False
    if any(phrase in text for phrase in PLANNING_PHRASES):
        return True
    words = _tokenize(text)
    return bool(words & PLANNING_HINTS and {"make", "need", "show", "want", "write"} & words)


def infer_file_creation_required(message: str, workspace_intent: str, slash_command_name: str = "") -> bool:
    """Estimate whether the request is likely to create files or folders."""
    slash_name = _normalize_slash_name(slash_command_name)
    if slash_name == "plan":
        return False

    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return False
    if any(phrase in text for phrase in FILE_CREATION_PHRASES):
        return True

    words = _tokenize(text)
    if {"create", "generate", "make", "scaffold", "write"} & words and FILE_HINTS & words:
        return True
    if workspace_intent == "broad_write" and {"template", "starter", "scaffold", "repo", "repository"} & words:
        return True
    return False


def infer_coding_loop_required(
    message: str,
    workspace_intent: str,
    enabled_tools: List[str],
    slash_command_name: str = "",
) -> bool:
    """Estimate whether the request should route through a coding-oriented loop."""
    slash_name = _normalize_slash_name(slash_command_name)
    if slash_name in {"code", "grep"}:
        return True
    if workspace_intent in {"focused_write", "broad_write"}:
        return True
    if any(tool_name in {"workspace.patch_file", "workspace.run_command"} for tool_name in enabled_tools):
        return True

    words = _tokenize(message)
    return bool(words & CODING_HINTS and (words & FILE_HINTS or "app" in words or "codebase" in words))


def infer_search_required(
    *,
    enabled_tools: List[str],
    slash_command_name: str = "",
    local_rag_requested: bool = False,
    web_search_requested: bool = False,
) -> bool:
    """Return whether any search or retrieval loop is part of this turn."""
    slash_name = _normalize_slash_name(slash_command_name)
    if slash_name in {"search", "grep"}:
        return True
    if local_rag_requested or web_search_requested:
        return True
    return any(
        tool_name in {"conversation.search_history", "web.search", "web.fetch_page", "workspace.grep"}
        for tool_name in enabled_tools
    )


def infer_planning_required(
    *,
    message: str,
    requested_mode: str,
    resolved_mode: str,
    workspace_intent: str,
    slash_command_name: str = "",
    auto_execute_workspace: bool = False,
    resume_saved_workspace: bool = False,
    execution_requested: bool = False,
) -> bool:
    """Return whether the turn should think in plan-oriented mode."""
    slash_name = _normalize_slash_name(slash_command_name)
    if slash_name == "plan":
        return True
    if auto_execute_workspace or resume_saved_workspace or execution_requested:
        return True
    if str(requested_mode or "").strip().lower() == "deep":
        return True
    if str(resolved_mode or "").strip().lower() == "deep":
        return True
    if workspace_intent == "broad_write":
        return True
    return infer_explicit_planning_request(message)


def infer_primary_skill(
    *,
    slash_command_name: str,
    compare_to_rag: bool,
    requires_search: bool,
    requires_file_creation: bool,
    requires_coding_loop: bool,
    requires_planning: bool,
    enabled_tools: List[str],
) -> str:
    """Pick the clearest primary skill label for analytics and UI summaries."""
    slash_name = _normalize_slash_name(slash_command_name)
    if slash_name == "search":
        return "search"
    if slash_name == "grep":
        return "grep"
    if slash_name == "plan":
        return "planning"
    if slash_name == "code":
        return "coding"
    if requires_planning and requires_coding_loop:
        return "plan_and_code"
    if requires_planning:
        return "planning"
    if requires_search and any(tool_name.startswith("web.") for tool_name in enabled_tools):
        return "search"
    if requires_search and "workspace.grep" in enabled_tools:
        return "grep"
    if requires_file_creation:
        return "file_creation"
    if requires_coding_loop:
        return "coding"
    if compare_to_rag:
        return "rag_compare"
    if enabled_tools:
        return "tool_loop"
    return "direct_answer"


def infer_execution_style(
    *,
    slash_command_name: str,
    requires_planning: bool,
    auto_execute_workspace: bool,
    resume_saved_workspace: bool,
    execution_requested: bool,
    enabled_tools: List[str],
) -> str:
    """Choose the simplest execution style for the current turn."""
    slash_name = _normalize_slash_name(slash_command_name)
    if slash_name in {"search", "grep"}:
        return "one_shot_skill"
    if slash_name == "plan":
        return "plan_preview" if not execution_requested else "plan_execution"
    if slash_name == "code":
        return "code_execution"
    if requires_planning:
        if auto_execute_workspace or resume_saved_workspace or execution_requested:
            return "plan_execution"
        return "plan_preview"
    if enabled_tools:
        return "one_shot_skill"
    return "direct_answer"


@dataclass
class TurnAssessment:
    """Structured summary of the app's skill-loop decision for one user turn."""

    slash_command_name: str = ""
    compare_to_rag: bool = False
    requires_search: bool = False
    requires_file_creation: bool = False
    requires_coding_loop: bool = False
    requires_planning: bool = False
    requires_workspace: bool = False
    requires_step_review: bool = False
    explicit_planning_request: bool = False
    workspace_intent: str = "none"
    enabled_tools: List[str] = field(default_factory=list)
    primary_skill: str = "direct_answer"
    execution_style: str = "direct_answer"
    needs_fresh_info: bool = False
    needs_search_citations: bool = False
    is_versioned_release_query: bool = False
    entity: str = ""
    time_sensitivity: str = "stable"
    answer_shape: str = "answer"
    route_confidence: float = 0.0
    route_reasoning: str = ""
    route_intake: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_metadata(self) -> Dict[str, Any]:
        """Return a JSON-safe payload for workflow metadata."""
        return asdict(self)


def build_turn_assessment(
    *,
    message: str,
    requested_mode: str,
    resolved_mode: str,
    workspace_intent: str,
    enabled_tools: List[str],
    workspace_requested: bool,
    has_attachment_context: bool = False,
    slash_command_name: str = "",
    local_rag_requested: bool = False,
    web_search_requested: bool = False,
    auto_execute_workspace: bool = False,
    resume_saved_workspace: bool = False,
    execution_requested: bool = False,
    workspace_run_commands_enabled: bool = False,
    route_intake: StructuredRouteIntake | None = None,
) -> TurnAssessment:
    """Build the explicit decision record for the current turn."""
    route = route_intake or StructuredRouteIntake()
    compare_to_rag = bool(
        has_attachment_context
        or local_rag_requested
        or "conversation.search_history" in enabled_tools
        or route.local_rag_requested
    )
    requires_search = bool(
        route.web_search_requested
        or infer_search_required(
            enabled_tools=enabled_tools,
            slash_command_name=slash_command_name,
            local_rag_requested=local_rag_requested,
            web_search_requested=web_search_requested,
        )
    )
    requires_file_creation = infer_file_creation_required(
        message,
        workspace_intent,
        slash_command_name=slash_command_name,
    )
    requires_coding_loop = infer_coding_loop_required(
        message,
        workspace_intent,
        enabled_tools,
        slash_command_name=slash_command_name,
    )
    explicit_planning_request = infer_explicit_planning_request(message)
    requires_planning = infer_planning_required(
        message=message,
        requested_mode=requested_mode,
        resolved_mode=resolved_mode,
        workspace_intent=workspace_intent,
        slash_command_name=slash_command_name,
        auto_execute_workspace=auto_execute_workspace,
        resume_saved_workspace=resume_saved_workspace,
        execution_requested=execution_requested,
    )
    requires_workspace = bool(
        workspace_requested
        or route.needs_workspace
        or workspace_intent != "none"
        or any(
            tool_name.startswith("workspace.") or tool_name == "spreadsheet.describe"
            for tool_name in enabled_tools
        )
    )
    requires_step_review = bool(
        requires_coding_loop
        or requires_file_creation
        or requires_planning
        or workspace_run_commands_enabled
        or any(tool_name == "workspace.run_command" for tool_name in enabled_tools)
        or TEST_REVIEW_HINTS & _tokenize(message)
    )
    primary_skill = infer_primary_skill(
        slash_command_name=slash_command_name,
        compare_to_rag=compare_to_rag,
        requires_search=requires_search,
        requires_file_creation=requires_file_creation,
        requires_coding_loop=requires_coding_loop,
        requires_planning=requires_planning,
        enabled_tools=enabled_tools,
    )
    execution_style = infer_execution_style(
        slash_command_name=slash_command_name,
        requires_planning=requires_planning,
        auto_execute_workspace=auto_execute_workspace,
        resume_saved_workspace=resume_saved_workspace,
        execution_requested=execution_requested,
        enabled_tools=enabled_tools,
    )

    notes: List[str] = []
    if compare_to_rag:
        notes.append("local_rag")
    if route.needs_fresh_info:
        notes.append("fresh_info")
    if requires_search:
        notes.append("search")
    if requires_file_creation:
        notes.append("file_creation")
    if requires_coding_loop:
        notes.append("coding_loop")
    if requires_planning:
        notes.append("planning")
    if requires_step_review:
        notes.append("review")

    return TurnAssessment(
        slash_command_name=_normalize_slash_name(slash_command_name),
        compare_to_rag=compare_to_rag,
        requires_search=requires_search,
        requires_file_creation=requires_file_creation,
        requires_coding_loop=requires_coding_loop,
        requires_planning=requires_planning,
        requires_workspace=requires_workspace,
        requires_step_review=requires_step_review,
        explicit_planning_request=explicit_planning_request,
        workspace_intent=str(workspace_intent or "none"),
        enabled_tools=list(enabled_tools),
        primary_skill=primary_skill,
        execution_style=execution_style,
        needs_fresh_info=route.needs_fresh_info,
        needs_search_citations=route.needs_search_citations,
        is_versioned_release_query=route.is_versioned_release_query,
        entity=route.entity,
        time_sensitivity=route.time_sensitivity,
        answer_shape=route.answer_shape,
        route_confidence=route.confidence,
        route_reasoning=route.reasoning,
        route_intake=route.as_metadata(),
        notes=notes,
    )


def format_turn_assessment_summary(assessment: TurnAssessment) -> str:
    """Render the routing decision as a compact user-facing skill-loop summary."""
    parts = [
        f"RAG {'yes' if assessment.compare_to_rag else 'no'}",
        f"search {'yes' if assessment.requires_search else 'no'}",
        f"files {'yes' if assessment.requires_file_creation else 'no'}",
        f"code {'yes' if assessment.requires_coding_loop else 'no'}",
        f"plan {'yes' if assessment.requires_planning else 'no'}",
        f"execute {assessment.execution_style.replace('_', ' ')}",
        f"review {'yes' if assessment.requires_step_review else 'no'}",
    ]
    if assessment.primary_skill and assessment.primary_skill != "direct_answer":
        parts.append(f"skill {assessment.primary_skill.replace('_', ' ')}")
    if assessment.slash_command_name:
        parts.append(f"slash /{assessment.slash_command_name}")
    return "Skill loop: " + " | ".join(parts)
