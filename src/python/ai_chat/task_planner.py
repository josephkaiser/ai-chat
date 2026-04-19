"""Heuristic planner for the first structured task-engine slice."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from src.python.ai_chat.task_types import TaskContext, TaskStep

STRUCTURED_READ_ONLY_TOOLS = frozenset({
    "workspace.list_files",
    "workspace.grep",
    "workspace.read_file",
})

SEARCH_QUERY_STOPWORDS = {
    "about",
    "after",
    "before",
    "could",
    "does",
    "explain",
    "file",
    "from",
    "have",
    "into",
    "look",
    "open",
    "please",
    "read",
    "repo",
    "repository",
    "review",
    "show",
    "that",
    "this",
    "what",
    "with",
    "workspace",
    "would",
}


def supports_structured_read_only_tools(allowed_tools: Sequence[str]) -> bool:
    """Return whether the allowed tools fit the current structured-engine slice."""
    cleaned = {
        str(tool_name).strip()
        for tool_name in allowed_tools
        if isinstance(tool_name, str) and str(tool_name).strip()
    }
    return bool(cleaned) and cleaned.issubset(STRUCTURED_READ_ONLY_TOOLS)


def summarize_structured_plan(steps: Sequence[TaskStep]) -> str:
    """Render a compact one-line summary for activity logs."""
    labels = [step.title for step in steps if step.kind != "finish"]
    if not labels:
        return "No structured steps were needed."
    return " -> ".join(labels)


def build_heuristic_task_plan(context: TaskContext) -> List[TaskStep]:
    """Build a short read-only plan grounded in explicit workspace paths."""
    steps: List[TaskStep] = []
    next_id = 1
    toolset = set(context.allowed_tools)

    for target in context.path_targets[:2]:
        if target.kind == "file" and "workspace.read_file" in toolset:
            steps.append(TaskStep(
                id=f"s{next_id}",
                kind="read",
                title=f"Read {target.path}",
                args={"path": target.path},
            ))
            next_id += 1
            continue

        if target.kind == "dir" and "workspace.list_files" in toolset:
            steps.append(TaskStep(
                id=f"s{next_id}",
                kind="list",
                title=f"List {target.path}",
                args={"path": target.path},
            ))
            next_id += 1
            query = build_search_query(context.user_message)
            if query and "workspace.grep" in toolset:
                steps.append(TaskStep(
                    id=f"s{next_id}",
                    kind="search",
                    title=f"Search {target.path}",
                    args={"path": target.path, "query": query},
                ))
                next_id += 1

    if not steps and "workspace.grep" in toolset:
        query = build_search_query(context.user_message)
        if query:
            steps.append(TaskStep(
                id=f"s{next_id}",
                kind="search",
                title=f"Search workspace for {query}",
                args={"path": ".", "query": query},
            ))
            next_id += 1

    if not steps and "workspace.list_files" in toolset:
        steps.append(TaskStep(
            id=f"s{next_id}",
            kind="list",
            title="List workspace root",
            args={"path": "."},
        ))
        next_id += 1

    steps.append(TaskStep(
        id=f"s{next_id}",
        kind="finish",
        title="Answer the user",
        args={"summary_hint": context.request_focus or "Summarize the inspected workspace evidence."},
    ))
    return steps


def build_search_query(message: str, limit: int = 6) -> str:
    """Extract a few durable terms for a grep step."""
    terms: List[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9._/-]*", str(message or "").lower()):
        if token in SEARCH_QUERY_STOPWORDS:
            continue
        if token.startswith("./") or "/" in token or "." in token:
            continue
        if len(token) < 3:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return " ".join(terms)
