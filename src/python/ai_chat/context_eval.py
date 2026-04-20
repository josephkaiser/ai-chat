"""Replayable evaluation harness for context policy and selection behavior."""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from src.python.ai_chat.context_policy_program import (
    DEFAULT_CONTEXT_POLICY_PROGRAM,
    ContextPolicyDecision,
    ContextPolicyInputs,
)
from src.python.ai_chat.context_selection_program import (
    DEFAULT_CONTEXT_SELECTION_PROGRAM,
    ContextCandidate,
    ContextSelectionInputs,
    ContextSelectionOutput,
)


@dataclass(frozen=True)
class ContextEvalExpectation:
    """Expected retrieval and selection behavior for one replay case."""

    retrieve_memory: bool | None = None
    retrieve_workspace_previews: bool | None = None
    min_memory_limit: int = 0
    min_workspace_preview_limit: int = 0
    required_selected_keys: Sequence[str] = field(default_factory=tuple)
    forbidden_selected_keys: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ContextEvalCase:
    """One replayable context-evaluation case."""

    name: str
    policy_inputs: ContextPolicyInputs
    selection_candidates: Sequence[ContextCandidate]
    selection_max_sections: int
    expectation: ContextEvalExpectation


@dataclass(frozen=True)
class ContextEvalResult:
    """Scored result for one replay case."""

    name: str
    passed: bool
    score: float
    passed_checks: int
    total_checks: int
    failed_checks: List[str]
    policy_decision: ContextPolicyDecision
    selection_output: ContextSelectionOutput


@dataclass(frozen=True)
class CapturedContextEvalResult:
    """Evaluated result for one captured replay case loaded from disk."""

    source_path: str
    capture: Dict[str, object]
    result: ContextEvalResult


DEFAULT_CONTEXT_EVAL_CASES_PATH = (
    pathlib.Path(__file__).resolve().parent / "fixtures" / "context_eval_cases.json"
)


def context_eval_expectation_from_dict(payload: Dict[str, object]) -> ContextEvalExpectation:
    """Normalize one expectation payload from JSON."""
    return ContextEvalExpectation(
        retrieve_memory=payload.get("retrieve_memory") if "retrieve_memory" in payload else None,
        retrieve_workspace_previews=(
            payload.get("retrieve_workspace_previews")
            if "retrieve_workspace_previews" in payload else None
        ),
        min_memory_limit=int(payload.get("min_memory_limit", 0) or 0),
        min_workspace_preview_limit=int(payload.get("min_workspace_preview_limit", 0) or 0),
        required_selected_keys=tuple(
            str(item).strip()
            for item in payload.get("required_selected_keys", [])
            if str(item).strip()
        ),
        forbidden_selected_keys=tuple(
            str(item).strip()
            for item in payload.get("forbidden_selected_keys", [])
            if str(item).strip()
        ),
    )


def context_eval_case_from_dict(payload: Dict[str, object]) -> ContextEvalCase:
    """Normalize one replay case payload from JSON."""
    policy_inputs_raw = payload.get("policy_inputs", {})
    expectation_raw = payload.get("expectation", {})
    candidate_payloads = payload.get("selection_candidates", [])

    if not isinstance(policy_inputs_raw, dict):
        raise ValueError("policy_inputs must be an object")
    if not isinstance(expectation_raw, dict):
        raise ValueError("expectation must be an object")
    if not isinstance(candidate_payloads, list):
        raise ValueError("selection_candidates must be a list")

    return ContextEvalCase(
        name=str(payload.get("name", "")).strip(),
        policy_inputs=ContextPolicyInputs(
            phase=str(policy_inputs_raw.get("phase", "")).strip(),
            request_text=str(policy_inputs_raw.get("request_text", "")).strip(),
            changed_files=tuple(
                str(item).strip()
                for item in policy_inputs_raw.get("changed_files", [])
                if str(item).strip()
            ),
            workspace_sample_paths=tuple(
                str(item).strip()
                for item in policy_inputs_raw.get("workspace_sample_paths", [])
                if str(item).strip()
            ),
            has_task_board=bool(policy_inputs_raw.get("has_task_board")),
            has_recent_feedback=bool(policy_inputs_raw.get("has_recent_feedback")),
            has_scope_gaps=bool(policy_inputs_raw.get("has_scope_gaps")),
            has_step_reports=bool(policy_inputs_raw.get("has_step_reports")),
        ),
        selection_candidates=tuple(
            ContextCandidate(
                key=str(candidate.get("key", "")).strip(),
                title=str(candidate.get("title", "")).strip(),
                content=str(candidate.get("content", "")).strip(),
                priority=int(candidate.get("priority", 100) or 100),
                required=bool(candidate.get("required")),
                tags=tuple(str(item).strip() for item in candidate.get("tags", []) if str(item).strip()),
                phase_hints=tuple(
                    str(item).strip()
                    for item in candidate.get("phase_hints", [])
                    if str(item).strip()
                ),
                metadata={
                    str(key).strip(): str(value).strip()
                    for key, value in dict(candidate.get("metadata", {})).items()
                    if str(key).strip()
                },
            )
            for candidate in candidate_payloads
            if isinstance(candidate, dict)
        ),
        selection_max_sections=int(payload.get("selection_max_sections", 6) or 6),
        expectation=context_eval_expectation_from_dict(expectation_raw),
    )


def serialize_context_eval_case(case: ContextEvalCase) -> Dict[str, object]:
    """Convert one replay case into a JSON-safe payload."""
    return {
        "name": case.name,
        "policy_inputs": {
            "phase": case.policy_inputs.phase,
            "request_text": case.policy_inputs.request_text,
            "changed_files": list(case.policy_inputs.changed_files),
            "workspace_sample_paths": list(case.policy_inputs.workspace_sample_paths),
            "has_task_board": case.policy_inputs.has_task_board,
            "has_recent_feedback": case.policy_inputs.has_recent_feedback,
            "has_scope_gaps": case.policy_inputs.has_scope_gaps,
            "has_step_reports": case.policy_inputs.has_step_reports,
        },
        "selection_candidates": [
            {
                "key": candidate.key,
                "title": candidate.title,
                "content": candidate.content,
                "priority": candidate.priority,
                "required": candidate.required,
                "tags": list(candidate.tags),
                "phase_hints": list(candidate.phase_hints),
                "metadata": dict(candidate.metadata),
            }
            for candidate in case.selection_candidates
        ],
        "selection_max_sections": case.selection_max_sections,
        "expectation": {
            "retrieve_memory": case.expectation.retrieve_memory,
            "retrieve_workspace_previews": case.expectation.retrieve_workspace_previews,
            "min_memory_limit": case.expectation.min_memory_limit,
            "min_workspace_preview_limit": case.expectation.min_workspace_preview_limit,
            "required_selected_keys": list(case.expectation.required_selected_keys),
            "forbidden_selected_keys": list(case.expectation.forbidden_selected_keys),
        },
    }


def load_context_eval_cases(path: pathlib.Path | str | None = None) -> List[ContextEvalCase]:
    """Load replay cases from a JSON fixture file."""
    target = pathlib.Path(path) if path is not None else DEFAULT_CONTEXT_EVAL_CASES_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("context eval fixture must be a list")
    return [context_eval_case_from_dict(item) for item in raw if isinstance(item, dict)]


def load_context_eval_case_payload(path: pathlib.Path | str) -> Dict[str, object]:
    """Load one serialized replay case payload from disk."""
    target = pathlib.Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("context eval payload must be an object")
    return raw


def evaluate_context_case(case: ContextEvalCase) -> ContextEvalResult:
    """Run one context replay case through policy and selection."""
    policy_decision = DEFAULT_CONTEXT_POLICY_PROGRAM.run(case.policy_inputs)
    selection_output = DEFAULT_CONTEXT_SELECTION_PROGRAM.run(
        ContextSelectionInputs(
            phase=case.policy_inputs.phase,
            request_text=case.policy_inputs.request_text,
            candidates=case.selection_candidates,
            max_sections=case.selection_max_sections,
        )
    )

    failed_checks: List[str] = []
    passed_checks = 0
    total_checks = 0

    def check(condition: bool, failure_message: str) -> None:
        nonlocal passed_checks, total_checks
        total_checks += 1
        if condition:
            passed_checks += 1
        else:
            failed_checks.append(failure_message)

    expected = case.expectation
    if expected.retrieve_memory is not None:
        check(
            policy_decision.retrieve_memory == expected.retrieve_memory,
            f"retrieve_memory expected {expected.retrieve_memory} but got {policy_decision.retrieve_memory}",
        )
    if expected.retrieve_workspace_previews is not None:
        check(
            policy_decision.retrieve_workspace_previews == expected.retrieve_workspace_previews,
            (
                "retrieve_workspace_previews expected "
                f"{expected.retrieve_workspace_previews} but got {policy_decision.retrieve_workspace_previews}"
            ),
        )
    if expected.min_memory_limit:
        check(
            int(policy_decision.memory_limit) >= int(expected.min_memory_limit),
            f"memory_limit expected >= {expected.min_memory_limit} but got {policy_decision.memory_limit}",
        )
    if expected.min_workspace_preview_limit:
        check(
            int(policy_decision.workspace_preview_limit) >= int(expected.min_workspace_preview_limit),
            (
                "workspace_preview_limit expected >= "
                f"{expected.min_workspace_preview_limit} but got {policy_decision.workspace_preview_limit}"
            ),
        )

    selected = set(selection_output.selected_keys)
    for key in expected.required_selected_keys:
        check(key in selected, f"required selected key missing: {key}")
    for key in expected.forbidden_selected_keys:
        check(key not in selected, f"forbidden selected key present: {key}")

    score = 1.0 if total_checks == 0 else passed_checks / total_checks
    return ContextEvalResult(
        name=case.name,
        passed=not failed_checks,
        score=round(score, 4),
        passed_checks=passed_checks,
        total_checks=total_checks,
        failed_checks=failed_checks,
        policy_decision=policy_decision,
        selection_output=selection_output,
    )


def summarize_context_eval_results(results: Sequence[ContextEvalResult]) -> Dict[str, object]:
    """Aggregate replay-case results into a compact summary."""
    materialized = list(results)
    total_cases = len(materialized)
    passed_cases = sum(1 for result in materialized if result.passed)
    average_score = (
        round(sum(float(result.score) for result in materialized) / total_cases, 4)
        if total_cases else 0.0
    )
    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "average_score": average_score,
        "failed": [
            {
                "name": result.name,
                "failed_checks": list(result.failed_checks),
            }
            for result in materialized
            if not result.passed
        ],
    }


def replay_captured_context_eval_payload(payload: Dict[str, object], *, source_path: str = "") -> CapturedContextEvalResult:
    """Evaluate one captured replay payload that may include extra capture metadata."""
    case = context_eval_case_from_dict(payload)
    capture = payload.get("capture", {})
    capture_payload = dict(capture) if isinstance(capture, dict) else {}
    capture_payload.setdefault("phase", case.policy_inputs.phase)
    return CapturedContextEvalResult(
        source_path=str(source_path or "").strip(),
        capture=capture_payload,
        result=evaluate_context_case(case),
    )


def find_context_eval_capture_files(
    workspace_root: pathlib.Path | str,
    *,
    limit: int = 100,
) -> List[pathlib.Path]:
    """Return recent captured context-eval payload files from one workspace."""
    root = pathlib.Path(workspace_root)
    capture_dir = root / ".ai" / "context-evals"
    if not capture_dir.exists() or not capture_dir.is_dir():
        return []
    files = [path for path in capture_dir.glob("*.json") if path.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)
    safe_limit = max(1, min(int(limit or 100), 500))
    return files[:safe_limit]


def replay_captured_context_eval_files(paths: Sequence[pathlib.Path | str]) -> List[CapturedContextEvalResult]:
    """Load and evaluate multiple captured replay payload files."""
    results: List[CapturedContextEvalResult] = []
    for path in paths:
        target = pathlib.Path(path)
        try:
            payload = load_context_eval_case_payload(target)
            results.append(replay_captured_context_eval_payload(payload, source_path=str(target)))
        except Exception:
            continue
    return results


def summarize_captured_context_eval_results(results: Sequence[CapturedContextEvalResult]) -> Dict[str, object]:
    """Aggregate replay results from captured workspace payloads."""
    materialized = list(results)
    eval_summary = summarize_context_eval_results([item.result for item in materialized])
    failed_check_counts = Counter(
        failed_check
        for item in materialized
        for failed_check in item.result.failed_checks
    )
    trigger_counts = Counter(
        str(item.capture.get("trigger", "")).strip() or "unknown"
        for item in materialized
    )
    phase_counts = Counter(
        (
            str(item.capture.get("phase", "")).strip()
            or (item.result.name.split("_", 2)[1] if "_" in item.result.name else "unknown")
        )
        for item in materialized
    )
    recent_failures = [
        {
            "source_path": item.source_path,
            "name": item.result.name,
            "score": item.result.score,
            "trigger": str(item.capture.get("trigger", "")).strip() or "unknown",
            "failed_checks": list(item.result.failed_checks),
            "selected_keys": list(item.result.selection_output.selected_keys),
        }
        for item in materialized
        if not item.result.passed
    ][:10]
    return {
        **eval_summary,
        "failed_check_counts": dict(failed_check_counts.most_common(12)),
        "trigger_counts": dict(trigger_counts.most_common()),
        "phase_counts": dict(phase_counts.most_common()),
        "recent_failures": recent_failures,
    }


DEFAULT_CONTEXT_EVAL_CASES: List[ContextEvalCase] = load_context_eval_cases()
