"""Replayable evaluation harness for context policy and selection behavior."""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

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
DEFAULT_CONTEXT_EVAL_CASES_DIR = (
    pathlib.Path(__file__).resolve().parent / "fixtures" / "context_eval_cases.d"
)
ACTIVE_CONTEXT_EVAL_REVIEW_STATUSES = ("", "candidate", "accepted")


@dataclass(frozen=True)
class ContextEvalFixtureMetadata:
    """Review metadata for one promoted context-eval fixture."""

    review_status: str = "candidate"
    promoted_at: str = ""
    updated_at: str = ""
    source_path: str = ""
    source_trigger: str = ""
    source_conversation_id: str = ""


def normalize_context_eval_review_status(value: str) -> str:
    """Normalize one fixture review status."""
    normalized = str(value or "").strip().lower()
    if normalized in {"suggested", "candidate", "accepted", "superseded"}:
        return normalized
    return "candidate"


def context_eval_fixture_metadata_from_dict(payload: Dict[str, object]) -> ContextEvalFixtureMetadata:
    """Normalize review metadata embedded in one fixture payload."""
    return ContextEvalFixtureMetadata(
        review_status=normalize_context_eval_review_status(str(payload.get("review_status", "candidate"))),
        promoted_at=str(payload.get("promoted_at", "")).strip(),
        updated_at=str(payload.get("updated_at", "")).strip(),
        source_path=str(payload.get("source_path", "")).strip(),
        source_trigger=str(payload.get("source_trigger", "")).strip(),
        source_conversation_id=str(payload.get("source_conversation_id", "")).strip(),
    )


def context_eval_fixture_metadata_from_payload(payload: Dict[str, object]) -> ContextEvalFixtureMetadata:
    """Extract fixture review metadata from one payload, if present."""
    raw = payload.get("fixture_metadata", {})
    if not isinstance(raw, dict):
        return ContextEvalFixtureMetadata()
    return context_eval_fixture_metadata_from_dict(raw)


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


def slugify_context_eval_case_name(name: str) -> str:
    """Return one filesystem-safe case name."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    return cleaned or "captured_context_eval_case"


def promoted_context_eval_case_path(
    case_name: str,
    *,
    fixtures_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Return the target path for one promoted context-eval fixture."""
    target_dir = pathlib.Path(fixtures_dir) if fixtures_dir is not None else DEFAULT_CONTEXT_EVAL_CASES_DIR
    return target_dir / f"{slugify_context_eval_case_name(case_name)}.json"


def normalize_promoted_context_eval_case_payload(
    payload: Dict[str, object],
    *,
    fixture_name: str = "",
    review_status: str = "candidate",
) -> Dict[str, object]:
    """Normalize one captured replay payload into a fixture payload."""
    case = context_eval_case_from_dict(payload)
    normalized_name = str(fixture_name or case.name).strip() or case.name
    normalized_case = ContextEvalCase(
        name=normalized_name,
        policy_inputs=case.policy_inputs,
        selection_candidates=case.selection_candidates,
        selection_max_sections=case.selection_max_sections,
        expectation=case.expectation,
    )
    capture = payload.get("capture", {})
    capture_payload = dict(capture) if isinstance(capture, dict) else {}
    normalized = serialize_context_eval_case(normalized_case)
    normalized["fixture_metadata"] = {
        "review_status": normalize_context_eval_review_status(review_status),
        "promoted_at": str(payload.get("promoted_at") or "").strip(),
        "updated_at": str(payload.get("promoted_at") or "").strip(),
        "source_path": str(payload.get("source_path") or "").strip(),
        "source_trigger": str(capture_payload.get("trigger", "")).strip(),
        "source_conversation_id": str(capture_payload.get("conversation_id", "")).strip(),
    }
    return normalized


def write_promoted_context_eval_case(
    payload: Dict[str, object],
    *,
    fixture_name: str = "",
    review_status: str = "candidate",
    fixtures_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Persist one normalized replay case into the promoted fixture directory."""
    normalized = normalize_promoted_context_eval_case_payload(
        payload,
        fixture_name=fixture_name,
        review_status=review_status,
    )
    target = promoted_context_eval_case_path(str(normalized.get("name", "")), fixtures_dir=fixtures_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def suggested_context_eval_case_name(bucket_key: str, *, phase: str = "") -> str:
    """Return one deterministic name for an auto-drafted replay fixture."""
    segments = ["suggested"]
    cleaned_bucket = str(bucket_key or "").strip().lower()
    if cleaned_bucket:
        segments.append(cleaned_bucket)
    return "_".join(segments)


def ensure_suggested_context_eval_case(
    payload: Dict[str, object],
    *,
    bucket_key: str,
    fixture_name: str = "",
    fixtures_dir: pathlib.Path | str | None = None,
) -> tuple[pathlib.Path, bool]:
    """Create one suggested replay fixture if it does not already exist."""
    capture = payload.get("capture", {})
    capture_payload = dict(capture) if isinstance(capture, dict) else {}
    target_name = (
        str(fixture_name or "").strip()
        or suggested_context_eval_case_name(
            bucket_key,
            phase=str(capture_payload.get("phase", "")).strip(),
        )
    )
    target = promoted_context_eval_case_path(target_name, fixtures_dir=fixtures_dir)
    if target.exists():
        return target, False
    normalized = normalize_promoted_context_eval_case_payload(
        payload,
        fixture_name=target_name,
        review_status="suggested",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target, True


def _load_context_eval_case_items_from_file(
    path: pathlib.Path,
    *,
    include_review_statuses: Optional[Sequence[str]] = ACTIVE_CONTEXT_EVAL_REVIEW_STATUSES,
) -> List[ContextEvalCase]:
    """Load one or many replay cases from a JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed_statuses = None if include_review_statuses is None else {
        normalize_context_eval_review_status(status) if status else ""
        for status in include_review_statuses
    }

    def include_payload(item: Dict[str, object]) -> bool:
        metadata = context_eval_fixture_metadata_from_payload(item)
        review_status = metadata.review_status if str(item.get("fixture_metadata") or "").strip() else ""
        if allowed_statuses is None:
            return True
        return review_status in allowed_statuses

    if isinstance(raw, list):
        return [
            context_eval_case_from_dict(item)
            for item in raw
            if isinstance(item, dict) and include_payload(item)
        ]
    if isinstance(raw, dict):
        if not include_payload(raw):
            return []
        return [context_eval_case_from_dict(raw)]
    raise ValueError("context eval fixture must be an object or list")


def load_context_eval_cases(
    path: pathlib.Path | str | None = None,
    *,
    include_review_statuses: Optional[Sequence[str]] = ACTIVE_CONTEXT_EVAL_REVIEW_STATUSES,
) -> List[ContextEvalCase]:
    """Load replay cases from a JSON fixture file."""
    if path is None:
        items = _load_context_eval_case_items_from_file(
            DEFAULT_CONTEXT_EVAL_CASES_PATH,
            include_review_statuses=include_review_statuses,
        )
        if DEFAULT_CONTEXT_EVAL_CASES_DIR.exists():
            for target in sorted(DEFAULT_CONTEXT_EVAL_CASES_DIR.glob("*.json")):
                items.extend(
                    _load_context_eval_case_items_from_file(
                        target,
                        include_review_statuses=include_review_statuses,
                    )
                )
        return items

    target = pathlib.Path(path)
    if target.is_dir():
        items: List[ContextEvalCase] = []
        for child in sorted(target.glob("*.json")):
            items.extend(
                _load_context_eval_case_items_from_file(
                    child,
                    include_review_statuses=include_review_statuses,
                )
            )
        return items
    return _load_context_eval_case_items_from_file(target, include_review_statuses=include_review_statuses)


def load_context_eval_case_payload(path: pathlib.Path | str) -> Dict[str, object]:
    """Load one serialized replay case payload from disk."""
    target = pathlib.Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("context eval payload must be an object")
    return raw


def list_promoted_context_eval_fixtures(
    fixtures_dir: pathlib.Path | str | None = None,
) -> List[Dict[str, object]]:
    """Return promoted fixture metadata for review surfaces."""
    root = pathlib.Path(fixtures_dir) if fixtures_dir is not None else DEFAULT_CONTEXT_EVAL_CASES_DIR
    if not root.exists() or not root.is_dir():
        return []
    items: List[Dict[str, object]] = []
    status_rank = {"suggested": 0, "candidate": 1, "accepted": 2, "superseded": 3}
    for target in sorted(root.glob("*.json")):
        try:
            payload = load_context_eval_case_payload(target)
            metadata = context_eval_fixture_metadata_from_payload(payload)
            items.append({
                "name": str(payload.get("name") or "").strip(),
                "path": str(target),
                "review_status": metadata.review_status,
                "promoted_at": metadata.promoted_at,
                "updated_at": metadata.updated_at,
                "source_path": metadata.source_path,
                "source_trigger": metadata.source_trigger,
                "source_conversation_id": metadata.source_conversation_id,
            })
        except Exception:
            continue
    items.sort(
        key=lambda item: (
            -int(status_rank.get(str(item.get("review_status") or ""), 99)),
            str(item.get("updated_at") or item.get("promoted_at") or ""),
            str(item.get("name") or ""),
        ),
        reverse=True,
    )
    return items


def resolve_promoted_context_eval_fixture_path(
    fixture_path: pathlib.Path | str,
    *,
    fixtures_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Validate and resolve one promoted fixture path."""
    target = pathlib.Path(fixture_path).expanduser()
    allowed_root = (pathlib.Path(fixtures_dir) if fixtures_dir is not None else DEFAULT_CONTEXT_EVAL_CASES_DIR).resolve()
    resolved_target = target.resolve()
    if allowed_root not in resolved_target.parents:
        raise ValueError("Fixture path must be inside the promoted fixture directory")
    if not resolved_target.exists() or not resolved_target.is_file():
        raise ValueError("Fixture file not found")
    return resolved_target


def load_promoted_context_eval_fixture_detail(
    fixture_path: pathlib.Path | str,
    *,
    fixtures_dir: pathlib.Path | str | None = None,
) -> Dict[str, object]:
    """Load one promoted fixture with review metadata for detail surfaces."""
    resolved_target = resolve_promoted_context_eval_fixture_path(fixture_path, fixtures_dir=fixtures_dir)
    payload = load_context_eval_case_payload(resolved_target)
    metadata = context_eval_fixture_metadata_from_payload(payload)
    return {
        "name": str(payload.get("name") or "").strip(),
        "path": str(resolved_target),
        "review_status": metadata.review_status,
        "promoted_at": metadata.promoted_at,
        "updated_at": metadata.updated_at,
        "source_path": metadata.source_path,
        "source_trigger": metadata.source_trigger,
        "source_conversation_id": metadata.source_conversation_id,
        "payload": payload,
    }


def update_promoted_context_eval_fixture_review_state(
    fixture_path: pathlib.Path | str,
    review_status: str,
    *,
    fixtures_dir: pathlib.Path | str | None = None,
    updated_at: str = "",
) -> pathlib.Path:
    """Update the review state for one promoted fixture."""
    resolved_target = resolve_promoted_context_eval_fixture_path(fixture_path, fixtures_dir=fixtures_dir)
    payload = load_context_eval_case_payload(resolved_target)
    metadata = context_eval_fixture_metadata_from_payload(payload)
    payload["fixture_metadata"] = {
        "review_status": normalize_context_eval_review_status(review_status),
        "promoted_at": metadata.promoted_at,
        "updated_at": str(updated_at or metadata.updated_at or metadata.promoted_at).strip(),
        "source_path": metadata.source_path,
        "source_trigger": metadata.source_trigger,
        "source_conversation_id": metadata.source_conversation_id,
    }
    resolved_target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved_target


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


def _humanize_context_key(key: str) -> str:
    """Convert an internal context key into a compact label."""
    return str(key or "").replace("_", " ").strip() or "unknown context"


def _recommendation_for_context_key(key: str, *, missing: bool) -> str:
    """Return a short fix recommendation for one context key."""
    normalized = str(key or "").strip()
    if normalized == "workspace_excerpts":
        return (
            "Promote workspace excerpts earlier in retrieval and selection for planning and verification turns."
            if missing
            else "Downrank workspace excerpts when the phase should stay conversational instead of file-grounded."
        )
    if normalized == "conversation_context":
        return (
            "Retain recent visible-chat context for follow-up turns that depend on user intent continuity."
            if missing
            else "Downrank chat history for evidence-heavy phases so file and artifact context win."
        )
    if normalized == "conversation_memory":
        return (
            "Promote compacted conversation memory earlier so short follow-ups and long-running chats do not depend on raw transcript replay."
            if missing
            else "Only keep compacted conversation memory when it materially helps the active turn instead of crowding out stronger evidence."
        )
    if normalized == "retrieved_history":
        return (
            "Increase memory retrieval or lower the selection threshold for relevant prior chat history."
            if missing
            else "Limit retrieved history in phases that should focus on workspace evidence."
        )
    if normalized == "step_reports":
        return (
            "Pull step reports into the candidate set before verify or synthesize so execution evidence is available."
            if missing
            else "Only keep step reports when the phase is evaluating work that actually produced them."
        )
    if normalized == "scope_gaps":
        return (
            "Preserve scope-gap evidence during planning and verification when open issues are present."
            if missing
            else "Drop scope-gap context when the request is a direct answer rather than a status review."
        )
    if normalized == "changed_files":
        return (
            "Prioritize changed-file evidence when summarizing or verifying recent edits."
            if missing
            else "Avoid changed-file context when no concrete file work happened in the turn."
        )
    return (
        f"Raise the rank of {_humanize_context_key(normalized)} in relevant phases."
        if missing
        else f"Lower the rank of {_humanize_context_key(normalized)} when it distracts from the active phase."
    )


def _triage_bucket_for_failed_check(failed_check: str) -> Dict[str, object]:
    """Map a failed replay check into a ranked triage bucket."""
    text = str(failed_check or "").strip()

    if text.startswith("required selected key missing: "):
        key = text.split(": ", 1)[1].strip()
        return {
            "bucket_key": f"missing_selected:{key}",
            "category": "selection_missing_key",
            "title": f"Missing context: {_humanize_context_key(key)}",
            "severity": "high",
            "severity_rank": 4,
            "recommendation": _recommendation_for_context_key(key, missing=True),
        }

    if text.startswith("forbidden selected key present: "):
        key = text.split(": ", 1)[1].strip()
        return {
            "bucket_key": f"forbidden_selected:{key}",
            "category": "selection_forbidden_key",
            "title": f"Wrong context kept: {_humanize_context_key(key)}",
            "severity": "high",
            "severity_rank": 4,
            "recommendation": _recommendation_for_context_key(key, missing=False),
        }

    retrieval_match = re.match(r"^(retrieve_[a-z_]+) expected (True|False) but got (True|False)$", text)
    if retrieval_match:
        field_name, expected, actual = retrieval_match.groups()
        is_workspace = field_name == "retrieve_workspace_previews"
        return {
            "bucket_key": f"policy:{field_name}:{expected}->{actual}",
            "category": "policy_retrieval",
            "title": (
                "Workspace retrieval policy mismatch"
                if is_workspace else "Memory retrieval policy mismatch"
            ),
            "severity": "high" if is_workspace else "medium",
            "severity_rank": 4 if is_workspace else 3,
            "recommendation": (
                "Adjust phase policy so file-grounded phases request workspace previews."
                if is_workspace
                else "Adjust phase policy so follow-up turns pull chat memory when intent depends on prior context."
            ),
        }

    limit_match = re.match(r"^(memory_limit|workspace_preview_limit) expected >= (\d+) but got (\d+)$", text)
    if limit_match:
        field_name, expected, actual = limit_match.groups()
        is_workspace = field_name == "workspace_preview_limit"
        return {
            "bucket_key": f"limit:{field_name}:{expected}->{actual}",
            "category": "policy_limit",
            "title": (
                "Workspace preview depth too low"
                if is_workspace
                else "Memory retrieval depth too low"
            ),
            "severity": "medium",
            "severity_rank": 2,
            "recommendation": (
                "Increase workspace preview depth for evidence-heavy phases so key files are available."
                if is_workspace
                else "Increase memory retrieval depth when short follow-ups need more conversation history."
            ),
        }

    return {
        "bucket_key": f"generic:{text or 'unknown'}",
        "category": "generic",
        "title": "Unclassified replay mismatch",
        "severity": "medium",
        "severity_rank": 1,
        "recommendation": "Review the failed check and add a specific policy or selection rule for it.",
    }


def _bucket_keys_for_context_eval_case(case: ContextEvalCase) -> List[str]:
    """Infer likely triage bucket keys covered by one promoted fixture."""
    keys: List[str] = []
    expectation = case.expectation
    for key in expectation.required_selected_keys:
        cleaned = str(key).strip()
        if cleaned:
            keys.append(f"missing_selected:{cleaned}")
    for key in expectation.forbidden_selected_keys:
        cleaned = str(key).strip()
        if cleaned:
            keys.append(f"forbidden_selected:{cleaned}")
    return keys


def _tool_failure_bucket_meta(
    *,
    lane: str,
    title: str,
    bucket_suffix: str,
    severity: str,
    severity_rank: int,
    recommendation: str,
) -> Dict[str, object]:
    """Build triage metadata for a tool-policy-derived failure bucket."""
    return {
        "bucket_key": f"tool:{lane}:{bucket_suffix}",
        "category": "tool_policy",
        "title": title,
        "severity": severity,
        "severity_rank": severity_rank,
        "recommendation": recommendation,
    }


def _tool_failure_buckets_for_capture(capture: Dict[str, object]) -> List[Dict[str, object]]:
    """Infer triage buckets from captured tool-policy traces."""
    tool_policy = capture.get("tool_policy")
    if not isinstance(tool_policy, dict):
        return []

    tool_names = {
        str(name).strip()
        for name in capture.get("tool_names", [])
        if str(name).strip()
    } if isinstance(capture.get("tool_names"), list) else set()
    workflow_status = str(capture.get("workflow_status") or "").strip().lower()
    error_text = str(capture.get("error_text") or "").strip().lower()

    def tool_used(*prefixes: str) -> bool:
        return any(
            any(name.startswith(prefix) for prefix in prefixes)
            for name in tool_names
        )

    permission_blocked = bool(
        error_text
        and any(
            marker in error_text
            for marker in (
                "permission denied",
                "permission was denied",
                "approval denied",
                "not approved",
                "not allowed",
                "requires approval",
                "denied for this chat",
            )
        )
    ) or workflow_status in {"blocked", "permission_blocked"}

    bucket_metas: List[Dict[str, object]] = []

    def maybe_add_bucket(
        *,
        requested: bool,
        available: bool,
        used: bool,
        lane: str,
        missing_title: str,
        missing_recommendation: str,
        unused_title: str,
        unused_recommendation: str,
        blocked_title: str,
        blocked_recommendation: str,
    ) -> None:
        if not requested:
            return
        if not available:
            bucket_metas.append(
                _tool_failure_bucket_meta(
                    lane=lane,
                    title=missing_title,
                    bucket_suffix="not_offered",
                    severity="high",
                    severity_rank=4,
                    recommendation=missing_recommendation,
                )
            )
            return
        if permission_blocked:
            bucket_metas.append(
                _tool_failure_bucket_meta(
                    lane=lane,
                    title=blocked_title,
                    bucket_suffix="permission_blocked",
                    severity="high",
                    severity_rank=4,
                    recommendation=blocked_recommendation,
                )
            )
            return
        if not used:
            bucket_metas.append(
                _tool_failure_bucket_meta(
                    lane=lane,
                    title=unused_title,
                    bucket_suffix="offered_but_unused",
                    severity="medium",
                    severity_rank=3,
                    recommendation=unused_recommendation,
                )
            )

    maybe_add_bucket(
        requested=bool(tool_policy.get("web_search_requested")),
        available=bool(tool_policy.get("has_web_tools")),
        used=tool_used("web.search", "web.fetch_page"),
        lane="web_search",
        missing_title="Web search was not offered",
        missing_recommendation="Expose web search tools for current-info turns before routing them into a weaker answer-only path.",
        unused_title="Web search was offered but not used",
        unused_recommendation="Tighten tool-choice policy so current-info turns actually invoke web search instead of stopping at a generic answer.",
        blocked_title="Web search was blocked by permissions",
        blocked_recommendation="Make web-search permission handling resumable so the requested lookup can continue after approval.",
    )
    maybe_add_bucket(
        requested=bool(tool_policy.get("workspace_requested") or str(tool_policy.get("workspace_intent") or "").strip().lower() not in {"", "none"}),
        available=bool(tool_policy.get("has_workspace_tools")),
        used=tool_used("workspace.list_files", "workspace.grep", "workspace.read_file", "workspace.patch_file"),
        lane="workspace",
        missing_title="Workspace tools were not offered",
        missing_recommendation="Keep workspace tools enabled when the user is asking to inspect or modify files in the current workspace.",
        unused_title="Workspace tools were offered but not used",
        unused_recommendation="Tighten tool-choice policy so file-grounded turns actually use workspace tools instead of staying in chat-only mode.",
        blocked_title="Workspace tools were blocked by permissions",
        blocked_recommendation="Make workspace-tool approvals resumable so file inspection or edits continue after permission is granted.",
    )
    maybe_add_bucket(
        requested=bool(tool_policy.get("render_requested")),
        available=bool(tool_policy.get("has_render_tools")),
        used=tool_used("workspace.render"),
        lane="render",
        missing_title="Render tools were not offered",
        missing_recommendation="Keep render tools available when the user is asking to preview or open a durable artifact in the viewer.",
        unused_title="Render tools were offered but not used",
        unused_recommendation="Prefer render/open actions when the turn is asking to show or preview an artifact instead of only describing it in chat.",
        blocked_title="Render tools were blocked by permissions",
        blocked_recommendation="Make render permission handling resumable so viewer-oriented turns do not stall after approval.",
    )
    maybe_add_bucket(
        requested=bool(tool_policy.get("local_rag_requested")),
        available=bool(tool_policy.get("has_history_tools")),
        used=tool_used("conversation.search_history"),
        lane="history",
        missing_title="History retrieval was not offered",
        missing_recommendation="Expose conversation-history retrieval when the request depends on prior turns or compacted memory is insufficient.",
        unused_title="History retrieval was offered but not used",
        unused_recommendation="Use history retrieval when follow-up turns depend on prior chat state instead of guessing from the latest message alone.",
        blocked_title="History retrieval was blocked by permissions",
        blocked_recommendation="Make history retrieval resumable when approval gates prevent the follow-up from grounding itself correctly.",
    )
    maybe_add_bucket(
        requested=bool(tool_policy.get("auto_execute_workspace") or tool_policy.get("resume_saved_workspace")),
        available=bool(tool_policy.get("has_execution_tools")),
        used=tool_used("workspace.run_command"),
        lane="execution",
        missing_title="Execution tools were not offered",
        missing_recommendation="Keep command-execution tools available when the turn is explicitly trying to continue or verify saved workspace work.",
        unused_title="Execution tools were offered but not used",
        unused_recommendation="Tighten tool-choice policy so resume/verify turns actually run the expected command instead of stalling in chat.",
        blocked_title="Execution tools were blocked by permissions",
        blocked_recommendation="Make execution approvals resumable so run/test/verify turns can continue after permission is granted.",
    )

    deduped: List[Dict[str, object]] = []
    seen_keys = set()
    for meta in bucket_metas:
        key = str(meta.get("bucket_key") or "")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(meta)
    return deduped


def promoted_fixture_bucket_coverage(
    fixtures_dir: pathlib.Path | str | None = None,
) -> Dict[str, Dict[str, object]]:
    """Return triage coverage metadata keyed by bucket key from promoted fixtures."""
    coverage: Dict[str, Dict[str, object]] = {}
    for fixture in list_promoted_context_eval_fixtures(fixtures_dir):
        try:
            detail = load_promoted_context_eval_fixture_detail(str(fixture.get("path") or ""), fixtures_dir=fixtures_dir)
            payload = detail.get("payload", {})
            if not isinstance(payload, dict):
                continue
            case = context_eval_case_from_dict(payload)
            review_status = normalize_context_eval_review_status(str(detail.get("review_status") or "candidate"))
            for bucket_key in _bucket_keys_for_context_eval_case(case):
                entry = coverage.setdefault(
                    bucket_key,
                    {
                        "bucket_key": bucket_key,
                        "total_fixtures": 0,
                        "suggested_count": 0,
                        "candidate_count": 0,
                        "accepted_count": 0,
                        "superseded_count": 0,
                        "fixtures": [],
                    },
                )
                entry["total_fixtures"] = int(entry["total_fixtures"]) + 1
                if review_status == "accepted":
                    entry["accepted_count"] = int(entry["accepted_count"]) + 1
                elif review_status == "suggested":
                    entry["suggested_count"] = int(entry["suggested_count"]) + 1
                elif review_status == "superseded":
                    entry["superseded_count"] = int(entry["superseded_count"]) + 1
                else:
                    entry["candidate_count"] = int(entry["candidate_count"]) + 1
                fixtures = entry.get("fixtures")
                if isinstance(fixtures, list) and len(fixtures) < 5:
                    fixtures.append(
                        {
                            "name": str(detail.get("name") or ""),
                            "path": str(detail.get("path") or ""),
                            "review_status": review_status,
                        }
                    )
        except Exception:
            continue
    return coverage


def summarize_captured_context_eval_results(results: Sequence[CapturedContextEvalResult]) -> Dict[str, object]:
    """Aggregate replay results from captured workspace payloads."""
    materialized = list(results)
    eval_summary = summarize_context_eval_results([item.result for item in materialized])
    fixture_coverage = promoted_fixture_bucket_coverage()
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
            "tool_policy": dict(item.capture.get("tool_policy", {})) if isinstance(item.capture.get("tool_policy"), dict) else {},
            "tool_names": list(item.capture.get("tool_names", [])) if isinstance(item.capture.get("tool_names"), list) else [],
            "workflow_status": str(item.capture.get("workflow_status", "")).strip(),
            "error_text": str(item.capture.get("error_text", "")).strip(),
        }
        for item in materialized
        if not item.result.passed
    ][:10]

    triage_buckets: Dict[str, Dict[str, object]] = {}
    for item in materialized:
        if item.result.passed:
            continue
        trigger = str(item.capture.get("trigger", "")).strip() or "unknown"
        phase = (
            str(item.capture.get("phase", "")).strip()
            or (item.result.name.split("_", 2)[1] if "_" in item.result.name else "unknown")
        )
        example_case = {
            "name": item.result.name,
            "source_path": item.source_path,
            "trigger": trigger,
            "phase": phase,
            "failed_checks": list(item.result.failed_checks),
            "selected_keys": list(item.result.selection_output.selected_keys),
            "tool_policy": dict(item.capture.get("tool_policy", {})) if isinstance(item.capture.get("tool_policy"), dict) else {},
            "tool_names": list(item.capture.get("tool_names", [])) if isinstance(item.capture.get("tool_names"), list) else [],
            "workflow_status": str(item.capture.get("workflow_status", "")).strip(),
            "error_text": str(item.capture.get("error_text", "")).strip(),
        }
        bucket_metas = [_triage_bucket_for_failed_check(failed_check) for failed_check in item.result.failed_checks]
        bucket_metas.extend(_tool_failure_buckets_for_capture(item.capture))
        for bucket_meta in bucket_metas:
            bucket_key = str(bucket_meta["bucket_key"])
            bucket = triage_buckets.setdefault(
                bucket_key,
                {
                    "key": bucket_key,
                    "category": bucket_meta["category"],
                    "title": bucket_meta["title"],
                    "severity": bucket_meta["severity"],
                    "severity_rank": bucket_meta["severity_rank"],
                    "recommendation": bucket_meta["recommendation"],
                    "failure_count": 0,
                    "case_refs": set(),
                    "trigger_counts": Counter(),
                    "phase_counts": Counter(),
                    "sample_failed_checks": [],
                    "example_cases": [],
                },
            )
            bucket["failure_count"] = int(bucket["failure_count"]) + 1
            bucket["case_refs"].add((item.source_path, item.result.name))
            bucket["trigger_counts"][trigger] += 1
            bucket["phase_counts"][phase] += 1
            sample_label = next(
                (
                    failed_check
                    for failed_check in item.result.failed_checks
                    if str(_triage_bucket_for_failed_check(failed_check).get("bucket_key") or "") == bucket_key
                ),
                str(bucket_meta["title"]),
            )
            if sample_label not in bucket["sample_failed_checks"] and len(bucket["sample_failed_checks"]) < 3:
                bucket["sample_failed_checks"].append(sample_label)
            if len(bucket["example_cases"]) < 3 and all(
                existing.get("source_path") != item.source_path or existing.get("name") != item.result.name
                for existing in bucket["example_cases"]
            ):
                bucket["example_cases"].append(example_case)

    def build_promotion_suggestion(
        bucket: Dict[str, object],
        coverage: Dict[str, object],
        case_count: int,
    ) -> Dict[str, object]:
        accepted_count = int(coverage.get("accepted_count", 0) or 0)
        suggested_count = int(coverage.get("suggested_count", 0) or 0)
        candidate_count = int(coverage.get("candidate_count", 0) or 0)
        failure_count = int(bucket["failure_count"])
        severity = str(bucket["severity"])
        should_suggest = (
            accepted_count == 0
            and suggested_count == 0
            and candidate_count == 0
            and (case_count >= 2 or failure_count >= 3 or severity == "high")
        )
        if case_count >= 2:
            reason = f"Repeated across {case_count} captured cases without fixture coverage."
        elif failure_count >= 3:
            reason = f"Repeated {failure_count} times in replay without fixture coverage."
        elif severity == "high":
            reason = "High-severity failure pattern without accepted or candidate fixture coverage."
        else:
            reason = "No promotion suggestion yet."
        return {
            "should_suggest": should_suggest,
            "reason": reason,
            "suggested_review_status": "candidate",
        }

    ranked_triage_buckets = []
    for bucket in triage_buckets.values():
        case_count = len(bucket["case_refs"])
        priority_score = round(
            float(bucket["failure_count"]) * float(bucket["severity_rank"]) + case_count * 0.25,
            4,
        )
        coverage = fixture_coverage.get(
            bucket["key"],
            {
                "bucket_key": bucket["key"],
                "total_fixtures": 0,
                "suggested_count": 0,
                "candidate_count": 0,
                "accepted_count": 0,
                "superseded_count": 0,
                "fixtures": [],
            },
        )
        ranked_triage_buckets.append(
            {
                "key": bucket["key"],
                "category": bucket["category"],
                "title": bucket["title"],
                "severity": bucket["severity"],
                "recommendation": bucket["recommendation"],
                "failure_count": bucket["failure_count"],
                "case_count": case_count,
                "priority_score": priority_score,
                "trigger_counts": dict(bucket["trigger_counts"].most_common()),
                "phase_counts": dict(bucket["phase_counts"].most_common()),
                "sample_failed_checks": list(bucket["sample_failed_checks"]),
                "example_cases": list(bucket["example_cases"]),
                "fixture_coverage": coverage,
                "promotion_suggestion": build_promotion_suggestion(bucket, coverage, case_count),
            }
        )
    ranked_triage_buckets.sort(
        key=lambda bucket: (
            -float(bucket["priority_score"]),
            -int(bucket["failure_count"]),
            -int(bucket["case_count"]),
            str(bucket["title"]),
        )
    )

    return {
        **eval_summary,
        "failed_check_counts": dict(failed_check_counts.most_common(12)),
        "trigger_counts": dict(trigger_counts.most_common()),
        "phase_counts": dict(phase_counts.most_common()),
        "recent_failures": recent_failures,
        "triage_bucket_count": len(ranked_triage_buckets),
        "top_triage_buckets": ranked_triage_buckets[:8],
        "recommended_fix": ranked_triage_buckets[0] if ranked_triage_buckets else None,
    }


DEFAULT_CONTEXT_EVAL_CASES: List[ContextEvalCase] = load_context_eval_cases()
