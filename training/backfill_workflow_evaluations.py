#!/usr/bin/env python3
"""Backfill deterministic workflow evaluations for existing executions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


AUTO_EVALUATOR = "workflow_autoeval_v1"
FEEDBACK_EVALUATOR = "workflow_feedback_v1"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill workflow_evaluations from existing workflow_executions."
    )
    parser.add_argument(
        "--db-path",
        default="data/chat.db",
        help="Path to the SQLite database. Defaults to data/chat.db.",
    )
    return parser.parse_args()


def parse_json_field(raw: Any, fallback: Any) -> Any:
    text = str(raw or "").strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return "{}"


def normalize_feedback(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"positive", "negative", "neutral"}:
        return text
    return "neutral"


def expected_final_outcome_for_workflow(workflow_name: str) -> str:
    normalized = str(workflow_name or "").strip()
    if normalized == "deep_orchestrated":
        return "completed_deep"
    if normalized.startswith("slash_"):
        return "completed_slash"
    if normalized == "direct_answer":
        return "completed_direct"
    if normalized.startswith("normal_") and "tool_loop" in normalized:
        return "completed_with_tools"
    return "completed_direct"


def workflow_steps(conn: sqlite3.Connection, execution_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT step_index, step_name, tool_name, result_ok, result_summary, result_json, auto_generated
        FROM workflow_steps
        WHERE execution_id = ?
        ORDER BY step_index ASC, id ASC
        """,
        (execution_id,),
    ).fetchall()
    return [
        {
            "step_index": int(row["step_index"] or 0),
            "step_name": str(row["step_name"] or ""),
            "tool_name": str(row["tool_name"] or ""),
            "result_ok": bool(int(row["result_ok"] or 0)),
            "result_summary": str(row["result_summary"] or ""),
            "result": parse_json_field(row["result_json"], {}),
            "auto_generated": bool(int(row["auto_generated"] or 0)),
        }
        for row in rows
    ]


def build_auto_evaluations(execution: sqlite3.Row, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    successful_steps = [step for step in steps if step.get("result_ok")]
    failed_steps = [step for step in steps if not step.get("result_ok")]
    successful_write_steps = [
        step for step in successful_steps
        if step.get("tool_name") in {"workspace.patch_file", "workspace.render"}
    ]
    artifact_paths = list(parse_json_field(execution["artifact_paths_json"], []))
    nontrivial_artifacts = [
        path for path in artifact_paths
        if str(path).strip() and str(path).strip() not in {".", "./"}
    ]
    last_successful_write_index = max(
        (step.get("step_index", 0) for step in successful_write_steps),
        default=0,
    )
    verification_after_write = any(
        step.get("step_index", 0) > last_successful_write_index
        and step.get("tool_name") in {
            "workspace.list_files", "workspace.grep", "workspace.read_file",
            "workspace.run_command", "spreadsheet.describe",
            "conversation.search_history", "web.search", "web.fetch_page",
        }
        for step in steps
    )

    normalized_status = str(execution["status"] or "").strip()
    normalized_outcome = str(execution["final_outcome"] or "").strip()
    error_text = str(execution["error_text"] or "").strip()
    tool_success_rate = (len(successful_steps) / max(len(steps), 1)) if steps else 1.0
    completion_passed = normalized_status == "completed" and not error_text
    route_alignment_passed = (
        normalized_status != "completed"
        or normalized_outcome == expected_final_outcome_for_workflow(execution["workflow_name"])
    )
    write_grounding_passed = True if not successful_write_steps else bool(nontrivial_artifacts)
    verification_passed = True if not successful_write_steps else verification_after_write

    return [
        {
            "metric": "completion_status",
            "score": 1.0 if completion_passed else 0.0,
            "passed": completion_passed,
            "notes": {
                "status": normalized_status,
                "final_outcome": normalized_outcome,
                "assistant_message_id": execution["assistant_message_id"],
                "error_text": error_text[:500],
            },
        },
        {
            "metric": "route_alignment",
            "score": 1.0 if route_alignment_passed else 0.0,
            "passed": route_alignment_passed,
            "notes": {
                "workflow_name": str(execution["workflow_name"] or ""),
                "expected_final_outcome": expected_final_outcome_for_workflow(execution["workflow_name"]),
                "observed_final_outcome": normalized_outcome,
            },
        },
        {
            "metric": "tool_success_rate",
            "score": round(tool_success_rate, 4),
            "passed": tool_success_rate >= 0.4,
            "notes": {
                "tool_count": len(steps),
                "successful_steps": len(successful_steps),
                "failed_steps": len(failed_steps),
                "failed_tool_names": [step.get("tool_name", "") for step in failed_steps[:8]],
            },
        },
        {
            "metric": "write_grounding",
            "score": 1.0 if write_grounding_passed else 0.0,
            "passed": write_grounding_passed,
            "notes": {
                "successful_write_steps": len(successful_write_steps),
                "artifact_paths": artifact_paths,
                "nontrivial_artifacts": nontrivial_artifacts,
            },
        },
        {
            "metric": "post_write_verification",
            "score": 1.0 if verification_passed else 0.0,
            "passed": verification_passed,
            "notes": {
                "successful_write_steps": len(successful_write_steps),
                "last_successful_write_index": last_successful_write_index,
                "verification_after_write": verification_after_write,
            },
        },
    ]


def build_feedback_evaluation(feedback: str) -> List[Dict[str, Any]]:
    normalized_feedback = normalize_feedback(feedback)
    score_map = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
    return [{
        "metric": "user_feedback",
        "score": score_map.get(normalized_feedback, 0.5),
        "passed": normalized_feedback != "negative",
        "notes": {"feedback": normalized_feedback},
    }]


def replace_evaluator_rows(
    conn: sqlite3.Connection,
    execution_id: str,
    evaluator: str,
    evaluations: List[Dict[str, Any]],
) -> None:
    conn.execute(
        "DELETE FROM workflow_evaluations WHERE execution_id = ? AND evaluator = ?",
        (execution_id, evaluator),
    )
    for evaluation in evaluations:
        conn.execute(
            """
            INSERT INTO workflow_evaluations
            (execution_id, evaluator, metric, score, passed, notes_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                evaluator,
                str(evaluation.get("metric", "")).strip(),
                evaluation.get("score"),
                1 if evaluation.get("passed") else 0,
                safe_json_dumps(evaluation.get("notes", {})),
                utcnow_iso(),
            ),
        )


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    executions = conn.execute(
        """
        SELECT
            id,
            assistant_message_id,
            workflow_name,
            final_outcome,
            status,
            error_text,
            artifact_paths_json,
            user_feedback
        FROM workflow_executions
        ORDER BY started_at ASC, id ASC
        """
    ).fetchall()

    refreshed = 0
    feedback_synced = 0
    for execution in executions:
        steps = workflow_steps(conn, str(execution["id"]))
        replace_evaluator_rows(
            conn,
            str(execution["id"]),
            AUTO_EVALUATOR,
            build_auto_evaluations(execution, steps),
        )
        refreshed += 1

        if execution["assistant_message_id"] is not None:
            replace_evaluator_rows(
                conn,
                str(execution["id"]),
                FEEDBACK_EVALUATOR,
                build_feedback_evaluation(execution["user_feedback"]),
            )
            feedback_synced += 1

    conn.commit()
    conn.close()

    print(f"Backfilled deterministic evaluations for {refreshed} workflow executions")
    print(f"Mirrored feedback evaluations for {feedback_synced} assistant messages")
    print(f"Database: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
