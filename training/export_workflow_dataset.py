#!/usr/bin/env python3
"""Export workflow-aware training datasets from the app's SQLite history."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCHEMA_VERSION = "workflow-export.v1"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export training-ready JSONL datasets from chat.db workflow history."
    )
    parser.add_argument(
        "--db-path",
        default="data/chat.db",
        help="Path to the SQLite database. Defaults to data/chat.db.",
    )
    parser.add_argument(
        "--out-dir",
        default="training/exports/latest",
        help="Directory to write JSONL outputs into. Defaults to training/exports/latest.",
    )
    parser.add_argument(
        "--max-history-messages",
        type=int,
        default=10,
        help="Maximum number of prior messages to include in each example. Defaults to 10.",
    )
    return parser.parse_args()


def parse_json_field(raw: Any, fallback: Any) -> Any:
    if raw is None:
        return fallback
    text = str(raw).strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def normalize_feedback(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"positive", "negative", "neutral"}:
        return text
    return ""


def compact_message(message: Dict[str, Any]) -> Dict[str, str]:
    return {
        "role": str(message.get("role", "")),
        "content": str(message.get("content", "")),
    }


def bounded_history(messages: List[Dict[str, Any]], end_index: int, max_history_messages: int) -> List[Dict[str, str]]:
    start_index = max(0, end_index - max(0, int(max_history_messages)))
    return [compact_message(message) for message in messages[start_index:end_index]]


def load_messages(conn: sqlite3.Connection) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[int, Dict[str, Any]], Dict[int, int]]:
    rows = conn.execute(
        """
        SELECT id, conversation_id, role, content, timestamp, feedback
        FROM messages
        ORDER BY conversation_id ASC, timestamp ASC, id ASC
        """
    ).fetchall()

    by_conversation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_id: Dict[int, Dict[str, Any]] = {}
    position_by_id: Dict[int, int] = {}

    for row in rows:
        message = {
            "id": int(row["id"]),
            "conversation_id": str(row["conversation_id"]),
            "role": str(row["role"] or ""),
            "content": str(row["content"] or ""),
            "timestamp": str(row["timestamp"] or ""),
            "feedback": normalize_feedback(row["feedback"]),
        }
        conversation_messages = by_conversation[message["conversation_id"]]
        position_by_id[message["id"]] = len(conversation_messages)
        conversation_messages.append(message)
        by_id[message["id"]] = message

    return dict(by_conversation), by_id, position_by_id


def load_workflow_executions(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            id,
            conversation_id,
            run_id,
            user_message_id,
            assistant_message_id,
            workflow_name,
            workflow_version,
            router_version,
            status,
            started_at,
            ended_at,
            final_outcome,
            error_text,
            user_feedback,
            tool_count,
            artifact_paths_json,
            route_metadata_json
        FROM workflow_executions
        ORDER BY started_at ASC, id ASC
        """
    ).fetchall()

    executions: List[Dict[str, Any]] = []
    for row in rows:
        executions.append({
            "id": str(row["id"]),
            "conversation_id": str(row["conversation_id"]),
            "run_id": str(row["run_id"]),
            "user_message_id": int(row["user_message_id"]) if row["user_message_id"] is not None else None,
            "assistant_message_id": int(row["assistant_message_id"]) if row["assistant_message_id"] is not None else None,
            "workflow_name": str(row["workflow_name"] or ""),
            "workflow_version": str(row["workflow_version"] or ""),
            "router_version": str(row["router_version"] or ""),
            "status": str(row["status"] or ""),
            "started_at": str(row["started_at"] or ""),
            "ended_at": str(row["ended_at"] or ""),
            "final_outcome": str(row["final_outcome"] or ""),
            "error_text": str(row["error_text"] or ""),
            "user_feedback": normalize_feedback(row["user_feedback"]),
            "tool_count": int(row["tool_count"] or 0),
            "artifact_paths": parse_json_field(row["artifact_paths_json"], []),
            "route_metadata": parse_json_field(row["route_metadata_json"], {}),
        })
    return executions


def load_workflow_steps(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT
            execution_id,
            step_index,
            step_name,
            tool_name,
            arguments_json,
            result_ok,
            result_summary,
            result_json,
            latency_ms,
            auto_generated,
            created_at
        FROM workflow_steps
        ORDER BY execution_id ASC, step_index ASC, id ASC
        """
    ).fetchall()

    steps_by_execution: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        steps_by_execution[str(row["execution_id"])].append({
            "step_index": int(row["step_index"] or 0),
            "step_name": str(row["step_name"] or ""),
            "tool_name": str(row["tool_name"] or ""),
            "arguments": parse_json_field(row["arguments_json"], {}),
            "result_ok": bool(int(row["result_ok"] or 0)),
            "result_summary": str(row["result_summary"] or ""),
            "result": parse_json_field(row["result_json"], {}),
            "latency_ms": int(row["latency_ms"] or 0),
            "auto_generated": bool(int(row["auto_generated"] or 0)),
            "created_at": str(row["created_at"] or ""),
        })
    return dict(steps_by_execution)


def load_workflow_evaluations(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT
            execution_id,
            evaluator,
            metric,
            score,
            passed,
            notes_json,
            created_at
        FROM workflow_evaluations
        ORDER BY execution_id ASC, created_at ASC, id ASC
        """
    ).fetchall()

    evaluations_by_execution: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        evaluations_by_execution[str(row["execution_id"])].append({
            "evaluator": str(row["evaluator"] or ""),
            "metric": str(row["metric"] or ""),
            "score": row["score"],
            "passed": bool(int(row["passed"] or 0)),
            "notes": parse_json_field(row["notes_json"], {}),
            "created_at": str(row["created_at"] or ""),
        })
    return dict(evaluations_by_execution)


def build_execution_maps(executions: Iterable[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    by_user_message_id: Dict[int, Dict[str, Any]] = {}
    by_assistant_message_id: Dict[int, Dict[str, Any]] = {}

    for execution in executions:
        user_message_id = execution.get("user_message_id")
        assistant_message_id = execution.get("assistant_message_id")
        if isinstance(user_message_id, int):
            by_user_message_id[user_message_id] = execution
        if isinstance(assistant_message_id, int):
            by_assistant_message_id[assistant_message_id] = execution

    return by_user_message_id, by_assistant_message_id


def build_conversation_sft_examples(
    messages_by_conversation: Dict[str, List[Dict[str, Any]]],
    execution_by_assistant_message_id: Dict[int, Dict[str, Any]],
    max_history_messages: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    for conversation_id, messages in messages_by_conversation.items():
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            execution = execution_by_assistant_message_id.get(message["id"])
            history = bounded_history(messages, index, max_history_messages)
            if not history:
                continue

            examples.append({
                "example_id": f"conversation-sft:{message['id']}",
                "schema_version": SCHEMA_VERSION,
                "dataset_kind": "conversation_sft",
                "conversation_id": conversation_id,
                "assistant_message_id": message["id"],
                "messages": history,
                "completion": message["content"],
                "metadata": {
                    "assistant_feedback": message.get("feedback", ""),
                    "workflow_execution_id": execution.get("id", "") if execution else "",
                    "workflow_name": execution.get("workflow_name", "") if execution else "",
                    "final_outcome": execution.get("final_outcome", "") if execution else "",
                    "user_feedback": execution.get("user_feedback", "") if execution else "",
                    "tool_count": execution.get("tool_count", 0) if execution else 0,
                },
            })

    return examples


def build_workflow_trace_examples(
    executions: List[Dict[str, Any]],
    steps_by_execution: Dict[str, List[Dict[str, Any]]],
    evaluations_by_execution: Dict[str, List[Dict[str, Any]]],
    messages_by_id: Dict[int, Dict[str, Any]],
    messages_by_conversation: Dict[str, List[Dict[str, Any]]],
    position_by_id: Dict[int, int],
    max_history_messages: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    for execution in executions:
        user_message_id = execution.get("user_message_id")
        assistant_message_id = execution.get("assistant_message_id")
        conversation_id = execution.get("conversation_id", "")
        conversation_messages = messages_by_conversation.get(conversation_id, [])
        user_message = messages_by_id.get(user_message_id) if isinstance(user_message_id, int) else None
        assistant_message = messages_by_id.get(assistant_message_id) if isinstance(assistant_message_id, int) else None

        history: List[Dict[str, str]] = []
        if user_message and user_message_id in position_by_id:
            history = bounded_history(
                conversation_messages,
                position_by_id[user_message_id],
                max_history_messages,
            )

        examples.append({
            "example_id": f"workflow-trace:{execution['id']}",
            "schema_version": SCHEMA_VERSION,
            "dataset_kind": "workflow_trace",
            "execution_id": execution["id"],
            "conversation_id": conversation_id,
            "workflow_name": execution.get("workflow_name", ""),
            "workflow_version": execution.get("workflow_version", ""),
            "router_version": execution.get("router_version", ""),
            "status": execution.get("status", ""),
            "final_outcome": execution.get("final_outcome", ""),
            "user_feedback": execution.get("user_feedback", ""),
            "tool_count": execution.get("tool_count", 0),
            "route_metadata": execution.get("route_metadata", {}),
            "artifact_paths": execution.get("artifact_paths", []),
            "history": history,
            "user_message": compact_message(user_message) if user_message else None,
            "assistant_message": compact_message(assistant_message) if assistant_message else None,
            "tool_steps": steps_by_execution.get(execution["id"], []),
            "evaluations": evaluations_by_execution.get(execution["id"], []),
            "error_text": execution.get("error_text", ""),
            "started_at": execution.get("started_at", ""),
            "ended_at": execution.get("ended_at", ""),
        })

    return examples


def build_evaluation_examples(
    executions: List[Dict[str, Any]],
    evaluations_by_execution: Dict[str, List[Dict[str, Any]]],
    messages_by_id: Dict[int, Dict[str, Any]],
    messages_by_conversation: Dict[str, List[Dict[str, Any]]],
    position_by_id: Dict[int, int],
    max_history_messages: int,
) -> List[Dict[str, Any]]:
    execution_by_id = {execution["id"]: execution for execution in executions}
    examples: List[Dict[str, Any]] = []

    for execution_id, evaluations in evaluations_by_execution.items():
        execution = execution_by_id.get(execution_id)
        if not execution:
            continue

        user_message_id = execution.get("user_message_id")
        if not isinstance(user_message_id, int):
            continue
        user_message = messages_by_id.get(user_message_id)
        if not user_message:
            continue

        conversation_id = execution.get("conversation_id", "")
        conversation_messages = messages_by_conversation.get(conversation_id, [])
        history = bounded_history(
            conversation_messages,
            position_by_id.get(user_message_id, 0),
            max_history_messages,
        )

        for eval_index, evaluation in enumerate(evaluations, start=1):
            examples.append({
                "example_id": f"evaluation:{execution_id}:{eval_index}",
                "schema_version": SCHEMA_VERSION,
                "dataset_kind": "evaluation_example",
                "execution_id": execution_id,
                "conversation_id": conversation_id,
                "workflow_name": execution.get("workflow_name", ""),
                "input": {
                    "history": history,
                    "user_message": user_message["content"],
                    "final_outcome": execution.get("final_outcome", ""),
                },
                "label": evaluation,
                "metadata": {
                    "user_feedback": execution.get("user_feedback", ""),
                    "tool_count": execution.get("tool_count", 0),
                },
            })

    return examples


def build_router_examples(
    executions: List[Dict[str, Any]],
    messages_by_id: Dict[int, Dict[str, Any]],
    messages_by_conversation: Dict[str, List[Dict[str, Any]]],
    position_by_id: Dict[int, int],
    max_history_messages: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    for execution in executions:
        user_message_id = execution.get("user_message_id")
        if not isinstance(user_message_id, int):
            continue

        user_message = messages_by_id.get(user_message_id)
        if not user_message:
            continue

        conversation_id = execution.get("conversation_id", "")
        conversation_messages = messages_by_conversation.get(conversation_id, [])
        history = bounded_history(
            conversation_messages,
            position_by_id.get(user_message_id, 0),
            max_history_messages,
        )
        route_metadata = execution.get("route_metadata", {})

        examples.append({
            "example_id": f"router:{execution['id']}",
            "schema_version": SCHEMA_VERSION,
            "dataset_kind": "router_example",
            "input": {
                "history": history,
                "user_message": user_message["content"],
            },
            "label": {
                "workflow_name": execution.get("workflow_name", ""),
                "mode": str(route_metadata.get("mode", "")),
                "workspace_intent": str(route_metadata.get("workspace_intent", "")),
                "enabled_tools": route_metadata.get("enabled_tools", []),
                "auto_execute_workspace": bool(route_metadata.get("auto_execute_workspace", False)),
            },
            "metadata": {
                "execution_id": execution["id"],
                "conversation_id": conversation_id,
                "user_feedback": execution.get("user_feedback", ""),
                "final_outcome": execution.get("final_outcome", ""),
            },
        })

    return examples


def build_preference_examples(
    messages_by_conversation: Dict[str, List[Dict[str, Any]]],
    execution_by_assistant_message_id: Dict[int, Dict[str, Any]],
    max_history_messages: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    for conversation_id, messages in messages_by_conversation.items():
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            execution = execution_by_assistant_message_id.get(message["id"])
            labels = [
                label
                for label in (
                    message.get("feedback", ""),
                    execution.get("user_feedback", "") if execution else "",
                )
                if label in {"positive", "negative"}
            ]
            if not labels:
                continue

            history = bounded_history(messages, index, max_history_messages)
            if not history:
                continue

            examples.append({
                "example_id": f"preference:{message['id']}",
                "schema_version": SCHEMA_VERSION,
                "dataset_kind": "preference_vote",
                "conversation_id": conversation_id,
                "assistant_message_id": message["id"],
                "messages": history,
                "candidate": message["content"],
                "label": labels[0],
                "metadata": {
                    "assistant_feedback": message.get("feedback", ""),
                    "workflow_feedback": execution.get("user_feedback", "") if execution else "",
                    "workflow_execution_id": execution.get("id", "") if execution else "",
                    "workflow_name": execution.get("workflow_name", "") if execution else "",
                },
            })

    return examples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(
    path: Path,
    *,
    db_path: Path,
    max_history_messages: int,
    table_counts: Dict[str, int],
    dataset_counts: Dict[str, int],
) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "db_path": str(db_path),
        "config": {
            "max_history_messages": max_history_messages,
        },
        "table_counts": table_counts,
        "dataset_counts": dataset_counts,
        "notes": [
            "Conversation SFT is derived from assistant turns and bounded prior history.",
            "Workflow traces are extracted from workflow_executions and workflow_steps.",
            "Router examples map request context to the observed workflow route.",
            "Preference votes only exist when assistant or workflow feedback is positive or negative.",
            "This export is suitable for workflow learning, but not sufficient on its own for full RLM training.",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_table(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return int(row["count"] or 0)


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        messages_by_conversation, messages_by_id, position_by_id = load_messages(conn)
        executions = load_workflow_executions(conn)
        steps_by_execution = load_workflow_steps(conn)
        evaluations_by_execution = load_workflow_evaluations(conn)
        _execution_by_user_message_id, execution_by_assistant_message_id = build_execution_maps(executions)

        conversation_sft = build_conversation_sft_examples(
            messages_by_conversation,
            execution_by_assistant_message_id,
            args.max_history_messages,
        )
        workflow_traces = build_workflow_trace_examples(
            executions,
            steps_by_execution,
            evaluations_by_execution,
            messages_by_id,
            messages_by_conversation,
            position_by_id,
            args.max_history_messages,
        )
        evaluation_examples = build_evaluation_examples(
            executions,
            evaluations_by_execution,
            messages_by_id,
            messages_by_conversation,
            position_by_id,
            args.max_history_messages,
        )
        router_examples = build_router_examples(
            executions,
            messages_by_id,
            messages_by_conversation,
            position_by_id,
            args.max_history_messages,
        )
        preference_votes = build_preference_examples(
            messages_by_conversation,
            execution_by_assistant_message_id,
            args.max_history_messages,
        )

        write_jsonl(out_dir / "conversation_sft.jsonl", conversation_sft)
        write_jsonl(out_dir / "workflow_traces.jsonl", workflow_traces)
        write_jsonl(out_dir / "evaluation_examples.jsonl", evaluation_examples)
        write_jsonl(out_dir / "router_examples.jsonl", router_examples)
        write_jsonl(out_dir / "preference_votes.jsonl", preference_votes)

        table_counts = {
            "messages": count_table(conn, "messages"),
            "workflow_executions": count_table(conn, "workflow_executions"),
            "workflow_steps": count_table(conn, "workflow_steps"),
            "workflow_evaluations": count_table(conn, "workflow_evaluations"),
        }
        dataset_counts = {
            "conversation_sft": len(conversation_sft),
            "workflow_traces": len(workflow_traces),
            "evaluation_examples": len(evaluation_examples),
            "router_examples": len(router_examples),
            "preference_votes": len(preference_votes),
        }
        write_manifest(
            out_dir / "manifest.json",
            db_path=db_path,
            max_history_messages=args.max_history_messages,
            table_counts=table_counts,
            dataset_counts=dataset_counts,
        )
    finally:
        conn.close()

    print(f"Exported training datasets from {db_path}")
    print(f"Output directory: {out_dir}")
    print(f"conversation_sft: {len(conversation_sft)}")
    print(f"workflow_traces: {len(workflow_traces)}")
    print(f"evaluation_examples: {len(evaluation_examples)}")
    print(f"router_examples: {len(router_examples)}")
    print(f"preference_votes: {len(preference_votes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
