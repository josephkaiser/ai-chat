#!/usr/bin/env python3
"""Shared training and inference helpers for the lightweight workflow router."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROUTER_MODEL_SCHEMA_VERSION = "workflow-router-model.v1"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./+-]+")
MIN_TOKEN_LENGTH = 2


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize_router_text(text: str) -> List[str]:
    """Normalize router text into a compact token sequence."""
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(str(text or ""))
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def render_router_input_text(history: Iterable[Dict[str, Any]], user_message: str) -> str:
    """Flatten history and the active request into one classifier input string."""
    lines: List[str] = []
    for message in history or []:
        role = str(message.get("role", "")).strip() or "message"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    cleaned_user_message = str(user_message or "").strip()
    if cleaned_user_message:
        lines.append(f"user: {cleaned_user_message}")
    return "\n".join(lines)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory."""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def extract_router_training_rows(
    records: Iterable[Dict[str, Any]],
    *,
    label_key: str = "workflow_name",
) -> List[Dict[str, Any]]:
    """Normalize exported router examples into rows for training or evaluation."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        input_payload = record.get("input", {}) if isinstance(record.get("input"), dict) else {}
        label_payload = record.get("label", {}) if isinstance(record.get("label"), dict) else {}
        label = str(label_payload.get(label_key, "")).strip()
        user_message = str(input_payload.get("user_message", "")).strip()
        history = input_payload.get("history", [])
        if not label or not isinstance(history, list):
            continue
        rows.append({
            "label": label,
            "user_message": user_message,
            "history": history,
            "text": render_router_input_text(history, user_message),
        })
    return rows


def tool_family_for_tool_name(tool_name: str) -> str:
    """Map a concrete tool name to a coarse policy family."""
    normalized = str(tool_name or "").strip()
    if normalized in {"workspace.list_files", "workspace.grep", "workspace.read_file", "spreadsheet.describe"}:
        return "workspace_read"
    if normalized == "workspace.patch_file":
        return "workspace_write"
    if normalized == "workspace.run_command":
        return "workspace_command"
    if normalized == "workspace.render":
        return "workspace_render"
    if normalized == "conversation.search_history":
        return "history_search"
    if normalized in {"web.search", "web.fetch_page"}:
        return "web_search"
    return "no_tool"


def extract_tool_policy_training_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build tool-policy training rows from exported workflow traces."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        history = record.get("history", [])
        user_message_payload = record.get("user_message")
        tool_steps = record.get("tool_steps", [])
        if not isinstance(history, list) or not isinstance(tool_steps, list):
            continue

        user_message = ""
        if isinstance(user_message_payload, dict):
            user_message = str(user_message_payload.get("content", "")).strip()
        elif isinstance(user_message_payload, str):
            user_message = str(user_message_payload).strip()

        label = "no_tool"
        for step in tool_steps:
            if not isinstance(step, dict):
                continue
            candidate = tool_family_for_tool_name(step.get("tool_name", ""))
            if candidate:
                label = candidate
                break

        rows.append({
            "label": label,
            "user_message": user_message,
            "history": history,
            "text": render_router_input_text(history, user_message),
            "workflow_name": str(record.get("workflow_name", "")).strip(),
            "execution_id": str(record.get("execution_id", "")).strip(),
        })

    return rows


def _logsumexp(values: List[float]) -> float:
    if not values:
        return 0.0
    ceiling = max(values)
    return ceiling + math.log(sum(math.exp(value - ceiling) for value in values))


def train_workflow_router(
    rows: Iterable[Dict[str, Any]],
    *,
    min_token_count: int = 1,
    max_vocab: int = 5000,
    alpha: float = 1.0,
    label_key: str = "workflow_name",
) -> Dict[str, Any]:
    """Train a tiny multinomial Naive Bayes router model."""
    prepared_rows: List[Dict[str, Any]] = []
    global_token_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row in rows:
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        tokens = tokenize_router_text(row.get("text", ""))
        prepared_row = dict(row)
        prepared_row["tokens"] = tokens
        prepared_rows.append(prepared_row)
        global_token_counts.update(tokens)
        label_counts.update([label])

    if not prepared_rows:
        raise ValueError("No router training rows were available.")

    vocab_items = [
        (token, count)
        for token, count in global_token_counts.items()
        if count >= max(1, int(min_token_count))
    ]
    vocab_items.sort(key=lambda item: (-item[1], item[0]))
    if max_vocab > 0:
        vocab_items = vocab_items[:max_vocab]
    vocabulary = [token for token, _ in vocab_items]
    vocabulary_set = set(vocabulary)

    label_token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    label_token_totals: Counter[str] = Counter()

    for row in prepared_rows:
        label = row["label"]
        filtered_tokens = [token for token in row["tokens"] if token in vocabulary_set]
        label_token_counts[label].update(filtered_tokens)
        label_token_totals[label] += len(filtered_tokens)

    total_examples = len(prepared_rows)
    vocab_size = max(len(vocabulary), 1)
    labels = sorted(label_counts.keys())

    log_priors: Dict[str, float] = {}
    default_log_probs: Dict[str, float] = {}
    token_log_probs: Dict[str, Dict[str, float]] = {}

    for label in labels:
        label_example_count = label_counts[label]
        total_tokens_for_label = label_token_totals[label]
        denominator = total_tokens_for_label + (alpha * vocab_size)
        log_priors[label] = math.log(label_example_count / total_examples)
        default_log_probs[label] = math.log(alpha / denominator)
        token_log_probs[label] = {
            token: math.log((count + alpha) / denominator)
            for token, count in label_token_counts[label].items()
        }

    model = {
        "schema_version": ROUTER_MODEL_SCHEMA_VERSION,
        "model_type": "multinomial_naive_bayes",
        "generated_at": utcnow_iso(),
        "label_key": label_key,
        "labels": labels,
        "vocabulary": vocabulary,
        "alpha": float(alpha),
        "min_token_count": int(min_token_count),
        "max_vocab": int(max_vocab),
        "log_priors": log_priors,
        "default_log_probs": default_log_probs,
        "token_log_probs": token_log_probs,
        "label_distribution": dict(label_counts),
        "training_examples": total_examples,
    }
    return model


def predict_workflow_router(
    model: Dict[str, Any],
    history: Iterable[Dict[str, Any]],
    user_message: str,
    *,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Predict the most likely workflow label for a new turn."""
    labels = [str(label) for label in model.get("labels", []) if str(label).strip()]
    if not labels:
        raise ValueError("Router model does not contain any labels.")

    vocabulary = set(str(token) for token in model.get("vocabulary", []))
    token_counts = Counter(
        token for token in tokenize_router_text(render_router_input_text(history, user_message))
        if token in vocabulary
    )

    raw_scores: Dict[str, float] = {}
    for label in labels:
        label_log_probs = model.get("token_log_probs", {}).get(label, {})
        default_log_prob = float(model.get("default_log_probs", {}).get(label, -100.0))
        score = float(model.get("log_priors", {}).get(label, 0.0))
        for token, count in token_counts.items():
            score += count * float(label_log_probs.get(token, default_log_prob))
        raw_scores[label] = score

    normalization = _logsumexp(list(raw_scores.values()))
    probabilities = {
        label: math.exp(score - normalization)
        for label, score in raw_scores.items()
    }
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    best_label, best_confidence = ranked[0]

    return {
        "label": best_label,
        "confidence": best_confidence,
        "top_labels": [
            {"label": label, "confidence": confidence}
            for label, confidence in ranked[:max(1, int(top_k))]
        ],
        "probabilities": probabilities,
        "tokens_used": int(sum(token_counts.values())),
    }


def evaluate_workflow_router(model: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate exact-label accuracy on a set of router rows."""
    total = 0
    correct = 0
    per_label: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for row in rows:
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        prediction = predict_workflow_router(
            model,
            row.get("history", []),
            str(row.get("user_message", "")),
        )
        total += 1
        per_label[label]["total"] += 1
        if prediction.get("label") == label:
            correct += 1
            per_label[label]["correct"] += 1

    accuracy = (correct / total) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_label": per_label,
    }


def save_router_model(path: Path, model: Dict[str, Any]) -> None:
    """Write a trained router model to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_router_model(path: Path) -> Dict[str, Any]:
    """Load a trained router model from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != ROUTER_MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported router model schema: {payload.get('schema_version')!r}"
        )
    return payload
