#!/usr/bin/env python3
"""Train a lightweight first-tool-family policy model from workflow traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_router import (
    evaluate_workflow_router,
    extract_tool_policy_training_rows,
    load_jsonl,
    save_router_model,
    train_workflow_router,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight tool-policy helper from workflow_traces.jsonl."
    )
    parser.add_argument(
        "--input",
        default="training/exports/latest/workflow_traces.jsonl",
        help="Path to workflow_traces.jsonl. Defaults to training/exports/latest/workflow_traces.jsonl.",
    )
    parser.add_argument(
        "--output",
        default="data/tool_policy_model.json",
        help="Path to write the trained tool policy model. Defaults to data/tool_policy_model.json.",
    )
    parser.add_argument(
        "--min-token-count",
        type=int,
        default=1,
        help="Discard tokens seen fewer than this many times. Defaults to 1.",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=5000,
        help="Maximum vocabulary size. Defaults to 5000.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing parameter. Defaults to 1.0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Tool policy training dataset not found: {input_path}")

    records = load_jsonl(input_path)
    rows = extract_tool_policy_training_rows(records)
    if not rows:
        raise SystemExit(f"No trainable tool-policy rows were found in {input_path}")

    model = train_workflow_router(
        rows,
        min_token_count=args.min_token_count,
        max_vocab=args.max_vocab,
        alpha=args.alpha,
        label_key="tool_family",
    )
    metrics = evaluate_workflow_router(model, rows)
    model["training_metrics"] = metrics
    model["training_source"] = str(input_path)
    model["task"] = "first_tool_family"

    save_router_model(output_path, model)

    print(f"Trained tool policy model from {input_path}")
    print(f"Output model: {output_path}")
    print(f"training_examples: {model.get('training_examples', 0)}")
    print(f"labels: {', '.join(model.get('labels', []))}")
    print(f"vocabulary_size: {len(model.get('vocabulary', []))}")
    print(f"training_accuracy: {metrics.get('accuracy', 0.0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
