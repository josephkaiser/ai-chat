# Training Scaffold

This folder is the first step toward making the repo learn from its own `chat.db` workflow history.

The current app already records useful supervision signals in SQLite:

- `messages` for turn-by-turn chat transcripts
- `workflow_executions` for routing and final workflow outcomes
- `workflow_steps` for tool traces
- `workflow_evaluations` for future evaluator scores
- `conversation_summaries` for compressed long-context state

That is enough to start training workflow-aware models in this repo without creating a separate project yet.

## What This Exporter Produces

Run the exporter and it writes JSONL datasets under `training/exports/latest` by default:

- `conversation_sft.jsonl`
  Assistant-turn supervised finetuning examples built from bounded chat history.
- `workflow_traces.jsonl`
  End-to-end workflow records with route metadata, tool steps, and final answers.
- `router_examples.jsonl`
  Inputs and labels for training a lightweight router to pick the right workflow.
- `evaluation_examples.jsonl`
  Evaluator rows from `workflow_evaluations`, ready for rubric or judge-model training when that table starts filling up.
- `preference_votes.jsonl`
  Positive or negative assistant/workflow feedback examples when those signals exist.
- `manifest.json`
  Export metadata, schema version, and dataset counts.

## Quick Start

```bash
./chat learn

# or run the pieces manually:
./chat export-training
./chat backfill-evaluations
./chat train-router
./chat train-tool-policy
```

Or call the exporter directly:

```bash
python3 training/export_workflow_dataset.py \
  --db-path data/chat.db \
  --out-dir training/exports/latest

python3 training/train_router.py \
  --input training/exports/latest/router_examples.jsonl \
  --output data/router_model.json

python3 training/train_tool_policy.py \
  --input training/exports/latest/workflow_traces.jsonl \
  --output data/tool_policy_model.json
```

## Why This Helps

This gives us a practical first learning loop:

1. Export supervision from real chats and workflow telemetry.
2. Finetune a small model for routing, planning style, or workflow imitation.
3. Hook that model back into the harness as a router, planner helper, or evaluator.
4. Keep collecting better traces and repeat.

The repo now includes the first hook-back paths for runtime learning:

- a lightweight learned workflow router trained from `router_examples.jsonl`
- a lightweight learned read-only tool advisor trained from `workflow_traces.jsonl`

Both are optional and can be enabled conservatively behind env flags.

## One-Command Refresh

If you just want the latest database turned into fresh helper models, use:

```bash
./chat learn
```

That command currently runs:

1. `./chat backfill-evaluations`
2. `./chat export-training`
3. `./chat train-router`
4. `./chat train-tool-policy`

This is realistic for the current lightweight in-repo learning loop.
It is not the same thing as full base-model finetuning or paper-style RLM training.

## What It Does Not Solve Yet

This is not a full paper-style RLM training pipeline by itself.

The current database does not log:

- prompt-object reads and slices
- recursive `sub_RLM(...)` style calls
- symbolic prompt operations inside a dedicated runtime

So this scaffold is strong enough for workflow learning and tool-policy learning, but true RLM training will also need new runtime instrumentation or synthetic recursive traces.

## Suggested Next Steps

After this export scaffold, the natural next steps are:

1. Train a small router on `router_examples.jsonl`.
2. Train a workflow imitation model on `workflow_traces.jsonl`.
3. Add evaluator writes into `workflow_evaluations`.
4. Instrument an RLM-lite runtime and export recursive traces into a new dataset family.
