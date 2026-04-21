# Architecture

## Project structure

```text
ai-chat/
├── app.py                           # Compatibility shim that re-exports src/python/harness.py
├── src/python/harness.py            # FastAPI app, WS loop, workspace APIs, tool runtime, file-session worker
├── src/python/ai_chat/
│   ├── routing_program.py           # Top-level routing seam
│   ├── turn_strategy.py             # Heuristic turn assessment
│   ├── runtime_layers.py            # Model-only runtime context layering
│   ├── deep_runtime.py              # Deep-session state + preview/execute lifecycle
│   ├── deep_flow.py                 # Deep execution route decisions
│   ├── task_engine.py               # Typed task runner for structured plans
│   ├── context_eval.py              # Replay capture + triage report logic
│   ├── context_assembler.py         # Retrieval-backed context building
│   ├── context_policy_program.py    # Context budget policy
│   ├── context_selection_program.py # Context candidate selection
│   ├── prompts.py                   # System, tool, and execution prompts
│   ├── thinking_stream.py           # Thinking/output stream parsing
│   └── workspace_reader.py          # Workspace read helpers and permission constants
├── src/web/
│   ├── index.html                   # Web UI shell
│   ├── app.ts                       # Frontend source
│   ├── app.js                       # Generated browser runtime
│   └── style.css                    # Layout and styling
├── chat                             # CLI to install/start/stop the stack
├── docker-compose.yml               # vLLM + chat-app services
├── docs/                            # Prose docs, plans, and design notes
└── data/chat.db                     # SQLite database
```

## How the pieces connect

- `app.py` is no longer the real backend implementation. It exists so Docker and local scripts can still launch the app through a stable entrypoint.
- `src/python/harness.py` is the actual backend. It mounts the frontend, exposes REST and WebSocket routes, persists SQLite state, runs the tool loop, and manages file-session/background-job behavior.
- Runtime code lives under `src/`; `docs/` is intentionally documentation-only.
- `routing_program.py` and `turn_strategy.py` decide whether a request should be answered directly, sent through a scoped tool loop, or upgraded into a deeper inspect/plan/execute flow.
- `runtime_layers.py` keeps model-only context such as active draft/file metadata out of the visible transcript.
- `deep_runtime.py`, `deep_flow.py`, and `task_engine.py` split the older monolithic deep-mode logic into smaller orchestration pieces.
- `context_eval.py` turns retries and negative feedback into replay captures and triage summaries, which the frontend surfaces as the Replay Triage panel.
- `src/web/app.ts` is intentionally small: it manages the workspace selector, replay triage, chat stream, file browser, and file preview UI.

## Main runtime model

- A **workspace** is a first-class catalog row backed by a real root path on disk.
- A **conversation** is the visible chat transcript attached to one workspace.
- A **file session** is durable per-target-file state that can outlive one visible turn.
- A **file-session job** records foreground or background work for that file.
- A **background focus** gives one file per workspace ownership of the background polish loop at a time.

## Persistence model

- SQLite stores conversations, messages, workspace rows, file sessions, file-session jobs, model state, and assistant feedback.
- Workspace files live on disk in their actual root directories, not only inside database blobs.
- File-session internals are stored as hidden workspace artifacts under `.ai-chat/`, including draft/spec files, candidate outputs, evaluations, and historical versions.
- Replay/context-eval captures live under `.ai/context-evals/` so failures can be inspected from the workspace browser.
- Managed Python environments live outside the workspace tree so installs can persist without polluting syncs.

## Tech stack

- Backend: Python 3.11, FastAPI, Uvicorn, httpx, SQLite
- Frontend: TypeScript source compiled into a checked-in browser bundle
- Model runtime: vLLM through an OpenAI-compatible API
- Transport: WebSocket `/ws/chat` plus buffered HTTP fallback at `/api/chat`

## Additional guides

- [Harness And Tools](harness.md) — routing, deep execution, tools, approvals, and background jobs
- [UI Features](ui.md) — the shipped workspace shell and replay triage UI
- [API](api.md) — HTTP and WebSocket surface
