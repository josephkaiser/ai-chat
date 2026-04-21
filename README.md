# AI Chat with vLLM

A self-hosted coding companion on **vLLM** (OpenAI-compatible API), with a FastAPI backend, a small TypeScript workspace UI, and a deeper server-side harness for repo work.

## Quick Start with `./chat`

After cloning the repo, `./chat` is the primary way to install and run the app:

```bash
./chat install
./chat start
./chat open
```

- `./chat install` pulls the vLLM image, builds the app container, and rebuilds the checked-in frontend bundle when needed.
- `./chat start` launches the stack using the installed/default model profile.
- `./chat open` opens the best available browser URL.

Useful commands:

```bash
./chat install   # Build images and prepare the default model
./chat start     # Start the app
./chat kickstart # Clean rebuild from the current repo snapshot, then start
./chat stop      # Stop the app
./chat restart   # Stop then start
./chat status    # Check what's running
./chat logs      # Tail logs from all services
./chat url       # Print the preferred browser URL
./chat open      # Open the preferred browser URL
```

`./chat start` reuses the installed/default model non-interactively, so chained commands like `./chat install start` and `./chat stop install start` work as expected.

## Source Layout

- `src/python/harness.py` — main FastAPI app, WebSocket loop, workspace APIs, tool loop, and file-session runtime
- `src/python/ai_chat/` — routing, deep-runtime orchestration, prompts, context selection, replay triage, and helpers
- `src/web/` — frontend web app (`index.html`, `app.ts`, generated `app.js`, `style.css`)
- `docs/` — prose documentation, plans, and design notes
- `app.py` — compatibility entrypoint used by Docker and `./chat`

## Features

- **Streaming chat** — token-by-token assistant replies over WebSocket, with HTTP fallback for buffered clients
- **Path-backed workspaces** — a shared workspace catalog keyed by real filesystem roots
- **Workspace browser and previews** — browse files and preview text, Markdown, HTML, images, PDFs, CSV, archives, and spreadsheets
- **Deep harness** — turn routing plus inspect/plan/execute/verify flows for repo work
- **Tool execution** — scoped file reads, patches, commands, spreadsheet inspection, conversation recall, and optional web search
- **Durable file sessions** — each target file can carry hidden draft/spec artifacts, versions, job summaries, and background polishing work
- **Replay triage** — retries and negative feedback can capture context-eval cases, summarized in the sidebar replay report
- **SQLite-backed memory** — conversations, feedback, workspace metadata, file sessions, and job state persist across restarts

## Recent Refactors

- **Backend split** — the real server now lives in `src/python/harness.py`; `app.py` is just a compatibility shim.
- **Orchestration extraction** — deep planning and execution logic moved into `src/python/ai_chat/` modules such as `deep_runtime.py`, `deep_flow.py`, `routing_program.py`, and `task_engine.py`.
- **Workspace identity cleanup** — workspaces are first-class catalog entries instead of being purely conversation-owned scratch directories.
- **File-session runtime** — durable per-file state, hidden `.ai-chat/` artifacts, foreground/background job records, and a background polish worker now back the document-style flows.
- **Replay report loop** — corrective feedback now feeds `.ai/context-evals/` captures and the in-app triage report for debugging routing/context failures.
- **Frontend simplification** — the shipped UI is now a cleaner workspace/chat/file-viewer shell with replay triage, while the backend still supports richer structured events.

## Frontend build

The served browser bundle lives at `src/web/app.js`, generated from `src/web/app.ts`.

```bash
npm run build:frontend
```

`./chat install`, `./chat start`, and local `npm start` rebuild that bundle automatically when the TypeScript source is newer than the generated browser file.

## System Requirements

The default stack is aimed at a local machine that can run vLLM with an NVIDIA GPU.

- **OS / runtime** — macOS, Linux, or another host that can run modern Docker and the `docker compose` plugin
- **GPU** — NVIDIA GPU with CUDA support plus the NVIDIA container runtime/toolkit available to Docker
- **VRAM** — `24GB` recommended for the default `14B` profile in [`docker-compose.yml`](/Users/joe/dev/ai-chat/docker-compose.yml)
- **Disk** — enough free SSD space for Docker images, Hugging Face model cache, and app data
- **Network** — internet access is needed for first-time image pulls, model downloads, and optional web search
- **Browser** — a current Chromium, Firefox, or Safari-class browser for the UI

## Runtime Defaults

The shipped app keeps the runtime surface fairly small:

- no in-browser Docker controls
- no PTY-backed terminal surface
- no legacy execute-code endpoint
- command execution stays workspace-scoped and argv-based on the backend

Once started, the launcher prints the best URL to open first:

- Tailscale HTTPS when `tailscale serve` is active
- Tailscale IP when Tailscale is available
- `http://localhost:8000` as the local fallback

If you prefer a trusted HTTPS URL inside your tailnet, run `tailscale serve --bg 8000` and then use `./chat open`.

Workspace roots are path-backed catalog entries. Legacy hosted runs may still exist under [`runs/`](/Users/joe/dev/ai-chat/runs), but the current model is a shared workspace catalog plus visible files on disk.

## Local Mac Development

If you develop on macOS but run inference on another machine, you do not need the full Docker stack locally.

Create a virtual environment and install the app dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the app locally while pointing it at your remote OpenAI-compatible server:

```bash
export VLLM_HOST=http://YOUR-SERVER:8000/v1
export DB_PATH=./data/chat.db
export MODEL_STATE_PATH=./data/model_state.json
export HF_CACHE_PATH=$HOME/.cache/huggingface
export RUNS_ROOT=./runs
export WORKSPACE_ROOT=./workspaces
python3 app.py
```

You can also use `npm start`, which rebuilds the frontend bundle and then launches `python3 app.py`.

Notes:

- `fastembed` runs on CPU, so semantic retrieval works fine on a Mac without CUDA.
- The checked-in `docker-compose.yml` is mainly for the bundled local `vllm` + app stack and assumes NVIDIA GPU support for the `vllm` service.
- `./chat install` downloads the current/default model profile automatically instead of prompting.
- `./chat start` reuses the installed/default model non-interactively so it can be chained after `install` or `stop`.

## Documentation

- [Configuration](docs/configuration.md) — model settings, launcher behavior, and runtime knobs
- [API](docs/api.md) — REST and WebSocket surface
- [Architecture](docs/architecture.md) — main modules and high-level data flow
- [Capability Playbook](docs/capabilities.md) — how to add new agent capabilities with docs, approvals, installs, and workflow entry points
- [Harness And Tools](docs/harness.md) — routing, deep execution, tools, approvals, and background jobs
- [UI Features](docs/ui.md) — current workspace shell, replay triage, chat flow, and file previews
