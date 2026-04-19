# AI Chat with vLLM

A self-hosted coding companion on **vLLM** (OpenAI-compatible API), with a small FastAPI + vanilla JS front end.

## Source Layout

- `src/python/harness.py` — main backend harness (FastAPI + WS + tool loop)
- `src/python/ai_chat/` — backend helper modules (prompts, routing, themes, readers)
- `src/web/` — frontend web app (`index.html`, `app.js`, `style.css`, `assets/`)
- `app.py` — compatibility entrypoint used by Docker and `./chat`

## Features

- **Streaming Responses** — Real-time token streaming via WebSocket (including optional “thinking” regions for supported models)
- **Context window** — History trimming with relevance scoring (tune model length in Compose / vLLM)
- **Chat search** — Search past messages in SQLite
- **Unified Skill Loop** — Each turn now gets an explicit RAG/search/file/code/plan/review assessment before the app chooses a one-shot skill, tool loop, or full inspect/plan/execute/verify run
- **Agent Harness** — Shared tool loop plus deep-mode orchestration for inspect/plan/execute/verify flows
- **Workspace Tools** — Per-turn file reads, patches, command execution, spreadsheet inspection, local RAG, and optional web search
- **Workspace UI** — Activity timeline, workspace browser, inline file viewer/editor, and downloads
- **Attachments** — Upload files into the conversation workspace and reuse them across turns
- **Voice** — Browser-recorded audio attachments plus server-side reply playback when native TTS is available; optional STT API remains available for direct clients
- **Markdown** — Full rendering with syntax highlighting
- **System Prompt** — Customizable per session
- **Model Controls** — Reasoning effort toggle and model availability/status feedback
- **Light/Dark Mode** — Theme support
- **Message Feedback** — Good/Bad response feedback with neutral clearing, used for context ranking

## System Requirements

The default stack is aimed at a local machine that can run vLLM with an NVIDIA GPU.

- **OS / runtime** — macOS, Linux, or another host that can run modern Docker and the `docker compose` plugin. The helper script also expects `bash` and `curl`.
- **GPU** — NVIDIA GPU with CUDA support plus the NVIDIA container runtime/toolkit available to Docker.
- **VRAM** — `24GB` recommended for the default `14B` profile in [`docker-compose.yml`](/Users/joe/dev/ai-chat/docker-compose.yml). `12GB` can work if you switch to the lighter `8B` profile described in [docs/configuration.md](/Users/joe/dev/ai-chat/docs/configuration.md).
- **System RAM** — enough to comfortably run Docker, the chat app, and model-loading overhead alongside your desktop workload. More headroom helps when downloading models and working with larger workspaces.
- **Disk** — enough free SSD space for Docker images, the Hugging Face model cache, and app data. Plan for tens of GB rather than a minimal install.
- **Network** — internet access is needed for first-time image pulls, model downloads, and optional web search.
- **Browser** — a current Chromium, Firefox, or Safari-class browser for the web UI.

## Prerequisites

- Docker with `docker compose`
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit or equivalent Docker GPU integration

## Install

```bash
./chat install
```

This pulls the vLLM image and builds the chat app container.

## Security defaults

The app keeps the runtime surface fairly small:

- No in-app Docker runtime controls
- No interactive PTY-backed terminal
- No legacy execute-code endpoint
- Workspace commands are approved per chat by executable name, then remembered for later turns in that chat only

## Usage

```bash
./chat start     # Start the app (idempotent)
./chat stop      # Stop the app (idempotent)
./chat restart   # Stop then start
./chat status    # Check what's running
./chat logs      # Tail logs from all services
./chat url       # Print the preferred browser URL
./chat open      # Open the preferred browser URL
```

Once started, the launcher prints the best URL to open first:

- Tailscale HTTPS when `tailscale serve` is active
- Tailscale IP when Tailscale is available
- `http://localhost:8000` as the local fallback

If you prefer a trusted HTTPS URL inside your tailnet, run `tailscale serve --bg 8000` and then use `./chat open`.

Conversation workspaces are stored per run under [`runs/`](/Users/joe/dev/ai-chat/runs) on the host via the default `/app/runs` bind mount, so files the assistant creates remain visible outside the container.

## Local Mac development

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

Notes:

- `fastembed` runs on CPU, so semantic retrieval works fine on a Mac without CUDA.
- The checked-in `docker-compose.yml` is mainly for the bundled local `vllm` + app stack and assumes NVIDIA GPU support for the `vllm` service.
- `./chat install` now downloads only the model you choose instead of pulling both local presets.
- `./chat start` prompts for which downloaded model to run for the current session.

## Documentation

- [Configuration](docs/configuration.md) — Model settings, GPU tuning, Docker services, where to edit prompts/themes
- [API](docs/api.md) — REST and WebSocket payloads
- [Architecture](docs/architecture.md) — Main modules and high-level data flow
- [Capability Playbook](docs/capabilities.md) — How to add new agent capabilities with docs, approvals, installs, and workflow entry points
- [Harness And Tools](docs/harness.md) — Tool loop, deep-mode orchestration, permissions, and workspace model
- [UI Features](docs/ui.md) — Document editor, workspace panel, and slash command behaviors
