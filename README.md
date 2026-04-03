# AI Chat with vLLM

A self-hosted coding companion on **vLLM** (OpenAI-compatible API), with a small FastAPI + vanilla JS front end.

## Features

- **Streaming Responses** — Real-time token streaming via WebSocket (including optional “thinking” regions for supported models)
- **Context window** — History trimming with relevance scoring (tune model length in Compose / vLLM)
- **Chat search** — Search past messages in SQLite
- **Agent Harness** — Shared tool loop plus deep-mode orchestration for inspect/plan/execute/verify flows
- **Workspace Tools** — Per-turn file reads, patches, command execution, spreadsheet inspection, local RAG, and optional web search
- **Workspace UI** — Activity timeline, workspace browser, inline file viewer/editor, downloads, and live terminal integration
- **Attachments** — Upload files into the conversation workspace and reuse them across turns
- **Voice** — Browser-recorded audio attachments plus server-side reply playback when native TTS is available; optional STT API remains available for direct clients
- **Markdown** — Full rendering with syntax highlighting
- **System Prompt** — Customizable per session
- **Model Controls** — Reasoning effort toggle, provider-aware model labels, and switchable model profiles
- **Light/Dark Mode** — Theme support
- **Dashboard** — Model status, cache info, restart, and redownload controls
- **Message Feedback** — Good/Bad response feedback with neutral clearing, used for context ranking
- **Log Viewer** — Real-time terminal logs via WebSocket

## System Requirements

The default stack is aimed at a local machine that can run vLLM with an NVIDIA GPU.

- **OS / runtime** — macOS, Linux, or another host that can run modern Docker and the `docker compose` plugin. The helper script also expects `bash` and `curl`.
- **GPU** — NVIDIA GPU with CUDA support plus the NVIDIA container runtime/toolkit available to Docker.
- **VRAM** — `24GB` recommended for the default `14B` profile in [`docker-compose.yml`](/Users/joe/dev/ai-chat/docker-compose.yml). `12GB` can work if you switch to the lighter `8B` profile described in [docs/configuration.md](/Users/joe/dev/ai-chat/docs/configuration.md).
- **System RAM** — enough to comfortably run Docker, the chat app, and model-loading overhead alongside your desktop workload. More headroom helps when downloading models and working with larger workspaces.
- **Disk** — enough free SSD space for Docker images, the Hugging Face model cache, app data, and optional voice models. Plan for tens of GB rather than a minimal install.
- **Network** — internet access is needed for first-time image pulls, model downloads, and optional web search.
- **Browser** — a current Chromium, Firefox, or Safari-class browser for the web UI. Microphone recording needs browser media-capture support.

Optional feature requirements:

- **Server TTS/STT** — the container already installs Piper/Whisper dependencies, but host-native TTS like macOS `say` is only available when you run outside Docker or deliberately expose host tools.
- **Dashboard runtime controls** — the sample local setup mounts the Docker socket so in-app restart/model-management actions work out of the box. If you want a more locked-down deployment, remove that mount and set `DOCKER_CONTROL_ENABLED=false`.
- **Interactive terminal** — disabled by default; enable it only if you want the PTY-backed workspace shell.

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

The sample local Docker setup now favors convenience for a trusted single-user machine:

- Docker-socket-backed runtime control is enabled so frontend model switching, restart, and cached-model activation work out of the box.
- The interactive workspace terminal is disabled unless you enable `INTERACTIVE_TERMINAL_ENABLED`.
- The legacy `/api/execute-code` endpoint is disabled unless you enable `EXECUTE_CODE_ENABLED`.
- Workspace commands are approved per chat by executable name, then remembered for later turns in that chat only.

If you want a stricter deployment, remove the Docker socket mount and set `DOCKER_CONTROL_ENABLED=false`. See [Configuration](docs/configuration.md) for the hardening flags.

## Usage

```bash
./chat start     # Start the app (idempotent)
./chat stop      # Stop the app (idempotent)
./chat restart   # Stop then start
./chat status    # Check what's running
./chat logs      # Tail logs from all services
```

Once started, open **http://localhost:8000**. The model loads in the background on first run.

Conversation workspaces are stored per run under [`runs/`](/Users/joe/dev/ai-chat/runs) on the host via the default `/app/runs` bind mount, so files the assistant creates remain visible outside the container.

## Documentation

- [Configuration](docs/configuration.md) — Model settings, GPU tuning, Docker services, where to edit prompts/themes
- [API](docs/api.md) — REST and WebSocket payloads
- [Architecture](docs/architecture.md) — Main modules and high-level data flow
- [Harness And Tools](docs/harness.md) — Tool loop, deep-mode orchestration, permissions, and workspace model
- [UI Features](docs/ui.md) — Workspace panel, terminal, attachments, voice, slash commands, and dashboard behaviors
