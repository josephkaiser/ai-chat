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
- **Voice** — Server-side transcription and reply playback when native STT/TTS tools are available
- **Markdown** — Full rendering with syntax highlighting
- **System Prompt** — Customizable per session
- **Model Controls** — Reasoning effort toggle plus switchable model profiles
- **Light/Dark Mode** — Theme support
- **Dashboard** — Model status, cache info, restart, and redownload controls
- **Message Feedback** — Thumbs up/down on responses, used for context ranking
- **Log Viewer** — Real-time terminal logs via WebSocket

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support

## Install

```bash
./chat install
```

This pulls the vLLM image and builds the chat app container.

## Security defaults

The default Docker setup now starts in a more locked-down posture:

- Docker-socket-backed runtime control is disabled unless you explicitly enable `DOCKER_CONTROL_ENABLED` and mount `/var/run/docker.sock`.
- The interactive workspace terminal is disabled unless you enable `INTERACTIVE_TERMINAL_ENABLED`.
- The legacy `/api/execute-code` endpoint is disabled unless you enable `EXECUTE_CODE_ENABLED`.
- Workspace commands are approved per chat by executable name, then remembered for later turns in that chat only.

See [Configuration](docs/configuration.md) for the hardening flags and the opt-in setup.

## Usage

```bash
./chat start     # Start the app (idempotent)
./chat stop      # Stop the app (idempotent)
./chat restart   # Stop then start
./chat status    # Check what's running
./chat logs      # Tail logs from all services
```

Once started, open **http://localhost:8000**. The model loads in the background on first run.

## Documentation

- [Configuration](docs/configuration.md) — Model settings, GPU tuning, Docker services, where to edit prompts/themes
- [API](docs/api.md) — REST and WebSocket payloads
- [Architecture](docs/architecture.md) — Main modules and high-level data flow
- [Harness And Tools](docs/harness.md) — Tool loop, deep-mode orchestration, permissions, and workspace model
- [UI Features](docs/ui.md) — Workspace panel, terminal, attachments, voice, slash commands, and dashboard behaviors
