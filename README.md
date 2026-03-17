# AI Chat with vLLM

A chat application with a web UI, running Qwen 3 8B on vLLM with a single GPU.

## Features

- **Web UI** — Responsive interface with light/dark mode
- **Streaming Responses** — Real-time token streaming via WebSocket
- **32K Context Window** — Smart history management
- **Web Search** — Automatic and manual web search via DuckDuckGo
- **Code Execution** — Sandboxed Python execution
- **File Browsing** — Read files and list directories
- **Chat Search** — Full-text search across all conversations
- **Markdown** — Full markdown rendering with syntax highlighting
- **System Prompt** — Customizable system prompt per session
- **Log Viewer** — Real-time terminal logs

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support

## Install

```bash
./chat install
```

This pulls the vLLM image and builds the chat app container.

## Usage

```bash
./chat start     # Start the app (idempotent — safe to run if already running)
./chat stop      # Stop the app  (idempotent — safe to run if already stopped)
./chat restart   # Stop then start
./chat status    # Check what's running
./chat logs      # Tail logs from all services
```

Once started, open **http://localhost:8000**. The web UI loads immediately — the model loads in the background on first run.

## Documentation

- [Configuration](docs/configuration.md) — Model settings, GPU tuning, Docker services
- [API](docs/api.md) — REST and WebSocket endpoints
- [Architecture](docs/architecture.md) — Project structure and tech stack
