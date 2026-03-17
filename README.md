# AI Chat with vLLM

A self-hosted chat application running Qwen 3 8B on vLLM with a single GPU.

## Features

- **Streaming Responses** — Real-time token streaming via WebSocket
- **32K Context** — Smart history management with relevance scoring
- **Web Search** — Automatic and manual search via DuckDuckGo
- **Code Execution** — Sandboxed Python execution
- **File Browsing** — Read files and list directories
- **Chat Search** — Full-text search across all conversations
- **Markdown** — Full rendering with syntax highlighting
- **System Prompt** — Customizable per session
- **Light/Dark Mode** — Theme support
- **Dashboard** — Model status, cache info, vLLM restart controls
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

- [Configuration](docs/configuration.md) — Model settings, GPU tuning, Docker services
- [API](docs/api.md) — REST and WebSocket endpoints
- [Architecture](docs/architecture.md) — Project structure and tech stack
