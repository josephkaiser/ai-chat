# AI Chat

AI Chat is a self-hosted coding companion with:

- a FastAPI backend
- a lightweight browser UI
- a local `./chat` launcher for install, rebuild, and run workflows
- a workspace-aware agent runtime with durable artifacts, replay triage, and tool execution

https://github.com/user-attachments/assets/8c4a06f3-d863-4d93-8e5d-dbc6f137a9e0

## Quick Start

After cloning the repo, `./chat` is the primary way to install and run the app:

```bash
./chat install
./chat start
./chat open
```

What these do:

- `./chat install` pulls the vLLM image, builds the app container, rebuilds the frontend bundle when needed, and prepares the configured model.
- `./chat start` starts the stack using the current model tune from your config.
- `./chat open` opens the best available browser URL.

Useful commands:

```bash
./chat install
./chat start
./chat kickstart
./chat stop
./chat restart
./chat status
./chat logs
./chat url
./chat open
```

`./chat` also supports chained commands like `./chat install start` and `./chat stop install start`.

## Helper Mode (default)

The app now boots a lean, all-purpose helper (`src/python/helper/`, UI in `src/helper_web/`) on
**Gemma 4 26B-A4B** (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) with the model's full **262,144-token** context
on a single 24 GB GPU (fp8 KV cache, image input enabled).

- Drop files or folders anywhere on the page, use 📎, or paste. Big pastes become attached files; pasted screenshots go to the model as images.
- Reads text/code, PDF (per page), DOCX, XLSX (per sheet), ZIP (text files inside; `node_modules`, `.git`, etc. skipped) and images.
- Checked files in the Files panel stay in context every turn. Copy buttons on every reply and code block, plus Save as `.md`, Retry and Edit.
- Each chat has its own `memory.md` (editable in the right panel). A new chat starts with a blank memory.
- Context handling per turn: everything that fits goes in verbatim, ordered to reuse vLLM's prefix cache. Older turns are summarised into `memory.md` once the conversation exceeds about 35% of the window. Files that don't fit are read in up to 64k-token passes; relevant notes (with line markers for citations) are saved to `memory.md` and the answer is written from them. `memory.md` is condensed when it grows past its budget.
- Chat data lives in `data/helper/chats/<id>/` (`chat.json`, `memory.md`, `files/`).

Tuning via env on `chat-app`: `HELPER_CHUNK_TOKENS_MAX`, `HELPER_HISTORY_SHARE`, `HELPER_MEMORY_TOKENS_MAX`,
`HELPER_REPLY_RESERVE`, `HELPER_VISION`, `HELPER_TEMPERATURE`. Set `LEGACY_HARNESS=1` to serve the previous
workspace/coding harness instead.

## System Requirements

The default setup is aimed at a machine that can run vLLM locally.

- Docker with the `docker compose` plugin
- NVIDIA GPU support available to Docker
- About 24 GB VRAM recommended for the default checked-in tune
- Enough SSD space for Docker images, model cache, and app data
- Internet access for initial image pulls, model downloads, and optional web search
- A modern browser for the UI

## What’s In This Repo

```text
.
├── app.py                  # compatibility entrypoint
├── chat                    # main local launcher
├── docs/                   # product, architecture, and runtime docs
├── scripts/                # build helpers
├── src/python/             # backend and runtime implementation
├── src/web/                # browser UI
├── tests/                  # automated coverage
├── Dockerfile
├── docker-compose.yml
└── README.md
```

Runtime state such as `data/` and workspace-generated `.ai/` artifacts is intentionally ignored so the repository stays clean for contributors and public browsing.

## Features

- Streaming chat over WebSocket, with HTTP fallback support
- Workspace browser with previews for text, Markdown, HTML, images, PDFs, CSV, archives, and spreadsheets
- Scoped tool execution for file reads, patches, commands, conversation recall, and optional web search
- Deep inspect/plan/execute/verify flows for repo work
- Durable file sessions, job summaries, and background polishing
- Replay triage driven by retries and negative feedback
- SQLite-backed persistence for conversations, workspaces, feedback, and file-session state

## Local Development

If you want to run the app without the bundled local vLLM stack, point it at a remote OpenAI-compatible server and start it with either:

```bash
npm start
```

or:

```bash
python3 app.py
```

after setting the relevant environment variables described in [docs/configuration.md](/Users/joe/dev/ai-chat/docs/configuration.md).

## Documentation

- [Docs Index](docs/README.md)
- [Configuration](docs/configuration.md)
- [API](docs/api.md)
- [Architecture](docs/architecture.md)
- [Capability Playbook](docs/capabilities.md)
- [Harness And Tools](docs/harness.md)
- [UI Features](docs/ui.md)
- [Roadmaps](docs/roadmaps/README.md)
