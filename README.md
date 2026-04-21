# AI Chat with vLLM

A self-hosted coding companion with a FastAPI backend, a lightweight workspace UI, and a `./chat` launcher for install, rebuild, and run workflows.

## Video Demo

<video src='assets/demo.mov'>

## Quick Start with `./chat`

After cloning the repo, `./chat` is the primary way to install and run the app:

```bash
./chat install
./chat start
./chat open
```

What these do:

- `./chat install` pulls the vLLM image, builds the app container, rebuilds the frontend bundle when needed, and prepares the default model.
- `./chat start` starts the stack using the installed/default model profile.
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

## System Requirements

The default setup is aimed at a machine that can run vLLM locally.

- Docker with the `docker compose` plugin
- NVIDIA GPU support available to Docker
- About 24 GB VRAM recommended for the default `14B` profile
- Enough SSD space for Docker images, model cache, and app data
- Internet access for initial image pulls, model downloads, and optional web search
- A modern browser for the UI

## Features

- Streaming chat over WebSocket, with HTTP fallback support
- Workspace browser with previews for text, Markdown, HTML, images, PDFs, CSV, archives, and spreadsheets
- Scoped tool execution for file reads, patches, commands, conversation recall, and optional web search
- Deep inspect/plan/execute/verify flows for repo work
- Durable file sessions, job summaries, and background polishing
- Replay triage driven by retries and negative feedback
- SQLite-backed persistence for conversations, workspaces, feedback, and file-session state

## Notes

- The launcher rebuilds `src/web/app.js` from `src/web/app.ts` when needed.
- `./chat kickstart` is the clean rebuild path for a freshly synced repo.
- Workspace roots are path-backed catalog entries; legacy hosted runs may still exist under [`runs/`](/Users/joe/dev/ai-chat/runs).

## Local Development

If you want to run the app without the bundled local vLLM stack, you can point it at a remote OpenAI-compatible server and start it with:

```bash
npm start
```

or:

```bash
python3 app.py
```

after setting the relevant environment variables described in [docs/configuration.md](/Users/joe/dev/ai-chat/docs/configuration.md).

## Documentation

- [Configuration](docs/configuration.md)
- [API](docs/api.md)
- [Architecture](docs/architecture.md)
- [Capability Playbook](docs/capabilities.md)
- [Harness And Tools](docs/harness.md)
- [UI Features](docs/ui.md)
