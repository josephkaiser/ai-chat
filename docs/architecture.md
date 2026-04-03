# Architecture

## Project structure

```
ai-chat/
├── app.py               # FastAPI app: routes, WebSockets, SQLite, vLLM client
├── themes.py            # Light/dark palettes → CSS custom properties for the UI
├── prompts.py           # Default system prompt text
├── thinking_stream.py   # Model stream → “thinking” vs visible answer (WebSocket framing)
├── chat                 # CLI to start/stop the stack
├── static/
│   ├── index.html       # Web UI template (Jinja)
│   ├── app.js           # Frontend: WebSocket client, markdown, theme application
│   └── style.css        # Layout and components (colors via CSS variables)
├── docker-compose.yml   # vLLM + chat-app services
├── Dockerfile           # Chat app image (copies the four Python modules above)
├── docs/                # Documentation
└── data/
    └── chat.db          # SQLite database (volume in Docker)
```

## How the pieces connect

- **`app.py`** is the entrypoint. It mounts static files, renders `index.html`, and implements REST + WebSocket handlers. It also now owns the chat harness, workspace APIs, terminal sessions, dashboard endpoints, voice pipeline, tool execution loop, per-conversation runs/workspaces, and workflow-execution persistence.
- **`themes.py`** defines small named palettes (e.g. paper, ink, panel). Those expand to the `--bg_primary`, `--text_secondary`, … keys that **`static/style.css`** already expects. At runtime, `GET /` injects JSON into the page; **`static/app.js`** calls `applyTheme()` to set `document.documentElement` styles.
- **`prompts.py`** holds **`DEFAULT_SYSTEM_PROMPT`** plus the tool-use and execution prompts that shape both normal tool turns and deep-mode planning/build flows.
- **`thinking_stream.py`** implements **`ThinkingStreamSplitter`**: it parses model output for paired thinking tags (must match **`THINK_TAG_PAIRS` in `static/app.js`**) and emits the WebSocket payloads (`think_start`, `think_token`, `think_end`, and normal `token`).
- **`static/app.js`** is now a substantial client runtime, not just a chat socket wrapper. It manages feature toggles, model/profile controls, reasoning mode, attachments, slash commands, workspace activity, the file modal, terminal streaming, dashboard actions, and voice playback/recording.

## Tech stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, httpx (OpenAI-compatible chat completions to vLLM)
- **Frontend:** Vanilla JS, Marked.js, highlight.js (CDN)
- **LLM:** vLLM (OpenAI-compatible API); model name is configured via env / compose (see [Configuration](configuration.md))
- **Database:** SQLite (conversations, messages, conversation summaries, per-conversation runs/workspaces, workflow execution telemetry, pet state, and assistant feedback for history scoring)
- **Streaming:** WebSocket `/ws/chat` for token and thinking-region events
- **Workspace terminal:** WebSocket `/ws/terminal/{conversation_id}` for PTY-backed command output
- **Dashboard / ops:** Optional Docker socket access from the chat container for vLLM restart and cache paths (see `GET /api/dashboard` and related endpoints in [API](api.md))

## Persistence model

- Every conversation gets a stable run id plus a dedicated workspace directory on disk.
- `messages.feedback` stores assistant feedback (`positive`, `negative`, `neutral`), which is surfaced through the conversation API and reused during message reranking.
- `workflow_executions` and `workflow_steps` capture how a turn was routed, which tools ran, what artifacts were produced, and whether any auto-generated verification steps were inserted. The `workflow_evaluations` table exists as a companion slot for later evaluator data.
- Voice artifacts live under `VOICE_ROOT`, outside the conversation workspace, and are cleaned up by cache pruning plus conversation/app reset flows.

## Additional guides

- [Harness And Tools](harness.md) — how the runtime chooses tools, executes them, and reports progress
- [UI Features](ui.md) — the modern chat/workspace UI surface and its major controls
