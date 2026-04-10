# Architecture

## Project structure

```text
ai-chat/
├── app.py               # FastAPI app: routes, WebSockets, SQLite, vLLM client
├── themes.py            # Light/dark palettes → CSS custom properties
├── prompts.py           # Default system prompt text
├── thinking_stream.py   # Model stream → “thinking” vs visible answer
├── turn_strategy.py     # Turn assessment and top-level routing decisions
├── deep_flow.py         # Deep-mode routing decisions
├── chat                 # CLI to start/stop the stack
├── static/
│   ├── index.html       # Web UI template
│   ├── app.js           # Frontend runtime
│   └── style.css        # Layout and components
├── docker-compose.yml   # vLLM + chat-app services
├── Dockerfile           # Chat app image
├── docs/                # Documentation
└── data/
    └── chat.db          # SQLite database
```

## How the pieces connect

- `app.py` is the entrypoint. It mounts static files, renders `index.html`, and implements REST and WebSocket handlers. It owns the chat harness, workspace APIs, voice pipeline, tool execution loop, and per-conversation workspaces.
- `turn_strategy.py` evaluates each user turn against the app’s main skill loop: local RAG, web search, file creation, coding mode, planning mode, execution mode, and verification needs.
- `deep_flow.py` decides what the deep execution pipeline should do next when a turn enters the inspect/plan/execute/verify path.
- `prompts.py` holds the default prompt plus tool-use and execution prompts used across normal and deep turns.
- `thinking_stream.py` parses model output into thinking and visible-answer streams. Its tag pairs must stay aligned with `THINK_TAG_PAIRS` in `static/app.js`.
- `themes.py` defines named palettes that expand into the CSS variables expected by `static/style.css`.
- `static/app.js` manages the client runtime: streaming chat events, slash commands, plan approval, workspace activity, file viewing/editing, attachments, and voice playback/recording.

## Tech stack

- Backend: Python 3.11, FastAPI, Uvicorn, httpx
- Frontend: Vanilla JS, Marked.js, highlight.js, CodeMirror
- LLM runtime: vLLM via an OpenAI-compatible API
- Database: SQLite for conversations, messages, summaries, per-conversation runs/workspaces, and assistant feedback
- Recent corrective user replies can also be mined from SQLite as implicit failure signals during feedback-driven repo-improvement passes.
- Streaming: WebSocket `/ws/chat`

## Persistence model

- Every conversation gets a stable run id plus a dedicated workspace directory on disk.
- Python package installs use a separate managed chat-scoped environment outside the workspace tree, so the workspace stays focused on user-visible files and artifacts.
- Assistant message feedback is stored in SQLite and reused during history ranking.
- Corrective user follow-ups can automatically mark the previous assistant turn as negative feedback, and feedback-driven deep runs can persist a recent-feedback digest into the workspace task state.
- Voice artifacts live under `VOICE_ROOT` and are cleaned up by pruning plus conversation/app reset flows.

## Additional guides

- [Harness And Tools](harness.md) — how turns are routed, tools run, and progress is reported
- [UI Features](ui.md) — the current chat/workspace UI surface
