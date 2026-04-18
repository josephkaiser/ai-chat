# Architecture

## Project structure

```text
ai-chat/
├── app.py               # FastAPI app: routes, WebSockets, SQLite, vLLM client
├── src/python/ai_chat/themes.py            # Light/dark palettes → CSS custom properties
├── src/python/ai_chat/prompts.py           # Default system prompt text
├── src/python/ai_chat/thinking_stream.py   # Model stream → “thinking” vs visible answer
├── src/python/ai_chat/turn_strategy.py     # Turn assessment and top-level routing decisions
├── src/python/ai_chat/deep_flow.py         # Deep-mode routing decisions
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

- `app.py` is the entrypoint. It mounts static files, renders `index.html`, and implements REST and WebSocket handlers. It owns the chat harness, workspace catalog APIs, voice pipeline, tool execution loop, and workspace-backed chat routing.
- `src/python/ai_chat/turn_strategy.py` evaluates each user turn against the app’s main skill loop: local RAG, web search, file creation, coding mode, planning mode, execution mode, and verification needs.
- `src/python/ai_chat/deep_flow.py` decides what the deep execution pipeline should do next when a turn enters the inspect/plan/execute/verify path.
- `src/python/ai_chat/prompts.py` holds the default prompt plus tool-use and execution prompts used across normal and deep turns.
- `src/python/ai_chat/thinking_stream.py` parses model output into thinking and visible-answer streams. Its tag pairs must stay aligned with `THINK_TAG_PAIRS` in `src/web/app.js`.
- `src/python/ai_chat/themes.py` defines named palettes that expand into the CSS variables expected by `src/web/style.css`.
- `src/web/app.js` manages the client runtime: streaming chat events, slash commands, plan approval, workspace activity, file viewing/editing, attachments, and voice playback/recording.

## Tech stack

- Backend: Python 3.11, FastAPI, Uvicorn, httpx
- Frontend: Vanilla JS, Marked.js, highlight.js, CodeMirror
- LLM runtime: vLLM via an OpenAI-compatible API
- Database: SQLite for conversations, messages, workspace catalog rows, runs, summaries, and assistant feedback
- Recent corrective user replies can also be mined from SQLite as implicit failure signals during feedback-driven repo-improvement passes.
- Streaming: WebSocket `/ws/chat`

## Persistence model

- Workspaces are first-class rows keyed by canonical absolute root path plus a user-facing display name.
- Conversations are ephemeral transcripts that attach to one workspace via `conversations.workspace_id`.
- Multiple conversations may point at the same workspace over time, so fresh chats can continue working in the same files without reopening an old thread.
- Python package installs use a managed workspace-scoped environment outside the workspace tree, so new conversations in the same workspace reuse the same environment.
- Assistant message feedback is stored in SQLite and reused during history ranking.
- Corrective user follow-ups can automatically mark the previous assistant turn as negative feedback, and feedback-driven deep runs can persist a recent-feedback digest into the workspace task state.
- Voice artifacts live under `VOICE_ROOT` and are cleaned up by pruning plus conversation/app reset flows.

## Additional guides

- [Harness And Tools](harness.md) — how turns are routed, tools run, and progress is reported
- [UI Features](ui.md) — the current chat/workspace UI surface
