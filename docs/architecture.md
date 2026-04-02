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

- **`app.py`** is the entrypoint. It mounts static files, renders `index.html`, and implements REST + WebSocket handlers. For orientation, read the module docstring at the top of `app.py`.
- **`themes.py`** defines small named palettes (e.g. paper, ink, panel). Those expand to the `--bg_primary`, `--text_secondary`, … keys that **`static/style.css`** already expects. At runtime, `GET /` injects JSON into the page; **`static/app.js`** calls `applyTheme()` to set `document.documentElement` styles.
- **`prompts.py`** holds **`DEFAULT_SYSTEM_PROMPT`**. The chat WebSocket uses it unless the client sends a custom `system_prompt`.
- **`thinking_stream.py`** implements **`ThinkingStreamSplitter`**: it parses model output for paired thinking tags (must match **`THINK_TAG_PAIRS` in `static/app.js`**) and emits the WebSocket payloads (`think_start`, `think_token`, `think_end`, and normal `token`).

## Tech stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, httpx (OpenAI-compatible chat completions to vLLM)
- **Frontend:** Vanilla JS, Marked.js, highlight.js (CDN)
- **LLM:** vLLM (OpenAI-compatible API); model name is configured via env / compose (see [Configuration](configuration.md))
- **Database:** SQLite (conversations, messages, optional feedback for history scoring)
- **Streaming:** WebSocket `/ws/chat` for token and thinking-region events
- **Dashboard / ops:** Optional Docker socket access from the chat container for vLLM restart and cache paths (see `GET /api/dashboard` and related endpoints in [API](api.md))
