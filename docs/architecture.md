# Architecture

## Project Structure

```
ai-chat/
├── app.py              # FastAPI application (API, WebSocket, vLLM client)
├── chat                # CLI to start/stop the app
├── static/
│   ├── index.html      # Web UI template
│   ├── app.js          # Frontend JavaScript
│   └── style.css       # Styles and themes
├── docker-compose.yml  # Docker services
├── Dockerfile          # Chat app container
├── docs/               # Documentation
└── data/
    └── chat.db         # SQLite database
```

## Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, httpx
- **Frontend:** Vanilla JS, Marked.js, highlight.js
- **LLM:** vLLM serving Qwen 3 8B (OpenAI-compatible API, streamed via httpx)
- **Database:** SQLite (conversations, messages with feedback)
- **Streaming:** WebSocket for real-time token delivery
- **Search:** DuckDuckGo (auto-triggered for queries about current events)
