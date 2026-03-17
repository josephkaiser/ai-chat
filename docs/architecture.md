# Architecture

## Project Structure

```
ai-chat/
├── app.py              # FastAPI application
├── chat                # CLI to start/stop the app
├── static/
│   ├── index.html      # Web UI template
│   └── app.js          # Frontend JavaScript
├── docker-compose.yml  # Docker services
├── Dockerfile          # Chat app container
├── docs/               # Documentation
└── data/
    └── chat.db         # SQLite database
```

## Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Frontend:** Vanilla JS, Marked.js, highlight.js
- **LLM:** vLLM serving Qwen 3 8B via OpenAI-compatible API
- **Database:** SQLite
- **Streaming:** WebSocket for real-time token delivery
