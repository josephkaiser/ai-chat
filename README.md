# AI Chat with vLLM

A chat application with a web UI, running Qwen 3 8B on vLLM with a single GPU.

## Features

- **Web UI** - Responsive interface with light/dark mode
- **Streaming Responses** - Real-time token streaming via WebSocket
- **32K Context Window** - Smart history management (75% for conversation history)
- **Web Search** - Automatic and manual web search via DuckDuckGo
- **Code Execution** - Sandboxed Python execution
- **File Browsing** - Read files and list directories
- **Chat Search** - Full-text search across all conversations
- **Markdown** - Full markdown rendering with syntax highlighting
- **System Prompt** - Customizable system prompt per session
- **Log Viewer** - Real-time terminal logs

## Quick Start

```bash
docker compose up -d

# Watch model download/loading (first time ~2-4GB)
docker compose logs -f vllm

# When ready, open http://localhost:8000
```

The web UI loads immediately — the model loads in the background.

## Configuration

Edit `docker-compose.yml` to change the model or vLLM settings:

```yaml
vllm:
  command: >
    --model Qwen/Qwen3-8B
    --gpu-memory-utilization 0.90
    --max-model-len 32768
    --swap-space 8
    --enable-prefix-caching
    --max-num-seqs 256
    --enable-chunked-prefill
```

Set `MODEL_NAME` in the `chat-app` environment to match.

### For Different GPUs

**24GB (RTX 3090/4090):** Default config works well.

**12GB (RTX 3060):**
```yaml
--gpu-memory-utilization 0.90
--max-model-len 16384
--swap-space 4
```

**40GB+ (A100/H100):**
```yaml
--gpu-memory-utilization 0.95
--max-model-len 65536
--swap-space 16
```

## API Endpoints

- `GET /` - Web interface
- `WebSocket /ws/chat` - Streaming chat
- `WebSocket /ws/logs` - Terminal logs
- `GET /api/conversations` - List conversations
- `GET /api/conversation/{id}` - Get conversation messages
- `POST /api/conversation/{id}/rename` - Rename conversation
- `DELETE /api/conversation/{id}` - Delete conversation
- `GET /api/search?query=...` - Search chat history
- `POST /api/web-search` - Search the web
- `POST /api/execute-code` - Execute Python code
- `GET /api/files/list?path=...` - List directory
- `GET /api/files/read?path=...` - Read file
- `GET /health` - Health check

## Restarting

```bash
# Restart just the chat app (fast, keeps model loaded)
docker compose restart chat-app

# Restart vLLM (reloads model, takes 2-5 min)
docker compose restart vllm

# View logs
docker compose logs -f
```

## Project Structure

```
ai-chat/
├── app.py              # FastAPI application
├── static/
│   ├── index.html      # Web UI template
│   └── app.js          # Frontend JavaScript
├── docker-compose.yml  # Docker services
├── Dockerfile          # Chat app container
└── data/
    └── chat.db         # SQLite database
```
