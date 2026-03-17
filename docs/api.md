# API Endpoints

## Pages

- `GET /` — Web interface

## WebSocket

- `WebSocket /ws/chat` — Streaming chat (JSON: start, token, done, error, message_id)
- `WebSocket /ws/logs` — Real-time terminal logs

## Conversations

- `GET /api/conversations` — List all conversations
- `GET /api/conversation/{id}` — Get conversation messages
- `POST /api/conversation/{id}/rename` — Rename conversation
- `DELETE /api/conversation/{id}` — Delete conversation

## Messages

- `POST /api/message/{id}/feedback` — Submit thumbs up/down (`positive` or `negative`)
- `POST /api/message/{id}/retry` — Get retry info for an assistant message

## Search

- `GET /api/search?query=...` — Search chat history
- `POST /api/web-search` — Search the web via DuckDuckGo

## Tools

- `POST /api/execute-code` — Execute Python code (5s timeout)
- `GET /api/files/list?path=...` — List directory contents
- `GET /api/files/read?path=...` — Read file contents (max 1MB)

## System

- `GET /health` — Health check (model availability)
- `GET /api/dashboard` — Model status, container info, cache details
- `POST /api/vllm/restart` — Restart vLLM container
- `POST /api/model/redownload` — Clear cache and re-download model
