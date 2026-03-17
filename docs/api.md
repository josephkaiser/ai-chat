# API Endpoints

- `GET /` — Web interface
- `WebSocket /ws/chat` — Streaming chat
- `WebSocket /ws/logs` — Terminal logs
- `GET /api/conversations` — List conversations
- `GET /api/conversation/{id}` — Get conversation messages
- `POST /api/conversation/{id}/rename` — Rename conversation
- `DELETE /api/conversation/{id}` — Delete conversation
- `GET /api/search?query=...` — Search chat history
- `POST /api/web-search` — Search the web
- `POST /api/execute-code` — Execute Python code
- `GET /api/files/list?path=...` — List directory
- `GET /api/files/read?path=...` — Read file
- `GET /health` — Health check
