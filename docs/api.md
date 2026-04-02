# API Endpoints

## Pages

- `GET /` — Web interface

## WebSocket

### `/ws/chat` (client → server)

Send JSON objects, for example:

- `message` — User text (required)
- `conversation_id` — Conversation UUID
- `system_prompt` — Optional override for the default in `prompts.py`

### `/ws/chat` (server → client)

JSON messages use a `type` field. Common values:

| `type`        | Purpose |
|---------------|---------|
| `start`       | Assistant turn beginning |
| `token`       | Visible answer text chunk |
| `think_start` | Beginning of model “thinking” region (collapsible in UI) |
| `think_token` | Thinking region text chunk |
| `think_end`   | End of thinking region |
| `message_id`  | SQLite id of the saved assistant message (`message_id` field) |
| `done`        | Turn complete |
| `error`       | Error (`content` has message text) |

### `/ws/logs`

- `WebSocket /ws/logs` — Log tail; messages use `type: log` and `content` (text).


## Conversations

- `GET /api/conversations` — List all conversations
- `GET /api/conversation/{id}` — Get conversation messages
- `POST /api/conversation/{id}/rename` — Rename conversation
- `DELETE /api/conversation/{id}` — Delete conversation

## Messages

- `POST /api/message/{id}/feedback` — Submit thumbs up/down (`positive` or `negative`)
- `POST /api/message/{id}/retry` — Get retry info for an assistant message

## Search

- `GET /api/search?query=...` — Search chat history (message content)

## Tools

- `POST /api/execute-code` — Execute Python code (5s timeout)
- `GET /api/files/list?path=...` — List directory contents
- `GET /api/files/read?path=...` — Read file contents (max 1MB)

## System

- `GET /health` — Health check (model availability)
- `GET /api/dashboard` — Model status, container info, cache details
- `POST /api/vllm/restart` — Restart vLLM container
- `POST /api/model/redownload` — Clear cache and re-download model
