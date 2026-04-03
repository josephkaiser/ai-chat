# API Endpoints

## Pages

- `GET /` — Web interface

## WebSocket

### `/ws/chat` (client → server)

Send JSON objects, for example:

- `message` — User text (required)
- `conversation_id` — Conversation UUID
- `system_prompt` — Optional override for the default in `prompts.py`; shorter prompts usually perform better on the 8B profile
- `features` — Per-turn tool/permission flags inferred by the UI
- `slash_command` — Optional structured slash intent for direct flows such as `/search`, `/grep`, `/plan`, and `/code`

### `/ws/chat` (server → client)

JSON messages use a `type` field.

Primary progress event:

| `type`       | Purpose |
|--------------|---------|
| `activity`   | Structured harness progress event. Fields: `phase`, `label`, `content`, optional `step_label` |

Recommended `activity.phase` values:

- `analyze` — Evaluate the prompt and current turn context
- `evaluate` — Choose mode, tools, and execution path
- `inspect` — Gather workspace facts before planning or solving
- `plan` — Build or revise an execution plan
- `execute` — Carry out workspace changes or plan steps
- `verify` — Check outputs, run validations, or review a draft
- `audit` — Compare requested scope against evidence
- `synthesize` — Prepare or refine the final answer
- `respond` — Answer directly without a fuller execution pipeline
- `blocked` — Surface a missing permission or other blocker
- `error` — Report a failed operation

Other common values:

| `type`          | Purpose |
|-----------------|---------|
| `start`         | Assistant turn beginning |
| `token`         | Visible answer text chunk (streamed during normal chat) |
| `think_start`   | Beginning of model “thinking” region (collapsible in UI) |
| `think_token`   | Thinking region text chunk |
| `think_end`     | End of thinking region |
| `assistant_note`| Intermediate assistant draft/note while work continues |
| `plan_ready`    | Execution plan preview ready for approval or explicit copy into the composer |
| `build_steps`   | Structured checklist state for deep-mode build steps |
| `tool_start`    | Tool invocation started |
| `tool_result`   | Tool finished, with summarized payload or error |
| `message_id`    | SQLite id of the saved assistant message (`message_id` field) |
| `done`          | Turn complete |
| `error`         | Error (`content` has message text) |

Legacy compatibility:

- `status` may still appear from older code paths or clients, but new harness progress should use `activity`.
- Audit progress is now reported through `activity` with `phase: "audit"` instead of a separate event type.

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

## Voice

- `GET /api/voice/status` — Report whether native server-side STT/TTS backends are available
- `POST /api/voice/transcribe` — Upload recorded audio as multipart form data (`file`) and receive `{ transcript, ... }`
- `POST /api/voice/speak` — Send `{ "text": "..." }` and receive a generated server audio URL
- `GET /api/voice/file/{filename}` — Fetch synthesized audio for browser playback

## System

- `GET /health` — Health check (model availability plus voice runtime summary)
- `GET /api/dashboard` — Model status, container info, cache details
- `GET /api/models/library` — List cached Hugging Face models plus active background download jobs
- `POST /api/vllm/restart` — Restart vLLM container
- `POST /api/models/library/download` — Start downloading a Hugging Face model into the shared cache
- `POST /api/models/library/activate` — Restart vLLM using a cached Hugging Face model from the discovery library
- `POST /api/models/library/delete` — Delete a cached Hugging Face model that is not configured as an active profile
- `POST /api/model/redownload` — Clear cache and re-download model
