# API Endpoints

## Pages

- `GET /` — Web interface

## WebSocket

### `/ws/chat` (client → server)

Send JSON objects with fields such as:

- `message` — User text
- `conversation_id` — Conversation UUID
- `system_prompt` — Optional prompt override
- `features` — Per-turn tool/permission flags inferred by the UI
- `slash_command` — Optional structured slash intent for `/search`, `/grep`, `/plan`, `/code`, or `/pip`

### `/ws/chat` (server → client)

The server streams JSON events with a `type` field.

Common event types:

- `start` — Assistant turn began
- `activity` — Structured progress update with `phase`, `label`, `content`, and optional `step_label`
- `assistant_note` — Intermediate draft/note while the turn continues
- `plan_ready` — Execution plan preview plus `execute_prompt` and `builder_steps`
- `build_steps` — Structured checklist progress
- `think_start`, `think_token`, `think_end` — Collapsible reasoning stream
- `token` — Visible answer stream
- `tool_start`, `tool_result`, `tool_error` — Tool lifecycle events
- `permission_required` — Pause for inline tool or command approval
- `final_replace` — Replace the in-progress draft with finalized text
- `message_id` — Saved assistant message id
- `canceled`, `done`, `error` — Terminal turn status

Notes:

- Command approvals now support scoped Python setup requests such as `pip install` and `python -m venv`.
- Long-running install/setup commands are expected to keep running until completion unless the user sends `stop` or `interrupt`.

Preferred `activity.phase` values:

- `analyze`
- `evaluate`
- `inspect`
- `plan`
- `execute`
- `verify`
- `audit`
- `synthesize`
- `respond`
- `blocked`
- `error`

## Conversations

- `GET /api/conversations` — List conversations
- `GET /api/conversation/{conversation_id}` — Get recent messages plus any saved pending plan preview
- `POST /api/conversation/{conversation_id}/rename` — Rename conversation
- `DELETE /api/conversation/{conversation_id}` — Delete the conversation, workspace, and scoped voice artifacts

## Messages

- `POST /api/message/{message_id}/feedback` — Save assistant feedback (`positive`, `negative`, `neutral`)
- `POST /api/message/{message_id}/retry` — Get retry info for an assistant message

## Search

- `GET /api/search?query=...` — Search chat history

## Workspace

- `GET /api/workspace/{conversation_id}` — Workspace metadata (`run_id`, absolute path, label)
- `POST /api/workspace/{conversation_id}/upload` — Upload files into the conversation workspace
- `GET /api/workspace/{conversation_id}/files?path=...` — List one workspace directory
- `GET /api/workspace/{conversation_id}/file?path=...` — Read a workspace file or structured preview payload
- `POST /api/workspace/{conversation_id}/file` — Save editor changes into a workspace file
- `GET /api/workspace/{conversation_id}/file/download?path=...` — Download one workspace file
- `GET /api/workspace/{conversation_id}/spreadsheet?path=...&sheet=...` — Spreadsheet preview/summary
- `GET /api/workspace/{conversation_id}/download` — Download the full conversation workspace as a zip

## Voice

- `GET /api/voice/status` — Report STT/TTS availability
- `POST /api/voice/transcribe` — Upload audio and receive a transcript
- `POST /api/voice/speak` — Generate reply audio for browser playback
- `GET /api/voice/file/{filename}` — Fetch synthesized audio

## System

- `GET /health` — Health check with model availability and voice runtime summary
- `POST /api/reset-all` — Reset chats, workspaces, cached voice artifacts, and related app data
