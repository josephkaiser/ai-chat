# API Endpoints

## Pages

- `GET /` — Web interface

## WebSocket

### `/ws/chat` (client → server)

Send JSON objects with fields such as:

- `message` — User text
- `conversation_id` — Conversation UUID
- `workspace_id` — Active workspace UUID for new chats or workspace switches
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
- `permission_required` is a blocking pause. If the user declines, the task stays paused until the capability is approved for that chat and resumed.
- Tool auto-approve applies to tool and command requests only. Plan execution still requires explicit plan approval in the UI.

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
- `GET /api/conversation/{conversation_id}` — Get recent messages, pending plan preview, and attached workspace metadata
- `POST /api/conversation/{conversation_id}/rename` — Rename conversation
- `DELETE /api/conversation/{conversation_id}` — Delete only the conversation transcript and scoped voice artifacts

## Messages

- `POST /api/message/{message_id}/feedback` — Save assistant feedback (`positive`, `negative`, `neutral`)
- `POST /api/message/{message_id}/retry` — Get retry info for an assistant message

## Search

- `GET /api/search?query=...` — Search chat history

## Workspaces

- `GET /api/workspaces` — List the shared workspace catalog and default selection
- `POST /api/workspaces` — Create a catalog entry for a workspace root
- `GET /api/workspaces/{workspace_id}` — Get workspace metadata
- `POST /api/workspaces/{workspace_id}/rename` — Rename a workspace in the catalog
- `DELETE /api/workspaces/{workspace_id}` — Remove a workspace from the catalog when no chats still reference it
- `GET /api/workspaces/{workspace_id}/files?path=...` — List one workspace directory
- `GET /api/workspaces/{workspace_id}/file?path=...` — Read a workspace file or structured preview payload
- `POST /api/workspaces/{workspace_id}/file` — Save editor changes into a workspace file
- `POST /api/workspaces/{workspace_id}/upload` — Upload files into the workspace
- `GET /api/workspaces/{workspace_id}/file/download?path=...` — Download one workspace file
- `GET /api/workspaces/{workspace_id}/spreadsheet?path=...&sheet=...` — Spreadsheet preview/summary
- `GET /api/workspaces/{workspace_id}/download` — Download the full workspace as a zip

Compatibility routes still exist under `/api/workspace/{conversation_id}/...`, but new clients should prefer the `workspace_id` routes above.

Workspace API notes:

- The workspace catalog is shared across the server deployment; there is no per-user isolation in this phase.
- Workspace roots are keyed by canonical absolute path. Display names are editable labels only.
- Directory listings hide dot-prefixed paths unless the request explicitly targets a hidden path.
- Directory listing items now include lightweight metadata such as `modified_at`, `content_kind`, and `kind` so the client can rank and preview artifacts.
- File reads can return non-text preview metadata. For images, the payload uses `content_kind: "image"` with binary preview metadata instead of raw file bytes.
- Tool results from `workspace.run_command` may include detected artifact metadata in `result.items`, plus `result.path` and `result.open_path` when there is a primary artifact worth surfacing in the viewer.

## Voice

- `GET /api/voice/status` — Report STT/TTS availability
- `POST /api/voice/transcribe` — Upload audio and receive a transcript
- `POST /api/voice/speak` — Generate reply audio for browser playback
- `GET /api/voice/file/{filename}` — Fetch synthesized audio

## System

- `GET /health` — Health check with model availability and voice runtime summary
- `POST /api/reset-all` — Reset chats, workspaces, cached voice artifacts, and related app data
