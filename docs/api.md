# API Endpoints

## Pages

- `GET /` — web interface

## WebSocket

### `/ws/chat` (client → server)

Common request fields:

- `message` — user text
- `conversation_id` — conversation UUID
- `workspace_id` — active workspace UUID
- `system_prompt` — optional override
- `mode` — usually `"deep"` in the shipped frontend
- `turn_kind` — visible/runtime turn classification
- `features` — optional per-turn feature flags
- `auto_approve_tool_permissions` — whether the client wants tool permissions auto-approved

Control messages also include:

- `type: "stop"` — cancel the active turn
- `type: "interrupt"` — stop and immediately reprompt
- `type: "permission_response"` — answer a pending approval request
- `type: "pong"` — heartbeat response

### `/ws/chat` (server → client)

The server streams JSON events with a `type` field.

Common event types:

- `start` — assistant turn began
- `activity` — progress update with `phase`, `label`, `content`, and optional step metadata
- `assistant_note` — intermediate draft/update
- `token` — visible answer token stream
- `final_replace` — replace the in-progress draft with finalized text
- `tool_start`, `tool_result`, `tool_error` — tool lifecycle events
- `permission_required` — blocking approval request for a tool or command
- `message_id` — persisted assistant message id
- `canceled`, `done`, `error` — terminal status
- `ping` — heartbeat

Richer structured events used by deeper flows include:

- `plan_ready`
- `build_steps`
- `scope_audit`
- `file_session_bound`
- `draft_bootstrap`

Preferred `activity.phase` values include:

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

## Buffered HTTP fallback

- `POST /api/chat` — run one chat turn through the same backend and return buffered events instead of streaming

## Conversations

- `GET /api/conversations` — list conversations
- `GET /api/conversation/{conversation_id}` — get recent messages, pending plan payload, and attached workspace metadata
- `POST /api/conversation/{conversation_id}/rename` — rename conversation
- `DELETE /api/conversation/{conversation_id}` — delete conversation transcript while leaving the workspace intact

## Messages and replay capture

- `POST /api/message/{message_id}/feedback` — save assistant feedback and optionally capture a replay case for negative feedback
- `POST /api/message/{message_id}/retry` — prepare a retry payload and capture a replay case
- `GET /api/search?query=...` — search chat history
- `GET /api/context-evals/report?conversation_id=...&workspace_id=...&limit=...` — build the replay-triage summary used by the sidebar

## Workspaces

- `GET /api/workspaces` — list the shared workspace catalog and default workspace
- `POST /api/workspaces` — create/register a workspace root
- `GET /api/workspaces/{workspace_id}` — get workspace metadata
- `POST /api/workspaces/{workspace_id}/rename` — rename the display label
- `DELETE /api/workspaces/{workspace_id}` — remove a workspace catalog row when no conversations still reference it
- `GET /api/workspaces/{workspace_id}/files?path=...` — list a workspace directory
- `GET /api/workspaces/{workspace_id}/file?path=...` — read a workspace file or structured preview payload
- `GET /api/workspaces/{workspace_id}/file/view?path=...` — raw file response for inline preview
- `GET /api/workspaces/{workspace_id}/file/download?path=...` — download one file
- `POST /api/workspaces/{workspace_id}/file` — write a file
- `POST /api/workspaces/{workspace_id}/upload` — upload files
- `POST /api/workspaces/{workspace_id}/archive/extract` — extract an archive into the workspace
- `GET /api/workspaces/{workspace_id}/spreadsheet?path=...&sheet=...` — spreadsheet preview/summary
- `GET /api/workspaces/{workspace_id}/download` — download the entire workspace as a zip

## File sessions

- `GET /api/workspaces/{workspace_id}/file-sessions` — list durable file sessions plus summarized current state
- `POST /api/workspaces/{workspace_id}/file-sessions/ensure` — create/reuse one file session for a target path
- `POST /api/workspaces/{workspace_id}/file-sessions/focus` — enable or disable background focus for one file
- `DELETE /api/workspaces/{workspace_id}/file-sessions/{file_session_id}` — delete a file session
- `GET /api/workspaces/{workspace_id}/file-sessions/{file_session_id}` — get the full file-session bundle
- `GET /api/workspaces/{workspace_id}/file-sessions/{file_session_id}/jobs` — list foreground/background jobs
- `POST /api/workspaces/{workspace_id}/file-session-jobs` — create a file-session job
- `POST /api/workspaces/{workspace_id}/file-session-jobs/{job_id}/status` — update one job status

File-session notes:

- file sessions bind a visible target file to hidden `.ai-chat/` draft/spec/candidate/version state
- file-session rows persist `job_summary` so clients can render current state without replaying every websocket event
- exactly one file per workspace owns the background-polish loop at a time

## Compatibility routes

Older conversation-shaped workspace routes still exist under `/api/workspace/{conversation_id}/...`.

There are also simple local debug routes:

- `GET /api/files/list?path=...`
- `GET /api/files/read?path=...`

New clients should prefer the `workspace_id` routes.

## System

- `GET /health` — runtime/model health, selected profile, and load-progress metadata
- `POST /api/reset-all` — reset chats, workspaces, voice/cache/runtime state, and related app data
