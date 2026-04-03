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
| `plan_ready`    | Execution plan preview ready for approval, editable build steps, and optional plan hydration (`plan`, `execute_prompt`, `builder_steps`) |
| `build_steps`   | Structured checklist state for deep-mode build steps |
| `tool_start`    | Tool invocation started |
| `tool_result`   | Tool finished, with summarized payload or error |
| `final_replace` | Replace the in-progress assistant draft with finalized response text |
| `command_approval_required` | Pause for per-chat executable approval before a workspace command runs |
| `message_id`    | SQLite id of the saved assistant message (`message_id` field) |
| `canceled`      | Active turn was interrupted by the user |
| `idle`          | A stop request arrived while no turn was running |
| `done`          | Turn complete |
| `error`         | Error (`content` has message text) |

Legacy compatibility:

- `status` may still appear from older code paths or clients, but new harness progress should use `activity`.
- Audit progress is now reported through `activity` with `phase: "audit"` instead of a separate event type.

### `/ws/logs`

- `WebSocket /ws/logs` — Log tail; messages use `type: log` and `content` (text).


## Conversations

- `GET /api/conversations` — List all conversations
- `GET /api/conversation/{conversation_id}` — Get recent raw conversation messages plus any saved `pending_plan` plan-preview payload for that chat (messages include message `id` and assistant `feedback`)
- `POST /api/conversation/{conversation_id}/rename` — Rename conversation
- `DELETE /api/conversation/{conversation_id}` — Delete the conversation, its workspace, and any scoped cached voice artifacts

## Messages

- `POST /api/message/{message_id}/feedback` — Submit assistant feedback (`positive`, `negative`, or `neutral`; assistant replies default to `neutral`, and the UI clears a rating by sending `neutral`)
- `POST /api/message/{message_id}/retry` — Get retry info for an assistant message

## Search

- `GET /api/search?query=...` — Search chat history (message content)

## Tools

- `POST /api/execute-code` — Execute Python code (5s timeout; disabled unless `EXECUTE_CODE_ENABLED=true`)
- `GET /api/files/list?path=...` — List directory contents
- `GET /api/files/read?path=...` — Read file contents (max 1MB)

## Workspace

- `GET /api/workspace/{conversation_id}` — Get conversation workspace metadata (`run_id`, absolute workspace path, label)
- `POST /api/workspace/{conversation_id}/upload` — Upload files into the conversation workspace (used by normal file attachments and recorded audio attachments; max 8 files per turn, 10MB per file)
- `GET /api/workspace/{conversation_id}/files?path=...` — List one workspace directory for the tree view
- `GET /api/workspace/{conversation_id}/file?path=...` — Read a workspace file or return a structured preview payload for supported document/spreadsheet formats
- `POST /api/workspace/{conversation_id}/file` — Save editor changes back into a workspace file (max 1MB)
- `GET /api/workspace/{conversation_id}/file/download?path=...` — Download one workspace file
- `GET /api/workspace/{conversation_id}/spreadsheet?path=...&sheet=...` — Get spreadsheet summary/preview data
- `GET /api/workspace/{conversation_id}/download` — Download the entire conversation workspace as a zip

## Voice

- `GET /api/voice/status` — Report whether native server-side STT/TTS backends are available
- `POST /api/voice/transcribe` — Upload audio as multipart form data (`file`) and receive `{ transcript, ... }`; accepts optional `conversation_id` for scoped temp artifact names. The built-in web UI currently records mic input as a workspace attachment instead of calling this endpoint directly.
- `POST /api/voice/speak` — Send `{ "text": "...", "conversation_id": "..."? }` and receive a generated server audio URL plus backend metadata
- `GET /api/voice/file/{filename}` — Fetch synthesized audio for browser playback

## System

- `GET /health` — Health check (model availability plus voice runtime summary)
- `POST /api/reset-all` — Reset chats, workspaces, cached voice artifacts, pet state, and related app data
- `GET /api/dashboard` — Model status, container info, cache details
- `GET /api/models/library` — List cached Hugging Face models plus active background download jobs
- `POST /api/vllm/restart` — Restart vLLM container
- `POST /api/models/library/download` — Start downloading a Hugging Face model into the shared cache
- `POST /api/models/library/activate` — Restart vLLM using a cached Hugging Face model from the discovery library
- `POST /api/models/library/delete` — Delete a cached Hugging Face model that is not configured as an active profile
- `POST /api/model/switch` — Switch the selected model profile
- `POST /api/model/redownload` — Clear cache and re-download model
