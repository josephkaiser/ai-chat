# Harness And Tools

This app now has a fairly opinionated server-side harness. The frontend is intentionally thin; most routing, tool execution, persistence, and background work happen in `src/python/harness.py` plus the orchestration modules under `src/python/ai_chat/`.

## Mental model

Each visible turn can follow one of three paths:

- **Direct answer** for simple questions
- **Scoped tool loop** for targeted reads, patches, commands, or web lookups
- **Deep execution** for inspect/plan/execute/verify style repo work

Separately, a turn can also bind itself to a **file session**, which lets the backend keep working on one file across visible turns.

## Main entry points

- `process_chat_turn()` in `src/python/harness.py` is the main request handler for both `/ws/chat` and `/api/chat`.
- `routing_program.py` and `turn_strategy.py` decide the top-level path.
- `runtime_layers.py` builds model-only context that should not be stored as visible transcript text.
- `deep_runtime.py` owns the deep-session lifecycle and callback seam.
- `task_engine.py` runs typed structured plans for newer task-style flows.

## Scoped tool loop

`run_tool_loop(...)` is still the core lightweight execution loop:

1. Build a prompt limited to the allowed tool set for this request.
2. Let the model emit one tool call.
3. Validate the tool name and JSON arguments.
4. Execute the tool and append a structured tool result.
5. Repeat until the model returns plain text or the step budget is exhausted.

Important guardrails:

- tools are scoped per request, not globally enabled
- `workspace.run_command` takes argv arrays rather than shell strings
- command execution snapshots workspace files before and after the run so new artifacts can be surfaced
- patching stays narrow and exact-match oriented
- direct-answer fallback logic tries to recover from incorrect “I can’t do that here” or “run this locally” model behavior

## Deep execution path

The deeper flow is now split across the harness and `src/python/ai_chat/deep_runtime.py`.

Typical phases are:

1. `evaluate`
2. `inspect`
3. `plan`
4. `execute`
5. `verify`
6. `audit`
7. `synthesize`

The backend can still emit richer structured events such as `plan_ready`, `build_steps`, and scope-audit updates even though the bundled frontend currently uses a simpler chat-and-hint experience.

## File sessions and background work

One of the biggest refactors is the file-session runtime.

- A file session is keyed by workspace plus normalized target path.
- Hidden artifacts for that file live under `.ai-chat/`, including drafts/specs, candidate outputs, evaluations, and versions.
- Foreground and background work are stored as durable `file_session_jobs` rows.
- One file per workspace can own the background-polish loop at a time.
- The background worker resumes queued jobs on startup and keeps iterating until work completes, is superseded, or loses focus.

This lets the backend treat “improve this file over time” as something more durable than one chat completion.

## Replay triage and failure signals

Another refactor is the replay/context-eval loop:

- explicit negative feedback can capture a replay case
- retries can also capture a replay case
- corrective user replies can be summarized into recent failure signals during deeper repo-improvement passes
- the workspace can expose these captures under `.ai/context-evals/`
- `/api/context-evals/report` builds a triage summary that the frontend renders in the Replay Triage sidebar

## Tool surface

The backend tool surface includes:

- `workspace.list_files`
- `workspace.grep`
- `workspace.read_file`
- `workspace.patch_file`
- `workspace.render`
- `workspace.run_command`
- `spreadsheet.describe`
- `conversation.search_history`
- `web.search`
- `web.fetch_page`

`workspace.render` and command-generated artifact detection can open HTML, image, Markdown, CSV, spreadsheet, and PDF outputs directly in the viewer.

## Approvals and the shipped frontend

The backend still supports blocking approval events for tools and commands via `permission_required`.

The current bundled frontend takes a simpler path:

- it sends turns in deep mode by default
- it opts into `auto_approve_tool_permissions: true`
- when the backend still emits `permission_required`, the frontend auto-responds with approval and continues

That means the protocol still supports explicit approval gates, but the shipped UI is currently optimized for a smoother local-workspace workflow.

## Related files

- `src/python/harness.py` — orchestration, routes, tool runtime, file-session worker
- `src/python/ai_chat/deep_runtime.py` — deep-session lifecycle
- `src/python/ai_chat/context_eval.py` — replay capture and report logic
- `src/web/app.ts` — current thin client
- `docs/api.md` — route and event reference
