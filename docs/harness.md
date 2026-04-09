# Harness And Tools

This app is no longer just a plain chat frontend over vLLM. It now has a server-side harness that can inspect the workspace, choose a scoped tool set, run tool calls in a controlled loop, and surface progress back to the UI.

## Mental model

There are two main execution paths per turn:

- **Normal mode** keeps the interaction lightweight. The server decides whether to answer directly or expose a small tool list for the current request.
- **Deep mode** uses a more explicit orchestration flow: inspect, plan, build, verify, audit, and synthesize.

In both modes, the browser streams structured progress so the user can see what the assistant is doing instead of waiting on a silent response.

## Normal-turn harness

`process_chat_turn()` in `app.py` is the entry point for `/ws/chat`.

For each request the server:

1. Saves the user message and optional attachment context.
2. Parses per-turn feature flags from the UI.
3. Classifies workspace intent.
4. Chooses a small allowed tool set with `select_enabled_tools(...)`.
5. Either streams a direct answer, enters `run_tool_loop(...)`, or auto-upgrades broad approved write requests into the deeper inspect/plan/build/verify flow.

`run_tool_loop(...)` is intentionally small:

- The model receives a tool-use system prompt plus the allowed tool names.
- The model may emit a single `<tool_call>{...}</tool_call>` block.
- The server validates the tool name and arguments.
- The server executes the tool and appends a structured `<tool_result>` message.
- After a successful patch, the server can infer one focused auto-verify command and run it immediately when command access is available.
- The loop repeats until the model returns plain text or the step budget is exhausted.

Important guardrails:

- Tool access is **scoped per request**, not globally.
- The server can restrict the prompt to a subset of tools with `build_filtered_tool_system_prompt(...)`.
- Tool budgets are capped by `tool_loop_step_limit_for_request(...)`.
- Token budgets are reduced for tool-oriented turns by `tool_loop_token_budget(...)`.
- `workspace.run_command` takes an argv array, not a shell string.
- `workspace.patch_file` uses exact-match edits so changes stay narrow and predictable.
- `workspace.render` is only exposed when the prompt strongly implies “render/preview/display this HTML”.
- Final answer text is post-processed to strip unsupported claims that a file was created or updated when no successful workspace write happened.
- Direct answers get one recovery pass if the model incorrectly claims an available tool capability is missing.
- Broad “build me an app/project/repo” requests with approved workspace edits can bypass the lightweight loop and run through the full workspace execution path automatically.

## Deep-mode harness

Deep mode uses a richer workflow in `orchestrated_chat(...)`.

Typical flow:

1. `evaluate` picks workspace-assisted or text-first deep mode.
2. `inspect` gathers repo facts, attachments, and conversation context.
3. `plan` builds a normalized execution plan.
4. `execute` carries out plan steps, usually with workspace tools enabled.
5. `verify` checks results and can run focused validations.
6. `audit` compares requested scope with verified evidence.
7. `synthesize` prepares the final response.

Two user-facing behaviors matter here:

- Some deep requests return a **plan preview** first via `plan_ready`, so the UI can offer approval-first execution without overwriting the user's composer draft.
- Explicit or auto-executed workspace build requests can proceed through the full inspect/build/verify path and stream step-by-step activity while work is happening.

## Tool surface

The current tool protocol is defined by `build_tool_system_prompt(...)` and executed by `execute_tool_call(...)`.

Available tools today:

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

Current search-specific behavior:

- `workspace.grep` is the preferred read-only repo search tool for code and text questions.
- `conversation.search_history` is for recall within the active conversation.
- `web.search` returns ordered results plus normalized domain metadata, and by default checks general web results alongside Wikipedia and Reddit result sets.
- For some topics, `web.search` also adds curated authoritative domains; philosophy queries now include Stanford Encyclopedia of Philosophy and Internet Encyclopedia of Philosophy result sets.
- Curated authoritative domains are fail-open and can be temporarily disabled automatically if they start failing repeatedly.
- `web.fetch_page` returns cleaned page text plus normalized source metadata so final answers can cite fetched pages.

How tools are exposed:

- **Workspace read tools** are available when the request clearly needs local files.
- **Workspace render** is added for HTML preview/display requests so the model can materialize a preview file directly in the viewer.
- **Write access** only appears when the user approves file changes for that turn.
- **Command execution** only appears when the user approves command execution for that turn.
- **Conversation search** depends on the `local_rag` feature toggle.
- **Web search** depends on the `web_search` feature toggle.

`allowed_workspace_tools(...)` is the core permission gate for file and command tools.

## Per-turn approvals

The UI does not blindly grant write or command access. `resolveTurnFeatures(...)` in `static/app.js` infers likely intent from the prompt and attachments, then asks for confirmation when a request looks like it wants to:

- create or edit workspace files
- run workspace commands

Those approvals become `FeatureFlags` on the server:

- `agent_tools`
- `workspace_write`
- `workspace_run_commands`
- `local_rag`
- `web_search`

This keeps casual chat cheap and safe while still allowing fully agentic coding turns when the user explicitly wants them.

## Workspace model

Each conversation gets its own workspace on the server. The harness uses that workspace for:

- uploaded attachments
- generated artifacts
- patched files
- command execution cwd
- downloadable zip exports

The browser can inspect the same workspace through the workspace panel and file modal, so tool output is visible outside the model transcript.

## Activity events

The harness reports structured progress through WebSocket `activity` events.

Common phases:

- `evaluate`
- `inspect`
- `plan`
- `execute`
- `verify`
- `audit`
- `synthesize`
- `respond`
- `error`
- `blocked`

The UI also receives:

- `tool_start`
- `tool_result`
- `assistant_note`
- `plan_ready`
- `build_steps`

Those events drive the workspace activity timeline, build-step checklist, and loading text.

## Related files

- `app.py` — orchestration, tool execution, workspace APIs, and voice APIs
- `prompts.py` — base prompts plus tool-use instructions
- `static/app.js` — feature toggles, per-turn approvals, activity rendering, workspace UI
- `docs/api.md` — endpoint and WebSocket payload reference
