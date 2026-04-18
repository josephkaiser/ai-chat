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
- Python capability setup uses a server-managed workspace environment outside the workspace, so installs do not flood the workspace tree or exports and can persist across fresh chats in the same workspace.
- `workspace.patch_file` uses exact-match edits so changes stay narrow and predictable.
- `workspace.render` is only exposed when the prompt strongly implies “render/preview/display this HTML”.
- `workspace.run_command` now snapshots workspace files before and after execution so the harness can surface newly created artifacts such as plots, reports, and exports.
- Final answer text is post-processed to strip unsupported claims that a file was created or updated when no successful workspace write happened.
- Direct answers get one recovery pass if the model incorrectly claims an available tool capability is missing.
- Direct answers also get a recovery pass when the model incorrectly hands execution back to the user with “run this locally” instructions even though command or render tools were available.
- Broad “build me an app/project/repo” requests with approved workspace edits can bypass the lightweight loop and run through the full workspace execution path automatically.

## Deep-mode harness

Deep mode uses a richer workflow in `orchestrated_chat(...)`.

Typical flow:

1. `evaluate` picks workspace-assisted or text-first deep mode.
2. `inspect` gathers repo facts, attachments, conversation context, and when the request is feedback-driven repo improvement, a recent corrective-feedback digest from `chat.db`.
3. `plan` builds a normalized execution plan.
4. `execute` carries out plan steps, usually with workspace tools enabled.
5. `verify` checks results and can run focused validations.
6. `audit` compares requested scope with verified evidence.
7. `synthesize` prepares the final response.

Two user-facing behaviors matter here:

- Some deep requests return a **plan preview** first via `plan_ready`, so the UI can offer approval-first execution without overwriting the user's composer draft.
- Explicit or auto-executed workspace build requests can proceed through the full inspect/build/verify path and stream step-by-step activity while work is happening.
- When a repo-improvement request explicitly mentions feedback, recent chats, `chat.db`, or the dev loop, deep mode can save a `.ai/recent-feedback.md` artifact and treat those corrective user replies as failure signals for the current pass.

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
- **Write access** is exposed for write-oriented requests, then paused behind an inline approval card the first time the model tries to edit files.
- **Command execution** is exposed for run/verify-oriented requests, then paused behind an inline approval card for each executable such as `git` or `python3`.
- **Python dependency setup** is routed through the same command runner, but uses more specific approvals for `python -m venv` and `pip install`.
- **Conversation search** stays available for recall-style prompts.
- **Web search** stays available for freshness-sensitive prompts, then pauses behind an inline approval card the first time live web access is used.

When a command creates or modifies visible files in the workspace, the tool result can now carry detected artifact metadata. Previewable outputs such as images, HTML, markdown, CSV, spreadsheets, and PDFs can be surfaced directly in the workspace viewer.

`allowed_workspace_tools(...)` is the core permission gate for file and command tools.

### Python capability pattern

For Python-heavy turns, the intended flow is:

1. Research packages or docs with `web.search` / `web.fetch_page` when package choice is uncertain.
2. Create or reuse the managed Python environment for the workspace when package work is needed.
   That environment is server-owned and stored outside the workspace root so workspace syncs stay lighter.
3. Install packages with pip into that managed environment.
4. Write scripts or artifacts into the workspace.
5. Verify with a focused command from the same environment.

## Runtime approvals

Tool enablement is no longer buried in settings. `resolveTurnFeatures(...)` in `src/web/app.js` now sends the server the relevant tool surface plus any remembered per-chat approvals, and the harness pauses only when a gated capability is actually used.

Current inline approval buckets are:

- workspace inspection
- workspace grep
- workspace edits / render writes
- web search / fetch
- per-executable commands such as `git`, `bash`, or `python3`
- scoped Python setup actions such as `python -m venv` and `pip install`

If the user denies one of those requests, the task pauses at that approval boundary and waits for the user to approve it and resume.
That denial is treated as a real block, not as optional context for the model to work around. The current task should resume only after the user approves it for this chat and says `continue`.

Plan approval is handled separately from per-tool approval. The composer’s tool auto-approve control can auto-approve tool and command requests for the current chat, but execution plans still require an explicit plan approval step in the UI.

Install-like Python setup commands are exempt from the short command timeout. They run until completion unless they fail or the user presses Stop / Interrupt, in which case the server cancels the subprocess cleanly.

## Workspace model

Workspaces are first-class catalog entries and conversations attach to them. The harness uses the attached workspace for:

- uploaded attachments
- generated artifacts
- patched files
- command execution cwd
- downloadable zip exports

The browser can inspect the same workspace through the activity rail, artifact list, file tree, and inline viewer, so tool output is visible outside the model transcript.
Deleting a conversation no longer deletes the attached workspace by default.
Dot-prefixed entries stay hidden in the browser workspace view unless a hidden path is targeted explicitly.
Previewable artifacts from `workspace.render` and from command-generated files can auto-open in the inline viewer to make execution feel more like live pair programming.

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
- `src/python/ai_chat/prompts.py` — base prompts plus tool-use instructions
- `src/web/app.js` — feature toggles, per-turn approvals, activity rendering, workspace UI
- `docs/api.md` — endpoint and WebSocket payload reference
