# Capability Playbook

This app works best when new capabilities are added as a small end-to-end workflow, not just as another exposed command.

## Core pattern

For a new capability, try to wire all of these together:

1. **Model-facing instructions**
   Add examples and constraints to `build_tool_system_prompt(...)` in [app.py](/Users/joe/dev/ai-chat/app.py) so the assistant knows:
   - which tool to call
   - what a good invocation looks like
   - when to research first
   - how to recover if approval is denied

2. **Granular approvals**
   Add a focused approval request in `build_tool_permission_request(...)` so the user sees the real action being requested.
   If the action is denied, the runtime should pause at that boundary and resume only after the user explicitly approves it.

   Good:
   - `Allow pip install?`
   - `Allow Python venv setup?`

   Less good:
   - `Allow python3?`

3. **Direct entry point**
   If the capability is common, add a slash command or deterministic workflow so users can invoke it intentionally without relying only on prompt wording.

4. **Workspace-first install path**
   Prefer reusable artifacts that live inside the conversation workspace.
   For Python work, the default pattern is:
   - create or reuse the managed chat Python environment
   - keep that environment server-owned and outside `runs/` so syncs stay lean
   - install dependencies there with pip
   - write scripts into the workspace
   - verify with the same managed environment
   - if the result is visual, save it as a previewable workspace artifact such as PNG, SVG, HTML, or PDF

5. **Tool-loop behavior**
   The assistant should:
   - inspect before acting when that improves accuracy
   - call web tools to verify package names/docs when useful
   - pause only when a gated tool is actually used
   - pause at the approval boundary if the user denies approval
   - keep long-running installs stoppable
   - avoid handing execution back with “run this locally” instructions when the app can still run or render the result
   - use workspace artifacts as durable memory when the context window is limited

## Python capability pattern

The current Python workflow is the reference example:

1. Use `web.search` or `web.fetch_page` when package choice or API usage is uncertain.
2. Create or reuse the managed chat Python environment if needed.
3. Install packages with `/pip` or `workspace.run_command`.
4. Write the Python script or artifact into the workspace.
5. Run a focused verify command in the same environment.
6. Save any plots or reports into the workspace so the UI can surface them automatically.

Example sequence:

- `/search best python package for sec filings api`
- `/pip sec-edgar-downloader pandas`
- `/code build a small script that downloads Broadcom filings into the workspace`

## When to add a new capability

A new capability is worth formalizing when it is:

- repeated across many chats
- awkward to express through plain language alone
- risky enough that approvals should be more explicit
- improved by reusable docs and examples

## Files to update

- [app.py](/Users/joe/dev/ai-chat/app.py) — tool prompt, approvals, direct workflows, runtime behavior
- [static/app.js](/Users/joe/dev/ai-chat/static/app.js) — slash menu, turn feature inference, approval UX
- [docs/harness.md](/Users/joe/dev/ai-chat/docs/harness.md) — harness/tool-loop behavior
- [docs/api.md](/Users/joe/dev/ai-chat/docs/api.md) — slash/API/event reference
- [docs/ui.md](/Users/joe/dev/ai-chat/docs/ui.md) — user-facing workflow entry points
