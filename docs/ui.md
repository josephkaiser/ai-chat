# UI Features

The UI is a compact workspace-oriented chat surface: shared workspaces, chat, progress, files, attachments, plan approval, and voice controls all live in one place.

## Main surface

The composer supports:

- reasoning effort switching (`Low` / `High`)
- attachments uploaded into the conversation workspace
- browser-recorded audio attachments from the mic button
- a per-chat **Tools** toggle for ask-first vs auto-approve tool and command use
- slash commands for common coding tasks
- send-or-interrupt behavior from the primary action button

The client preserves a handful of browser preferences in `localStorage`, including speech playback settings, theme choice, the last active `workspace_id`, and per-chat tool approval preferences.

## Workspace catalog

The main menu now includes a shared workspace catalog.

- users can create a managed workspace under the server workspace root
- users can register an existing absolute folder path as a workspace
- renaming a workspace only changes the display label, not the root path
- selecting a different workspace starts a fresh chat so the existing transcript stays attached to its original workspace

## Settings and About

The Settings panel exposes:

- **Auto-Speak Replies** — queue fresh assistant replies through server TTS when available
- **Speech Speed** — adjust playback speed for generated audio
- **Appearance** and **System Prompt** controls
- **Reset App Data** danger zone

Tool approvals are no longer managed from Settings. The main menu now keeps **About** separate for the author note and project intent, while plan approval and runtime permission approval happen inline near the composer when needed.

## Workspace panel

When agent tools are enabled, the workspace area shows:

- an **Activity Log** with harness phases, tool calls, plan events, and finalization markers
- a **Recent Artifacts** rail for files the assistant just created or touched
- an **All Files** tree for files created, uploaded, or edited during the conversation
- refresh and download actions for the current workspace
- shared file access across conversations attached to the same workspace

The activity timeline is populated from `activity`, `tool_start`, `tool_result`, `assistant_note`, and `plan_ready` events.
Dot-prefixed paths stay hidden in the browser view unless the user explicitly targets a hidden path.

## File viewer and editor

Selecting a workspace file opens the inline viewer/editor in the workspace area.

Current capabilities include:

- editable text files
- markdown preview
- HTML preview
- image preview for files such as PNG, JPG, GIF, SVG, and WebP
- delimited file previews
- spreadsheet summaries with sheet switching
- PDF preview
- save-back into the current conversation workspace

When the assistant uses `workspace.render`, the UI automatically opens the generated HTML file. Previewable artifacts detected after `workspace.run_command`, especially plots and other generated images, can also auto-open in the inline viewer so the user sees more of what happened during execution.

## Attachments and workspace artifacts

Files are uploaded before send and stored in the active workspace. The composer shows them as removable chips until the request is sent.

That means attachments are first-class workspace inputs:

- the assistant sees summarized attachment context
- uploaded files appear in the workspace tree
- follow-up turns can keep working with those artifacts

Recorded audio follows the same path. If the user records audio without typing text, the client sends a small prompt asking the assistant to review the attached recording.

## Slash commands

The composer has a slash menu for common workflows:

- `/search`
- `/grep`
- `/plan`
- `/code`
- `/pip`

These commands take a structured execution path instead of only inserting prompt text.

## Deep-mode UX

Deep mode is an explicit inspect/plan/execute/verify flow rather than just “think harder.”

The UI surfaces:

- a reasoning-effort control
- streaming activity updates across phases
- structured build-step progress
- plan previews with editable build steps
- direct approval of an execution plan from the composer

Document mode hides approval affordances and routes the visible draft through the background agent flow automatically.

## Related files

- `src/web/index.html` — UI structure
- `src/web/app.js` — client runtime, activity rendering, workspace tools, slash commands, and voice
- `src/web/style.css` — layout and component styling
- `app.py` — chat, workspace, and voice APIs
