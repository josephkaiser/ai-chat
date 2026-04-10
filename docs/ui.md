# UI Features

The UI is a compact workspace-oriented chat surface: chat, progress, files, attachments, plan approval, and voice controls all live in one place.

## Main surface

The composer supports:

- reasoning effort switching (`Low` / `High`)
- attachments uploaded into the conversation workspace
- browser-recorded audio attachments from the mic button
- slash commands for common coding tasks
- send-or-interrupt behavior from the primary action button

The client also preserves several preferences in `localStorage`, including feature toggles, speech playback settings, and workspace panel visibility.

## Settings and feature toggles

The Settings panel exposes:

- **Agent Dev Tools** — allow workspace-aware tool execution
- **Workspace Panel** — show or hide the workspace UI
- **Local RAG** — allow conversation-history recall
- **Web Search** — allow freshness-sensitive web lookups
- **Auto-Speak Replies** — queue fresh assistant replies through server TTS when available
- **Speech Speed** — adjust playback speed for generated audio
- **Appearance** and **System Prompt** controls
- **Reset App Data** danger zone

## Workspace panel

When agent tools are enabled, the workspace panel shows:

- an **Activity Log** with harness phases, tool calls, plan events, and finalization markers
- a **Workspace Tree** for files created, uploaded, or edited during the conversation
- refresh and download actions for the current workspace

The activity timeline is populated from `activity`, `tool_start`, `tool_result`, `assistant_note`, and `plan_ready` events.

## File viewer and editor

Selecting a workspace file opens the inline viewer/editor in the workspace area.

Current capabilities include:

- editable text files
- markdown preview
- HTML preview
- delimited file previews
- spreadsheet summaries with sheet switching
- save-back into the current conversation workspace

When the assistant uses `workspace.render`, the UI automatically opens the generated HTML file.

## Attachments and workspace artifacts

Files are uploaded before send and stored in the conversation workspace. The composer shows them as removable chips until the request is sent.

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

## Voice features

The built-in voice UX is split across two paths:

- browser microphone capture records locally and uploads an audio attachment into the workspace
- spoken reply playback calls `/api/voice/speak` and plays the returned server audio file

When auto-speak is enabled, fresh assistant replies are spoken in order so one reply does not cut off another.

## Related files

- `static/index.html` — UI structure
- `static/app.js` — client runtime, activity rendering, workspace tools, slash commands, and voice
- `static/style.css` — layout and component styling
- `app.py` — chat, workspace, and voice APIs
