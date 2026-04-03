# UI Features

The original docs describe a smaller chat interface. The current UI is closer to a lightweight agent workspace: chat, activity timeline, file browser, editor, terminal, voice controls, model/runtime controls, and plan handoff all live in the same surface.

## Main chat surface

The main composer in `static/index.html` and `static/app.js` now supports:

- model profile switching from the composer
- reasoning effort switching (`Low` / `High`)
- attachments uploaded into the conversation workspace
- slash commands for common coding tasks
- voice dictation
- send-or-interrupt behavior from the primary action button

The app also preserves several client-side preferences in `localStorage`, including feature toggles, speech playback settings, and workspace panel visibility.

## Settings and feature toggles

The Settings panel exposes the main runtime affordances:

- **Agent Dev Tools** enables the assistant's workspace-aware tool flow.
- **Workspace Panel** shows or hides the right-side workspace UI.
- **Local RAG** allows conversation-history search for recall-heavy prompts.
- **Web Search** allows freshness-sensitive web lookups.
- **Auto-Speak Replies** plays assistant responses through the server TTS pipeline when available.
- **Speech Speed** adjusts playback rate for generated audio.

These toggles affect both UI behavior and the server-side feature flags sent with each request.

## Workspace panel

When agent tools are enabled and the viewport is large enough, the UI can open a dedicated workspace panel.

It includes:

- an **Activity** timeline showing harness phases, tool calls, plan events, and finalization markers
- a **Workspace Tree** for files created, uploaded, or edited during the conversation
- refresh and download actions for the current workspace

The activity timeline is populated from WebSocket `activity`, `tool_start`, `tool_result`, `assistant_note`, and `plan_ready` events.

## File viewer and editor

Selecting a workspace file opens a modal reader/editor.

Current capabilities include:

- editable text files
- markdown preview
- HTML preview
- delimited file previews
- spreadsheet summaries with sheet switching
- save-back into the current conversation workspace

This gives users a way to inspect or lightly edit generated artifacts without leaving the app.

## Terminal integration

The workspace UI also includes a PTY-backed terminal view tied to the current conversation workspace.

Key details:

- the browser connects to `/ws/terminal/{conversation_id}`
- command output streams live into the UI
- command runs launched by `workspace.run_command` are reflected in terminal activity
- the terminal can be cleared, resized, and reused for the active conversation workspace

This makes the harness feel observable instead of opaque when a turn runs local commands.

## Attachments and workspace artifacts

File attachments are uploaded before send and stored in the conversation workspace. The composer shows them as removable chips until the request is sent.

That means attachments are first-class workspace inputs rather than opaque blobs:

- the assistant sees a summarized attachment context in the prompt
- uploaded files appear in the workspace tree
- follow-up turns can keep working with those artifacts

Assistant replies can also be recovered into the workspace after the fact:

- the `Add to WS` action on an assistant message stages fenced code blocks into `recovered/.../files/`
- the app generates a companion `README.md` with the non-code notes from that reply
- this creates a reviewable staging area instead of forcing users to perfectly re-prompt the model

## Slash commands

The composer has a slash menu for common workflows. Current commands include:

- `/new`
- `/clear`
- `/web`
- `/files`
- `/read`
- `/edit`
- `/review`
- `/fix`
- `/plan`
- `/explain`
- `/attach`
- `/add`

Some commands also enable the related feature toggle automatically, such as turning on web search or agent tools before inserting the prompt template.

## Deep-mode UX

Deep mode is no longer just "think harder." The UI now exposes several orchestration artifacts:

- a reasoning-effort control in the composer
- streaming activity updates across inspect/plan/execute/verify phases
- structured build-step progress
- plan previews that can be loaded back into the composer for approval-first execution

On mobile, the title area doubles as a quick reasoning toggle, while the workspace panel is intentionally suppressed to keep the layout manageable.

## Voice features

The browser can use the server-side voice pipeline for both directions:

- record audio and send it to `/api/voice/transcribe`
- request spoken playback from `/api/voice/speak`

The UI detects runtime availability from `/api/voice/status` and updates controls accordingly. If STT or TTS is missing on the server, the relevant controls stay disabled and the user gets a clear note explaining why.

## Dashboard and model controls

The dashboard and composer expose operational controls that were not part of the original UI:

- live model/runtime health
- active model profile visibility
- model profile switching
- cache visibility
- model-library discovery with Hugging Face URL/repo downloads
- cached-model activation directly from the discovery page
- cached-model browsing and delete actions for non-profile models
- load timing details including elapsed time, prior average, and ETA
- vLLM restart actions
- model redownload actions

Together, these make the app usable as both a chat client and a small self-hosted model console.

## Related files

- `static/index.html` — UI structure
- `static/app.js` — client runtime, activity rendering, workspace tools, slash commands, voice, dashboard
- `static/style.css` — layout and component styling
- `app.py` — workspace, terminal, dashboard, and voice APIs
