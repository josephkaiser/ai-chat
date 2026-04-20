# UI Features

The shipped frontend is now a slimmer workspace shell. It focuses on four things: choosing a workspace, reviewing replay-triage signals, chatting with the backend, and previewing workspace files.

## Main layout

`src/web/index.html` renders a three-part shell:

- a left sidebar for workspace selection, replay triage, and chat history
- a central chat panel with the message stream and composer
- a right-side viewer panel for browsing and previewing workspace files

The mobile/compact layout can toggle the workspace sidebar and file viewer independently.

## Current chat flow

The bundled frontend is intentionally light:

- the composer is a plain textarea plus send/stop button
- messages stream over `/ws/chat`
- the client sends turns with `mode: "deep"` and `auto_approve_tool_permissions: true`
- runtime progress is mostly shown as short status text in the composer hint area

The current browser client does not expose the older voice, slash-command, or plan-editor surfaces described in earlier docs.

## Workspace selector and chat list

The sidebar includes:

- a workspace selector
- a workspace refresh button
- a settings button
- a conversation list with rename/delete actions
- a `New Chat` button

The only browser preference currently persisted in `localStorage` is `lastWorkspaceId`.

## Replay Triage panel

One newer UI surface is the **Replay Triage** panel.

It shows:

- replay capture counts and failure counts
- a recommended next fix
- top triage buckets
- recent failure samples
- a drill-down into one captured replay case

If a replay capture file lives inside the workspace under `.ai/context-evals/`, the panel can jump straight to that file in the viewer.

## File browser and previewer

The viewer panel lets the user browse the active workspace and preview files inline.

Current preview types include:

- plain text and code
- Markdown
- HTML
- images
- PDFs
- CSV tables
- archive entry listings

Spreadsheet previews are served by the backend through the workspace API, even though the current frontend keeps the presentation simple.

When the backend reports `tool_result.payload.open_path`, the frontend automatically opens that file in the viewer. This is how rendered HTML, plots, and other generated artifacts can pop into view after a tool run.

## Settings

The current settings overlay is intentionally small. It shows a lightweight runtime summary and exposes a single destructive action:

- `Reset App Data`

That reset clears chats, workspaces, and related runtime state through `/api/reset-all`.

## Related files

- `src/web/index.html` — shell structure
- `src/web/app.ts` — client runtime and event handling
- `src/web/style.css` — responsive layout and visual system
- `src/python/harness.py` — APIs and WebSocket events consumed by the client
