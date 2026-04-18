# Plan: Path-backed workspaces with a catalog UI (single shared userspace)

## Purpose

Refactor the app so **durable execution state** (filesystem + managed Python env + command sandbox) is keyed to a **stable workspace** tied to a **real root directory**, while the **chat transcript** stays ephemeral per session. The UI presents **named workspaces** (catalog UX); persistence follows **path-keyed** semantics (Implementation model 3 in UI, model 2 on disk).

This document is the **authoritative checklist** for agents implementing the change.

## Goals

- **Workspace-first:** Opening or creating a “workspace” picks a canonical **absolute root path** on the server host. All `workspace.*` tools, uploads, viewer, and downloads operate under that root (subject to existing path-safety checks).
- **Fresh-chat friendly:** New conversations **attach** to an existing workspace row; users are not required to reuse old chats to continue work. Multiple chats may reference the same workspace over time.
- **Catalog UI:** Users see a list of named workspaces (create, rename in UI, optional “open existing folder,” remove from list). Display names are **not** the source of truth; **resolved path** is.
- **Shared userspace:** No authentication or per-user isolation in this phase. Everyone hitting the same server shares one logical catalog (acceptable for local/single-operator deployments).

## Non-goals (this phase)

- Multi-user accounts, ACLs, or per-row permissions.
- Remote workspaces over SSH or cloud mount protocols.
- Automatic git hosting or repo hosting product features beyond what the agent already does via commands.
- Replacing SQLite with another DB.

## Current state (as of this plan)

- `src/python/harness.py` implements SQLite schema (`conversations`, `runs`, …). Each **conversation** has at most one **run** (`runs.conversation_id` is `UNIQUE`). `ensure_run_for_conversation` materializes `RUNS_ROOT/<run-id>/` and `…/workspace/`.
- `get_workspace_path(conversation_id)` and tool execution ultimately resolve the workspace directory from that run.
- Managed Python environments live under `MANAGED_PYTHON_ENVS_ROOT` keyed by **conversation id** (`get_managed_python_env_path`), not by workspace path.
- REST routes under `/api/workspace/{conversation_id}/…` and WebSocket chat identify the workspace implicitly via **conversation id** (`docs/api.md`).

## Target architecture

### Data model

Introduce a first-class **`workspaces`** (name TBD; avoid clashing with env `WORKSPACE_ROOT`) table, for example:

| Column | Notes |
|--------|--------|
| `id` | Stable UUID primary key (string). |
| `display_name` | User-facing label for the catalog. |
| `root_path` | Canonical **absolute** path; normalized and validated on write. Unique per server deployment unless explicitly allowed otherwise. |
| `created_at` / `updated_at` | Metadata. |

**Link conversations to workspaces:**

- Add `conversations.workspace_id` (nullable at first for migration, then required for new rows).
- Either **drop** `runs.conversation_id UNIQUE` and allow many conversations per run, or **merge** “run” into “workspace execution record.” Preferred direction: **one row in `runs` (or renamed table) per workspace**, not per conversation:
  - `runs.id` or `workspace_id` owns `sandbox_path` where `sandbox_path` is the **same as** `workspaces.root_path`, or `sandbox_path` points at a designated subtree (see “Layout” below).

**Managed Python environment:** Key `venv` location by **`workspace_id`** (or stable hash of resolved `root_path`), **not** `conversation_id`. Optionally keep a symlink or legacy migration path from old per-conversation env paths.

### Layout options (choose one in implementation; document the choice in `docs/configuration.md`)

1. **Direct root:** `workspaces.root_path` **is** the directory the user chose (e.g. `~/dev/my-app`). Tools use it as cwd roots for file ops and commands. No nested `runs/.../workspace` unless temporarily for migration.
2. **Hosted mirror:** Each workspace has `root_path` under `WORKSPACE_ROOT/<workspace_id>/` with optional “import” from user path — more isolation, more sync complexity. Prefer (1) unless security requirements change.

Default recommendation for this repo: **(1) direct root**, aligned with “path-keyed.”

### API & routing

- Add REST endpoints to **CRUD workspace catalog entries** and **resolve** path normalization errors (duplicate path, missing directory, path escapes).
- Evolve workspace file APIs from `/api/workspace/{conversation_id}/…` to either:
  - **`/api/workspaces/{workspace_id}/…`** for static resources, and keep conversation only for chat history, or
  - Keep path shape but resolve via **lookup: conversation → workspace** (incremental migration path).

WebSocket `/ws/chat` should accept (or require) **`workspace_id`** alongside **`conversation_id`**, or derive workspace from conversation server-side once `conversations.workspace_id` is always set.

Update **`docs/api.md`** in the same PR as the new routes.

### Frontend (`src/web/app.js`, `static/index.html`, `static/style.css`)

- **Entry flow:** Before or alongside “new chat,” user picks **active workspace** from catalog, or creates one (display name + directory create or picker — exact UX left to implementer; must work in browser constraints: may need server-side “create folder under default root” if no native folder picker).
- **Persistence:** Client stores `workspace_id` in `localStorage` (or equivalent) as “last active workspace” for new chats.
- **Conversation list:** May show workspace name/label per thread; optional filter “chats in this workspace.”

### Backend harness refactor (`src/python/harness.py`, related `src/python/ai_chat/*`)

Central refactor: replace `get_workspace_path(conversation_id)` with **`get_workspace_path_for_conversation`** that:

1. Loads `conversation → workspace_id → root_path`, or
2. Resolves `workspace_id` directly for tool loops.

**Functions that today take `conversation_id` only for workspace/env resolution** (grep and update call sites systematically):

- Workspace read/write, `workspace.run_command` cwd and allowed paths.
- `build_workspace_command_env`, `normalize_command_for_managed_python`, managed venv creation, `delete_managed_python_env` (trigger on workspace delete, not only conversation delete).
- Attachment upload paths, zip download, `delete_run_workspace` semantics — split **“delete chat transcript”** from **“delete workspace disk”** (destructive; confirm in UI).

**Prompt copy** in `src/python/ai_chat/prompts.py`: adjust references from “conversation workspace” to “project workspace” where helpful; keep instructions about durable files.

### Voice and ancillary paths

- If voice artifacts are keyed by conversation only, keep that for isolation of audio; no need to move voice storage to workspace unless product requires it. Document any coupling.

## Migration

1. **DB migration:** Create `workspaces` table; backfill one workspace per existing conversation from current `runs.sandbox_path` / `get_workspace_path` behavior (derive `root_path` = `…/workspace` inside existing run layout).
2. **Link:** Set `conversations.workspace_id` for every existing conversation.
3. **Managed env:** Optionally copy or symlink old per-conversation venv to first workspace id mapping when first used; or regenerate venv on next `pip` (simpler but slower). Document tradeoff in PR.
4. **Disk:** Do not delete existing `runs/` trees until a later cleanup task; new workspaces use direct paths going forward.

Add a short **`docs/migration-workspaces.md`** or a section in **`docs/configuration.md`** describing one-time upgrade steps and rollback limits (optional; only if migration is non-trivial).

## Testing & verification

- **Unit tests:** Any new path normalization and ` workspaces.root_path` uniqueness constraints (`tests/`).
- **Integration:** Create two conversations attached to the same `workspace_id`; verify both see the same file tree and share the same managed env.
- **Backward compatibility:** One existing conversation still loads and tools resolve after migration.

## Documentation updates (required in scope)

- `docs/architecture.md` — persistence model section.
- `docs/api.md` — new/changed endpoints.
- `docs/harness.md` — workspace and managed Python keying.
- `docs/ui.md` — workspace catalog and “new chat” flow.

## Suggested implementation order

1. Schema + migration + internal resolver: `conversation_id` → `workspace` → `root_path` (keep old code paths working via adapter).
2. Key managed Python env by `workspace_id`; migration shim from old paths.
3. REST workspace catalog + switch workspace file APIs to `workspace_id` (or dual-read old routes).
4. Frontend: catalog UI + attach conversation to workspace on create.
5. Remove dead conversation-only workspace assumptions; tighten `DELETE` semantics (conversation vs workspace).
6. Docs sweep and manual QA checklist.

## Risks / explicit decisions for implementers

- **Concurrent writes:** Same workspace from two chats → same as two users editing one folder; acceptable for v1; optional file-lock or “workspace busy” later.
- **Renaming host paths:** If user moves folder on disk, DB `root_path` is stale; provide “Relink path” or a `.ai-chat-workspace.json` marker file **inside** the root with `workspace_id` for repair (optional enhancement).
- **Security:** Single shared catalog does not reduce path traversal risk; keep `STRICT_WORKSPACE_COMMAND_PATHS` and existing containment checks, now against `workspaces.root_path`.

## Success criteria

- User can **create named workspace → pick path → start new chat** without reopening an old thread, and **see prior files** in that root.
- **Managed pip installs** persist for that workspace across **new** conversations sharing the same workspace id.
- **Deleting a chat** does not delete the shared workspace disk by default.
