# Plan: File-Session-First Draft Runtime

## Purpose

Refactor the app from a chat-first coding assistant into a **file-session-first document runtime**:

- the **user edits a durable draft/spec file**
- the **agent creates and maintains a separate generated file**
- the **draft is the source of intent**
- the **agent runtime treats the draft, generated file, search results, tests, git state, and versions as external environment**

This is the authoritative working plan for implementing that product direction.

## Product Thesis

The app should feel like a lightweight document tool:

- one filename at the top, like `index.html`
- one draft editor where the user writes natural-language structure, outline, and wishes
- one generated output view beside it
- no prompt tennis in a visible chat window

The user is not “chatting with a coding agent.” The user is **editing an evolving brief** and the agent is **interpreting that brief into a better static artifact**.

## Core Principles

### 1. The draft is the prompt

The visible draft document is the agent’s prompt surface.

- do not bounce the prompt back and forth through chat
- do not depend on the user re-explaining intent in a transcript
- store the prompt as a durable file in the workspace

### 2. The prompt lives outside the model window

We want an RLM-like runtime shape:

- the draft/spec file lives in the environment, not only in context
- the generated artifact lives in the environment
- intermediate state lives in durable storage
- the model receives focused slices, summaries, diffs, and metadata

This keeps the model context narrow while letting the product operate over long-lived, growing files and histories.

### 3. The file session is the primary unit

Each target file has a durable **file session**.

A file session owns:

- target path
- hidden draft/spec path
- generated file path
- attached conversation/session id for hidden agent turns
- version history
- queued jobs
- search/test/git memory

### 4. Most user edits should stay local

Not every draft change should trigger a full rewrite.

Default behavior should prefer:

- inline patch
- section patch
- targeted regeneration

Full-file regeneration should be reserved for genuine structural changes.

### 5. Foreground and background work must be separate

Fast reactive work and slow exploratory work should not share the same loop.

## Target Runtime

### Foreground lane

Purpose:

- respond to live user edits
- keep the generated file aligned with the latest draft structure
- preserve flow during rapid typing

Characteristics:

- debounced
- narrow context
- low latency
- single-file by default
- minimal tool use

Typical operations:

- diff draft snapshot
- classify change scope
- patch one section of the target file
- validate syntax or renderability
- refresh preview

### Background lane

Purpose:

- improve quality without interrupting the user
- search for better supporting content
- test alternatives
- run validations
- compare candidate outputs

Characteristics:

- queue-backed
- snapshot-based
- cancellable / supersedable
- can use search, tests, and git-backed experiments

Typical operations:

- web or workspace search
- content expansion
- alternative generation
- lint/test/render checks
- git experiment branches or worktrees
- reviewer/evaluator passes

## Execution Modes

### Live

Used for rapid user edits.

- target: very fast
- no heavy search
- no broad testing
- patch the current file from the latest draft diff

### Build

Used for a stronger realization pass.

- may rework structure
- may run lightweight validation
- still focused on the active file session

### Research

Used in background.

- search for better supporting material
- gather references or examples
- prepare candidate improvements

### Optimize

Used in background.

- run tests
- compare alternatives
- review or score candidates
- promote only better results

## Data Model

### File sessions

`file_sessions` should remain the durable source of file identity.

Each record should own:

- `id`
- `workspace_id`
- `path`
- `spec_path`
- `conversation_id`
- `created_at`
- `updated_at`
- `title`

### File session jobs

Add durable `file_session_jobs` for both foreground and background work.

Each job should include:

- `id`
- `workspace_id`
- `file_session_id`
- `lane` = `foreground | background`
- `job_kind`
- `status` = `queued | running | completed | failed | canceled | superseded`
- `payload_json`
- `title`
- `source_conversation_id`
- `created_at`
- `updated_at`
- `started_at`
- `finished_at`
- `error_text`

## Queue Rules

### Foreground jobs

- represent the current live draft-to-output realization
- new foreground work supersedes stale foreground work for the same file session
- should track the latest draft snapshot, not arbitrary old state

### Background jobs

- operate on snapshots
- must not overwrite newer user intent blindly
- should rebase or re-evaluate before promotion

## Search, Testing, and Git

These are background capabilities first, not the default foreground loop.

### Search

Use search to improve:

- factual content
- examples
- references
- implementation quality
- reusable project content

### Testing and review

Use testing and review to:

- verify generated code or structured files
- compare alternatives
- prevent regressions

### Git

Use git as an experiment engine:

- branch or worktree for candidate variations
- compare outputs
- promote only accepted results

Git is internal infrastructure here, not the primary user-facing model.

## UI Direction

The user-facing surface should stay minimal:

- filename header
- draft editor
- generated output view

Hidden or secondary infrastructure:

- job queues
- search results
- testing outputs
- git experiments
- internal agent traces

Version history and file explorer stay useful, but they support the file session rather than exposing chat transcripts.

## Implementation Phases

### Phase 1: File-session foundation

- file session becomes the primary unit of identity
- backend owns file-to-session mapping
- frontend follows backend session identity
- draft and generated file are clearly separated

Status:

- in progress

### Phase 2: Durable queue

- add `file_session_jobs`
- support foreground and background lanes
- supersede stale foreground work
- expose job state through backend APIs

Status:

- started

### Phase 3: Foreground realization runtime

- classify draft diffs
- choose `inline_patch | section_patch | regenerate`
- keep the generated file close to the draft with low latency

### Phase 4: Background runtime

- research lane
- optimize lane
- candidate generation and evaluation
- promotion rules for accepted improvements

### Phase 5: Search, test, and git integration

- background search
- validation/testing
- git-backed experiments
- reviewer/evaluator passes

### Phase 6: Remove chat-first assumptions

- rename remaining chat/conversation concepts in product copy
- demote or remove visible transcript-centric UI
- treat hidden conversation state strictly as runtime plumbing

## Immediate Next Steps

1. Finish wiring the new durable file-session job queue into the live draft realization flow.
2. Add job listing and status use on the client side for hidden runtime bookkeeping.
3. Rename more backend/frontend “conversation/chat” assumptions to “file session” where product-facing.
4. Add a first background job type for research or optimization snapshots.
5. Introduce candidate evaluation before background work can replace the visible generated file.

## Success Criteria

- The user edits a draft/spec file and sees a generated file update beside it.
- The visible draft acts as the durable prompt.
- The generated file is improved without requiring visible chat turns.
- File identity, versions, and hidden agent context persist through backend-managed file sessions.
- Fast foreground updates do not require large-context prompt stuffing.
- Slower search/test/git work can happen in the background and only promote better results.
