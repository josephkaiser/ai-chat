# HUMAN.md

# Wolfy Human Operator Guide

## Purpose

This file is for the human operator responsible for turning Wolfy from a codebase into a living open-source project.

The human does not need to do the core application development work. The agent handles that.

The human is responsible for everything that connects the repo to the outside world:
- pushing code
- maintaining GitHub settings
- wiring up CI/release systems
- deciding what ships
- creating releases
- publishing prebuilt images
- documenting install steps that depend on external settings
- building in public
- running the Wolfy social account and public posts
- later, handling deployment and production operations

The human and the agent must work together:
- the agent improves the code and repo
- the human makes the repo visible, runnable, released, and public

---

## Division of labor

### Agent owns
- app development
- refactors
- internal docs
- architecture cleanup
- metrics support
- extension points
- local DX improvements
- workflow file creation

### Human owns
- GitHub repository configuration
- branch/release discipline
- package registry setup
- GitHub Actions permissions and secrets
- release publishing
- Docker image visibility and package settings
- deployment settings
- production decisions
- public communication
- social media account
- momentum and consistency

---

## Immediate human goals

Your near-term job is to help Wolfy become:

- understandable from the outside
- easy to try
- easy to trust
- visibly active
- publicly legible as a small serious open-source project

Do not optimize for business infrastructure yet.
Do not optimize for market capture yet.
Optimize for:
- coherence
- usability
- visibility
- consistency
- momentum

---

## Phase 1: set the repo up to be seen clearly

### 1. Clean up the GitHub repo presentation
Make sure the repo front page gives a strong first impression.

Actions:
- set a clear repo description
- pin the repo if relevant
- add topics/tags
- set the homepage field later if you create a demo site
- choose a clean open-source license if not already set
- add a good social preview image later when visuals exist

Suggested topical tags:
- local-llm
- ai-chat
- docker
- open-source
- rag
- assistant
- self-hosted
- gpu
- nvidia
- llama or relevant model family if appropriate

### 2. Review the README after agent updates it
Your job is not to rewrite it from scratch unless needed. Your job is to make sure it matches public reality.

Check:
- is the hook strong?
- is quickstart accurate?
- are requirements honest?
- does it promise only what works?
- does it fit the tone you want?

---

## Phase 2: wire up GitHub Actions and package publishing

### 3. Enable GitHub Actions appropriately
When the agent adds workflow files, you must verify repository settings allow them to run.

Actions:
- make sure Actions are enabled for the repo
- review workflow permissions
- ensure `GITHUB_TOKEN` has permissions needed for package publishing if using GHCR
- verify whether package write permissions are allowed in workflow settings
- check branch protection rules if you want CI required later

### 4. Set up release image publishing
If you want users to pull images instead of building locally, wire up the registry side.

Actions:
- decide whether to use GHCR or Docker Hub
- if using GHCR, confirm package visibility and naming
- if using Docker Hub, create/access credentials and add needed secrets
- test a tagged release flow
- verify the image can actually be pulled publicly
- document the exact image name in the README

### 5. Validate the workflows manually
Do one dry run yourself.

Check:
- CI builds successfully
- smoke test actually means something
- release workflow tags are correct
- image names are consistent
- README instructions match the real image names and commands

---

## Phase 3: shipping discipline

### 6. Establish a simple release pattern
Do not overcomplicate versioning.

Use something like:
- `main` for current active development
- tags like `v0.1.0`, `v0.2.0`
- optional `latest` image tag
- optional `dev` image tag later if useful

Your job:
- cut releases intentionally
- write short release notes
- note what changed and what users should try
- avoid silent breaking changes without explanation

### 7. Keep a lightweight public roadmap
You do not need a giant PM system.

Do:
- open a few issues that reflect real priorities
- label a few “good first issue” items
- keep one visible roadmap doc or milestone set
- make the repo feel alive

---

## Phase 4: build in public

### 8. Create and run the Wolfy public account
This can be on X, Bluesky, or whatever platform you prefer. The goal is not growth hacking. The goal is consistent public breadcrumbs.

The account should:
- show progress
- share benchmarks
- show design choices
- show tradeoffs
- show experiments
- show the project’s personality

You are not trying to posture as a startup CEO.
You are documenting a real build.

### 9. Posting strategy
Post artifacts, not vague claims.

Good post types:
- quick demo clips
- benchmark screenshots
- “I simplified X”
- “Wolfy now runs in Y setup”
- “Added personal RAG groundwork”
- “Trying to make local AI useful without a giant stack”
- “Runs on a 4090 with these numbers”
- “Made the startup path much cleaner”

Bad post types:
- generic hustle posts
- vague “big things coming”
- inflated claims
- pretending features work if they don’t

### 10. Posting cadence
Consistency beats volume.

A workable cadence:
- 1-3 small updates per week
- 1 bigger milestone post when you ship something notable
- occasional screenshot/demo/benchmark post
- occasional reflective post on design choices

### 11. Social voice
Keep it:
- practical
- curious
- honest
- technically grounded
- slightly playful
- not overmarketed

Good tone:
- “Built this because I wanted local AI chat to be simpler and more hackable.”
- “Trying to make a useful personal AI that runs on real hardware.”
- “Still early, but this is getting fun.”

---

## Phase 5: benchmark and proof loop

### 12. Turn metrics into public proof
Once the agent adds metrics or a benchmark path, your job is to package it into understandable proof.

Publish things like:
- setup used
- GPU used
- model used
- quantization/settings
- time to first token
- tokens/sec
- rough memory use
- any retrieval latency if relevant

Keep it reproducible. Avoid hype.

### 13. Record demos
When the app is stable enough:
- record a short terminal-to-browser quickstart
- record a short response demo
- record a benchmark/stats walkthrough
- later add visual proof to the README

These assets are reusable across the repo, release notes, and social posts.

---

## Phase 6: human deployment responsibilities later

This is not the current focus, but when you are ready, deployment becomes your responsibility.

That includes:
- cloud or server choice
- secrets management
- production image rollout
- DNS/domain
- reverse proxy
- auth
- monitoring
- logs
- uptime
- incident response

The agent can help prepare code and deployment files, but the human must make the final operational choices and configure the external systems.

---

## Human checklist before each public ship

Before you push or announce something, check:

- Does the README match reality?
- Does quickstart actually work?
- Do workflow files pass?
- Are release tags/version names consistent?
- Are images publicly pullable if promised?
- Are claims in the post true?
- Is there at least one concrete thing a stranger can try?
- Does this preserve the project’s identity?

---

## Human checklist before enabling a release image flow

- registry chosen
- registry account ready
- package visibility understood
- repo workflow permissions checked
- secrets added if needed
- tags agreed on
- one release tested end-to-end
- pull instructions verified from a clean machine if possible

---

## Human checklist for social/media operation

- profile name matches project
- bio explains what Wolfy is
- repo link is present
- first few posts establish the project clearly
- screenshots or terminal clips saved as reusable assets
- keep a simple note file of future post ideas
- post when there is real progress, not because you feel pressure

---

## Harmony contract between human and agent

### The human should give the agent
- clear priorities
- current repo reality
- constraints
- what is actually in scope
- feedback on what should ship next

### The agent should give the human
- code and docs that are easier to ship
- clear notes on manual setup needed
- understandable workflow files
- minimal operational surprises
- measurable progress

### Shared rule
Do not let the public story drift away from the actual repo.

If the repo is simple, say it is simple.
If a feature is partial, say it is partial.
If something is experimental, say it is experimental.

That honesty is part of the brand.

---

## Suggested near-term operator sequence

1. Let the agent improve README, startup flow, docs, and metrics path
2. Review and merge those changes
3. Configure GitHub Actions and package publishing
4. Test a clean release flow
5. Publish one release with clear notes
6. Make 2-3 public posts showing what Wolfy is and why it exists
7. Keep posting progress as the agent improves the app
8. Add screenshots/demo later when ready
9. Grow the repo through consistency, not pressure

---

## Final operator reminder

You are not required to pretend Wolfy is bigger than it is.

Your job is to make it visible, runnable, and credible while the agent makes it better.

That is enough.
