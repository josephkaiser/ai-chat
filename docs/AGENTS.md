# AGENTS.md

# Wolfy Agent Operating Guide

## Mission

You are working on **Wolfy**, a lightweight, portable, open-source local AI chat app designed to run on consumer hardware, especially a single GPU such as an RTX 4090.

Wolfy is intended to become a strong open-source kernel for:

- local chat
- personal RAG
- useful assistant workflows
- learning how local AI systems work
- future capability expansion as consumer compute improves

Wolfy is **not** trying to become a bloated framework, enterprise platform, or overbuilt orchestration system.

The design target is the “Mazda RX-7” of local AI software:
compact, capable, efficient, understandable, and extensible through clever design rather than bloat.

---

## Role boundaries

### Agent responsibilities
The agent develops and improves the application and repository itself.

The agent may:
- refactor code
- improve structure and clarity
- add docs inside the repo
- improve local startup flow
- add instrumentation and benchmarks
- add extension points for RAG and tools
- add tests and CI workflow files
- improve portability and developer experience

The agent does **not** own:
- production deployment
- GitHub repo settings
- GitHub Actions secrets
- package registry credentials
- social media posting
- release operations
- hosted infrastructure setup
- domain, auth, billing, or business operations

Those are human responsibilities.

---

## Product thesis

Wolfy should become:

> A lightweight, portable, open-source personal AI chat app and learning tool that helps people run useful local assistants with personal knowledge and real work capability on consumer hardware.

All work should support that direction.

---

## Core principles

### 1. Simplicity over sprawl
Prefer a few understandable components over introducing more systems, libraries, abstractions, or services.

### 2. Portability over lock-in
Changes should make Wolfy easier to run on real machines. Avoid unnecessary dependence on one vendor, platform, or hosted pattern unless clearly isolated.

### 3. Learning value matters
The repo should teach. Favor readable code, visible boundaries, and understandable docs.

### 4. Capability-per-pound
Every added feature must justify its complexity. Wolfy should gain meaningful capability without becoming heavy.

### 5. Real work matters
Wolfy is not only a chat UI. It should become a useful personal assistant foundation, especially through retrieval and practical tooling.

### 6. Preserve project character
Do not flatten Wolfy into a generic template. Keep it human, pragmatic, and slightly playful without becoming unserious.

### 7. Refactor with restraint
Prefer small, reversible refactors. Avoid giant rewrites unless they are clearly necessary.

---

## Current project assumptions

Assume the repo currently has:

- a working chat app
- 2 Docker containers
- a bash script used to compose/start the app
- open-source code hosted on GitHub
- a target runtime on a single GPU
- an early-stage architecture that needs tightening, not replacement

Work from the repo as it exists today. Do not invent a brand-new platform architecture unless explicitly asked.

---

## Agent goals for this enhancement cycle

The current enhancement cycle aims to make Wolfy:

- easier to understand
- easier to run
- easier to fork
- easier to benchmark
- easier to extend
- better documented
- ready for personal RAG and assistant growth
- credible as a small but serious open-source project

---

## Priority workstreams

### A. Repository framing and first-run experience
Improve the repo so a new developer can understand what Wolfy is and how to run it very quickly.

Expected outputs may include:
- stronger `README.md`
- a clear hook near the top
- a 1-minute quickstart
- exact requirements
- common issues
- short explanation of why Wolfy exists

### B. Startup and environment hardening
Make the current startup path deterministic and inspectable.

Expected outputs may include:
- clearer startup script behavior
- improved environment variable handling
- `.env.example`
- explicit local persistence paths
- obvious ports and config knobs
- clearer health expectations

### C. Minimal architecture documentation
Make the repo legible to contributors.

Expected outputs may include:
- `ARCHITECTURE.md`
- concise explanation of container responsibilities
- request flow documentation
- clear extension points for retrieval/tools/metrics

### D. Refactor for extension readiness
Refactor only enough to make future work clean.

Target boundaries include:
- UI
- request handling
- inference/model adapter
- retrieval
- tool logic
- persistence/session logic
- config
- metrics/logging

### E. Performance and observability
Support a credible “runs on a 4090” story.

Prefer measuring:
- time to first token
- total response time
- output token count if available
- tokens per second if available
- retrieval latency if used
- model load/init time if measurable
- memory usage if easy to expose

A lightweight debug or stats surface is acceptable. Do not add a heavy observability stack.

### F. GitHub Actions support files
The agent may add workflow files to the repo for:
- CI
- image build/release

The agent should not assume those workflows are live until the human configures repo settings and secrets.

### G. Personal RAG foundation
Prepare Wolfy to support personal knowledge grounding.

This should be modular and local-friendly. Avoid overbuilding a retrieval platform.

### H. Assistant capability foundation
Prepare a small, understandable tool/action layer for useful work.

Do not introduce a giant agent framework.

### I. Contributor friendliness
Help the repo become easier for others to contribute to.

Expected outputs may include:
- `CONTRIBUTING.md`
- roadmap docs
- developer notes
- good extension point docs

---

## Execution order

When unclear, prefer this order:

1. Improve README and quickstart
2. Harden startup/config/environment flow
3. Add architecture documentation
4. Refactor for clean boundaries
5. Add lightweight metrics and benchmark support
6. Add CI and release workflow files
7. Improve contributor docs
8. Prepare personal RAG foundation
9. Prepare assistant/tool foundation

---

## Definition of done

A task is aligned when it moves Wolfy toward these outcomes:

- a new developer can understand Wolfy quickly
- local startup is less fragile
- configuration is clearer
- the codebase is easier to extend
- benchmarking is possible
- personal RAG has a clean home
- assistant/tool capability has a clean home
- repo docs reflect reality
- the project feels like a durable open-source kernel, not a one-off experiment

---

## Non-goals

Avoid drifting into these unless explicitly instructed:

- Kubernetes migration
- hosted multi-tenant SaaS architecture
- billing/auth/platform work
- per-user GPU container orchestration
- giant framework rewrites
- enterprise complexity
- growth hacking or marketing strategy
- abstract “AI operating system” ambitions

---

## Development style

### Make reality visible
Prefer explicit docs, visible config, and understandable control flow.

### Keep changes scoped
Do not mix unrelated refactors and new features in one change.

### Favor stable interfaces
Where future growth is likely, introduce small clear interfaces rather than hard-coding new paths.

### Document assumptions
If a design depends on assumptions about models, runtime, storage, or hardware, document them.

### Preserve behavior unless improving it
Refactors should not casually break the current workflow.

---

## File targets agents may create or improve

Common repo artifacts for this phase include:

- `README.md`
- `.env.example`
- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
- `docs/CONFIG.md`
- `docs/BENCHMARKS.md`
- `docs/RAG.md`
- `docs/ROADMAP.md`
- `.github/workflows/ci.yml`
- `.github/workflows/release-images.yml`

---

## Coordination contract with the human operator

The human operator is responsible for:
- pushing to GitHub
- managing repo settings
- enabling GitHub Actions as needed
- configuring secrets and package permissions
- creating releases
- deploying production systems
- running social media and public updates
- deciding what is actually shipped externally

The agent should therefore:
- leave clear docs for human-required setup
- keep workflow files understandable
- note any required secrets, permissions, or manual repo settings
- avoid assuming access to external accounts or infrastructure

---

## Agent decision checklist

Before making a change, ask:

- Does this make Wolfy easier to run?
- Does this make Wolfy easier to understand?
- Does this improve usefulness on real consumer hardware?
- Does this preserve portability?
- Does this help future RAG or assistant capability?
- Does this avoid unnecessary system weight?
- Does this respect the human/operator boundary?

If the answer is mostly no, the change is probably out of scope.

---

## Tone

Wolfy should feel:
- compact
- practical
- technically honest
- open to tinkering
- a little playful
- not hype-heavy
- not corporate

Protect that tone in code comments, docs, and feature naming.
