# Deep Mode v2: Critique Loop + Dual-Model Diversity + Tiny Model Layer

## Goal

Upgrade the existing deep mode pipeline with three improvements:
1. **Critique/refinement loop** — self-correcting quality gate after review
2. **Dual-model agents** — two different models for diverse parallel thinking
3. **Tiny model layer** — a small fast model handling lightweight tasks (routing, critique, context compression) so the big models focus on generation

---

## Current State

- Single model: Qwen3-14B-AWQ via vLLM on one GPU (0.95 memory utilization)
- Pipeline: decompose → 2 parallel agents (same model) → review/merge → done
- All calls go through `vllm_chat_complete()` / `vllm_chat_stream()` hitting one vLLM instance

---

## New Pipeline

```
        ┌─────────────────────────────────┐
        │         TINY MODEL (0.6B)       │
        │  router · critique · compress   │
        └────────┬────────────────────────┘
                 │
          route (normal/deep)
                 │
           ┌─────┴──── deep ────┐
           ▼                     │
      decompose (14B)            │
           │                     │
   agent_a (14B) ──┐             │
                   ├── parallel  │
   agent_b (7B) ──┘             │
           │                     │
      review/merge (14B)         │
           │                     │
      critique (0.6B) ←────┐    │
           │               │    │
        pass? ── no ──→ refine (14B)
           │
          yes
           │
      stream final answer
```

### Three Model Tiers

| Tier | Model | Role | Speed |
|------|-------|------|-------|
| **Tiny (0.6B)** | Qwen3-0.6B-AWQ | Router, critique, context compression | ~0.2-0.5s per call |
| **Medium (7B)** | Qwen3-7B-AWQ or external API | Agent B (diverse second perspective) | ~3-5s per call |
| **Large (14B)** | Qwen3-14B-AWQ (current) | Decompose, Agent A, review, refine | ~5-8s per call |

### Phases

1. **Route** (NEW, tiny model) — Classify the query: does it need deep mode? What type of task is it? This lets the app auto-select deep mode for complex queries even if the user didn't toggle it, and skip deep mode overhead for simple questions.
2. **Compress** (NEW, tiny model, optional) — If conversation history is long (>6 messages), summarize it into a compact context block. Keeps agent prompts focused and within token budgets.
3. **Decompose** (14B) — Same as today. Splits query into two sub-tasks.
4. **Parallel agents** — Agent A (14B) and Agent B (7B or external API). Different model sizes/families produce genuinely different reasoning.
5. **Review/merge** (14B) — Combines both outputs into a draft answer.
6. **Critique** (NEW, tiny model) — Fast pass/fail check on the draft. The 0.6B model can output `{"pass": true/false, "issues": "..."}` in ~0.3s vs ~2s for 14B. This is essentially classification — a tiny model's sweet spot.
7. **Refine** (14B, conditional) — If critique fails, the big model fixes the specific issues. Re-critique. Max 2 cycles.
8. **Stream** — Final answer to the user.

---

## GPU Memory Layout (4090 24GB)

Three vLLM containers sharing one GPU:

```
┌──────────────────────────────────────────────────┐
│                   RTX 4090 (24GB)                │
├──────────────────────┬───────────┬───────────────┤
│   Qwen3-14B-AWQ     │ Qwen3-7B  │  Qwen3-0.6B  │
│   ~8.5GB weights     │ ~4.5GB wt │  ~0.5GB wt   │
│   +4.7GB KV cache    │ +1.5GB KV │  +1.4GB KV   │
│   = ~13.2GB          │ = ~6.0GB  │  = ~1.9GB    │
│   (0.55 util)        │ (0.25 ut) │  (0.08 util) │
├──────────────────────┴───────────┴───────────────┤
│   CUDA overhead + fragmentation: ~2.9GB          │
│   Total: 13.2 + 6.0 + 1.9 + 2.9 = 24.0GB       │
└──────────────────────────────────────────────────┘
```

| Container | Model | `--gpu-memory-utilization` | VRAM | `--max-model-len` | `--max-num-seqs` |
|-----------|-------|---------------------------|------|--------------------|------------------|
| `vllm` | Qwen3-14B-AWQ | 0.55 | ~13.2GB | 8192 | 8 |
| `vllm-7b` | Qwen3-7B-AWQ | 0.25 | ~6.0GB | 4096 | 4 |
| `vllm-tiny` | Qwen3-0.6B-AWQ | 0.08 | ~1.9GB | 2048 | 16 |

**Key tradeoff vs current setup:** The 14B drops from 0.95 → 0.55 utilization. This means:
- KV cache shrinks from ~14GB to ~4.7GB
- `--max-num-seqs` should drop from 16 → 8 (fewer concurrent sequences)
- Single-request latency is unchanged (same model, same weights)
- Throughput under concurrent load decreases — but deep mode is inherently sequential (one user waiting), so this matters less than it sounds

The 7B only needs to handle one request at a time (Agent B), so 4 max seqs with a short context is plenty.

The 0.6B handles routing/critique/compression — all short-output tasks. 16 max seqs at 2048 context is generous for JSON classification.

### Fallback: 4B instead of 7B

If 0.55 utilization for the 14B is too constraining (truncated responses, OOM on long contexts), swap the 7B for Qwen3-4B-AWQ:

| Container | Model | `--gpu-memory-utilization` | VRAM |
|-----------|-------|---------------------------|------|
| `vllm` | Qwen3-14B-AWQ | 0.65 | ~15.6GB |
| `vllm-4b` | Qwen3-4B-AWQ | 0.15 | ~3.6GB |
| `vllm-tiny` | Qwen3-0.6B-AWQ | 0.08 | ~1.9GB |

This gives the 14B more breathing room (0.65 vs 0.55) while still getting a meaningfully different Agent B.

---

## Tiny Model Layer (Qwen3-0.6B)

### Why a Tiny Model

Several pipeline tasks don't need 14B-level intelligence — they're classification, extraction, or summarization. Using 14B for these wastes GPU cycles and adds latency. A 0.6B model handles them in a fraction of the time on minimal VRAM.

### What It Handles

| Task | Input | Output | Why tiny is enough |
|------|-------|--------|--------------------|
| **Query routing** | User message | `{"mode": "normal"\|"deep", "type": "code"\|"explain"\|"debug"\|"general"}` | Binary/multi-class classification — tiny models excel at this |
| **Critique** | Draft response + original question | `{"pass": true/false, "issues": "..."}` | Pass/fail judgment with brief justification — no generation needed |
| **Context compression** | Last N messages of conversation history | 2-3 sentence summary | Summarization of structured text — well within 0.6B capability |

---

## Dual-Model Agent Strategy

Agent A (14B) and Agent B (7B) run in parallel via `asyncio.gather()`, each as a request to its own vLLM container. The 7B reasons differently — more concise, more direct, sometimes catching practical issues the 14B over-thinks. The 14B reviewer merges the best of both.

Optional external API override: when `AGENT_B_PROVIDER` is set to `"openai"` or `"google"`, Agent B uses an external model instead of the local 7B for maximum diversity. Falls back to local 7B (or 14B if 7B isn't configured).

---

## File Changes

### 1. `docker-compose.yml` — Multi-model vLLM + new env vars

Two approaches for serving multiple local models:

**Option A: Single vLLM with multi-model (preferred if supported by vLLM version)**
```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    ports:
      - "8001:8000"
    command: >
      --model Qwen/Qwen3-14B-AWQ
      --host 0.0.0.0
      --port 8000
      --gpu-memory-utilization 0.85
      --max-model-len 8192
      --enable-prefix-caching
      --max-num-seqs 16
      --enable-chunked-prefill
      --quantization awq_marlin
      --trust-remote-code
      --enforce-eager
    # ... (unchanged volumes, deploy, restart)

  vllm-tiny:
    image: vllm/vllm-openai:latest
    container_name: vllm-tiny
    ports:
      - "8002:8000"
    command: >
      --model Qwen/Qwen3-0.6B-AWQ
      --host 0.0.0.0
      --port 8000
      --gpu-memory-utilization 0.10
      --max-model-len 4096
      --enable-prefix-caching
      --max-num-seqs 32
      --quantization awq_marlin
      --trust-remote-code
      --enforce-eager
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  chat-app:
    # ...
    environment:
      - VLLM_HOST=http://vllm:8000/v1
      - MODEL_NAME=Qwen/Qwen3-14B-AWQ
      - TINY_VLLM_HOST=http://vllm-tiny:8000/v1
      - TINY_MODEL_NAME=Qwen/Qwen3-0.6B-AWQ
      - HF_CACHE_PATH=/cache/huggingface
      # Agent B model — "local" (uses AGENT_B_MODEL_NAME on vLLM), "openai", "google", or empty (same as primary)
      - AGENT_B_PROVIDER=
      - AGENT_B_MODEL_NAME=
      - AGENT_B_API_KEY=
    depends_on:
      - vllm
      - vllm-tiny
```

Note: GPU utilization split — 14B at 0.85 + 0.6B at 0.10 = 0.95 total. The 0.6B model is so small it barely impacts the 14B's throughput. If a local 7B is used for Agent B, that needs a second GPU or further reduction (14B at 0.55, 7B at 0.35, 0.6B at 0.05).

**Option B: Run 0.6B on CPU via llama.cpp (no GPU impact at all)**
If GPU headroom is too tight, run the tiny model on CPU. Qwen3-0.6B generates short outputs (JSON classification, brief summaries) so CPU latency is acceptable (~0.3-0.8s). Use an Ollama container or a llama-cpp-python sidecar instead of vllm-tiny.

### 2. `app.py` — Configuration for three model tiers

#### 2a. Config section — add after existing config vars

```python
# --- Tiny model (routing, critique, compression) ---
TINY_VLLM_HOST = os.getenv("TINY_VLLM_HOST", "").strip()
TINY_MODEL_NAME = os.getenv("TINY_MODEL_NAME", "").strip()
TINY_MODEL_ENABLED = bool(TINY_VLLM_HOST and TINY_MODEL_NAME)

# --- Agent B model (secondary agent for dual-model deep mode) ---
AGENT_B_PROVIDER = os.getenv("AGENT_B_PROVIDER", "").strip().lower()  # "local", "openai", "google", or ""
AGENT_B_MODEL_NAME = os.getenv("AGENT_B_MODEL_NAME", "").strip()
AGENT_B_API_KEY = os.getenv("AGENT_B_API_KEY", "").strip()
AGENT_B_ENABLED = bool(AGENT_B_PROVIDER and AGENT_B_MODEL_NAME)

logger.info("Tiny model: %s", f"{TINY_VLLM_HOST} ({TINY_MODEL_NAME})" if TINY_MODEL_ENABLED else "disabled")
logger.info("Agent B: %s", f"{AGENT_B_PROVIDER}/{AGENT_B_MODEL_NAME}" if AGENT_B_ENABLED else "disabled (using primary)")
```

#### 2b. Add `tiny_model_complete()` — after `vllm_chat_complete()`

```python
async def tiny_model_complete(messages: list, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Fast completion from the tiny model. Falls back to primary vLLM if tiny is not configured."""
    host = TINY_VLLM_HOST if TINY_MODEL_ENABLED else VLLM_HOST
    model = TINY_MODEL_NAME if TINY_MODEL_ENABLED else MODEL_NAME
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        resp = await client.post(
            f"{host}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

#### 2c. Add `agent_b_complete()` — after `tiny_model_complete()`

```python
async def agent_b_complete(messages: list, max_tokens: int = 4096, temperature: float = 0.25) -> str:
    """Completion for Agent B. Routes to local vLLM, external API, or falls back to primary."""
    if not AGENT_B_ENABLED:
        return await vllm_chat_complete(messages, max_tokens=max_tokens, temperature=temperature)

    if AGENT_B_PROVIDER == "local":
        # Different model on the same or different vLLM instance
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            resp = await client.post(
                f"{VLLM_HOST}/chat/completions",
                json={
                    "model": AGENT_B_MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    elif AGENT_B_PROVIDER in ("openai", "google"):
        if AGENT_B_PROVIDER == "openai":
            url = "https://api.openai.com/v1/chat/completions"
        else:
            url = "https://generativelanguage.googleapis.com/v1beta/chat/completions"
        headers = {"Authorization": f"Bearer {AGENT_B_API_KEY}"}
        body = {
            "model": AGENT_B_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unknown AGENT_B_PROVIDER: {AGENT_B_PROVIDER}")
```

#### 2d. Add query router — new function, called before mode selection

```python
async def route_query(message: str, history: list) -> dict:
    """Use tiny model to classify query complexity and type. Returns {"mode": "normal"|"deep", "type": "..."}."""
    context = ""
    if history:
        recent = history[-3:]
        context = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in recent)

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{message}"},
    ]
    raw = await tiny_model_complete(messages, max_tokens=128, temperature=0.05)
    raw = strip_stream_special_tokens(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"mode": "normal", "type": "general"}  # safe default
```

#### 2e. Add context compressor — new function

```python
async def compress_context(history: list) -> str:
    """Use tiny model to compress long conversation history into a brief summary."""
    if len(history) <= 6:
        # Short enough — just format directly
        return "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])

    full_context = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history)
    messages = [
        {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
        {"role": "user", "content": full_context},
    ]
    summary = await tiny_model_complete(messages, max_tokens=512, temperature=0.1)
    return strip_stream_special_tokens(summary)
```

#### 2f. Modify `orchestrated_chat()` — use new model tiers

Parallel agents phase becomes:
```python
output_a, output_b = await asyncio.gather(
    vllm_chat_complete(agent_a_messages, max_tokens=max_tokens),
    agent_b_complete(agent_b_messages, max_tokens=max_tokens),
)
```

Critique phase uses tiny model instead of 14B:
```python
critique_raw = await tiny_model_complete(critique_messages, max_tokens=256, temperature=0.05)
```

Context building uses compressor:
```python
context = await compress_context(history)
```

#### 2g. Modify `chat_websocket` — auto-routing

```python
mode = data.get('mode', 'auto')

if mode == 'auto':
    route = await route_query(message, history)
    mode = route.get("mode", "normal")
    logger.info("Auto-routed to %s (type: %s)", mode, route.get("type", "unknown"))
```

The frontend toggle gets a third state: "Auto" (default), "Deep" (force), "Normal" (force).

### 3. `prompts.py` — Add all new prompts

```python
ROUTER_SYSTEM_PROMPT = """Classify this query. Output ONLY valid JSON, no markdown.
{"mode": "normal"|"deep", "type": "code"|"explain"|"debug"|"review"|"general"}

Route to "deep" when the query:
- Requires generating AND validating code
- Asks for comparison of multiple approaches
- Involves multi-step reasoning or debugging
- Is complex enough to benefit from parallel analysis

Route to "normal" when:
- Simple factual question or short code snippet
- Follow-up that builds on prior context
- Clarification or refinement of previous answer"""

COMPRESS_SYSTEM_PROMPT = """Summarize this conversation into a brief context block (3-5 sentences max).
Preserve: key decisions, code snippets referenced, the current topic, any constraints mentioned.
Drop: greetings, acknowledgments, repetition, resolved tangents.
Output the summary only — no preamble."""

CRITIQUE_SYSTEM_PROMPT = """You are a quality reviewer. Given the user's original question and a draft response, evaluate whether the draft is correct, complete, and well-structured.

Output ONLY valid JSON — no markdown fences, no explanation. Format:
{"pass": true/false, "issues": "brief description of problems, or empty string if pass is true"}

Evaluation criteria:
- Correctness: Are code examples syntactically valid and logically correct? Are factual claims accurate?
- Completeness: Does the response fully address the user's question?
- Coherence: Is the response well-organized and free of contradictions or redundancy?
- Conciseness: Is it appropriately concise without missing key information?

Be strict but fair. Only fail the draft if there are genuine problems worth fixing — not stylistic nitpicks."""

REFINE_SYSTEM_PROMPT = """You are improving a draft response based on specific feedback.

Rules:
- Fix ONLY the issues identified — do not rewrite parts that are already good.
- Preserve the structure, tone, and any code that is correct.
- If the issue is about missing information, add it in the right place.
- If the issue is about incorrect code, fix the code and verify the fix.
- Do not mention that you are refining or that there were issues."""
```

### 4. `app.py` — Update imports

```python
from prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    COMPRESS_SYSTEM_PROMPT,
)
```

### 5. `static/app.js` — Three-state mode toggle

Update the toggle to cycle through: Auto → Deep → Normal → Auto.
The frontend already handles `{"type": "status"}` messages, so all new status updates display automatically.

### 6. Critique/refine loop (unchanged from earlier plan section)

After review/merge, the critique uses `tiny_model_complete()` instead of `vllm_chat_complete()`. Refine still uses 14B. Max 2 cycles. See section 2f above.

---

## Latency Budget

Rough estimates with the tiny model layer:

| Phase | Model | Current | New (best case) | New (worst, 2 refines) |
|-------|-------|---------|------------------|------------------------|
| Route | 0.6B | — | ~0.3s | ~0.3s |
| Compress | 0.6B | — | ~0.4s (if needed) | ~0.4s |
| Decompose | 14B | ~3s | ~3s | ~3s |
| Parallel agents | 14B + 7B/API | ~8s | ~8s (wall clock) | ~8s |
| Review | 14B | ~6s | ~6s | ~6s |
| Critique | 0.6B | — | ~0.3s | ~0.3s × 3 = ~0.9s |
| Refine | 14B | — | 0s (passed) | ~6s × 2 = ~12s |
| Stream final | — | included | ~1s | ~1s |
| **Total** | | **~17s** | **~19s** | **~32s** |

Key improvement: critique at 0.6B takes ~0.3s vs ~2s on 14B. This makes the quality gate nearly free in the pass case, and saves ~5s across the worst case compared to using 14B for critique.

The router adds ~0.3s to every request but can **save ~17s** on simple queries by skipping deep mode entirely.

---

## Configuration Summary

| Env Var | Default | Description |
|---------|---------|-------------|
| `TINY_VLLM_HOST` | `""` (disabled) | URL for tiny model vLLM instance |
| `TINY_MODEL_NAME` | `""` | e.g., `"Qwen/Qwen3-0.6B-AWQ"` |
| `AGENT_B_PROVIDER` | `""` (use primary) | `"local"`, `"openai"`, `"google"`, or empty |
| `AGENT_B_MODEL_NAME` | `""` | e.g., `"Qwen/Qwen3-7B-AWQ"`, `"gpt-4o-mini"` |
| `AGENT_B_API_KEY` | `""` | API key (only for openai/google providers) |

All new features degrade gracefully:
- No tiny model configured → route/critique/compress use the 14B (slower but works)
- No Agent B configured → both agents use 14B (current behavior)
- No external API key → Agent B uses local model or falls back to 14B

---

## Implementation Order

Suggested phased rollout to validate each piece independently:

1. **Phase 1: Critique loop only** — Add critique/refine to existing pipeline. No new models. Validates the self-correction mechanism.
2. **Phase 2: Tiny model layer** — Add vllm-tiny container with 0.6B. Move critique + router to it. Validates VRAM fit and latency gains.
3. **Phase 3: Dual-model agents** — Add Agent B (local 7B or external API). Validates diversity benefit.
4. **Phase 4: Chat tool protocol for workspace/sandbox** — Let the model call the persistent workspace and sandbox during a chat turn instead of only answering with plain text.

---

## Next Phase: Chat Tool Protocol

### Why This Is the Next High-Value Step

The workspace/sandbox foundation is now in place: conversations have persistent workspaces, the server can expose workspace files, and code can already be executed server-side. But the chat model still behaves like a pure text generator. It cannot autonomously inspect files, write changes, or run commands as part of solving the user's request.

That means we have the infrastructure, but not the control loop. The next phase should add a **small, explicit tool protocol** so the model can use those capabilities during chat.

### Goal

Extend the chat loop from:

`user message -> model text -> stream to UI`

to:

`user message -> model decides -> optional tool call -> server executes -> tool result fed back -> repeat -> final answer`

Success means the model can do all three during one turn:

1. Read from the conversation workspace
2. Write/update files in that workspace
3. Run sandboxed commands, then use the results in its final answer

### Minimal Protocol

Because the current vLLM chat path does not give us a native function-calling loop, use a small text protocol that the backend can parse reliably.

Assistant outputs must be exactly one of:

1. A normal user-facing answer
2. A single tool call block in this format:

```text
<tool_call>
{"id":"call_1","name":"workspace.read_file","arguments":{"path":"src/app.py"}}
</tool_call>
```

Tool results are injected back into the conversation by the server as:

```text
<tool_result>
{"id":"call_1","ok":true,"result":{...}}
</tool_result>
```

Keep v1 intentionally narrow:

- One tool call per assistant turn
- Max 6 tool turns per user message
- If parsing fails, treat the output as a final assistant answer
- Only stream the final user-facing answer; tool calls stay server-driven

This avoids needing true native function calling while still enabling a robust tool loop.

### Initial Tool Set

Keep the first tool surface small and directly mapped to the APIs/helpers we already have or can add safely:

| Tool | Purpose | Notes |
|------|---------|-------|
| `workspace.list_files` | List a directory in the current conversation workspace | Read-only, safe default for exploration |
| `workspace.read_file` | Read a text file from the workspace | Enforce size cap like current file APIs |
| `workspace.write_file` | Create/overwrite a file in the workspace | Needed so the model can actually produce artifacts |
| `workspace.run_command` | Run a sandboxed command in the workspace and capture stdout/stderr/exit code | This is the bridge from "can write" to "can verify" |

Optional but useful right after v1:

- `workspace.mkdir`
- `workspace.delete_file`
- `workspace.patch_file`

Do not start with arbitrary broad tools. A tiny, boring tool surface is easier to secure and debug.

### Backend Changes (`app.py`)

Add a server-side tool loop that wraps the existing chat completion path:

1. Build the model prompt with tool instructions and available tool schemas
2. Call the model non-streaming
3. Detect whether the response is a normal answer or a `<tool_call>` block
4. Validate tool name and arguments
5. Execute the tool against the current conversation workspace/sandbox
6. Append the tool result to the message list
7. Repeat until the assistant returns a plain final answer or the loop limit is hit
8. Stream only that final answer to the UI

Concrete additions:

- Add `run_tool_loop(...)` or similar helper near `orchestrated_chat()`
- Add `parse_tool_call(raw: str) -> dict | None`
- Add `execute_tool_call(conversation_id: str, call: dict) -> dict`
- Reuse workspace path helpers (`get_workspace_path`, `resolve_workspace_relative_path`) instead of making HTTP calls back into the same app
- Add a safe command runner helper for `workspace.run_command`

Important constraint for v1:

- Keep normal chat behavior unchanged when the model does not emit a tool call

### Command Execution Shape

The current app already has sandboxed Python execution, but the model needs a more general command primitive for coding workflows. Add a bounded command runner specifically for the workspace:

```json
{"name":"workspace.run_command","arguments":{"command":["python3","main.py"],"cwd":"."}}
```

Result shape:

```json
{
  "ok": true,
  "result": {
    "stdout": "...",
    "stderr": "...",
    "returncode": 0,
    "cwd": "."
  }
}
```

Guardrails:

- `cwd` must resolve inside the conversation workspace
- Command must be passed as an argv array, not a shell string
- Timeout stays short in v1 (for example 5-10 seconds)
- Return truncated stdout/stderr with explicit truncation markers

### Prompt Changes (`prompts.py`)

Add a dedicated tool-use system prompt, separate from the normal answer prompt:

```python
TOOL_USE_SYSTEM_PROMPT = """You may use tools to inspect the workspace, write files, and run commands.

When you need a tool, output ONLY:
<tool_call>
{"id":"call_n","name":"tool.name","arguments":{...}}
</tool_call>

Rules:
- Use at most one tool call at a time.
- Prefer reading before writing.
- After receiving a tool result, either make the next tool call or provide the final answer.
- When you are done, return a normal user-facing answer with no tool tags.
- Never invent tool results.
"""
```

The model should only see the tools we actually support. Keep schemas concrete and examples short.

### UI / WebSocket Changes (`static/app.js`)

The UI does not need a full agent console yet, but it should expose tool activity so the chat does not feel frozen during tool turns.

Add lightweight websocket events such as:

- `tool_start` — tool name + short summary
- `tool_result` — success/failure + concise preview
- `tool_error` — validation or execution failure

This can render similarly to current status updates. We do not need full stdout streaming in phase 1; a compact activity log is enough.

### Suggested Rollout Inside This Phase

1. **Phase 4a: Read-only protocol** — Implement the tool loop with `workspace.list_files` and `workspace.read_file` only.
2. **Phase 4b: Write support** — Add `workspace.write_file` and let the model create/update files.
3. **Phase 4c: Run support** — Add `workspace.run_command`, short timeouts, and stdout/stderr capture.
4. **Phase 4d: UX polish** — Show tool activity in the chat UI and include concise server logs.

This sequencing lets us prove the loop before we let the model mutate files or execute commands.

---

## Testing

1. **Router accuracy** — Batch of 20 queries (mix of simple/complex). Verify auto-routing matches what a human would choose >80% of the time.
2. **Critique calibration** — Simple correct answer should pass. Intentionally flawed answer (wrong code, missing info) should fail with useful issues.
3. **Critique speed** — 0.6B critique should complete in <0.5s. Benchmark against 14B doing the same task.
4. **Context compression** — Long conversation (10+ messages). Verify compressed context preserves key information and fits in agent prompt budget.
5. **Dual-model diversity** — Same query, compare Agent A (14B) vs Agent B (7B) outputs. Should show different approaches/emphasis, not just shortened versions.
6. **Full pipeline** — End-to-end deep mode with all three tiers. Measure total latency. Compare answer quality vs current single-model deep mode.
7. **Graceful degradation** — Disable tiny model, disable Agent B, disable both. Verify each fallback path works.
8. **VRAM pressure** — Monitor GPU memory with all models loaded. Verify no OOM under concurrent requests.
9. **Tool call parsing** — Valid `<tool_call>` blocks execute; malformed blocks fall back safely to plain assistant text.
10. **Read-only loop** — Ask the model to inspect workspace files and summarize what it found. Verify it can chain multiple reads before answering.
11. **Write loop** — Ask the model to create or edit a file in the conversation workspace. Verify path safety and file contents.
12. **Run loop** — Ask the model to run a simple command in the workspace and explain the output. Verify timeout, truncation, and exit-code handling.
13. **Loop bounds** — Force repeated tool usage and confirm the server stops after the max turn count with a graceful failure message.
14. **Security checks** — Attempt `../` path escapes, absolute paths, and disallowed commands. Verify every case is rejected cleanly.
