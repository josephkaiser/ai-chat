# Deep Mode v2: Critique Loop + Dual-Model Diversity

## Goal

Upgrade the existing deep mode pipeline from a single-pass `decompose → agents → merge` to a multi-cycle pipeline with a critique/refinement loop, and use **two different models** for the parallel agent phase to get genuinely diverse reasoning.

---

## Current State

- Single model: Qwen3-14B-AWQ via vLLM on one GPU (0.95 memory utilization)
- Pipeline: decompose → 2 parallel agents (same model) → review/merge → done
- All calls go through `vllm_chat_complete()` / `vllm_chat_stream()` hitting one vLLM instance

---

## New Pipeline

```
decompose
    ↓
agent_a (Model A) ──┐
                     ├── parallel
agent_b (Model B) ──┘
    ↓
review/merge
    ↓
critique  ←─────────┐
    ↓                │
  pass? ── no ──→ refine
    │
   yes
    ↓
stream final answer
```

### Phases

1. **Decompose** — Same as today. Qwen splits query into two sub-tasks. No change.
2. **Parallel agents** — Agent A uses Model A, Agent B uses Model B. Each gets its sub-task.
3. **Review/merge** — Same as today. Combines both outputs into a draft answer.
4. **Critique** (NEW) — A critic evaluates the draft for correctness, completeness, and coherence. Outputs a JSON verdict: `{"pass": true/false, "issues": "..."}`.
5. **Refine** (NEW, conditional) — If critique fails, send the draft + critique back to the model for a targeted fix. Then re-critique. Max 2 refinement cycles to bound latency.
6. **Stream** — Final approved answer streams to the user (same as today's review stream, but may be the refined version).

---

## Dual-Model Strategy

### The Problem

Two instances of the same model think similarly — same training biases, same blind spots. Parallel calls give speed but not true diversity of thought.

### Options for a Second Model

The GPU is fully allocated to Qwen3-14B-AWQ. Options for Model B:

| Option | Pros | Cons |
|--------|------|------|
| **A. External API (OpenAI/Anthropic/Google)** | True diversity, high quality, zero GPU cost | Adds latency (network), costs money, requires API key, external dependency |
| **B. Second vLLM instance with smaller model** | Local, no API cost | Needs GPU headroom — would require shrinking Qwen's allocation or a second GPU |
| **C. llama.cpp / Ollama for Model B** | Can run a small model on CPU or with minimal GPU | CPU inference is slow for 7B+; only viable for small models (1-3B) |
| **D. Mixed: vLLM multi-model** | vLLM supports serving multiple models via `--served-model-name` | Still GPU-bound; two 14B models won't fit, but a 14B + 3B might |

### Recommended: Option A (External API) as primary, Option D as fallback

**Primary approach:** Use an external API (e.g., OpenAI `gpt-4o-mini` or Google `gemini-2.0-flash`) for Agent B. This gives:
- Genuinely different reasoning patterns and training data
- No GPU pressure — the second model runs remotely
- `gpt-4o-mini` / `gemini-2.0-flash` are fast and cheap (~$0.15-0.30/1M input tokens)
- Both agents run in parallel, so the wall-clock time is `max(agent_a, agent_b)` — external API latency is often comparable to local 14B inference

**Fallback:** If no API key is configured, fall back to using the local Qwen model for both agents (current behavior). This keeps the app functional without external dependencies.

---

## File Changes

### 1. `docker-compose.yml` — Add env vars for second model

```yaml
environment:
  - VLLM_HOST=http://vllm:8000/v1
  - MODEL_NAME=Qwen/Qwen3-14B-AWQ
  - HF_CACHE_PATH=/cache/huggingface
  # Second model for dual-model deep mode (optional)
  - SECONDARY_MODEL_PROVIDER=        # "openai", "google", or empty to disable
  - SECONDARY_MODEL_NAME=            # e.g. "gpt-4o-mini", "gemini-2.0-flash"
  - SECONDARY_MODEL_API_KEY=         # API key for the secondary provider
```

### 2. `app.py` — New configuration + secondary model client

#### 2a. Config section — add after existing config vars

```python
# --- Secondary model for dual-model deep mode (optional) ---
SECONDARY_MODEL_PROVIDER = os.getenv("SECONDARY_MODEL_PROVIDER", "").strip().lower()
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "")
SECONDARY_MODEL_API_KEY = os.getenv("SECONDARY_MODEL_API_KEY", "")
DUAL_MODEL_ENABLED = bool(SECONDARY_MODEL_PROVIDER and SECONDARY_MODEL_NAME and SECONDARY_MODEL_API_KEY)

if DUAL_MODEL_ENABLED:
    logger.info(f"Dual-model deep mode enabled: secondary={SECONDARY_MODEL_PROVIDER}/{SECONDARY_MODEL_NAME}")
else:
    logger.info("Dual-model deep mode disabled — secondary model not configured")
```

#### 2b. Add `secondary_model_complete()` — after `vllm_chat_complete()`

```python
async def secondary_model_complete(messages: list, max_tokens: int = 4096, temperature: float = 0.25) -> str:
    """Chat completion from the secondary model provider (OpenAI-compatible or Google)."""
    if SECONDARY_MODEL_PROVIDER == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {SECONDARY_MODEL_API_KEY}"}
        body = {
            "model": SECONDARY_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    elif SECONDARY_MODEL_PROVIDER == "google":
        url = f"https://generativelanguage.googleapis.com/v1beta/chat/completions"
        headers = {"Authorization": f"Bearer {SECONDARY_MODEL_API_KEY}"}
        body = {
            "model": SECONDARY_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        raise ValueError(f"Unknown secondary provider: {SECONDARY_MODEL_PROVIDER}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

Note: Google's Gemini API now supports the OpenAI-compatible chat completions format, so both providers use the same response parsing.

#### 2c. Modify `orchestrated_chat()` — parallel agents phase

Replace the current parallel agent call:
```python
output_a, output_b = await asyncio.gather(
    vllm_chat_complete(agent_a_messages, max_tokens=max_tokens),
    vllm_chat_complete(agent_b_messages, max_tokens=max_tokens),
)
```

With:
```python
agent_b_fn = (
    secondary_model_complete if DUAL_MODEL_ENABLED else vllm_chat_complete
)
output_a, output_b = await asyncio.gather(
    vllm_chat_complete(agent_a_messages, max_tokens=max_tokens),
    agent_b_fn(agent_b_messages, max_tokens=max_tokens),
)
```

#### 2d. Add critique/refine loop — after the review phase, before streaming

Replace the current review-then-stream block with:

```python
# --- Phase 3: Review (non-streaming now, since we need to critique it) ---
await websocket.send_json({"type": "status", "content": "Reviewing and combining..."})

review_user_content = (
    f"User's question:\n{message}\n\n"
    f"Response A ({agent_a_role}):\n{output_a}\n\n"
    f"Response B ({agent_b_role}):\n{output_b}"
)
review_messages = [
    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
    {"role": "user", "content": review_user_content},
]
draft = await vllm_chat_complete(review_messages, max_tokens=max_tokens)
draft = strip_stream_special_tokens(draft)

# --- Phase 4: Critique/refine loop (max 2 cycles) ---
MAX_REFINE_CYCLES = 2
for cycle in range(MAX_REFINE_CYCLES):
    await websocket.send_json({
        "type": "status",
        "content": f"Quality check {cycle + 1}/{MAX_REFINE_CYCLES}..."
    })

    critique_messages = [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"User's original question:\n{message}\n\n"
            f"Draft response:\n{draft}"
        )},
    ]
    critique_raw = await vllm_chat_complete(critique_messages, max_tokens=1024, temperature=0.1)
    critique_raw = strip_stream_special_tokens(critique_raw)

    try:
        critique = json.loads(critique_raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", critique_raw, re.DOTALL)
        if match:
            critique = json.loads(match.group())
        else:
            break  # Can't parse critique — accept the draft

    if critique.get("pass", True):
        logger.info("Critique passed on cycle %d", cycle + 1)
        break

    # Refine
    issues = critique.get("issues", "unspecified issues")
    logger.info("Critique cycle %d issues: %s", cycle + 1, issues)
    await websocket.send_json({
        "type": "status",
        "content": f"Refining: {issues[:80]}..."
    })

    refine_messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"User's question:\n{message}\n\n"
            f"Current draft:\n{draft}\n\n"
            f"Issues found:\n{issues}\n\n"
            "Produce an improved version that fixes these issues."
        )},
    ]
    draft = await vllm_chat_complete(refine_messages, max_tokens=max_tokens)
    draft = strip_stream_special_tokens(draft)

# --- Phase 5: Stream the final answer ---
await websocket.send_json({"type": "start"})
# Stream the pre-computed draft to the user token-by-token for consistent UX
for i in range(0, len(draft), 4):
    chunk = draft[i:i+4]
    await websocket.send_json({"type": "token", "content": chunk})
return draft
```

### 3. `prompts.py` — Add critique and refine prompts

```python
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

### 4. `prompts.py` — Update imports in `app.py`

```python
from prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
)
```

### 5. `static/app.js` — Update status message handling (no change needed)

The frontend already handles `{"type": "status"}` messages, so the new critique/refine status updates will display automatically.

---

## Latency Budget

Rough estimates for the extended pipeline:

| Phase | Current | New (best case) | New (worst case, 2 refine cycles) |
|-------|---------|------------------|-------------------------------------|
| Decompose | ~3s | ~3s | ~3s |
| Parallel agents | ~8s | ~8s (wall clock) | ~8s |
| Review | ~6s (streaming) | ~6s (non-streaming) | ~6s |
| Critique | — | ~2s | ~2s × 3 = ~6s |
| Refine | — | 0s (passed) | ~6s × 2 = ~12s |
| Stream final | included above | ~1s | ~1s |
| **Total** | **~17s** | **~20s** | **~36s** |

The worst case (2 full refine cycles) adds ~19s. In practice, most responses should pass on the first critique, keeping the overhead to ~2-3s.

---

## Configuration Summary

| Env Var | Default | Description |
|---------|---------|-------------|
| `SECONDARY_MODEL_PROVIDER` | `""` (disabled) | `"openai"` or `"google"` |
| `SECONDARY_MODEL_NAME` | `""` | e.g., `"gpt-4o-mini"`, `"gemini-2.0-flash"` |
| `SECONDARY_MODEL_API_KEY` | `""` | API key for the provider |

When no secondary model is configured, deep mode works exactly as it does today (both agents use local Qwen) plus the new critique loop.

---

## Testing

1. **Critique pass** — Simple question, draft should pass on first critique. Verify no unnecessary refine cycles.
2. **Critique fail → refine** — Ask something with nuance (e.g., "explain the difference between concurrency and parallelism with Go examples"). If the first draft misses something, the critique should catch it and the refine should fix it.
3. **Dual-model diversity** — With secondary model configured, verify Agent A and Agent B outputs have meaningfully different perspectives. Compare quality vs single-model mode.
4. **Fallback** — With no secondary model configured, verify deep mode still works (both agents use Qwen).
5. **Max cycles** — Force a bad draft (e.g., very low temperature) and verify the loop stops after 2 refine cycles, not infinitely.
6. **Latency** — Measure end-to-end time for critique-pass vs critique-fail-once vs max-cycles.
