# Deep Mode: Multi-Agent Orchestration Pipeline

## Overview

Add a "Deep" mode toggle to the chat app. When active, the user's prompt goes through a 3-phase pipeline instead of a single model call:

1. **Decompose** — Model splits the user's query into 2 parallel sub-tasks (non-streaming, JSON output)
2. **Execute** — Both sub-tasks run in parallel against the same vLLM instance (non-streaming)
3. **Review** — Combined outputs sent to the model with fresh context for final synthesis (streaming to user)

vLLM handles concurrent requests via continuous batching — no second container needed.

Fallback: if decompose fails to produce valid JSON, fall back to normal single-call mode.

---

## File Changes

### 1. `prompts.py` — Add two new prompts

Append after `DEFAULT_SYSTEM_PROMPT`:

```python
DECOMPOSE_SYSTEM_PROMPT = """You are a task planner. Given a user's query and conversation context, split the work into exactly 2 parallel sub-tasks for two AI agents.

Output ONLY valid JSON — no markdown fences, no explanation, no preamble. Format:
{"strategy":"brief description","agent_a":{"role":"focus area","prompt":"self-contained prompt"},"agent_b":{"role":"focus area","prompt":"self-contained prompt"}}

Splitting guidelines:
- Each prompt must be self-contained — include any code or context the agent needs.
- Code generation: one writes the code, one writes tests or validates edge cases.
- Explanation: split by sub-topic, or one explains and one gives worked examples.
- Debugging: one diagnoses root cause, one proposes fixes.
- If the task cannot be meaningfully split, have one agent do the main work and the other review/critique it.
- Keep prompts focused. Do not repeat these instructions in the prompts."""

REVIEW_SYSTEM_PROMPT = """You are combining two AI responses into one excellent answer for the user.

Rules:
- Produce a single, coherent response — not a comparison of the two inputs.
- Keep the best code, explanations, and insights from both.
- Fix errors or contradictions; remove redundancy.
- Match the style expected by a coding companion: concise explanation then code.
- Do not mention that two agents were involved or that you are reviewing anything."""
```

---

### 2. `app.py` — Backend pipeline

#### 2a. Update import line

Change:
```python
from prompts import DEFAULT_SYSTEM_PROMPT
```
To:
```python
from prompts import DEFAULT_SYSTEM_PROMPT, DECOMPOSE_SYSTEM_PROMPT, REVIEW_SYSTEM_PROMPT
```

#### 2b. Add `vllm_chat_complete()` — place right after `vllm_chat_stream()`

```python
async def vllm_chat_complete(messages: list, max_tokens: int = 4096, temperature: float = 0.25) -> str:
    """Non-streaming chat completion from vLLM."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        resp = await client.post(
            f"{VLLM_HOST}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

#### 2c. Add `orchestrated_chat()` — place right before the `chat_websocket` handler

```python
async def orchestrated_chat(websocket: WebSocket, message: str, history: list,
                            system_prompt: str, max_tokens: int) -> str:
    """Decompose -> parallel agents -> review. Returns final response text."""

    # --- Phase 1: Decompose ---
    await websocket.send_json({"type": "status", "content": "Analyzing query..."})

    context = ""
    if history:
        recent = history[-6:]
        context = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)

    decompose_messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation context:\n{context}\n\nUser's new query:\n{message}"},
    ]

    raw = await vllm_chat_complete(decompose_messages, max_tokens=1024, temperature=0.1)
    cleaned = strip_stream_special_tokens(raw)

    # Parse JSON — try raw first, then extract from surrounding text
    import re as _re
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _re.search(r"\{.*\}", cleaned, _re.DOTALL)
        if match:
            plan = json.loads(match.group())
        else:
            raise ValueError("Decompose did not produce valid JSON")

    strategy = plan.get("strategy", "")
    agent_a_role = plan["agent_a"]["role"]
    agent_b_role = plan["agent_b"]["role"]
    agent_a_prompt = plan["agent_a"]["prompt"]
    agent_b_prompt = plan["agent_b"]["prompt"]

    logger.info(f"Deep mode strategy: {strategy}")

    # --- Phase 2: Parallel execution ---
    await websocket.send_json({"type": "status", "content": f"Working in parallel: {strategy}"})

    agent_a_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": agent_a_prompt},
    ]
    agent_b_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": agent_b_prompt},
    ]

    output_a, output_b = await asyncio.gather(
        vllm_chat_complete(agent_a_messages, max_tokens=max_tokens),
        vllm_chat_complete(agent_b_messages, max_tokens=max_tokens),
    )

    output_a = strip_stream_special_tokens(output_a)
    output_b = strip_stream_special_tokens(output_b)

    # --- Phase 3: Review (streamed to user) ---
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

    await websocket.send_json({"type": "start"})

    splitter = ThinkingStreamSplitter()
    async for token in vllm_chat_stream(review_messages, max_tokens=max_tokens):
        for payload in splitter.feed(token):
            await websocket.send_json(payload)
    for payload in splitter.finalize():
        await websocket.send_json(payload)

    return strip_stream_special_tokens(splitter.full_response)
```

#### 2d. Modify `chat_websocket` handler — replace the try block inside `while True`

Current logic (save message, build messages, stream, save response) needs to branch on mode. The new structure:

```python
try:
    save_message(conv_id, 'user', message)
    history = get_conversation_history(conv_id, current_query=None)
    system_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
    max_tokens = MAX_COMPLETION_TOKENS

    mode = data.get('mode', 'normal')
    deep_succeeded = False

    if mode == 'deep':
        try:
            full_response = await orchestrated_chat(
                websocket, message, history, system_prompt, max_tokens
            )
            assistant_message_id = save_message(conv_id, 'assistant', full_response)
            await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
            await websocket.send_json({'type': 'done'})
            deep_succeeded = True
        except Exception as e:
            logger.warning(f"Deep mode failed, falling back: {e}")
            await websocket.send_json({'type': 'status', 'content': 'Deep mode failed, using normal mode...'})

    if not deep_succeeded:
        # --- existing normal path (unchanged) ---
        messages = [{'role': 'system', 'content': system_prompt}]
        for msg in history:
            messages.append({'role': msg['role'], 'content': msg['content']})

        await websocket.send_json({'type': 'start'})

        logger.info(f"Processing message for conv {conv_id}, max_tokens: {max_tokens}")

        splitter = ThinkingStreamSplitter()
        async for token in vllm_chat_stream(messages, max_tokens=max_tokens):
            for payload in splitter.feed(token):
                await websocket.send_json(payload)
        for payload in splitter.finalize():
            await websocket.send_json(payload)

        full_response = strip_stream_special_tokens(splitter.full_response)
        assistant_message_id = save_message(conv_id, 'assistant', full_response)
        await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
        await websocket.send_json({'type': 'done'})

except httpx.HTTPStatusError as e:
    # ... (keep existing error handlers unchanged)
```

---

### 3. `static/app.js` — Frontend toggle + status messages

#### 3a. Add state variable — after `let pastedBlocks = [];` (line 11)

```javascript
let deepMode = localStorage.getItem('deepMode') === 'true';
```

#### 3b. Add toggle function — after `enterWelcomeMode()`

```javascript
function toggleDeepMode() {
    deepMode = !deepMode;
    localStorage.setItem('deepMode', deepMode);
    const btn = document.getElementById('deepToggle');
    if (btn) btn.classList.toggle('active', deepMode);
}
```

#### 3c. Handle `status` messages — in `ws.onmessage`, add this branch before the `done` check

```javascript
} else if (data.type === 'status') {
    const loadingText = document.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = data.content;
```

#### 3d. Send mode in WebSocket message — in `sendMessage()`, update the `ws.send` call

Change:
```javascript
ws.send(JSON.stringify({
    message: message,
    conversation_id: currentConvId,
    system_prompt: localStorage.getItem('customSystemPrompt') || null
}));
```
To:
```javascript
ws.send(JSON.stringify({
    message: message,
    conversation_id: currentConvId,
    system_prompt: localStorage.getItem('customSystemPrompt') || null,
    mode: deepMode ? 'deep' : 'normal'
}));
```

#### 3e. Initialize toggle state on load — at the bottom, after the `_input` block

```javascript
const _deepBtn = document.getElementById('deepToggle');
if (_deepBtn) _deepBtn.classList.toggle('active', deepMode);
```

---

### 4. `static/index.html` — Add toggle button

In the `.input-container` div, add the deep toggle button before the send button:

Change:
```html
<button id="send" onclick="sendMessage()">&#10148;</button>
```
To:
```html
<button id="deepToggle" class="deep-toggle" onclick="toggleDeepMode()" title="Deep mode: multi-agent pipeline">Deep</button>
<button id="send" onclick="sendMessage()">&#10148;</button>
```

---

### 5. `static/style.css` — Toggle styling

Add before the `/* === Loading Animation === */` section:

```css
/* === Deep Mode Toggle === */
.deep-toggle {
    background: transparent;
    border: 1px solid var(--bg_tertiary);
    border-radius: 14px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text_tertiary);
    cursor: pointer;
    transition: all 0.2s;
    margin: 7px 0 7px 8px;
    flex-shrink: 0;
    line-height: 1.4;
}
.deep-toggle:hover {
    color: var(--text_secondary);
    border-color: var(--text_tertiary);
}
.deep-toggle.active {
    background: var(--accent_primary);
    color: #fff;
    border-color: var(--accent_primary);
}
```

---

## Testing

1. **Normal mode** — should work exactly as before (no regressions)
2. **Deep mode, simple query** — e.g. "what does += do in python" — verify decompose produces valid JSON, review produces coherent output
3. **Deep mode, code generation** — e.g. "write a binary search in rust" — expect one agent writes code, other writes tests
4. **Deep mode fallback** — if decompose fails (bad JSON), should fall back to normal mode with a status message
5. **Toggle persistence** — deep mode state should survive page reload (localStorage)
6. **Status messages** — loading text should update through phases: "Analyzing..." → "Working in parallel: ..." → "Reviewing..." → streamed answer
