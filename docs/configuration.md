# Configuration

Edit `docker-compose.yml` to change the model or vLLM settings:

```yaml
vllm:
  command: >
    --model Qwen/Qwen3-8B
    --gpu-memory-utilization 0.90
    --max-model-len 32768
    --swap-space 8
    --enable-prefix-caching
    --max-num-seqs 256
    --enable-chunked-prefill
```

Set `MODEL_NAME` in the `chat-app` environment to match.

## Chat application behavior

These are **code defaults** (no extra env vars required):

- **Default system prompt** — Edit `prompts.py` (`DEFAULT_SYSTEM_PROMPT`).
- **Light/dark UI colors** — Edit `themes.py` (named palettes at the top of `_light_tokens()` / `_dark_tokens()`). Variable names must stay aligned with `static/style.css`.
- **Thinking tags in the stream** — `thinking_stream.py` (`THINK_TAG_PAIRS`) must match `THINK_TAG_PAIRS` in `static/app.js`.

Other tunables in **`app.py`** (module-level constants / env):

- `MAX_COMPLETION_TOKENS` — From env `MAX_COMPLETION_TOKENS` (default `4096`).
- `VLLM_HOST`, `DB_PATH`, `HF_CACHE_PATH` — See comments near the top of `app.py`.

The Docker image copies **`app.py`**, **`themes.py`**, **`prompts.py`**, and **`thinking_stream.py`** together (see `Dockerfile`).

## GPU Configurations

**24GB (RTX 3090/4090):** Default config works well.

**12GB (RTX 3060):**
```yaml
--gpu-memory-utilization 0.90
--max-model-len 16384
--swap-space 4
```

**40GB+ (A100/H100):**
```yaml
--gpu-memory-utilization 0.95
--max-model-len 65536
--swap-space 16
```

## Docker Services

The application runs two containers via `docker-compose.yml`:

- **vllm** — vLLM OpenAI-compatible inference server (port 8001)
- **chat-app** — FastAPI web application (port 8000)

Both are set to `restart: unless-stopped`, so they will survive reboots.

### Restarting Individual Services

```bash
# Restart just the chat app (fast, keeps model loaded)
docker compose restart chat-app

# Restart vLLM (reloads model, takes 2-5 min)
docker compose restart vllm

# View logs
docker compose logs -f
```
