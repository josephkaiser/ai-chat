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

## Switchable model profiles

The app can keep one primary `14B` profile and expose a dashboard switch that hard-restarts `vLLM` into an `8B` profile. The active profile is persisted in `/app/data/model_state.json`.

Configure these env vars on `chat-app`:

```yaml
environment:
  - VLLM_HOST=http://vllm:8000/v1
  - MODEL_NAME=Qwen/Qwen3-14B-AWQ
  - MODEL_14B_NAME=Qwen/Qwen3-14B-AWQ
  - MODEL_14B_ARGS=--gpu-memory-utilization 0.95 --max-model-len 8192 --enable-prefix-caching --max-num-seqs 16 --enable-chunked-prefill --quantization awq_marlin --trust-remote-code --enforce-eager
  - MODEL_8B_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
  - MODEL_8B_ARGS=--gpu-memory-utilization 0.90 --max-model-len 8192 --enable-prefix-caching --max-num-seqs 16 --enable-chunked-prefill --enforce-eager
  - DEFAULT_MODEL_PROFILE=14b
```

Use the dashboard action to switch profiles. The app will:

- Confirm the switch in the UI
- Stop and remove the current `vllm` container
- Recreate `vllm` with the selected model command
- Start polling health until the new model is ready

## Chat application behavior

These are **code defaults** (no extra env vars required):

- **Default system prompt** — Edit `prompts.py` (`DEFAULT_SYSTEM_PROMPT`). Keep it compact if you want strong 8B performance.
- **Light/dark UI colors** — Edit `themes.py` (named palettes at the top of `_light_tokens()` / `_dark_tokens()`). Variable names must stay aligned with `static/style.css`.
- **Thinking tags in the stream** — `thinking_stream.py` (`THINK_TAG_PAIRS`) must match `THINK_TAG_PAIRS` in `static/app.js`.
- **Deep-mode fast path for 8B** — The lighter profile uses a deterministic confirmation note, a heuristic execution plan fallback, a tighter tool-step cap, and skips the final critique/refine loop for lower latency.

Other tunables in **`app.py`** (module-level constants / env):

- `MAX_COMPLETION_TOKENS` — From env `MAX_COMPLETION_TOKENS` (default `4096`).
- `VLLM_HOST`, `DB_PATH`, `HF_CACHE_PATH` — See comments near the top of `app.py`.

The Docker image copies **`app.py`**, **`themes.py`**, **`prompts.py`**, and **`thinking_stream.py`** together (see `Dockerfile`).

## Server Voice Pipeline

The browser can now record audio and hand it to the server for transcription, and assistant replies can be synthesized on the server and returned as audio files.

Relevant env vars on `chat-app`:

```yaml
environment:
  - VOICE_ROOT=/app/data/voice
  - VOICE_INPUT_SIZE_LIMIT=15728640
  - VOICE_COMMAND_TIMEOUT_SECONDS=180
  - VOICE_TTS_VOICE=Samantha
  - VOICE_STT_LANGUAGE=en
  - PIPER_MODEL=/app/data/voice/models/en_US-lessac-high.onnx
  - VOICE_TTS_COMMAND=
  - VOICE_STT_COMMAND=
```

Behavior:

- If `VOICE_TTS_COMMAND` is empty and the server can run `say`, the app uses native macOS `say` for TTS.
- If `VOICE_STT_COMMAND` is empty and the server can run `whisper`, the app uses native `whisper` CLI for STT with `VOICE_STT_LANGUAGE` as the language hint.
- If either tool is unavailable, the corresponding voice feature is disabled in the UI with a clear status note.

Command templates support these placeholders:

- `{input}` — Uploaded audio file path for STT
- `{output}` — Generated audio output path for TTS
- `{output_dir}` — Output directory for tool artifacts
- `{transcript}` — Preferred transcript path
- `{textfile}` — UTF-8 text file containing the assistant reply for TTS

Examples:

```yaml
- VOICE_TTS_COMMAND=say -v Samantha -o {output} -f {textfile}
- VOICE_STT_COMMAND=whisper {input} --model turbo --language en --output_format txt --output_dir {output_dir}
```

Notes:

- The UI speech-speed control now supports up to `2.0x` playback for generated replies.
- For a more natural English voice on macOS, start with `VOICE_TTS_VOICE=Samantha` and try other English `say` voices such as `Karen`, `Moira`, or `Daniel` if you prefer a different accent or cadence.
- For Piper fallback, the project now defaults to `en_US-lessac-high`, which is a better English-quality choice than the previous `en_US-amy-medium`, at the cost of a larger/slower model.
- In Docker, native host tools like macOS `say` are usually not available inside the container unless you deliberately provide them.
- For a host-native deployment on macOS, leaving `VOICE_TTS_COMMAND` empty is enough to pick up `say`.
- The current built-in fallback writes AIFF output when using `say`, which the browser then plays back from the server.

## Hardening flags

The app now ships with the riskiest host-control features disabled by default. Configure these env vars on `chat-app` only if you explicitly want them:

```yaml
environment:
  - DOCKER_CONTROL_ENABLED=false
  - INTERACTIVE_TERMINAL_ENABLED=false
  - EXECUTE_CODE_ENABLED=false
  - STRICT_WORKSPACE_COMMAND_PATHS=true
```

- `DOCKER_CONTROL_ENABLED` gates Docker-socket-backed dashboard actions such as restart, model switching, and cache redownload.
- `INTERACTIVE_TERMINAL_ENABLED` gates the PTY-backed interactive shell in the workspace panel.
- `EXECUTE_CODE_ENABLED` gates the legacy `/api/execute-code` endpoint.
- `STRICT_WORKSPACE_COMMAND_PATHS` rejects workspace command arguments that reference paths outside the current conversation workspace.
- Command execution is additionally allowlisted per chat in the UI: when the assistant first tries an executable such as `pytest` or `npm`, the user is asked to allow that executable for the current chat only.

If you enable Docker control, you must also mount `/var/run/docker.sock` into the `chat-app` container deliberately.

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

By default the sample compose file does not mount the Docker socket into `chat-app`. That keeps dashboard runtime controls read-only unless you opt in.

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
