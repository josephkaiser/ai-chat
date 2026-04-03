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

## Host requirements

The checked-in Docker setup assumes a local host that can run GPU-backed vLLM:

- **Docker** — modern Docker plus the `docker compose` plugin.
- **GPU runtime** — NVIDIA GPU support exposed to Docker. The sample compose file reserves an NVIDIA device for the `vllm` service.
- **VRAM target** — the default `14B` AWQ profile is meant for roughly `24GB`-class GPUs. If your machine is closer to `12GB`, start with the lighter `8B` profile and the reduced settings in [GPU Configurations](#gpu-configurations).
- **Disk** — enough free space for Docker images, the Hugging Face cache mounted at `~/.cache/huggingface`, project data in `./data`, and optional voice models downloaded by `./chat install`.
- **Network** — required during install for container pulls, Piper voice-model download, and any later model downloads.

The helper script `./chat` also assumes `bash` and `curl` are available on the host.

## Switchable model profiles

The app can keep a primary `14B` profile and expose additional switchable profiles such as `8B (Meta)` and `Gemma 4 E4B (Google)`. The active profile is persisted in `/app/data/model_state.json`.

Configure these env vars on `chat-app`:

```yaml
environment:
  - VLLM_HOST=http://vllm:8000/v1
  - MODEL_NAME=Qwen/Qwen3-14B-AWQ
  - MODEL_14B_NAME=Qwen/Qwen3-14B-AWQ
  - MODEL_14B_ARGS=--gpu-memory-utilization 0.95 --max-model-len 8192 --enable-prefix-caching --max-num-seqs 16 --enable-chunked-prefill --quantization awq_marlin --trust-remote-code --enforce-eager
  - MODEL_8B_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
  - MODEL_8B_ARGS=--gpu-memory-utilization 0.90 --max-model-len 8192 --enable-prefix-caching --max-num-seqs 16 --enable-chunked-prefill --enforce-eager
  - MODEL_GEMMA_4_NAME=google/gemma-4-E4B-it
  - MODEL_GEMMA_4_ARGS=--gpu-memory-utilization 0.85 --max-model-len 8192 --enable-prefix-caching --max-num-seqs 12 --enable-chunked-prefill --enforce-eager
  - DEFAULT_MODEL_PROFILE=14b
```

Use the `Models` button in the composer or menu to open the dashboard and library view.

For live profile switching or cached-model activation, the server must have Docker control enabled. The sample local compose file now mounts `/var/run/docker.sock` and sets `DOCKER_CONTROL_ENABLED=true`, so switching works in the default trusted local setup. If you turn that off, the UI still exposes the Hugging Face library so you can download and delete caches, but it will not pretend that profile switching is available.

When Docker control is enabled, the app will:

- Confirm the switch in the UI
- Stop and remove the current `vllm` container
- Recreate `vllm` with the selected model command
- Start polling health until the new model is ready

## Chat application behavior

These are **code defaults** (no extra env vars required):

- **Default system prompt** — Edit `prompts.py` (`DEFAULT_SYSTEM_PROMPT`). Keep it compact if you want strong 8B performance.
- **Light/dark UI colors** — Edit `themes.py` (named palettes at the top of `_light_tokens()` / `_dark_tokens()`). Variable names must stay aligned with `static/style.css`.
- **Thinking tags in the stream** — `thinking_stream.py` (`THINK_TAG_PAIRS`) must match `THINK_TAG_PAIRS` in `static/app.js`.
- **Deep-mode fast path for smaller profiles** — The lighter profiles use a deterministic confirmation note, a heuristic execution plan fallback, a tighter tool-step cap, and skip the final critique/refine loop for lower latency.

Other tunables in **`app.py`** (module-level constants / env):

- `MAX_COMPLETION_TOKENS` — From env `MAX_COMPLETION_TOKENS` (default `4096`).
- `VLLM_HOST`, `DB_PATH`, `HF_CACHE_PATH` — See comments near the top of `app.py`.
- `CURATED_SOURCE_FAILURE_THRESHOLD` / `CURATED_SOURCE_DISABLE_MINUTES` — Control how quickly auto-added curated web domains are temporarily disabled after repeated failures.
- `VOICE_STORAGE_LIMIT_BYTES` — Caps retained server-owned voice artifacts before oldest-first pruning kicks in.

The Docker image copies **`app.py`**, **`themes.py`**, **`prompts.py`**, and **`thinking_stream.py`** together (see `Dockerfile`).

## Server Voice Pipeline

The current voice stack has two different paths:

- The built-in web mic button records in the browser with `MediaRecorder` and uploads the clip into the conversation workspace as a normal attachment.
- Assistant reply playback is generated on the server and returned as an audio file.
- The server-side transcription endpoint still exists for direct API use or custom clients, but the stock web UI does not need it just to record and attach audio.

Relevant env vars on `chat-app`:

```yaml
environment:
  - VOICE_ROOT=/app/data/voice
  - VOICE_INPUT_SIZE_LIMIT=15728640
  - VOICE_COMMAND_TIMEOUT_SECONDS=180
  - VOICE_STORAGE_LIMIT_BYTES=3221225472
  - VOICE_TTS_VOICE=Samantha
  - VOICE_STT_LANGUAGE=en
  - PIPER_MODEL=/app/data/voice/models/en_US-lessac-high.onnx
  - VOICE_TTS_COMMAND=
  - VOICE_STT_COMMAND=
```

Behavior:

- If `VOICE_TTS_COMMAND` is empty and the server can run `say`, the app uses native macOS `say` for TTS.
- If `VOICE_STT_COMMAND` is empty and the server can run `whisper`, the app uses native `whisper` CLI for STT with `VOICE_STT_LANGUAGE` as the language hint.
- If TTS is unavailable, reply playback is disabled in the UI with a clear status note.
- Browser-side audio recording does not depend on `VOICE_STT_COMMAND`; it only needs browser mic support because the recording is uploaded as an attachment.
- `POST /api/voice/transcribe` still depends on STT availability if you call it directly.

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
- Temporary STT inputs/transcripts are deleted after each transcription request.
- Generated TTS text/audio is stored under `VOICE_ROOT` and pruned oldest-first when it grows past `VOICE_STORAGE_LIMIT_BYTES`.
- Deleting a conversation or running the full app reset also removes any voice artifacts scoped to that conversation.
- For a more natural English voice on macOS, start with `VOICE_TTS_VOICE=Samantha` and try other English `say` voices such as `Karen`, `Moira`, or `Daniel` if you prefer a different accent or cadence.
- For Piper fallback, the project now defaults to `en_US-lessac-high`, which is a better English-quality choice than the previous `en_US-amy-medium`, at the cost of a larger/slower model.
- In Docker, native host tools like macOS `say` are usually not available inside the container unless you deliberately provide them.
- For a host-native deployment on macOS, leaving `VOICE_TTS_COMMAND` empty is enough to pick up `say`.
- The current built-in fallback writes AIFF output when using `say`, which the browser then plays back from the server.

## Web search fan-out

The search tool now does more than one plain web query when it helps answer quality:

- general web search runs alongside Wikipedia and Reddit result sets when no domain filter is provided
- curated authoritative domains can be auto-added for some topics; philosophy queries currently fan out to Stanford Encyclopedia of Philosophy and the Internet Encyclopedia of Philosophy
- curated sources are fail-open: if they start failing repeatedly, the app temporarily disables just those domains and continues returning the rest of the result sets

Relevant env vars on `chat-app`:

```yaml
environment:
  - CURATED_SOURCE_FAILURE_THRESHOLD=2
  - CURATED_SOURCE_DISABLE_MINUTES=240
```

## Hardening flags

If you want a stricter deployment than the default local setup, configure these env vars on `chat-app`:

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

When `DOCKER_CONTROL_ENABLED=true`, `chat-app` also needs `/var/run/docker.sock` mounted.

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

The sample compose file mounts the Docker socket into `chat-app` so dashboard runtime controls work out of the box for a trusted local machine. Remove that mount and set `DOCKER_CONTROL_ENABLED=false` if you want a read-only setup instead.

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
