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

- Docker with the `docker compose` plugin
- NVIDIA GPU support exposed to Docker
- Enough disk for Docker images, the Hugging Face cache, app data, and optional voice models
- `bash` and `curl` for the `./chat` helper script

Conversation workspaces are bind-mounted to the host under `./runs` so generated files remain visible outside the container.

## Model profiles

The app keeps a primary model profile plus optional additional configured profiles. The selected profile is persisted in `/app/data/model_state.json`.

Example `chat-app` environment:

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

## Chat application behavior

Code defaults and useful tunables:

- `prompts.py` — default system prompt and execution prompts
- `themes.py` — light/dark UI palettes
- `thinking_stream.py` — thinking tag pairs, which must match `static/app.js`
- `MAX_COMPLETION_TOKENS` — env-backed completion cap
- `VLLM_HOST`, `DB_PATH`, `HF_CACHE_PATH` — core runtime paths/settings
- `CURATED_SOURCE_FAILURE_THRESHOLD` and `CURATED_SOURCE_DISABLE_MINUTES` — web-search source fan-out behavior
- `VOICE_STORAGE_LIMIT_BYTES` — cap retained server-owned voice artifacts
- `STRICT_WORKSPACE_COMMAND_PATHS` — reject workspace command arguments that point outside the current conversation workspace

The Docker image copies `app.py`, `themes.py`, `prompts.py`, `thinking_stream.py`, `turn_strategy.py`, and `deep_flow.py`.

## Server voice pipeline

The built-in voice stack has two paths:

- the web mic button records in the browser and uploads the clip into the workspace as an attachment
- assistant reply playback is generated on the server and returned as an audio file

Useful env vars:

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

## GPU configurations

24GB-class GPUs: the default config is the intended target.

12GB-class GPUs:

```yaml
--gpu-memory-utilization 0.90
--max-model-len 16384
--swap-space 4
```

40GB+ GPUs:

```yaml
--gpu-memory-utilization 0.95
--max-model-len 65536
--swap-space 16
```

## Docker services

The application runs two containers via `docker-compose.yml`:

- `vllm` — OpenAI-compatible inference server
- `chat-app` — FastAPI web application

The compose file also mounts `./runs:/app/runs`, which is where per-conversation workspaces live.

### Restarting individual services

```bash
docker compose restart chat-app
docker compose restart vllm
docker compose logs -f
```
