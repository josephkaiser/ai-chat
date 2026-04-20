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
- enough disk for Docker images, the Hugging Face cache, and app data
- `bash` and `curl` for the `./chat` helper script

Workspace roots are path-backed catalog entries. Managed workspace directories are typically bind-mounted under `./workspaces`, while legacy hosted runs may still exist under `./runs` after migration.

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

- `src/python/harness.py` — main backend runtime, routes, tool loop, workspace/file-session behavior
- `src/python/ai_chat/prompts.py` — default system prompt and execution prompts
- `src/python/ai_chat/themes.py` — UI palettes
- `src/python/ai_chat/thinking_stream.py` — thinking tag pairs, which must match `src/web/app.js`
- `MAX_COMPLETION_TOKENS` — env-backed completion cap
- `VLLM_HOST`, `DB_PATH`, `HF_CACHE_PATH` — core runtime paths/settings
- `COMMAND_TIMEOUT_SECONDS` — default timeout for ordinary workspace commands
- `CURATED_SOURCE_FAILURE_THRESHOLD` and `CURATED_SOURCE_DISABLE_MINUTES` — web-search source fan-out behavior
- `STRICT_WORKSPACE_COMMAND_PATHS` — reject workspace command arguments that point outside the current workspace root

Install/setup note:

- Python capability setup commands such as `python -m venv` and `pip install` are exempt from `COMMAND_TIMEOUT_SECONDS` so package installs can finish naturally.
- Those long-running commands are still stoppable from the UI via Stop / Interrupt.

The Docker image launches through `app.py`, which immediately hands off to `src/python/harness.py`. The frontend bundle is also checked at startup so stale `src/web/app.js` output is rebuilt when needed.

## Launcher notes

- `./chat install` is non-interactive and prepares the current runtime profile or `DEFAULT_MODEL_PROFILE` (falling back to `14b`).
- `./chat start` is non-interactive and reuses the installed/default downloaded profile.
- The launcher accepts chained commands in one invocation, so `./chat install start` and `./chat stop install start` run sequentially.

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

The compose file also mounts `./workspaces:/app/workspaces`, `./runs:/app/runs`, and `./python-envs:/app/python-envs` so path-backed workspaces, legacy hosted runs, and managed Python environments all persist across restarts.

## Workspace migration notes

- Existing conversations are backfilled into a new `workspaces` table on startup.
- Each migrated conversation keeps its prior hosted workspace path as the workspace root.
- New workspaces use direct root paths from the catalog instead of creating a fresh workspace per conversation.
- Legacy `runs/` trees are not deleted automatically during migration.

## Restarting individual services

```bash
docker compose restart chat-app
docker compose restart vllm
docker compose logs -f
```
