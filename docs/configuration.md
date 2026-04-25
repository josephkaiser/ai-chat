# Configuration

Edit [config/model-defaults.env](/Users/joe/dev/ai-chat/config/model-defaults.env) to change the safe repo defaults in one place:

```dotenv
DEFAULT_MODEL_PROFILE=14b
MODEL_NAME=Qwen/Qwen3-14B-AWQ
MODEL_GPU_MEMORY_UTILIZATION=0.75
MODEL_MAX_MODEL_LEN=16384
MODEL_ENABLE_PREFIX_CACHING=1
MODEL_MAX_NUM_SEQS=1
MODEL_ENABLE_CHUNKED_PREFILL=1
MODEL_QUANTIZATION=awq_marlin
MODEL_TRUST_REMOTE_CODE=1
MODEL_ENFORCE_EAGER=1
MODEL_SWAP_SPACE=
MODEL_EXTRA_ARGS=
```

`./chat` reads this file, writes the active selection into `.runtime-model.env`, and `chat-app` also loads the same defaults file on startup when explicit environment overrides are absent.

For machine-specific tuning, copy [config/model-overrides.local.env.sample](/Users/joe/dev/ai-chat/config/model-overrides.local.env.sample) to `config/model-overrides.local.env`. That local override file is git-ignored, so it is safe for personal VRAM tuning, host-specific endpoints, and similar machine-only settings.

Load order is:

1. exported process environment
2. `config/model-overrides.local.env`
3. `config/model-defaults.env`

This means each tuning knob is a one-line edit. For example:

- change model: `MODEL_NAME=...`
- change VRAM cap: `MODEL_GPU_MEMORY_UTILIZATION=0.80`
- change context: `MODEL_MAX_MODEL_LEN=16384`
- change concurrency: `MODEL_MAX_NUM_SEQS=1`
- add one-off flags: `MODEL_EXTRA_ARGS=--swap-space 8`

New users can clone the repo and run the app with only the checked-in defaults. You only need the local override file if you want machine-specific customization.

## Host requirements

The checked-in Docker setup assumes a local host that can run GPU-backed vLLM:

- Docker with the `docker compose` plugin
- NVIDIA GPU support exposed to Docker
- enough disk for Docker images, the Hugging Face cache, and app data
- `bash` and `curl` for the `./chat` helper script

Workspace roots are path-backed catalog entries. Managed workspace directories are typically bind-mounted under `./workspaces`, while legacy hosted runs may still exist under `./runs` after migration.

## Runtime model tuning

The public config surface is now generic `MODEL_*` tuning knobs. The launcher still keeps a lightweight internal `14b` state key for backward compatibility, but the intended user-facing interface is:

- choose `MODEL_NAME`
- tune `MODEL_GPU_MEMORY_UTILIZATION`
- tune `MODEL_MAX_MODEL_LEN`
- tune `MODEL_MAX_NUM_SEQS`
- add one-off flags with `MODEL_EXTRA_ARGS`

Legacy `MODEL_14B_*` overrides are still accepted as compatibility fallbacks, but new configs should prefer the generic names.

The generated runtime env passed into Docker Compose contains the selected model plus the shared profile defaults:

```yaml
environment:
  - VLLM_HOST=http://vllm:8000/v1
  - MODEL_NAME=${VLLM_SELECTED_MODEL_NAME}
  - MODEL_ARGS=${VLLM_SELECTED_MODEL_ARGS}
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

- `./chat install` is non-interactive and prepares the currently configured model tune.
- `./chat start` is non-interactive and reuses the installed/configured model tune.
- The launcher accepts chained commands in one invocation, so `./chat install start` and `./chat stop install start` run sequentially.

## GPU configurations

24GB-class GPUs: the default config is the intended target.

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
