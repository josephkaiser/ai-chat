# Configuration

Edit `docker-compose.yml` to change the model or vLLM settings:

```yaml
vllm:
  command: >
    --model ServiceNow-AI/Apriel-1.6-15b-Thinker
    --gpu-memory-utilization 0.90
    --max-model-len 32768
    --swap-space 8
    --enable-prefix-caching
    --max-num-seqs 256
    --enable-chunked-prefill
```

Set `MODEL_NAME` in the `chat-app` environment to match.

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
