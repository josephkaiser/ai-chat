# AI Chat with vLLM - GPU Optimized

AI chat web server that can be run on a single GPU server (e.g. a desktop PC) using vLLM and WebAI. 

**Next Steps**
- Quantization to run larger models / cheaper inference

## Quick Start

```bash
# One command - everything auto-configured for GPU
docker-compose up -d

# Wait for model download (first time, ~2GB)
docker-compose logs -f vllm

# When ready, open:
# http://localhost:8000
```

vLLM will auto-detect your GPU and optimize its settings accordingly.

## Configuration

### Current Setup (Optimized)

```yaml
vllm:
  command: >
    --model meta-llama/Llama-3.2-3B-Instruct
    --gpu-memory-utilization 0.90    # Use 90% of GPU memory
    --max-model-len 4096             # Max context length
    --dtype auto                      # Auto-detect best dtype
```

### For Different GPUs

**RTX 3090 / 4090 (24GB):**
```yaml
--gpu-memory-utilization 0.90
--max-model-len 8192
--dtype float16
```

**A100 (40GB/80GB):**
```yaml
--gpu-memory-utilization 0.95
--max-model-len 16384
--dtype bfloat16
```

**RTX 3060 (12GB):**
```yaml
--gpu-memory-utilization 0.85
--max-model-len 2048
--dtype float16
```

### Use Larger Models

**Llama 3.1 8B:**
```yaml
--model meta-llama/Llama-3.1-8B-Instruct
--gpu-memory-utilization 0.90
--max-model-len 4096
```

**Mistral 7B:**
```yaml
--model mistralai/Mistral-7B-Instruct-v0.3
--gpu-memory-utilization 0.90
--max-model-len 8192
```

**Qwen 2.5 7B Coder:**
```yaml
--model Qwen/Qwen2.5-Coder-7B-Instruct
--gpu-memory-utilization 0.90
--max-model-len 4096
```

Update `app.py` too:
```python
stream = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",  # Match docker-compose
    messages=messages,
    stream=True,
)
```

## Advanced Features

### Enable Speculative Decoding (Faster!)
```yaml
command: >
  --model meta-llama/Llama-3.2-3B-Instruct
  --speculative-model meta-llama/Llama-3.2-1B-Instruct
  --num-speculative-tokens 5
  --gpu-memory-utilization 0.85
```

30-50% faster generation!

### Tensor Parallelism (Multi-GPU)
```yaml
command: >
  --model meta-llama/Llama-3.1-8B-Instruct
  --tensor-parallel-size 2  # Use 2 GPUs
  --gpu-memory-utilization 0.90
```

### Quantization (Save Memory)
```yaml
command: >
  --model meta-llama/Llama-3.1-8B-Instruct
  --quantization awq         # or 'gptq', 'squeezellm'
  --gpu-memory-utilization 0.95
```

Run larger models on smaller GPUs!

## 📁 What's Inside

```
ai-chat-vllm/
├── app.py              # vLLM-optimized chat app
├── docker-compose.yml  # GPU-optimized configuration
├── Dockerfile         # App container
└── README.md          # This file

Created automatically:
└── data/
    └── chat.db        # Conversations
```

## System Requirements

**Minimum:**
- GPU: RTX 3060 (12GB VRAM)
- RAM: 16GB
- Storage: 10GB free

**Recommended:**
- GPU: RTX 4090 or A100 (24GB+ VRAM)
- RAM: 32GB+
- Storage: 20GB+ free

**Optimal:**
- GPU: A100 80GB or H100
- RAM: 64GB+
- Storage: 50GB+ (for multiple models)

## First Run

```bash
docker-compose up -d

# Watch vLLM startup (shows GPU detection)
docker-compose logs -f vllm
```

**You'll see:**
```
Detected GPU: NVIDIA GeForce RTX 4090
GPU Memory: 24GB
Loading model meta-llama/Llama-3.2-3B-Instruct...
Model loaded successfully
Starting server on 0.0.0.0:8000
```

**First run:** 2-3 minutes (downloads model)
**Later runs:** 30-60 seconds (model cached)

## Monitoring

### Check GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop (prettier)
nvtop
```

### Check vLLM Stats
```bash
# vLLM metrics endpoint
curl http://localhost:8001/metrics
```

### View Logs
```bash
# All services
docker-compose logs -f

# Just vLLM
docker-compose logs -f vllm

# Just chat app
docker-compose logs -f chat-app
```

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce memory usage
--gpu-memory-utilization 0.80
--max-model-len 2048
```

### Model Download Fails
```bash
# Pre-download model
docker run --gpus all -v vllm_cache:/root/.cache/huggingface \
  vllm/vllm-openai \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --download-dir /root/.cache/huggingface
```

### Slow Generation
```bash
# Check GPU is actually being used
nvidia-smi

# Check vLLM detected GPU
docker-compose logs vllm | grep GPU
```

### Port Already in Use
```yaml
# Change vLLM port in docker-compose.yml
ports:
  - "8002:8000"  # Changed from 8001
```

## Benchmarking

### Test Single User Speed
```bash
# Time a simple query
time curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

### Test Concurrent Users
```bash
# Install apache bench
apt-get install apache2-utils

# 100 requests, 10 concurrent
ab -n 100 -c 10 -p request.json -T application/json \
  http://localhost:8001/v1/chat/completions
```

## vLLM vs Ollama Decision

**Use vLLM (this version) if:**
- You have a GPU server
- You need maximum performance
- You have multiple users
- You're deploying to production
- You want advanced features (batching, quantization)

**Use Ollama if:**
- You're on a laptop/desktop
- You have a single user
- You want simpler setup
- You're prototyping
- You don't need maximum performance

## Security

For production:
1. Add authentication to the chat app
2. Use HTTPS (reverse proxy with nginx/Caddy)
3. Restrict vLLM port (only accessible to chat-app)
4. Rate limiting on endpoints
5. Monitor GPU usage and set alerts

## Scaling

### Vertical (Bigger GPU)
- Upgrade to A100/H100
- Use larger models (70B+)
- Increase batch size

### Horizontal (More GPUs)
```yaml
# Tensor parallelism across GPUs
--tensor-parallel-size 4
```

### Load Balancing
Run multiple vLLM instances behind nginx:
```nginx
upstream vllm_backend {
    server vllm1:8000;
    server vllm2:8000;
    server vllm3:8000;
}
```
