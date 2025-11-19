# AI Chat with vLLM - Full-Featured Chat Interface

A comprehensive AI chat application with a modern web UI, running on a single GPU server using vLLM. Features include dynamic model switching, web search, code execution, file system access, and intelligent context management.

## Features

### Core Features
- ✅ **Modern Web UI** - Clean, responsive interface with light/dark mode
- ✅ **Dynamic Model Switching** - Switch between models on-the-fly with confirmation dialogs
- ✅ **Large Context Window** - 32K token context with smart history management (uses up to 75% for conversation history)
- ✅ **Quantization Support** - AWQ Marlin quantization for efficient GPU memory usage
- ✅ **Job Queue System** - Efficient batching to minimize model switches
- ✅ **Model Health Monitoring** - Automatic recovery if model becomes unresponsive

### Advanced Features
- ✅ **Automatic Web Search** - AI automatically searches the web for current events, recent information, and up-to-date facts
- ✅ **Manual Web Search** - Search Google, Wikipedia, Reddit, GitHub, and Stack Overflow via UI button
- ✅ **Code Execution** - Sandboxed Python code execution with resource limits
- ✅ **File System Access** - Read files and traverse directories (with security checks)
- ✅ **Chat History Search** - Full-text search through all conversations
- ✅ **Feedback & Retry** - Provide feedback on responses and retry with extended thinking
- ✅ **Terminal Log Viewer** - Real-time view of vLLM logs and model status with helpful startup messages
- ✅ **Prefix Caching** - Faster responses for repeated prompts
- ✅ **Chunked Prefill** - Optimized context processing
- ✅ **Fast Startup** - Web UI loads immediately, model loading happens in background

### UI Features
- ✅ **Collapsible Sidebar** - Conversation history with search
- ✅ **Markdown Rendering** - Full markdown support with syntax highlighting
- ✅ **Copy to Clipboard** - Easy copying of code and responses
- ✅ **Timestamps** - Message timestamps for all conversations
- ✅ **Mobile Responsive** - Works great on mobile devices with optimized touch interactions
- ✅ **Auto-resizing Input** - Text input grows with content
- ✅ **Status Indicator** - Visual connection status (green=connected, yellow=booting/loading, red=disconnected)
- ✅ **Theme Support** - Light and dark modes with customizable colors

## Quick Start

```bash
# One command - everything auto-configured for GPU
docker compose up -d

# Wait for model download (first time, ~2-4GB)
docker compose logs -f vllm

# When ready, open:
# http://localhost:8000
```

**Note:** Uses Docker Compose v2 (`docker compose`). For v1, use `docker-compose` instead.

vLLM will auto-detect your GPU and optimize its settings accordingly.

### Restarting Services

```bash
# Restart everything
docker compose restart

# Restart only the chat-app (faster, keeps model loaded)
docker compose restart chat-app

# Restart only vLLM (reloads model, takes 2-5 minutes)
docker compose restart vllm

# View logs
docker compose logs -f chat-app
docker compose logs -f vllm
```

### Gated Models (Llama, etc.)

Some models like Llama require a Hugging Face token to access. To use these models:

1. Get your Hugging Face token from https://huggingface.co/settings/tokens
2. Request access to the model (e.g., https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
3. Create a `.env` file in the project root:

```bash
echo "HF_TOKEN=your_token_here" > .env
```

The `.env` file is gitignored for security. The token will be automatically passed to vLLM containers when switching models.

**Security Note:** Never commit your `.env` file. It's already in `.gitignore`.

## Current Configuration

The default setup is optimized for a 24GB GPU (RTX 3090/4090):

```yaml
vllm:
  command: >
    --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
    --gpu-memory-utilization 0.95      # Use 95% of GPU memory
    --max-model-len 32768               # 32K context window
    --quantization awq_marlin           # AWQ Marlin quantization
    --swap-space 8                      # 8GB swap for overflow
    --enable-prefix-caching             # Cache common prefixes
    --max-num-seqs 256                  # Handle up to 256 concurrent sequences
    --enable-chunked-prefill            # Optimized context processing
```

### Context Management

The system intelligently manages context:
- **75% of context window** (24K tokens) reserved for conversation history
- **25% of context window** (8K tokens) for system prompt + new input + response
- History is loaded from most recent backwards, maximizing context usage
- Database indexes ensure fast history retrieval even for long conversations

### For Different GPUs

**RTX 3090 / 4090 (24GB):**
```yaml
--gpu-memory-utilization 0.95
--max-model-len 32768
--swap-space 8
```

**A100 (40GB/80GB):**
```yaml
--gpu-memory-utilization 0.95
--max-model-len 65536  # Can go even higher
--swap-space 16
```

**RTX 3060 (12GB):**
```yaml
--gpu-memory-utilization 0.90
--max-model-len 16384
--swap-space 4
```

## Web Interface

### Main Features

1. **Model Switching** - Click the model button (🤖) in the header to switch models
   - Confirmation dialog warns about experimental nature
   - Status console shows progress
   - Automatic recovery to default model on failure

2. **Web Search** - Two ways to search:
   - **Automatic**: The AI automatically searches the web when you ask about current events, recent information, or topics that may have changed (e.g., "What's the latest news about...", "Current status of...", "Recent updates on...")
   - **Manual**: Click "🔍 Web" button to search online
     - Select sources: Google, Wikipedia, Reddit, GitHub, Stack Overflow
     - Results open in new tabs
     - Press Enter to search, Escape to close

3. **Code Execution** - The AI can execute Python code
   - Use `[EXECUTE_CODE:python]` in prompts
   - Sandboxed with 5-second timeout
   - Results included in response

4. **File Access** - The AI can read files and list directories
   - Use `[READ_FILE:path/to/file]` to read files
   - Use `[LIST_DIR:path/to/directory]` to list directories
   - Security checks prevent directory traversal

5. **Search History** - Search box in sidebar searches all conversations
   - Full-text search across all messages
   - Results grouped by conversation
   - Click to jump to conversation

6. **Feedback & Retry** - Each AI response has buttons for:
   - 👍 Positive feedback
   - 👎 Negative feedback
   - 🔄 Retry (same prompt)
   - 🔄 Retry with extended thinking

### Keyboard Shortcuts

- **Escape** - Close any open modal/popup
- **Enter** - Submit message (in input box)
- **Enter** - Trigger search (in web search modal)

## API Endpoints

### Chat
- `GET /` - Web interface
- `WebSocket /ws/chat` - Streaming chat
- `WebSocket /ws/logs` - Terminal logs

### Models
- `GET /api/models` - List available models
- `GET /api/model/status` - Get model switch status and history
- `POST /api/model/switch` - Switch to a different model
- `POST /api/model/select` - Select model for conversation (queued)

### Conversations
- `GET /api/conversations` - List all conversations
- `GET /api/conversation/{id}` - Get conversation with messages
- `POST /api/conversation/{id}/rename` - Rename conversation
- `DELETE /api/conversation/{id}` - Delete conversation

### Messages
- `POST /api/message/{id}/feedback` - Submit feedback (positive/negative)
- `POST /api/message/{id}/retry` - Retry message generation

### Search
- `GET /api/search?query=...` - Search chat history
- `POST /api/web-search` - Search the web

### Code & Files
- `POST /api/execute-code` - Execute Python code in sandbox
- `GET /api/files/list?path=...` - List directory contents
- `GET /api/files/read?path=...` - Read file content

### Health
- `GET /health` - Check model availability and status

## System Requirements

**Minimum:**
- GPU: RTX 3060 (12GB VRAM)
- RAM: 16GB
- Storage: 10GB free
- CUDA: 12.1+ (or 11.8 with `cu118` image tag)

**Recommended:**
- GPU: RTX 4090 or A100 (24GB+ VRAM)
- RAM: 32GB+
- Storage: 20GB+ free (for model cache)
- CUDA: 12.8+ (or 13.0+)

**Optimal:**
- GPU: A100 80GB or H100
- RAM: 64GB+
- Storage: 50GB+ (for multiple models)
- CUDA: Latest

## First Run

```bash
docker compose up -d

# Watch vLLM startup (shows GPU detection)
docker compose logs -f vllm
```

**You'll see:**
```
Detected GPU: NVIDIA GeForce RTX 4090
GPU Memory: 24GB
Loading model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ...
Model loaded successfully
Starting server on 0.0.0.0:8000
```

**First run:** 2-5 minutes (downloads model ~2-4GB)
**Later runs:** 30-60 seconds (model cached, faster with prefix caching)

**Note:** The web UI loads immediately - you don't have to wait for the model to finish loading. The status indicator shows the connection state, and you can monitor progress in the Log Viewer (📋 Logs button).

## Monitoring

### Check GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop (prettier)
nvtop
```

### View Logs
```bash
# All services
docker compose logs -f

# Just vLLM
docker compose logs -f vllm

# Just chat app
docker compose logs -f chat-app
```

### Check Health
```bash
# API health endpoint
curl http://localhost:8000/health

# vLLM metrics (if enabled)
curl http://localhost:8001/metrics
```

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce memory usage
--gpu-memory-utilization 0.80
--max-model-len 16384
--swap-space 4
```

### Model Download Fails
```bash
# Check Hugging Face token (for gated models)
cat .env | grep HF_TOKEN

# Pre-download model manually
docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=your_token \
  vllm/vllm-openai:latest \
  python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct-AWQ')"
```

### Slow Generation
```bash
# Check GPU is actually being used
nvidia-smi

# Check vLLM detected GPU
docker compose logs vllm | grep GPU

# Verify prefix caching is working (should see cache hits in logs)
docker compose logs vllm | grep cache
```

### Port Already in Use
```yaml
# Change ports in docker-compose.yml
vllm:
  ports:
    - "8002:8000"  # Changed from 8001

chat-app:
  ports:
    - "8001:8000"  # Changed from 8000
```

### Model Switch Fails
- Check logs: `docker compose logs chat-app | grep "Model switch"`
- Verify Docker socket is mounted: `docker compose exec chat-app ls /var/run/docker.sock`
- Check model name is correct in `app.py` `AVAILABLE_MODELS`
- For gated models, ensure `HF_TOKEN` is set in `.env`

### Application Stalls on Startup
- The app loads the UI immediately and checks for model in background (non-blocking startup)
- Check model status in the web UI (status indicator in header - yellow means loading, green means ready)
- Open the Log Viewer (📋 Logs button) to see detailed startup progress and helpful messages
- View logs: `docker compose logs -f chat-app`
- Model health monitor will attempt recovery automatically
- Boot time has been optimized with reduced wait times for faster startup

## Performance Tips

1. **Keep vLLM Running** - Only restart `chat-app` when updating code, not `vLLM`
   ```bash
   docker compose restart chat-app  # Fast restart
   ```

2. **Use Model Cache** - Models are cached in `~/.cache/huggingface`, so restarts are faster

3. **Prefix Caching** - Repeated prompts (like system prompts) are cached for faster responses

4. **Context Management** - The system automatically uses up to 75% of context window for history

5. **Database Indexes** - Conversation history queries are optimized with indexes

## Security

For production:
1. Add authentication to the chat app
2. Use HTTPS (reverse proxy with nginx/Caddy)
3. Restrict vLLM port (only accessible to chat-app via Docker network)
4. Rate limiting on endpoints
5. Monitor GPU usage and set alerts
6. Review file system access paths (currently allows reading from project directory)

## Project Structure

```
ai-chat/
├── app.py              # Main FastAPI application
├── theme_config.py     # UI theme configuration (colors, fonts, dimensions)
├── docker-compose.yml  # Docker services configuration
├── Dockerfile         # Chat app container
├── .env               # Environment variables (gitignored)
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── data/              # Created automatically
    └── chat.db        # SQLite database (conversations, messages)
```

## Advanced Configuration

### Custom Models

Edit `app.py` to add models to `AVAILABLE_MODELS`:

```python
AVAILABLE_MODELS = [
    {
        "id": "your-model-id",
        "name": "Your Model Name",
        "quantized": False,
        "command": [
            "--model", "your-model-id",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "32768",
            # ... other vLLM args
        ]
    },
]
```

### Theme Customization

Edit `theme_config.py` to customize:
- Colors (light/dark mode)
- Fonts and sizes
- Dimensions (sidebar width, message width, etc.)
- Animations

### Context Window Tuning

Adjust in `docker-compose.yml`:
- `--max-model-len` - Maximum context window
- `--swap-space` - SSD swap space for overflow (GB)
- `--gpu-memory-utilization` - GPU memory usage (0.0-1.0)

Adjust in `app.py`:
- `get_conversation_history()` - Percentage of context for history (currently 75%)
- `estimate_tokens_needed()` - Maximum response tokens (currently 24K)

## Scaling

### Vertical (Bigger GPU)
- Upgrade to A100/H100
- Use larger models (70B+)
- Increase `--max-model-len` to 65K+
- Increase `--swap-space` for more overflow capacity

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

## License

See LICENSE file for details.
