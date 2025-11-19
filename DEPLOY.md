# Deployment Guide

## After Pulling Changes from GitHub

When you pull code changes and want to refresh your Docker instance:

### Quick Refresh (Recommended)

```bash
# Navigate to your project directory
cd /path/to/ai-chat

# Pull latest changes from GitHub
git pull origin main

# Rebuild and restart the chat-app container (keeps vLLM running)
docker compose build chat-app
docker compose up -d chat-app
```

**Why this works:** The `chat-app` container contains your code (app.py, theme_config.py), so rebuilding it picks up your changes. The `vllm` container doesn't need to restart - it keeps the model loaded.

### Full Rebuild (If Quick Refresh Doesn't Work)

```bash
cd /path/to/ai-chat
git pull origin main

# Rebuild everything from scratch
docker compose build --no-cache chat-app
docker compose up -d chat-app
```

### Complete Restart (Only if Needed)

```bash
cd /path/to/ai-chat
git pull origin main

# Stop everything, rebuild, and restart
docker compose down
docker compose build
docker compose up -d
```

**Note:** This will reload the model, which takes 2-5 minutes. Only do this if you need to restart vLLM too.

### 2. Verify Changes Are Applied

```bash
# Check if the new files are in the container
docker exec ai-chat-vllm ls -la /app/

# Check if theme_config.py exists
docker exec ai-chat-vllm cat /app/theme_config.py

# View container logs
docker compose logs -f chat-app
```

### 3. Clear Browser Cache

In your browser:
- **Chrome/Edge**: Press `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac) for hard refresh
- **Firefox**: Press `Ctrl+F5` (Windows) or `Cmd+Shift+R` (Mac)
- Or open DevTools (F12) → Right-click refresh button → "Empty Cache and Hard Reload"

### 4. Quick Deploy Script

Use the included `deploy.sh` script:

```bash
chmod +x deploy.sh
./deploy.sh
```

Or manually:
```bash
cd /path/to/ai-chat
git pull origin main
docker compose build chat-app
docker compose up -d chat-app
```

## Cleaning Up Dynamically Created Containers

When you switch models, the app creates new vLLM containers dynamically (e.g., `vllm-qwen-qwen2.5-coder-7b-instruct-awq`). These aren't managed by docker-compose, so `docker compose down` won't stop them.

### Quick Cleanup

Use the cleanup script:
```bash
chmod +x cleanup.sh
./cleanup.sh
```

This stops and removes all dynamically created vLLM containers (keeps the main `vllm` container from docker-compose).

### Manual Cleanup

```bash
# List all vLLM containers
docker ps -a | grep vllm

# Stop and remove a specific container
docker rm -f vllm-qwen-qwen2.5-coder-7b-instruct-awq

# Or stop all vLLM containers (except the main one)
docker ps -a --filter "name=vllm-" --format "{{.Names}}" | grep -v "^vllm$" | xargs -r docker rm -f
```

### Full Cleanup (Stop Everything)

**Option 1: Use the stop script (recommended)**
```bash
./stop.sh
```

**Option 2: Manual cleanup**
```bash
# Stop docker-compose managed containers
docker compose down

# Clean up dynamically created containers
./cleanup.sh
```

The `stop.sh` script does both steps automatically.

## Troubleshooting

### If changes still don't appear:

1. **Force rebuild without cache:**
   ```bash
   docker compose build --no-cache chat-app
   docker compose up -d chat-app
   ```

2. **Check container is using new code:**
   ```bash
   docker exec ai-chat-vllm cat /app/app.py | head -20
   ```

3. **Restart the container:**
   ```bash
   docker compose restart chat-app
   ```

4. **Check for errors:**
   ```bash
   docker compose logs chat-app | tail -50
   ```

## Note

This guide uses **Docker Compose v2** (`docker compose` without hyphen). 
If you're using v1 (`docker-compose`), replace `docker compose` with `docker-compose` in all commands.

