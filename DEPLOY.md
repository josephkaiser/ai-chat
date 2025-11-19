# Deployment Guide

## After Pushing Changes to GitHub

When you push code changes and deploy on your Ubuntu server, you need to rebuild the Docker containers:

### 1. On Your Ubuntu Server

```bash
# Navigate to your project directory
cd /path/to/ai-chat

# Pull latest changes from GitHub
git pull origin main

# Rebuild the chat-app container (this rebuilds with new code)
docker compose build chat-app

# Restart the chat-app container
docker compose up -d chat-app

# Or rebuild and restart everything
docker compose down
docker compose build
docker compose up -d
```

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

