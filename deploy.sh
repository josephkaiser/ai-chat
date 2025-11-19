#!/bin/bash
# Quick deployment script for Ubuntu server
# Usage: ./deploy.sh
# Uses Docker Compose v2 (docker compose)

set -e

echo "🚀 Starting deployment..."

# Pull latest code
echo "📥 Pulling latest code from GitHub..."
git pull origin main

# Optional: Clean up dynamically created vLLM containers
# Uncomment the next 2 lines if you want to clean them up during deployment
# echo "🧹 Cleaning up old vLLM containers..."
# ./cleanup.sh

# Rebuild the chat-app container
echo "🔨 Rebuilding chat-app container..."
docker compose build chat-app

# Restart the container
echo "🔄 Restarting chat-app container..."
docker compose up -d chat-app

# Wait a moment for container to start
sleep 2

# Check if container is running
if ! docker ps | grep -q ai-chat-vllm; then
    echo "❌ Container failed to start. Check logs:"
    docker compose logs chat-app
    exit 1
fi

echo "✅ Container is running!"
echo "📋 Showing logs until model is ready..."
echo "⏳ Waiting for model to load (this may take a few minutes)..."
echo ""

# Function to check if model is ready via health endpoint
check_model_ready() {
    local container_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ai-chat-vllm 2>/dev/null)
    if [ -z "$container_ip" ]; then
        # Try localhost if we can't get container IP
        curl -s http://localhost:8000/health 2>/dev/null | grep -q '"model_available":true' && return 0 || return 1
    else
        curl -s http://${container_ip}:8000/health 2>/dev/null | grep -q '"model_available":true' && return 0 || return 1
    fi
}

# Show logs and monitor for model readiness
MAX_WAIT=600  # 10 minutes max
ELAPSED=0
CHECK_INTERVAL=5  # Check every 5 seconds

# Start showing logs in the background
docker compose logs -f chat-app &
LOGS_PID=$!

# Monitor for model readiness
while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
    
    # Check if model is ready
    if check_model_ready; then
        echo ""
        echo "✅ Model is ready!"
        # Kill the logs process
        kill $LOGS_PID 2>/dev/null || true
        wait $LOGS_PID 2>/dev/null || true
        break
    fi
    
    # Show progress every 30 seconds
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo ""
        echo "⏳ Still waiting... (${ELAPSED}s elapsed)"
    fi
done

# Clean up logs process if still running
kill $LOGS_PID 2>/dev/null || true
wait $LOGS_PID 2>/dev/null || true

# Final check
if check_model_ready; then
    echo ""
    echo "✨ Deployment complete! Model is ready to use."
    echo "💡 Don't forget to hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)"
else
    echo ""
    echo "⚠️  Deployment complete, but model may still be loading."
    echo "💡 Check the logs above or use the Log Viewer in the web UI to monitor progress."
    echo "💡 The model will be available once it finishes loading."
fi

