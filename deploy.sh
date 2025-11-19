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
if docker ps | grep -q ai-chat-vllm; then
    echo "✅ Container is running!"
    echo "📋 Recent logs:"
    docker compose logs --tail=20 chat-app
else
    echo "❌ Container failed to start. Check logs:"
    docker compose logs chat-app
    exit 1
fi

echo "✨ Deployment complete!"
echo "💡 Don't forget to hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)"

