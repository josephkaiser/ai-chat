#!/bin/bash
# Stop all containers (docker-compose managed + dynamically created)
# Usage: ./stop.sh

echo "🛑 Stopping all containers..."

# First, stop docker-compose managed containers
echo "📦 Stopping docker-compose containers..."
docker compose down

# Then, clean up dynamically created vLLM containers
echo "🧹 Cleaning up dynamically created vLLM containers..."
containers=$(docker ps -a --filter "name=vllm-" --format "{{.Names}}" | grep -v "^vllm$")

if [ -z "$containers" ]; then
    echo "✅ No dynamically created vLLM containers found"
else
    echo "$containers" | while read container; do
        docker rm -f "$container" 2>/dev/null && echo "  ✓ Removed $container" || echo "  ⚠️ Failed to remove $container"
    done
fi

echo "✅ All containers stopped!"

