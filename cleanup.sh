#!/bin/bash
# Cleanup script to stop dynamically created vLLM containers
# These containers are created by the app when switching models
# and aren't managed by docker-compose

echo "🧹 Cleaning up dynamically created vLLM containers..."

# Stop and remove all containers that start with "vllm-" but aren't the main "vllm" container
containers=$(docker ps -a --filter "name=vllm-" --format "{{.Names}}" | grep -v "^vllm$")

if [ -z "$containers" ]; then
    echo "✅ No dynamically created vLLM containers found"
else
    echo "📦 Found containers to clean up:"
    echo "$containers" | while read container; do
        echo "  - $container"
    done
    
    echo ""
    echo "🛑 Stopping and removing containers..."
    echo "$containers" | while read container; do
        docker rm -f "$container" 2>/dev/null && echo "  ✓ Removed $container" || echo "  ⚠️ Failed to remove $container"
    done
    
    echo "✅ Cleanup complete!"
fi

