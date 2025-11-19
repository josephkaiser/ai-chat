#!/bin/bash
# Quick script to check vLLM status

echo "🔍 Checking vLLM container status..."
echo ""

# Check if container is running
if docker ps | grep -q "vllm$"; then
    echo "✅ vLLM container is running"
    CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' vllm 2>/dev/null)
    echo "   Container status: $CONTAINER_STATUS"
else
    echo "❌ vLLM container is NOT running"
    echo "   Start it with: docker compose up -d vllm"
    exit 1
fi

echo ""
echo "📋 Recent vLLM logs (last 50 lines):"
echo "-----------------------------------"
docker logs vllm --tail=50

echo ""
echo ""
echo "🔌 Testing vLLM API connection..."
echo "-----------------------------------"

# Check if API is responding
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8001/health 2>/dev/null)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ vLLM health endpoint is responding (HTTP $HTTP_CODE)"
    
    # Check if models endpoint works
    MODELS_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8001/v1/models 2>/dev/null)
    MODELS_HTTP_CODE=$(echo "$MODELS_RESPONSE" | tail -n1)
    if [ "$MODELS_HTTP_CODE" = "200" ]; then
        echo "✅ vLLM models endpoint is responding (HTTP $MODELS_HTTP_CODE)"
        echo ""
        echo "📦 Available models:"
        echo "$MODELS_RESPONSE" | head -n -1 | python3 -m json.tool 2>/dev/null || echo "$MODELS_RESPONSE" | head -n -1
        echo ""
        echo "✅ Model is ready! You should be able to chat now."
    else
        echo "⚠️  vLLM models endpoint returned HTTP $MODELS_HTTP_CODE"
        echo "   The model may still be loading. Check logs above for progress."
        echo "   Look for messages like 'Uvicorn running' and 'Application startup complete'"
    fi
else
    echo "❌ vLLM health endpoint returned HTTP ${HTTP_CODE:-unknown}"
    echo "   The model may still be loading. Check logs above for progress."
fi

echo ""
echo "🔍 Checking from inside chat-app container..."
echo "---------------------------------------------"
if docker exec ai-chat-vllm curl -s http://vllm:8000/health > /dev/null 2>&1; then
    echo "✅ Network connection from chat-app to vLLM is working"
else
    echo "⚠️  Cannot connect from chat-app to vLLM"
    echo "   This could mean:"
    echo "   1. The model is still loading (most likely)"
    echo "   2. There's a network issue between containers"
    echo "   Check vLLM logs: docker logs vllm -f"
fi

echo ""
echo "💡 Tips:"
echo "   - If the model is still loading, wait 2-5 minutes"
echo "   - Watch vLLM logs: docker logs vllm -f"
echo "   - Look for 'Uvicorn running' and model loading messages"
echo "   - Once you see 'Application startup complete', the model should be ready"

