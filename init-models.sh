#!/bin/sh

echo "🚀 Pulling initial models..."

# Wait for Ollama to be ready
sleep 5

# Pull your preferred models
ollama pull qwen2.5-coder:7b
#ollama pull qwen2.5-coder:32b
#ollama pull mistral:7b
#ollama pull llama3.2:3b

echo "✓ Models ready!"
