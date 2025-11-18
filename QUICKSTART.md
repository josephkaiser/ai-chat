# Quick Start Guide - AI Chat Enhanced

## ⚡ Get Started in 5 Minutes

### Step 1: Prerequisites Check
```bash
# Check if Docker is installed
docker --version
docker-compose --version

# Check if Ollama is running
curl http://localhost:11434/api/tags
```

### Step 2: Start the Application
```bash
# Navigate to the directory
cd ai-chat-enhanced

# Start everything
docker-compose up -d

# Watch the logs
docker-compose logs -f
```

### Step 3: Wait for Ready
Look for these messages in the logs:
```
✓ Database initialized
✓ Embedding model loaded
🚀 Starting AI Chat with RAG & Search (Enhanced Version)...
📍 http://0.0.0.0:8000
```

### Step 4: Open Browser
Go to: http://localhost:8000

### Step 5: First Use

**Upload a Document:**
1. Click "Upload Document"
2. Choose a `.txt`, `.md`, or `.pdf` file
3. Wait for "Document added" confirmation

**Start Chatting:**
1. Type your message in the input box
2. Press Enter or click "Send"
3. Toggle RAG 📚 to search your documents
4. Toggle Search 🔍 for web results

## 🎨 Interface Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar                    │  Main Chat Area               │
│  ┌──────────────────┐      │  ┌─────────────────────────┐ │
│  │ + New Chat       │      │  │ AI Chat with RAG        │ │
│  └──────────────────┘      │  │ [RAG] [Search] [Model]  │ │
│                             │  └─────────────────────────┘ │
│  📄 Documents               │                              │
│  ├─ Upload Document         │  💬 Messages appear here    │
│  ├─ doc1.pdf               │                              │
│  └─ doc2.txt               │                              │
│                             │                              │
│  💬 Conversations           │  ┌─────────────────────────┐ │
│  ├─ Latest Chat            │  │ Type your message...    │ │
│  ├─ Previous Chat          │  │ [Send]                  │ │
│  └─ Older Chat             │  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Common Tasks

### Upload Multiple Documents
```bash
# Upload multiple files at once
for file in docs/*.pdf; do
    curl -X POST -F "file=@$file" http://localhost:8000/api/upload
done
```

### Export a Conversation
```bash
# Get conversation data
curl http://localhost:8000/api/conversation/{id} > conversation.json
```

### Check Health
```bash
curl http://localhost:8000/health
```

### View Logs
```bash
# All logs
docker-compose logs -f

# Just app logs
docker-compose logs -f chat-app

# Last 100 lines
docker-compose logs --tail=100 chat-app
```

## 🔧 Keyboard Shortcuts

- **Enter**: Send message
- **Shift + Enter**: New line in message
- **Esc**: Clear input (in future version)

## 💡 Tips

1. **Enable RAG** when asking about your documents
2. **Enable Search** when asking about current events
3. **Use both** for comprehensive answers combining your docs and web info
4. **Start specific questions** for better RAG results
5. **Choose the right model** based on your task

## 🐛 Quick Troubleshooting

**Problem: Can't connect**
```bash
# Check if running
docker-compose ps

# Restart
docker-compose restart

# Check Ollama
curl http://localhost:11434/api/tags
```

**Problem: Slow responses**
- Use smaller models (Llama 3.2 3B)
- Enable GPU support in docker-compose.yml
- Reduce document chunk size in app.py

**Problem: WebSocket disconnects**
```bash
# Check logs for errors
docker-compose logs -f chat-app

# Restart the app
docker-compose restart chat-app
```

## 🎓 Example Questions

**With RAG enabled:**
- "What are the main points in my document?"
- "Summarize the key findings from my reports"
- "Find information about [topic] in my documents"

**With Search enabled:**
- "What's the latest news on [topic]?"
- "Current price of [stock/crypto]"
- "Recent developments in [field]"

**With both enabled:**
- "How does this news relate to my documents?"
- "Compare my analysis with current market trends"
- "Update my report with latest information"

## 📞 Getting Help

1. Check the logs: `docker-compose logs -f`
2. Review [README.md](README.md) for detailed docs
3. See [IMPROVEMENTS.md](IMPROVEMENTS.md) for technical details
4. Check health endpoint: `curl localhost:8000/health`

## 🚀 Next Steps

Once comfortable with basics:
1. Explore the [full README](README.md)
2. Customize settings in `app.py`
3. Try different models
4. Set up GPU support
5. Configure for production

---

**Ready to go! 🎉**

Open http://localhost:8000 and start chatting!
