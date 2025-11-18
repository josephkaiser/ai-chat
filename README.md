# AI Chat with RAG & Web Search - Enhanced Edition

A self-hosted AI chat application with document understanding (RAG) and web search capabilities. This enhanced version includes improved error handling, better UI/UX, and additional features.

## ✨ New Features

### Improvements Over Original
- **Enhanced Error Handling**: Comprehensive error handling with proper logging
- **Better Database Management**: Context managers, indexes, and improved schema
- **Improved UI/UX**: Modern dark theme with smooth animations
- **Better WebSocket Handling**: Proper reconnection logic and error recovery
- **Performance Optimizations**: Database indexing, better caching strategies
- **Enhanced RAG**: Better chunking algorithm, similarity thresholds
- **Conversation Management**: Archive, delete, and better organization
- **Security Improvements**: Input validation, better error messages
- **Logging System**: Comprehensive logging for debugging
- **Source Attribution**: Track which documents/sources were used
- **Better Code Organization**: Modular structure with type hints

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- (Optional) NVIDIA GPU + nvidia-docker for GPU support
- Ollama installed and running on your host machine

### Installation

1. **Clone or download this directory**

2. **Start the application**
   ```bash
   docker-compose up -d
   ```

3. **Wait for models to download** (first run only, ~5-10 minutes)
   ```bash
   docker-compose logs -f
   ```

4. **Open your browser**
   ```
   http://localhost:8000
   ```

### GPU Support

Uncomment the GPU section in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## 📖 Usage

### Document Upload (RAG)
1. Click "Upload Document" in the sidebar
2. Select a `.txt`, `.md`, or `.pdf` file
3. Document will be automatically chunked and indexed
4. Enable "RAG" toggle to search your documents

### Web Search
1. Enable "Search" toggle in the header
2. Ask questions that benefit from current information
3. Results are cached for 1 hour

### Conversations
- Create new chats with the "+ New Chat" button
- Click on any conversation in the sidebar to load it
- Conversations are automatically titled based on first message

### Model Selection
Choose from available models in the dropdown:
- **Qwen Coder 7B**: Best for coding and technical tasks
- **Mistral 7B**: General purpose model
- **Llama 3.2 3B**: Faster, lightweight model

## 🏗️ Architecture

### Components
- **FastAPI**: Web framework with async support
- **WebSocket**: Real-time streaming responses
- **SQLite**: Local database for conversations and documents
- **Sentence Transformers**: Document embeddings for RAG
- **Ollama**: LLM backend
- **BeautifulSoup**: Web scraping for search

### Database Schema
```sql
conversations (
  id TEXT PRIMARY KEY,
  title TEXT,
  model TEXT,
  created_at TEXT,
  updated_at TEXT,
  message_count INTEGER,
  is_archived BOOLEAN
)

messages (
  id INTEGER PRIMARY KEY,
  conversation_id TEXT,
  role TEXT,
  content TEXT,
  timestamp TEXT,
  tokens INTEGER,
  used_rag BOOLEAN,
  used_search BOOLEAN,
  rag_sources TEXT,
  search_queries TEXT
)

documents (
  id TEXT PRIMARY KEY,
  filename TEXT,
  content TEXT,
  chunk_index INTEGER,
  embedding BLOB,
  file_size INTEGER,
  created_at TEXT
)

search_cache (
  query TEXT PRIMARY KEY,
  results TEXT,
  timestamp TEXT
)
```

## 🔧 Configuration

### Environment Variables
- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)
- `DATA_DIR`: Data directory path (default: `/app/data`)
- `HF_HOME`: Hugging Face cache directory (default: `/app/data/models`)

### Customization
Edit `app.py` to modify:
- Chunk size and overlap (default: 500 chars with 50 overlap)
- RAG similarity threshold (default: 0.3)
- Search cache duration (default: 1 hour)
- Max tokens, temperature, etc.

## 📊 API Endpoints

### REST API
- `GET /`: Web interface
- `GET /health`: Health check
- `POST /api/upload`: Upload document
- `GET /api/documents`: List documents
- `DELETE /api/document/{filename}`: Delete document
- `GET /api/conversations`: List conversations
- `GET /api/conversation/{id}`: Get conversation messages
- `DELETE /api/conversation/{id}`: Delete conversation
- `GET /api/models`: List available Ollama models

### WebSocket
- `WS /ws/chat`: Real-time chat with streaming responses

## 🛠️ Development

### Local Development (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access at http://localhost:8000
```

### Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test document upload
curl -X POST -F "file=@test.txt" http://localhost:8000/api/upload

# Test WebSocket (requires wscat)
wscat -c ws://localhost:8000/ws/chat
```

## 📁 File Structure
```
ai-chat-enhanced/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Docker Compose config
├── README.md            # This file
└── data/                # Created at runtime
    ├── chat.db          # SQLite database
    └── models/          # Cached ML models
```

## 🔍 Troubleshooting

### Application won't start
```bash
# Check logs
docker-compose logs -f

# Restart services
docker-compose restart

# Rebuild if needed
docker-compose up -d --build
```

### Ollama connection issues
- Ensure Ollama is running on your host
- Check `OLLAMA_HOST` environment variable
- Verify `host.docker.internal` resolves correctly

### WebSocket disconnects
- Check browser console for errors
- Verify firewall settings
- Check Docker network configuration

### RAG not finding documents
- Verify documents uploaded successfully
- Check minimum similarity threshold (0.3 default)
- Ensure RAG toggle is enabled

### Out of memory errors
- Reduce chunk size in code
- Use smaller embedding models
- Limit concurrent requests
- Add memory limits to docker-compose.yml:
  ```yaml
  deploy:
    resources:
      limits:
        memory: 4G
  ```

## 🔒 Security Considerations

### For Production Use
1. **Enable HTTPS**: Use reverse proxy (nginx/Caddy)
2. **Add Authentication**: Implement user authentication
3. **Rate Limiting**: Add rate limiting to API endpoints
4. **Input Validation**: Already included but review for your use case
5. **CORS Settings**: Restrict allowed origins
6. **Network Isolation**: Use Docker networks properly
7. **Secrets Management**: Use Docker secrets or environment files

### Example nginx config
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🎯 Performance Tips

1. **Use GPU**: Uncomment GPU section in docker-compose.yml
2. **Increase Cache**: Adjust search cache duration
3. **Optimize Chunks**: Tune chunk size for your documents
4. **Use Faster Models**: Switch to smaller/quantized models
5. **Add Redis**: For better caching (requires code modification)
6. **Database Optimization**: Already includes indexes

## 🤝 Contributing

This is an enhanced version of the original AI chat application. Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share your use cases

## 📝 Changelog

### Enhanced Version
- ✅ Improved error handling throughout
- ✅ Better database schema with indexes
- ✅ Context managers for database connections
- ✅ Comprehensive logging system
- ✅ Enhanced UI with modern design
- ✅ Better WebSocket reconnection logic
- ✅ Source attribution for RAG and search
- ✅ Input validation with Pydantic
- ✅ Better chunking algorithm
- ✅ Health check endpoint improvements
- ✅ Type hints throughout codebase
- ✅ Improved documentation

## 📄 License

This is an educational project. Use and modify as needed.

## 🙏 Acknowledgments

- Built with FastAPI, Ollama, and Sentence Transformers
- Inspired by modern chat interfaces
- Enhanced for production use

## 📞 Support

For issues or questions:
1. Check the logs: `docker-compose logs -f`
2. Review troubleshooting section
3. Check Ollama documentation
4. Verify system requirements

## 🔮 Future Enhancements

Potential improvements for future versions:
- [ ] Multi-user support with authentication
- [ ] Conversation export (JSON, Markdown)
- [ ] Advanced search filters
- [ ] Image upload and analysis
- [ ] Voice input/output
- [ ] Conversation sharing
- [ ] Custom prompt templates
- [ ] Model fine-tuning interface
- [ ] Analytics dashboard
- [ ] Mobile app
- [ ] Plugins system
- [ ] Integration with external APIs
- [ ] Collaborative features
- [ ] Advanced RAG techniques (HyDE, etc.)

---

**Made with ❤️ for the AI community**
