# AI Chat with RAG & Web Search

Self-hosted AI chat application with document understanding and web search.

## Quick Start

### Prerequisites
- Docker
- Docker Compose
- (Optional) NVIDIA GPU + nvidia-docker for GPU support

### Installation

1. Clone/download this directory
2. Run:
```bash
   docker-compose up -d
```
3. Wait for models to download (first run only, ~5-10 minutes)
4. Open http://localhost:8000

### GPU Support

Uncomment the GPU section in docker-compose.yml:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Usage

- Upload documents via the UI
- Enable RAG to search your documents
- Enable Search to query the web
- All data stored in `./data/` directory

### Stopping
```bash
docker-compose down
```

### Backup

Your data is in `./data/chat.db` - just copy this file!

### Updating
```bash
docker-compose pull
docker-compose up -d
```

### First Time Setup
```bash
# 1. Create the directory
mkdir ai-chat && cd ai-chat

# 2. Create all the files above (or download them)

# 3. Make the init script executable
chmod +x init-models.sh

# 4. Start everything
docker-compose up -d

# 5. Watch the logs
docker-compose logs -f

# 6. Wait for "Models ready!" message

# 7. Open browser
open http://localhost:8000
```

**7. .dockerignore**
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
.env
.venv
data/
*.db
.git/
.gitignore
