FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    python-multipart \
    uvicorn[standard] \
    websockets \
    httpx \
    huggingface_hub \
    beautifulsoup4 \
    jinja2 \
    aiofiles \
    pandas \
    openpyxl

# System deps for voice tools (STT/TTS) and text PDF extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Voice: OpenAI Whisper (STT) and Piper (TTS)
RUN pip install --no-cache-dir openai-whisper piper-tts

# Copy application (app imports themes, prompts, thinking_stream, workflow_router)
COPY app.py themes.py prompts.py thinking_stream.py workflow_router.py .
COPY static/ static/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "app.py"]
