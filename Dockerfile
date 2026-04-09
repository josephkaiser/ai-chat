FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-voice.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# System deps for voice tools (STT/TTS) and text PDF extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Voice: OpenAI Whisper (STT) and Piper (TTS)
RUN pip install --no-cache-dir -r requirements-voice.txt

# Copy only the current top-level Python modules used by the app entrypoint.
COPY app.py embeddings.py themes.py prompts.py thinking_stream.py deep_flow.py turn_strategy.py ./
COPY static/ static/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "app.py"]
