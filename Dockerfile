FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    websockets \
    httpx \
    beautifulsoup4 \
    jinja2 \
    aiofiles

# Copy application (app imports themes, prompts, thinking_stream)
COPY app.py themes.py prompts.py thinking_stream.py .
COPY static/ static/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "app.py"]
