FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    websockets \
    httpx \
    duckduckgo-search \
    beautifulsoup4 \
    jinja2 \
    aiofiles

# Copy app and static files
COPY app.py .
COPY static/ static/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "app.py"]
