FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] websockets openai docker httpx duckduckgo-search beautifulsoup4

# Copy app and config
COPY app.py theme_config.py .

# Copy out folder (for gallery or other static assets)
# Note: The 'out' folder must exist in the build context
# If it doesn't exist, create an empty folder: mkdir -p out
COPY out/ /app/out/

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "app.py"]
