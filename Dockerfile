FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] websockets openai

# Copy app and config
COPY app.py theme_config.py .

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "app.py"]
