#!/usr/bin/env python3
"""
AI Chat with vLLM - Simplified single-model setup
- Uses vLLM with Qwen 3.5 via httpx (no OpenAI SDK)
- Markdown support for code and formatted text
- Web search, code execution, file browsing
- Theme support (light/dark)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import sqlite3
from datetime import datetime
import logging
import traceback
import io
import sys
import os
import asyncio
import json
import re
import httpx
import shutil
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# Theme colors
COLORS_LIGHT = {
    'bg_primary': '#fffefb', 'bg_secondary': '#f5f0e8', 'bg_tertiary': '#e9eced',
    'bg_quaternary': '#f5f0e8',
    'text_primary': '#366c9c', 'text_secondary': '#8194b1', 'text_tertiary': '#b0c9df',
    'accent_primary': '#8194b1', 'accent_hover': '#366c9c', 'accent_secondary': '#2563eb',
    'msg_user_bg': '#e9eced', 'msg_user_text': '#366c9c',
    'msg_assistant_bg': '#fffefb', 'msg_assistant_text': '#366c9c',
    'status_connected': '#10b981', 'status_disconnected': '#ef4444',
    'btn_primary': '#8194b1', 'btn_primary_hover': '#366c9c',
    'btn_danger': '#ef4444', 'btn_danger_hover': '#dc2626',
    'btn_secondary': '#e9eced', 'btn_secondary_hover': '#b0c9df',
    'modal_overlay': 'rgba(0,0,0,0.5)', 'modal_bg': '#fffefb',
    'scrollbar_track': '#f5f0e8', 'scrollbar_thumb': '#b0c9df', 'scrollbar_thumb_hover': '#8194b1',
}
COLORS_DARK = {
    'bg_primary': '#1a1a1a', 'bg_secondary': '#242424', 'bg_tertiary': '#333333',
    'bg_quaternary': '#242424',
    'text_primary': '#e0e0e0', 'text_secondary': '#a0a0a0', 'text_tertiary': '#666666',
    'accent_primary': '#8194b1', 'accent_hover': '#b0c9df', 'accent_secondary': '#4a9eff',
    'msg_user_bg': '#333333', 'msg_user_text': '#e0e0e0',
    'msg_assistant_bg': '#242424', 'msg_assistant_text': '#e0e0e0',
    'status_connected': '#10b981', 'status_disconnected': '#ef4444',
    'btn_primary': '#8194b1', 'btn_primary_hover': '#b0c9df',
    'btn_danger': '#ef4444', 'btn_danger_hover': '#dc2626',
    'btn_secondary': '#333333', 'btn_secondary_hover': '#444444',
    'modal_overlay': 'rgba(0,0,0,0.8)', 'modal_bg': '#242424',
    'scrollbar_track': '#242424', 'scrollbar_thumb': '#333333', 'scrollbar_thumb_hover': '#444444',
}

# Setup logging with capture
log_capture = io.StringIO()

class TeeHandler(logging.Handler):
    """Handler that writes to both stdout and StringIO"""
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + '\n')
            self.stream.flush()
        except Exception:
            self.handleError(record)

stdout_handler = logging.StreamHandler(sys.stdout)
capture_handler = TeeHandler(log_capture)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)
capture_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stdout_handler, capture_handler],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "/app/data/chat.db"
VLLM_HOST = os.getenv("VLLM_HOST", "http://vllm:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/cache/huggingface")

logger.info(f"Using vLLM at {VLLM_HOST} with model {MODEL_NAME}")

# ==================== vLLM Client (httpx) ====================

async def vllm_chat_stream(messages: list, max_tokens: int = 4096, temperature: float = 0.3):
    """Stream chat completions from vLLM using httpx"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        async with client.stream(
            "POST",
            f"{VLLM_HOST}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.15,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

async def vllm_health_check() -> bool:
    """Check if vLLM is healthy and model is loaded"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(VLLM_HOST.replace('/v1', '') + '/health')
            if resp.status_code == 200:
                models_resp = await client.get(f"{VLLM_HOST}/models")
                return models_resp.status_code == 200
    except Exception:
        pass
    return False

# ==================== Database ====================

def init_db():
    """Initialize database"""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id TEXT PRIMARY KEY,
                      title TEXT,
                      created_at TEXT,
                      updated_at TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS messages
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      conversation_id TEXT,
                      role TEXT,
                      content TEXT,
                      timestamp TEXT,
                      feedback TEXT,
                      FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')

        try:
            c.execute('ALTER TABLE messages ADD COLUMN feedback TEXT')
        except sqlite3.OperationalError:
            pass

        try:
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conv_timestamp ON messages(conversation_id, timestamp)')
        except sqlite3.OperationalError:
            pass

        conn.commit()
        conn.close()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

try:
    init_db()
except Exception as e:
    logger.error(f"Critical: Failed to initialize database: {e}")

# ==================== Models ====================

class RenameRequest(BaseModel):
    title: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: str

# ==================== Helper Functions ====================

def estimate_tokens_needed(prompt: str) -> int:
    """Estimate tokens needed based on user request"""
    prompt_lower = prompt.lower()

    page_matches = re.findall(r'(\d+)\s*pages?', prompt_lower)
    if page_matches:
        return int(page_matches[0]) * 700

    word_matches = re.findall(r'(\d+)\s*words?', prompt_lower)
    if word_matches:
        return int(int(word_matches[0]) * 1.3)

    token_matches = re.findall(r'(\d+)\s*tokens?', prompt_lower)
    if token_matches:
        return int(token_matches[0])

    if any(word in prompt_lower for word in ['long', 'detailed', 'comprehensive', 'extensive', 'thorough', 'complete']):
        return 20000

    estimated = len(prompt) * 0.25 * 10
    return min(max(int(estimated), 2048), 24000)

def calculate_message_relevance_score(msg: Dict, current_query: str, message_index: int, total_messages: int) -> float:
    """Calculate relevance and quality score for a message"""
    score = 0.0

    recency_ratio = message_index / max(total_messages, 1)
    score += 0.3 * recency_ratio

    feedback = msg.get('feedback', '').lower()
    if feedback == 'positive':
        score += 0.4
    elif feedback == 'negative':
        score -= 0.2

    if current_query:
        query_words = set(current_query.lower().split())
        content_words = set(msg.get('content', '').lower().split())
        common_words = query_words.intersection(content_words)
        if common_words:
            score += 0.3 * min(len(common_words) / max(len(query_words), 1), 1.0)

    content_length = len(msg.get('content', ''))
    if content_length > 500:
        score += 0.1
    if content_length > 1000:
        score += 0.1

    if msg.get('role') == 'user':
        score += 0.05

    return score

def get_conversation_history(conv_id: str, limit: int = None, max_tokens: int = None, current_query: str = None) -> List[Dict]:
    """Get conversation history, selecting messages by quality and relevance"""
    if max_tokens is None:
        model_max_len = 32768
        max_tokens = int(model_max_len * 0.75)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''SELECT id, role, content, timestamp, feedback FROM messages
                 WHERE conversation_id = ?
                 ORDER BY timestamp ASC''', (conv_id,))

    all_messages = []
    for row in c.fetchall():
        all_messages.append({
            'id': row[0], 'role': row[1], 'content': row[2],
            'timestamp': row[3], 'feedback': row[4] if row[4] else ''
        })

    conn.close()

    if not all_messages:
        return []

    total_messages = len(all_messages)
    min_recent_messages = min(10, total_messages // 4)

    recent_messages = all_messages[-min_recent_messages:]
    recent_tokens = sum((len(msg['content']) // 4) + 10 for msg in recent_messages)

    older_messages = all_messages[:-min_recent_messages] if min_recent_messages < total_messages else []

    scored_messages = []
    for idx, msg in enumerate(older_messages):
        score = calculate_message_relevance_score(msg, current_query or '', idx, len(older_messages))
        scored_messages.append((score, msg))

    scored_messages.sort(key=lambda x: x[0], reverse=True)

    selected_messages = list(recent_messages)
    total_tokens = recent_tokens
    remaining_tokens = max_tokens - total_tokens

    for score, msg in scored_messages:
        msg_tokens = (len(msg['content']) // 4) + 10
        if total_tokens + msg_tokens <= max_tokens and remaining_tokens > 0:
            msg_timestamp = msg['timestamp']
            insert_pos = 0
            for i, existing_msg in enumerate(selected_messages):
                if existing_msg['timestamp'] > msg_timestamp:
                    insert_pos = i
                    break
                insert_pos = i + 1
            selected_messages.insert(insert_pos, msg)
            total_tokens += msg_tokens
            remaining_tokens -= msg_tokens

    for msg in selected_messages:
        msg.pop('id', None)
        msg.pop('feedback', None)

    if limit and len(selected_messages) > limit:
        selected_messages = selected_messages[-limit:]

    return selected_messages

def search_messages(query: str, limit: int = 20) -> List[Dict]:
    """Search through all messages"""
    if not query or len(query.strip()) < 2:
        return []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    search_term = f"%{query.strip()}%"
    c.execute('''SELECT m.conversation_id, m.role, m.content, m.timestamp, c.title
                 FROM messages m
                 LEFT JOIN conversations c ON m.conversation_id = c.id
                 WHERE m.content LIKE ?
                 ORDER BY m.timestamp DESC
                 LIMIT ?''', (search_term, limit))

    results = []
    for row in c.fetchall():
        results.append({
            'conversation_id': row[0], 'role': row[1], 'content': row[2],
            'timestamp': row[3], 'conversation_title': row[4] or 'Untitled'
        })

    conn.close()
    return results

def save_message(conv_id: str, role: str, content: str) -> int:
    """Save message to database and return message ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('SELECT id FROM conversations WHERE id = ?', (conv_id,))
    if not c.fetchone():
        title = content[:50] + "..." if len(content) > 50 else content
        c.execute('''INSERT INTO conversations (id, title, created_at, updated_at)
                     VALUES (?, ?, ?, ?)''',
                  (conv_id, title, datetime.now().isoformat(), datetime.now().isoformat()))

    c.execute('''INSERT INTO messages (conversation_id, role, content, timestamp)
                 VALUES (?, ?, ?, ?)''',
              (conv_id, role, content, datetime.now().isoformat()))

    message_id = c.lastrowid

    c.execute('UPDATE conversations SET updated_at = ? WHERE id = ?',
              (datetime.now().isoformat(), conv_id))

    conn.commit()
    conn.close()
    return message_id

def update_message_feedback(message_id: int, feedback: str):
    """Update feedback for a message"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE messages SET feedback = ? WHERE id = ?', (feedback, message_id))
    conn.commit()
    conn.close()

def get_message_by_id(message_id: int) -> Optional[Dict]:
    """Get a message by its ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, conversation_id, role, content, timestamp, feedback
                 FROM messages WHERE id = ?''', (message_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {
            'id': row[0], 'conversation_id': row[1], 'role': row[2],
            'content': row[3], 'timestamp': row[4], 'feedback': row[5]
        }
    return None

# ==================== FastAPI App ====================

app = FastAPI(title="AI Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pathlib
_base_dir = pathlib.Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_base_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(_base_dir / "static"))

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an advanced AI assistant with deep reasoning capabilities. You must think through problems thoroughly and validate your reasoning internally, but only present final, polished conclusions to the user.

CRITICAL REASONING PROTOCOL - INTERNAL REASONING, POLISHED OUTPUT:

**1. INTERNAL REASONING PROCESS:**
   - Think through problems step-by-step internally
   - Validate each logical step before proceeding
   - Question your own assumptions
   - Consider counter-arguments before concluding
   - Verify calculations and logic chains

   BUT: Do NOT show this reasoning process to the user. Only present final, validated conclusions.

**2. OUTPUT QUALITY STANDARDS:**
   - Only output statements that have passed your internal validation
   - Present conclusions clearly and directly, without reasoning brackets
   - Do not use phrases like "[Reasoning: ...]", "[Conclusion: ...]", "[Statement: ...]", "[Analysis: ...]", etc.
   - Write naturally as if you've already completed the reasoning
   - Be confident in your answers because they've been validated internally

**3. FOR DIFFERENT TASK TYPES:**
   - **Math/Logic**: Calculate internally, verify, then present the answer with explanation
   - **Code**: Think through algorithm choice, test internally, then present clean, working code
   - **Analysis**: Consider multiple perspectives internally, then present a balanced conclusion
   - **Creative**: Explore ideas internally, evaluate options, then present refined solutions

**4. THINKING STYLE:**
   - Be methodical: reason internally -> validate -> present polished conclusion
   - Be thorough: don't skip validation steps internally
   - Be self-critical: question your own reasoning before presenting
   - Be clear: write naturally without reasoning brackets or meta-commentary
   - Be confident: present validated conclusions directly

Remember: Do your reasoning internally for quality, but present only polished, final conclusions to the user."""

def filter_reasoning_text(text):
    """Remove reasoning brackets and meta-commentary from text"""
    if not text:
        return text
    text = re.sub(r'\[Reasoning:[^\]]+\]\s*->\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[Reasoning:[^\]]+\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\[(?:Conclusion|Statement|Analysis|Evaluation|Decision|Verification|Step \d+|Fact|Claim|Acknowledged uncertainty|Verified|Test Result):[^\]]+\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\s*->\s*', ' ', text)
    text = re.sub(r'\[[A-Z][a-z]+(?::[^\]]+)?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

# ==================== API Endpoints ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "themes_json": json.dumps({"light": COLORS_LIGHT, "dark": COLORS_DARK}),
        })
    except Exception as e:
        logger.error(f"Error generating home page: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error loading page: {str(e)}")

@app.get("/api/conversations")
async def get_conversations():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT c.id, c.title, c.created_at, c.updated_at,
                     m.content, m.timestamp
                     FROM conversations c
                     LEFT JOIN (
                         SELECT conversation_id, content, timestamp,
                                ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY timestamp DESC) as rn
                         FROM messages
                     ) m ON c.id = m.conversation_id AND m.rn = 1
                     ORDER BY c.updated_at DESC''')

        conversations = []
        for row in c.fetchall():
            conversations.append({
                'id': row[0], 'title': row[1], 'created_at': row[2],
                'updated_at': row[3],
                'last_message': row[4] if row[4] else '',
                'last_message_timestamp': row[5] if row[5] else row[3]
            })

        conn.close()
        return {'conversations': conversations}
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return {'conversations': []}

@app.post("/api/message/{message_id}/feedback")
async def submit_feedback(message_id: int, request: dict):
    feedback = request.get('feedback', '').strip()
    if feedback not in ['positive', 'negative']:
        raise HTTPException(status_code=400, detail="Feedback must be 'positive' or 'negative'")

    try:
        update_message_feedback(message_id, feedback)
        return {"status": "success", "message": f"Feedback recorded: {feedback}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/message/{message_id}/retry")
async def retry_message(message_id: int, request: dict):
    try:
        message = get_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        if message['role'] != 'assistant':
            raise HTTPException(status_code=400, detail="Can only retry assistant messages")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT id, content FROM messages
                     WHERE conversation_id = ? AND role = 'user' AND id < ?
                     ORDER BY id DESC LIMIT 1''',
                  (message['conversation_id'], message_id))
        user_row = c.fetchone()
        conn.close()

        if not user_row:
            raise HTTPException(status_code=404, detail="Original user message not found")

        return {
            "status": "ready",
            "conversation_id": message['conversation_id'],
            "prompt": user_row[1],
            "message": "Ready to retry. Send this prompt again."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        history = get_conversation_history(conversation_id, limit=100, current_query=None)
        return {'messages': history}
    except Exception as e:
        return {'messages': []}

@app.post("/api/conversation/{conversation_id}/rename")
async def rename_conversation(conversation_id: str, request: RenameRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('UPDATE conversations SET title = ? WHERE id = ?',
                  (request.title, conversation_id))
        conn.commit()
        conn.close()
        return {'status': 'success'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        conn.commit()
        conn.close()
        return {'status': 'success'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def search_chats(query: str):
    try:
        results = search_messages(query, limit=50)
        return {'results': results, 'count': len(results)}
    except Exception as e:
        return {'results': [], 'count': 0}

@app.post("/api/web-search")
async def web_search(request: dict):
    query = request.get('query', '').strip()
    sources = request.get('sources', ['google', 'wikipedia', 'reddit'])
    max_results = min(request.get('max_results', 10), 20)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    results = []
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))

            for result in search_results:
                url = result.get('href', '')
                if 'wikipedia.org' in url.lower():
                    source_type = 'wikipedia'
                elif 'reddit.com' in url.lower():
                    source_type = 'reddit'
                elif 'github.com' in url.lower():
                    source_type = 'github'
                elif any(x in url.lower() for x in ['stackoverflow.com', 'stackexchange.com']):
                    source_type = 'stackoverflow'
                else:
                    source_type = 'google'

                if source_type in sources or (source_type == 'general' and 'google' in sources):
                    results.append({
                        'title': result.get('title', ''), 'url': url,
                        'snippet': result.get('body', ''), 'source': source_type
                    })

        return {'results': results, 'count': len(results), 'query': query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/execute-code")
async def execute_code(request: dict):
    code = request.get('code', '').strip()
    language = request.get('language', 'python').lower()

    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    if len(code) > 10000:
        raise HTTPException(status_code=400, detail="Code too long (max 10KB)")

    try:
        if language == 'python':
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                process = await asyncio.create_subprocess_exec(
                    'python3', temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, 'PYTHONPATH': ''}
                )

                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
                    return {
                        'success': process.returncode == 0,
                        'stdout': stdout.decode('utf-8', errors='replace'),
                        'stderr': stderr.decode('utf-8', errors='replace'),
                        'returncode': process.returncode
                    }
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return {'success': False, 'error': 'Code execution timed out (5 second limit)'}
            finally:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        else:
            return {'success': False, 'error': f'Unsupported language: {language}. Use Python.'}
    except Exception as e:
        return {'success': False, 'error': f'Execution error: {str(e)}'}

@app.get("/api/files/list")
async def list_files(path: str = "."):
    try:
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(os.path.abspath('.')):
            raise HTTPException(status_code=403, detail="Access denied")

        items = []
        if os.path.isdir(abs_path):
            for item in sorted(os.listdir(abs_path)):
                item_path = os.path.join(abs_path, item)
                try:
                    items.append({
                        'name': item,
                        'path': os.path.relpath(item_path, os.path.abspath('.')),
                        'type': 'directory' if os.path.isdir(item_path) else 'file',
                        'size': os.path.getsize(item_path) if os.path.isfile(item_path) else None
                    })
                except (OSError, PermissionError):
                    continue

        return {'path': path, 'items': items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/read")
async def read_file_content(path: str):
    try:
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(os.path.abspath('.')):
            raise HTTPException(status_code=403, detail="Access denied")
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="File not found")
        if os.path.getsize(abs_path) > 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 1MB)")

        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        return {'path': path, 'content': content, 'size': len(content), 'lines': content.count('\n') + 1}
    except (HTTPException):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WebSocket Handlers ====================

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            data = await websocket.receive_json()

            message = data.get('message', '').strip()
            conv_id = data.get('conversation_id')

            if not message:
                await websocket.send_json({'type': 'error', 'content': 'Empty message'})
                continue

            custom_system_prompt = data.get('system_prompt')

            try:
                # Save user message
                save_message(conv_id, 'user', message)

                # Get history
                history = get_conversation_history(conv_id, current_query=message)

                # Build messages
                system_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
                messages = [{'role': 'system', 'content': system_prompt}]

                for msg in history:
                    messages.append({'role': msg['role'], 'content': msg['content']})

                # Auto web search for queries that might need current info
                search_keywords = ['current', 'recent', 'latest', 'today', 'now', '2024', '2025', '2026', 'news', 'update', 'happening']
                query_lower = message.lower()
                if any(kw in query_lower for kw in search_keywords):
                    try:
                        await websocket.send_json({'type': 'token', 'content': 'Searching for up-to-date information...\n\n'})
                        with DDGS() as ddgs:
                            search_results = list(ddgs.text(message, max_results=3))
                            if search_results:
                                search_context = "\n\nWeb search results:\n"
                                for i, r in enumerate(search_results, 1):
                                    search_context += f"\n[{i}] {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')[:300]}\n"
                                messages.append({
                                    'role': 'system',
                                    'content': f'Web search results for the user\'s query:\n{search_context}\n\nUse this information to provide accurate responses.'
                                })
                    except Exception as e:
                        logger.warning(f"Web search failed: {e}")

                messages.append({'role': 'user', 'content': message})

                # Signal start
                await websocket.send_json({'type': 'start'})

                # Calculate max tokens
                max_tokens = estimate_tokens_needed(message)
                logger.info(f"Processing message for conv {conv_id}, max_tokens: {max_tokens}")

                # Stream from vLLM
                full_response = ""
                stream_buffer = ""
                sent_length = 0
                buffer_size = 300

                async for token in vllm_chat_stream(messages, max_tokens=max_tokens):
                    full_response += token
                    stream_buffer += token

                    if len(stream_buffer) > buffer_size:
                        trim_amount = len(stream_buffer) - buffer_size
                        stream_buffer = stream_buffer[trim_amount:]
                        sent_length = max(0, sent_length - trim_amount)

                    filtered_buffer = filter_reasoning_text(stream_buffer)
                    new_content = filtered_buffer[sent_length:]
                    if new_content:
                        await websocket.send_json({'type': 'token', 'content': new_content})
                        sent_length = len(filtered_buffer)

                # Filter full response
                full_response = filter_reasoning_text(full_response)

                # Save and send completion
                assistant_message_id = save_message(conv_id, 'assistant', full_response)
                await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
                await websocket.send_json({'type': 'done'})

            except httpx.HTTPStatusError as e:
                logger.error(f"vLLM API error: {e}")
                await websocket.send_json({'type': 'error', 'content': f'vLLM error: {e.response.status_code}. Is the model loaded?'})
            except httpx.ConnectError:
                await websocket.send_json({'type': 'error', 'content': 'Cannot connect to vLLM. Is it running?'})
            except Exception as e:
                logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
                await websocket.send_json({'type': 'error', 'content': f'Error: {str(e)}'})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.websocket("/ws/logs")
async def logs_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        existing_logs = log_capture.getvalue()
        if existing_logs:
            await websocket.send_json({'type': 'log', 'content': existing_logs})

        last_size = len(existing_logs)

        while True:
            await asyncio.sleep(0.5)
            current_logs = log_capture.getvalue()
            if len(current_logs) > last_size:
                await websocket.send_json({'type': 'log', 'content': current_logs[last_size:]})
                last_size = len(current_logs)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Log WebSocket error: {e}")

@app.get("/health")
async def health():
    model_ok = await vllm_health_check()
    return {
        "status": "healthy" if model_ok else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "model_available": model_ok,
        "message": f"Model {MODEL_NAME} is ready" if model_ok else "Model is loading or unavailable"
    }

# ==================== Dashboard ====================

def get_cache_info():
    """Get HuggingFace model cache info"""
    model_dir = MODEL_NAME.replace("/", "--")
    cache_dir = os.path.join(HF_CACHE_PATH, "hub", f"models--{model_dir}")

    if not os.path.exists(cache_dir):
        return {"status": "not_downloaded", "size_bytes": 0, "size_display": "0 B", "last_modified": None}

    total_size = 0
    file_count = 0
    for dirpath, _, filenames in os.walk(cache_dir):
        for f in filenames:
            try:
                total_size += os.path.getsize(os.path.join(dirpath, f))
                file_count += 1
            except OSError:
                pass

    snapshots_dir = os.path.join(cache_dir, "snapshots")
    has_snapshots = os.path.exists(snapshots_dir) and bool(os.listdir(snapshots_dir))

    last_modified = None
    try:
        last_modified = datetime.fromtimestamp(os.path.getmtime(cache_dir)).isoformat()
    except Exception:
        pass

    if total_size > 1024**3:
        size_display = f"{total_size / (1024**3):.1f} GB"
    elif total_size > 1024**2:
        size_display = f"{total_size / (1024**2):.1f} MB"
    else:
        size_display = f"{total_size / 1024:.1f} KB"

    return {
        "status": "valid" if has_snapshots else "incomplete",
        "size_bytes": total_size,
        "size_display": size_display,
        "file_count": file_count,
        "last_modified": last_modified,
    }

@app.get("/api/dashboard")
async def get_dashboard():
    model_ok = await vllm_health_check()

    # Get vLLM container status via Docker socket
    container_status = None
    try:
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("http://localhost/containers/vllm/json")
            if resp.status_code == 200:
                info = resp.json()
                container_status = {
                    "status": info.get("State", {}).get("Status"),
                    "running": info.get("State", {}).get("Running"),
                    "started_at": info.get("State", {}).get("StartedAt"),
                }
    except Exception:
        pass

    cache_info = get_cache_info()

    return {
        "model_name": MODEL_NAME,
        "vllm_host": VLLM_HOST,
        "model_available": model_ok,
        "container": container_status,
        "cache": cache_info,
    }

@app.post("/api/vllm/restart")
async def restart_vllm():
    try:
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.post("http://localhost/containers/vllm/restart", params={"t": 10})
            if resp.status_code == 204:
                return {"status": "success", "message": "vLLM is restarting. Model will reload in a few minutes."}
            else:
                return {"status": "error", "message": f"Restart failed (HTTP {resp.status_code})"}
    except Exception as e:
        logger.error(f"Failed to restart vLLM: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/model/redownload")
async def redownload_model():
    try:
        # Stop vLLM first
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post("http://localhost/containers/vllm/stop", params={"t": 10})

        # Delete model cache
        model_dir = MODEL_NAME.replace("/", "--")
        cache_dir = os.path.join(HF_CACHE_PATH, "hub", f"models--{model_dir}")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"Deleted model cache: {cache_dir}")

        # Start vLLM (will re-download on boot)
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post("http://localhost/containers/vllm/start")

        return {"status": "success", "message": "Cache cleared. vLLM is restarting and will re-download the model."}
    except Exception as e:
        logger.error(f"Failed to redownload model: {e}")
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("AI Chat Application Starting...")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"vLLM: {VLLM_HOST}")
    logger.info("=" * 60)

    # Check model availability in background
    async def wait_for_model():
        for i in range(120):  # Wait up to 10 minutes
            if await vllm_health_check():
                logger.info(f"Model {MODEL_NAME} is ready!")
                return
            if i % 6 == 0:
                logger.info(f"Waiting for model to load... ({i*5}s)")
            await asyncio.sleep(5)
        logger.warning("Model did not become available within 10 minutes")

    asyncio.create_task(wait_for_model())

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting AI Chat with vLLM ({MODEL_NAME})...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
