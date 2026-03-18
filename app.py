#!/usr/bin/env python3
"""
AI Chat with vLLM - Simplified single-model setup
- Uses vLLM with Apriel 1.6 15B via httpx (no OpenAI SDK)
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
MODEL_NAME = os.getenv("MODEL_NAME", "ServiceNow-AI/Apriel-1.6-15b-Thinker")
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
        return 30000

    estimated = len(prompt) * 0.25 * 10
    return min(max(int(estimated), 4096), 30000)

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
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Be concise and direct. Match the complexity of your response to the complexity of the question — simple questions get short answers, complex questions get thorough answers.

Never show your internal reasoning, thinking process, drafts, or self-corrections. Just give the final answer."""

SEARCH_ADDENDUM = """
---
WEB SEARCH RESULTS (use these to answer the user — cite sources with URLs):
{results}

Instructions for using search results:
- Use the provided data to answer the question accurately with specific facts, numbers, dates, and details
- Cite sources inline using [Source Name](URL) markdown links
- Synthesize information from multiple results when relevant
- Do NOT say "according to search results" — just present the information naturally"""

def filter_reasoning_text(text):
    """Remove <think> blocks, reasoning brackets, and meta-commentary from text"""
    if not text:
        return text
    # Strip model special tokens (<|eot_id|>, <|im_end|>, etc.)
    text = re.sub(r'<\|[^|]*\|>', '', text)
    # Strip <think>...</think> blocks (thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip unclosed <think> blocks (model cut off mid-thought)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    # Strip text-based reasoning blocks (e.g., "Thinking Process: ...")
    _reasoning_headers = ['thinking process', 'thought process', 'internal reasoning',
                          'my reasoning', 'wait,', 'wait -', 'let me think',
                          'let me check', 'let me verify', 'let me consider',
                          'hmm,', 'okay so', 'okay let', 'decision:',
                          'correction:', 'drafting', 'first, let me',
                          'i need to think', 'i need to consider']
    _first_line = text.lstrip().split('\n')[0].lower().replace('*', '').strip().rstrip(':')
    if any(_first_line.startswith(h) for h in _reasoning_headers):
        paragraphs = re.split(r'\n\s*\n', text)
        result_parts = []
        found_content = False
        for i, para in enumerate(paragraphs):
            if found_content:
                result_parts.append(para)
                continue
            if i == 0:
                continue
            stripped = para.strip()
            if not stripped:
                continue
            first_para_line = stripped.split('\n')[0]
            looks_like_reasoning = bool(re.match(
                r'^\s*(?:\d+[\.\):]|\*\*[A-Z]|\*\s|[-•]|#{1,3}\s)',
                first_para_line
            )) or bool(re.match(
                r'^\s*(?:Wait|Hmm|Actually,|Let me |Re-evaluat|Decision:|Safest|Correction:|Draft:)',
                first_para_line, re.IGNORECASE
            ))
            if not looks_like_reasoning and len(stripped) > 20:
                found_content = True
                result_parts.append(para)
        text = '\n\n'.join(result_parts) if result_parts else ""
    text = re.sub(r'\[Reasoning:[^\]]+\]\s*->\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[Reasoning:[^\]]+\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\[(?:Conclusion|Statement|Analysis|Evaluation|Decision|Verification|Step \d+|Fact|Claim|Acknowledged uncertainty|Verified|Test Result):[^\]]+\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\s*->\s*', ' ', text)
    text = re.sub(r'\[[A-Z][a-z]+(?::[^\]]+)?\]', '', text)
    text = re.sub(r'[^\S\n]+', ' ', text)  # Collapse horizontal whitespace only, preserve newlines
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
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
            "model_name": MODEL_NAME,
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

# ==================== Web Search Logic ====================

def should_web_search(message: str) -> bool:
    """Decide if a query would benefit from web search results."""
    msg = message.lower().strip()

    # Explicit search requests — always honour, no length/greeting gate
    search_triggers = ['search online', 'search the web', 'search the internet',
                       'look up', 'look online', 'google ', 'find online',
                       'web search', 'browse for', 'search for ',
                       'look it up', 'look this up', 'search this',
                       'search that', 'look that up', 'look them up']
    if any(s in msg for s in search_triggers):
        return True
    # "search <something>" at the start of the message (e.g. "search ebay for legos")
    if msg.startswith('search '):
        return True

    # Too short to need search (only checked AFTER explicit triggers)
    if len(msg) < 15:
        return False

    # Skip greetings, thanks, simple instructions
    skip_starts = ['hello', 'hi ', 'hey', 'thanks', 'thank you', 'bye', 'please ', 'ok', 'sure', 'yes', 'no']
    if any(msg.startswith(s) for s in skip_starts) and len(msg) < 40:
        return False

    # Skip coding/math/creative tasks — these don't need web search
    code_signals = [
        'write a function', 'write code', 'write a script', 'fix this', 'debug',
        'refactor', 'implement', 'def ', 'class ', 'function(', 'calculate',
        'solve', 'write a poem', 'write a story', 'explain this code',
        'convert this', 'translate this code', 'optimize this', 'review this',
        '```', 'import ', 'print(', 'return ', 'for i in', 'console.log',
    ]
    if any(sig in msg for sig in code_signals):
        return False

    # Shopping / product / buying queries
    shopping_patterns = [
        r'\b(buy|purchase|shop for|shopping for|order)\b',
        r'\b(find me|get me|show me|recommend)\b.{3,}',
        r'\$\d+',                                       # dollar amounts
        r'\b\d+\s*dollars\b',
        r'\b(cheap|affordable|budget|expensive|price|pricing|cost of)\b',
        r'\b(best|top|good)\b.{1,30}\b(under|around|for|near)\b',
        r'\b(where (can|do|to) (i |we )?(buy|get|find|order))\b',
        r'\b(in stock|available|for sale|deal on|deals on|discount)\b',
        r'\b(review|reviews|rating|ratings|comparison)\b.{1,20}\b(of|for|on)\b',
    ]
    if any(re.search(pat, msg) for pat in shopping_patterns):
        return True

    # Temporal markers — strong signal for needing current info
    temporal = [
        'today', 'yesterday', 'this week', 'this month', 'this year',
        'right now', 'currently', 'latest', 'recent', 'newest',
        'last week', 'last month', 'last year', 'upcoming', 'schedule',
    ]
    # Also match year references 2024+
    if any(t in msg for t in temporal) or re.search(r'\b20(2[4-9]|[3-9]\d)\b', msg):
        return True

    # Current events / real-world lookups
    current_signals = [
        'news', 'election', 'stock', 'price of', 'weather', 'score',
        'release date', 'announced', 'launched', 'worth', 'net worth',
        'ceo of', 'president of', 'population of', 'capital of',
        'happening', 'update on', 'status of', 'reviews of',
    ]
    if any(sig in msg for sig in current_signals):
        return True

    # Factual lookup patterns
    lookup_patterns = [
        r'\b(who|what|when|where|how) (is|was|are|were|do|does|did|to|can|should) (the |a |i |you )?\w+.{5,}',
        r'how (much|many) (does|do|did|is|are|will)',
        r'(latest|current|recent|new) .{3,} (version|release|update|price|status|news)',
        r'(compare|difference between|vs\.?|versus) .+ (and|vs)',
        r'(best|top|recommended) .{3,} (for|in|of) \d{4}',
        r'\b(why) (is|was|are|were|does|do|did|can|would|should) .{10,}',
    ]
    if any(re.search(pat, msg) for pat in lookup_patterns):
        return True

    return False

def clean_search_query(message: str, history: list = None) -> str:
    """Strip conversational fluff to produce a better search engine query.

    If the cleaned query is too vague (pronouns / short references like "one",
    "it", "that"), fall back to the last assistant reply to build context.
    """
    q = message.strip()
    # Remove common prefixes that confuse search engines
    prefixes = [
        r'^(can you |could you |please |hey,? |hi,? )',
        r'^(search online for |search the web for |search the internet for )',
        r'^(search for |look up |look online for |google |find online )',
        r'^(web search |browse for |find me |find )',
        r'^(search online |search this|search that|look it up|look this up|look that up|look them up)',
        r'^(tell me about |what is |what are |who is |where can i )',
    ]
    lower = q.lower()
    for pat in prefixes:
        m = re.match(pat, lower)
        if m:
            q = q[m.end():]
            lower = q.lower()
    q = q.strip() or message.strip()

    # If the cleaned query is just a pronoun / too vague, use conversation context
    vague_tokens = {'one', 'it', 'that', 'this', 'them', 'those', 'these', 'some', 'any'}
    words = set(q.lower().split())
    if words and words.issubset(vague_tokens | {'a', 'the', 'an', 'for', 'me', 'please'}):
        if history:
            # Walk backwards to find the last user message (the one before this)
            for msg in reversed(history):
                if msg.get('role') == 'user':
                    prev = msg['content'].strip()
                    # Use the previous user query as context
                    return clean_search_query(prev)
        # If no usable history, return the original message
        return message.strip()

    return q

# ==================== Enhanced Web Search with Content Extraction ====================

async def fetch_page_content(url: str, max_chars: int = 1500) -> str:
    """Fetch a URL and extract clean text content using BeautifulSoup.

    Uses source-specific extractors for Wikipedia, Reddit, GitHub,
    StackOverflow, and falls back to generic main-content extraction.
    """
    try:
        # Wikipedia: use their REST API for clean structured content
        wiki_match = re.match(r'https?://(\w+)\.wikipedia\.org/wiki/(.+)', url)
        if wiki_match:
            lang, title = wiki_match.group(1), wiki_match.group(2)
            api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(api_url, headers={'Accept': 'application/json'})
                if resp.status_code == 200:
                    data = resp.json()
                    extract = data.get('extract', '')
                    desc = data.get('description', '')
                    result = f"{desc}\n\n{extract}" if desc else extract
                    return result[:max_chars] if result else ""

        # Reddit: rewrite to old.reddit.com (new Reddit is JS-rendered)
        fetch_url = url
        if 'reddit.com' in url and 'old.reddit.com' not in url:
            fetch_url = url.replace('www.reddit.com', 'old.reddit.com').replace('://reddit.com', '://old.reddit.com')

        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
        }
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(fetch_url)
            if resp.status_code != 200:
                return ""
            content_type = resp.headers.get('content-type', '')
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                return ""

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Strip non-content tags
            for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                             'aside', 'form', 'iframe', 'noscript', 'svg', 'button']):
                tag.decompose()

            # --- Reddit ---
            if 'reddit.com' in url:
                parts = []
                # Post title
                title_el = soup.find('a', class_='title')
                if title_el:
                    parts.append(title_el.get_text(strip=True))
                # Self-text (post body)
                selftext = soup.find('div', class_='usertext-body')
                if selftext:
                    parts.append(selftext.get_text(separator='\n', strip=True)[:600])
                # Top comments
                comments = soup.find_all('div', class_='usertext-body')
                for c in comments[1:6]:  # skip first (post body), get top 5 comments
                    ct = c.get_text(separator=' ', strip=True)[:250]
                    if ct and len(ct) > 30:
                        parts.append(f"Comment: {ct}")
                if parts:
                    return '\n---\n'.join(parts)[:max_chars]

            # --- GitHub ---
            if 'github.com' in url:
                # Try README article
                readme = soup.find('article', class_='markdown-body')
                if readme:
                    return readme.get_text(separator='\n', strip=True)[:max_chars]
                # Repo about/description
                about = soup.find('p', class_='f4') or soup.find('p', {'itemprop': 'description'})
                if about:
                    desc = about.get_text(strip=True)
                    # Also grab topic tags
                    topics = soup.find_all('a', class_='topic-tag')
                    tags = ', '.join(t.get_text(strip=True) for t in topics[:10])
                    result = desc
                    if tags:
                        result += f"\nTopics: {tags}"
                    return result[:max_chars]

            # --- StackOverflow / StackExchange ---
            if 'stackoverflow.com' in url or 'stackexchange.com' in url:
                parts = []
                # Question
                q_div = soup.find('div', class_='s-prose')
                if q_div:
                    parts.append("Q: " + q_div.get_text(separator=' ', strip=True)[:500])
                # Answers (accepted first, then top-voted)
                answer_divs = soup.find_all('div', class_='s-prose')
                for a in answer_divs[1:3]:  # top 2 answers
                    parts.append("A: " + a.get_text(separator=' ', strip=True)[:500])
                if parts:
                    return '\n---\n'.join(parts)[:max_chars]

            # --- Generic extraction ---
            main = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup.find('body')
            if main:
                text = main.get_text(separator='\n', strip=True)
                lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 20]
                return '\n'.join(lines)[:max_chars]

            return ""
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return ""


def _source_priority(url: str) -> int:
    """Score a URL by source quality for content-fetch prioritization."""
    url_lower = url.lower()
    for domain, score in [
        ('wikipedia.org', 10), ('stackoverflow.com', 9), ('stackexchange.com', 8),
        ('github.com', 7), ('reddit.com', 7), ('arxiv.org', 7),
        ('.gov', 6), ('.edu', 6),
    ]:
        if domain in url_lower:
            return score
    return 3


async def enhanced_web_search(query: str, original_message: str) -> str:
    """Multi-source web search with page content extraction.

    1. Runs DDG text search (+ news search for temporal queries) in parallel
    2. Prioritizes high-value sources (Wikipedia, SO, GitHub, Reddit)
    3. Fetches full page content from top 3 URLs in parallel
    4. Formats everything within a ~6000-char budget for the context window
    """
    msg_lower = original_message.lower()
    is_news = any(w in msg_lower for w in [
        'news', 'latest', 'recent', 'announced', 'launched', 'update',
        'happening', 'today', 'yesterday', 'this week',
    ])

    # --- DDG searches in parallel ---
    def _ddg_text(q):
        with DDGS() as ddgs:
            return list(ddgs.text(q, region='us-en', max_results=8))

    def _ddg_news(q):
        try:
            with DDGS() as ddgs:
                return list(ddgs.news(q, region='us-en', max_results=5))
        except Exception:
            return []

    tasks = [asyncio.to_thread(_ddg_text, query)]
    if is_news:
        tasks.append(asyncio.to_thread(_ddg_news, query))

    search_outputs = await asyncio.gather(*tasks, return_exceptions=True)

    text_results = search_outputs[0] if not isinstance(search_outputs[0], Exception) else []
    news_results = []
    if is_news and len(search_outputs) > 1:
        news_results = search_outputs[1] if not isinstance(search_outputs[1], Exception) else []

    if not text_results and not news_results:
        return ""

    # --- Prioritize and pick top 3 unique domains to fetch content from ---
    text_results.sort(key=lambda r: _source_priority(r.get('href', '')), reverse=True)

    fetch_urls = []
    seen_domains = set()
    for r in text_results:
        url = r.get('href', '')
        if not url:
            continue
        m = re.search(r'://([^/]+)', url)
        base = '.'.join(m.group(1).split('.')[-2:]) if m else url
        if base not in seen_domains:
            seen_domains.add(base)
            fetch_urls.append(url)
        if len(fetch_urls) >= 3:
            break

    # --- Fetch page content in parallel ---
    page_contents = {}
    if fetch_urls:
        fetch_results = await asyncio.gather(
            *[fetch_page_content(u) for u in fetch_urls],
            return_exceptions=True,
        )
        for url, content in zip(fetch_urls, fetch_results):
            if isinstance(content, str) and content.strip():
                page_contents[url] = content

    # --- Format results within budget ---
    BUDGET = 6000  # chars — leaves room for system prompt + history + response tokens
    used = 0
    parts = []

    # News results first (most timely)
    if news_results:
        parts.append("RECENT NEWS:")
        for n in news_results[:4]:
            entry = f"- {n.get('title', '')} ({n.get('date', '')})\n  {n.get('body', '')[:200]}\n  {n.get('url', '')}"
            if used + len(entry) > BUDGET:
                break
            parts.append(entry)
            used += len(entry)
        parts.append("")

    # Text results with fetched content where available
    parts.append("SEARCH RESULTS:")
    for i, r in enumerate(text_results, 1):
        title = r.get('title', '')
        url = r.get('href', '')
        snippet = r.get('body', '')[:300]
        fetched = page_contents.get(url, '')

        if fetched:
            entry = f"\n[{i}] {title}\nURL: {url}\n{fetched}"
        else:
            entry = f"\n[{i}] {title}\nURL: {url}\n{snippet}"

        if used + len(entry) > BUDGET:
            remaining = BUDGET - used
            if remaining > 150:
                entry = entry[:remaining]
                parts.append(entry)
                used += len(entry)
            break

        parts.append(entry)
        used += len(entry)

    return '\n'.join(parts)


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

            # ── Slash commands ──────────────────────────────────────
            if message.startswith('/'):
                cmd_parts = message.split(None, 1)
                cmd = cmd_parts[0].lower()
                cmd_arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ''
                handled = True

                if cmd == '/help':
                    help_text = (
                        "**Available commands:**\n"
                        "| Command | Description |\n"
                        "|---------|-------------|\n"
                        "| `/help` | Show this help |\n"
                        "| `/search <query>` | Force a web search |\n"
                        "| `/think <message>` | Force extended thinking |\n"
                        "| `/system <prompt>` | Set custom system prompt |\n"
                        "| `/reset` | Reset system prompt to default |\n"
                        "| `/model` | Show current model info |\n"
                        "| `/temp <0.0–1.0>` | Set temperature for next message |\n"
                        "| `/clear` | Clear conversation history |"
                    )
                    await websocket.send_json({'type': 'start'})
                    await websocket.send_json({'type': 'token', 'content': help_text})
                    await websocket.send_json({'type': 'done'})

                elif cmd == '/model':
                    healthy = await vllm_health_check()
                    status = "online" if healthy else "offline"
                    await websocket.send_json({'type': 'start'})
                    await websocket.send_json({'type': 'token', 'content': f'**Model:** `{MODEL_NAME}`\n**Status:** {status}'})
                    await websocket.send_json({'type': 'done'})

                elif cmd == '/system':
                    if not cmd_arg:
                        current = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
                        await websocket.send_json({'type': 'start'})
                        await websocket.send_json({'type': 'token', 'content': f'**Current system prompt:**\n\n{current}'})
                        await websocket.send_json({'type': 'done'})
                    else:
                        await websocket.send_json({'type': 'set_system_prompt', 'content': cmd_arg})
                        await websocket.send_json({'type': 'start'})
                        await websocket.send_json({'type': 'token', 'content': f'System prompt updated.'})
                        await websocket.send_json({'type': 'done'})

                elif cmd == '/reset':
                    await websocket.send_json({'type': 'set_system_prompt', 'content': ''})
                    await websocket.send_json({'type': 'start'})
                    await websocket.send_json({'type': 'token', 'content': 'System prompt reset to default.'})
                    await websocket.send_json({'type': 'done'})

                elif cmd == '/clear':
                    if conv_id:
                        conn = sqlite3.connect(DB_PATH)
                        conn.execute('DELETE FROM messages WHERE conversation_id = ?', (conv_id,))
                        conn.commit()
                        conn.close()
                    await websocket.send_json({'type': 'start'})
                    await websocket.send_json({'type': 'token', 'content': 'Conversation history cleared.'})
                    await websocket.send_json({'type': 'done'})

                elif cmd == '/temp':
                    try:
                        val = float(cmd_arg) if cmd_arg else -1
                        if not (0.0 <= val <= 2.0):
                            raise ValueError
                        await websocket.send_json({'type': 'set_temperature', 'value': val})
                        await websocket.send_json({'type': 'start'})
                        await websocket.send_json({'type': 'token', 'content': f'Temperature set to **{val}**.'})
                        await websocket.send_json({'type': 'done'})
                    except ValueError:
                        await websocket.send_json({'type': 'error', 'content': 'Usage: `/temp 0.7` (range 0.0–2.0)'})

                elif cmd == '/search':
                    if not cmd_arg:
                        await websocket.send_json({'type': 'error', 'content': 'Usage: `/search <query>`'})
                    else:
                        # Force search then send to model — fall through to normal flow
                        message = cmd_arg
                        data['_force_search'] = True
                        handled = False

                elif cmd == '/think':
                    if not cmd_arg:
                        await websocket.send_json({'type': 'error', 'content': 'Usage: `/think <message>`'})
                    else:
                        message = cmd_arg
                        data['_force_think'] = True
                        handled = False

                else:
                    await websocket.send_json({'type': 'error', 'content': f'Unknown command: `{cmd}`. Type `/help` for available commands.'})

                if handled:
                    continue

            try:
                # Save user message
                save_message(conv_id, 'user', message)

                # Get history
                history = get_conversation_history(conv_id, current_query=message)

                # Build messages
                system_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
                # Thinking mode: /think forces it on, simple queries turn it off
                force_think = data.get('_force_think', False)
                msg_lower = message.lower().strip()
                is_simple = not force_think and (
                    len(message.split()) < 10 or any(
                        msg_lower.startswith(s) for s in ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'ok', 'yes', 'no']
                    )
                )
                if is_simple:
                    system_prompt = "/no_think\n" + system_prompt
                messages = [{'role': 'system', 'content': system_prompt}]

                for msg in history:
                    messages.append({'role': msg['role'], 'content': msg['content']})

                # Auto web search when the query likely needs current/factual info
                do_search = data.get('_force_search', False) or should_web_search(message)
                if do_search:
                    # Force no-think mode — the model just needs to summarise results
                    if not system_prompt.startswith('/no_think'):
                        system_prompt = "/no_think\n" + system_prompt
                        messages[0] = {'role': 'system', 'content': system_prompt}
                    try:
                        search_query = clean_search_query(message, history=history)
                        await websocket.send_json({'type': 'token', 'content': 'Searching the web...\n\n'})
                        search_context = await enhanced_web_search(search_query, message)
                        if search_context:
                            messages[0]['content'] += SEARCH_ADDENDUM.format(results=search_context)
                            logger.info(f"Enhanced search injected ({len(search_context)} chars) for query: {search_query}")
                        else:
                            logger.warning(f"Search returned no results for query: {search_query}")
                    except Exception as e:
                        logger.warning(f"Web search failed for query '{search_query}': {e}", exc_info=True)

                # Signal start
                await websocket.send_json({'type': 'start'})

                # Calculate max tokens
                max_tokens = estimate_tokens_needed(message)
                logger.info(f"Processing message for conv {conv_id}, max_tokens: {max_tokens}")

                # Stream from vLLM — filter <think> blocks and text-based reasoning before sending to user
                full_response = ""
                in_think_block = False
                think_buffer = ""
                # Text-based reasoning detection (e.g., "Thinking Process: ...")
                reasoning_detected = False
                reasoning_check_done = False
                REASONING_PREFIXES = ['thinking process', 'thought process', 'internal reasoning',
                                      'my reasoning', 'reasoning:', 'wait,', 'wait -',
                                      'let me think', 'let me check', 'let me verify',
                                      'let me consider', 'let me re-read',
                                      'i need to think', 'i need to consider',
                                      'hmm,', 'okay so', 'okay let',
                                      'decision:', 'correction:', 'drafting',
                                      'final plan:', 'first, let me']
                # Broader reasoning indicators (regex fallback for first ~50 chars)
                _REASONING_INDICATORS = re.compile(
                    r'(?:re-reading|system (?:instruction|prompt)|'
                    r'I (?:must|should|need to|will|cannot|can\'t|don\'t) |'
                    r'drafting|refining|final plan|fact[\s-]?check|'
                    r'let me (?:recall|verify|think|check|re-read)|'
                    r'wait.*(?:re-read|check|one more|actually)|'
                    r'this (?:means|implies|suggests)|'
                    r'looking at (?:the|this)|'
                    r'acknowledge|limitation|browsing capabilit)',
                    re.IGNORECASE
                )
                ws_disconnected = False
                next_rep_check = 300  # first repetition check at 300 chars

                temperature = data.get('temperature', 0.3)
                stream_gen = vllm_chat_stream(messages, max_tokens=max_tokens, temperature=temperature)
                try:
                    async for token in stream_gen:
                        # Strip leaked model special tokens
                        token = re.sub(r'<\|[^|]*\|>', '', token)
                        if not token:
                            continue
                        full_response += token

                        # Repetition loop detection — stop if model is repeating itself
                        if len(full_response) >= next_rep_check:
                            next_rep_check = len(full_response) + 200
                            window = full_response[-150:]
                            if window in full_response[:-150]:
                                logger.warning(f"Repetition loop detected in conv {conv_id}, stopping generation")
                                break

                        # If text-based reasoning was detected, suppress all streaming output
                        if reasoning_detected:
                            continue

                        # Buffer tokens to detect and skip reasoning blocks
                        think_buffer += token

                        # Check for text-based reasoning at the start of response
                        if not reasoning_check_done:
                            check = think_buffer.lower().lstrip().replace('*', '')
                            if len(check) < 50:
                                # Could still be a reasoning prefix, keep buffering
                                if any(p.startswith(check) for p in REASONING_PREFIXES):
                                    continue
                                # Check if buffer already matches (prefix + extra chars like ":")
                                if any(check.startswith(p) for p in REASONING_PREFIXES):
                                    reasoning_detected = True
                                    reasoning_check_done = True
                                    think_buffer = ""
                                    logger.info(f"Reasoning suppressed (prefix) in conv {conv_id}")
                                    continue
                                # Secondary: broader reasoning indicators
                                if len(check) > 15 and _REASONING_INDICATORS.search(check):
                                    reasoning_detected = True
                                    reasoning_check_done = True
                                    think_buffer = ""
                                    logger.info(f"Reasoning suppressed (indicator) in conv {conv_id}")
                                    continue
                                # Not enough chars yet, keep buffering
                                if len(check) < 30:
                                    continue
                                # Doesn't match any prefix or indicator, proceed normally
                                reasoning_check_done = True
                            else:
                                if any(check.startswith(p) for p in REASONING_PREFIXES):
                                    reasoning_detected = True
                                    reasoning_check_done = True
                                    think_buffer = ""
                                    logger.info(f"Reasoning suppressed (prefix) in conv {conv_id}")
                                    continue
                                if _REASONING_INDICATORS.search(check):
                                    reasoning_detected = True
                                    reasoning_check_done = True
                                    think_buffer = ""
                                    logger.info(f"Reasoning suppressed (indicator) in conv {conv_id}")
                                    continue
                                reasoning_check_done = True

                        if not in_think_block:
                            # Check if we're entering a think block
                            if '<think>' in think_buffer:
                                # Send everything before the <think> tag
                                before = think_buffer.split('<think>')[0]
                                if before:
                                    try:
                                        await websocket.send_json({'type': 'token', 'content': before})
                                    except Exception:
                                        ws_disconnected = True
                                        break
                                in_think_block = True
                                think_buffer = think_buffer.split('<think>', 1)[1]
                            elif '<think' in think_buffer and '>' not in think_buffer.split('<think')[-1]:
                                # Partial <think tag, keep buffering
                                pass
                            else:
                                # No think tag, flush buffer
                                try:
                                    await websocket.send_json({'type': 'token', 'content': think_buffer})
                                except Exception:
                                    ws_disconnected = True
                                    break
                                think_buffer = ""
                        else:
                            # Inside think block — check for closing tag
                            if '</think>' in think_buffer:
                                # Discard everything up to and including </think>
                                after = think_buffer.split('</think>', 1)[1]
                                think_buffer = after
                                in_think_block = False
                                # Flush any remaining content after the close tag
                                if think_buffer:
                                    try:
                                        await websocket.send_json({'type': 'token', 'content': think_buffer})
                                    except Exception:
                                        ws_disconnected = True
                                        break
                                    think_buffer = ""
                finally:
                    # Always close the stream — stops vLLM generation on disconnect/loop
                    await stream_gen.aclose()

                if ws_disconnected:
                    logger.info(f"Client disconnected during streaming for conv {conv_id}")
                    full_response = filter_reasoning_text(full_response)
                    if full_response:
                        save_message(conv_id, 'assistant', full_response)
                    break  # exit the while True WebSocket loop

                # Flush any remaining buffer (shouldn't happen normally)
                if think_buffer and not in_think_block and not reasoning_detected:
                    try:
                        await websocket.send_json({'type': 'token', 'content': think_buffer})
                    except Exception:
                        break

                # If reasoning was suppressed during streaming, filter and send cleaned response
                if reasoning_detected:
                    cleaned = filter_reasoning_text(full_response)
                    if cleaned:
                        await websocket.send_json({'type': 'token', 'content': cleaned})

                # Filter full response for saving (preserves <think> tags and markdown)
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
