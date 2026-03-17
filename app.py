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
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# Theme configuration fallback
COLORS = {
    'bg_primary': '#0f172a', 'bg_secondary': '#1e293b', 'bg_tertiary': '#334155',
    'bg_quaternary': '#1e293b',
    'text_primary': '#f1f5f9', 'text_secondary': '#94a3b8', 'text_tertiary': '#64748b',
    'accent_primary': '#10b981', 'accent_hover': '#059669', 'accent_secondary': '#2563eb',
    'msg_user_bg': '#2563eb', 'msg_user_text': '#ffffff',
    'msg_assistant_bg': '#1e293b', 'msg_assistant_text': '#f1f5f9',
    'status_connected': '#10b981', 'status_disconnected': '#ef4444',
    'btn_primary': '#10b981', 'btn_primary_hover': '#059669',
    'btn_danger': '#ef4444', 'btn_danger_hover': '#dc2626',
    'btn_secondary': '#334155', 'btn_secondary_hover': '#475569',
    'modal_overlay': 'rgba(0,0,0,0.8)', 'modal_bg': '#1e293b',
    'scrollbar_track': '#1e293b', 'scrollbar_thumb': '#334155', 'scrollbar_thumb_hover': '#475569',
}
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
DIMENSIONS = {'sidebar_width': '200px', 'sidebar_collapsed_width': '50px',
              'border_radius': '8px', 'border_radius_small': '6px',
              'message_max_width': '700px', 'input_padding': '15px', 'header_padding': '20px 30px'}
FONTS = {'family': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
         'size_base': '14px', 'size_small': '13px', 'size_large': '24px'}
ANIMATIONS = {'transition_speed': '0.2s', 'slide_duration': '0.3s'}

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

def generate_css(mode='light'):
    """Generate CSS from theme configuration"""
    # Select color scheme based on mode
    colors = COLORS_DARK if mode == 'dark' else COLORS_LIGHT
    return f"""
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        html {{
            height: 100%;
            width: 100%;
            overflow: hidden;
        }}
        
        body {{
            font-family: {FONTS['family']};
            background: {colors['bg_primary']};
            color: {colors['text_primary']};
            height: 100vh;
            width: 100vw;
            display: flex;
            overflow: hidden;
            font-size: {FONTS['size_base']};
        }}
        
        .sidebar {{
            width: {DIMENSIONS['sidebar_width']};
            background: {colors['bg_secondary']};
            border-right: 3px solid {colors['accent_primary']};
            display: flex;
            flex-direction: column;
            z-index: 100;
            transition: transform {ANIMATIONS['transition_speed']}, width {ANIMATIONS['transition_speed']};
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            overflow: hidden;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
        }}
        
        .sidebar.collapsed {{
            width: {DIMENSIONS['sidebar_collapsed_width']};
            transform: translateX(0);
        }}
        
        .sidebar:not(.collapsed) {{
            transform: translateX(0);
        }}
        
        /* Sidebar overlay when open */
        .sidebar-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3);
            z-index: 99;
            backdrop-filter: blur(2px);
            transition: opacity 0.3s ease;
        }}
        
        .sidebar-overlay.show {{
            display: block;
        }}
        
        .sidebar-toggle {{
            width: 32px;
            height: 32px;
            background: {colors['accent_primary']};
            border: none;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            color: {colors['bg_primary']};
            transition: all {ANIMATIONS['transition_speed']};
        }}
        
        .sidebar-toggle:hover {{
            background: {colors['accent_hover']};
        }}
        
        .sidebar.collapsed .sidebar-content {{
            display: none;
        }}
        
        .sidebar.collapsed .search-container {{
            display: none;
        }}
        
        /* New Chat button for collapsed sidebar */
        .new-chat-icon-btn {{
            display: none;
            width: 40px;
            height: 40px;
            margin: 16px auto 0;
            background: {colors['btn_primary']};
            color: {colors['bg_primary']};
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 20px;
            align-items: center;
            justify-content: center;
            transition: all {ANIMATIONS['transition_speed']};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .sidebar.collapsed .new-chat-icon-btn {{
            display: flex;
        }}
        
        .new-chat-icon-btn:hover {{
            background: {colors['btn_primary_hover']};
            transform: scale(1.05);
        }}
        
        .new-chat-icon-btn:active {{
            transform: scale(0.95);
        }}
        
        .sidebar-header {{
            padding: 20px;
            background: {colors['bg_secondary']};
            border: none;
        }}
        
        .new-chat-btn {{
            width: 100%;
            padding: 12px;
            background: {colors['accent_primary']};
            color: {colors['bg_primary']};
            border: none;
            border-radius: 4px;
            font-size: {FONTS['size_base']};
            font-weight: 600;
            cursor: pointer;
            transition: all {ANIMATIONS['transition_speed']};
            font-family: {FONTS['family']};
        }}
        
        .new-chat-btn:hover {{ 
            background: {colors['accent_hover']}; 
            color: {colors['bg_primary']};
            transform: translateX(3px);
        }}
        
        .search-container {{
            padding: 15px;
            background: {colors['bg_secondary']};
            border: none;
        }}
        
        .search-input {{
            width: 100%;
            padding: 10px 14px;
            background: {colors['bg_tertiary']};
            border: none;
            border-radius: 4px;
            color: {colors['text_primary']};
            font-size: {FONTS['size_small']};
            font-family: {FONTS['family']};
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: {colors['accent_primary']};
            box-shadow: 0 0 0 2px {colors['accent_primary']}33;
        }}
        
        .search-input::placeholder {{
            color: {colors['text_tertiary']};
        }}
        
        .search-results {{
            max-height: 400px;
            overflow-y: auto;
            margin-top: 10px;
        }}
        
        .search-result-item {{
            padding: 12px 16px;
            margin: 5px 0;
            background: {colors['bg_primary']};
            border: 2px solid {colors['bg_tertiary']};
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            color: {colors['text_primary']};
        }}
        
        .search-result-item:hover {{
            background: {colors['btn_secondary']};
            border-color: {colors['accent_primary']};
            color: {colors['text_primary']};
            transform: translateX(3px);
        }}
        
        .search-result-title {{
            font-weight: 600;
            font-size: {FONTS['size_small']};
            color: {colors['text_primary']};
            margin-bottom: 4px;
        }}
        
        .search-result-preview {{
            font-size: 12px;
            color: {colors['text_secondary']};
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .search-result-meta {{
            font-size: 11px;
            color: {colors['text_tertiary']};
            margin-top: 4px;
        }}
        
        .search-no-results {{
            padding: 20px;
            text-align: center;
            color: {colors['text_tertiary']};
            font-size: {FONTS['size_small']};
        }}
        
        .conversations {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }}
        
        .conv-item {{
            padding: 12px 16px;
            margin-bottom: 8px;
            background: {colors['bg_tertiary']};
            border: none;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            color: {colors['text_primary']};
            transition: all 0.2s;
            gap: 10px;
        }}
        
        .conv-item:hover {{ 
            background: {colors['btn_secondary']}; 
            color: {colors['text_primary']};
            transform: translateX(3px);
        }}
        .conv-item.active {{ 
            background: {colors['accent_primary']}; 
            color: {colors['bg_primary']}; 
        }}
        
        .conv-content {{
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .conv-title {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: {FONTS['size_small']};
            font-weight: 600;
        }}
        
        .conv-preview {{
            font-size: 12px;
            color: {colors['text_secondary']};
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            line-height: 1.4;
            max-height: 2.8em;
            transition: all 0.2s;
        }}
        
        .conv-item:hover .conv-preview {{
            -webkit-line-clamp: 4;
            max-height: 5.6em;
            color: {colors['text_primary']};
        }}
        
        .conv-timestamp {{
            font-size: 11px;
            color: {colors['text_tertiary']};
            margin-top: 2px;
        }}
        
        .conv-item.active .conv-preview,
        .conv-item.active .conv-timestamp {{
            color: {colors['bg_primary']};
            opacity: 0.9;
        }}
        
        .conv-actions {{
            display: none;
            gap: 5px;
        }}
        
        .conv-item:hover .conv-actions {{ display: flex; }}
        
        .conv-btn {{
            padding: 4px 8px;
            background: {colors['btn_secondary']};
            border: none;
            border-radius: 4px;
            color: {colors['bg_primary']};
            cursor: pointer;
            font-size: 11px;
        }}
        
        .conv-btn:hover {{ background: {colors['btn_secondary_hover']}; }}
        .conv-btn.delete {{ background: {colors['btn_danger']}; }}
        .conv-btn.delete:hover {{ background: {colors['btn_danger_hover']}; }}
        
        .main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            min-width: 0;
            overflow: hidden;
            height: 100vh;
            margin-left: 0;
            transition: margin-left {ANIMATIONS['transition_speed']};
        }}
        
        /* When sidebar is open, add margin to main content */
        .sidebar:not(.collapsed) ~ .main {{
            margin-left: 0;
        }}
        
        .header {{
            padding: {DIMENSIONS['header_padding']};
            background: {colors['bg_secondary']};
            border-bottom: 3px solid {colors['accent_primary']};
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header-left {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .header-left .sidebar-toggle {{
            margin-right: 8px;
        }}
        
        .header-right {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .header h1 {{ 
            font-size: {FONTS['size_large']}; 
            font-weight: 600;
            color: {colors['text_primary']};
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .model-selector {{
            padding: 8px 12px;
            background: {colors['bg_primary']};
            border: 1px solid {colors['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            color: {colors['text_primary']};
            font-size: {FONTS['size_small']};
            cursor: pointer;
            min-width: 200px;
        }}
        
        .model-selector:focus {{ outline: none; border-color: {colors['accent_primary']}; }}
        .model-selector:hover {{ border-color: {colors['accent_primary']}; }}
        
        .status {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: {FONTS['size_base']};
            color: {colors['text_secondary']};
        }}
        
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {colors['status_connected']};
            animation: pulse 2s infinite;
        }}
        
        .status-dot.disconnected {{ background: {colors['status_disconnected']}; animation: none; }}
        
        .status-dot.booting {{
            background: #ffa500;
            animation: pulse 1s infinite;
        }}
        
        .status-progress-container {{
            margin-top: 8px;
            width: 200px;
        }}
        
        .status-progress-bar {{
            width: 100%;
            height: 4px;
            background: {colors['bg_tertiary']};
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 4px;
            position: relative;
        }}
        
        .status-progress-bar::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: {colors['accent_primary']};
            border-radius: 2px;
            transition: width 0.3s ease;
            width: var(--progress, 0%);
        }}
        
        .status-progress-text {{
            font-size: {FONTS['size_small']};
            color: {colors['text_secondary']};
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .messages {{
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            align-items: center;
            justify-content: flex-start;
            max-width: 100%;
            width: 100%;
            background: {colors['bg_primary']};
            min-height: 0;
        }}
        
        .message {{
            width: 100%;
            max-width: {DIMENSIONS['message_max_width']};
            padding: 20px 24px;
            line-height: 1.7;
            animation: slideIn {ANIMATIONS['slide_duration']} ease-out;
            font-size: {FONTS['size_base']};
            border: 2px solid {colors['accent_primary']};
            border-radius: 8px;
            transition: all 0.2s;
            box-sizing: border-box;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(5px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .message.user {{
            background: {colors['msg_user_bg']};
            color: {colors['msg_user_text']};
            text-align: right;
            margin-left: auto;
            margin-right: 0;
            border-color: {colors['accent_primary']};
        }}
        
        .message.user:hover {{
            border-color: {colors['accent_hover']};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        
        .message.assistant {{
            background: {colors['msg_assistant_bg']};
            color: {colors['msg_assistant_text']};
            text-align: left;
            margin-left: 0;
            margin-right: auto;
            position: relative;
            border-color: {colors['accent_primary']};
        }}
        
        .message.assistant:hover {{
            border-color: {colors['accent_hover']};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        
        .message-timestamp {{
            font-size: 11px;
            color: #b0c9df;
            margin-top: 4px;
            opacity: 0.8;
        }}
        
        .message.user .message-timestamp {{
            text-align: right;
        }}
        
        .message.assistant .message-timestamp {{
            text-align: left;
        }}
        
        .message.assistant pre {{
            position: relative;
            background: {colors['bg_tertiary'] if mode == 'dark' else '#e9eced'};
            padding: 12px 16px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 8px 0;
            border: 2px solid {colors['accent_primary']};
            font-family: 'Courier New', 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            line-height: 1.4;
        }}
        
        .message.assistant code {{
            background: {colors['bg_tertiary'] if mode == 'dark' else '#e9eced'};
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            border: 1px solid {colors['accent_primary']};
            color: {colors['text_primary']};
        }}
        
        .message.assistant pre code {{
            background: transparent;
            padding: 0;
            color: {colors['text_primary']};
            border: none;
        }}
        
        /* Syntax highlighting styles */
        .message.assistant pre.hljs {{
            background: {colors['bg_tertiary'] if mode == 'dark' else '#e9eced'};
            color: {colors['text_primary']};
        }}
        
        .message.assistant .hljs-keyword {{ color: {colors['text_primary']}; font-weight: 600; }}
        .message.assistant .hljs-string {{ color: {colors['accent_secondary']}; }}
        .message.assistant .hljs-comment {{ color: {colors['text_tertiary']}; font-style: italic; }}
        .message.assistant .hljs-number {{ color: {colors['accent_primary']}; }}
        .message.assistant .hljs-function {{ color: {colors['text_primary']}; }}
        .message.assistant .hljs-variable {{ color: {colors['text_secondary']}; }}
        .message.assistant .hljs-title {{ color: {colors['text_primary']}; }}
        .message.assistant .hljs-type {{ color: {colors['text_secondary']}; }}
        
        /* Markdown Typography Styles */
        .message.assistant h1 {{
            font-size: 1.8em;
            font-weight: 700;
            color: {colors['text_primary']};
            margin: 20px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 3px solid {colors['accent_primary']};
            line-height: 1.3;
        }}
        
        .message.assistant h2 {{
            font-size: 1.5em;
            font-weight: 700;
            color: {colors['text_primary']};
            margin: 18px 0 10px 0;
            padding-bottom: 6px;
            border-bottom: 2px solid {colors['bg_tertiary']};
            line-height: 1.3;
        }}
        
        .message.assistant h3 {{
            font-size: 1.3em;
            font-weight: 600;
            color: {colors['text_primary']};
            margin: 16px 0 8px 0;
            line-height: 1.4;
        }}
        
        .message.assistant h4 {{
            font-size: 1.15em;
            font-weight: 600;
            color: {colors['text_secondary']};
            margin: 14px 0 6px 0;
            line-height: 1.4;
        }}
        
        .message.assistant h5 {{
            font-size: 1.05em;
            font-weight: 600;
            color: {colors['text_secondary']};
            margin: 12px 0 6px 0;
            line-height: 1.4;
        }}
        
        .message.assistant h6 {{
            font-size: 1em;
            font-weight: 600;
            color: {colors['text_secondary']};
            margin: 10px 0 4px 0;
            line-height: 1.4;
        }}
        
        .message.assistant p {{
            margin: 10px 0;
            line-height: 1.7;
            color: {colors['text_primary']};
        }}
        
        .message.assistant strong,
        .message.assistant b {{
            font-weight: 700;
            color: {colors['text_primary']};
            background: {colors['bg_tertiary']}33;
            padding: 1px 3px;
            border-radius: 2px;
        }}
        
        .message.assistant em,
        .message.assistant i {{
            font-style: italic;
            color: {colors['text_secondary']};
        }}
        
        .message.assistant strong em,
        .message.assistant em strong,
        .message.assistant b i,
        .message.assistant i b {{
            font-weight: 700;
            font-style: italic;
            color: {colors['text_primary']};
        }}
        
        .message.assistant ul,
        .message.assistant ol {{
            margin: 12px 0;
            padding-left: 30px;
            color: {colors['text_primary']};
        }}
        
        .message.assistant ul {{
            list-style-type: disc;
        }}
        
        .message.assistant ol {{
            list-style-type: decimal;
        }}
        
        .message.assistant li {{
            margin: 6px 0;
            line-height: 1.6;
            color: {colors['text_primary']};
        }}
        
        .message.assistant li p {{
            margin: 4px 0;
        }}
        
        .message.assistant ul ul,
        .message.assistant ol ol,
        .message.assistant ul ol,
        .message.assistant ol ul {{
            margin: 4px 0;
        }}
        
        .message.assistant blockquote {{
            margin: 16px 0;
            padding: 12px 16px;
            border-left: 4px solid {colors['accent_primary']};
            background: {colors['bg_secondary']};
            border-radius: 4px;
            color: {colors['text_secondary']};
            font-style: italic;
        }}
        
        .message.assistant blockquote p {{
            margin: 4px 0;
        }}
        
        .message.assistant blockquote p:first-child {{
            margin-top: 0;
        }}
        
        .message.assistant blockquote p:last-child {{
            margin-bottom: 0;
        }}
        
        .message.assistant a {{
            color: {colors['text_secondary']};
            text-decoration: underline;
            text-decoration-color: {colors['bg_tertiary']};
            transition: all 0.2s;
        }}
        
        .message.assistant a:hover {{
            color: {colors['accent_primary']};
            text-decoration-color: {colors['accent_primary']};
            background: {colors['bg_tertiary']}33;
            padding: 1px 2px;
            border-radius: 2px;
        }}
        
        .message.assistant hr {{
            border: none;
            border-top: 2px solid {colors['bg_tertiary']};
            margin: 20px 0;
        }}
        
        .message.assistant table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            border: 2px solid {colors['bg_tertiary']};
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .message.assistant thead {{
            background: {colors['bg_secondary']};
        }}
        
        .message.assistant th {{
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            color: {colors['text_primary']};
            border-bottom: 2px solid {colors['accent_primary']};
        }}
        
        .message.assistant td {{
            padding: 8px 12px;
            border-bottom: 1px solid {colors['bg_tertiary']};
            color: {colors['text_primary']};
        }}
        
        .message.assistant tbody tr:last-child td {{
            border-bottom: none;
        }}
        
        .message.assistant tbody tr:hover {{
            background: rgba(176,201,223,0.2);
        }}
        
        .message.assistant img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            border: 2px solid #b0c9df;
            margin: 12px 0;
        }}
        
        /* Copy button styles */
        .copy-btn {{
            position: absolute;
            top: 8px;
            right: 8px;
            background: {COLORS['btn_secondary']};
            border: none;
            border-radius: 4px;
            padding: 6px 10px;
            cursor: pointer;
            font-size: 12px;
            color: {COLORS['text_primary']};
            opacity: 0;
            transition: opacity 0.2s;
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        
        .message.assistant:hover .copy-btn,
        .message.assistant pre:hover .copy-btn {{
            opacity: 1;
        }}
        
        .copy-btn:hover {{
            background: {colors['btn_secondary_hover']};
        }}
        
        .copy-btn.copied {{
            background: {colors['accent_secondary']};
            color: {colors['bg_primary']};
        }}
        
        .loading {{
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: {DIMENSIONS['message_max_width']};
            width: 100%;
            padding: 8px 0;
            background: transparent;
            margin-left: 0;
            margin-right: auto;
        }}
        
        .loading-spinner {{ display: flex; gap: 5px; }}
        
        .loading-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {colors['accent_primary']};
            animation: bounce 1.4s infinite ease-in-out both;
        }}
        
        .loading-dot:nth-child(1) {{ animation-delay: -0.32s; }}
        .loading-dot:nth-child(2) {{ animation-delay: -0.16s; }}
        
        @keyframes bounce {{
            0%, 80%, 100% {{ transform: scale(0); opacity: 0.5; }}
            40% {{ transform: scale(1); opacity: 1; }}
        }}
        
        .loading-text {{ color: {colors['text_secondary']}; font-size: {FONTS['size_base']}; }}
        
        .input-area {{
            padding: 20px 24px;
            background: {colors['bg_quaternary']};
            border-top: 3px solid {colors['accent_primary']};
            display: flex;
            justify-content: center;
            flex-shrink: 0;
            width: 100%;
            box-sizing: border-box;
        }}
        
        .input-container {{
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: {DIMENSIONS['message_max_width']};
            width: 100%;
            background: {colors['bg_primary']};
            border: 2px solid {colors['accent_primary']};
            border-radius: 8px;
            padding: 14px 18px;
            cursor: text;
            box-sizing: border-box;
        }}
        
        .input-container:focus-within {{
            border-color: {colors['accent_hover']};
            box-shadow: 0 0 0 2px rgba(129,148,177,0.1);
        }}
        
        #input {{
            flex: 1;
            padding: 0;
            background: transparent;
            color: {colors['text_primary']};
            border: none;
            border-radius: 0;
            font-size: {FONTS['size_base']};
            resize: none;
            font-family: {FONTS['family']};
            min-height: 24px;
            max-height: 66vh;
            overflow-y: auto;
            overflow-x: hidden;
            line-height: 1.5;
            width: 100%;
            box-sizing: border-box;
        }}
        
        #input:focus {{ outline: none; }}
        #input:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        #input::placeholder {{ color: #b0c9df; }}
        
        #send {{
            width: 36px;
            height: 36px;
            padding: 0;
            background: #8194b1;
            color: #fff4de;
            border: 2px solid #8194b1;
            border-radius: 4px;
            cursor: pointer;
            transition: all {ANIMATIONS['transition_speed']};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }}
        
        #send:hover:not(:disabled) {{
            background: #b0c9df;
            border-color: #8194b1;
            transform: translateX(2px);
        }}
        #send:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}
        #send::before {{
            content: '';
        }}
        
        /* Settings Menu (Bottom Left) */
        .settings-menu {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 1000;
        }}
        
        .settings-menu-toggle {{
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: {colors['accent_primary']};
            color: {colors['bg_primary']};
            border: 3px solid {colors['accent_primary']};
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15), inset 0 2px 4px rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            z-index: 1001;
            position: relative;
            overflow: visible;
        }}
        
        .settings-menu-toggle:hover {{
            background: {colors['accent_hover']};
            border-color: {colors['accent_hover']};
            transform: scale(1.1) rotate(15deg);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2), inset 0 2px 4px rgba(255, 255, 255, 0.3);
        }}
        
        .settings-menu-toggle:active {{
            transform: scale(0.95) rotate(15deg);
        }}
        
        .gear-icon {{
            width: 28px;
            height: 28px;
            stroke: currentColor;
            stroke-width: 2.5;
            stroke-linecap: round;
            stroke-linejoin: round;
            filter: drop-shadow(0 1px 1px rgba(0, 0, 0, 0.2));
            animation: none;
        }}
        
        .settings-menu-toggle:hover .gear-icon {{
            animation: rotateGear 2s linear infinite;
        }}
        
        @keyframes rotateGear {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        
        .settings-menu-content {{
            position: absolute;
            bottom: 70px;
            left: 0;
            background: {colors['bg_secondary']};
            border: 2px solid {colors['accent_primary']};
            border-radius: {DIMENSIONS['border_radius']};
            padding: 8px;
            display: none;
            flex-direction: column;
            gap: 4px;
            min-width: 200px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            opacity: 0;
            transform: translateY(10px) scale(0.95);
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            pointer-events: none;
        }}
        
        .settings-menu-content.show {{
            display: flex;
            opacity: 1;
            transform: translateY(0) scale(1);
            pointer-events: auto;
        }}
        
        .settings-menu-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            background: {colors['bg_primary']};
            border: 1px solid {colors['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            color: {colors['text_primary']};
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: {FONTS['size_base']};
            width: 100%;
            text-align: left;
        }}
        
        .settings-menu-item:hover {{
            background: {colors['btn_secondary']};
            border-color: {colors['accent_primary']};
            transform: translateX(4px);
        }}
        
        .settings-icon {{
            font-size: 20px;
            width: 24px;
            text-align: center;
        }}
        
        .settings-label {{
            flex: 1;
            font-weight: 500;
        }}
        
        .settings-model-name {{
            font-size: {FONTS['size_small']};
            color: {colors['text_secondary']};
            font-weight: normal;
        }}
        
        .settings-dropdown {{
            position: absolute;
            bottom: 100%;
            left: 0;
            margin-bottom: 8px;
            background: {colors['bg_secondary']};
            border: 2px solid {colors['accent_primary']};
            border-radius: {DIMENSIONS['border_radius']};
            padding: 8px;
            min-width: 250px;
            max-height: 300px;
            overflow-y: auto;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: {COLORS['scrollbar_track']}; }}
        ::-webkit-scrollbar-thumb {{ background: {COLORS['scrollbar_thumb']}; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {COLORS['scrollbar_thumb_hover']}; }}
        
        .welcome {{
            text-align: center;
            color: {colors['text_secondary']};
            padding: 60px 40px;
            max-width: {DIMENSIONS['message_max_width']};
            width: 100%;
            margin: 0 auto;
        }}
        
        .welcome h2 {{
            font-size: 28px;
            margin-bottom: 12px;
            color: {colors['text_primary']};
            font-weight: 500;
        }}
        
        .welcome p {{
            font-size: {FONTS['size_base']};
            color: {colors['text_secondary']};
            color: {COLORS['text_secondary']};
        }}
        
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: {COLORS['modal_overlay']};
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }}
        
        .modal.show {{ display: flex; }}
        
        .modal-content {{
            background: {COLORS['modal_bg']};
            padding: 30px;
            border-radius: {DIMENSIONS['border_radius']};
            min-width: 400px;
            border: 1px solid {COLORS['bg_tertiary']};
        }}
        
        .modal-content h3 {{ margin-bottom: 20px; }}
        
        .modal-content input {{
            width: 100%;
            padding: 12px;
            background: {COLORS['bg_primary']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            color: {COLORS['text_primary']};
            margin-bottom: 20px;
        }}
        
        .modal-buttons {{
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }}
        
        .modal-btn {{
            padding: 10px 20px;
            border: none;
            border-radius: {DIMENSIONS['border_radius_small']};
            font-weight: 600;
            cursor: pointer;
        }}
        
        .modal-btn.cancel {{
            background: {COLORS['btn_secondary']};
            color: white;
        }}
        
        .modal-btn.confirm {{
            background: {COLORS['btn_primary']};
            color: white;
        }}
        
        /* Terminal Log Viewer */
        .web-search-btn {{
            padding: 8px 12px;
            background: {colors['btn_secondary']};
            color: {colors['text_primary']};
            border: 2px solid {colors['accent_primary']};
            border-radius: 4px;
            cursor: pointer;
            font-size: {FONTS['size_small']};
            transition: all {ANIMATIONS['transition_speed']};
            font-family: {FONTS['family']};
        }}
        
        .web-search-btn:hover {{
            background: {colors['btn_secondary_hover']};
            border-color: {colors['accent_hover']};
        }}
        
        .log-viewer-btn {{
            padding: 8px 12px;
            background: {colors['btn_secondary']};
            color: {colors['text_primary']};
            border: 2px solid {colors['accent_primary']};
            border-radius: 4px;
            cursor: pointer;
            font-size: {FONTS['size_small']};
            transition: all {ANIMATIONS['transition_speed']};
            font-family: {FONTS['family']};
        }}
        
        .log-viewer-btn:hover {{
            background: {colors['btn_secondary_hover']};
            border-color: {colors['accent_hover']};
        }}
        
        .log-viewer {{
            display: none;
            position: fixed;
            bottom: 70px;
            left: 20px;
            width: 500px;
            height: 400px;
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius']};
            z-index: 99;
            flex-direction: column;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .log-viewer.show {{ display: flex; }}
        
        /* Web Search Modal */
        .web-search-modal {{
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 800px;
            max-height: 80vh;
            background: {colors['modal_bg']};
            border: 3px solid {colors['accent_primary']};
            border-radius: {DIMENSIONS['border_radius']};
            z-index: 1001;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            flex-direction: column;
        }}
        
        .web-search-modal.show {{ display: flex; }}
        
        .web-search-header {{
            padding: 16px 20px;
            background: {colors['bg_secondary']};
            border-bottom: 2px solid {colors['accent_primary']};
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .web-search-header h3 {{
            font-size: {FONTS['size_large']};
            font-weight: 600;
            color: {colors['text_primary']};
        }}
        
        .web-search-close {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: {colors['text_primary']};
            padding: 0;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .web-search-close:hover {{
            color: {colors['accent_primary']};
        }}
        
        .web-search-content {{
            padding: 20px;
            overflow-y: auto;
            flex: 1;
        }}
        
        .web-search-form {{
            margin-bottom: 20px;
        }}
        
        .web-search-input-group {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .web-search-input {{
            flex: 1;
            padding: 12px 16px;
            border: 2px solid {colors['accent_primary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            font-size: {FONTS['size_base']};
            font-family: {FONTS['family']};
            background: {colors['bg_primary']};
            color: {colors['text_primary']};
        }}
        
        .web-search-input:focus {{
            outline: none;
            border-color: {colors['accent_hover']};
        }}
        
        .web-search-btn-submit {{
            padding: 12px 24px;
            background: {colors['btn_primary']};
            color: white;
            border: 2px solid {colors['accent_primary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            cursor: pointer;
            font-size: {FONTS['size_base']};
            font-weight: 600;
            transition: all {ANIMATIONS['transition_speed']};
        }}
        
        .web-search-btn-submit:hover {{
            background: {colors['btn_primary_hover']};
        }}
        
        .web-search-sources {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .web-search-source-checkbox {{
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 8px 12px;
            background: {colors['bg_secondary']};
            border: 1px solid {colors['accent_primary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            cursor: pointer;
        }}
        
        .web-search-source-checkbox input {{
            cursor: pointer;
        }}
        
        .web-search-results {{
            margin-top: 20px;
        }}
        
        .web-search-result {{
            padding: 15px;
            margin-bottom: 15px;
            background: {colors['bg_secondary']};
            border: 1px solid {colors['accent_primary']};
            border-radius: {DIMENSIONS['border_radius_small']};
        }}
        
        .web-search-result-title {{
            font-size: {FONTS['size_base']};
            font-weight: 600;
            color: {colors['accent_primary']};
            margin-bottom: 8px;
        }}
        
        .web-search-result-title a {{
            color: {colors['accent_primary']};
            text-decoration: none;
        }}
        
        .web-search-result-title a:hover {{
            text-decoration: underline;
        }}
        
        .web-search-result-url {{
            font-size: {FONTS['size_small']};
            color: {colors['text_secondary']};
            margin-bottom: 8px;
        }}
        
        .web-search-result-snippet {{
            font-size: {FONTS['size_base']};
            color: {colors['text_primary']};
            line-height: 1.5;
        }}
        
        .web-search-result-source {{
            display: inline-block;
            padding: 4px 8px;
            background: {colors['accent_primary']};
            color: white;
            border-radius: 4px;
            font-size: {FONTS['size_small']};
            margin-top: 8px;
        }}
        
        .web-search-loading {{
            text-align: center;
            padding: 20px;
            color: {colors['text_secondary']};
        }}
        
        .log-viewer-header {{
            padding: 12px 15px;
            background: {COLORS['bg_primary']};
            border-bottom: 1px solid {COLORS['bg_tertiary']};
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .log-viewer-header h3 {{
            font-size: {FONTS['size_base']};
            font-weight: 600;
        }}
        
        .log-viewer-close {{
            background: none;
            border: none;
            color: {COLORS['text_secondary']};
            cursor: pointer;
            font-size: 18px;
            padding: 0;
            width: 24px;
            height: 24px;
        }}
        
        .log-viewer-content {{
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            font-family: {FONTS['family']};
            font-size: 12px;
            color: {COLORS['text_primary']};
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        /* Model Toggle */
        .theme-toggle-btn {{
            width: 40px;
            height: 40px;
            background: {colors['btn_secondary']};
            border: 2px solid {colors['accent_primary']};
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: {colors['text_primary']};
            transition: all {ANIMATIONS['transition_speed']};
        }}
        
        .theme-toggle-btn:hover {{
            background: {colors['btn_secondary_hover']};
            border-color: {colors['accent_hover']};
            transform: translateY(-1px);
        }}
        
        .model-toggle {{
            position: relative;
        }}
        
        .model-toggle-btn {{
            padding: 10px 18px;
            background: #e9eced;
            border: 2px solid #b0c9df;
            border-radius: 4px;
            color: #8194b1;
            font-size: {FONTS['size_base']};
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 180px;
            justify-content: space-between;
            font-family: {FONTS['family']};
        }}
        
        .model-toggle-btn:hover {{
            border-color: #8194b1;
            background: #b0c9df;
            color: #fff4de;
            transform: translateY(-1px);
        }}
        
        .model-toggle-btn.switching {{
            opacity: 0.6;
            cursor: not-allowed;
            border-color: {COLORS['accent_primary']};
        }}
        
        
        .model-dropdown {{
            display: none;
            position: absolute;
            top: 100%;
            right: 0;
            margin-top: 5px;
            background: #fff4de;
            border: 3px solid #8194b1;
            border-radius: 4px;
            min-width: 250px;
            max-height: 400px;
            overflow-y: auto;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(129,148,177,0.3);
        }}
        
        .model-dropdown.show {{
            display: block;
        }}
        
        .model-option {{
            padding: 12px 16px;
            cursor: pointer;
            border-bottom: 2px solid #b0c9df;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #8194b1;
            transition: all 0.2s;
        }}
        
        .model-option:last-child {{
            border-bottom: none;
        }}
        
        .model-option:hover {{
            background: #b0c9df;
            color: #fff4de;
            transform: translateX(3px);
        }}
        
        .model-option.active {{
            background: #8194b1;
            color: #fff4de;
            border-color: #8194b1;
        }}
        
        .model-option-name {{
            font-weight: 500;
        }}
        
        .model-option-badge {{
            font-size: 10px;
            padding: 2px 6px;
            background: {COLORS['bg_primary']};
            border-radius: 4px;
            color: {COLORS['text_secondary']};
        }}
        
        .model-option.active .model-option-badge {{
            background: rgba(255,255,255,0.2);
            color: white;
        }}
        
        /* Model Switch Status Console */
        .model-status-console {{
            position: fixed;
            top: 80px;
            right: 20px;
            width: 350px;
            max-height: 200px;
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius']};
            z-index: 999;
            display: none;
            flex-direction: column;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .model-status-console.show {{
            display: flex;
        }}
        
        .model-status-header {{
            padding: 10px 15px;
            background: {COLORS['bg_primary']};
            border-bottom: 1px solid {COLORS['bg_tertiary']};
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: {FONTS['size_small']};
            font-weight: 600;
        }}
        
        .model-status-close {{
            background: none;
            border: none;
            color: {COLORS['text_secondary']};
            cursor: pointer;
            font-size: 16px;
            padding: 0;
            width: 20px;
            height: 20px;
        }}
        
        .model-status-content {{
            padding: 15px;
            font-size: {FONTS['size_small']};
        }}
        
        .model-status-message {{
            margin-bottom: 10px;
            color: {COLORS['text_primary']};
        }}
        
        .model-status-progress {{
            width: 100%;
            height: 6px;
            background: {COLORS['bg_primary']};
            border-radius: 3px;
            overflow: hidden;
            margin-top: 10px;
        }}
        
        .model-status-progress-bar {{
            height: 100%;
            background: {COLORS['accent_primary']};
            transition: width 0.3s;
        }}
        
        .model-status-status {{
            margin-top: 8px;
            font-size: 11px;
            color: {COLORS['text_secondary']};
        }}
        
        .model-status-status.success {{
            color: {COLORS['status_connected']};
        }}
        
        .model-status-status.error {{
            color: {COLORS['status_disconnected']};
        }}
        
        /* Boot Menu Styles */
        .boot-menu {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #e9eced;
            z-index: 10000;
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: {FONTS['family']};
        }}
        
        .boot-menu.show {{
            display: flex;
        }}
        
        .boot-menu-content {{
            background: #fff4de;
            border: 3px solid #8194b1;
            padding: 40px;
            max-width: 800px;
            width: 90%;
            box-shadow: 0 8px 32px rgba(129,148,177,0.3);
        }}
        
        .boot-menu-header {{
            text-align: center;
            margin-bottom: 30px;
            color: #8194b1;
        }}
        
        .boot-menu-header h1 {{
            font-size: 24px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #8194b1;
        }}
        
        .boot-menu-header p {{
            font-size: 14px;
            color: #8194b1;
        }}
        
        .boot-menu-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        
        .boot-menu-item {{
            padding: 12px 16px;
            margin: 8px 0;
            background: #e9eced;
            border: 2px solid #b0c9df;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #8194b1;
        }}
        
        .boot-menu-item:hover {{
            background: #b0c9df;
            border-color: #8194b1;
            transform: translateX(5px);
            color: #fff4de;
        }}
        
        .boot-menu-item.selected {{
            background: #8194b1;
            color: #fff4de;
            border-color: #8194b1;
        }}
        
        .boot-menu-item.loading {{
            opacity: 0.6;
            cursor: not-allowed;
        }}
        
        .boot-menu-item-name {{
            font-weight: 600;
        }}
        
        .boot-menu-item-badge {{
            font-size: 11px;
            padding: 2px 8px;
            background: #b0c9df;
            border-radius: 3px;
            color: #8194b1;
        }}
        
        .boot-menu-item:hover .boot-menu-item-badge {{
            background: rgba(255,255,255,0.3);
            color: #fff4de;
        }}
        
        .boot-menu-item.selected .boot-menu-item-badge {{
            background: rgba(255,255,255,0.3);
            color: #fff4de;
        }}
        
        .boot-menu-instructions {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #b0c9df;
            text-align: center;
            color: #8194b1;
            font-size: 12px;
        }}
        
        .boot-menu-error {{
            margin-top: 20px;
            padding: 12px;
            background: #ff7863;
            color: white;
            border-radius: 4px;
            font-size: 13px;
            display: none;
        }}
        
        .boot-menu-error.show {{
            display: block;
        }}
        
        /* Model Switch Confirmation Modal */
        .model-switch-confirm {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.6);
            z-index: 10001;
            display: none;
            align-items: center;
            justify-content: center;
        }}
        
        .model-switch-confirm.show {{
            display: flex;
        }}
        
        .model-switch-confirm-content {{
            background: #fff4de;
            border: 3px solid #8194b1;
            padding: 30px;
            max-width: 500px;
            width: 90%;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        
        .model-switch-confirm-header {{
            color: #8194b1;
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .model-switch-confirm-warning {{
            background: #fff4de;
            border-left: 4px solid #ff7863;
            padding: 12px;
            margin: 15px 0;
            color: #8194b1;
            font-size: 14px;
        }}
        
        .model-switch-confirm-info {{
            color: #8194b1;
            font-size: 14px;
            margin: 15px 0;
            line-height: 1.6;
        }}
        
        .model-switch-confirm-buttons {{
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            margin-top: 20px;
        }}
        
        .model-switch-confirm-btn {{
            padding: 10px 20px;
            border: 2px solid #8194b1;
            background: #e9eced;
            color: #8194b1;
            cursor: pointer;
            border-radius: 4px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        
        .model-switch-confirm-btn:hover {{
            background: #b0c9df;
            transform: translateY(-1px);
        }}
        
        .model-switch-confirm-btn.primary {{
            background: #8194b1;
            color: #fff4de;
        }}
        
        .model-switch-confirm-btn.primary:hover {{
            background: #b0c9df;
            border-color: #b0c9df;
        }}
        
        /* Mobile Responsive Styles */
        @media (max-width: 768px) {{
            html, body {{
                height: 100%;
                width: 100%;
                overflow: hidden;
            }}
            
            body {{
                flex-direction: column;
                height: 100vh;
                min-height: 100vh;
            }}
            
            .sidebar {{
                width: 280px;
                max-width: 85%;
                height: 100vh;
                border-right: 3px solid {colors['accent_primary']};
                border-bottom: none;
                position: fixed;
                top: 0;
                left: 0;
                z-index: 100;
                transform: translateX(-100%);
                transition: transform 0.3s ease;
                overflow-y: auto;
            }}
            
            .sidebar.show {{
                transform: translateX(0);
            }}
            
            .sidebar.collapsed {{
                width: 280px;
                transform: translateX(-100%);
            }}
            
            /* Overlay when sidebar is open */
            .sidebar-overlay {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                z-index: 99;
                backdrop-filter: blur(2px);
            }}
            
            .sidebar-overlay.show {{
                display: block;
            }}
            
            .main {{
                width: 100%;
                margin-left: 0;
                padding-top: 70px;
            }}
            
            .messages {{
                padding: 20px 15px;
            }}
            
            .message {{
                max-width: 95%;
            }}
            
            .header {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 98;
                padding: 12px 15px;
                background: {colors['bg_secondary']};
                border-bottom: 3px solid {colors['accent_primary']};
            }}
            
            .header h1 {{
                font-size: 20px;
            }}
            
            .messages {{
                padding: 20px 15px;
                margin-top: 70px;
                padding-bottom: env(safe-area-inset-bottom, 0px);
                padding-bottom: calc(env(safe-area-inset-bottom, 0px) + 120px);
            }}
            
            .message {{
                max-width: 95%;
                padding: 16px 18px;
                font-size: 18px;
                line-height: 1.6;
            }}
            
            .input-area {{
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                padding: 12px;
                padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px));
                background: {colors['bg_quaternary']};
                border-top: 3px solid {colors['accent_primary']};
                z-index: 97;
                /* Rise with keyboard */
                transform: translateY(0);
                transition: transform 0.25s ease-out;
            }}
            
            .settings-menu {{
                z-index: 98;
                bottom: calc(80px + env(safe-area-inset-bottom, 0px));
            }}
            
            /* When keyboard is open, input area rises */
            @supports (-webkit-touch-callout: none) {{
                .input-area {{
                    position: -webkit-sticky;
                    bottom: 0;
                }}
            }}
            
            .input-container {{
                max-width: 100%;
            }}
            
            #input {{
                max-height: 50vh;
                font-size: 18px; /* Larger font, prevents zoom on iOS */
                line-height: 1.5;
                padding: 12px;
            }}
            
            /* Swipe gestures for conversations */
            .conversation-item {{
                position: relative;
                overflow: hidden;
            }}
            
            .conversation-item .swipe-actions {{
                position: absolute;
                top: 0;
                right: 0;
                height: 100%;
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 0 16px;
                background: {colors['btn_danger']};
                transform: translateX(100%);
                transition: transform 0.3s ease;
            }}
            
            .conversation-item.swipe-left .swipe-actions {{
                transform: translateX(0);
            }}
            
            .conversation-item .swipe-actions button {{
                padding: 8px 12px;
                background: {colors['btn_primary']};
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            
            /* Hide swipe actions on desktop */
            @media (min-width: 769px) {{
                .conversation-item .swipe-actions {{
                    display: none !important;
                }}
            }}
            
            .model-toggle-btn {{
                font-size: 12px;
                padding: 6px 10px;
            }}
            
            .model-dropdown {{
                right: 10px;
                left: auto;
                width: calc(100vw - 20px);
                max-width: 300px;
            }}
            
            .status {{
                font-size: 11px;
            }}
            
            .log-viewer-btn {{
                padding: 6px 10px;
                font-size: 12px;
            }}
            
            .log-viewer {{
                width: 100%;
                height: 80vh;
                max-width: 100%;
                left: 0;
                right: 0;
            }}
            
            .model-status-console {{
                width: calc(100% - 20px);
                left: 10px;
                right: 10px;
            }}
            
            .modal-content {{
                width: 90%;
                max-width: 90%;
                padding: 20px;
            }}
            
            .boot-menu-content {{
                width: 95%;
                max-width: 95%;
                padding: 20px;
            }}
            
            .conversations {{
                max-height: calc(40vh - 100px);
            }}
            
            .search-container {{
                padding: 8px;
            }}
            
            .settings-menu {{
                bottom: 15px;
                left: 15px;
            }}
            
            .settings-menu-toggle {{
                width: 50px;
                height: 50px;
                font-size: 20px;
            }}
            
            .settings-menu-content {{
                bottom: 65px;
                min-width: 180px;
            }}
        }}
        
        @media (max-width: 480px) {{
            .header h1 {{
                font-size: 16px;
            }}
            
            .message {{
                font-size: 14px;
                padding: 10px 12px;
            }}
            
            .model-toggle-btn span {{
                display: none;
            }}
            
            .model-toggle-btn span:first-child {{
                display: inline;
            }}
            
            #input {{
                font-size: 16px;
            }}
            
            .welcome {{
                padding: 30px 20px;
            }}
            
            .welcome h2 {{
                font-size: 22px;
            }}
        }}
        
    """


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

@app.get("/api/css")
async def serve_css(mode: str = "light"):
    try:
        css = generate_css(mode)
        return Response(content=css, media_type="text/css")
    except Exception as e:
        logger.error(f"Error generating CSS: {e}")
        return Response(content="/* CSS generation error */", media_type="text/css")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, mode: str = "light"):
    try:
        colors = COLORS_DARK if mode == 'dark' else COLORS_LIGHT
        return templates.TemplateResponse("index.html", {
            "request": request,
            "mode": mode,
            "colors_json": json.dumps(colors),
            "theme_colors": type('Colors', (), colors)(),
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
