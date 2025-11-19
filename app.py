#!/usr/bin/env python3
"""
AI Chat with vLLM - Simplified UI with Customizations
- Centralized theme configuration (theme_config.py)
- Model selection with preset models (including quantized versions)
- Terminal log viewer (button in bottom left)
- Markdown support for code and formatted text
- Future: Search feature for Google, Wikipedia, Reddit, etc.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import sqlite3
from datetime import datetime
import logging
from openai import OpenAI
import traceback
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

# Import theme configuration
try:
    from theme_config import COLORS, DIMENSIONS, FONTS, ANIMATIONS
except ImportError:
    # Fallback if theme_config not found
    COLORS = {
        'bg_primary': '#0f172a', 'bg_secondary': '#1e293b', 'bg_tertiary': '#334155',
        'text_primary': '#f1f5f9', 'text_secondary': '#94a3b8',
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
    DIMENSIONS = {'sidebar_width': '280px', 'border_radius': '8px', 'border_radius_small': '6px',
                  'message_max_width': '70%', 'input_padding': '15px', 'header_padding': '20px 30px'}
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

# Create handler that writes to both stdout and StringIO
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
VLLM_HOST = "http://vllm:8000/v1"

# Available models (including quantized versions)
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-Coder-3B-Instruct", "name": "Qwen 2.5 Coder 3B", "quantized": False},
    {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "name": "Qwen 2.5 Coder 7B", "quantized": False},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "name": "Llama 3.2 3B", "quantized": False},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "name": "Llama 3.2 1B (Quantized)", "quantized": True},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B", "quantized": False},
]

# Default model
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

app = FastAPI(title="AI Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client for vLLM
try:
    client = OpenAI(base_url=VLLM_HOST, api_key="dummy")
    logger.info(f"✓ Connected to vLLM at {VLLM_HOST}")
except Exception as e:
    logger.error(f"Failed to initialize vLLM client: {e}")

# ==================== Database ====================

def init_db():
    """Initialize database"""
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
                  FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')
    
    conn.commit()
    conn.close()
    logger.info("✓ Database initialized")

init_db()

# ==================== Models ====================

class RenameRequest(BaseModel):
    title: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    model: Optional[str] = None

# ==================== Helper Functions ====================

def get_conversation_history(conv_id: str, limit: int = 10) -> List[Dict]:
    """Get recent conversation history"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT role, content FROM messages 
                 WHERE conversation_id = ? 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (conv_id, limit))
    
    messages = []
    for row in c.fetchall():
        messages.append({'role': row[0], 'content': row[1]})
    
    conn.close()
    return list(reversed(messages))

def save_message(conv_id: str, role: str, content: str):
    """Save message to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create conversation if needed
    c.execute('SELECT id FROM conversations WHERE id = ?', (conv_id,))
    if not c.fetchone():
        title = content[:50] + "..." if len(content) > 50 else content
        c.execute('''INSERT INTO conversations (id, title, created_at, updated_at)
                     VALUES (?, ?, ?, ?)''',
                  (conv_id, title, datetime.now().isoformat(), 
                   datetime.now().isoformat()))
    
    # Save message
    c.execute('''INSERT INTO messages (conversation_id, role, content, timestamp)
                 VALUES (?, ?, ?, ?)''',
              (conv_id, role, content, datetime.now().isoformat()))
    
    # Update conversation
    c.execute('UPDATE conversations SET updated_at = ? WHERE id = ?',
              (datetime.now().isoformat(), conv_id))
    
    conn.commit()
    conn.close()

# ==================== Generate CSS from Theme ====================

def generate_css():
    """Generate CSS from theme configuration"""
    return f"""
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: {FONTS['family']};
            background: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            height: 100vh;
            display: flex;
            font-size: {FONTS['size_base']};
        }}
        
        .sidebar {{
            width: {DIMENSIONS['sidebar_width']};
            background: {COLORS['bg_secondary']};
            border-right: 1px solid {COLORS['bg_tertiary']};
            display: flex;
            flex-direction: column;
        }}
        
        .sidebar-header {{
            padding: 20px;
            border-bottom: 1px solid {COLORS['bg_tertiary']};
        }}
        
        .new-chat-btn {{
            width: 100%;
            padding: 12px;
            background: {COLORS['btn_primary']};
            color: white;
            border: none;
            border-radius: {DIMENSIONS['border_radius']};
            font-size: {FONTS['size_base']};
            font-weight: 600;
            cursor: pointer;
            transition: all {ANIMATIONS['transition_speed']};
        }}
        
        .new-chat-btn:hover {{ background: {COLORS['btn_primary_hover']}; }}
        
        .conversations {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }}
        
        .conv-item {{
            padding: 12px;
            margin-bottom: 5px;
            background: {COLORS['bg_primary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .conv-item:hover {{ background: {COLORS['bg_tertiary']}; }}
        .conv-item.active {{ background: {COLORS['accent_primary']}; color: white; }}
        
        .conv-title {{
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: {FONTS['size_small']};
        }}
        
        .conv-actions {{
            display: none;
            gap: 5px;
        }}
        
        .conv-item:hover .conv-actions {{ display: flex; }}
        
        .conv-btn {{
            padding: 4px 8px;
            background: {COLORS['btn_secondary']};
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 11px;
        }}
        
        .conv-btn:hover {{ background: {COLORS['btn_secondary_hover']}; }}
        .conv-btn.delete {{ background: {COLORS['btn_danger']}; }}
        .conv-btn.delete:hover {{ background: {COLORS['btn_danger_hover']}; }}
        
        .main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }}
        
        .header {{
            padding: {DIMENSIONS['header_padding']};
            background: {COLORS['bg_secondary']};
            border-bottom: 1px solid {COLORS['bg_tertiary']};
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header-left {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .header h1 {{ font-size: {FONTS['size_large']}; font-weight: 600; }}
        
        .model-selector {{
            padding: 8px 12px;
            background: {COLORS['bg_primary']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            color: {COLORS['text_primary']};
            font-size: {FONTS['size_small']};
            cursor: pointer;
        }}
        
        .model-selector:focus {{ outline: none; border-color: {COLORS['accent_primary']}; }}
        
        .status {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: {FONTS['size_base']};
            color: {COLORS['text_secondary']};
        }}
        
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {COLORS['status_connected']};
            animation: pulse 2s infinite;
        }}
        
        .status-dot.disconnected {{ background: {COLORS['status_disconnected']}; animation: none; }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .messages {{
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .message {{
            max-width: {DIMENSIONS['message_max_width']};
            padding: 15px 20px;
            border-radius: {DIMENSIONS['border_radius']};
            line-height: 1.6;
            animation: slideIn {ANIMATIONS['slide_duration']} ease-out;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .message.user {{
            background: {COLORS['msg_user_bg']};
            color: {COLORS['msg_user_text']};
            align-self: flex-end;
        }}
        
        .message.assistant {{
            background: {COLORS['msg_assistant_bg']};
            color: {COLORS['msg_assistant_text']};
            align-self: flex-start;
            border: 1px solid {COLORS['bg_tertiary']};
        }}
        
        .message.assistant pre {{
            background: {COLORS['bg_primary']};
            padding: 12px;
            border-radius: {DIMENSIONS['border_radius_small']};
            overflow-x: auto;
            margin: 10px 0;
        }}
        
        .message.assistant code {{
            background: {COLORS['bg_primary']};
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        
        .message.assistant pre code {{
            background: transparent;
            padding: 0;
        }}
        
        .loading {{
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: {DIMENSIONS['message_max_width']};
            padding: 15px 20px;
            background: {COLORS['msg_assistant_bg']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius']};
            align-self: flex-start;
        }}
        
        .loading-spinner {{ display: flex; gap: 5px; }}
        
        .loading-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {COLORS['accent_primary']};
            animation: bounce 1.4s infinite ease-in-out both;
        }}
        
        .loading-dot:nth-child(1) {{ animation-delay: -0.32s; }}
        .loading-dot:nth-child(2) {{ animation-delay: -0.16s; }}
        
        @keyframes bounce {{
            0%, 80%, 100% {{ transform: scale(0); opacity: 0.5; }}
            40% {{ transform: scale(1); opacity: 1; }}
        }}
        
        .loading-text {{ color: {COLORS['text_secondary']}; font-size: {FONTS['size_base']}; }}
        
        .input-area {{
            padding: 20px 30px;
            background: {COLORS['bg_secondary']};
            border-top: 1px solid {COLORS['bg_tertiary']};
        }}
        
        .input-container {{ display: flex; gap: 15px; }}
        
        #input {{
            flex: 1;
            padding: {DIMENSIONS['input_padding']};
            background: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius']};
            font-size: {FONTS['size_base']};
            resize: none;
            font-family: inherit;
        }}
        
        #input:focus {{ outline: none; border-color: {COLORS['accent_primary']}; }}
        #input:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        
        #send {{
            padding: 15px 30px;
            background: {COLORS['btn_primary']};
            color: white;
            border: none;
            border-radius: {DIMENSIONS['border_radius']};
            font-size: {FONTS['size_base']};
            font-weight: 600;
            cursor: pointer;
            transition: all {ANIMATIONS['transition_speed']};
        }}
        
        #send:hover:not(:disabled) {{ background: {COLORS['btn_primary_hover']}; transform: translateY(-1px); }}
        #send:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: {COLORS['scrollbar_track']}; }}
        ::-webkit-scrollbar-thumb {{ background: {COLORS['scrollbar_thumb']}; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {COLORS['scrollbar_thumb_hover']}; }}
        
        .welcome {{
            text-align: center;
            color: {COLORS['text_secondary']};
            padding: 40px;
        }}
        
        .welcome h2 {{ font-size: 20px; margin-bottom: 10px; color: {COLORS['text_primary']}; }}
        
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
        .log-viewer-btn {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            padding: 10px 15px;
            background: {COLORS['btn_secondary']};
            color: white;
            border: none;
            border-radius: {DIMENSIONS['border_radius']};
            cursor: pointer;
            font-size: {FONTS['size_small']};
            z-index: 100;
            transition: all {ANIMATIONS['transition_speed']};
        }}
        
        .log-viewer-btn:hover {{ background: {COLORS['btn_secondary_hover']}; }}
        
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
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: {COLORS['text_primary']};
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    """

# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web interface"""
    css = generate_css()
    models_json = str(AVAILABLE_MODELS).replace("'", '"')
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chat</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        {css}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
        </div>
        <div class="conversations" id="conversations"></div>
    </div>
    
    <div class="main">
        <div class="header">
            <div class="header-left">
                <h1>AI Chat</h1>
                <select class="model-selector" id="modelSelector" onchange="updateModel()">
                    {''.join([f'<option value="{m["id"]}">{m["name"]}</option>' for m in AVAILABLE_MODELS])}
                </select>
            </div>
            <div class="status">
                <div class="status-dot disconnected" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </div>
        
        <div class="messages" id="messages">
            <div class="welcome">
                <h2>Welcome! 👋</h2>
                <p>Start a conversation with AI</p>
            </div>
        </div>
        
        <div class="input-area">
            <div class="input-container">
                <textarea 
                    id="input" 
                    placeholder="Type your message..." 
                    rows="2"
                    onkeydown="handleKeyDown(event)"
                ></textarea>
                <button id="send" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>
    
    <button class="log-viewer-btn" onclick="toggleLogViewer()">📋 Logs</button>
    
    <div class="log-viewer" id="logViewer">
        <div class="log-viewer-header">
            <h3>Terminal Output</h3>
            <button class="log-viewer-close" onclick="toggleLogViewer()">×</button>
        </div>
        <div class="log-viewer-content" id="logContent"></div>
    </div>
    
    <div class="modal" id="renameModal">
        <div class="modal-content">
            <h3>Rename Conversation</h3>
            <input type="text" id="renameInput" placeholder="Enter new title...">
            <div class="modal-buttons">
                <button class="modal-btn cancel" onclick="closeRenameModal()">Cancel</button>
                <button class="modal-btn confirm" onclick="confirmRename()">Rename</button>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let logWs = null;
        let currentConvId = generateId();
        let isGenerating = false;
        let renameConvId = null;
        let currentModel = '{DEFAULT_MODEL}';
        let markedOptions = {{
            breaks: true,
            gfm: true,
            highlight: function(code, lang) {{
                return code;
            }}
        }};
        
        function generateId() {{
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {{
                const r = Math.random() * 16 | 0;
                return (c == 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            }});
        }}
        
        function updateStatus(connected) {{
            document.getElementById('statusDot').className = 
                'status-dot ' + (connected ? '' : 'disconnected');
            document.getElementById('statusText').textContent = 
                connected ? 'Connected' : 'Disconnected';
        }}
        
        function connectWS() {{
            try {{
                ws = new WebSocket(`ws://${{location.host}}/ws/chat`);
                
                ws.onopen = () => {{
                    console.log('Connected');
                    updateStatus(true);
                }};
                
                ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'start') {{
                        removeLoading();
                        addMessage('', 'assistant');
                    }} else if (data.type === 'token') {{
                        appendToLastMessage(data.content);
                    }} else if (data.type === 'done') {{
                        isGenerating = false;
                        document.getElementById('send').disabled = false;
                        document.getElementById('input').disabled = false;
                        document.getElementById('input').focus();
                        loadConversations();
                        renderMarkdown();
                    }} else if (data.type === 'error') {{
                        removeLoading();
                        addMessage('⚠️ ' + data.content, 'assistant');
                        isGenerating = false;
                        document.getElementById('send').disabled = false;
                        document.getElementById('input').disabled = false;
                    }}
                }};
                
                ws.onerror = () => updateStatus(false);
                ws.onclose = () => {{
                    updateStatus(false);
                    setTimeout(connectWS, 2000);
                }};
            }} catch (e) {{
                setTimeout(connectWS, 2000);
            }}
        }}
        
        function connectLogWS() {{
            try {{
                logWs = new WebSocket(`ws://${{location.host}}/ws/logs`);
                
                logWs.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    if (data.type === 'log') {{
                        const logContent = document.getElementById('logContent');
                        logContent.textContent += data.content;
                        logContent.scrollTop = logContent.scrollHeight;
                    }}
                }};
                
                logWs.onerror = () => {{
                    setTimeout(connectLogWS, 2000);
                }};
                
                logWs.onclose = () => {{
                    setTimeout(connectLogWS, 2000);
                }};
            }} catch (e) {{
                setTimeout(connectLogWS, 2000);
            }}
        }}
        
        function toggleLogViewer() {{
            const viewer = document.getElementById('logViewer');
            viewer.classList.toggle('show');
            if (viewer.classList.contains('show') && !logWs) {{
                connectLogWS();
            }}
        }}
        
        function updateModel() {{
            currentModel = document.getElementById('modelSelector').value;
        }}
        
        function sendMessage() {{
            const input = document.getElementById('input');
            const message = input.value.trim();
            
            if (!message || !ws || ws.readyState !== WebSocket.OPEN || isGenerating) return;
            
            document.querySelector('.welcome')?.remove();
            
            addMessage(message, 'user');
            input.value = '';
            
            showLoading();
            
            isGenerating = true;
            document.getElementById('send').disabled = true;
            document.getElementById('input').disabled = true;
            
            ws.send(JSON.stringify({{
                message: message,
                conversation_id: currentConvId,
                model: currentModel
            }}));
        }}
        
        function addMessage(content, role) {{
            const msg = document.createElement('div');
            msg.className = `message ${{role}}`;
            msg.textContent = content;
            if (role === 'assistant') {{
                msg.dataset.needsMarkdown = 'true';
            }}
            document.getElementById('messages').appendChild(msg);
            scrollToBottom();
            return msg;
        }}
        
        function renderMarkdown() {{
            const messages = document.querySelectorAll('.message.assistant[data-needs-markdown="true"]');
            messages.forEach(msg => {{
                const content = msg.textContent;
                try {{
                    msg.innerHTML = marked.parse(content, markedOptions);
                    msg.dataset.needsMarkdown = 'false';
                    msg.dataset.rendered = 'true';
                }} catch (e) {{
                    console.error('Markdown render error:', e);
                }}
            }});
        }}
        
        function showLoading() {{
            const loading = document.createElement('div');
            loading.className = 'loading';
            loading.id = 'loadingIndicator';
            loading.innerHTML = `
                <div class="loading-spinner">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                <div class="loading-text">Thinking...</div>
            `;
            document.getElementById('messages').appendChild(loading);
            scrollToBottom();
        }}
        
        function removeLoading() {{
            document.getElementById('loadingIndicator')?.remove();
        }}
        
        function appendToLastMessage(content) {{
            const messages = document.getElementById('messages');
            const lastMsg = messages.lastElementChild;
            if (lastMsg?.classList.contains('assistant')) {{
                const currentText = lastMsg.textContent || '';
                lastMsg.textContent = currentText + content;
                scrollToBottom();
            }}
        }}
        
        function scrollToBottom() {{
            const messages = document.getElementById('messages');
            messages.scrollTop = messages.scrollHeight;
        }}
        
        function handleKeyDown(event) {{
            if (event.key === 'Enter' && !event.shiftKey) {{
                event.preventDefault();
                sendMessage();
            }}
        }}
        
        async function loadConversations() {{
            const resp = await fetch('/api/conversations');
            const data = await resp.json();
            
            const container = document.getElementById('conversations');
            container.innerHTML = '';
            
            data.conversations.forEach(conv => {{
                const item = document.createElement('div');
                item.className = 'conv-item' + (conv.id === currentConvId ? ' active' : '');
                item.innerHTML = `
                    <div class="conv-title">${{conv.title}}</div>
                    <div class="conv-actions">
                        <button class="conv-btn" onclick="event.stopPropagation(); renameConv('${{conv.id}}', '${{conv.title.replace(/'/g, "\\\\'")}}')">✏️</button>
                        <button class="conv-btn delete" onclick="event.stopPropagation(); deleteConv('${{conv.id}}')">🗑️</button>
                    </div>
                `;
                item.onclick = () => loadConversation(conv.id);
                container.appendChild(item);
            }});
        }}
        
        async function loadConversation(id) {{
            currentConvId = id;
            const resp = await fetch(`/api/conversation/${{id}}`);
            const data = await resp.json();
            
            const messages = document.getElementById('messages');
            messages.innerHTML = '';
            
            data.messages.forEach(msg => {{
                addMessage(msg.content, msg.role);
            }});
            renderMarkdown();
            loadConversations();
        }}
        
        function newChat() {{
            currentConvId = generateId();
            document.getElementById('messages').innerHTML = 
                '<div class="welcome"><h2>New Conversation</h2><p>Start chatting!</p></div>';
            loadConversations();
        }}
        
        function renameConv(id, currentTitle) {{
            renameConvId = id;
            document.getElementById('renameInput').value = currentTitle;
            document.getElementById('renameModal').classList.add('show');
            document.getElementById('renameInput').focus();
        }}
        
        function closeRenameModal() {{
            document.getElementById('renameModal').classList.remove('show');
            renameConvId = null;
        }}
        
        async function confirmRename() {{
            const newTitle = document.getElementById('renameInput').value.trim();
            if (!newTitle || !renameConvId) return;
            
            await fetch(`/api/conversation/${{renameConvId}}/rename`, {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{title: newTitle}})
            }});
            
            closeRenameModal();
            loadConversations();
        }}
        
        async function deleteConv(id) {{
            if (!confirm('Delete this conversation?')) return;
            
            await fetch(`/api/conversation/${{id}}`, {{method: 'DELETE'}});
            
            if (id === currentConvId) {{
                newChat();
            }} else {{
                loadConversations();
            }}
        }}
        
        connectWS();
        loadConversations();
        document.getElementById('input').focus();
        document.getElementById('modelSelector').value = currentModel;
    </script>
</body>
</html>
    """

@app.get("/api/models")
async def get_models():
    """Get available models"""
    return {"models": AVAILABLE_MODELS}

@app.get("/api/conversations")
async def get_conversations():
    """Get all conversations"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT id, title, created_at, updated_at 
                     FROM conversations 
                     ORDER BY updated_at DESC''')
        
        conversations = []
        for row in c.fetchall():
            conversations.append({
                'id': row[0],
                'title': row[1],
                'created_at': row[2],
                'updated_at': row[3]
            })
        
        conn.close()
        return {'conversations': conversations}
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return {'conversations': []}

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation messages"""
    try:
        history = get_conversation_history(conversation_id, limit=100)
        return {'messages': history}
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return {'messages': []}

@app.post("/api/conversation/{conversation_id}/rename")
async def rename_conversation(conversation_id: str, request: RenameRequest):
    """Rename a conversation"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('UPDATE conversations SET title = ? WHERE id = ?',
                  (request.title, conversation_id))
        conn.commit()
        conn.close()
        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Error renaming conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        conn.commit()
        conn.close()
        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket for streaming chat"""
    await websocket.accept()
    logger.info("WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            message = data.get('message', '').strip()
            conv_id = data.get('conversation_id')
            model = data.get('model', DEFAULT_MODEL)
            
            if not message:
                await websocket.send_json({'type': 'error', 'content': 'Empty message'})
                continue
            
            logger.info(f"Received message: {message[:50]}... (model: {model})")
            
            try:
                # Save user message
                save_message(conv_id, 'user', message)
                
                # Get history
                history = get_conversation_history(conv_id)
                
                # Build messages
                messages = []
                for msg in history[-10:]:
                    messages.append({'role': msg['role'], 'content': msg['content']})
                messages.append({'role': 'user', 'content': message})
                
                # Signal start
                await websocket.send_json({'type': 'start'})
                
                # Stream from vLLM
                full_response = ""
                
                logger.info(f"Calling vLLM with model: {model}")
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=2048,
                    temperature=0.7,
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        
                        await websocket.send_json({
                            'type': 'token',
                            'content': token
                        })
                
                logger.info(f"Generated {len(full_response)} chars")
                
                # Save response
                save_message(conv_id, 'assistant', full_response)
                
                await websocket.send_json({'type': 'done'})
                
            except Exception as e:
                logger.error(f"Generation error: {e}\n{traceback.format_exc()}")
                await websocket.send_json({
                    'type': 'error',
                    'content': f'Error: {str(e)}. Check vLLM is running.'
                })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")

@app.websocket("/ws/logs")
async def logs_websocket(websocket: WebSocket):
    """WebSocket for streaming terminal logs"""
    await websocket.accept()
    logger.info("Log WebSocket connected")
    
    try:
        # Send existing logs
        existing_logs = log_capture.getvalue()
        if existing_logs:
            await websocket.send_json({
                'type': 'log',
                'content': existing_logs
            })
        
        # Keep connection alive and send new logs
        import asyncio
        last_size = len(existing_logs)
        
        while True:
            await asyncio.sleep(0.5)
            current_logs = log_capture.getvalue()
            if len(current_logs) > last_size:
                new_content = current_logs[last_size:]
                await websocket.send_json({
                    'type': 'log',
                    'content': new_content
                })
                last_size = len(current_logs)
            
    except WebSocketDisconnect:
        logger.info("Log WebSocket disconnected")
    except Exception as e:
        logger.error(f"Log WebSocket error: {e}")

@app.get("/health")
async def health():
    """Health check"""
    try:
        # Test vLLM connection
        models = client.models.list()
        return {
            "status": "healthy",
            "vllm_connected": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "vllm_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Chat with vLLM...")
    logger.info("📍 http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
