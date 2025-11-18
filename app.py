#!/usr/bin/env python3
"""
AI Chat with vLLM - Full Featured
- Word-by-word streaming
- Conversation management (delete, rename)
- Chat history sidebar
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "/app/data/chat.db"
VLLM_HOST = "http://vllm:8000/v1"

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

# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #f1f5f9;
            height: 100vh;
            display: flex;
        }
        
        .sidebar {
            width: 280px;
            background: #1e293b;
            border-right: 1px solid #334155;
            display: flex;
            flex-direction: column;
        }
        
        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #334155;
        }
        
        .new-chat-btn {
            width: 100%;
            padding: 12px;
            background: #10b981;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        }
        
        .new-chat-btn:hover { background: #059669; }
        
        .conversations {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        
        .conv-item {
            padding: 12px;
            margin-bottom: 5px;
            background: #0f172a;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
        }
        
        .conv-item:hover { background: #334155; }
        .conv-item.active { background: #10b981; color: white; }
        
        .conv-title {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 13px;
        }
        
        .conv-actions {
            display: none;
            gap: 5px;
        }
        
        .conv-item:hover .conv-actions { display: flex; }
        
        .conv-btn {
            padding: 4px 8px;
            background: #334155;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 11px;
        }
        
        .conv-btn:hover { background: #475569; }
        .conv-btn.delete { background: #ef4444; }
        .conv-btn.delete:hover { background: #dc2626; }
        
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            padding: 20px 30px;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 { font-size: 24px; font-weight: 600; }
        
        .badge {
            background: #10b981;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #94a3b8;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }
        
        .status-dot.disconnected { background: #ef4444; animation: none; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .message {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 12px;
            line-height: 1.6;
            animation: slideIn 0.3s ease-out;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: #2563eb;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant {
            background: #1e293b;
            color: #f1f5f9;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            border: 1px solid #334155;
        }
        
        .loading {
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: 70%;
            padding: 15px 20px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            align-self: flex-start;
        }
        
        .loading-spinner { display: flex; gap: 5px; }
        
        .loading-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        
        .loading-dot:nth-child(1) { animation-delay: -0.32s; }
        .loading-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        
        .loading-text { color: #94a3b8; font-size: 14px; }
        
        .input-area {
            padding: 20px 30px;
            background: #1e293b;
            border-top: 1px solid #334155;
        }
        
        .input-container { display: flex; gap: 15px; }
        
        #input {
            flex: 1;
            padding: 15px;
            background: #0f172a;
            color: #f1f5f9;
            border: 1px solid #334155;
            border-radius: 8px;
            font-size: 14px;
            resize: none;
            font-family: inherit;
        }
        
        #input:focus { outline: none; border-color: #10b981; }
        #input:disabled { opacity: 0.5; cursor: not-allowed; }
        
        #send {
            padding: 15px 30px;
            background: #10b981;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        #send:hover:not(:disabled) { background: #059669; transform: translateY(-1px); }
        #send:disabled { opacity: 0.5; cursor: not-allowed; }
        
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1e293b; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
        
        .welcome {
            text-align: center;
            color: #94a3b8;
            padding: 40px;
        }
        
        .welcome h2 { font-size: 20px; margin-bottom: 10px; color: #f1f5f9; }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        
        .modal.show { display: flex; }
        
        .modal-content {
            background: #1e293b;
            padding: 30px;
            border-radius: 12px;
            min-width: 400px;
            border: 1px solid #334155;
        }
        
        .modal-content h3 { margin-bottom: 20px; }
        
        .modal-content input {
            width: 100%;
            padding: 12px;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            color: #f1f5f9;
            margin-bottom: 20px;
        }
        
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        
        .modal-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
        }
        
        .modal-btn.cancel {
            background: #334155;
            color: white;
        }
        
        .modal-btn.confirm {
            background: #10b981;
            color: white;
        }
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
            <h1>🚀 AI Chat <span class="badge">vLLM</span></h1>
            <div class="status">
                <div class="status-dot disconnected" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </div>
        
        <div class="messages" id="messages">
            <div class="welcome">
                <h2>Welcome! 👋</h2>
                <p>GPU-accelerated AI chat with conversation memory</p>
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
        let currentConvId = generateId();
        let isGenerating = false;
        let renameConvId = null;
        
        function generateId() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
                const r = Math.random() * 16 | 0;
                return (c == 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            });
        }
        
        function updateStatus(connected) {
            document.getElementById('statusDot').className = 
                'status-dot ' + (connected ? '' : 'disconnected');
            document.getElementById('statusText').textContent = 
                connected ? 'Connected' : 'Disconnected';
        }
        
        function connectWS() {
            try {
                ws = new WebSocket(`ws://${location.host}/ws/chat`);
                
                ws.onopen = () => {
                    console.log('Connected');
                    updateStatus(true);
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'start') {
                        removeLoading();
                        addMessage('', 'assistant');
                    } else if (data.type === 'token') {
                        appendToLastMessage(data.content);
                    } else if (data.type === 'done') {
                        isGenerating = false;
                        document.getElementById('send').disabled = false;
                        document.getElementById('input').disabled = false;
                        document.getElementById('input').focus();
                        loadConversations();
                    } else if (data.type === 'error') {
                        removeLoading();
                        addMessage('⚠️ ' + data.content, 'assistant');
                        isGenerating = false;
                        document.getElementById('send').disabled = false;
                        document.getElementById('input').disabled = false;
                    }
                };
                
                ws.onerror = () => updateStatus(false);
                ws.onclose = () => {
                    updateStatus(false);
                    setTimeout(connectWS, 2000);
                };
            } catch (e) {
                setTimeout(connectWS, 2000);
            }
        }
        
        function sendMessage() {
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
            
            ws.send(JSON.stringify({
                message: message,
                conversation_id: currentConvId
            }));
        }
        
        function addMessage(content, role) {
            const msg = document.createElement('div');
            msg.className = `message ${role}`;
            msg.textContent = content;
            document.getElementById('messages').appendChild(msg);
            scrollToBottom();
            return msg;
        }
        
        function showLoading() {
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
        }
        
        function removeLoading() {
            document.getElementById('loadingIndicator')?.remove();
        }
        
        function appendToLastMessage(content) {
            const messages = document.getElementById('messages');
            const lastMsg = messages.lastElementChild;
            if (lastMsg?.classList.contains('assistant')) {
                lastMsg.textContent += content;
                scrollToBottom();
            }
        }
        
        function scrollToBottom() {
            const messages = document.getElementById('messages');
            messages.scrollTop = messages.scrollHeight;
        }
        
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        async function loadConversations() {
            const resp = await fetch('/api/conversations');
            const data = await resp.json();
            
            const container = document.getElementById('conversations');
            container.innerHTML = '';
            
            data.conversations.forEach(conv => {
                const item = document.createElement('div');
                item.className = 'conv-item' + (conv.id === currentConvId ? ' active' : '');
                item.innerHTML = `
                    <div class="conv-title">${conv.title}</div>
                    <div class="conv-actions">
                        <button class="conv-btn" onclick="event.stopPropagation(); renameConv('${conv.id}', '${conv.title.replace(/'/g, "\\'")}')">✏️</button>
                        <button class="conv-btn delete" onclick="event.stopPropagation(); deleteConv('${conv.id}')">🗑️</button>
                    </div>
                `;
                item.onclick = () => loadConversation(conv.id);
                container.appendChild(item);
            });
        }
        
        async function loadConversation(id) {
            currentConvId = id;
            const resp = await fetch(`/api/conversation/${id}`);
            const data = await resp.json();
            
            const messages = document.getElementById('messages');
            messages.innerHTML = '';
            
            data.messages.forEach(msg => addMessage(msg.content, msg.role));
            loadConversations();
        }
        
        function newChat() {
            currentConvId = generateId();
            document.getElementById('messages').innerHTML = 
                '<div class="welcome"><h2>New Conversation</h2><p>Start chatting!</p></div>';
            loadConversations();
        }
        
        function renameConv(id, currentTitle) {
            renameConvId = id;
            document.getElementById('renameInput').value = currentTitle;
            document.getElementById('renameModal').classList.add('show');
            document.getElementById('renameInput').focus();
        }
        
        function closeRenameModal() {
            document.getElementById('renameModal').classList.remove('show');
            renameConvId = null;
        }
        
        async function confirmRename() {
            const newTitle = document.getElementById('renameInput').value.trim();
            if (!newTitle || !renameConvId) return;
            
            await fetch(`/api/conversation/${renameConvId}/rename`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({title: newTitle})
            });
            
            closeRenameModal();
            loadConversations();
        }
        
        async function deleteConv(id) {
            if (!confirm('Delete this conversation?')) return;
            
            await fetch(`/api/conversation/${id}`, {method: 'DELETE'});
            
            if (id === currentConvId) {
                newChat();
            } else {
                loadConversations();
            }
        }
        
        connectWS();
        loadConversations();
        document.getElementById('input').focus();
    </script>
</body>
</html>
    """

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
            
            if not message:
                await websocket.send_json({'type': 'error', 'content': 'Empty message'})
                continue
            
            logger.info(f"Received message: {message[:50]}...")
            
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
                
                logger.info("Calling vLLM...")
                stream = client.chat.completions.create(
                    model="Qwen/Qwen2.5-Coder-3B-Instruct",
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
