#!/usr/bin/env python3
"""
AI Chat with Conversation Memory - vLLM Backend
Optimized for GPU servers with better throughput
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List, Dict
import sqlite3
from datetime import datetime
import logging
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "/app/data/chat.db"
VLLM_HOST = "http://vllm:8000/v1"

app = FastAPI(title="AI Chat with vLLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client for vLLM
client = OpenAI(
    base_url=VLLM_HOST,
    api_key="dummy"  # vLLM doesn't require real API key
)

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
    <title>AI Chat - vLLM</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #f1f5f9;
            height: 100vh;
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
        
        .header h1 {
            font-size: 24px;
            font-weight: 600;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #94a3b8;
        }
        
        .badge {
            background: #10b981;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }
        
        .status-dot.disconnected {
            background: #ef4444;
            animation: none;
        }
        
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
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
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
            border-bottom-left-radius: 4px;
        }
        
        .loading-spinner {
            display: flex;
            gap: 5px;
        }
        
        .loading-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        
        .loading-dot:nth-child(1) {
            animation-delay: -0.32s;
        }
        
        .loading-dot:nth-child(2) {
            animation-delay: -0.16s;
        }
        
        @keyframes bounce {
            0%, 80%, 100% {
                transform: scale(0);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }
        
        .loading-text {
            color: #94a3b8;
            font-size: 14px;
        }
        
        .input-area {
            padding: 20px 30px;
            background: #1e293b;
            border-top: 1px solid #334155;
        }
        
        .input-container {
            display: flex;
            gap: 15px;
        }
        
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
        
        #input:focus {
            outline: none;
            border-color: #10b981;
        }
        
        #input:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
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
        
        #send:hover:not(:disabled) {
            background: #059669;
            transform: translateY(-1px);
        }
        
        #send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1e293b;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }
        
        .welcome {
            text-align: center;
            color: #94a3b8;
            padding: 40px;
        }
        
        .welcome h2 {
            font-size: 20px;
            margin-bottom: 10px;
            color: #f1f5f9;
        }
        
        .welcome .feature {
            display: inline-block;
            margin: 5px;
            padding: 6px 12px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 6px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            🚀 AI Chat
            <span class="badge">vLLM Powered</span>
        </h1>
        <div class="status">
            <div class="status-dot disconnected" id="statusDot"></div>
            <span id="statusText">Connecting...</span>
        </div>
    </div>
    
    <div class="messages" id="messages">
        <div class="welcome">
            <h2>⚡ GPU-Accelerated AI Chat</h2>
            <p style="margin: 15px 0;">Powered by vLLM for maximum performance</p>
            <div>
                <span class="feature">🔥 Fast Inference</span>
                <span class="feature">💾 Conversation Memory</span>
                <span class="feature">🎯 High Throughput</span>
            </div>
        </div>
    </div>
    
    <div class="input-area">
        <div class="input-container">
            <textarea 
                id="input" 
                placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)" 
                rows="2"
                onkeydown="handleKeyDown(event)"
            ></textarea>
            <button id="send" onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        let ws = null;
        let currentConvId = generateId();
        let isGenerating = false;
        
        function generateId() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }
        
        function updateStatus(connected) {
            const dot = document.getElementById('statusDot');
            const text = document.getElementById('statusText');
            
            if (connected) {
                dot.classList.remove('disconnected');
                text.textContent = 'Connected';
            } else {
                dot.classList.add('disconnected');
                text.textContent = 'Disconnected';
            }
        }
        
        function connectWS() {
            try {
                ws = new WebSocket(`ws://${location.host}/ws/chat`);
                
                ws.onopen = () => {
                    console.log('Connected');
                    updateStatus(true);
                };
                
                ws.onmessage = (event) => {
                    try {
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
                        } else if (data.type === 'error') {
                            removeLoading();
                            addMessage('⚠️ Error: ' + data.content, 'assistant');
                            isGenerating = false;
                            document.getElementById('send').disabled = false;
                            document.getElementById('input').disabled = false;
                        }
                    } catch (e) {
                        console.error('Error:', e);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    updateStatus(false);
                };
                
                ws.onclose = () => {
                    console.log('Disconnected');
                    updateStatus(false);
                    setTimeout(connectWS, 2000);
                };
            } catch (e) {
                console.error('Connection error:', e);
                setTimeout(connectWS, 2000);
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            
            if (!message || !ws || ws.readyState !== WebSocket.OPEN || isGenerating) {
                return;
            }
            
            const welcome = document.querySelector('.welcome');
            if (welcome) welcome.remove();
            
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
            const messages = document.getElementById('messages');
            const msg = document.createElement('div');
            msg.className = `message ${role}`;
            msg.textContent = content;
            messages.appendChild(msg);
            messages.scrollTop = messages.scrollHeight;
            return msg;
        }
        
        function showLoading() {
            const messages = document.getElementById('messages');
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
            messages.appendChild(loading);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function removeLoading() {
            const loading = document.getElementById('loadingIndicator');
            if (loading) loading.remove();
        }
        
        function appendToLastMessage(content) {
            const messages = document.getElementById('messages');
            const lastMsg = messages.lastElementChild;
            if (lastMsg && lastMsg.classList.contains('assistant')) {
                lastMsg.textContent += content;
                messages.scrollTop = messages.scrollHeight;
            }
        }
        
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        connectWS();
        document.getElementById('input').focus();
    </script>
</body>
</html>
    """

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
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
            
            logger.info(f"Received: {message[:50]}...")
            
            # Save user message
            save_message(conv_id, 'user', message)
            
            # Get conversation history
            history = get_conversation_history(conv_id)
            
            # Build messages for vLLM
            messages = []
            for msg in history[-10:]:
                messages.append({'role': msg['role'], 'content': msg['content']})
            
            messages.append({'role': 'user', 'content': message})
            
            try:
                # Signal start
                await websocket.send_json({'type': 'start'})
                
                # Stream from vLLM
                full_response = ""
                
                stream = client.chat.completions.create(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=messages,
                    stream=True,
                    max_tokens=2048,
                    temperature=0.7,
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        
                        # Send token to client
                        await websocket.send_json({
                            'type': 'token',
                            'content': token
                        })
                
                # Save assistant message
                save_message(conv_id, 'assistant', full_response)
                
                # Signal completion
                await websocket.send_json({'type': 'done'})
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                await websocket.send_json({
                    'type': 'error',
                    'content': f'Failed to generate response: {str(e)}'
                })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Chat with vLLM...")
    logger.info("📍 http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
