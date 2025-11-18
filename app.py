#!/usr/bin/env python3
"""
AI Chat with RAG & Web Search - Complete Docker Version
Includes: RAG, Web Search, Multi-model support, SQLite storage
"""

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import sqlite3
import ollama
import numpy as np
from datetime import datetime
import uuid
import aiohttp
from bs4 import BeautifulSoup
import PyPDF2
import io
import os

# ==================== Configuration ====================

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
DATA_DIR = os.getenv('DATA_DIR', '/app/data')
DB_PATH = os.path.join(DATA_DIR, 'chat.db')
MODELS_CACHE = os.getenv('TRANSFORMERS_CACHE', '/app/data/models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE, exist_ok=True)

print(f"📁 Data directory: {DATA_DIR}")
print(f"🗄️  Database: {DB_PATH}")
print(f"🤖 Ollama host: {OLLAMA_HOST}")

app = FastAPI(title="AI Chat with RAG & Search")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama client
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Lazy load embedding model
embedder = None

def get_embedder():
    """Lazy load embedding model"""
    global embedder
    if embedder is None:
        print("📥 Loading embedding model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODELS_CACHE)
        print("✓ Embedding model loaded")
    return embedder

# ==================== Database Setup ====================

def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initialize SQLite database"""
    conn = get_db()
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id TEXT PRIMARY KEY,
                  title TEXT,
                  model TEXT,
                  created_at TEXT,
                  updated_at TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conversation_id TEXT,
                  role TEXT,
                  content TEXT,
                  timestamp TEXT,
                  tokens INTEGER,
                  used_rag BOOLEAN DEFAULT 0,
                  used_search BOOLEAN DEFAULT 0,
                  FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  content TEXT,
                  chunk_index INTEGER,
                  embedding BLOB,
                  created_at TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS search_cache
                 (query TEXT PRIMARY KEY,
                  results TEXT,
                  timestamp TEXT)''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized")

init_db()

# ==================== Models ====================

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: str = "qwen2.5-coder:7b"
    use_rag: bool = True
    use_search: bool = False

# ==================== RAG Functions ====================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def add_document_to_rag(filename: str, content: str) -> dict:
    """Add document to RAG database"""
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(content)
    
    model = get_embedder()
    conn = get_db()
    c = conn.cursor()
    
    added_chunks = 0
    for idx, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue
            
        embedding = model.encode(chunk)
        
        c.execute('''INSERT INTO documents 
                     (id, filename, content, chunk_index, embedding, created_at)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (f"{doc_id}_{idx}", filename, chunk, idx, 
                   embedding.tobytes(), datetime.now().isoformat()))
        added_chunks += 1
    
    conn.commit()
    conn.close()
    
    return {
        'document_id': doc_id,
        'filename': filename,
        'chunks': added_chunks,
        'total_chars': len(content)
    }

def search_rag(query: str, top_k: int = 5) -> List[dict]:
    """Search documents using semantic similarity"""
    model = get_embedder()
    query_embedding = model.encode(query)
    
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, filename, content, embedding FROM documents')
    
    results = []
    for row in c.fetchall():
        doc_id, filename, content, embedding_bytes = row
        doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        results.append({
            'doc_id': doc_id,
            'filename': filename,
            'content': content,
            'similarity': float(similarity)
        })
    
    conn.close()
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def get_rag_context(query: str, top_k: int = 3) -> str:
    """Get RAG context for a query"""
    results = search_rag(query, top_k)
    
    if not results:
        return ""
    
    context_parts = []
    for i, result in enumerate(results, 1):
        if result['similarity'] > 0.3:
            context_parts.append(
                f"[Source {i}: {result['filename']}]\n{result['content']}"
            )
    
    if not context_parts:
        return ""
    
    return "\n\n---\n\n".join(context_parts)

# ==================== Web Search Functions ====================

async def web_search(query: str, num_results: int = 5) -> List[dict]:
    """Search the web using DuckDuckGo"""
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT results, timestamp FROM search_cache WHERE query = ?', (query,))
        cached = c.fetchone()
        
        if cached:
            import json
            timestamp = datetime.fromisoformat(cached[1])
            age = (datetime.now() - timestamp).total_seconds()
            if age < 3600:
                conn.close()
                return json.loads(cached[0])
        
        async with aiohttp.ClientSession() as session:
            url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            async with session.get(url, headers=headers, timeout=10) as response:
                html = await response.text()
        
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        for result_div in soup.find_all('div', class_='result')[:num_results]:
            try:
                title_elem = result_div.find('a', class_='result__a')
                snippet_elem = result_div.find('a', class_='result__snippet')
                url_elem = result_div.find('a', class_='result__url')
                
                if title_elem and snippet_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'snippet': snippet_elem.get_text(strip=True),
                        'url': url_elem.get_text(strip=True) if url_elem else ''
                    })
            except:
                continue
        
        import json
        c.execute('''INSERT OR REPLACE INTO search_cache (query, results, timestamp)
                     VALUES (?, ?, ?)''',
                  (query, json.dumps(results), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def format_search_results(results: List[dict]) -> str:
    """Format search results for LLM"""
    if not results:
        return ""
    
    formatted = ["Recent web search results:\n"]
    for i, result in enumerate(results, 1):
        formatted.append(f"{i}. {result['title']}")
        formatted.append(f"   {result['snippet']}")
        if result.get('url'):
            formatted.append(f"   URL: {result['url']}")
        formatted.append("")
    
    return "\n".join(formatted)

# ==================== Database Functions ====================

def get_conversation_history(conv_id: str) -> List[dict]:
    """Get messages from conversation"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT role, content FROM messages 
                 WHERE conversation_id = ? 
                 ORDER BY timestamp ASC''', (conv_id,))
    messages = [{'role': row[0], 'content': row[1]} for row in c.fetchall()]
    conn.close()
    return messages

def save_message(conv_id: str, role: str, content: str, 
                 used_rag: bool = False, used_search: bool = False):
    """Save message to database"""
    conn = get_db()
    c = conn.cursor()
    
    c.execute('SELECT id FROM conversations WHERE id = ?', (conv_id,))
    if not c.fetchone():
        title = content[:50] + "..." if len(content) > 50 else content
        c.execute('''INSERT INTO conversations 
                     (id, title, model, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?)''',
                  (conv_id, title, 'qwen2.5-coder:7b', 
                   datetime.now().isoformat(), datetime.now().isoformat()))
    
    c.execute('''INSERT INTO messages 
                 (conversation_id, role, content, timestamp, used_rag, used_search)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (conv_id, role, content, datetime.now().isoformat(), 
               used_rag, used_search))
    
    c.execute('UPDATE conversations SET updated_at = ? WHERE id = ?',
              (datetime.now().isoformat(), conv_id))
    
    conn.commit()
    conn.close()

def list_conversations():
    """List all conversations"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT id, title, model, updated_at 
                 FROM conversations 
                 ORDER BY updated_at DESC LIMIT 50''')
    convs = [{'id': row[0], 'title': row[1], 'model': row[2], 'updated_at': row[3]} 
             for row in c.fetchall()]
    conn.close()
    return convs

def list_documents():
    """List all uploaded documents"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT DISTINCT filename, COUNT(*) as chunks, 
                 MIN(created_at) as uploaded_at
                 FROM documents 
                 GROUP BY filename 
                 ORDER BY uploaded_at DESC''')
    docs = [{'filename': row[0], 'chunks': row[1], 'uploaded_at': row[2]} 
            for row in c.fetchall()]
    conn.close()
    return docs

# ==================== API Endpoints ====================
@app.get("/health")
async def health():
    """Health check - basic liveness probe"""
    return {'status': 'healthy'}

@app.get("/health/full")
async def health_full():
    """Full health check including Ollama"""
    try:
        models = ollama_client.list()
        return {
            'status': 'healthy',
            'ollama': 'connected',
            'models': len(models.get('models', []))
        }
    except Exception as e:
        return {
            'status': 'degraded',
            'ollama': 'disconnected',
            'error': str(e)
        }

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket for streaming chat"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            conv_id = data.get('conversation_id') or str(uuid.uuid4())
            message = data['message']
            model = data.get('model', 'qwen2.5-coder:7b')
            use_rag = data.get('use_rag', True)
            use_search = data.get('use_search', False)
            
            save_message(conv_id, 'user', message)
            history = get_conversation_history(conv_id)
            
            context_parts = []
            
            if use_rag:
                rag_context = get_rag_context(message)
                if rag_context:
                    context_parts.append(f"=== RELEVANT DOCUMENTS ===\n{rag_context}\n")
            
            if use_search:
                search_results = await web_search(message)
                if search_results:
                    search_context = format_search_results(search_results)
                    context_parts.append(f"=== WEB SEARCH RESULTS ===\n{search_context}\n")
            
            if context_parts:
                enhanced_message = (
                    f"{' '.join(context_parts)}\n"
                    f"=== USER QUESTION ===\n{message}\n\n"
                    f"Please answer based on the context provided."
                )
                history[-1]['content'] = enhanced_message
            
            full_response = ""
            stream = ollama_client.chat(
                model=model,
                messages=history,
                stream=True
            )
            
            for chunk in stream:
                content = chunk['message']['content']
                full_response += content
                await websocket.send_json({
                    'type': 'chunk',
                    'content': content,
                    'conversation_id': conv_id
                })
            
            save_message(conv_id, 'assistant', full_response, use_rag, use_search)
            
            await websocket.send_json({
                'type': 'done',
                'conversation_id': conv_id,
                'used_rag': use_rag and bool(context_parts),
                'used_search': use_search
            })
            
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document for RAG"""
    try:
        content_bytes = await file.read()
        
        if file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
        elif file.filename.endswith(('.txt', '.md')):
            content = content_bytes.decode('utf-8')
        else:
            return {'error': 'Unsupported file type. Use .txt, .md, or .pdf'}
        
        result = add_document_to_rag(file.filename, content)
        
        return {
            'status': 'success',
            'message': f"Added {result['chunks']} chunks from {file.filename}",
            'details': result
        }
        
    except Exception as e:
        return {'error': str(e)}

@app.post("/api/add_text")
async def add_text(data: dict):
    """Add raw text to RAG"""
    try:
        result = add_document_to_rag(
            data.get('filename', 'text_input.txt'),
            data['content']
        )
        return {
            'status': 'success',
            'message': f"Added {result['chunks']} chunks",
            'details': result
        }
    except Exception as e:
        return {'error': str(e)}

@app.get("/api/documents")
async def get_documents():
    """List documents"""
    return {'documents': list_documents()}

@app.delete("/api/document/{filename}")
async def delete_document(filename: str):
    """Delete document"""
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM documents WHERE filename = ?', (filename,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return {'status': 'deleted', 'chunks_removed': deleted}

@app.get("/api/conversations")
async def get_conversations():
    """List conversations"""
    return {'conversations': list_conversations()}

@app.get("/api/conversation/{conv_id}")
async def get_conversation(conv_id: str):
    """Get conversation"""
    messages = get_conversation_history(conv_id)
    return {'messages': messages}

@app.delete("/api/conversation/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete conversation"""
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE conversation_id = ?', (conv_id,))
    c.execute('DELETE FROM conversations WHERE id = ?', (conv_id,))
    conn.commit()
    conn.close()
    return {'status': 'deleted'}

@app.get("/api/models")
async def get_models():
    """Get available models"""
    try:
        models = ollama_client.list()
        return {'models': [m['name'] for m in models.get('models', [])]}
    except Exception as e:
        return {'models': [], 'error': str(e)}

# ==================== Web UI ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chat with RAG & Search</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
        }
        .sidebar {
            width: 260px;
            background: #111;
            border-right: 1px solid #222;
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid #222;
        }
        .new-chat-btn, .upload-btn {
            width: 100%;
            padding: 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #fff;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .new-chat-btn:hover, .upload-btn:hover { background: #222; }
        .sidebar-section {
            padding: 16px;
            border-bottom: 1px solid #222;
            max-height: 200px;
            overflow-y: auto;
        }
        .section-title {
            font-size: 12px;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .conversations {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        .conv-item, .doc-item {
            padding: 10px;
            margin: 4px 0;
            background: #1a1a1a;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
        }
        .conv-item:hover, .doc-item:hover { background: #222; }
        .conv-item.active { background: #2a2a2a; border-left: 3px solid #0084ff; }
        .doc-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .doc-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .doc-delete {
            color: #ff4444;
            cursor: pointer;
            padding: 4px 8px;
            margin-left: 8px;
        }
        .doc-delete:hover { color: #ff6666; }
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .header {
            padding: 16px 24px;
            background: #111;
            border-bottom: 1px solid #222;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 18px; font-weight: 600; }
        .controls {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
        }
        .toggle input { cursor: pointer; }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }
        .message {
            max-width: 700px;
            margin: 0 auto 20px;
            padding: 16px;
            border-radius: 8px;
            line-height: 1.6;
        }
        .message.user {
            background: #0084ff;
            color: white;
            margin-left: auto;
        }
        .message.assistant {
            background: #1a1a1a;
            border: 1px solid #222;
            white-space: pre-wrap;
        }
        .message-meta {
            font-size: 11px;
            color: #00ff88;
            margin-top: 8px;
            opacity: 0.7;
        }
        .input-area {
            padding: 24px;
            background: #111;
            border-top: 1px solid #222;
        }
        .input-container {
            max-width: 700px;
            margin: 0 auto;
            display: flex;
            gap: 12px;
        }
        textarea {
            flex: 1;
            padding: 12px 16px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
            font-family: inherit;
            resize: none;
            outline: none;
        }
        textarea:focus { border-color: #0084ff; }
        button {
            padding: 12px 24px;
            background: #0084ff;
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover { background: #0073e6; }
        button:disabled { background: #333; cursor: not-allowed; }
        select {
            padding: 8px 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #fff;
            border-radius: 6px;
            font-size: 13px;
        }
        input[type="file"] { display: none; }
        .no-docs {
            color: #666;
            font-size: 12px;
            padding: 8px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                📄 Upload Document
            </button>
            <input type="file" id="fileInput" accept=".txt,.md,.pdf" onchange="uploadFile(event)">
        </div>
        
        <div class="sidebar-section">
            <div class="section-title">Documents (RAG)</div>
            <div id="documents"><div class="no-docs">No documents uploaded</div></div>
        </div>
        
        <div class="conversations" id="conversations"></div>
    </div>
    
    <div class="main">
        <div class="header">
            <h1>AI Chat with RAG & Search</h1>
            <div class="controls">
                <label class="toggle">
                    <input type="checkbox" id="useRag" checked>
                    <span>📚 RAG</span>
                </label>
                <label class="toggle">
                    <input type="checkbox" id="useSearch">
                    <span>🔍 Search</span>
                </label>
                <select id="model">
                    <option value="qwen2.5-coder:7b">Qwen Coder 7B</option>
                    <option value="mistral:7b">Mistral 7B</option>
                    <option value="llama3.2:3b">Llama 3.2 3B</option>
                </select>
            </div>
        </div>
        
        <div class="messages" id="messages"></div>
        
        <div class="input-area">
            <div class="input-container">
                <textarea 
                    id="input" 
                    placeholder="Ask me anything... (RAG will search your uploaded documents)" 
                    rows="3"
                    onkeydown="handleKeyDown(event)"
                ></textarea>
                <button id="send" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        let currentConvId = null;
        let ws = null;
        
        function connectWS() {
            ws = new WebSocket(`ws://${location.host}/ws/chat`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'chunk') {
                    currentConvId = data.conversation_id;
                    appendToLastMessage(data.content);
                } else if (data.type === 'done') {
                    if (data.used_rag || data.used_search) {
                        const meta = [];
                        if (data.used_rag) meta.push('📚 Used RAG');
                        if (data.used_search) meta.push('🔍 Used Web Search');
                        appendMetaToLastMessage(meta.join(' • '));
                    }
                    document.getElementById('send').disabled = false;
                    loadConversations();
                }
            };
            
            ws.onerror = () => setTimeout(connectWS, 1000);
            ws.onclose = () => setTimeout(connectWS, 1000);
        }
        
        connectWS();
        
        function sendMessage() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            document.getElementById('send').disabled = true;
            
            addMessage('', 'assistant');
            
            ws.send(JSON.stringify({
                message: message,
                conversation_id: currentConvId,
                model: document.getElementById('model').value,
                use_rag: document.getElementById('useRag').checked,
                use_search: document.getElementById('useSearch').checked
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
        
        function appendToLastMessage(content) {
            const messages = document.getElementById('messages');
            const lastMsg = messages.lastElementChild;
            if (lastMsg) {
                lastMsg.textContent += content;
                messages.scrollTop = messages.scrollHeight;
            }
        }
        
        function appendMetaToLastMessage(meta) {
            const messages = document.getElementById('messages');
            const lastMsg = messages.lastElementChild;
            if (lastMsg) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'message-meta';
                metaDiv.textContent = meta;
                lastMsg.appendChild(metaDiv);
            }
        }
        
        async function uploadFile(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const resp = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await resp.json();
                
                if (data.status === 'success') {
                    alert(data.message);
                    loadDocuments();
                } else {
                    alert('Error: ' + (data.error || 'Upload failed'));
                }
            } catch (e) {
                alert('Upload error: ' + e.message);
            }
            
            event.target.value = '';
        }
        
        async function loadDocuments() {
            const resp = await fetch('/api/documents');
            const data = await resp.json();
            
            const container = document.getElementById('documents');
            container.innerHTML = '';
            
            if (data.documents.length === 0) {
                container.innerHTML = '<div class="no-docs">No documents uploaded</div>';
                return;
            }
            
            data.documents.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'doc-item';
                item.innerHTML = `
                    <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
                    <span class="doc-delete" onclick="deleteDoc('${doc.filename}')" title="Delete">✕</span>
                `;
                container.appendChild(item);
            });
        }
        
        async function deleteDoc(filename) {
            if (!confirm(`Delete ${filename}?`)) return;
            
            await fetch(`/api/document/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });
            loadDocuments();
        }
        
        async function loadConversations() {
            const resp = await fetch('/api/conversations');
            const data = await resp.json();
            
            const container = document.getElementById('conversations');
            container.innerHTML = '';
            
            data.conversations.forEach(conv => {
                const item = document.createElement('div');
                item.className = 'conv-item';
                if (conv.id === currentConvId) item.classList.add('active');
                item.textContent = conv.title || 'New Chat';
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
            
            data.messages.forEach(msg => {
                addMessage(msg.content, msg.role);
            });
            
            loadConversations();
        }
        
        function newChat() {
            currentConvId = null;
            document.getElementById('messages').innerHTML = '';
        }
        
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        loadConversations();
        loadDocuments();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Chat with RAG & Search...")
    print(f"📍 http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
