#!/usr/bin/env python3
"""
AI Chat with RAG & Web Search - Enhanced Version
Features: RAG, Web Search, Multi-model support, SQLite storage, improved UI/UX
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict, Any
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
import json
import asyncio
import logging
from contextlib import contextmanager
import traceback

# ==================== Logging Setup ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
DATA_DIR = os.getenv('DATA_DIR', '/app/data')
DB_PATH = os.path.join(DATA_DIR, 'chat.db')
MODELS_CACHE = os.getenv('HF_HOME', '/app/data/models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE, exist_ok=True)

logger.info(f"📁 Data directory: {DATA_DIR}")
logger.info(f"🗄️  Database: {DB_PATH}")
logger.info(f"🤖 Ollama host: {OLLAMA_HOST}")

app = FastAPI(title="AI Chat with RAG & Search - Enhanced")

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
    """Lazy load embedding model with error handling"""
    global embedder
    if embedder is None:
        try:
            logger.info("📥 Loading embedding model...")
            embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODELS_CACHE)
            logger.info("✓ Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return embedder

# ==================== Database Setup ====================

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize SQLite database with improved schema"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            c.execute('''CREATE TABLE IF NOT EXISTS conversations
                         (id TEXT PRIMARY KEY,
                          title TEXT,
                          model TEXT,
                          created_at TEXT,
                          updated_at TEXT,
                          message_count INTEGER DEFAULT 0,
                          is_archived BOOLEAN DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS messages
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          conversation_id TEXT,
                          role TEXT,
                          content TEXT,
                          timestamp TEXT,
                          tokens INTEGER,
                          used_rag BOOLEAN DEFAULT 0,
                          used_search BOOLEAN DEFAULT 0,
                          rag_sources TEXT,
                          search_queries TEXT,
                          FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS documents
                         (id TEXT PRIMARY KEY,
                          filename TEXT,
                          content TEXT,
                          chunk_index INTEGER,
                          embedding BLOB,
                          file_size INTEGER,
                          created_at TEXT)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS search_cache
                         (query TEXT PRIMARY KEY,
                          results TEXT,
                          timestamp TEXT)''')
            
            # Create indexes for better performance
            c.execute('''CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                         ON messages(conversation_id)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_documents_filename 
                         ON documents(filename)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_conversations_updated 
                         ON conversations(updated_at DESC)''')
            
            conn.commit()
            logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

init_db()

# ==================== Models ====================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    model: str = "qwen2.5-coder:7b"
    use_rag: bool = True
    use_search: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100, le=8000)

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    file_size: int
    created_at: str

# ==================== RAG Functions ====================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks with improved logic"""
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        # Try to break at sentence boundaries
        if end < text_length:
            # Look for sentence endings
            for delimiter in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']:
                last_pos = chunk.rfind(delimiter)
                if last_pos > chunk_size // 2:
                    chunk = text[start:start + last_pos + len(delimiter)]
                    end = start + last_pos + len(delimiter)
                    break
        
        chunk = chunk.strip()
        if len(chunk) > 30:  # Only add substantial chunks
            chunks.append(chunk)
        
        start = end - overlap if end < text_length else text_length
    
    return chunks

async def add_document_to_rag(filename: str, content: str) -> Dict[str, Any]:
    """Add document to RAG database with improved error handling"""
    try:
        doc_id = str(uuid.uuid4())
        chunks = chunk_text(content)
        
        if not chunks:
            raise ValueError("No valid chunks could be extracted from document")
        
        model = get_embedder()
        
        with get_db() as conn:
            c = conn.cursor()
            
            added_chunks = 0
            for idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:
                    continue
                
                try:
                    embedding = model.encode(chunk)
                    
                    c.execute('''INSERT INTO documents 
                                 (id, filename, content, chunk_index, embedding, file_size, created_at)
                                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
                              (f"{doc_id}_{idx}", filename, chunk, idx, 
                               embedding.tobytes(), len(content), datetime.now().isoformat()))
                    added_chunks += 1
                except Exception as e:
                    logger.error(f"Failed to add chunk {idx}: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"✓ Added document: {filename} ({added_chunks} chunks)")
        
        return {
            'document_id': doc_id,
            'filename': filename,
            'chunks': added_chunks,
            'total_chars': len(content),
            'file_size': len(content)
        }
    except Exception as e:
        logger.error(f"Error adding document to RAG: {e}")
        raise

def search_rag(query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
    """Search documents using semantic similarity with improved scoring"""
    try:
        model = get_embedder()
        query_embedding = model.encode(query)
        
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT id, filename, content, embedding FROM documents')
            
            results = []
            for row in c.fetchall():
                doc_id, filename, content, embedding_bytes = row
                doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                if similarity >= min_similarity:
                    results.append({
                        'doc_id': doc_id,
                        'filename': filename,
                        'content': content,
                        'similarity': float(similarity)
                    })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return []

def get_rag_context(query: str, top_k: int = 3) -> tuple[str, List[str]]:
    """Get RAG context for a query, returns context and source filenames"""
    results = search_rag(query, top_k)
    
    if not results:
        return "", []
    
    context_parts = []
    sources = []
    
    for i, result in enumerate(results, 1):
        if result['similarity'] > 0.3:
            context_parts.append(
                f"[Source {i}: {result['filename']} (relevance: {result['similarity']:.2f})]\n{result['content']}"
            )
            if result['filename'] not in sources:
                sources.append(result['filename'])
    
    if not context_parts:
        return "", []
    
    return "\n\n---\n\n".join(context_parts), sources

# ==================== Web Search Functions ====================

async def web_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search the web using DuckDuckGo with caching"""
    try:
        # Check cache first
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT results, timestamp FROM search_cache WHERE query = ?', (query,))
            cached = c.fetchone()
            
            if cached:
                timestamp = datetime.fromisoformat(cached['timestamp'])
                age = (datetime.now() - timestamp).total_seconds()
                if age < 3600:  # 1 hour cache
                    return json.loads(cached['results'])
        
        # Perform search
        url = "https://html.duckduckgo.com/html/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data={'q': query}, headers=headers, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Search returned status {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                for result in soup.select('.result')[:num_results]:
                    title_elem = result.select_one('.result__a')
                    snippet_elem = result.select_one('.result__snippet')
                    
                    if title_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'url': title_elem.get('href', ''),
                            'snippet': snippet_elem.get_text(strip=True) if snippet_elem else ''
                        })
                
                # Cache results
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute('''INSERT OR REPLACE INTO search_cache (query, results, timestamp)
                                 VALUES (?, ?, ?)''',
                              (query, json.dumps(results), datetime.now().isoformat()))
                    conn.commit()
                
                return results
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []

async def get_search_context(query: str) -> tuple[str, List[str]]:
    """Get web search context for a query"""
    results = await web_search(query)
    
    if not results:
        return "", []
    
    context_parts = []
    queries = [query]
    
    for i, result in enumerate(results[:3], 1):
        context_parts.append(
            f"[Web Result {i}: {result['title']}]\n{result['snippet']}\nURL: {result['url']}"
        )
    
    return "\n\n---\n\n".join(context_parts), queries

# ==================== Conversation Management ====================

def create_conversation(model: str) -> str:
    """Create a new conversation"""
    conv_id = str(uuid.uuid4())
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO conversations (id, title, model, created_at, updated_at, message_count)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (conv_id, "New Chat", model, datetime.now().isoformat(), 
                   datetime.now().isoformat(), 0))
        conn.commit()
    return conv_id

def update_conversation_title(conv_id: str, message: str):
    """Update conversation title based on first message"""
    title = message[:50] + "..." if len(message) > 50 else message
    with get_db() as conn:
        c = conn.cursor()
        c.execute('UPDATE conversations SET title = ? WHERE id = ?', (title, conv_id))
        conn.commit()

def save_message(conv_id: str, role: str, content: str, 
                 used_rag: bool = False, used_search: bool = False,
                 rag_sources: List[str] = None, search_queries: List[str] = None,
                 tokens: int = 0):
    """Save a message to the database"""
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO messages 
                     (conversation_id, role, content, timestamp, tokens, 
                      used_rag, used_search, rag_sources, search_queries)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (conv_id, role, content, datetime.now().isoformat(), tokens,
                   used_rag, used_search, 
                   json.dumps(rag_sources) if rag_sources else None,
                   json.dumps(search_queries) if search_queries else None))
        
        c.execute('''UPDATE conversations 
                     SET updated_at = ?, message_count = message_count + 1 
                     WHERE id = ?''',
                  (datetime.now().isoformat(), conv_id))
        conn.commit()

# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the enhanced web interface"""
    return get_html_interface()

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Test database
        with get_db() as conn:
            conn.execute("SELECT 1")
        
        # Test Ollama connection
        ollama_client.list()
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for RAG"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file type
        allowed_extensions = {'.txt', '.md', '.pdf'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        if file_ext == '.pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = "\n\n".join(page.extract_text() for page in pdf_reader.pages)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF parsing error: {str(e)}")
        else:
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File encoding not supported")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document contains no text")
        
        # Add to RAG
        result = await add_document_to_rag(file.filename, text)
        
        return {
            'status': 'success',
            'message': f"Added {file.filename} with {result['chunks']} chunks",
            'data': result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents():
    """Get list of uploaded documents"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('''SELECT DISTINCT filename, COUNT(*) as chunks, 
                         MAX(file_size) as file_size, MAX(created_at) as created_at
                         FROM documents 
                         GROUP BY filename 
                         ORDER BY created_at DESC''')
            
            documents = []
            for row in c.fetchall():
                documents.append({
                    'filename': row['filename'],
                    'chunks': row['chunks'],
                    'file_size': row['file_size'],
                    'created_at': row['created_at']
                })
            
            return {'documents': documents}
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/document/{filename}")
async def delete_document(filename: str):
    """Delete a document from RAG"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM documents WHERE filename = ?', (filename,))
            deleted = c.rowcount
            conn.commit()
        
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {'status': 'success', 'message': f'Deleted {deleted} chunks'}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def get_conversations():
    """Get list of conversations"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('''SELECT id, title, model, created_at, updated_at, message_count 
                         FROM conversations 
                         WHERE is_archived = 0
                         ORDER BY updated_at DESC 
                         LIMIT 50''')
            
            conversations = []
            for row in c.fetchall():
                conversations.append({
                    'id': row['id'],
                    'title': row['title'],
                    'model': row['model'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'message_count': row['message_count']
                })
            
            return {'conversations': conversations}
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation with messages"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('''SELECT role, content, timestamp, used_rag, used_search,
                         rag_sources, search_queries
                         FROM messages 
                         WHERE conversation_id = ? 
                         ORDER BY timestamp ASC''', (conversation_id,))
            
            messages = []
            for row in c.fetchall():
                msg = {
                    'role': row['role'],
                    'content': row['content'],
                    'timestamp': row['timestamp'],
                    'used_rag': bool(row['used_rag']),
                    'used_search': bool(row['used_search'])
                }
                
                if row['rag_sources']:
                    msg['rag_sources'] = json.loads(row['rag_sources'])
                if row['search_queries']:
                    msg['search_queries'] = json.loads(row['search_queries'])
                
                messages.append(msg)
            
            return {'messages': messages}
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
            c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
            conn.commit()
        
        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models():
    """Get available Ollama models"""
    try:
        models = ollama_client.list()
        return {'models': [model['name'] for model in models['models']]}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {'models': ['qwen2.5-coder:7b', 'mistral:7b', 'llama3.2:3b']}

# ==================== WebSocket Chat ====================

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """Enhanced WebSocket endpoint for streaming chat"""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
                
                message = data.get('message', '').strip()
                if not message:
                    await websocket.send_json({'type': 'error', 'content': 'Empty message'})
                    continue
                
                conv_id = data.get('conversation_id')
                model = data.get('model', 'qwen2.5-coder:7b')
                use_rag = data.get('use_rag', True)
                use_search = data.get('use_search', False)
                temperature = data.get('temperature', 0.7)
                
                # Create conversation if needed
                if not conv_id:
                    conv_id = create_conversation(model)
                    await websocket.send_json({
                        'type': 'conversation_created',
                        'conversation_id': conv_id
                    })
                
                # Save user message
                save_message(conv_id, 'user', message)
                
                # Update title on first message
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute('SELECT message_count FROM conversations WHERE id = ?', (conv_id,))
                    row = c.fetchone()
                    if row and row['message_count'] == 1:
                        update_conversation_title(conv_id, message)
                
                # Build context
                context_parts = []
                rag_sources = []
                search_queries = []
                
                if use_rag:
                    rag_context, sources = get_rag_context(message)
                    if rag_context:
                        context_parts.append(f"**Relevant Documents:**\n{rag_context}")
                        rag_sources = sources
                
                if use_search:
                    search_context, queries = await get_search_context(message)
                    if search_context:
                        context_parts.append(f"**Web Search Results:**\n{search_context}")
                        search_queries = queries
                
                # Build prompt
                if context_parts:
                    prompt = f"{chr(10).join(context_parts)}\n\n**User Question:**\n{message}\n\nPlease provide a comprehensive answer based on the context above."
                else:
                    prompt = message
                
                # Stream response
                full_response = ""
                try:
                    stream = ollama_client.chat(
                        model=model,
                        messages=[{'role': 'user', 'content': prompt}],
                        stream=True,
                        options={'temperature': temperature}
                    )
                    
                    for chunk in stream:
                        if chunk and 'message' in chunk and 'content' in chunk['message']:
                            content = chunk['message']['content']
                            full_response += content
                            await websocket.send_json({
                                'type': 'chunk',
                                'content': content,
                                'conversation_id': conv_id
                            })
                    
                    # Save assistant message
                    save_message(
                        conv_id, 'assistant', full_response,
                        used_rag=use_rag and bool(rag_sources),
                        used_search=use_search and bool(search_queries),
                        rag_sources=rag_sources,
                        search_queries=search_queries
                    )
                    
                    await websocket.send_json({
                        'type': 'done',
                        'conversation_id': conv_id,
                        'used_rag': bool(rag_sources),
                        'used_search': bool(search_queries),
                        'rag_sources': rag_sources,
                        'search_queries': search_queries
                    })
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg)
                    await websocket.send_json({
                        'type': 'error',
                        'content': error_msg
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    'type': 'error',
                    'content': 'Invalid JSON'
                })
            except Exception as e:
                logger.error(f"Error in message processing: {e}\n{traceback.format_exc()}")
                await websocket.send_json({
                    'type': 'error',
                    'content': str(e)
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")

# ==================== HTML Interface ====================

def get_html_interface():
    """Return the enhanced HTML interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chat with RAG & Search - Enhanced</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #2563eb;
            --primary-dark: #1e40af;
            --secondary: #64748b;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --border: #475569;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            height: 100vh;
        }
        
        /* Sidebar */
        .sidebar {
            width: 280px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border);
        }
        
        .sidebar-header h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: var(--text-primary);
        }
        
        .new-chat-btn {
            width: 100%;
            padding: 12px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .new-chat-btn:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }
        
        .sidebar-section {
            padding: 15px;
            border-bottom: 1px solid var(--border);
        }
        
        .section-title {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }
        
        .upload-btn {
            width: 100%;
            padding: 10px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .upload-btn:hover {
            background: var(--border);
        }
        
        #fileInput {
            display: none;
        }
        
        .documents, .conversations {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        
        .doc-item, .conv-item {
            padding: 10px;
            margin-bottom: 5px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
        }
        
        .doc-item:hover, .conv-item:hover {
            background: var(--border);
            transform: translateX(2px);
        }
        
        .conv-item.active {
            background: var(--primary);
            color: white;
        }
        
        .doc-name {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .doc-delete {
            margin-left: 10px;
            padding: 2px 6px;
            background: var(--danger);
            color: white;
            border-radius: 3px;
            font-size: 12px;
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        
        .doc-delete:hover {
            opacity: 1;
        }
        
        .no-docs {
            padding: 20px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 13px;
        }
        
        /* Main Chat Area */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-primary);
        }
        
        .header {
            padding: 20px 30px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 20px;
            font-weight: 600;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
            user-select: none;
        }
        
        .toggle:hover {
            background: var(--border);
        }
        
        .toggle input[type="checkbox"] {
            appearance: none;
            width: 16px;
            height: 16px;
            border: 2px solid var(--border);
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .toggle input[type="checkbox"]:checked {
            background: var(--success);
            border-color: var(--success);
        }
        
        select#model {
            padding: 8px 12px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }
        
        /* Messages */
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .message {
            max-width: 75%;
            padding: 15px 20px;
            border-radius: 12px;
            line-height: 1.6;
            position: relative;
            animation: slideIn 0.3s ease-out;
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
            background: var(--primary);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant {
            background: var(--bg-secondary);
            color: var(--text-primary);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            border: 1px solid var(--border);
        }
        
        .message-meta {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .message code {
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .message pre {
            background: var(--bg-tertiary);
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 10px 0;
        }
        
        .message pre code {
            background: none;
            padding: 0;
        }
        
        /* Input Area */
        .input-area {
            padding: 20px 30px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
        }
        
        .input-container {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }
        
        #input {
            flex: 1;
            padding: 15px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            resize: none;
            font-family: inherit;
            max-height: 200px;
            transition: border-color 0.2s;
        }
        
        #input:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        #send {
            padding: 15px 30px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
        }
        
        #send:hover:not(:disabled) {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }
        
        #send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-tertiary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--secondary);
        }
        
        /* Loading indicator */
        .typing-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h2>🤖 AI Chat</h2>
                <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
            </div>
            
            <div class="sidebar-section">
                <div class="section-title">📄 Documents</div>
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    Upload Document
                </button>
                <input type="file" id="fileInput" accept=".txt,.md,.pdf" onchange="uploadFile(event)">
            </div>
            
            <div class="sidebar-section" style="flex: 1; display: flex; flex-direction: column;">
                <div class="section-title">Documents (RAG)</div>
                <div class="documents" id="documents">
                    <div class="no-docs">No documents uploaded</div>
                </div>
            </div>
            
            <div class="sidebar-section" style="flex: 2; display: flex; flex-direction: column;">
                <div class="section-title">💬 Conversations</div>
                <div class="conversations" id="conversations"></div>
            </div>
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
    </div>
    
    <script>
        let currentConvId = null;
        let ws = null;
        let reconnectAttempts = 0;
        const MAX_RECONNECT_ATTEMPTS = 5;
        
        function connectWS() {
            try {
                ws = new WebSocket(`ws://${location.host}/ws/chat`);
                
                ws.onopen = () => {
                    console.log('WebSocket connected');
                    reconnectAttempts = 0;
                };
                
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    } catch (e) {
                        console.error('Error parsing message:', e);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
                
                ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                        reconnectAttempts++;
                        setTimeout(connectWS, 1000 * reconnectAttempts);
                    }
                };
            } catch (e) {
                console.error('Error connecting WebSocket:', e);
            }
        }
        
        function handleWebSocketMessage(data) {
            if (data.type === 'chunk') {
                currentConvId = data.conversation_id;
                appendToLastMessage(data.content);
            } else if (data.type === 'done') {
                if (data.used_rag || data.used_search) {
                    const meta = [];
                    if (data.used_rag) {
                        meta.push(`📚 RAG (${data.rag_sources?.length || 0} sources)`);
                    }
                    if (data.used_search) {
                        meta.push('🔍 Web Search');
                    }
                    appendMetaToLastMessage(meta.join(' • '));
                }
                document.getElementById('send').disabled = false;
                loadConversations();
            } else if (data.type === 'conversation_created') {
                currentConvId = data.conversation_id;
            } else if (data.type === 'error') {
                addErrorMessage(data.content);
                document.getElementById('send').disabled = false;
            }
        }
        
        connectWS();
        
        function sendMessage() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;
            
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
        
        function addErrorMessage(content) {
            const messages = document.getElementById('messages');
            const msg = document.createElement('div');
            msg.className = 'message assistant';
            msg.style.borderColor = 'var(--danger)';
            msg.textContent = '⚠️ Error: ' + content;
            messages.appendChild(msg);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function appendToLastMessage(content) {
            const messages = document.getElementById('messages');
            const lastMsg = messages.lastElementChild;
            if (lastMsg && lastMsg.classList.contains('assistant')) {
                lastMsg.textContent += content;
                messages.scrollTop = messages.scrollHeight;
            }
        }
        
        function appendMetaToLastMessage(meta) {
            const messages = document.getElementById('messages');
            const lastMsg = messages.lastElementChild;
            if (lastMsg && lastMsg.classList.contains('assistant')) {
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
                    alert(`✓ ${data.message}`);
                    loadDocuments();
                } else {
                    alert('Error: ' + (data.error || data.detail || 'Upload failed'));
                }
            } catch (e) {
                alert('Upload error: ' + e.message);
            }
            
            event.target.value = '';
        }
        
        async function loadDocuments() {
            try {
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
            } catch (e) {
                console.error('Error loading documents:', e);
            }
        }
        
        async function deleteDoc(filename) {
            if (!confirm(`Delete ${filename}?`)) return;
            
            try {
                await fetch(`/api/document/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                });
                loadDocuments();
            } catch (e) {
                alert('Delete error: ' + e.message);
            }
        }
        
        async function loadConversations() {
            try {
                const resp = await fetch('/api/conversations');
                const data = await resp.json();
                
                const container = document.getElementById('conversations');
                container.innerHTML = '';
                
                if (data.conversations.length === 0) {
                    container.innerHTML = '<div class="no-docs">No conversations</div>';
                    return;
                }
                
                data.conversations.forEach(conv => {
                    const item = document.createElement('div');
                    item.className = 'conv-item';
                    if (conv.id === currentConvId) item.classList.add('active');
                    item.textContent = conv.title || 'New Chat';
                    item.onclick = () => loadConversation(conv.id);
                    container.appendChild(item);
                });
            } catch (e) {
                console.error('Error loading conversations:', e);
            }
        }
        
        async function loadConversation(id) {
            try {
                currentConvId = id;
                const resp = await fetch(`/api/conversation/${id}`);
                const data = await resp.json();
                
                const messages = document.getElementById('messages');
                messages.innerHTML = '';
                
                data.messages.forEach(msg => {
                    const msgEl = addMessage(msg.content, msg.role);
                    if (msg.used_rag || msg.used_search) {
                        const meta = [];
                        if (msg.used_rag) meta.push('📚 RAG');
                        if (msg.used_search) meta.push('🔍 Search');
                        const metaDiv = document.createElement('div');
                        metaDiv.className = 'message-meta';
                        metaDiv.textContent = meta.join(' • ');
                        msgEl.appendChild(metaDiv);
                    }
                });
                
                loadConversations();
            } catch (e) {
                console.error('Error loading conversation:', e);
            }
        }
        
        function newChat() {
            currentConvId = null;
            document.getElementById('messages').innerHTML = '';
            loadConversations();
        }
        
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        // Initialize
        loadConversations();
        loadDocuments();
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Chat with RAG & Search (Enhanced Version)...")
    logger.info(f"📍 http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
