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
import os
import docker
import asyncio
import subprocess
import json
from contextlib import redirect_stdout, redirect_stderr
from threading import Lock

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
VLLM_HOST = os.getenv("VLLM_HOST", "http://vllm:8000/v1")
VLLM_CONTAINER = os.getenv("VLLM_CONTAINER", "vllm")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Hugging Face token for gated models

# Available models with their vLLM command configurations
AVAILABLE_MODELS = [
    {
        "id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "name": "Qwen 2.5 Coder 3B",
        "quantized": False,
        "command": [
            "--model", "Qwen/Qwen2.5-Coder-3B-Instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "4096"
        ]
    },
    {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "name": "Llama 3.2 3B",
        "quantized": False,
        "command": [
            "--model", "meta-llama/Llama-3.2-3B-Instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "4096"
        ]
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 1B",
        "quantized": True,
        "command": [
            "--model", "meta-llama/Llama-3.2-1B-Instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "4096"
        ]
    },
    {
        "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "name": "Qwen 2.5 Coder 7B",
        "quantized": False,
        "command": [
            "--model", "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "4096"
        ]
    },
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral 7B",
        "quantized": False,
        "command": [
            "--model", "mistralai/Mistral-7B-Instruct-v0.3",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "8192"
        ]
    },
]

# Default model
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Current model state
current_model = DEFAULT_MODEL
model_switching = False
model_switch_lock = Lock()
model_switch_status = {"status": "ready", "message": "", "progress": 0}

# Initialize Docker client
try:
    docker_client = docker.from_env()
    logger.info("✓ Docker client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Docker client: {e}")
    docker_client = None

# Initialize OpenAI client
try:
    client = OpenAI(base_url=VLLM_HOST, api_key="dummy")
    logger.info(f"✓ Connected to vLLM at {VLLM_HOST}")
except Exception as e:
    logger.error(f"Failed to initialize vLLM client: {e}")
    client = None

app = FastAPI(title="AI Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model switching functions
async def switch_model(new_model_id: str):
    """Switch vLLM to a new model by restarting the container with docker-compose"""
    global current_model, model_switching, model_switch_status, client
    
    with model_switch_lock:
        if model_switching:
            return {"status": "error", "message": "Model switch already in progress"}
        
        if new_model_id == current_model:
            return {"status": "success", "message": f"Model {new_model_id} is already active"}
        
        # Find model config
        model_config = next((m for m in AVAILABLE_MODELS if m["id"] == new_model_id), None)
        if not model_config:
            return {"status": "error", "message": f"Model {new_model_id} not found"}
        
        model_switching = True
        model_switch_status = {"status": "switching", "message": f"Switching to {model_config['name']}...", "progress": 10}
    
    try:
        logger.info(f"🔄 MODEL SWITCH INITIATED: {current_model} → {new_model_id} ({model_config['name']})")
        
        if not docker_client:
            raise Exception("Docker client not available")
        
        import time
        
        # Step 1: Stop the vllm container
        model_switch_status["progress"] = 20
        model_switch_status["message"] = "Stopping vLLM container..."
        logger.info("📦 Step 1/5: Stopping current vLLM container...")
        try:
            container = docker_client.containers.get(VLLM_CONTAINER)
            container.stop(timeout=10)
            logger.info("✓ Container stopped successfully")
        except docker.errors.NotFound:
            logger.warning(f"⚠️ Container {VLLM_CONTAINER} not found, may already be stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping container: {e}")
            raise
        
        # Step 2: Remove the container
        model_switch_status["progress"] = 40
        model_switch_status["message"] = "Removing old container..."
        logger.info("🗑️ Step 2/5: Removing old container...")
        try:
            container = docker_client.containers.get(VLLM_CONTAINER)
            container.remove(force=True)
            logger.info("✓ Container removed successfully")
        except docker.errors.NotFound:
            logger.info("ℹ️ Container already removed")
        except Exception as e:
            logger.warning(f"⚠️ Error removing container: {e}")
        
        # Step 3: Start container with new model
        model_switch_status["progress"] = 50
        model_switch_status["message"] = "Starting container with new model..."
        logger.info(f"🚀 Step 3/5: Starting container with new model: {model_config['name']}")
        logger.info(f"   Model ID: {new_model_id}")
        logger.info(f"   Command: {' '.join(model_config['command'])}")
        
        try:
            # Prepare environment variables
            env_vars = {}
            if HF_TOKEN:
                env_vars["HF_TOKEN"] = HF_TOKEN
                logger.info("🔑 Using Hugging Face token for gated model access")
            
            # Start new container with new model command
            container = docker_client.containers.run(
                "vllm/vllm-openai:v0.5.4",
                name=VLLM_CONTAINER,
                command=model_config["command"],
                ports={"8000/tcp": 8001},
                environment=env_vars if env_vars else None,
                detach=True,
                remove=False,
                device_requests=[
                    docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])
                ],
                restart_policy={"Name": "unless-stopped"}
            )
            logger.info(f"✓ Container started successfully (ID: {container.id[:12]})")
        except Exception as e:
            logger.error(f"❌ Error starting container: {e}")
            raise
        
        # Step 4: Wait for model to load
        model_switch_status["progress"] = 60
        model_switch_status["message"] = "Loading new model (this may take a few minutes)..."
        logger.info("⏳ Step 4/5: Waiting for model to load (this may take several minutes)...")
        
        max_wait = 300  # 5 minutes max
        wait_time = 0
        while wait_time < max_wait:
            time.sleep(5)
            wait_time += 5
            try:
                # Check if container is running
                container.reload()
                if container.status == "running":
                    # Try to connect to vLLM
                    try:
                        test_client = OpenAI(base_url=VLLM_HOST, api_key="dummy", timeout=5)
                        test_client.models.list()
                        # Success!
                        logger.info(f"✓ Model is ready! (loaded in {wait_time}s)")
                        break
                    except Exception as e:
                        model_switch_status["progress"] = 60 + int((wait_time / max_wait) * 30)
                        model_switch_status["message"] = f"Model loading... ({wait_time}s)"
                        if wait_time % 30 == 0:  # Log every 30 seconds
                            logger.info(f"⏳ Still loading... ({wait_time}s elapsed)")
                        continue
                else:
                    raise Exception(f"Container status: {container.status}")
            except Exception as e:
                logger.warning(f"⚠️ Error checking container: {e}")
        
        if wait_time >= max_wait:
            logger.error("❌ Model loading timeout after 5 minutes")
            raise Exception("Model loading timeout after 5 minutes")
        
        # Step 5: Update client and model
        model_switch_status["progress"] = 95
        model_switch_status["message"] = "Finalizing..."
        logger.info("🔧 Step 5/5: Finalizing model switch...")
        
        previous_model = current_model
        client = OpenAI(base_url=VLLM_HOST, api_key="dummy")
        current_model = new_model_id
        
        model_switch_status = {"status": "success", "message": f"Successfully switched to {model_config['name']}", "progress": 100}
        logger.info(f"✅ MODEL SWITCH COMPLETE: Successfully switched to {model_config['name']} ({new_model_id})")
        logger.info(f"   Previous model: {previous_model}")
        logger.info(f"   New model: {new_model_id}")
        
    except Exception as e:
        logger.error(f"❌ MODEL SWITCH FAILED: {str(e)}")
        logger.error(f"   Error details: {traceback.format_exc()}")
        model_switch_status = {"status": "error", "message": f"Error: {str(e)}", "progress": 0}
    finally:
        model_switching = False
    
    return model_switch_status

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
    c.execute('''SELECT role, content, timestamp FROM messages 
                 WHERE conversation_id = ? 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (conv_id, limit))
    
    messages = []
    for row in c.fetchall():
        messages.append({'role': row[0], 'content': row[1], 'timestamp': row[2]})
    
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
            z-index: 1;
            transition: width {ANIMATIONS['transition_speed']};
            position: relative;
        }}
        
        .sidebar.collapsed {{
            width: {DIMENSIONS['sidebar_collapsed_width']};
        }}
        
        .sidebar-toggle {{
            position: absolute;
            top: 16px;
            right: -12px;
            width: 24px;
            height: 24px;
            background: {COLORS['bg_primary']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10;
            font-size: 10px;
            color: {COLORS['text_primary']};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .sidebar-toggle:hover {{
            background: {COLORS['bg_tertiary']};
        }}
        
        .sidebar.collapsed .sidebar-content {{
            display: none;
        }}
        
        .sidebar.collapsed .sidebar-toggle {{
            right: -12px;
        }}
        
        /* New Chat button for collapsed sidebar */
        .new-chat-icon-btn {{
            display: none;
            width: 40px;
            height: 40px;
            margin: 16px auto 0;
            background: {COLORS['btn_primary']};
            color: {COLORS['bg_primary']};
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
            background: {COLORS['btn_primary_hover']};
            transform: scale(1.05);
        }}
        
        .new-chat-icon-btn:active {{
            transform: scale(0.95);
        }}
        
        .sidebar-header {{
            padding: 20px;
            border-bottom: 1px solid {COLORS['bg_tertiary']};
        }}
        
        .new-chat-btn {{
            width: 100%;
            padding: 12px;
            background: {COLORS['btn_primary']};
            color: {COLORS['bg_primary']};
            border: none;
            border-radius: {DIMENSIONS['border_radius']};
            font-size: {FONTS['size_base']};
            font-weight: 600;
            cursor: pointer;
            transition: all {ANIMATIONS['transition_speed']};
        }}
        
        .new-chat-btn:hover {{ background: {COLORS['btn_primary_hover']}; color: {COLORS['bg_primary']}; }}
        
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
        .conv-item.active {{ background: {COLORS['accent_primary']}; color: {COLORS['bg_primary']}; }}
        
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
        
        .header-left .sidebar-toggle {{
            margin-right: 8px;
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
            min-width: 200px;
        }}
        
        .model-selector:focus {{ outline: none; border-color: {COLORS['accent_primary']}; }}
        .model-selector:hover {{ border-color: {COLORS['accent_primary']}; }}
        
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
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            align-items: center;
            max-width: 100%;
        }}
        
        .message {{
            width: 100%;
            max-width: {DIMENSIONS['message_max_width']};
            padding: 8px 0;
            line-height: 1.6;
            animation: slideIn {ANIMATIONS['slide_duration']} ease-out;
            font-size: {FONTS['size_message']};
            background: transparent;
            border: none;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(5px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .message.user {{
            color: #366c9c;
            text-align: right;
            margin-left: auto;
        }}
        
        .message.assistant {{
            color: #1a1a1a;
            text-align: left;
            margin-right: auto;
            position: relative;
        }}
        
        .message-timestamp {{
            font-size: 11px;
            color: {COLORS['text_tertiary']};
            margin-top: 4px;
            opacity: 0.7;
        }}
        
        .message.user .message-timestamp {{
            text-align: right;
        }}
        
        .message.assistant .message-timestamp {{
            text-align: left;
        }}
        
        .message.assistant pre {{
            position: relative;
            background: #f5f5f5;
            padding: 12px 16px;
            border-radius: {DIMENSIONS['border_radius_small']};
            overflow-x: auto;
            margin: 8px 0;
            border: 1px solid #e0e0e0;
        }}
        
        .message.assistant code {{
            background: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        .message.assistant pre code {{
            background: transparent;
            padding: 0;
            color: #1a1a1a;
        }}
        
        /* Syntax highlighting styles */
        .message.assistant pre.hljs {{
            background: #f5f5f5;
            color: #1a1a1a;
        }}
        
        .message.assistant .hljs-keyword {{ color: #366c9c; }}
        .message.assistant .hljs-string {{ color: #91b461; }}
        .message.assistant .hljs-comment {{ color: #999999; font-style: italic; }}
        .message.assistant .hljs-number {{ color: #fd7589; }}
        .message.assistant .hljs-function {{ color: #67a2cb; }}
        .message.assistant .hljs-variable {{ color: #5898b7; }}
        .message.assistant .hljs-title {{ color: #67a2cb; }}
        .message.assistant .hljs-type {{ color: #5898b7; }}
        
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
            background: {COLORS['btn_secondary_hover']};
        }}
        
        .copy-btn.copied {{
            background: {COLORS['accent_secondary']};
            color: white;
        }}
        
        .loading {{
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: {DIMENSIONS['message_max_width']};
            width: 100%;
            padding: 8px 0;
            background: transparent;
            margin-right: auto;
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
            padding: 16px 24px;
            background: {COLORS['bg_quaternary']};
            border-top: 1px solid {COLORS['bg_tertiary']};
            display: flex;
            justify-content: center;
        }}
        
        .input-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            max-width: {DIMENSIONS['message_max_width']};
            width: 100%;
            background: {COLORS['bg_primary']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius']};
            padding: 8px 12px;
        }}
        
        #input {{
            flex: 1;
            padding: 8px 12px;
            background: transparent;
            color: {COLORS['text_primary']};
            border: none;
            border-radius: 0;
            font-size: {FONTS['size_base']};
            resize: none;
            font-family: inherit;
            min-height: 24px;
            max-height: 120px;
        }}
        
        #input:focus {{ outline: none; }}
        #input:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        #input::placeholder {{ color: {COLORS['text_tertiary']}; }}
        
        #send {{
            width: 32px;
            height: 32px;
            padding: 0;
            background: transparent;
            color: {COLORS['text_primary']};
            border: none;
            border-radius: 50%;
            cursor: pointer;
            transition: all {ANIMATIONS['transition_speed']};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }}
        
        #send:hover:not(:disabled) {{
            background: {COLORS['bg_tertiary']};
            transform: scale(1.1);
        }}
        #send:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}
        #send::before {{
            content: '→';
        }}
        
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: {COLORS['scrollbar_track']}; }}
        ::-webkit-scrollbar-thumb {{ background: {COLORS['scrollbar_thumb']}; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {COLORS['scrollbar_thumb_hover']}; }}
        
        .welcome {{
            text-align: center;
            color: {COLORS['text_secondary']};
            padding: 60px 40px;
            max-width: {DIMENSIONS['message_max_width']};
            width: 100%;
            margin: 0 auto;
        }}
        
        .welcome h2 {{
            font-size: 28px;
            margin-bottom: 12px;
            color: {COLORS['text_primary']};
            font-weight: 500;
        }}
        
        .welcome p {{
            font-size: {FONTS['size_base']};
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
            z-index: 1000;
            transition: all {ANIMATIONS['transition_speed']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
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
        
        /* Model Toggle */
        .model-toggle {{
            position: relative;
        }}
        
        .model-toggle-btn {{
            padding: 10px 18px;
            background: {COLORS['bg_primary']};
            border: 2px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius_small']};
            color: {COLORS['text_primary']};
            font-size: {FONTS['size_base']};
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 180px;
            justify-content: space-between;
        }}
        
        .model-toggle-btn:hover {{
            border-color: {COLORS['accent_primary']};
            background: {COLORS['bg_tertiary']};
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
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['bg_tertiary']};
            border-radius: {DIMENSIONS['border_radius']};
            min-width: 250px;
            max-height: 400px;
            overflow-y: auto;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .model-dropdown.show {{
            display: block;
        }}
        
        .model-option {{
            padding: 12px 15px;
            cursor: pointer;
            border-bottom: 1px solid {COLORS['bg_tertiary']};
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .model-option:last-child {{
            border-bottom: none;
        }}
        
        .model-option:hover {{
            background: {COLORS['bg_tertiary']};
        }}
        
        .model-option.active {{
            background: {COLORS['accent_primary']};
            color: white;
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
            font-family: 'Courier New', monospace;
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
    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        {css}
    </style>
</head>
<body>
    <!-- Model Switch Confirmation Modal -->
    <div class="model-switch-confirm" id="modelSwitchConfirm">
        <div class="model-switch-confirm-content">
            <div class="model-switch-confirm-header">⚠️ Experimental Feature Warning</div>
            <div class="model-switch-confirm-warning">
                <strong>Model switching is an experimental feature that may fail.</strong>
            </div>
            <div class="model-switch-confirm-info">
                <p><strong>Switching to:</strong> <span id="confirmModelName"></span></p>
                <p>⚠️ <strong>Please note:</strong></p>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>Model switching can take several minutes</li>
                    <li>The process may fail and require manual intervention</li>
                    <li>You will be returned to the model selection menu if it fails</li>
                    <li>All current chat sessions will be preserved</li>
                </ul>
            </div>
            <div class="model-switch-confirm-buttons">
                <button class="model-switch-confirm-btn" onclick="cancelModelSwitch()">Cancel</button>
                <button class="model-switch-confirm-btn primary" onclick="proceedWithModelSwitch()">Continue</button>
            </div>
        </div>
    </div>
    
    <!-- Boot Menu -->
    <div class="boot-menu" id="bootMenu">
        <div class="boot-menu-content">
            <div class="boot-menu-header">
                <h1>AI Chat - Model Selection</h1>
                <p>Select a model to load. Use arrow keys to navigate, Enter to select.</p>
            </div>
            <ul class="boot-menu-list" id="bootMenuList">
                {''.join([f'''
                <li class="boot-menu-item" data-model-id="{m['id']}" onclick="selectModelFromBootMenu('{m['id']}')">
                    <span class="boot-menu-item-name">{m['name']}</span>
                    <span class="boot-menu-item-badge">{'Quantized' if m.get('quantized') else 'Standard'}</span>
                </li>
                ''' for m in AVAILABLE_MODELS])}
            </ul>
            <div class="boot-menu-error" id="bootMenuError"></div>
            <div class="boot-menu-instructions">
                <p>↑↓ Navigate | Enter Select | ESC Cancel</p>
            </div>
        </div>
    </div>
    
    <div class="sidebar" id="sidebar">
        <button class="new-chat-icon-btn" onclick="newChat()" title="New Chat">+</button>
        <div class="sidebar-content">
            <div class="sidebar-header">
                <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
            </div>
            <div class="conversations" id="conversations"></div>
        </div>
    </div>
    
    <div class="main">
        <div class="header">
            <div class="header-left">
                <button class="sidebar-toggle" onclick="toggleSidebar()" title="Toggle sidebar" id="sidebarToggle">📖</button>
                <h1>AI Chat</h1>
            </div>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div class="model-toggle" style="position: relative;">
                    <button class="model-toggle-btn" id="modelToggleBtn" onclick="toggleModelDropdown(event)" title="Click to switch model" style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px;">🤖</span>
                        <span id="currentModelName" style="font-weight: 500;">Loading...</span>
                        <span style="font-size: 10px; margin-left: 5px;">▼</span>
                    </button>
                    <div class="model-dropdown" id="modelDropdown" style="display: none;">
                        {''.join([f'''
                        <div class="model-option" id="model-{m["id"]}" onclick="confirmModelSwitch('{m["id"]}', '{m["name"]}')">
                            <span class="model-option-name">{m["name"]}</span>
                            <span class="model-option-badge">{'Quantized' if m.get('quantized') else 'Standard'}</span>
                        </div>
                        ''' for m in AVAILABLE_MODELS])}
                    </div>
                </div>
                <div class="status">
                    <div class="status-dot disconnected" id="statusDot"></div>
                    <span id="statusText">Connecting...</span>
                </div>
            </div>
        </div>
        
        <div class="model-status-console" id="modelStatusConsole">
            <div class="model-status-header">
                <span>Model Switch Status</span>
                <button class="model-status-close" onclick="closeModelStatus()">×</button>
            </div>
            <div class="model-status-content">
                <div class="model-status-message" id="modelStatusMessage">Ready</div>
                <div class="model-status-progress">
                    <div class="model-status-progress-bar" id="modelStatusProgress" style="width: 0%"></div>
                </div>
                <div class="model-status-status" id="modelStatusStatus"></div>
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
                <button id="send" onclick="sendMessage()" title="Send message"></button>
            </div>
        </div>
    </div>
    
    <button class="log-viewer-btn" onclick="toggleLogViewer()" title="View Terminal Logs">📋 Logs</button>
    
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
        let markedReady = false;
        
        // Initialize marked library when ready
        function initMarked() {{
            if (typeof marked !== 'undefined') {{
                markedReady = true;
                marked.setOptions({{
                    breaks: true,
                    gfm: true,
                    headerIds: false,
                    mangle: false
                }});
                console.log('Marked library loaded');
                renderMarkdown(); // Render any existing messages
            }} else {{
                setTimeout(initMarked, 50);
            }}
        }}
        
        // Start checking for marked library
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initMarked);
        }} else {{
            initMarked();
        }}
        
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
                        // Wait a bit for marked to be ready, then render
                        setTimeout(() => {{
                            renderMarkdown();
                        }}, 100);
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
        
        function toggleSidebar() {{
            const sidebar = document.getElementById('sidebar');
            const toggle = document.getElementById('sidebarToggle');
            sidebar.classList.toggle('collapsed');
            toggle.textContent = sidebar.classList.contains('collapsed') ? '📕' : '📖';
        }}
        
        function toggleLogViewer() {{
            const viewer = document.getElementById('logViewer');
            viewer.classList.toggle('show');
            if (viewer.classList.contains('show') && !logWs) {{
                connectLogWS();
            }}
        }}
        
        // Model switching functions
        let modelStatusPollInterval = null;
        
        function toggleModelDropdown(event) {{
            if (event) event.stopPropagation();
            const dropdown = document.getElementById('modelDropdown');
            if (dropdown) {{
                const isShowing = dropdown.style.display === 'block' || dropdown.classList.contains('show');
                if (isShowing) {{
                    dropdown.style.display = 'none';
                    dropdown.classList.remove('show');
                }} else {{
                    dropdown.style.display = 'block';
                    dropdown.classList.add('show');
                }}
            }} else {{
                console.error('Model dropdown not found!');
            }}
        }}
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(event) {{
            const toggle = document.getElementById('modelToggleBtn');
            const dropdown = document.getElementById('modelDropdown');
            if (toggle && dropdown && !toggle.contains(event.target) && !dropdown.contains(event.target)) {{
                dropdown.style.display = 'none';
                dropdown.classList.remove('show');
            }}
        }});
        
        // Model switch confirmation
        let pendingModelSwitch = null;
        
        function confirmModelSwitch(modelId, modelName) {{
            // Close dropdown
            document.getElementById('modelDropdown').classList.remove('show');
            
            // Store pending switch
            pendingModelSwitch = modelId;
            
            // Update modal with model name
            document.getElementById('confirmModelName').textContent = modelName;
            
            // Show confirmation modal
            document.getElementById('modelSwitchConfirm').classList.add('show');
        }}
        
        function cancelModelSwitch() {{
            pendingModelSwitch = null;
            document.getElementById('modelSwitchConfirm').classList.remove('show');
        }}
        
        function proceedWithModelSwitch() {{
            if (pendingModelSwitch) {{
                const modelId = pendingModelSwitch;
                pendingModelSwitch = null;
                document.getElementById('modelSwitchConfirm').classList.remove('show');
                switchModel(modelId);
            }}
        }}
        
        // Close confirmation modal on Escape key
        document.addEventListener('keydown', function(event) {{
            const confirmModal = document.getElementById('modelSwitchConfirm');
            if (event.key === 'Escape' && confirmModal && confirmModal.classList.contains('show')) {{
                cancelModelSwitch();
            }}
        }});
        
        async function switchModel(modelId) {{
            return new Promise((resolve, reject) => {{
                // Close dropdown
                document.getElementById('modelDropdown').classList.remove('show');
                
                // Show status console
                const console = document.getElementById('modelStatusConsole');
                console.classList.add('show');
                
                // Disable toggle button
                const btn = document.getElementById('modelToggleBtn');
                btn.classList.add('switching');
                btn.disabled = true;
                
                fetch('/api/model/switch', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{model_id: modelId}})
                }}).then(response => response.json()).then(data => {{
                    if (data.status === 'initiated') {{
                        // Start polling for status
                        startModelStatusPolling();
                        resolve();
                    }} else {{
                        reject(new Error(data.message || 'Failed to initiate model switch'));
                    }}
                }}).catch(e => {{
                    console.error('Error switching model:', e);
                    updateModelStatus({{
                        status: 'error',
                        message: 'Failed to initiate model switch',
                        progress: 0
                    }});
                    reject(e);
                }});
            }});
        }}
        
        function startModelStatusPolling() {{
            if (modelStatusPollInterval) {{
                clearInterval(modelStatusPollInterval);
            }}
            
            modelStatusPollInterval = setInterval(async () => {{
                try {{
                    const response = await fetch('/api/model/status');
                    const data = await response.json();
                    
                    updateModelStatus(data.status);
                    updateCurrentModel(data.current_model);
                    
                    if (data.status.status === 'success' || data.status.status === 'error') {{
                        clearInterval(modelStatusPollInterval);
                        modelStatusPollInterval = null;
                        
                        // Re-enable toggle after a delay
                        setTimeout(() => {{
                            const btn = document.getElementById('modelToggleBtn');
                            btn.classList.remove('switching');
                            btn.disabled = false;
                        }}, 2000);
                        
                        // If error, show boot menu
                        if (data.status.status === 'error') {{
                            setTimeout(() => {{
                                showBootMenu();
                                showBootMenuError(data.status.message || 'Model switch failed');
                            }}, 2000);
                        }}
                    }}
                }} catch (e) {{
                    console.error('Error polling model status:', e);
                }}
            }}, 1000); // Poll every second
        }}
        
        function updateModelStatus(status) {{
            const messageEl = document.getElementById('modelStatusMessage');
            const progressEl = document.getElementById('modelStatusProgress');
            const statusEl = document.getElementById('modelStatusStatus');
            
            messageEl.textContent = status.message || 'Ready';
            progressEl.style.width = (status.progress || 0) + '%';
            
            if (status.status === 'success') {{
                statusEl.textContent = '✓ Success';
                statusEl.className = 'model-status-status success';
            }} else if (status.status === 'error') {{
                statusEl.textContent = '✗ Error';
                statusEl.className = 'model-status-status error';
            }} else {{
                statusEl.textContent = 'In progress...';
                statusEl.className = 'model-status-status';
            }}
        }}
        
        const availableModels = {json.dumps(AVAILABLE_MODELS)};
        
        function updateCurrentModel(modelId) {{
            currentModel = modelId;
            const model = availableModels.find(m => m.id === modelId);
            if (model) {{
                const nameEl = document.getElementById('currentModelName');
                if (nameEl) {{
                    nameEl.textContent = model.name;
                }}
                
                // Update active state in dropdown
                document.querySelectorAll('.model-option').forEach(opt => {{
                    opt.classList.remove('active');
                }});
                const activeOpt = document.getElementById(`model-${{modelId}}`);
                if (activeOpt) {{
                    activeOpt.classList.add('active');
                }}
            }}
        }}
        
        function closeModelStatus() {{
            document.getElementById('modelStatusConsole').classList.remove('show');
        }}
        
        async function loadCurrentModel() {{
            try {{
                const response = await fetch('/api/model/status');
                const data = await response.json();
                updateCurrentModel(data.current_model);
                
                if (data.switching) {{
                    startModelStatusPolling();
                    document.getElementById('modelStatusConsole').classList.add('show');
                }}
            }} catch (e) {{
                console.error('Error loading current model:', e);
            }}
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
        
        function addMessage(content, role, timestamp = null) {{
            const msg = document.createElement('div');
            msg.className = `message ${{role}}`;
            
            // Create content wrapper
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            msg.appendChild(contentDiv);
            
            // Add timestamp
            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'message-timestamp';
            if (timestamp) {{
                timestampDiv.textContent = formatTimestamp(timestamp);
            }} else {{
                timestampDiv.textContent = formatTimestamp(new Date().toISOString());
            }}
            msg.appendChild(timestampDiv);
            
            if (role === 'assistant') {{
                msg.dataset.needsMarkdown = 'true';
                msg.dataset.originalContent = content; // Store original content for copying
            }}
            document.getElementById('messages').appendChild(msg);
            scrollToBottom();
            return msg;
        }}
        
        function formatTimestamp(isoString) {{
            const date = new Date(isoString);
            const now = new Date();
            const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            const messageDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
            
            const timeStr = date.toLocaleTimeString('en-US', {{ hour: 'numeric', minute: '2-digit', hour12: true }});
            
            if (messageDate.getTime() === today.getTime()) {{
                return timeStr;
            }} else {{
                const yesterday = new Date(today);
                yesterday.setDate(yesterday.getDate() - 1);
                if (messageDate.getTime() === yesterday.getTime()) {{
                    return `Yesterday ${{timeStr}}`;
                }} else {{
                    const dateStr = date.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
                    return `${{dateStr}} ${{timeStr}}`;
                }}
            }}
        }}
        
        function copyToClipboard(text, button) {{
            navigator.clipboard.writeText(text).then(() => {{
                const originalText = button.innerHTML;
                button.innerHTML = '✓ Copied';
                button.classList.add('copied');
                setTimeout(() => {{
                    button.innerHTML = originalText;
                    button.classList.remove('copied');
                }}, 2000);
            }}).catch(err => {{
                console.error('Failed to copy:', err);
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.opacity = '0';
                document.body.appendChild(textArea);
                textArea.select();
                try {{
                    document.execCommand('copy');
                    const originalText = button.innerHTML;
                    button.innerHTML = '✓ Copied';
                    button.classList.add('copied');
                    setTimeout(() => {{
                        button.innerHTML = originalText;
                        button.classList.remove('copied');
                    }}, 2000);
                }} catch (e) {{
                    console.error('Fallback copy failed:', e);
                }}
                document.body.removeChild(textArea);
            }});
        }}
        
        function renderMarkdown() {{
            if (!markedReady || typeof marked === 'undefined') {{
                console.log('Marked library not ready yet, will retry...');
                setTimeout(renderMarkdown, 100);
                return;
            }}
            
            const messages = document.querySelectorAll('.message.assistant[data-needs-markdown="true"]');
            messages.forEach(msg => {{
                const contentDiv = msg.querySelector('.message-content');
                if (!contentDiv) return;
                
                const content = contentDiv.textContent || contentDiv.innerText;
                if (!content || content.trim() === '') return;
                
                // Preserve timestamp
                const timestampDiv = msg.querySelector('.message-timestamp');
                
                try {{
                    if (typeof marked !== 'undefined') {{
                        const html = marked.parse(content);
                        const originalContent = msg.dataset.originalContent || content;
                        contentDiv.innerHTML = html;
                        msg.dataset.needsMarkdown = 'false';
                        msg.dataset.rendered = 'true';
                        
                        // Restore timestamp if it was removed
                        if (timestampDiv && !msg.querySelector('.message-timestamp')) {{
                            msg.appendChild(timestampDiv);
                        }}
                        
                        // Add copy button for the entire message
                        if (!msg.querySelector('.copy-btn.message-copy-btn')) {{
                            const copyBtn = document.createElement('button');
                            copyBtn.className = 'copy-btn message-copy-btn';
                            copyBtn.innerHTML = '📋 Copy';
                            copyBtn.title = 'Copy to clipboard';
                            copyBtn.onclick = (e) => {{
                                e.stopPropagation();
                                copyToClipboard(originalContent, copyBtn);
                            }};
                            msg.appendChild(copyBtn);
                        }}
                        
                        // Apply syntax highlighting to code blocks
                        if (typeof hljs !== 'undefined') {{
                            msg.querySelectorAll('pre code').forEach(block => {{
                                hljs.highlightElement(block);
                            }});
                        }}
                        
                        // Add copy buttons to code blocks
                        msg.querySelectorAll('pre').forEach(preBlock => {{
                            // Check if copy button already exists
                            if (preBlock.querySelector('.copy-btn')) return;
                            
                            const codeText = preBlock.querySelector('code')?.textContent || preBlock.textContent;
                            const copyBtn = document.createElement('button');
                            copyBtn.className = 'copy-btn';
                            copyBtn.innerHTML = '📋 Copy';
                            copyBtn.title = 'Copy code to clipboard';
                            copyBtn.onclick = (e) => {{
                                e.stopPropagation();
                                copyToClipboard(codeText, copyBtn);
                            }};
                            preBlock.appendChild(copyBtn);
                        }});
                    }} else {{
                        console.error('Marked library not available');
                    }}
                }} catch (e) {{
                    console.error('Markdown render error:', e);
                    // Fallback to plain text if markdown fails
                    contentDiv.textContent = content;
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
                const contentDiv = lastMsg.querySelector('.message-content');
                if (contentDiv) {{
                    const currentText = contentDiv.textContent || '';
                    contentDiv.textContent = currentText + content;
                    // Update original content for copying
                    if (lastMsg.dataset.originalContent !== undefined) {{
                        lastMsg.dataset.originalContent = currentText + content;
                    }}
                }}
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
                addMessage(msg.content, msg.role, msg.timestamp);
            }});
            setTimeout(() => {{
                renderMarkdown();
            }}, 100);
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
        
        // Escape key handler to close modals/popups
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                // Close rename modal if open
                const renameModal = document.getElementById('renameModal');
                if (renameModal && renameModal.classList.contains('show')) {{
                    closeRenameModal();
                    return;
                }}
                
                // Close model status console if open
                const modelStatusConsole = document.getElementById('modelStatusConsole');
                if (modelStatusConsole && modelStatusConsole.classList.contains('show')) {{
                    closeModelStatus();
                    return;
                }}
                
                // Close log viewer if open
                const logViewer = document.getElementById('logViewer');
                if (logViewer && logViewer.classList.contains('show')) {{
                    toggleLogViewer();
                    return;
                }}
                
                // Close model dropdown if open
                const modelDropdown = document.getElementById('modelDropdown');
                if (modelDropdown && (modelDropdown.style.display === 'block' || modelDropdown.classList.contains('show'))) {{
                    modelDropdown.style.display = 'none';
                    modelDropdown.classList.remove('show');
                    return;
                }}
            }}
        }});
        
        // Boot Menu Functions
        let bootMenuSelectedIndex = 0;
        let bootMenuItems = [];
        
        function showBootMenu() {{
            const bootMenu = document.getElementById('bootMenu');
            bootMenu.classList.add('show');
            bootMenuItems = Array.from(document.querySelectorAll('.boot-menu-item'));
            bootMenuSelectedIndex = 0;
            updateBootMenuSelection();
            document.getElementById('bootMenuError').classList.remove('show');
        }}
        
        function hideBootMenu() {{
            const bootMenu = document.getElementById('bootMenu');
            bootMenu.classList.remove('show');
        }}
        
        function updateBootMenuSelection() {{
            bootMenuItems.forEach((item, index) => {{
                item.classList.toggle('selected', index === bootMenuSelectedIndex);
            }});
            if (bootMenuItems[bootMenuSelectedIndex]) {{
                bootMenuItems[bootMenuSelectedIndex].scrollIntoView({{ block: 'nearest' }});
            }}
        }}
        
        function selectModelFromBootMenu(modelId) {{
            const item = bootMenuItems.find(el => el.dataset.modelId === modelId);
            if (item && !item.classList.contains('loading')) {{
                bootMenuItems.forEach(i => i.classList.add('loading'));
                document.getElementById('bootMenuError').classList.remove('show');
                
                switchModel(modelId).then(() => {{
                    // Check if model is now available
                    setTimeout(async () => {{
                        try {{
                            const healthResponse = await fetch('/health');
                            const health = await healthResponse.json();
                            if (health.model_available) {{
                                hideBootMenu();
                                bootMenuItems.forEach(i => i.classList.remove('loading'));
                                loadCurrentModel();
                            }} else {{
                                showBootMenuError(health.error || 'Model failed to load');
                                bootMenuItems.forEach(i => i.classList.remove('loading'));
                            }}
                        }} catch (e) {{
                            showBootMenuError('Failed to verify model status');
                            bootMenuItems.forEach(i => i.classList.remove('loading'));
                        }}
                    }}, 3000);
                }}).catch(error => {{
                    showBootMenuError(error.message || 'Failed to switch model');
                    bootMenuItems.forEach(i => i.classList.remove('loading'));
                }});
            }}
        }}
        
        function showBootMenuError(message) {{
            const errorDiv = document.getElementById('bootMenuError');
            errorDiv.textContent = `Error: ${{message}}`;
            errorDiv.classList.add('show');
        }}
        
        // Boot menu keyboard navigation
        document.addEventListener('keydown', function(event) {{
            const bootMenu = document.getElementById('bootMenu');
            if (!bootMenu.classList.contains('show')) return;
            
            if (event.key === 'ArrowDown') {{
                event.preventDefault();
                bootMenuSelectedIndex = (bootMenuSelectedIndex + 1) % bootMenuItems.length;
                updateBootMenuSelection();
            }} else if (event.key === 'ArrowUp') {{
                event.preventDefault();
                bootMenuSelectedIndex = (bootMenuSelectedIndex - 1 + bootMenuItems.length) % bootMenuItems.length;
                updateBootMenuSelection();
            }} else if (event.key === 'Enter') {{
                event.preventDefault();
                const selectedItem = bootMenuItems[bootMenuSelectedIndex];
                if (selectedItem) {{
                    selectModelFromBootMenu(selectedItem.dataset.modelId);
                }}
            }} else if (event.key === 'Escape') {{
                // Don't close boot menu on escape - user must select a model
            }}
        }});
        
        // Check model availability on startup
        async function checkModelAvailability() {{
            try {{
                const response = await fetch('/health');
                const health = await response.json();
                if (!health.model_available) {{
                    showBootMenu();
                }}
            }} catch (e) {{
                console.error('Failed to check model availability:', e);
                showBootMenu();
            }}
        }}
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Page loaded, initializing...');
            
            // Check model availability first
            checkModelAvailability();
            
            // Load current model immediately
            loadCurrentModel();
            
            // Check for model toggle button
            const modelBtn = document.getElementById('modelToggleBtn');
            if (modelBtn) {{
                console.log('Model toggle button found');
            }} else {{
                console.error('Model toggle button not found!');
            }}
            
            const logBtn = document.querySelector('.log-viewer-btn');
            if (logBtn) {{
                console.log('Log button found');
            }} else {{
                console.error('Log button not found!');
            }}
        }});
        
        connectWS();
        loadConversations();
        
        // Load current model on startup
        setTimeout(() => {{
            loadCurrentModel();
        }}, 500);
        
        document.getElementById('input').focus();
    </script>
</body>
</html>
    """

@app.get("/api/models")
async def get_models():
    """Get available models"""
    return {
        "models": AVAILABLE_MODELS,
        "current_model": current_model,
        "switching": model_switching
    }

@app.get("/api/model/status")
async def get_model_status():
    """Get current model and switch status"""
    return {
        "current_model": current_model,
        "switching": model_switching,
        "status": model_switch_status
    }

class ModelSwitchRequest(BaseModel):
    model_id: str

@app.post("/api/model/switch")
async def switch_model_endpoint(request: ModelSwitchRequest):
    """Switch to a different model"""
    # Run switch in background
    import asyncio
    asyncio.create_task(switch_model(request.model_id))
    
    return {"status": "initiated", "message": "Model switch started"}

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
                
                # Check if model is switching
                if model_switching:
                    await websocket.send_json({
                        'type': 'error',
                        'content': 'Model is currently switching. Please wait...'
                    })
                    continue
                
                # Check if requested model matches current model
                if model != current_model:
                    await websocket.send_json({
                        'type': 'error',
                        'content': f'Model {model} is not active. Current model: {current_model}. Please switch models first.'
                    })
                    continue
                
                # Use current client
                if not client:
                    await websocket.send_json({
                        'type': 'error',
                        'content': 'vLLM client not available. Model may be loading...'
                    })
                    continue
                
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
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "current_model": current_model,
        "model_available": False
    }
    
    # Check if current model is working
    if client:
        try:
            models = client.models.list()
            status["model_available"] = True
            status["status"] = "healthy"
        except Exception as e:
            status["model_available"] = False
            status["status"] = "unhealthy"
            status["error"] = str(e)
    else:
        status["model_available"] = False
        status["status"] = "unhealthy"
        status["error"] = "No vLLM client available"
    
    return status

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Chat with vLLM...")
    logger.info("📍 http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
