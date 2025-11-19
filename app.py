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
import re
from contextlib import redirect_stdout, redirect_stderr
from threading import Lock
import httpx
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# Import theme configuration
try:
    from theme_config import COLORS, COLORS_LIGHT, COLORS_DARK, DIMENSIONS, FONTS, ANIMATIONS
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
VLLM_PORT = 8001  # Host port for vLLM

# Log HF_TOKEN status (but not the actual token)
if HF_TOKEN:
    logger.info(f"✓ Hugging Face token loaded ({len(HF_TOKEN)} chars)")
else:
    logger.warning("⚠️ No HF_TOKEN found. Gated models (like Llama) will not work.")
    logger.warning("   Create a .env file with: HF_TOKEN=your_token_here")

# Available models with their vLLM command configurations
# Note: With vLLM latest, AWQ quantization is supported via --quantization awq flag
AVAILABLE_MODELS = [
    {
        "id": "Qwen/Qwen2.5-14B-Instruct",
        "name": "Qwen 2.5 14B Instruct",
        "quantized": False,
        "command": [
            "--model", "Qwen/Qwen2.5-14B-Instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "16384"
        ]
    },
    {
        "id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "name": "Qwen 2.5 Coder 32B",
        "quantized": False,
        "command": [
            "--model", "Qwen/Qwen2.5-Coder-32B-Instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "8192"
        ]
    },
    {
        "id": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "name": "Qwen 2.5 Coder 7B (AWQ Marlin - Fast)",
        "quantized": True,
        "command": [
            "--model", "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "16384",
            "--quantization", "awq_marlin"
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
            "--max-model-len", "8192"
        ]
    },
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
]

# Default model - Qwen 2.5 Coder 7B (AWQ Marlin) - Fast and efficient
# Fits in 24GB GPU with 16K context window
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"

# Current model state
current_model = DEFAULT_MODEL
model_switching = False
model_switch_lock = Lock()
model_switch_status = {"status": "ready", "message": "", "progress": 0}
model_switch_history = []  # List of dicts: {"timestamp": str, "from": str, "to": str, "status": str, "duration": float}
model_containers = {}  # Cache of container names per model: {model_id: container_name}

# Job queue system for efficient model switching
from collections import deque
from asyncio import Queue
import uuid

class Job:
    def __init__(self, prompt: str, model_id: str, websocket: WebSocket, conversation_id: str):
        self.id = str(uuid.uuid4())
        self.prompt = prompt
        self.model_id = model_id
        self.websocket = websocket
        self.conversation_id = conversation_id
        self.created_at = datetime.now()

job_queue = Queue()  # Queue of Job objects
queue_processor_running = False
queue_processor_task = None
selected_models = {}  # Store selected model per conversation: {conversation_id: model_id}

async def process_job_queue():
    """Process jobs from the queue, batching by model to minimize switches"""
    global queue_processor_running, current_model, client
    
    queue_processor_running = True
    logger.info("🚀 Job queue processor started")
    
    while True:
        try:
            # Collect jobs, grouping by model
            jobs_by_model = {}
            batch_timeout = 2.0  # Wait up to 2 seconds to batch requests
            
            # Get first job
            try:
                first_job = await asyncio.wait_for(job_queue.get(), timeout=batch_timeout)
                jobs_by_model[first_job.model_id] = [first_job]
            except asyncio.TimeoutError:
                continue  # No jobs, wait again
            
            # Collect more jobs for the same model (quick batch)
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < 0.5:  # 500ms batching window
                try:
                    job = await asyncio.wait_for(job_queue.get(), timeout=0.1)
                    if job.model_id not in jobs_by_model:
                        jobs_by_model[job.model_id] = []
                    jobs_by_model[job.model_id].append(job)
                except asyncio.TimeoutError:
                    break
            
            # Process each model's batch
            for model_id, jobs in jobs_by_model.items():
                logger.info(f"📦 Processing {len(jobs)} job(s) for model: {model_id}")
                
                # Switch model if needed
                if model_id != current_model:
                    logger.info(f"🔄 Switching to model: {model_id} (current: {current_model})")
                    try:
                        switch_result = await switch_model(model_id)
                        # Check if switch failed (but not recovered)
                        if switch_result.get("status") == "error":
                            logger.warning(f"⚠️ Model switch had issues, but continuing with current state")
                            # If switch failed and we're now on default model, update model_id for jobs
                            if current_model == DEFAULT_MODEL:
                                model_id = DEFAULT_MODEL
                                logger.info(f"🔄 Using default model for queued jobs")
                        elif switch_result.get("status") == "recovered":
                            # Switch failed but we recovered to default
                            model_id = DEFAULT_MODEL
                            logger.info(f"🔄 Using recovered default model for queued jobs")
                    except Exception as switch_error:
                        logger.error(f"❌ Model switch error in queue processor: {switch_error}")
                        # Try to recover
                        try:
                            await recover_to_default_model()
                            model_id = DEFAULT_MODEL
                        except:
                            pass
                        # Continue processing with whatever model is available
                
                # Process all jobs for this model
                for job in jobs:
                    try:
                        await process_single_job(job)
                    except Exception as e:
                        logger.error(f"❌ Error processing job {job.id}: {e}")
                        try:
                            await job.websocket.send_json({
                                'type': 'error',
                                'content': f'Error processing request: {str(e)}'
                            })
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"❌ Queue processor error: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(1)  # Wait before retrying

def estimate_tokens_needed(prompt: str) -> int:
    """Estimate tokens needed based on user request"""
    prompt_lower = prompt.lower()
    
    # Check for explicit length requests
    
    # Look for page counts
    page_matches = re.findall(r'(\d+)\s*pages?', prompt_lower)
    if page_matches:
        pages = int(page_matches[0])
        # Rough estimate: 1 page ≈ 500-800 tokens
        return pages * 700
    
    # Look for word counts
    word_matches = re.findall(r'(\d+)\s*words?', prompt_lower)
    if word_matches:
        words = int(word_matches[0])
        # Rough estimate: 1 word ≈ 1.3 tokens
        return int(words * 1.3)
    
    # Look for token counts
    token_matches = re.findall(r'(\d+)\s*tokens?', prompt_lower)
    if token_matches:
        return int(token_matches[0])
    
    # Look for "long", "detailed", "comprehensive", "extensive"
    if any(word in prompt_lower for word in ['long', 'detailed', 'comprehensive', 'extensive', 'thorough', 'complete']):
        return 20000  # High limit for detailed requests (leaves ~12k for input with 32k context)
    
    # Default based on prompt length
    # Rough estimate: 1 character ≈ 0.25 tokens
    # With 32k context window, we can allow much longer responses
    # Use up to 24k tokens for response (leaving 8k for system + input + overhead)
    estimated = len(prompt) * 0.25 * 10  # 10x multiplier for response
    return min(max(int(estimated), 2048), 24000)  # Between 2k and 24k (leaves ~8k for input)

async def process_single_job(job: Job):
    """Process a single job with the current model"""
    global client
    
    try:
        # Save user message
        save_message(job.conversation_id, 'user', job.prompt)
        
        # Get history with relevance-based selection (pass current query for relevance matching)
        history = get_conversation_history(job.conversation_id, current_query=job.prompt)
        
        # Build messages with system prompt for reasoning
        messages = []
        
        # Enhanced system prompt for deep reasoning with internal validation
        system_prompt = """You are an advanced AI assistant with deep reasoning capabilities. You must think through problems thoroughly and validate your reasoning internally, but only present final, polished conclusions to the user.

CRITICAL REASONING PROTOCOL - INTERNAL REASONING, POLISHED OUTPUT:

**1. INTERNAL REASONING PROCESS:**
   - Think through problems step-by-step internally
   - Validate each logical step before proceeding
   - Question your own assumptions
   - Consider counter-arguments before concluding
   - Verify calculations and logic chains
   - Acknowledge when you're making assumptions
   
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

**4. CODE EXECUTION CAPABILITY:**
   You have access to a sandboxed code execution environment. When writing code:
   - Test your code before presenting it as final
   - When you need to test code, include: [EXECUTE_CODE:python] or [EXECUTE_CODE:javascript] followed by the code
   - The system will execute your code in a secure sandbox and return the output
   - Use this to verify code correctness, test algorithms, and debug issues
   - Always test code examples you provide, especially for programming questions
   - Show the test results in your response: [Test Result: ...] → [Verified: Code works/needs fixes]

**5. FILE SYSTEM ACCESS:**
   You can read files and traverse directories to help with code analysis:
   - Use [READ_FILE:path/to/file] to read file contents
   - Use [LIST_DIR:path/to/directory] to list directory contents
   - This helps you understand codebases, analyze project structure, and provide better programming assistance
   - Always read relevant files when answering questions about code or projects

**6. THINKING STYLE:**
   - Be methodical: reason internally → validate → present polished conclusion
   - Be thorough: don't skip validation steps internally
   - Be self-critical: question your own reasoning before presenting
   - Be clear: write naturally without reasoning brackets or meta-commentary
   - Be confident: present validated conclusions directly

Remember: Do your reasoning internally for quality, but present only polished, final conclusions to the user. Never use reasoning brackets, conclusion markers, or other meta-commentary in your output."""
        
        messages.append({'role': 'system', 'content': system_prompt})
        
        # Use full history (already token-limited by get_conversation_history)
        for msg in history:
            messages.append({'role': msg['role'], 'content': msg['content']})
        messages.append({'role': 'user', 'content': job.prompt})
        
        # Signal start
        await job.websocket.send_json({'type': 'start'})
        
        # Check if model is switching
        if model_switching:
            await job.websocket.send_json({
                'type': 'error',
                'content': 'Model is currently switching. Please wait...'
            })
            return
        
        # Use current client
        if not client:
            await job.websocket.send_json({
                'type': 'error',
                'content': 'vLLM client not available. Model may be loading...'
            })
            return
        
        # Calculate max_tokens based on user request
        max_tokens = estimate_tokens_needed(job.prompt)
        logger.info(f"📤 Processing job {job.id} with model: {current_model}, max_tokens: {max_tokens}")
        
        # Stream from vLLM
        full_response = ""
        
        stream = client.chat.completions.create(
            model=current_model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more deterministic, thoughtful reasoning
            top_p=0.95,  # Higher top_p for more comprehensive exploration of reasoning paths
            frequency_penalty=0.2,  # Moderate penalty to reduce repetition in reasoning
            presence_penalty=0.15,  # Encourage exploring different aspects of the problem
        )
        
        # Function to filter out reasoning brackets and meta-commentary
        def filter_reasoning_text(text):
            """Remove reasoning brackets and meta-commentary from text"""
            if not text:
                return text
            # Remove [Reasoning: ...] → patterns (handles multi-line)
            text = re.sub(r'\[Reasoning:[^\]]+\]\s*→\s*', '', text, flags=re.DOTALL)
            # Remove standalone reasoning brackets
            text = re.sub(r'\[Reasoning:[^\]]+\]', '', text, flags=re.DOTALL)
            # Remove conclusion/statement/analysis brackets
            text = re.sub(r'\[(?:Conclusion|Statement|Analysis|Evaluation|Decision|Verification|Step \d+|Fact|Claim|Acknowledged uncertainty|Verified|Test Result):[^\]]+\]', '', text, flags=re.DOTALL)
            # Remove arrow-only patterns (leftover from reasoning)
            text = re.sub(r'\s*→\s*', ' ', text)
            # Remove any remaining bracket patterns that look like reasoning
            text = re.sub(r'\[[A-Z][a-z]+(?::[^\]]+)?\]', '', text)
            # Clean up extra spaces and newlines
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            text = re.sub(r'\n\s*\n+', '\n\n', text)  # Normalize multiple newlines
            return text.strip()
        
        # Buffer for filtering multi-token patterns during streaming
        stream_buffer = ""
        sent_length = 0  # Track how much filtered content we've already sent
        buffer_size = 300  # Keep last 300 chars for pattern matching
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                stream_buffer += token
                
                # Keep buffer size manageable
                if len(stream_buffer) > buffer_size:
                    # Trim from front, but keep track of what we've sent
                    trim_amount = len(stream_buffer) - buffer_size
                    stream_buffer = stream_buffer[trim_amount:]
                    sent_length = max(0, sent_length - trim_amount)
                
                # Filter the entire buffer to catch multi-token patterns
                filtered_buffer = filter_reasoning_text(stream_buffer)
                
                # Send only the new filtered content that hasn't been sent yet
                new_content = filtered_buffer[sent_length:]
                if new_content:
                    await job.websocket.send_json({
                        'type': 'token',
                        'content': new_content
                    })
                    sent_length = len(filtered_buffer)
        
        logger.info(f"✅ Job {job.id} completed ({len(full_response)} chars)")
        
        # Filter reasoning brackets from full response before processing
        full_response = filter_reasoning_text(full_response)
        
        # Detect and execute code if requested
        code_execution_pattern = r'\[EXECUTE_CODE:(python|javascript|js)\](.*?)(?=\[|$)'
        code_matches = re.finditer(code_execution_pattern, full_response, re.DOTALL)
        
        # Detect file access requests
        file_read_pattern = r'\[READ_FILE:(.+?)\]'
        file_read_matches = re.finditer(file_read_pattern, full_response)
        
        dir_list_pattern = r'\[LIST_DIR:(.+?)\]'
        dir_list_matches = re.finditer(dir_list_pattern, full_response)
        
        for match in code_matches:
            language = match.group(1)
            code_block = match.group(2).strip()
            
            # Extract code from markdown code blocks if present
            code_block = re.sub(r'^```\w*\n', '', code_block)
            code_block = re.sub(r'\n```$', '', code_block)
            code_block = code_block.strip()
            
            if code_block:
                logger.info(f"🔧 Executing {language} code for job {job.id}")
                
                # Execute code
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        exec_response = await client.post(
                            f"http://127.0.0.1:8000/api/execute-code",
                            json={'code': code_block, 'language': language}
                        )
                        exec_result = exec_response.json()
                        
                        # Append execution results to response
                        if exec_result.get('success'):
                            result_text = f"\n\n[Test Result: Success]\n"
                            if exec_result.get('stdout'):
                                result_text += f"Output:\n{exec_result['stdout']}\n"
                            if exec_result.get('result'):
                                result_text += f"Result: {exec_result['result']}\n"
                            full_response += result_text
                            
                            # Send result to client
                            await job.websocket.send_json({
                                'type': 'token',
                                'content': result_text
                            })
                        else:
                            error_text = f"\n\n[Test Result: Error]\n{exec_result.get('error', 'Unknown error')}\n"
                            if exec_result.get('stderr'):
                                error_text += f"Error details: {exec_result['stderr']}\n"
                            full_response += error_text
                            
                            await job.websocket.send_json({
                                'type': 'token',
                                'content': error_text
                            })
                except Exception as e:
                    logger.error(f"Code execution request failed: {e}")
                    error_msg = f"\n\n[Test Result: Execution request failed - {str(e)}]\n"
                    full_response += error_msg
                    await job.websocket.send_json({
                        'type': 'token',
                        'content': error_msg
                    })
        
        # Handle file read requests
        for match in file_read_matches:
            file_path = match.group(1).strip()
            logger.info(f"📖 Reading file: {file_path} for job {job.id}")
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    file_response = await client.get(
                        f"http://127.0.0.1:8000/api/files/read",
                        params={'path': file_path}
                    )
                    if file_response.status_code == 200:
                        file_data = file_response.json()
                        file_text = f"\n\n[File: {file_path}]\n```\n{file_data['content']}\n```\n"
                        full_response += file_text
                        await job.websocket.send_json({
                            'type': 'token',
                            'content': file_text
                        })
                    else:
                        error_text = f"\n\n[File Read Error: {file_response.text}]\n"
                        full_response += error_text
                        await job.websocket.send_json({
                            'type': 'token',
                            'content': error_text
                        })
            except Exception as e:
                logger.error(f"File read request failed: {e}")
                error_msg = f"\n\n[File Read Error: {str(e)}]\n"
                full_response += error_msg
                await job.websocket.send_json({
                    'type': 'token',
                    'content': error_msg
                })
        
        # Handle directory listing requests
        for match in dir_list_matches:
            dir_path = match.group(1).strip()
            logger.info(f"📁 Listing directory: {dir_path} for job {job.id}")
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    dir_response = await client.get(
                        f"http://127.0.0.1:8000/api/files/list",
                        params={'path': dir_path}
                    )
                    if dir_response.status_code == 200:
                        dir_data = dir_response.json()
                        dir_text = f"\n\n[Directory: {dir_path}]\n"
                        for item in dir_data['items']:
                            dir_text += f"{'📁' if item['type'] == 'directory' else '📄'} {item['name']}\n"
                        dir_text += "\n"
                        full_response += dir_text
                        await job.websocket.send_json({
                            'type': 'token',
                            'content': dir_text
                        })
                    else:
                        error_text = f"\n\n[Directory List Error: {dir_response.text}]\n"
                        full_response += error_text
                        await job.websocket.send_json({
                            'type': 'token',
                            'content': error_text
                        })
            except Exception as e:
                logger.error(f"Directory list request failed: {e}")
                error_msg = f"\n\n[Directory List Error: {str(e)}]\n"
                full_response += error_msg
                await job.websocket.send_json({
                    'type': 'token',
                    'content': error_msg
                })
        
        # Save response and get message ID
        assistant_message_id = save_message(job.conversation_id, 'assistant', full_response)
        
        # Send message ID to client for feedback/retry buttons
        await job.websocket.send_json({
            'type': 'message_id',
            'message_id': assistant_message_id
        })
        
        await job.websocket.send_json({'type': 'done'})
        
    except Exception as e:
        logger.error(f"❌ Job processing error: {e}\n{traceback.format_exc()}")
        await job.websocket.send_json({
            'type': 'error',
            'content': f'Error: {str(e)}. Check vLLM is running.'
        })

# Initialize Docker client
try:
    docker_client = docker.from_env()
    logger.info("✓ Docker client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Docker client: {e}")
    docker_client = None

def get_model_container_name(model_id: str) -> str:
    """Get container name for a model (creates if doesn't exist)"""
    # Sanitize model ID for container name
    safe_name = model_id.replace("/", "-").replace("_", "-").lower()
    return f"vllm-{safe_name}"

def ensure_model_container(model_id: str, model_config: dict) -> str:
    """Ensure a container exists for a model, create if needed"""
    container_name = get_model_container_name(model_id)
    
    if not docker_client:
        raise Exception("Docker client not available")
    
    try:
        # Check if container already exists
        container = docker_client.containers.get(container_name)
        logger.info(f"✓ Container {container_name} already exists")
        return container_name
    except docker.errors.NotFound:
        # Container doesn't exist, create it (but don't start it)
        logger.info(f"📦 Creating container for model: {model_id}")
        
        env_vars = {}
        if HF_TOKEN:
            env_vars["HF_TOKEN"] = HF_TOKEN
            env_vars["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN  # vLLM may use either
            logger.info("🔑 Using Hugging Face token for model access")
        else:
            # Check if this is a gated model (Llama models, large Qwen models)
            is_gated = (
                "llama" in model_id.lower() or 
                "meta-llama" in model_id.lower() or
                "72b" in model_id.lower() or
                "70b" in model_id.lower()
            )
            if is_gated:
                logger.warning(f"⚠️ WARNING: No HF_TOKEN set, but model {model_id} may be gated!")
                logger.warning("   You need to:")
                logger.warning("   1. Create a free account at https://huggingface.co/join")
                logger.warning(f"   2. Visit https://huggingface.co/{model_id.replace('/', '/')} and request access if gated")
                logger.warning("   3. Get a token from https://huggingface.co/settings/tokens (Read access)")
                logger.warning("   4. Set HF_TOKEN in your .env file")
        
        container = docker_client.containers.create(
            "vllm/vllm-openai:latest",
            name=container_name,
            command=model_config["command"],
            ports={"8000/tcp": VLLM_PORT},
            environment=env_vars if env_vars else None,
            device_requests=[
                docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])
            ],
            restart_policy={"Name": "unless-stopped"}
        )
        logger.info(f"✓ Container {container_name} created (ID: {container.id[:12]})")
        return container_name

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
        import time
        
        switch_start_time = time.time()
        switch_timestamp = datetime.now().isoformat()
        
        logger.info("=" * 80)
        logger.info(f"🔄 MODEL SWITCH INITIATED")
        logger.info(f"   Timestamp: {switch_timestamp}")
        logger.info(f"   From: {current_model}")
        logger.info(f"   To: {new_model_id} ({model_config['name']})")
        logger.info("=" * 80)
        
        # Log recent switch history
        if model_switch_history:
            logger.info("📜 Recent Model Switch History:")
            for i, hist in enumerate(model_switch_history[-5:], 1):  # Show last 5 switches
                status_icon = "✅" if hist["status"] == "success" else "❌"
                duration_str = f"{hist.get('duration', 0):.1f}s" if hist.get('duration') else "N/A"
                logger.info(f"   {i}. {hist['timestamp']} | {status_icon} {hist['from']} → {hist['to']} ({duration_str})")
        else:
            logger.info("📜 No previous model switches recorded")
        logger.info("-" * 80)
        
        if not docker_client:
            raise Exception("Docker client not available")
        
        # Step 1: Stop all model containers (coordinator pattern)
        model_switch_status["progress"] = 20
        model_switch_status["message"] = "Stopping other model containers..."
        logger.info("📦 Step 1/4: Stopping all model containers...")
        
        stopped_count = 0
        
        # First, stop the legacy "vllm" container if it exists
        try:
            legacy_container = docker_client.containers.get(VLLM_CONTAINER)
            if legacy_container.status == "running":
                logger.info(f"🛑 Stopping legacy container: {VLLM_CONTAINER}")
                legacy_container.stop(timeout=10)
                stopped_count += 1
        except docker.errors.NotFound:
            pass  # Legacy container doesn't exist, that's fine
        except Exception as e:
            logger.warning(f"⚠️ Error stopping legacy container: {e}")
        
        # Stop all model-specific containers
        for model in AVAILABLE_MODELS:
            container_name = get_model_container_name(model["id"])
            try:
                container = docker_client.containers.get(container_name)
                container.reload()  # Refresh status
                if container.status == "running":
                    logger.info(f"🛑 Stopping container: {container_name}")
                    container.stop(timeout=10)
                    stopped_count += 1
            except docker.errors.NotFound:
                pass  # Container doesn't exist, that's fine
            except Exception as e:
                logger.warning(f"⚠️ Error stopping {container_name}: {e}")
        
        # Wait for containers to fully stop and port to be released
        if stopped_count > 0:
            logger.info(f"⏳ Waiting for containers to stop and port {VLLM_PORT} to be released...")
            for i in range(10):  # Wait up to 5 seconds
                time.sleep(0.5)
                # Check if any container is still using port 8001
                port_in_use = False
                try:
                    # Check all containers for port usage
                    all_containers = docker_client.containers.list(all=True)
                    for cont in all_containers:
                        try:
                            cont.reload()
                            if cont.status == "running":
                                # Check port bindings
                                port_bindings = cont.attrs.get('HostConfig', {}).get('PortBindings', {})
                                for port_config in port_bindings.values():
                                    if port_config and any(binding.get('HostPort') == str(VLLM_PORT) for binding in port_config):
                                        port_in_use = True
                                        break
                        except:
                            pass
                except:
                    pass
                
                if not port_in_use:
                    logger.info(f"✓ Port {VLLM_PORT} is now available")
                    break
            else:
                logger.warning(f"⚠️ Port {VLLM_PORT} may still be in use, proceeding anyway...")
        
        logger.info(f"✓ Stopped {stopped_count} container(s)")
        
        # Step 2: Ensure target model container exists
        model_switch_status["progress"] = 40
        model_switch_status["message"] = "Preparing model container..."
        logger.info(f"🔧 Step 2/4: Ensuring container exists for {model_config['name']}...")
        
        target_container_name = ensure_model_container(new_model_id, model_config)
        model_containers[new_model_id] = target_container_name
        
        # Step 3: Start the target container
        model_switch_status["progress"] = 50
        model_switch_status["message"] = "Starting model container..."
        logger.info(f"🚀 Step 3/4: Starting container: {target_container_name}")
        
        try:
            container = docker_client.containers.get(target_container_name)
            
            # Check if already running
            container.reload()
            if container.status == "running":
                logger.info(f"✓ Container {target_container_name} is already running")
            else:
                # Double-check port is available before starting
                logger.info(f"🔍 Verifying port {VLLM_PORT} is available...")
                time.sleep(1)  # Extra wait before starting
                
                try:
                    container.start()
                    logger.info(f"✓ Container {target_container_name} started")
                    time.sleep(2)  # Brief pause for container to initialize
                except docker.errors.APIError as api_err:
                    if "port is already allocated" in str(api_err).lower() or "bind" in str(api_err).lower():
                        logger.error(f"❌ Port {VLLM_PORT} is still in use. Attempting to find and stop conflicting container...")
                        # Try to find what's using the port
                        all_containers = docker_client.containers.list(all=True)
                        for cont in all_containers:
                            try:
                                cont.reload()
                                if cont.status == "running" and cont.name != target_container_name:
                                    port_bindings = cont.attrs.get('HostConfig', {}).get('PortBindings', {})
                                    for port_config in port_bindings.values():
                                        if port_config and any(binding.get('HostPort') == str(VLLM_PORT) for binding in port_config):
                                            logger.info(f"🛑 Found container using port {VLLM_PORT}: {cont.name}, stopping it...")
                                            cont.stop(timeout=10)
                                            time.sleep(2)
                                            # Retry starting
                                            container.start()
                                            logger.info(f"✓ Container {target_container_name} started after clearing port")
                                            time.sleep(2)
                                            break
                            except:
                                pass
                        else:
                            # If we didn't find it, raise the original error
                            raise api_err
                    else:
                        raise api_err
                
        except docker.errors.NotFound:
            logger.error(f"❌ Container {target_container_name} not found after creation")
            raise Exception(f"Container {target_container_name} was not created properly")
        except Exception as e:
            logger.error(f"❌ Error starting container: {e}")
            raise
        
        # Step 4: Wait for model to load
        model_switch_status["progress"] = 60
        model_switch_status["message"] = "Loading new model (this may take a few minutes)..."
        logger.info("⏳ Step 4/4: Waiting for model to load (this may take several minutes)...")
        
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
        logger.info("🔧 Finalizing model switch...")
        
        previous_model = current_model
        client = OpenAI(base_url=VLLM_HOST, api_key="dummy")
        current_model = new_model_id
        
        switch_duration = time.time() - switch_start_time
        
        model_switch_status = {"status": "success", "message": f"Successfully switched to {model_config['name']}", "progress": 100}
        
        # Record successful switch in history
        model_switch_history.append({
            "timestamp": switch_timestamp,
            "from": previous_model,
            "to": new_model_id,
            "status": "success",
            "duration": switch_duration
        })
        # Keep only last 20 switches
        if len(model_switch_history) > 20:
            model_switch_history.pop(0)
        
        logger.info("=" * 80)
        logger.info(f"✅ MODEL SWITCH COMPLETE")
        logger.info(f"   Duration: {switch_duration:.1f} seconds ({switch_duration/60:.1f} minutes)")
        logger.info(f"   Previous model: {previous_model}")
        logger.info(f"   New model: {new_model_id} ({model_config['name']})")
        logger.info(f"   Total switches recorded: {len(model_switch_history)}")
        logger.info("=" * 80)
        
    except Exception as e:
        switch_duration = time.time() - switch_start_time if 'switch_start_time' in locals() else 0
        
        # Record failed switch in history
        model_switch_history.append({
            "timestamp": switch_timestamp if 'switch_timestamp' in locals() else datetime.now().isoformat(),
            "from": current_model,
            "to": new_model_id,
            "status": "failed",
            "duration": switch_duration,
            "error": str(e)
        })
        # Keep only last 20 switches
        if len(model_switch_history) > 20:
            model_switch_history.pop(0)
        
        logger.error("=" * 80)
        logger.error(f"❌ MODEL SWITCH FAILED")
        logger.error(f"   Duration: {switch_duration:.1f} seconds")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Error details: {traceback.format_exc()}")
        logger.error("=" * 80)
        
        # Attempt to recover by loading default model
        logger.info("🔄 Attempting to recover by loading default model...")
        model_switch_status = {"status": "error", "message": f"Switch failed: {str(e)}. Attempting to recover...", "progress": 0}
        
        try:
            # Only attempt recovery if we're not already on the default model
            if current_model != DEFAULT_MODEL:
                logger.info(f"🔄 Attempting to load default model: {DEFAULT_MODEL}")
                recovery_result = await recover_to_default_model()
                if recovery_result.get("success"):
                    model_switch_status = {
                        "status": "recovered",
                        "message": f"Switch failed, but recovered to default model ({DEFAULT_MODEL})",
                        "progress": 100
                    }
                    logger.info("✅ Successfully recovered to default model")
                else:
                    model_switch_status = {
                        "status": "error",
                        "message": f"Switch failed and recovery failed: {recovery_result.get('error', 'Unknown error')}",
                        "progress": 0
                    }
                    logger.error(f"❌ Recovery to default model also failed: {recovery_result.get('error')}")
            else:
                logger.warning("⚠️ Already on default model, cannot recover")
                model_switch_status = {
                    "status": "error",
                    "message": f"Switch failed and already on default model: {str(e)}",
                    "progress": 0
                }
        except Exception as recovery_error:
            logger.error(f"❌ Recovery attempt failed: {recovery_error}\n{traceback.format_exc()}")
            model_switch_status = {
                "status": "error",
                "message": f"Switch failed and recovery failed: {str(recovery_error)}",
                "progress": 0
            }
    finally:
        model_switching = False
    
    return model_switch_status

async def recover_to_default_model():
    """Attempt to recover by loading the default model"""
    global current_model, client, model_switch_status
    
    try:
        logger.info("🔄 Starting recovery to default model...")
        model_switch_status = {"status": "switching", "message": "Recovering to default model...", "progress": 10}
        
        # Find default model config
        default_config = next((m for m in AVAILABLE_MODELS if m["id"] == DEFAULT_MODEL), None)
        if not default_config:
            return {"success": False, "error": f"Default model {DEFAULT_MODEL} not found in available models"}
        
        # Stop all containers first
        if docker_client:
            try:
                # Stop legacy container
                try:
                    legacy_container = docker_client.containers.get(VLLM_CONTAINER)
                    if legacy_container.status == "running":
                        legacy_container.stop(timeout=10)
                except docker.errors.NotFound:
                    pass
                
                # Stop all model containers
                for model in AVAILABLE_MODELS:
                    container_name = get_model_container_name(model["id"])
                    try:
                        container = docker_client.containers.get(container_name)
                        if container.status == "running":
                            container.stop(timeout=10)
                    except docker.errors.NotFound:
                        pass
            except Exception as e:
                logger.warning(f"⚠️ Error stopping containers during recovery: {e}")
        
        # Wait a bit for ports to clear
        import time
        time.sleep(2)
        
        # Ensure default model container exists
        if docker_client:
            target_container_name = ensure_model_container(DEFAULT_MODEL, default_config)
            
            # Start the default model container
            try:
                container = docker_client.containers.get(target_container_name)
                container.reload()
                if container.status != "running":
                    container.start()
                    time.sleep(3)  # Brief wait for container to start
                
                # Wait for model to be ready (allow enough time for model loading)
                max_wait = 600  # 10 minutes for recovery (models can take 5+ minutes to load)
                wait_time = 0
                last_log_check = 0
                
                while wait_time < max_wait:
                    time.sleep(5)
                    wait_time += 5
                    try:
                        container.reload()
                        
                        # Check if container is still running
                        if container.status != "running":
                            logger.warning(f"⚠️ Container {target_container_name} is not running (status: {container.status})")
                            # Try to restart it
                            try:
                                container.start()
                                time.sleep(3)
                                continue
                            except Exception as e:
                                logger.error(f"❌ Failed to restart container: {e}")
                                return {"success": False, "error": f"Container not running: {container.status}"}
                        
                        # Check container logs for loading progress (every 30 seconds)
                        if wait_time - last_log_check >= 30:
                            try:
                                logs = container.logs(tail=20).decode('utf-8', errors='ignore')
                                # Estimate progress based on log content
                                progress_estimate = min(60 + int((wait_time / max_wait) * 35), 95)
                                if 'Starting to load model' in logs or 'Loading model' in logs:
                                    logger.info(f"📦 Model is loading... (checking logs)")
                                    model_switch_status["progress"] = progress_estimate
                                    model_switch_status["message"] = f"Loading model... ({wait_time}s)"
                                elif 'Uvicorn running' in logs or 'Application startup complete' in logs:
                                    logger.info(f"✅ vLLM server started, checking API...")
                                    model_switch_status["progress"] = 90
                                    model_switch_status["message"] = "API server starting..."
                                else:
                                    model_switch_status["progress"] = progress_estimate
                                    model_switch_status["message"] = f"Loading... ({wait_time}s)"
                            except:
                                # Fallback progress estimate
                                progress_estimate = min(60 + int((wait_time / max_wait) * 35), 95)
                                model_switch_status["progress"] = progress_estimate
                                model_switch_status["message"] = f"Loading... ({wait_time}s)"
                            last_log_check = wait_time
                        else:
                            # Update progress even when not checking logs
                            progress_estimate = min(60 + int((wait_time / max_wait) * 35), 95)
                            model_switch_status["progress"] = progress_estimate
                            model_switch_status["message"] = f"Loading... ({wait_time}s)"
                        
                        # Try to connect to the API
                        test_client = OpenAI(base_url=VLLM_HOST, api_key="dummy", timeout=10)
                        test_client.models.list()
                        # Success!
                        client = OpenAI(base_url=VLLM_HOST, api_key="dummy")
                        current_model = DEFAULT_MODEL
                        model_switch_status = {
                            "status": "ready",
                            "current_model": DEFAULT_MODEL,
                            "message": f"Recovered to {DEFAULT_MODEL}"
                        }
                        logger.info(f"✅ Default model recovered and ready! (loaded in {wait_time}s)")
                        return {"success": True, "message": f"Recovered to {DEFAULT_MODEL}"}
                    except Exception as e:
                        # Only log every 30 seconds to avoid spam
                        if wait_time % 30 == 0:
                            logger.info(f"⏳ Recovery: Still loading default model... ({wait_time}s) - {str(e)[:100]}")
                        continue
                
                # Check container status one more time before giving up
                try:
                    container.reload()
                    if container.status == "running":
                        # Container is running but API not responding - might be stuck
                        logger.error(f"❌ Container running but API not responding after {max_wait}s")
                        return {"success": False, "error": f"Model API not responding after {max_wait}s. Container status: {container.status}"}
                    else:
                        return {"success": False, "error": f"Container not running after {max_wait}s. Status: {container.status}"}
                except Exception as e:
                    return {"success": False, "error": f"Default model loading timeout after {max_wait}s: {str(e)}"}
            except Exception as e:
                return {"success": False, "error": f"Failed to start default model container: {str(e)}"}
        else:
            return {"success": False, "error": "Docker client not available for recovery"}
            
    except Exception as e:
        logger.error(f"❌ Recovery error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}

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
                  feedback TEXT,
                  FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')
    
    # Add feedback column if it doesn't exist (for existing databases)
    try:
        c.execute('ALTER TABLE messages ADD COLUMN feedback TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Create indexes for faster queries (especially for conversation history)
    try:
        c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conv_timestamp ON messages(conversation_id, timestamp)')
    except sqlite3.OperationalError:
        pass  # Indexes might already exist
    
    conn.commit()
    conn.close()
    logger.info("✓ Database initialized with indexes")

init_db()

# ==================== Models ====================

class RenameRequest(BaseModel):
    title: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    model: Optional[str] = None

# ==================== Helper Functions ====================

def calculate_message_relevance_score(msg: Dict, current_query: str, message_index: int, total_messages: int) -> float:
    """Calculate relevance and quality score for a message
    
    Args:
        msg: Message dict with role, content, timestamp, feedback, id
        current_query: Current user query for relevance matching
        message_index: Index of message in conversation (0 = oldest, total-1 = newest)
        total_messages: Total number of messages in conversation
    
    Returns:
        Relevance score (higher = more relevant/important)
    """
    score = 0.0
    
    # 1. Recency score (0.0 to 1.0) - more recent = higher, but not the only factor
    recency_ratio = message_index / max(total_messages, 1)
    recency_score = 0.3 * recency_ratio  # Up to 30% of score from recency
    score += recency_score
    
    # 2. Feedback score (quality indicator)
    feedback = msg.get('feedback', '').lower()
    if feedback == 'positive':
        score += 0.4  # High boost for positive feedback
    elif feedback == 'negative':
        score -= 0.2  # Penalty for negative feedback
    
    # 3. Relevance to current query (keyword matching)
    if current_query:
        query_words = set(current_query.lower().split())
        content_lower = msg.get('content', '').lower()
        content_words = set(content_lower.split())
        
        # Calculate word overlap
        common_words = query_words.intersection(content_words)
        if common_words:
            # More overlap = more relevant
            relevance_ratio = len(common_words) / max(len(query_words), 1)
            score += 0.3 * min(relevance_ratio, 1.0)  # Up to 30% from relevance
    
    # 4. Message length score (longer messages often more informative)
    content_length = len(msg.get('content', ''))
    if content_length > 500:  # Substantial messages
        score += 0.1
    if content_length > 1000:  # Very detailed messages
        score += 0.1
    
    # 5. User messages get slight boost (they define the conversation direction)
    if msg.get('role') == 'user':
        score += 0.05
    
    return score

def get_conversation_history(conv_id: str, limit: int = None, max_tokens: int = None, current_query: str = None) -> List[Dict]:
    """Get conversation history, selecting messages by quality and relevance
    
    Args:
        conv_id: Conversation ID
        limit: Maximum number of messages (None = unlimited, use token limit instead)
        max_tokens: Maximum tokens to include (None = auto-calculate based on model context window)
        current_query: Current user query for relevance matching (optional)
    """
    # Auto-calculate max_tokens based on model's context window
    if max_tokens is None:
        # Default: Use 75% of context window for history, leave 25% for system prompt + response
        # For 32k context: 24k for history, 8k for system + response
        # For 16k context: 12k for history, 4k for system + response
        model_max_len = 32768  # Current max-model-len setting
        max_tokens = int(model_max_len * 0.75)  # Use 75% for history
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get all messages with feedback for this conversation
    c.execute('''SELECT id, role, content, timestamp, feedback FROM messages 
                 WHERE conversation_id = ? 
                 ORDER BY timestamp ASC''', (conv_id,))
    
    all_messages = []
    for row in c.fetchall():
        all_messages.append({
            'id': row[0],
            'role': row[1],
            'content': row[2],
            'timestamp': row[3],
            'feedback': row[4] if row[4] else ''
        })
    
    conn.close()
    
    if not all_messages:
        return []
    
    # Strategy: Always include most recent messages, then fill with highest-scoring older messages
    total_messages = len(all_messages)
    min_recent_messages = min(10, total_messages // 4)  # Always include at least 25% most recent, or 10 messages
    
    # Step 1: Always include the most recent messages (they're usually most relevant)
    recent_messages = all_messages[-min_recent_messages:]
    recent_tokens = sum((len(msg['content']) // 4) + 10 for msg in recent_messages)
    
    # Step 2: Score all older messages (excluding the recent ones we already included)
    older_messages = all_messages[:-min_recent_messages] if min_recent_messages < total_messages else []
    
    scored_messages = []
    for idx, msg in enumerate(older_messages):
        score = calculate_message_relevance_score(
            msg, 
            current_query or '', 
            idx, 
            len(older_messages)
        )
        scored_messages.append((score, msg))
    
    # Sort by score (highest first)
    scored_messages.sort(key=lambda x: x[0], reverse=True)
    
    # Step 3: Select highest-scoring messages that fit in remaining token budget
    selected_messages = list(recent_messages)  # Start with recent messages
    total_tokens = recent_tokens
    
    remaining_tokens = max_tokens - total_tokens
    
    for score, msg in scored_messages:
        msg_tokens = (len(msg['content']) // 4) + 10
        if total_tokens + msg_tokens <= max_tokens and remaining_tokens > 0:
            # Insert in chronological order (find where this message should go)
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
    
    # Remove metadata we don't need in the response
    for msg in selected_messages:
        msg.pop('id', None)
        msg.pop('feedback', None)
    
    # If we have a limit and haven't hit token limit, respect message limit
    if limit and len(selected_messages) > limit:
        # Keep most recent messages if we hit the limit
        selected_messages = selected_messages[-limit:]
    
    logger.debug(f"📚 Loaded {len(selected_messages)} messages (~{total_tokens} tokens) from conversation {conv_id} using quality/relevance selection")
    return selected_messages

def search_messages(query: str, limit: int = 20) -> List[Dict]:
    """Search through all messages"""
    if not query or len(query.strip()) < 2:
        return []
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Search in message content (case-insensitive)
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
            'conversation_id': row[0],
            'role': row[1],
            'content': row[2],
            'timestamp': row[3],
            'conversation_title': row[4] or 'Untitled'
        })
    
    conn.close()
    return results

def save_message(conv_id: str, role: str, content: str) -> int:
    """Save message to database and return message ID"""
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
    
    message_id = c.lastrowid
    
    # Update conversation
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
            'id': row[0],
            'conversation_id': row[1],
            'role': row[2],
            'content': row[3],
            'timestamp': row[4],
            'feedback': row[5]
        }
    return None

# ==================== Generate CSS from Theme ====================

def generate_css(mode='light'):
    """Generate CSS from theme configuration"""
    # Select color scheme based on mode
    colors = COLORS_DARK if mode == 'dark' else COLORS_LIGHT
    return f"""
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: {FONTS['family']};
            background: {colors['bg_primary']};
            color: {colors['text_primary']};
            height: 100vh;
            display: flex;
            font-size: {FONTS['size_base']};
        }}
        
        .sidebar {{
            width: {DIMENSIONS['sidebar_width']};
            background: {colors['bg_secondary']};
            border-right: 3px solid {colors['accent_primary']};
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
            width: 32px;
            height: 32px;
            background: {colors['btn_secondary']};
            border: 2px solid {colors['accent_primary']};
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            color: {colors['text_primary']};
            transition: all {ANIMATIONS['transition_speed']};
        }}
        
        .sidebar-toggle:hover {{
            background: {colors['btn_secondary_hover']};
            border-color: {colors['accent_hover']};
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
            border-bottom: 2px solid #8194b1;
        }}
        
        .new-chat-btn {{
            width: 100%;
            padding: 12px;
            background: #e9eced;
            color: #8194b1;
            border: 2px solid #b0c9df;
            border-radius: 4px;
            font-size: {FONTS['size_base']};
            font-weight: 600;
            cursor: pointer;
            transition: all {ANIMATIONS['transition_speed']};
            font-family: {FONTS['family']};
        }}
        
        .new-chat-btn:hover {{ 
            background: #b0c9df; 
            border-color: #8194b1; 
            color: #fff4de;
            transform: translateX(3px);
        }}
        
        .search-container {{
            padding: 15px;
            border-bottom: 2px solid #8194b1;
        }}
        
        .search-input {{
            width: 100%;
            padding: 10px 14px;
            background: #e9eced;
            border: 2px solid #b0c9df;
            border-radius: 4px;
            color: #8194b1;
            font-size: {FONTS['size_small']};
            font-family: {FONTS['family']};
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: #8194b1;
            box-shadow: 0 0 0 2px rgba(129,148,177,0.1);
        }}
        
        .search-input::placeholder {{
            color: #b0c9df;
        }}
        
        .search-results {{
            max-height: 400px;
            overflow-y: auto;
            margin-top: 10px;
        }}
        
        .search-result-item {{
            padding: 12px 16px;
            margin: 5px 0;
            background: #e9eced;
            border: 2px solid #b0c9df;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            color: #8194b1;
        }}
        
        .search-result-item:hover {{
            background: #b0c9df;
            border-color: #8194b1;
            color: #fff4de;
            transform: translateX(3px);
        }}
        
        .search-result-title {{
            font-weight: 600;
            font-size: {FONTS['size_small']};
            color: #8194b1;
            margin-bottom: 4px;
        }}
        
        .search-result-preview {{
            font-size: 12px;
            color: #5898b7;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .search-result-meta {{
            font-size: 11px;
            color: #b0c9df;
            margin-top: 4px;
        }}
        
        .search-no-results {{
            padding: 20px;
            text-align: center;
            color: {COLORS['text_tertiary']};
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
            background: #e9eced;
            border: 2px solid #b0c9df;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #8194b1;
            transition: all 0.2s;
        }}
        
        .conv-item:hover {{ 
            background: #b0c9df; 
            border-color: #8194b1;
            color: #fff4de;
            transform: translateX(3px);
        }}
        .conv-item.active {{ 
            background: #8194b1; 
            color: #fff4de; 
            border-color: #8194b1;
        }}
        
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
        
        .header h1 {{ 
            font-size: {FONTS['size_large']}; 
            font-weight: 600;
            color: #8194b1;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
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
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            align-items: center;
            max-width: 100%;
            background: {colors['bg_primary']};
        }}
        
        .message {{
            width: 100%;
            max-width: {DIMENSIONS['message_max_width']};
            padding: 16px 20px;
            line-height: 1.6;
            animation: slideIn {ANIMATIONS['slide_duration']} ease-out;
            font-size: {FONTS['size_base']};
            border: 2px solid {colors['accent_primary']};
            border-radius: 4px;
            transition: all 0.2s;
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
            border-color: {colors['accent_primary']};
        }}
        
        .message.user:hover {{
            border-color: #8194b1;
            background: #b0c9df;
            color: #fff4de;
        }}
        
        .message.assistant {{
            background: {colors['msg_assistant_bg']};
            color: {colors['msg_assistant_text']};
            text-align: left;
            margin-right: auto;
            position: relative;
            border-color: #8194b1;
        }}
        
        .message.assistant:hover {{
            border-color: #b0c9df;
            box-shadow: 0 4px 12px rgba(129,148,177,0.2);
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
        
        .message.assistant .hljs-keyword {{ color: #8194b1; font-weight: 600; }}
        .message.assistant .hljs-string {{ color: #91b461; }}
        .message.assistant .hljs-comment {{ color: #b0c9df; font-style: italic; }}
        .message.assistant .hljs-number {{ color: #ff7863; }}
        .message.assistant .hljs-function {{ color: #8194b1; }}
        .message.assistant .hljs-variable {{ color: #5898b7; }}
        .message.assistant .hljs-title {{ color: #8194b1; }}
        .message.assistant .hljs-type {{ color: #5898b7; }}
        
        /* Markdown Typography Styles */
        .message.assistant h1 {{
            font-size: 1.8em;
            font-weight: 700;
            color: #8194b1;
            margin: 20px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 3px solid #8194b1;
            line-height: 1.3;
        }}
        
        .message.assistant h2 {{
            font-size: 1.5em;
            font-weight: 700;
            color: #8194b1;
            margin: 18px 0 10px 0;
            padding-bottom: 6px;
            border-bottom: 2px solid #b0c9df;
            line-height: 1.3;
        }}
        
        .message.assistant h3 {{
            font-size: 1.3em;
            font-weight: 600;
            color: #8194b1;
            margin: 16px 0 8px 0;
            line-height: 1.4;
        }}
        
        .message.assistant h4 {{
            font-size: 1.15em;
            font-weight: 600;
            color: #5898b7;
            margin: 14px 0 6px 0;
            line-height: 1.4;
        }}
        
        .message.assistant h5 {{
            font-size: 1.05em;
            font-weight: 600;
            color: #5898b7;
            margin: 12px 0 6px 0;
            line-height: 1.4;
        }}
        
        .message.assistant h6 {{
            font-size: 1em;
            font-weight: 600;
            color: #5898b7;
            margin: 10px 0 4px 0;
            line-height: 1.4;
        }}
        
        .message.assistant p {{
            margin: 10px 0;
            line-height: 1.7;
            color: #8194b1;
        }}
        
        .message.assistant strong,
        .message.assistant b {{
            font-weight: 700;
            color: #8194b1;
            background: rgba(129,148,177,0.1);
            padding: 1px 3px;
            border-radius: 2px;
        }}
        
        .message.assistant em,
        .message.assistant i {{
            font-style: italic;
            color: #5898b7;
        }}
        
        .message.assistant strong em,
        .message.assistant em strong,
        .message.assistant b i,
        .message.assistant i b {{
            font-weight: 700;
            font-style: italic;
            color: #8194b1;
        }}
        
        .message.assistant ul,
        .message.assistant ol {{
            margin: 12px 0;
            padding-left: 30px;
            color: #8194b1;
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
            color: #8194b1;
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
            border-left: 4px solid #8194b1;
            background: #e9eced;
            border-radius: 4px;
            color: #5898b7;
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
            color: #5898b7;
            text-decoration: underline;
            text-decoration-color: #b0c9df;
            transition: all 0.2s;
        }}
        
        .message.assistant a:hover {{
            color: #8194b1;
            text-decoration-color: #8194b1;
            background: rgba(129,148,177,0.1);
            padding: 1px 2px;
            border-radius: 2px;
        }}
        
        .message.assistant hr {{
            border: none;
            border-top: 2px solid #b0c9df;
            margin: 20px 0;
        }}
        
        .message.assistant table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            border: 2px solid #b0c9df;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .message.assistant thead {{
            background: #e9eced;
        }}
        
        .message.assistant th {{
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            color: #8194b1;
            border-bottom: 2px solid #8194b1;
        }}
        
        .message.assistant td {{
            padding: 8px 12px;
            border-bottom: 1px solid #b0c9df;
            color: #8194b1;
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
            background: {colors['bg_quaternary']};
            border-top: 3px solid {colors['accent_primary']};
            display: flex;
            justify-content: center;
        }}
        
        .input-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            max-width: {DIMENSIONS['message_max_width']};
            width: 100%;
            background: {colors['bg_primary']};
            border: 2px solid {colors['accent_primary']};
            border-radius: 4px;
            padding: 12px 16px;
            cursor: text;
        }}
        
        .input-container:focus-within {{
            border-color: {colors['accent_hover']};
            box-shadow: 0 0 0 2px rgba(129,148,177,0.1);
        }}
        
        #input {{
            flex: 1;
            padding: 0;
            background: transparent;
            color: #8194b1;
            border: none;
            border-radius: 0;
            font-size: {FONTS['size_base']};
            resize: none;
            font-family: {FONTS['family']};
            min-height: 24px;
            max-height: 66vh;
            overflow-y: auto;
            line-height: 1.5;
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
            body {{
                flex-direction: column;
                height: auto;
                min-height: 100vh;
            }}
            
            .sidebar {{
                width: 66.67%;
                max-width: 66.67%;
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
                width: 66.67%;
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
            }}
            
            .sidebar.show ~ .sidebar-overlay,
            .sidebar.show + .main::before {{
                display: block;
            }}
            
            .main {{
                width: 100%;
                margin-left: 0;
                padding-top: 70px;
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

# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def home(mode: str = "light"):
    """Serve the web interface"""
    css = generate_css(mode)
    models_json = str(AVAILABLE_MODELS).replace("'", '"')
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
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
            <div class="search-container">
                <input type="text" class="search-input" id="searchInput" placeholder="🔍 Search chats..." oninput="handleSearch(this.value)" />
                <div class="search-results" id="searchResults" style="display: none;"></div>
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
                <button class="web-search-btn" onclick="toggleWebSearch()" title="Search the web">🔍 Web</button>
                <button class="log-viewer-btn" onclick="toggleLogViewer()" title="View Terminal Logs">📋 Logs</button>
                <button class="theme-toggle-btn" id="themeToggle" onclick="toggleTheme()" title="Toggle light/dark mode">🌙</button>
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
                <div class="status-progress-container" id="statusProgressContainer" style="display: none;">
                    <div class="status-progress-bar" id="statusProgressBar"></div>
                    <span class="status-progress-text" id="statusProgressText"></span>
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
            <div class="input-container" onclick="focusInput()">
                <textarea 
                    id="input" 
                    placeholder="Type your message..." 
                    rows="2"
                    onkeydown="handleKeyDown(event)"
                    oninput="autoResizeTextarea(this)"
                ></textarea>
                <button id="send" onclick="sendMessage()" title="Send message"></button>
            </div>
        </div>
    </div>
    
    <div class="log-viewer" id="logViewer">
        <div class="log-viewer-header">
            <h3>Terminal Output</h3>
            <button class="log-viewer-close" onclick="toggleLogViewer()">×</button>
        </div>
        <div class="log-viewer-content" id="logContent"></div>
    </div>
    
    <!-- Web Search Modal -->
    <div class="web-search-modal" id="webSearchModal">
        <div class="web-search-header">
            <h3>🔍 Web Search</h3>
            <button class="web-search-close" onclick="toggleWebSearch()">×</button>
        </div>
        <div class="web-search-content">
            <div class="web-search-form">
                <div class="web-search-input-group">
                    <input type="text" class="web-search-input" id="webSearchInput" placeholder="Search the web..." />
                    <button class="web-search-btn-submit" onclick="performWebSearch()">Search</button>
                </div>
                <div class="web-search-sources">
                    <label class="web-search-source-checkbox">
                        <input type="checkbox" value="google" checked />
                        <span>Google</span>
                    </label>
                    <label class="web-search-source-checkbox">
                        <input type="checkbox" value="wikipedia" checked />
                        <span>Wikipedia</span>
                    </label>
                    <label class="web-search-source-checkbox">
                        <input type="checkbox" value="reddit" checked />
                        <span>Reddit</span>
                    </label>
                    <label class="web-search-source-checkbox">
                        <input type="checkbox" value="github" />
                        <span>GitHub</span>
                    </label>
                    <label class="web-search-source-checkbox">
                        <input type="checkbox" value="stackoverflow" />
                        <span>Stack Overflow</span>
                    </label>
                </div>
            </div>
            <div class="web-search-results" id="webSearchResults"></div>
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
        let logWs = null;
        let currentConvId = generateId();
        let isGenerating = false;
        let renameConvId = null;
        let currentModel = '{DEFAULT_MODEL}';
        let modelAvailable = false;
        let markedReady = false;
        
        // Initialize marked library when ready
        function initMarked() {{
            if (typeof marked !== 'undefined') {{
                markedReady = true;
                marked.setOptions({{
                    breaks: true,
                    gfm: true,
                    headerIds: false,
                    mangle: false,
                    pedantic: false,
                    sanitize: false,
                    smartLists: true,
                    smartypants: true
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
        
        function updateStatus(status, progress = null, message = null) {{
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const progressContainer = document.getElementById('statusProgressContainer');
            const progressBar = document.getElementById('statusProgressBar');
            const progressText = document.getElementById('statusProgressText');
            
            // Update modelAvailable based on status
            if (status === true || status === 'connected') {{
                modelAvailable = true;
                document.getElementById('input').disabled = false;
                document.getElementById('send').disabled = false;
                progressContainer.style.display = 'none';
            }} else if (status === 'booting' || status === 'loading') {{
                modelAvailable = false;
                document.getElementById('input').disabled = true;
                document.getElementById('send').disabled = true;
                progressContainer.style.display = 'block';
            }} else {{
                modelAvailable = false;
                document.getElementById('input').disabled = true;
                document.getElementById('send').disabled = true;
                progressContainer.style.display = 'none';
            }}
            
            // Update status display
            if (status === 'booting' || status === 'loading') {{
                statusDot.className = 'status-dot booting';
                statusText.textContent = message || 'Loading model...';
                
                // Update progress bar
                if (progress !== null && progress !== undefined) {{
                    // Set CSS variable for the ::before pseudo-element
                    progressBar.style.setProperty('--progress', progress + '%');
                    progressText.textContent = progress + '%';
                }} else {{
                    progressBar.style.setProperty('--progress', '0%');
                    progressText.textContent = '';
                }}
            }} else if (status === true) {{
                statusDot.className = 'status-dot';
                statusText.textContent = 'Connected';
                progressContainer.style.display = 'none';
            }} else {{
                statusDot.className = 'status-dot disconnected';
                statusText.textContent = 'Disconnected';
                progressContainer.style.display = 'none';
            }}
        }}
        
        // Model loading progress polling
        let modelProgressInterval = null;
        
        function startModelProgressPolling() {{
            if (modelProgressInterval) {{
                clearInterval(modelProgressInterval);
            }}
            
            modelProgressInterval = setInterval(async () => {{
                try {{
                    // Check health endpoint
                    const healthResponse = await fetch('/health');
                    const health = await healthResponse.json();
                    
                    if (health.model_available) {{
                        updateStatus(true);
                        if (modelProgressInterval) {{
                            clearInterval(modelProgressInterval);
                            modelProgressInterval = null;
                        }}
                        return;
                    }}
                    
                    // Check model status for progress info
                    const statusResponse = await fetch('/api/model/status');
                    const statusData = await statusResponse.json();
                    
                    if (statusData.status && statusData.status.progress !== undefined) {{
                        const progress = statusData.status.progress;
                        const message = statusData.status.message || 'Loading model...';
                        updateStatus('loading', progress, message);
                    }} else {{
                        // Estimate progress based on time (rough estimate)
                        // This is a fallback if no explicit progress is available
                        updateStatus('loading', null, 'Loading model...');
                    }}
                }} catch (e) {{
                    // Silently fail - don't spam console
                }}
            }}, 2000); // Poll every 2 seconds
        }}
        
        function connectWS() {{
            try {{
                ws = new WebSocket(`ws://${{location.host}}/ws/chat`);
                
                ws.onopen = () => {{
                    console.log('Connected');
                    // Check if model is actually available (might be booting)
                    fetch('/health').then(r => r.json()).then(health => {{
                        if (health.model_available) {{
                            modelAvailable = true;
                            updateStatus(true);
                            if (modelProgressInterval) {{
                                clearInterval(modelProgressInterval);
                                modelProgressInterval = null;
                            }}
                        }} else {{
                            modelAvailable = false;
                            updateStatus('loading');
                            // Start polling for progress
                            startModelProgressPolling();
                            // Disable input when model not available
                            document.getElementById('input').disabled = true;
                            document.getElementById('send').disabled = true;
                        }}
                    }}).catch(() => {{
                        updateStatus(true); // Default to connected if health check fails
                    }});
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
                        
                        // Check for model loading messages and estimate progress
                        const logText = data.content.toLowerCase();
                        let estimatedProgress = null;
                        let progressMessage = 'Loading model...';
                        
                        if (logText.includes('starting to load model') || 
                            logText.includes('loading model') ||
                            (logText.includes('model_runner') && logText.includes('load'))) {{
                            estimatedProgress = 20;
                            progressMessage = 'Starting to load model...';
                            updateStatus('loading', estimatedProgress, progressMessage);
                            if (!modelProgressInterval) {{
                                startModelProgressPolling();
                            }}
                        }} else if (logText.includes('weight_utils') && logText.includes('model weights')) {{
                            estimatedProgress = 40;
                            progressMessage = 'Loading model weights...';
                            updateStatus('loading', estimatedProgress, progressMessage);
                        }} else if (logText.includes('initializing') || logText.includes('initialization')) {{
                            estimatedProgress = 60;
                            progressMessage = 'Initializing model...';
                            updateStatus('loading', estimatedProgress, progressMessage);
                        }} else if (logText.includes('uvicorn.run') || logText.includes('application startup complete')) {{
                            estimatedProgress = 90;
                            progressMessage = 'Starting API server...';
                            updateStatus('loading', estimatedProgress, progressMessage);
                        }} else if (logText.includes('api server version')) {{
                            estimatedProgress = 95;
                            progressMessage = 'API server ready, verifying...';
                            updateStatus('loading', estimatedProgress, progressMessage);
                            // Wait a bit then check if actually connected
                            setTimeout(() => {{
                                // Check health to see if model is actually ready
                                fetch('/health').then(r => r.json()).then(health => {{
                                    if (health.model_available) {{
                                        updateStatus(true);
                                        if (modelProgressInterval) {{
                                            clearInterval(modelProgressInterval);
                                            modelProgressInterval = null;
                                        }}
                                    }}
                                }}).catch(() => {{}});
                            }}, 2000);
                        }}
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
        
        function toggleTheme() {{
            const currentMode = localStorage.getItem('theme') || 'light';
            const newMode = currentMode === 'light' ? 'dark' : 'light';
            localStorage.setItem('theme', newMode);
            window.location.href = `/?mode=${{newMode}}`;
        }}
        
        // Set theme icon on load
        document.addEventListener('DOMContentLoaded', function() {{
            const currentMode = localStorage.getItem('theme') || 'light';
            const themeBtn = document.getElementById('themeToggle');
            if (themeBtn) {{
                themeBtn.textContent = currentMode === 'light' ? '🌙' : '☀️';
            }}
        }});
        
        function toggleSidebar() {{
            const sidebar = document.getElementById('sidebar');
            const toggle = document.getElementById('sidebarToggle');
            // On mobile, toggle show/hide instead of collapse
            if (window.innerWidth <= 768) {{
                sidebar.classList.toggle('show');
                // Add/remove overlay
                let overlay = document.querySelector('.sidebar-overlay');
                if (!overlay) {{
                    overlay = document.createElement('div');
                    overlay.className = 'sidebar-overlay';
                    overlay.onclick = () => {{
                        sidebar.classList.remove('show');
                        overlay.remove();
                    }};
                    document.body.appendChild(overlay);
                }}
                if (sidebar.classList.contains('show')) {{
                    overlay.style.display = 'block';
                }} else {{
                    overlay.style.display = 'none';
                }}
            }} else {{
                sidebar.classList.toggle('collapsed');
                toggle.textContent = sidebar.classList.contains('collapsed') ? '📕' : '📖';
            }}
        }}
        
        function toggleLogViewer() {{
            const viewer = document.getElementById('logViewer');
            viewer.classList.toggle('show');
            if (viewer.classList.contains('show') && !logWs) {{
                connectLogWS();
            }}
        }}
        
        function toggleWebSearch() {{
            const modal = document.getElementById('webSearchModal');
            modal.classList.toggle('show');
            if (modal.classList.contains('show')) {{
                document.getElementById('webSearchInput').focus();
            }} else {{
                document.getElementById('webSearchResults').innerHTML = '';
            }}
        }}
        
        async function performWebSearch() {{
            const query = document.getElementById('webSearchInput').value.trim();
            const resultsDiv = document.getElementById('webSearchResults');
            
            if (!query) {{
                resultsDiv.innerHTML = '<div class="web-search-loading">Please enter a search query</div>';
                return;
            }}
            
            // Get selected sources
            const checkboxes = document.querySelectorAll('.web-search-source-checkbox input:checked');
            const sources = Array.from(checkboxes).map(cb => cb.value);
            
            if (sources.length === 0) {{
                resultsDiv.innerHTML = '<div class="web-search-loading">Please select at least one source</div>';
                return;
            }}
            
            resultsDiv.innerHTML = '<div class="web-search-loading">Searching...</div>';
            
            try {{
                const response = await fetch('/api/web-search', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        query: query,
                        sources: sources,
                        max_results: 10
                    }})
                }});
                
                const data = await response.json();
                
                if (data.results && data.results.length > 0) {{
                    let html = `<div style="margin-bottom: 15px; color: {colors['text_secondary']};">Found ${{data.count}} results for "${{data.query}}"</div>`;
                    data.results.forEach(result => {{
                        html += `
                            <div class="web-search-result">
                                <div class="web-search-result-title">
                                    <a href="${{result.url}}" target="_blank">${{result.title}}</a>
                                </div>
                                <div class="web-search-result-url">${{result.url}}</div>
                                <div class="web-search-result-snippet">${{result.snippet}}</div>
                                <span class="web-search-result-source">${{result.source}}</span>
                            </div>
                        `;
                    }});
                    resultsDiv.innerHTML = html;
                }} else {{
                    resultsDiv.innerHTML = '<div class="web-search-loading">No results found</div>';
                }}
            }} catch (error) {{
                resultsDiv.innerHTML = `<div class="web-search-loading">Error: ${{error.message}}</div>`;
            }}
        }}
        
        // Allow Enter key to trigger search
        document.addEventListener('DOMContentLoaded', () => {{
            const searchInput = document.getElementById('webSearchInput');
            if (searchInput) {{
                searchInput.addEventListener('keypress', (e) => {{
                    if (e.key === 'Enter') {{
                        performWebSearch();
                    }}
                }});
            }}
        }});
        
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
        
        function autoResizeTextarea(textarea) {{
            // Reset height to get accurate scrollHeight
            textarea.style.height = 'auto';
            // Calculate new height (min 24px, max 66vh)
            const maxHeight = window.innerHeight * 0.66; // 2/3 of viewport height
            const newHeight = Math.min(Math.max(textarea.scrollHeight, 24), maxHeight);
            textarea.style.height = newHeight + 'px';
        }}
        
        function focusInput() {{
            const input = document.getElementById('input');
            if (input && !input.disabled) {{
                input.focus();
                // Position cursor at end of text
                const len = input.value.length;
                input.setSelectionRange(len, len);
            }}
        }}
        
        function sendMessage() {{
            const input = document.getElementById('input');
            const message = input.value.trim();
            
            // Check if model is available
            if (!modelAvailable) {{
                // Silently prevent submission - don't show error
                return;
            }}
            
            if (!message || !ws || ws.readyState !== WebSocket.OPEN || isGenerating) return;
            
            document.querySelector('.welcome')?.remove();
            
            addMessage(message, 'user');
            input.value = '';
            // Reset textarea height after sending
            autoResizeTextarea(input);
            
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
        
        function downloadAsJSON(content, filename = 'chat-response.json') {{
            const data = {{
                content: content,
                timestamp: new Date().toISOString(),
                type: 'chat_response'
            }};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
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
        
        let searchTimeout = null;
        
        async function handleSearch(query) {{
            const searchResults = document.getElementById('searchResults');
            const conversations = document.getElementById('conversations');
            
            // Clear previous timeout
            if (searchTimeout) {{
                clearTimeout(searchTimeout);
            }}
            
            if (!query || query.trim().length < 2) {{
                searchResults.style.display = 'none';
                conversations.style.display = 'block';
                return;
            }}
            
            // Debounce search
            searchTimeout = setTimeout(async () => {{
                try {{
                    const resp = await fetch(`/api/search?query=${{encodeURIComponent(query.trim())}}`);
                    const data = await resp.json();
                    
                    conversations.style.display = 'none';
                    searchResults.style.display = 'block';
                    searchResults.innerHTML = '';
                    
                    if (data.results && data.results.length > 0) {{
                        // Group results by conversation
                        const grouped = {{}};
                        data.results.forEach(result => {{
                            if (!grouped[result.conversation_id]) {{
                                grouped[result.conversation_id] = [];
                            }}
                            grouped[result.conversation_id].push(result);
                        }});
                        
                        // Display grouped results
                        Object.keys(grouped).forEach(convId => {{
                            const results = grouped[convId];
                            const firstResult = results[0];
                            
                            const item = document.createElement('div');
                            item.className = 'search-result-item';
                            item.onclick = () => {{
                                loadConversation(convId);
                                document.getElementById('searchInput').value = '';
                                searchResults.style.display = 'none';
                                conversations.style.display = 'block';
                            }};
                            
                            const preview = firstResult.content.substring(0, 100);
                            const timestamp = formatTimestamp(firstResult.timestamp);
                            
                            item.innerHTML = `
                                <div class="search-result-title">${{firstResult.conversation_title}}</div>
                                <div class="search-result-preview">${{preview}}${{firstResult.content.length > 100 ? '...' : ''}}</div>
                                <div class="search-result-meta">${{results.length}} match${{results.length > 1 ? 'es' : ''}} • ${{timestamp}}</div>
                            `;
                            
                            searchResults.appendChild(item);
                        }});
                    }} else {{
                        searchResults.innerHTML = '<div class="search-no-results">No results found</div>';
                    }}
                }} catch (e) {{
                    console.error('Search error:', e);
                    searchResults.innerHTML = '<div class="search-no-results">Error searching</div>';
                }}
            }}, 300); // 300ms debounce
        }}
        
        async function loadConversations() {{
            const resp = await fetch('/api/conversations');
            const data = await resp.json();
            
            const container = document.getElementById('conversations');
            container.innerHTML = '';
            
            data.conversations.forEach(conv => {{
                const item = document.createElement('div');
                item.className = 'conv-item conversation-item' + (conv.id === currentConvId ? ' active' : '');
                
                // Desktop: show buttons on hover, Mobile: use swipe gestures
                if (window.innerWidth <= 768) {{
                    // Mobile: swipe actions
                    item.innerHTML = `
                        <div class="conv-title">${{conv.title}}</div>
                        <div class="swipe-actions">
                            <button onclick="event.stopPropagation(); renameConv('${{conv.id}}', '${{conv.title.replace(/'/g, "\\\\'")}}')">Rename</button>
                            <button onclick="event.stopPropagation(); deleteConv('${{conv.id}}')" style="background: #fd7589;">Delete</button>
                        </div>
                    `;
                }} else {{
                    // Desktop: hover buttons
                    item.innerHTML = `
                        <div class="conv-title">${{conv.title}}</div>
                        <div class="conv-actions">
                            <button class="conv-btn" onclick="event.stopPropagation(); renameConv('${{conv.id}}', '${{conv.title.replace(/'/g, "\\\\'")}}')">✏️</button>
                            <button class="conv-btn delete" onclick="event.stopPropagation(); deleteConv('${{conv.id}}')">🗑️</button>
                        </div>
                    `;
                }}
                
                item.onclick = () => loadConversation(conv.id);
                
                // Add swipe gesture handling ONLY for mobile
                if (window.innerWidth <= 768) {{
                    let startX = 0;
                    let currentX = 0;
                    let isDragging = false;
                    
                    item.addEventListener('touchstart', (e) => {{
                        startX = e.touches[0].clientX;
                        isDragging = false;
                    }});
                    
                    item.addEventListener('touchmove', (e) => {{
                        currentX = e.touches[0].clientX;
                        const diff = startX - currentX;
                        
                        if (Math.abs(diff) > 10) {{
                            isDragging = true;
                        }}
                        
                        if (isDragging && diff > 0) {{
                            e.preventDefault();
                            item.style.transform = `translateX(-${{Math.min(diff, 150)}}px)`;
                        }}
                    }});
                    
                    item.addEventListener('touchend', (e) => {{
                        const diff = startX - currentX;
                        
                        if (isDragging && diff > 50) {{
                            item.classList.add('swipe-left');
                            item.style.transform = 'translateX(-150px)';
                        }} else {{
                            item.classList.remove('swipe-left');
                            item.style.transform = '';
                        }}
                        
                        isDragging = false;
                    }});
                }}
                
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
                // Close web search modal if open
                const webSearchModal = document.getElementById('webSearchModal');
                if (webSearchModal && webSearchModal.classList.contains('show')) {{
                    toggleWebSearch();
                    return;
                }}
                
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
                
                // Get current conversation ID (or create new one)
                let convId = currentConversationId;
                if (!convId) {{
                    convId = generateId();
                    currentConversationId = convId;
                }}
                
                // Just select the model (don't switch immediately)
                fetch('/api/model/select', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        model_id: modelId,
                        conversation_id: convId
                    }})
                }}).then(response => response.json()).then(data => {{
                    if (data.status === 'selected') {{
                        hideBootMenu();
                        bootMenuItems.forEach(i => i.classList.remove('loading'));
                        loadCurrentModel();
                        // Show success message
                        const modelName = data.model_name;
                        console.log(`Model ${{modelName}} selected for this session`);
                    }} else {{
                        showBootMenuError(data.message || 'Failed to select model');
                        bootMenuItems.forEach(i => i.classList.remove('loading'));
                    }}
                }}).catch(error => {{
                    showBootMenuError(error.message || 'Failed to select model');
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
        
        // Check model availability on startup (boot menu disabled - using default model)
        async function checkModelAvailability() {{
            try {{
                const response = await fetch('/health');
                const health = await response.json();
                if (!health.model_available) {{
                    // Don't show boot menu - just log the issue
                    console.warn('Model not available, but continuing with default');
                }}
            }} catch (e) {{
                console.error('Failed to check model availability:', e);
                // Don't show boot menu on error either
            }}
        }}
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Page loaded, initializing...');
            
            // Check model availability (but don't show boot menu)
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
        
        // Start checking for model loading progress on page load
        setTimeout(() => {{
            fetch('/health').then(r => r.json()).then(health => {{
                if (!health.model_available) {{
                    updateStatus('loading');
                    startModelProgressPolling();
                }}
            }}).catch(() => {{}});
        }}, 1000);
        
        // Initialize textarea auto-resize
        const input = document.getElementById('input');
        if (input) {{
            autoResizeTextarea(input);
            // Handle window resize to recalculate max height
            window.addEventListener('resize', () => {{
                autoResizeTextarea(input);
            }});
        }}
        
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
        "status": model_switch_status,
        "switch_history": model_switch_history[-10:]  # Last 10 switches
    }

class ModelSwitchRequest(BaseModel):
    model_id: str

@app.post("/api/model/switch")
async def switch_model_endpoint(request: ModelSwitchRequest):
    """Switch to a different model"""
    try:
        # Run switch in background with error handling
        async def switch_with_recovery():
            try:
                result = await switch_model(request.model_id)
                # If switch failed, recovery is already attempted in switch_model
                return result
            except Exception as e:
                logger.error(f"❌ Unhandled error in model switch: {e}\n{traceback.format_exc()}")
                # Final fallback: try to recover to default
                try:
                    await recover_to_default_model()
                except:
                    pass
                raise
        
        asyncio.create_task(switch_with_recovery())
        return {"status": "initiated", "message": "Model switch started"}
    except Exception as e:
        logger.error(f"❌ Error initiating model switch: {e}")
        # Try immediate recovery
        try:
            await recover_to_default_model()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to initiate model switch: {str(e)}")

class SelectModelRequest(BaseModel):
    model_id: str
    conversation_id: str

@app.post("/api/model/select")
async def select_model_endpoint(request: SelectModelRequest):
    """Select a model for a conversation (doesn't switch immediately)"""
    global selected_models
    
    # Validate model exists
    model_config = next((m for m in AVAILABLE_MODELS if m["id"] == request.model_id), None)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Model {request.model_id} not found")
    
    # Store selected model for this conversation
    selected_models[request.conversation_id] = request.model_id
    logger.info(f"📌 Model {request.model_id} selected for conversation {request.conversation_id}")
    
    return {
        "status": "selected",
        "model_id": request.model_id,
        "model_name": model_config["name"],
        "message": f"Model {model_config['name']} selected. It will be used when you send a message."
    }

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

@app.post("/api/message/{message_id}/feedback")
async def submit_feedback(message_id: int, request: dict):
    """Submit feedback for a message (positive or negative)"""
    feedback = request.get('feedback', '').strip()
    if feedback not in ['positive', 'negative']:
        raise HTTPException(status_code=400, detail="Feedback must be 'positive' or 'negative'")
    
    try:
        update_message_feedback(message_id, feedback)
        logger.info(f"📝 Feedback '{feedback}' submitted for message {message_id}")
        return {"status": "success", "message": f"Feedback recorded: {feedback}"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/message/{message_id}/retry")
async def retry_message(message_id: int, request: dict):
    """Retry generating a message with optional extended thinking"""
    extended_thinking = request.get('extended_thinking', False)
    
    try:
        # Get the message to retry
        message = get_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        if message['role'] != 'assistant':
            raise HTTPException(status_code=400, detail="Can only retry assistant messages")
        
        # Get the user message that prompted this response
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
        
        user_message_id, user_prompt = user_row
        
        # Create a new job for retry
        # We'll need to create a WebSocket connection or use a different approach
        # For now, return the prompt and let the client handle it
        return {
            "status": "ready",
            "conversation_id": message['conversation_id'],
            "prompt": user_prompt,
            "extended_thinking": extended_thinking,
            "message": "Ready to retry. Send this prompt again."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying message: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation messages"""
    try:
        history = get_conversation_history(conversation_id, limit=100, current_query=None)
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

@app.get("/api/search")
async def search_chats(query: str):
    """Search through all chat messages"""
    try:
        results = search_messages(query, limit=50)
        return {'results': results, 'count': len(results)}
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        return {'results': [], 'count': 0}

@app.post("/api/web-search")
async def web_search(request: dict):
    """Search the web using DuckDuckGo"""
    query = request.get('query', '').strip()
    sources = request.get('sources', ['google', 'wikipedia', 'reddit'])  # Default to all sources
    max_results = request.get('max_results', 10)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    if max_results > 20:
        max_results = 20  # Limit to 20 results
    
    results = []
    
    try:
        with DDGS() as ddgs:
            # Search with DuckDuckGo (searches across Google, Wikipedia, Reddit, etc.)
            search_results = list(ddgs.text(query, max_results=max_results))
            
            for result in search_results:
                source_type = 'general'
                url = result.get('href', '')
                
                # Determine source type from URL
                if 'wikipedia.org' in url.lower():
                    source_type = 'wikipedia'
                elif 'reddit.com' in url.lower():
                    source_type = 'reddit'
                elif 'github.com' in url.lower():
                    source_type = 'github'
                elif any(x in url.lower() for x in ['stackoverflow.com', 'stackexchange.com']):
                    source_type = 'stackoverflow'
                else:
                    source_type = 'google'  # Default to Google for general web results
                
                # Only include if source is in requested sources, or if 'google' is requested and it's a general result
                if source_type in sources or (source_type == 'general' and 'google' in sources):
                    results.append({
                        'title': result.get('title', ''),
                        'url': url,
                        'snippet': result.get('body', ''),
                        'source': source_type
                    })
        
        return {
            'results': results,
            'count': len(results),
            'query': query
        }
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/execute-code")
async def execute_code(request: dict):
    """Execute code in a sandboxed environment"""
    code = request.get('code', '').strip()
    language = request.get('language', 'python').lower()
    
    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    
    # Security: Limit code length
    if len(code) > 10000:
        raise HTTPException(status_code=400, detail="Code too long (max 10KB)")
    
    try:
        if language == 'python':
            # Use subprocess for better isolation
            import tempfile
            import shutil
            
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute in subprocess with timeout and resource limits
                process = await asyncio.create_subprocess_exec(
                    'python3', temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, 'PYTHONPATH': ''}  # Clear PYTHONPATH for security
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=5.0
                    )
                    
                    stdout_output = stdout.decode('utf-8', errors='replace')
                    stderr_output = stderr.decode('utf-8', errors='replace')
                    
                    return {
                        'success': process.returncode == 0,
                        'stdout': stdout_output,
                        'stderr': stderr_output,
                        'returncode': process.returncode
                    }
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return {
                        'success': False,
                        'error': 'Code execution timed out (5 second limit)',
                        'stdout': '',
                        'stderr': 'Execution timeout'
                    }
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        elif language == 'javascript' or language == 'js':
            # For JavaScript, we'd need Node.js - for now, return not implemented
            return {
                'success': False,
                'error': 'JavaScript execution not yet implemented. Use Python for now.'
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    except Exception as e:
        logger.error(f"Code execution error: {e}\n{traceback.format_exc()}")
        return {
            'success': False,
            'error': f'Execution error: {str(e)}'
        }

@app.get("/api/files/list")
async def list_files(path: str = "."):
    """List files and directories in a given path"""
    try:
        # Security: Prevent directory traversal attacks
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(os.path.abspath('.')):
            raise HTTPException(status_code=403, detail="Access denied: Path outside allowed directory")
        
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
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/read")
async def read_file_content(path: str):
    """Read content of a file"""
    try:
        # Security: Prevent directory traversal attacks
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(os.path.abspath('.')):
            raise HTTPException(status_code=403, detail="Access denied: Path outside allowed directory")
        
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Limit file size (1MB)
        if os.path.getsize(abs_path) > 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 1MB)")
        
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        return {
            'path': path,
            'content': content,
            'size': len(content),
            'lines': content.count('\n') + 1
        }
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not a text file")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket for streaming chat - queues jobs for efficient processing"""
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
            
            # Get selected model for this conversation (or default)
            model_id = selected_models.get(conv_id, DEFAULT_MODEL)
            
            logger.info(f"📥 Queuing message for model: {model_id} (conv: {conv_id})")
                
            # Create job and add to queue
            job = Job(
                prompt=message,
                model_id=model_id,
                websocket=websocket,
                conversation_id=conv_id
            )
            
            await job_queue.put(job)
            logger.info(f"✅ Job {job.id} queued (model: {model_id})")
            
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

async def ensure_model_loaded():
    """Ensure a model is loaded on startup or after errors"""
    global current_model, client
    
    try:
        # Check if current model is actually working
        if client:
            try:
                client.models.list()
                logger.info(f"✓ Model {current_model} is already loaded and working")
                return True
            except:
                logger.warning(f"⚠️ Model {current_model} client exists but not responding")
                client = None
        
        # Try to connect to see if any model is running
        try:
            test_client = OpenAI(base_url=VLLM_HOST, api_key="dummy", timeout=5)
            test_client.models.list()
            client = test_client
            logger.info(f"✓ Found working model at {VLLM_HOST}")
            return True
        except:
            logger.warning(f"⚠️ No model responding at {VLLM_HOST}, will attempt to load default model")
        
        # No working model found, try to load default
        logger.info(f"🔄 No working model found, attempting to load default model: {DEFAULT_MODEL}")
        logger.info("💡 This may take 5-10 minutes if the model needs to load. The web UI will remain available.")
        recovery_result = await recover_to_default_model()
        
        if recovery_result.get("success"):
            logger.info("✅ Default model loaded successfully on startup")
            return True
        else:
            error_msg = recovery_result.get('error', 'Unknown error')
            logger.error(f"❌ Failed to load default model on startup: {error_msg}")
            logger.warning("⚠️ Model health monitor will continue attempting recovery in the background")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error ensuring model is loaded: {e}\n{traceback.format_exc()}")
        return False

async def model_health_monitor():
    """Background task to monitor model health and recover if needed"""
    global client, current_model
    
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Check if model is still working
            if client:
                try:
                    client.models.list()
                    # Model is working
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ Model health check failed: {e}")
                    client = None
            
            # Model not working, attempt recovery
            if not client:
                logger.warning("⚠️ Model not responding, attempting recovery...")
                recovery_result = await recover_to_default_model()
                if recovery_result.get("success"):
                    logger.info("✅ Model recovered successfully")
                else:
                    logger.error(f"❌ Model recovery failed: {recovery_result.get('error')}")
                    
        except Exception as e:
            logger.error(f"❌ Error in model health monitor: {e}")
            await asyncio.sleep(60)  # Wait longer on error

@app.on_event("startup")
async def startup_event():
    """Start the job queue processor and ensure model is loaded on app startup"""
    global queue_processor_task
    
    # Start job queue processor immediately (non-blocking)
    if not queue_processor_running:
        queue_processor_task = asyncio.create_task(process_job_queue())
        logger.info("✅ Job queue processor task created")
    
    # Start model health monitor
    asyncio.create_task(model_health_monitor())
    logger.info("✅ Model health monitor started")
    
    # Check for model in background (don't block startup)
    # This allows the web UI to load immediately
    asyncio.create_task(ensure_model_loaded())
    logger.info("🔄 Checking for model availability in background...")

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Chat with vLLM...")
    logger.info("📍 http://0.0.0.0:8000")
    # Start queue processor
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(process_job_queue())
    uvicorn.run(app, host="0.0.0.0", port=8000)
