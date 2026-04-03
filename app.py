#!/usr/bin/env python3
"""
Coding companion with vLLM — short technical answers with code.

Repo map (start here if you are new):
  app.py           — FastAPI routes, WebSockets, SQLite helpers, vLLM httpx client
  themes.py        — Light/dark palettes → CSS variables for static/style.css
  prompts.py       — Default system prompt text
  thinking_stream.py — Split model output into "thinking" vs visible answer for the UI
  static/          — index.html, app.js, style.css (vanilla front end)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import pathlib
import pty
import re
import shlex
import shutil
import socket
import sqlite3
import struct
import sys
import signal
import tempfile
import traceback
import uuid
import zipfile
from html import unescape
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse, urlunparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import fcntl
import termios

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    import pandas as pd
except Exception:
    pd = None

from prompts import (
    CONVERSATION_SUMMARY_SYSTEM_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    DEEP_BUILD_SYSTEM_PROMPT,
    DEEP_DIRECT_SYSTEM_PROMPT,
    DEEP_INSPECT_SYSTEM_PROMPT,
    DEEP_SYNTHESIZE_SYSTEM_PROMPT,
    DEEP_VERIFY_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    TOOL_USE_SYSTEM_PROMPT,
)
from themes import COLORS_DARK, COLORS_LIGHT
from thinking_stream import ThinkingStreamSplitter, strip_stream_special_tokens

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

# --- Runtime configuration (override with env vars; see docs/configuration.md) ---
DB_PATH = "/app/data/chat.db"
VLLM_HOST = os.getenv("VLLM_HOST", "http://vllm:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-14B-AWQ")
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/cache/huggingface")
WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT", str(pathlib.Path("/app/workspaces")))
PET_ROOT = os.getenv("PET_ROOT", str(pathlib.Path("/app/pet")))
RUNS_ROOT = os.getenv("RUNS_ROOT", str(pathlib.Path("/app/runs")))
VOICE_ROOT = os.getenv("VOICE_ROOT", str(pathlib.Path("/app/data/voice")))
MODEL_STATE_PATH = os.getenv("MODEL_STATE_PATH", "/app/data/model_state.json")
MODEL_14B_NAME = os.getenv("MODEL_14B_NAME", MODEL_NAME)
MODEL_14B_ARGS = os.getenv(
    "MODEL_14B_ARGS",
    "--gpu-memory-utilization 0.95 --max-model-len 8192 --enable-prefix-caching "
    "--max-num-seqs 16 --enable-chunked-prefill --quantization awq_marlin "
    "--trust-remote-code --enforce-eager",
)
MODEL_8B_NAME = os.getenv("MODEL_8B_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
MODEL_8B_ARGS = os.getenv(
    "MODEL_8B_ARGS",
    "--gpu-memory-utilization 0.90 --max-model-len 8192 --enable-prefix-caching "
    "--max-num-seqs 16 --enable-chunked-prefill --enforce-eager",
)
CUSTOM_MODEL_ARGS = os.getenv("CUSTOM_MODEL_ARGS", MODEL_14B_ARGS)
DEFAULT_MODEL_PROFILE = os.getenv("DEFAULT_MODEL_PROFILE", "14b").strip().lower()
DEEP_CRITIQUE_ENABLED = os.getenv("DEEP_CRITIQUE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
TOOL_LOOP_ENABLED = os.getenv("TOOL_LOOP_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
TOOL_LOOP_MAX_STEPS = int(os.getenv("TOOL_LOOP_MAX_STEPS", "6"))
AUTO_VERIFY_AFTER_PATCH = os.getenv("AUTO_VERIFY_AFTER_PATCH", "1").strip().lower() not in {"0", "false", "no"}
AUTO_VERIFY_MAX_RUNS = max(0, int(os.getenv("AUTO_VERIFY_MAX_RUNS", "2")))
WORKSPACE_FILE_SIZE_LIMIT = 1024 * 1024
WORKSPACE_WRITE_SIZE_LIMIT = 1024 * 1024
COMMAND_TIMEOUT_SECONDS = float(os.getenv("COMMAND_TIMEOUT_SECONDS", "8"))
COMMAND_OUTPUT_LIMIT = int(os.getenv("COMMAND_OUTPUT_LIMIT", "12000"))
TOOL_RESULT_TEXT_LIMIT = int(os.getenv("TOOL_RESULT_TEXT_LIMIT", "40000"))
WEB_PAGE_TEXT_LIMIT = int(os.getenv("WEB_PAGE_TEXT_LIMIT", "16000"))
WEB_FETCH_PAGE_MAX_PER_TURN = max(1, int(os.getenv("WEB_FETCH_PAGE_MAX_PER_TURN", "3")))
UPLOAD_FILE_SIZE_LIMIT = 10 * 1024 * 1024
VOICE_INPUT_SIZE_LIMIT = int(os.getenv("VOICE_INPUT_SIZE_LIMIT", str(15 * 1024 * 1024)))
VOICE_COMMAND_TIMEOUT_SECONDS = float(os.getenv("VOICE_COMMAND_TIMEOUT_SECONDS", "180"))
VOICE_TTS_COMMAND = os.getenv("VOICE_TTS_COMMAND", "").strip()
VOICE_STT_COMMAND = os.getenv("VOICE_STT_COMMAND", "").strip()
VOICE_STT_LANGUAGE = os.getenv("VOICE_STT_LANGUAGE", "en").strip() or "en"
VOICE_TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "Samantha").strip() or "Samantha"
MAX_ATTACHMENTS_PER_MESSAGE = 8
SPREADSHEET_PREVIEW_ROWS = 8
SPREADSHEET_MAX_COLUMNS = 40
SPREADSHEET_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm"}
WORKSPACE_SIGNAL_VERBS = {
    "inspect", "read", "open", "show", "list", "search", "find", "grep",
    "edit", "change", "update", "patch", "refactor", "create", "write", "add",
    "delete", "remove", "rename", "run", "execute", "test", "build", "compile",
    "debug", "fix", "implement",
}
WORKSPACE_SIGNAL_NOUNS = {
    "workspace", "repo", "repository", "codebase", "project", "folder", "directory",
    "file", "files", "code", "app", "module", "script", "source", "test", "tests",
}
WORKSPACE_TEMPLATE_TERMS = {
    "template", "starter", "scaffold", "boilerplate", "example", "sample",
    "saas", "mvp", "skeleton", "seed", "bootstrap", "generate",
}


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


DOCKER_CONTROL_ENABLED = env_flag("DOCKER_CONTROL_ENABLED", False)
INTERACTIVE_TERMINAL_ENABLED = env_flag("INTERACTIVE_TERMINAL_ENABLED", False)
EXECUTE_CODE_ENABLED = env_flag("EXECUTE_CODE_ENABLED", False)
STRICT_WORKSPACE_COMMAND_PATHS = env_flag("STRICT_WORKSPACE_COMMAND_PATHS", True)

MODEL_PROFILES: Dict[str, Dict[str, str]] = {
    "14b": {
        "key": "14b",
        "label": "14B",
        "name": MODEL_14B_NAME,
        "args": MODEL_14B_ARGS,
    },
    "8b": {
        "key": "8b",
        "label": "8B",
        "name": MODEL_8B_NAME,
        "args": MODEL_8B_ARGS,
    },
}
if DEFAULT_MODEL_PROFILE not in MODEL_PROFILES:
    DEFAULT_MODEL_PROFILE = "14b"


def _read_model_state_payload() -> Dict[str, Any]:
    """Load persisted model runtime state from disk."""
    try:
        with open(MODEL_STATE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _read_selected_model_state() -> tuple[str, Optional[str], Dict[str, Any], Dict[str, Any]]:
    """Load selected model, custom model name, load history, and loading state."""
    payload = _read_model_state_payload()
    selected = str(payload.get("active_profile", "")).strip().lower()
    custom_model_name = str(payload.get("custom_model_name", "")).strip() or None
    if selected not in MODEL_PROFILES and not custom_model_name:
        selected = DEFAULT_MODEL_PROFILE
    elif selected not in MODEL_PROFILES and custom_model_name:
        selected = "custom"

    load_history = payload.get("load_history")
    if not isinstance(load_history, dict):
        load_history = {}

    loading = payload.get("loading")
    if not isinstance(loading, dict):
        loading = {}

    return selected or DEFAULT_MODEL_PROFILE, custom_model_name, load_history, loading


ACTIVE_MODEL_PROFILE, ACTIVE_CUSTOM_MODEL_NAME, MODEL_LOAD_HISTORY, MODEL_LOADING_STATUS = _read_selected_model_state()
ACTIVE_MODEL_LOCK: asyncio.Lock | None = None
TERMINAL_SESSIONS: Dict[str, TerminalSession] = {}
TERMINAL_LOCKS: Dict[str, asyncio.Lock] = {}
MODEL_DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
MODEL_DOWNLOAD_LOCK: asyncio.Lock | None = None
COMMAND_APPROVAL_WAITERS: Dict[str, Dict[str, Any]] = {}


def build_model_profile(key: str, name: str, args: str, label: Optional[str] = None) -> Dict[str, str]:
    """Create a model profile dictionary."""
    return {
        "key": key,
        "label": label or key.upper(),
        "name": name,
        "args": args,
    }


def get_active_model_profile() -> Dict[str, str]:
    """Return the selected profile metadata."""
    profile = MODEL_PROFILES.get(ACTIVE_MODEL_PROFILE)
    if profile:
        return profile
    if ACTIVE_CUSTOM_MODEL_NAME:
        return build_model_profile("custom", ACTIVE_CUSTOM_MODEL_NAME, CUSTOM_MODEL_ARGS, label="Custom")
    return MODEL_PROFILES[DEFAULT_MODEL_PROFILE]


def get_active_model_name() -> str:
    """Return the model name currently selected for vLLM requests."""
    return get_active_model_profile()["name"]


def is_fast_profile_active() -> bool:
    """Return whether the lighter-weight model profile is active."""
    return get_active_model_profile()["key"] == "8b"


def persist_model_state():
    """Persist selected model metadata plus load history to disk."""
    os.makedirs(os.path.dirname(MODEL_STATE_PATH), exist_ok=True)
    payload = {
        "active_profile": ACTIVE_MODEL_PROFILE,
        "custom_model_name": ACTIVE_CUSTOM_MODEL_NAME,
        "load_history": MODEL_LOAD_HISTORY,
        "loading": MODEL_LOADING_STATUS,
        "updated_at": datetime.now().isoformat(),
    }
    with open(MODEL_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def persist_active_model_selection(profile_key: str, custom_model_name: Optional[str] = None):
    """Persist the selected model/profile so the UI survives restarts."""
    global ACTIVE_MODEL_PROFILE, ACTIVE_CUSTOM_MODEL_NAME
    ACTIVE_MODEL_PROFILE = profile_key
    ACTIVE_CUSTOM_MODEL_NAME = (custom_model_name or "").strip() or None
    persist_model_state()


def build_vllm_command(profile_key: Optional[str] = None, model_name: Optional[str] = None) -> List[str]:
    """Build the vLLM argv for a profile or custom model."""
    if model_name:
        profile = build_model_profile(profile_key or "custom", model_name, CUSTOM_MODEL_ARGS, label="Custom")
    else:
        profile = MODEL_PROFILES[profile_key or DEFAULT_MODEL_PROFILE]
    return [
        "--model", profile["name"],
        "--host", "0.0.0.0",
        "--port", "8000",
        *shlex.split(profile["args"]),
    ]


logger.info(f"Using vLLM at {VLLM_HOST} with model {get_active_model_name()}")
logger.info("Available model profiles: %s", ", ".join(f"{key}={value['name']}" for key, value in MODEL_PROFILES.items()))
logger.info("Workspace root: %s", WORKSPACE_ROOT)

HOSTNAME_ADJECTIVES = (
    "Bright", "Clever", "Cosmic", "Daring", "Electric", "Golden", "Lucky",
    "Merry", "Nova", "Radiant", "Rocket", "Sonic", "Spark", "Starlit",
    "Sunny", "Swift", "Velvet", "Wild",
)
HOSTNAME_NOUNS = (
    "Anchor", "Atlas", "Beacon", "Camp", "Canvas", "Comet", "Cove", "Den",
    "Forge", "Garden", "Grove", "Harbor", "Hideout", "Lab", "Loft", "Nest",
    "Orbit", "Outpost", "Studio", "Trail",
)
HOSTNAME_SKIP_TOKENS = {
    "local", "localhost", "home", "host", "hostname", "node", "server", "srv",
    "desktop", "laptop", "computer", "machine", "mac", "macbook", "macbookpro",
    "macbook-air", "macmini", "imac", "pc", "pro", "air", "mini",
    "linux", "ubuntu", "debian",
    "fedora", "arch", "windows",
}


def _title_case_token(token: str) -> str:
    """Render a hostname token as a human-readable title fragment."""
    if token.isupper() and len(token) <= 4:
        return token
    if token.isdigit():
        return token
    return token[:1].upper() + token[1:].lower()


def build_dynamic_app_title(hostname: str) -> str:
    """Turn a raw hostname into a friendlier, deterministic codename."""
    raw = (hostname or "").strip()
    if not raw:
        return "Lucky Harbor"

    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    adjective = HOSTNAME_ADJECTIVES[digest[0] % len(HOSTNAME_ADJECTIVES)]
    fallback_noun = HOSTNAME_NOUNS[digest[1] % len(HOSTNAME_NOUNS)]

    parts = [part for part in re.split(r"[^A-Za-z0-9]+", raw) if part]
    meaningful_parts = []
    numeric_parts = []
    for part in parts:
        lowered = part.lower()
        if part.isdigit():
            numeric_parts.append(part)
            continue
        if len(part) <= 2 or lowered in HOSTNAME_SKIP_TOKENS:
            continue
        if re.fullmatch(r"[a-f0-9]{6,}", lowered):
            continue
        meaningful_parts.append(_title_case_token(part))

    anchor = " ".join(meaningful_parts[:2]).strip() or fallback_noun
    suffix = numeric_parts[-1][-2:] if numeric_parts else f"{digest[2] % 100:02d}"
    return f"{adjective} {anchor} {suffix}"


RAW_HOSTNAME = socket.gethostname() or "localhost"
APP_TITLE = "Compy"

# Completion budget: ceiling only. The model still ends the stream when it predicts EOS
# (end of sequence); it does not "fill" unused max_tokens. Tune via env if replies truncate.
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "4096"))

WORKSPACE_ROOT_PATH = pathlib.Path(WORKSPACE_ROOT).resolve()
WORKSPACE_ROOT_PATH.mkdir(parents=True, exist_ok=True)
PET_ROOT_PATH = pathlib.Path(PET_ROOT).resolve()
PET_ROOT_PATH.mkdir(parents=True, exist_ok=True)
RUNS_ROOT_PATH = pathlib.Path(RUNS_ROOT).resolve()
RUNS_ROOT_PATH.mkdir(parents=True, exist_ok=True)
VOICE_ROOT_PATH = pathlib.Path(VOICE_ROOT).resolve()
VOICE_ROOT_PATH.mkdir(parents=True, exist_ok=True)
PIPER_DEFAULT_MODEL = pathlib.Path(
    os.getenv("PIPER_MODEL", str(VOICE_ROOT_PATH / "models" / "en_US-lessac-high.onnx"))
)


def sanitize_conversation_id(conversation_id: str) -> str:
    """Restrict conversation IDs to a safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", (conversation_id or "").strip())
    return cleaned[:120] or "default"


def sanitize_uploaded_filename(filename: str) -> str:
    """Normalize uploaded filenames while preserving an extension when possible."""
    raw_name = pathlib.Path((filename or "").strip()).name
    if not raw_name:
        raw_name = "attachment"

    suffix = pathlib.Path(raw_name).suffix[:16]
    stem = pathlib.Path(raw_name).stem or "attachment"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80] or "attachment"
    safe_name = f"{stem}{suffix}"
    return safe_name[:120]


def ensure_unique_workspace_filename(workspace: pathlib.Path, filename: str) -> str:
    """Avoid clobbering an existing workspace file during upload."""
    candidate = sanitize_uploaded_filename(filename)
    path = workspace / candidate
    if not path.exists():
        return candidate

    stem = pathlib.Path(candidate).stem
    suffix = pathlib.Path(candidate).suffix
    for index in range(2, 1000):
        candidate = f"{stem}-{index}{suffix}"
        if not (workspace / candidate).exists():
            return candidate
    raise HTTPException(status_code=500, detail="Unable to allocate a unique filename")


def sanitize_relative_workspace_path(path_value: str, fallback: str = "snippet.txt") -> str:
    """Normalize a user/model-suggested relative path into safe workspace components."""
    raw = str(path_value or "").strip().replace("\\", "/")
    if not raw:
        return sanitize_uploaded_filename(fallback)

    parts = []
    for part in raw.split("/"):
        cleaned = part.strip()
        if not cleaned or cleaned in {".", ".."}:
            continue
        parts.append(sanitize_uploaded_filename(cleaned))

    if not parts:
        return sanitize_uploaded_filename(fallback)
    return "/".join(parts[:8])


RECOVERED_CODE_EXTENSIONS = {
    "bash": ".sh",
    "c": ".c",
    "cpp": ".cpp",
    "csharp": ".cs",
    "css": ".css",
    "csv": ".csv",
    "go": ".go",
    "html": ".html",
    "java": ".java",
    "javascript": ".js",
    "js": ".js",
    "json": ".json",
    "jsx": ".jsx",
    "markdown": ".md",
    "md": ".md",
    "python": ".py",
    "py": ".py",
    "ruby": ".rb",
    "rust": ".rs",
    "shell": ".sh",
    "sql": ".sql",
    "text": ".txt",
    "toml": ".toml",
    "ts": ".ts",
    "tsx": ".tsx",
    "typescript": ".ts",
    "xml": ".xml",
    "yaml": ".yml",
    "yml": ".yml",
}
RECOVERED_CODE_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
RECOVERED_FILENAME_HINT_RE = re.compile(
    r"(?:save(?:\s+it)?\s+as|write(?:\s+this)?\s+to|file(?:name)?|named|call(?:ed)?|create\s+)"
    r"\s*`?([A-Za-z0-9][A-Za-z0-9._/\-]{0,140}\.[A-Za-z0-9]{1,12})`?",
    re.IGNORECASE,
)


def infer_code_extension(language_hint: str) -> str:
    """Map a fenced-code language token to a common file extension."""
    token = str(language_hint or "").strip().lower()
    if not token:
        return ".txt"
    token = token.split()[0]
    return RECOVERED_CODE_EXTENSIONS.get(token, ".txt")


def infer_filename_from_text(text: str) -> Optional[str]:
    """Extract a likely filename reference from prose near a code block."""
    for match in RECOVERED_FILENAME_HINT_RE.finditer(text or ""):
        candidate = match.group(1)
        if candidate and "://" not in candidate:
            return sanitize_relative_workspace_path(candidate)

    for candidate in re.findall(r"`([A-Za-z0-9][A-Za-z0-9._/\-]{0,140}\.[A-Za-z0-9]{1,12})`", text or ""):
        if "://" not in candidate:
            return sanitize_relative_workspace_path(candidate)
    return None


def infer_filename_for_code_block(info: str, context_before: str, full_message: str, index: int, total_blocks: int) -> str:
    """Choose a stable filename for a recovered fenced code block."""
    info = str(info or "").strip()
    info_tokens = [token.strip("()[]{}<>,;:") for token in info.split() if token.strip()]

    if len(info_tokens) > 1:
        for token in info_tokens[1:]:
            if "." in token and "://" not in token:
                return sanitize_relative_workspace_path(token)
    elif info_tokens:
        lone = info_tokens[0]
        if "." in lone and "://" not in lone:
            return sanitize_relative_workspace_path(lone)

    context_candidate = infer_filename_from_text(context_before)
    if context_candidate:
        return context_candidate

    if total_blocks == 1:
        message_candidate = infer_filename_from_text(full_message)
        if message_candidate:
            return message_candidate

    extension = infer_code_extension(info_tokens[0] if info_tokens else "")
    return sanitize_relative_workspace_path(f"snippet-{index}{extension}")


def strip_fenced_code_blocks(markdown_text: str) -> str:
    """Remove fenced code blocks so notes can keep the surrounding explanation."""
    return RECOVERED_CODE_BLOCK_RE.sub("", markdown_text or "")


def build_recovered_notes_markdown(
    source_label: str,
    created_paths: List[str],
    message_content: str,
    recovered_at: str,
) -> str:
    """Create a README capturing the non-code guidance from a recovered chat response."""
    notes_body = strip_fenced_code_blocks(message_content).strip()
    lines = [
        "# Recovered Chat Output",
        "",
        f"- Source: {source_label}",
        f"- Recovered at: {recovered_at}",
        "",
        "## Files",
    ]
    if created_paths:
        lines.extend(f"- `{path}`" for path in created_paths)
    else:
        lines.append("- No fenced code blocks were found in the source message.")

    lines.extend([
        "",
        "## Notes",
        notes_body or "No separate prose notes were present in the source message.",
        "",
        "## Review",
        "Treat this folder as a staging area. Review the recovered files here before moving them into the main repo.",
        "",
    ])
    return "\n".join(lines)


def utcnow_iso() -> str:
    """Return a stable timestamp string for persisted records."""
    return datetime.now().isoformat()


def get_pet_root() -> pathlib.Path:
    """Return the persistent pet home directory."""
    return PET_ROOT_PATH


def ensure_pet_dirs() -> pathlib.Path:
    """Create the persistent pet home layout if missing."""
    pet_root = get_pet_root()
    for rel_path in (
        "state",
        "memory",
        "capabilities",
        "capabilities/scripts",
        "capabilities/prompts",
        "capabilities/templates",
        "capabilities/workflows",
        "assets",
        "journals",
    ):
        (pet_root / rel_path).mkdir(parents=True, exist_ok=True)
    return pet_root


def get_run_root(run_id: str, create: bool = True) -> pathlib.Path:
    """Return the root directory for a run."""
    run_root = (RUNS_ROOT_PATH / sanitize_conversation_id(run_id)).resolve()
    if RUNS_ROOT_PATH not in run_root.parents and run_root != RUNS_ROOT_PATH:
        raise ValueError("Run path escaped runs root")
    if create:
        run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def get_run_workspace_root(run_id: str, create: bool = True) -> pathlib.Path:
    """Return the workspace directory for a run."""
    workspace = (get_run_root(run_id, create=create) / "workspace").resolve()
    if create:
        workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def build_run_id(conversation_id: str) -> str:
    """Derive a stable run id from a conversation id."""
    return f"run-{sanitize_conversation_id(conversation_id)}"


def get_run_record(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Return the run metadata for a conversation, if any."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''SELECT id, conversation_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count
           FROM runs
           WHERE conversation_id = ?''',
        (conversation_id,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "conversation_id": row[1],
        "title": row[2] or "",
        "status": row[3] or "active",
        "sandbox_path": row[4],
        "started_at": row[5],
        "ended_at": row[6],
        "summary": row[7] or "",
        "promoted_count": int(row[8] or 0),
    }


def ensure_run_for_conversation(conversation_id: str, title: Optional[str] = None) -> Dict[str, Any]:
    """Create or fetch the run record that owns a conversation's workspace."""
    existing = get_run_record(conversation_id)
    if existing:
        workspace = get_run_workspace_root(existing["id"], create=True)
        if existing["sandbox_path"] != str(workspace):
            conn = sqlite3.connect(DB_PATH)
            conn.execute('UPDATE runs SET sandbox_path = ? WHERE id = ?', (str(workspace), existing["id"]))
            conn.commit()
            conn.close()
            existing["sandbox_path"] = str(workspace)
        return existing

    run_id = build_run_id(conversation_id)
    workspace = get_run_workspace_root(run_id, create=True)
    now = utcnow_iso()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''INSERT OR REPLACE INTO runs
           (id, conversation_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (run_id, conversation_id, title or "", "active", str(workspace), now, None, "", 0),
    )
    try:
        conn.execute('UPDATE conversations SET run_id = ? WHERE id = ?', (run_id, conversation_id))
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    return get_run_record(conversation_id) or {
        "id": run_id,
        "conversation_id": conversation_id,
        "title": title or "",
        "status": "active",
        "sandbox_path": str(workspace),
        "started_at": now,
        "ended_at": None,
        "summary": "",
        "promoted_count": 0,
    }


def get_workspace_path(conversation_id: str, create: bool = True) -> pathlib.Path:
    """Return the persistent workspace path for a conversation's run."""
    run = get_run_record(conversation_id)
    if not run:
        if not create:
            workspace = (RUNS_ROOT_PATH / sanitize_conversation_id(build_run_id(conversation_id)) / "workspace").resolve()
            if RUNS_ROOT_PATH not in workspace.parents:
                raise ValueError("Workspace path escaped runs root")
            return workspace
        run = ensure_run_for_conversation(conversation_id)
    return get_run_workspace_root(run["id"], create=create)


def get_voice_dir(kind: str, create: bool = True) -> pathlib.Path:
    """Return a server-owned directory for transient voice artifacts."""
    safe_kind = re.sub(r"[^A-Za-z0-9._-]+", "_", (kind or "misc").strip()) or "misc"
    target = (VOICE_ROOT_PATH / safe_kind).resolve()
    if VOICE_ROOT_PATH not in target.parents and target != VOICE_ROOT_PATH:
        raise ValueError("Voice path escaped voice root")
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def build_voice_command(
    template: str,
    replacements: Dict[str, str],
    fallback: Optional[List[str]] = None,
) -> List[str]:
    """Expand a configurable native command template into argv."""
    candidate = (template or "").strip()
    if not candidate:
        return list(fallback or [])
    argv = shlex.split(candidate)
    return [part.format(**replacements) for part in argv]


def default_tts_command(output_path: pathlib.Path, text_path: pathlib.Path) -> List[str]:
    """Return a built-in native TTS command when available."""
    say_path = shutil.which("say")
    if say_path:
        return [say_path, "-v", VOICE_TTS_VOICE, "-o", str(output_path), "-f", str(text_path)]
    piper_path = shutil.which("piper")
    if piper_path and PIPER_DEFAULT_MODEL.exists():
        return [
            "sh", "-c",
            f"{shlex.quote(piper_path)} --model {shlex.quote(str(PIPER_DEFAULT_MODEL))} "
            f"--output_file {shlex.quote(str(output_path))} < {shlex.quote(str(text_path))}",
        ]
    return []


def default_stt_command(input_path: pathlib.Path, output_dir: pathlib.Path) -> List[str]:
    """Return a built-in native STT command when available."""
    whisper_path = shutil.which("whisper")
    if not whisper_path:
        return []
    return [
        whisper_path,
        str(input_path),
        "--model", "turbo",
        "--language", VOICE_STT_LANGUAGE,
        "--output_format", "txt",
        "--output_dir", str(output_dir),
    ]


async def run_native_voice_command(argv: List[str], timeout: float = VOICE_COMMAND_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Run a native voice tool and capture structured output."""
    if not argv:
        raise HTTPException(status_code=503, detail="Voice tool is not configured on the server")
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Voice tool not found: {argv[0]}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise HTTPException(status_code=504, detail="Voice tool timed out") from exc
    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if process.returncode != 0:
        detail = stderr_text or stdout_text or f"Voice tool failed with exit code {process.returncode}"
        raise HTTPException(status_code=500, detail=detail)
    return {
        "command": argv,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "returncode": process.returncode,
    }


def guess_audio_media_type(path: pathlib.Path) -> str:
    """Map a generated audio filename to an HTTP media type."""
    suffix = path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".wave": "audio/wav",
        ".aiff": "audio/aiff",
        ".aif": "audio/aiff",
        ".caf": "audio/x-caf",
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm",
    }.get(suffix, "application/octet-stream")


def infer_tts_output_suffix(template: str) -> str:
    """Guess the output suffix for a configured TTS command."""
    candidate = (template or "").strip()
    if not candidate:
        # No custom command — match whatever the built-in default produces.
        if shutil.which("say"):
            return ".aiff"
        return ".wav"  # piper outputs WAV
    try:
        argv = shlex.split(candidate)
    except ValueError:
        return ".wav"
    for token in argv:
        if "{output}" in token:
            suffix = pathlib.Path(token.replace("{output}", "voice.wav")).suffix
            return suffix or ".wav"
    return ".wav"


def read_transcript_output(output_dir: pathlib.Path, input_path: pathlib.Path, transcript_path: pathlib.Path) -> str:
    """Load transcript text written by the native STT tool."""
    candidates = [transcript_path, output_dir / f"{input_path.stem}.txt"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    raise HTTPException(status_code=500, detail="Transcription completed but no transcript file was produced")


def get_voice_runtime_summary() -> Dict[str, Any]:
    """Describe whether server-side native voice tools are available."""
    say_path = shutil.which("say")
    whisper_path = shutil.which("whisper")
    piper_path = shutil.which("piper")
    piper_ok = bool(piper_path and PIPER_DEFAULT_MODEL.exists())
    tts_available = bool(VOICE_TTS_COMMAND or say_path or piper_ok)
    stt_available = bool(VOICE_STT_COMMAND or whisper_path)
    tts_backend = "custom" if VOICE_TTS_COMMAND else ("say" if say_path else ("piper" if piper_ok else "unavailable"))
    stt_backend = "custom" if VOICE_STT_COMMAND else ("whisper" if whisper_path else "unavailable")
    return {
        "tts_available": tts_available,
        "stt_available": stt_available,
        "tts_backend": tts_backend,
        "stt_backend": stt_backend,
        "tts_voice": VOICE_TTS_VOICE,
        "stt_language": VOICE_STT_LANGUAGE,
        "voice_root": str(VOICE_ROOT_PATH),
    }


def load_pet_profile() -> Optional[Dict[str, Any]]:
    """Return the singleton agent profile."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''SELECT id, name, species, status, theme_primary, theme_secondary, theme_accent,
                  avatar_style, curious, cautious, playful, talkative, independent,
                  custom_backstory, system_prompt_suffix, created_at, updated_at, last_active_at,
                  hunger, energy, last_fed_at,
                  llm_temperature, llm_top_p, llm_frequency_penalty, llm_presence_penalty, llm_max_tokens
           FROM pet_profile
           WHERE id = 1'''
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "species": row[2],
        "status": row[3],
        "theme_primary": row[4],
        "theme_secondary": row[5],
        "theme_accent": row[6],
        "avatar_style": row[7],
        "curious": int(row[8]),
        "cautious": int(row[9]),
        "playful": int(row[10]),
        "talkative": int(row[11]),
        "independent": int(row[12]),
        "custom_backstory": row[13] or "",
        "system_prompt_suffix": row[14] or "",
        "created_at": row[15],
        "updated_at": row[16],
        "last_active_at": row[17],
        "hunger": int(row[18] or 35) if len(row) > 18 else 35,
        "energy": int(row[19] or 65) if len(row) > 19 else 65,
        "last_fed_at": row[20] if len(row) > 20 else None,
        "llm_temperature": float(row[21]) if len(row) > 21 and row[21] is not None else 0.25,
        "llm_top_p": float(row[22]) if len(row) > 22 and row[22] is not None else 0.95,
        "llm_frequency_penalty": float(row[23]) if len(row) > 23 and row[23] is not None else 0.2,
        "llm_presence_penalty": float(row[24]) if len(row) > 24 and row[24] is not None else 0.15,
        "llm_max_tokens": int(row[25]) if len(row) > 25 and row[25] is not None else 4096,
    }


def get_pet_stats() -> Dict[str, Any]:
    """Return aggregate stats for the singleton pet."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total_runs = c.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
    successful_runs = c.execute("SELECT COUNT(*) FROM runs WHERE status IN ('completed', 'success', 'active')").fetchone()[0]
    capabilities_count = c.execute('SELECT COUNT(*) FROM pet_capabilities WHERE status != ?', ('archived',)).fetchone()[0]
    memories_count = c.execute('SELECT COUNT(*) FROM pet_memories').fetchone()[0]
    last_active_row = c.execute('SELECT MAX(updated_at) FROM conversations').fetchone()
    conn.close()
    return {
        "total_runs": int(total_runs or 0),
        "successful_runs": int(successful_runs or 0),
        "capabilities_count": int(capabilities_count or 0),
        "memories_count": int(memories_count or 0),
        "last_active_at": last_active_row[0] if last_active_row else None,
        "hunger": int(profile.get("hunger", 35)) if (profile := load_pet_profile()) else 35,
        "energy": int(profile.get("energy", 65)) if profile else 65,
        "last_fed_at": profile.get("last_fed_at") if profile else None,
    }


def write_pet_profile_snapshot(profile: Dict[str, Any]):
    """Persist a JSON snapshot of the pet profile into the pet home."""
    ensure_pet_dirs()
    snapshot = {
        key: value
        for key, value in profile.items()
        if key not in {"id"}
    }
    with (get_pet_root() / "profile.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def upsert_pet_profile(payload: Dict[str, Any], create: bool = False) -> Dict[str, Any]:
    """Create or update the singleton pet profile."""
    existing = load_pet_profile()
    if create and existing:
        raise HTTPException(status_code=400, detail="Agent profile already exists")
    if not create and not existing:
        raise HTTPException(status_code=404, detail="Agent profile not initialized")

    base = existing or {}

    def int_field(name: str, default: int) -> int:
        raw = payload.get(name, base.get(name, default))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
        return max(0, min(100, value))

    def float_field(name: str, default: float, lo: float, hi: float) -> float:
        raw = payload.get(name, base.get(name, default))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = default
        return max(lo, min(hi, value))

    now = utcnow_iso()
    profile = {
        "id": 1,
        "name": str(payload.get("name", base.get("name", "Compy"))).strip() or "Compy",
        "species": str(payload.get("species", base.get("species", "agent"))).strip() or "agent",
        "status": str(payload.get("status", base.get("status", "active"))).strip() or "active",
        "theme_primary": str(payload.get("theme_primary", base.get("theme_primary", "#2563eb"))).strip() or "#2563eb",
        "theme_secondary": str(payload.get("theme_secondary", base.get("theme_secondary", "#0f172a"))).strip() or "#0f172a",
        "theme_accent": str(payload.get("theme_accent", base.get("theme_accent", "#dbeafe"))).strip() or "#dbeafe",
        "avatar_style": str(payload.get("avatar_style", base.get("avatar_style", "agent"))).strip() or "agent",
        "curious": int_field("curious", 60),
        "cautious": int_field("cautious", 60),
        "playful": int_field("playful", 40),
        "talkative": int_field("talkative", 45),
        "independent": int_field("independent", 55),
        "custom_backstory": str(payload.get("custom_backstory", base.get("custom_backstory", ""))).strip(),
        "system_prompt_suffix": str(payload.get("system_prompt_suffix", base.get("system_prompt_suffix", ""))).strip(),
        "created_at": base.get("created_at", now),
        "updated_at": now,
        "last_active_at": base.get("last_active_at"),
        "hunger": max(0, min(100, int(payload.get("hunger", base.get("hunger", 35))))),
        "energy": max(0, min(100, int(payload.get("energy", base.get("energy", 65))))),
        "last_fed_at": payload.get("last_fed_at", base.get("last_fed_at")),
        "llm_temperature": float_field("llm_temperature", 0.25, 0.0, 2.0),
        "llm_top_p": float_field("llm_top_p", 0.95, 0.0, 1.0),
        "llm_frequency_penalty": float_field("llm_frequency_penalty", 0.2, -2.0, 2.0),
        "llm_presence_penalty": float_field("llm_presence_penalty", 0.15, -2.0, 2.0),
        "llm_max_tokens": max(64, min(32768, int(float_field("llm_max_tokens", 4096, 64, 32768)))),
    }

    ensure_pet_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''INSERT OR REPLACE INTO pet_profile
           (id, name, species, status, theme_primary, theme_secondary, theme_accent,
            avatar_style, curious, cautious, playful, talkative, independent,
            custom_backstory, system_prompt_suffix, created_at, updated_at, last_active_at,
            hunger, energy, last_fed_at,
            llm_temperature, llm_top_p, llm_frequency_penalty, llm_presence_penalty, llm_max_tokens)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            profile["id"], profile["name"], profile["species"], profile["status"],
            profile["theme_primary"], profile["theme_secondary"], profile["theme_accent"],
            profile["avatar_style"], profile["curious"], profile["cautious"], profile["playful"],
            profile["talkative"], profile["independent"], profile["custom_backstory"],
            profile["system_prompt_suffix"], profile["created_at"], profile["updated_at"],
            profile["last_active_at"], profile["hunger"], profile["energy"], profile["last_fed_at"],
            profile["llm_temperature"], profile["llm_top_p"], profile["llm_frequency_penalty"],
            profile["llm_presence_penalty"], profile["llm_max_tokens"],
        ),
    )
    conn.commit()
    conn.close()
    write_pet_profile_snapshot(profile)
    return profile


def ensure_default_agent_profile() -> Dict[str, Any]:
    """Ensure the install has a default productivity-agent profile."""
    existing = load_pet_profile()
    if existing:
        return existing
    return upsert_pet_profile(
        {
            "name": "Compy",
            "species": "agent",
            "status": "active",
            "theme_primary": "#2563eb",
            "theme_secondary": "#0f172a",
            "theme_accent": "#dbeafe",
            "avatar_style": "agent",
            "curious": 62,
            "cautious": 72,
            "playful": 0,
            "talkative": 32,
            "independent": 70,
            "custom_backstory": "A steady workspace-first coding agent focused on additive progress and safe execution.",
            "system_prompt_suffix": "Prefer durable progress in workspace files. Keep changes additive, inspectable, and low-risk.",
            "hunger": 0,
            "energy": 100,
            "llm_temperature": 0.25,
            "llm_top_p": 0.95,
            "llm_frequency_penalty": 0.2,
            "llm_presence_penalty": 0.15,
            "llm_max_tokens": 4096,
        },
        create=True,
    )


def touch_pet_activity():
    """Update the pet's last-active timestamp when it handles a turn."""
    profile = load_pet_profile()
    if not profile:
        return
    now = utcnow_iso()
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE pet_profile SET last_active_at = ?, updated_at = ? WHERE id = 1', (now, now))
    conn.commit()
    conn.close()


def feed_pet(kind: str = "snack", note: str = "", source_run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Refresh the agent's internal operating state."""
    profile = load_pet_profile()
    if not profile:
        return None

    normalized = (kind or "snack").strip().lower()
    delta_map = {
        "snack": {"hunger": -12, "energy": 6},
        "meal": {"hunger": -24, "energy": 10},
        "knowledge": {"hunger": -10, "energy": 8},
        "capability": {"hunger": -14, "energy": 12},
        "task": {"hunger": -8, "energy": 5},
    }
    delta = delta_map.get(normalized, delta_map["snack"])
    now = utcnow_iso()
    hunger = max(0, min(100, int(profile.get("hunger", 35)) + delta["hunger"]))
    energy = max(0, min(100, int(profile.get("energy", 65)) + delta["energy"]))

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''UPDATE pet_profile
           SET hunger = ?, energy = ?, last_fed_at = ?, updated_at = ?
           WHERE id = 1''',
        (hunger, energy, now, now),
    )
    conn.execute(
        'INSERT INTO pet_events (type, payload_json, created_at) VALUES (?, ?, ?)',
        (
            "fed",
            json.dumps({"kind": normalized, "note": note, "source_run_id": source_run_id, "hunger": hunger, "energy": energy}),
            now,
        ),
    )
    conn.commit()
    conn.close()
    return load_pet_profile()


BOND_MAX_PETS_PER_DAY = 12
BOND_AFFECTION_PER_PET = 3
BOND_DAILY_DECAY = 8
BOND_MAX_AFFECTION = 100


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _bond_mood(affection: int, streak: int) -> str:
    if affection >= 70 and streak >= 3:
        return "happy"
    if affection >= 40:
        return "content"
    if affection >= 15:
        return "lonely"
    return "neglected"


def get_pet_bond() -> Dict[str, Any]:
    """Load bond state, apply daily decay, and persist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT affection, pets_today, last_pet_date, streak, last_visit_date FROM pet_profile WHERE id = 1'
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return {"affection": 50, "pets_today": 0, "streak": 0, "mood": "content",
                "max_pets_per_day": BOND_MAX_PETS_PER_DAY, "capped": False}

    affection, pets_today, last_pet_date, streak, last_visit_date = (
        int(row[0] or 50), int(row[1] or 0), row[2], int(row[3] or 0), row[4],
    )
    today = _today_str()
    changed = False

    # Reset daily pet count if new day
    if last_pet_date != today:
        pets_today = 0
        changed = True

    # Apply decay for missed days
    if last_visit_date and last_visit_date != today:
        try:
            last_dt = datetime.strptime(last_visit_date, "%Y-%m-%d")
            days_away = (datetime.strptime(today, "%Y-%m-%d") - last_dt).days
        except ValueError:
            days_away = 1
        if days_away > 0:
            affection = max(0, affection - BOND_DAILY_DECAY * days_away)
            changed = True
        if days_away > 1:
            streak = 0
            changed = True

    if last_visit_date != today:
        last_visit_date = today
        changed = True

    if changed:
        conn.execute(
            '''UPDATE pet_profile
               SET affection = ?, pets_today = ?, streak = ?, last_visit_date = ?, updated_at = ?
               WHERE id = 1''',
            (affection, pets_today, streak, today, utcnow_iso()),
        )
        conn.commit()
    conn.close()

    return {
        "affection": affection,
        "pets_today": pets_today,
        "streak": streak,
        "mood": _bond_mood(affection, streak),
        "max_pets_per_day": BOND_MAX_PETS_PER_DAY,
        "capped": pets_today >= BOND_MAX_PETS_PER_DAY,
    }


def register_pet_action() -> Dict[str, Any]:
    """Record a single petting interaction."""
    bond = get_pet_bond()
    if bond["capped"]:
        return bond

    today = _today_str()
    affection = min(BOND_MAX_AFFECTION, bond["affection"] + BOND_AFFECTION_PER_PET)
    pets_today = bond["pets_today"] + 1
    streak = bond["streak"]

    conn = sqlite3.connect(DB_PATH)
    # Advance streak if first pet of the day
    c = conn.cursor()
    c.execute('SELECT last_pet_date FROM pet_profile WHERE id = 1')
    row = c.fetchone()
    last_pet_date = row[0] if row else None
    if last_pet_date != today:
        streak += 1

    conn.execute(
        '''UPDATE pet_profile
           SET affection = ?, pets_today = ?, last_pet_date = ?, streak = ?,
               last_visit_date = ?, updated_at = ?
           WHERE id = 1''',
        (affection, pets_today, today, streak, today, utcnow_iso()),
    )
    conn.execute(
        'INSERT INTO pet_events (type, payload_json, created_at) VALUES (?, ?, ?)',
        ("petted", json.dumps({"affection": affection, "streak": streak, "pets_today": pets_today}), utcnow_iso()),
    )
    conn.commit()
    conn.close()

    return {
        "affection": affection,
        "pets_today": pets_today,
        "streak": streak,
        "mood": _bond_mood(affection, streak),
        "max_pets_per_day": BOND_MAX_PETS_PER_DAY,
        "capped": pets_today >= BOND_MAX_PETS_PER_DAY,
    }


def list_pet_memories(limit: int = 200) -> List[Dict[str, Any]]:
    """Return stored long-term memories for the pet."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''SELECT id, kind, title, content, importance, source_run_id, created_at, updated_at
           FROM pet_memories
           ORDER BY importance DESC, updated_at DESC
           LIMIT ?''',
        (max(1, min(limit, 500)),),
    )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "kind": row[1],
            "title": row[2],
            "content": row[3],
            "importance": int(row[4] or 0),
            "source_run_id": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }
        for row in rows
    ]


def create_pet_memory(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a durable pet memory."""
    now = utcnow_iso()
    importance = max(0, min(100, int(payload.get("importance", 50))))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''INSERT INTO pet_memories
           (kind, title, content, importance, source_run_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (
            str(payload.get("kind", "note")).strip() or "note",
            str(payload.get("title", "")).strip() or "Untitled memory",
            str(payload.get("content", "")).strip(),
            importance,
            str(payload.get("source_run_id", "")).strip() or None,
            now,
            now,
        ),
    )
    memory_id = c.lastrowid
    conn.commit()
    conn.close()
    return next((item for item in list_pet_memories(limit=500) if item["id"] == memory_id), {})


def update_pet_memory_record(memory_id: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a durable pet memory."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, kind, title, content, importance FROM pet_memories WHERE id = ?', (memory_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    updated = {
        "kind": str(payload.get("kind", row[1])).strip() or row[1],
        "title": str(payload.get("title", row[2])).strip() or row[2],
        "content": str(payload.get("content", row[3])).strip() if payload.get("content") is not None else row[3],
        "importance": max(0, min(100, int(payload.get("importance", row[4])))),
    }
    c.execute(
        '''UPDATE pet_memories
           SET kind = ?, title = ?, content = ?, importance = ?, updated_at = ?
           WHERE id = ?''',
        (updated["kind"], updated["title"], updated["content"], updated["importance"], utcnow_iso(), memory_id),
    )
    conn.commit()
    conn.close()
    return next((item for item in list_pet_memories(limit=500) if item["id"] == memory_id), None)


def delete_pet_memory_record(memory_id: int) -> bool:
    """Delete a durable pet memory."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM pet_memories WHERE id = ?', (memory_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def list_pet_capabilities(limit: int = 200) -> List[Dict[str, Any]]:
    """Return learned capabilities for the pet."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''SELECT id, name, kind, path, description, source_run_id, status, created_at, updated_at
           FROM pet_capabilities
           ORDER BY updated_at DESC
           LIMIT ?''',
        (max(1, min(limit, 500)),),
    )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "name": row[1],
            "kind": row[2],
            "path": row[3],
            "description": row[4] or "",
            "source_run_id": row[5],
            "status": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        }
        for row in rows
    ]


def ensure_unique_pet_capability_path(kind: str, name: str, source_filename: str) -> pathlib.Path:
    """Choose a durable path for a promoted capability."""
    ensure_pet_dirs()
    bucket = {
        "script": "capabilities/scripts",
        "prompt": "capabilities/prompts",
        "template": "capabilities/templates",
        "workflow": "capabilities/workflows",
    }.get(kind, "capabilities")
    target_dir = get_pet_root() / bucket
    target_dir.mkdir(parents=True, exist_ok=True)
    source_name = sanitize_uploaded_filename(source_filename or name or "capability")
    stem = pathlib.Path(source_name).stem or sanitize_uploaded_filename(name or "capability")
    suffix = pathlib.Path(source_name).suffix
    candidate = target_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = target_dir / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=500, detail="Unable to allocate capability path")


def promote_run_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a run artifact into the pet home and register it as a capability."""
    run_id = str(payload.get("run_id", "")).strip()
    source_path = str(payload.get("source_path", "")).strip()
    if not run_id or not source_path:
        raise HTTPException(status_code=400, detail="run_id and source_path are required")

    run_workspace = get_run_workspace_root(run_id, create=False)
    source = (run_workspace / source_path).resolve()
    if run_workspace not in source.parents:
        raise HTTPException(status_code=403, detail="Source path escaped run workspace")
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Source artifact not found")

    target = ensure_unique_pet_capability_path(
        str(payload.get("kind", "artifact")).strip() or "artifact",
        str(payload.get("name", source.stem)).strip() or source.stem,
        source.name,
    )
    shutil.copy2(source, target)
    rel_target = target.relative_to(get_pet_root()).as_posix()
    now = utcnow_iso()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''INSERT INTO pet_capabilities
           (name, kind, path, description, source_run_id, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            str(payload.get("name", source.stem)).strip() or source.stem,
            str(payload.get("kind", "artifact")).strip() or "artifact",
            rel_target,
            str(payload.get("description", "")).strip(),
            run_id,
            "active",
            now,
            now,
        ),
    )
    capability_id = c.lastrowid
    c.execute(
        'UPDATE runs SET promoted_count = COALESCE(promoted_count, 0) + 1 WHERE id = ?',
        (run_id,),
    )
    conn.commit()
    conn.close()
    capability = next((item for item in list_pet_capabilities(limit=500) if item["id"] == capability_id), None)
    if not capability:
        raise HTTPException(status_code=500, detail="Capability promotion failed")
    return capability


def update_pet_capability_record(capability_id: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update capability metadata."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, description, status FROM pet_capabilities WHERE id = ?', (capability_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    name = str(payload.get("name", row[1])).strip() or row[1]
    description = str(payload.get("description", row[2] or "")).strip() if payload.get("description") is not None else (row[2] or "")
    status = str(payload.get("status", row[3])).strip() or row[3]
    c.execute(
        '''UPDATE pet_capabilities
           SET name = ?, description = ?, status = ?, updated_at = ?
           WHERE id = ?''',
        (name, description, status, utcnow_iso(), capability_id),
    )
    conn.commit()
    conn.close()
    return next((item for item in list_pet_capabilities(limit=500) if item["id"] == capability_id), None)


def delete_pet_capability_record(capability_id: int) -> bool:
    """Delete a capability record and best-effort remove the backing file."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT path FROM pet_capabilities WHERE id = ?', (capability_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    rel_path = row[0]
    c.execute('DELETE FROM pet_capabilities WHERE id = ?', (capability_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    target = (get_pet_root() / rel_path).resolve()
    if target.exists() and get_pet_root() in target.parents:
        try:
            target.unlink()
        except OSError:
            pass
    return deleted


def summarize_pet_memories(message: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Select a few relevant memories to inject into the prompt."""
    if limit is None:
        limit = 2 if is_fast_profile_active() else 4
    memories = list_pet_memories(limit=200)
    if not memories:
        return []
    query_words = set(re.findall(r"[a-z0-9_+-]+", (message or "").lower()))
    ranked = []
    for memory in memories:
        haystack = f"{memory['title']} {memory['content']} {memory['kind']}".lower()
        overlap = len(query_words.intersection(set(re.findall(r"[a-z0-9_+-]+", haystack))))
        score = (memory["importance"] / 100.0) + (0.35 if overlap else 0) + min(overlap, 4) * 0.08
        ranked.append((score, memory))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory in ranked[:limit]]


def summarize_pet_capabilities(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return active capabilities for prompt injection."""
    if limit is None:
        limit = 4 if is_fast_profile_active() else 8
    capabilities = [item for item in list_pet_capabilities(limit=200) if item.get("status") != "archived"]
    return capabilities[:limit]


def build_pet_prompt_block(message: str) -> str:
    """Render persistent agent identity, memory, and capability context into the system prompt."""
    profile = ensure_default_agent_profile()
    if not profile:
        return ""
    memory_lines = summarize_pet_memories(message)
    capability_lines = summarize_pet_capabilities()
    fast_profile = is_fast_profile_active()
    memory_limit = 120 if fast_profile else 220
    capability_limit = 100 if fast_profile else 180
    parts = [
        f"You are {profile['name']}, a persistent local {profile['species']} agent on this server.",
        "Profile:",
        f"- Status: {profile['status']}",
        "- Use durable artifacts when they help.",
        "- Avoid destructive actions unless the user explicitly asks for them.",
    ]
    if profile.get("custom_backstory"):
        parts.append(f"- Mission: {profile['custom_backstory']}")
    if profile.get("system_prompt_suffix"):
        parts.append(f"- Extra guidance: {profile['system_prompt_suffix']}")
    if memory_lines:
        parts.append("Relevant memory:")
        parts.extend(f"- {item['title']}: {truncate_output(item['content'], memory_limit)}" for item in memory_lines)
    if capability_lines:
        parts.append("Capabilities:")
        parts.extend(
            f"- {item['name']}: {truncate_output(item['description'] or item['path'], capability_limit)}"
            for item in capability_lines
        )
    return "\n".join(parts)


def resolve_workspace_relative_path(conversation_id: str, rel_path: str = "") -> pathlib.Path:
    """Resolve a user/model path inside the conversation workspace."""
    workspace = get_workspace_path(conversation_id)
    target = (workspace / (rel_path or "")).resolve()
    if target != workspace and workspace not in target.parents:
        raise HTTPException(status_code=403, detail="Access denied")
    return target


def workspace_command_allows_argument(argument: str, workspace: pathlib.Path) -> bool:
    """Reject command arguments that reference paths outside the workspace."""
    token = str(argument or "").strip()
    if not token or token == "-":
        return True
    if "\x00" in token:
        return False
    if "://" in token or token.startswith("-"):
        return True

    normalized = token.replace("\\", "/")
    if normalized.startswith("/"):
        try:
            resolved = pathlib.Path(normalized).resolve(strict=False)
        except Exception:
            return False
        return resolved == workspace or workspace in resolved.parents

    path_like = "/" in normalized or normalized in {".", ".."} or normalized.startswith("./") or normalized.startswith("../")
    if not path_like:
        return True

    try:
        resolved = (workspace / normalized).resolve(strict=False)
    except Exception:
        return False
    return resolved == workspace or workspace in resolved.parents


def normalize_allowed_command_key(value: str) -> str:
    """Normalize a persisted command-approval token."""
    return str(value or "").strip().lower()


def command_permission_key(
    conversation_id: str,
    command: List[str],
    cwd_path: pathlib.Path,
) -> str:
    """Build a stable permission key for a command invocation."""
    if not command:
        return ""

    executable = str(command[0] or "").strip()
    if not executable:
        return ""

    normalized = executable.replace("\\", "/")
    if "/" not in normalized:
        return f"exec:{pathlib.Path(normalized).name.lower()}"

    workspace = get_workspace_path(conversation_id)
    resolved_exec = pathlib.Path(executable).expanduser()
    if not resolved_exec.is_absolute():
        resolved_exec = (cwd_path / resolved_exec).resolve(strict=False)
    else:
        resolved_exec = resolved_exec.resolve(strict=False)

    if resolved_exec != workspace and workspace not in resolved_exec.parents:
        raise ValueError("Executable path must stay inside the workspace")

    return f"workspace:{format_workspace_path(resolved_exec, workspace).lower()}"


def is_command_allowlisted(
    conversation_id: str,
    command: List[str],
    cwd_path: pathlib.Path,
    features: FeatureFlags,
) -> bool:
    """Return whether the command's executable has been approved for this chat."""
    if not command:
        return False
    key = command_permission_key(conversation_id, command, cwd_path)
    if not key:
        return False
    allowed = {
        normalize_allowed_command_key(item)
        for item in (features.allowed_commands or [])
        if isinstance(item, str) and normalize_allowed_command_key(item)
    }
    return key in allowed


def validate_workspace_command(
    conversation_id: str,
    command: List[str],
    cwd_path: pathlib.Path,
    features: Optional[FeatureFlags] = None,
) -> None:
    """Conservatively validate workspace command inputs before execution."""
    if features is not None and not is_command_allowlisted(conversation_id, command, cwd_path, features):
        command_key = command_permission_key(conversation_id, command, cwd_path)
        raise ValueError(
            f"Command '{command_key or command[0]}' is not approved for this chat"
        )

    workspace = get_workspace_path(conversation_id)
    executable = command[0].strip()

    if "/" in executable:
        resolved_exec = pathlib.Path(executable).expanduser()
        if not resolved_exec.is_absolute():
            resolved_exec = (cwd_path / resolved_exec).resolve(strict=False)
        else:
            resolved_exec = resolved_exec.resolve(strict=False)
        if resolved_exec != workspace and workspace not in resolved_exec.parents:
            raise ValueError("Executable path must stay inside the workspace")

    if not STRICT_WORKSPACE_COMMAND_PATHS:
        return

    for index, part in enumerate(command[1:], start=1):
        if not workspace_command_allows_argument(part, workspace):
            raise ValueError(f"command argument {index} references a path outside the workspace")


def ensure_docker_control_enabled() -> None:
    """Fail closed unless the operator explicitly enabled Docker control."""
    if not DOCKER_CONTROL_ENABLED:
        raise HTTPException(status_code=403, detail="Docker control is disabled on this server")
    if not os.path.exists("/var/run/docker.sock"):
        raise HTTPException(status_code=503, detail="Docker socket is not mounted")


def delete_run_workspace(conversation_id: str):
    """Delete the run sandbox for a conversation if it exists."""
    run = get_run_record(conversation_id)
    if not run:
        return
    sandbox_path = pathlib.Path(run["sandbox_path"]).resolve()
    if sandbox_path.exists() and (RUNS_ROOT_PATH == sandbox_path or RUNS_ROOT_PATH in sandbox_path.parents):
        shutil.rmtree(sandbox_path.parent, ignore_errors=True)


def reset_directory_contents(root: pathlib.Path) -> None:
    """Remove everything inside a safe app-owned directory, then recreate it."""
    resolved_root = root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    for child in resolved_root.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except FileNotFoundError:
            continue


async def reset_application_state() -> None:
    """Wipe persisted chat data, workspaces, pet artifacts, and transient runtime state."""
    sessions = list(TERMINAL_SESSIONS.values())
    TERMINAL_SESSIONS.clear()
    TERMINAL_LOCKS.clear()
    for session in sessions:
        await _close_terminal_session(session)

    for waiter in COMMAND_APPROVAL_WAITERS.values():
        future = waiter.get("future")
        if future and not future.done():
            future.cancel()
    COMMAND_APPROVAL_WAITERS.clear()

    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute('DELETE FROM conversation_summaries')
        c.execute('DELETE FROM messages')
        c.execute('DELETE FROM conversations')
        c.execute('DELETE FROM runs')
        c.execute('DELETE FROM pet_memories')
        c.execute('DELETE FROM pet_capabilities')
        c.execute('DELETE FROM pet_events')
        c.execute('DELETE FROM pet_profile')
        try:
            c.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('messages', 'pet_memories', 'pet_capabilities', 'pet_events')"
            )
        except sqlite3.OperationalError:
            pass
        if sqlite_has_fts(conn):
            try:
                c.execute(f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES ('rebuild')")
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()

    reset_directory_contents(RUNS_ROOT_PATH)
    reset_directory_contents(WORKSPACE_ROOT_PATH)
    reset_directory_contents(PET_ROOT_PATH)
    for kind in ("input", "transcripts", "tts-text", "tts-audio"):
        reset_directory_contents(get_voice_dir(kind, create=True))

    ensure_pet_dirs()
    ensure_default_agent_profile()


def format_workspace_path(path: pathlib.Path, workspace: pathlib.Path) -> str:
    """Render a workspace path using workspace-relative POSIX separators."""
    if path == workspace:
        return "."
    return path.relative_to(workspace).as_posix()


def is_spreadsheet_path(path: str) -> bool:
    """Return whether a workspace-relative path looks like a spreadsheet."""
    return pathlib.Path(path or "").suffix.lower() in SPREADSHEET_SUPPORTED_EXTENSIONS


def serialize_spreadsheet_value(value: Any) -> Any:
    """Convert DataFrame values into compact JSON-safe primitives."""
    if value is None:
        return None
    if pd is not None and pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def dataframe_preview_records(df: "pd.DataFrame", max_rows: int = SPREADSHEET_PREVIEW_ROWS) -> List[Dict[str, Any]]:
    """Return a compact record preview for the first few spreadsheet rows."""
    frame = df.head(max_rows).copy()
    frame.columns = [str(col) for col in frame.columns]
    records: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({str(key): serialize_spreadsheet_value(value) for key, value in row.items()})
    return records


def dataframe_column_summaries(df: "pd.DataFrame") -> List[Dict[str, Any]]:
    """Describe spreadsheet columns for model-side analysis."""
    columns: List[Dict[str, Any]] = []
    for col in list(df.columns)[:SPREADSHEET_MAX_COLUMNS]:
        series = df[col]
        non_null_series = series.dropna()
        summary: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "nulls": int(series.isna().sum()),
            "sample_values": [serialize_spreadsheet_value(v) for v in non_null_series.head(3).tolist()],
        }

        if pd is not None and pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                summary["stats"] = {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "sum": float(numeric.sum()),
                }
        columns.append(summary)
    return columns


def load_spreadsheet_summary(target: pathlib.Path, sheet: Optional[str] = None) -> Dict[str, Any]:
    """Load a spreadsheet-like file and return a compact summary for the model."""
    suffix = target.suffix.lower()
    if suffix not in SPREADSHEET_SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported spreadsheet type: {suffix or 'unknown'}")
    if target.stat().st_size > UPLOAD_FILE_SIZE_LIMIT:
        raise ValueError("Spreadsheet too large (max 10MB)")
    if pd is None:
        raise ValueError("Spreadsheet support is unavailable because pandas is not installed")

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(target, sep=sep)
        return {
            "file_type": suffix.lstrip("."),
            "sheet": pathlib.Path(target.name).name,
            "sheet_names": [pathlib.Path(target.name).name],
            "row_count": int(len(df.index)),
            "column_count": int(len(df.columns)),
            "columns": dataframe_column_summaries(df),
            "preview_rows": dataframe_preview_records(df),
        }

    workbook = pd.ExcelFile(target)
    sheet_names = workbook.sheet_names
    chosen_sheet = sheet or (sheet_names[0] if sheet_names else None)
    if not chosen_sheet:
        raise ValueError("Workbook has no readable sheets")
    if chosen_sheet not in sheet_names:
        raise ValueError(f"Sheet not found: {chosen_sheet}")

    df = workbook.parse(chosen_sheet)
    workbook_sheets = []
    for name in sheet_names[:12]:
        sheet_df = workbook.parse(name)
        workbook_sheets.append({
            "name": name,
            "row_count": int(len(sheet_df.index)),
            "column_count": int(len(sheet_df.columns)),
            "columns": [str(col) for col in list(sheet_df.columns)[:SPREADSHEET_MAX_COLUMNS]],
        })

    return {
        "file_type": suffix.lstrip("."),
        "sheet": chosen_sheet,
        "sheet_names": sheet_names,
        "workbook_sheets": workbook_sheets,
        "row_count": int(len(df.index)),
        "column_count": int(len(df.columns)),
        "columns": dataframe_column_summaries(df),
        "preview_rows": dataframe_preview_records(df),
    }


def format_attachment_context(paths: List[str]) -> str:
    """Render uploaded file references into a compact prompt block."""
    cleaned = [str(path).strip() for path in paths if str(path).strip()]
    if not cleaned:
        return ""
    lines = ["Attached files are available in the workspace:"]
    lines.extend(f"- {path}" for path in cleaned[:MAX_ATTACHMENTS_PER_MESSAGE])
    return "\n".join(lines)


def extract_artifact_references(message: str) -> List[str]:
    """Extract explicit artifact references from user-visible chat text."""
    refs: List[str] = []
    for match in re.findall(r"\[\[artifact:([^\]]+)\]\]", message or "", flags=re.IGNORECASE):
        cleaned = str(match).strip()
        if cleaned and cleaned not in refs:
            refs.append(cleaned)
    return refs


def extract_workspace_path_references(message: str) -> List[str]:
    """Extract likely workspace-relative file references from a message."""
    refs: List[str] = []
    for match in re.findall(r"(`([^`]+)`|([\w./-]+\.[A-Za-z0-9]+))", message or ""):
        candidate = str(match[1] or match[2] or "").strip()
        if not candidate:
            continue
        if candidate.startswith("[[artifact:") or "://" in candidate:
            continue
        cleaned = candidate.strip("./")
        if not cleaned or cleaned in refs:
            continue
        refs.append(cleaned)
    return refs


def truncate_output(text: str, limit: int = COMMAND_OUTPUT_LIMIT) -> str:
    """Trim long command output while making truncation explicit."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} chars]"


def build_tool_system_prompt(base_system_prompt: str) -> str:
    """Combine the base assistant prompt with the tool protocol and supported tools."""
    tool_schema = """Available tools:
- workspace.list_files {"path":"."}
- workspace.grep {"query":"FeatureFlags","path":".","glob":"*.py","limit":20}
- workspace.read_file {"path":"src/app.py"}
- workspace.patch_file {"path":"src/app.py","edits":[{"old_text":"before","new_text":"after","expected_count":1}]}
- workspace.patch_file {"path":"notes/todo.txt","create":true,"new_content":"hello"}
- workspace.run_command {"command":["python3","main.py"],"cwd":"."}
- spreadsheet.describe {"path":"model.xlsx","sheet":"Assumptions"}
- conversation.search_history {"query":"retry logic","limit":5}
- web.search {"query":"python async docs","limit":5}
- web.search {"query":"history of C programming language","domains":["wikipedia.org"],"limit":5}
- web.fetch_page {"url":"https://en.wikipedia.org/wiki/Python_(programming_language)"}

Constraints:
- Paths are always relative to the current conversation workspace.
- workspace.grep searches text files in the workspace and returns matching file paths and lines.
- workspace.patch_file uses exact-match replacements; prefer small edits.
- workspace.run_command takes an argv array, never a shell string.
- spreadsheet.describe is for CSV, TSV, and Excel inspection.
- conversation.search_history searches the current conversation only.
- web.search is for fresh or explicitly web-based questions, supports optional domain filters, and returns ordered results with normalized domains.
- web.fetch_page reads a web page after search and returns normalized domain/content metadata for citation-ready summaries.
- Write reusable artifacts to the workspace and mention the path briefly.
- Ask for another tool only when needed; otherwise answer."""
    return f"{base_system_prompt.strip()}\n\n{TOOL_USE_SYSTEM_PROMPT.strip()}\n\n{tool_schema}"


def build_filtered_tool_system_prompt(base_system_prompt: str, allowed_tools: List[str]) -> str:
    """Scope the tool prompt to a subset of tools for a specific phase."""
    filtered = "\n".join(f"- {tool_name}" for tool_name in allowed_tools)
    return (
        f"{build_tool_system_prompt(base_system_prompt)}\n\n"
        f"Allowed tools for this phase:\n{filtered}\n"
        "Do not call any tool outside this allowed list."
    )


def build_effective_system_prompt(base_system_prompt: str, user_message: str) -> str:
    """Attach pet identity and long-term context to the active system prompt."""
    pet_block = build_pet_prompt_block(user_message)
    if not pet_block:
        return base_system_prompt
    return f"{base_system_prompt.strip()}\n\n{pet_block}"


@dataclass
class DeepSession:
    """Shared state for a deep-mode request."""
    websocket: WebSocket
    conversation_id: str
    message: str
    history: List[Dict[str, str]]
    system_prompt: str
    max_tokens: int
    features: "FeatureFlags"
    context: str = ""
    workspace_enabled: bool = False
    workspace_facts: str = ""
    workspace_snapshot: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    task_board_path: str = ".ai/task-board.md"
    task_state_path: str = ".ai/task-state.json"
    build_step_summaries: List[str] = field(default_factory=list)
    step_reports: List[Dict[str, Any]] = field(default_factory=list)
    build_summary: str = ""
    changed_files: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    verification_summary: str = ""
    scope_audit: Dict[str, Any] = field(default_factory=dict)
    scope_audit_summary: str = ""
    draft_response: str = ""
    final_response: str = ""
    resumed: bool = False

    def history_messages(self) -> List[Dict[str, str]]:
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.history]


@dataclass
class ToolLoopOutcome:
    """Result of a server-side tool loop."""
    final_text: str
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


def fetched_web_sources(tool_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Collect unique fetched-page metadata for citation and audit flows."""
    sources: Dict[str, Dict[str, str]] = {}
    for entry in tool_results:
        call = entry.get("call", {})
        result = entry.get("result", {})
        if call.get("name") != "web.fetch_page" or not result.get("ok"):
            continue
        payload = result.get("result", {})
        if not isinstance(payload, dict):
            continue
        url = canonicalize_http_url(str(payload.get("final_url") or payload.get("url") or "").strip())
        if not url:
            continue
        if url not in sources:
            sources[url] = {
                "url": url,
                "title": str(payload.get("title", "")).strip(),
                "domain": str(payload.get("domain", "")).strip(),
            }
    return list(sources.values())


def answer_has_required_source_grounding(answer: str, sources: List[Dict[str, str]]) -> bool:
    """Return whether an answer already cites fetched sources and includes a sources line."""
    text = str(answer or "")
    if not text or not sources:
        return True
    has_sources_line = bool(re.search(r"(?im)^sources:\s*", text))
    body = re.split(r"(?im)^sources:\s*", text, maxsplit=1)[0]
    has_inline_markdown_citation = any(
        source.get("url")
        and re.search(rf"\[[^\]]+\]\({re.escape(source['url'])}\)", body)
        for source in sources
    )
    return has_sources_line and has_inline_markdown_citation


def append_sources_line(answer: str, sources: List[Dict[str, str]]) -> str:
    """Ensure the final answer ends with a deterministic sources line."""
    cleaned = str(answer or "").rstrip()
    if not sources:
        return cleaned
    entries = []
    for source in sources:
        url = source.get("url", "").strip()
        if not url:
            continue
        label = source.get("title", "").strip() or source.get("domain", "").strip() or url
        entries.append(f"[{label}]({url})")
    if not entries:
        return cleaned
    sources_line = "Sources: " + ", ".join(entries)
    if re.search(r"(?im)^sources:\s*", cleaned):
        cleaned = re.sub(r"(?im)^sources:\s*.*$", sources_line, cleaned)
        return cleaned
    separator = "\n\n" if cleaned else ""
    return f"{cleaned}{separator}{sources_line}"


async def repair_answer_with_citations(
    draft_answer: str,
    sources: List[Dict[str, str]],
    max_tokens: int,
) -> str:
    """Rewrite a fetched-page answer so it includes inline citations and a sources line."""
    if not sources:
        return draft_answer

    source_lines = []
    for source in sources:
        url = source.get("url", "").strip()
        title = source.get("title", "").strip() or source.get("domain", "").strip() or url
        domain = source.get("domain", "").strip()
        source_lines.append(f"- {title} | {domain or 'unknown domain'} | {url}")

    repair_messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the draft answer so it includes inline Markdown citations for web-backed claims "
                "and ends with a `Sources:` line listing the provided URLs. "
                "Preserve the substance, keep it concise, and do not add new factual claims."
            ),
        },
        {
            "role": "user",
            "content": (
                "Draft answer:\n"
                f"{draft_answer.strip()}\n\n"
                "Fetched sources you may cite:\n"
                + "\n".join(source_lines)
            ),
        },
    ]
    repaired = strip_stream_special_tokens(
        await vllm_chat_complete(repair_messages, max_tokens=min(max_tokens, 1024), temperature=0.1)
    ).strip()
    if not repaired:
        return append_sources_line(draft_answer, sources)
    return append_sources_line(repaired, sources)


async def finalize_tool_loop_answer(
    final_text: str,
    tool_results: List[Dict[str, Any]],
    max_tokens: int,
) -> ToolLoopOutcome:
    """Apply citation repair when fetched web pages informed the final answer."""
    sources = fetched_web_sources(tool_results)
    cleaned = str(final_text or "").strip()
    if sources and not answer_has_required_source_grounding(cleaned, sources):
        cleaned = await repair_answer_with_citations(cleaned, sources, max_tokens)
    return ToolLoopOutcome(final_text=cleaned, tool_results=tool_results)


@dataclass
class TerminalSession:
    """Persistent PTY-backed shell session for a conversation workspace."""
    conversation_id: str
    workspace_path: pathlib.Path
    process: asyncio.subprocess.Process
    master_fd: int
    reader_task: asyncio.Task


@dataclass
class FeatureFlags:
    """Per-request feature switches from the UI."""
    agent_tools: bool = True
    workspace_write: bool = False
    workspace_run_commands: bool = False
    local_rag: bool = True
    web_search: bool = False
    allowed_commands: List[str] = field(default_factory=list)


async def wait_for_command_approval(
    websocket: WebSocket,
    conversation_id: str,
    command: List[str],
    command_key: str,
    cwd: str = ".",
    step_label: Optional[str] = None,
) -> bool:
    """Pause the active tool loop until the user approves or denies the command."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    COMMAND_APPROVAL_WAITERS[conversation_id] = {
        "future": future,
        "command_key": command_key,
    }
    try:
        await websocket.send_json({
            "type": "command_approval_required",
            "command": command,
            "command_key": command_key,
            "cwd": cwd or ".",
            "content": f"Approve '{command_key or 'command'}' for this chat to continue.",
        })
        await send_activity_event(
            websocket,
            "blocked",
            "Blocked",
            f"Waiting for approval to run '{command_key or 'command'}' in this chat.",
            step_label=step_label,
        )
        return await future
    finally:
        current = COMMAND_APPROVAL_WAITERS.get(conversation_id)
        if current and current.get("future") is future:
            COMMAND_APPROVAL_WAITERS.pop(conversation_id, None)


class PatchApplicationError(ValueError):
    """Structured patch failure that can be surfaced back to the model."""

    def __init__(self, message: str, details: Dict[str, Any]):
        super().__init__(message)
        self.details = details


def normalize_deep_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce planner output into the structure the deep pipeline expects."""
    normalized = dict(plan or {})
    if not isinstance(normalized.get("strategy"), str) or not normalized["strategy"].strip():
        normalized["strategy"] = "Inspect, build, verify, and synthesize."
    if not isinstance(normalized.get("deliverable"), str) or not normalized["deliverable"].strip():
        normalized["deliverable"] = "A strong final response grounded in the workspace."

    builder_steps = normalized.get("builder_steps")
    if not isinstance(builder_steps, list):
        builder_steps = []
    normalized["builder_steps"] = [
        str(step).strip() for step in builder_steps if str(step).strip()
    ][:5]
    if not normalized["builder_steps"]:
        normalized["builder_steps"] = [
            "Inspect the most relevant local files or artifacts.",
            "Build or revise the main deliverable in the workspace.",
            "Run a focused check and tighten the result.",
        ]

    verifier_checks = normalized.get("verifier_checks")
    if not isinstance(verifier_checks, list):
        verifier_checks = []
    normalized["verifier_checks"] = [
        str(step).strip() for step in verifier_checks if str(step).strip()
    ][:4]
    if not normalized["verifier_checks"]:
        normalized["verifier_checks"] = [
            "Verify the changed files or artifact contents directly.",
            "Run the most relevant local command or inspection step.",
        ]

    for agent_key, default_role in (("agent_a", "builder"), ("agent_b", "verifier")):
        agent = normalized.get(agent_key)
        if not isinstance(agent, dict):
            agent = {}
        role = str(agent.get("role", default_role)).strip() or default_role
        prompt = str(agent.get("prompt", "")).strip()
        if not prompt:
            if agent_key == "agent_a":
                prompt = (
                    "Carry out the planned build steps, use the workspace actively, "
                    "and produce the main deliverable."
                )
            else:
                prompt = (
                    "Review the build critically, verify the important assumptions, "
                    "and call out any gaps or regressions."
                )
        normalized[agent_key] = {"role": role, "prompt": prompt}

    return normalized


def format_deep_plan_note(plan: Dict[str, Any]) -> str:
    """Render a short user-visible plan summary for deep mode."""
    builder_steps = plan.get("builder_steps", [])
    verifier_checks = plan.get("verifier_checks", [])
    lines = [
        "Plan for this turn:",
        f"- Strategy: {plan.get('strategy', 'Inspect, build, verify, and synthesize.')}",
        f"- Deliverable: {plan.get('deliverable', 'A strong final response grounded in the workspace.')}",
    ]
    if builder_steps:
        lines.append("- Build steps:")
        lines.extend(f"  {idx}. {step}" for idx, step in enumerate(builder_steps, start=1))
    if verifier_checks:
        lines.append("- Verification:")
        lines.extend(f"  {idx}. {step}" for idx, step in enumerate(verifier_checks, start=1))
    return "\n".join(lines)


def capture_workspace_snapshot(conversation_id: str, max_entries: int = 40, max_depth: int = 4) -> Dict[str, Any]:
    """Capture a deterministic summary of the current workspace contents."""
    workspace = get_workspace_path(conversation_id)
    sample_paths: List[str] = []
    total_files = 0
    total_dirs = 0
    user_file_count = 0

    for root, dirs, files in os.walk(workspace):
        root_path = pathlib.Path(root)
        rel_root = pathlib.Path(".") if root_path == workspace else root_path.relative_to(workspace)
        depth = 0 if rel_root == pathlib.Path(".") else len(rel_root.parts)
        dirs[:] = sorted(d for d in dirs if d != ".git")
        if depth >= max_depth:
            dirs[:] = []
        total_dirs += len(dirs)

        for filename in sorted(files):
            total_files += 1
            rel_path = pathlib.Path(filename) if rel_root == pathlib.Path(".") else rel_root / filename
            rel_text = rel_path.as_posix()
            if not rel_text.startswith(".ai/"):
                user_file_count += 1
            if len(sample_paths) < max_entries:
                sample_paths.append(rel_text)

    top_level: List[str] = []
    if workspace.exists():
        for item in sorted(workspace.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            top_level.append(item.name + ("/" if item.is_dir() else ""))

    return {
        "root": str(workspace),
        "total_files": total_files,
        "total_dirs": total_dirs,
        "user_file_count": user_file_count,
        "sample_paths": sample_paths,
        "top_level": top_level[:20],
    }


def format_workspace_snapshot(snapshot: Dict[str, Any]) -> str:
    """Render a short, concrete workspace summary for planning and auditing."""
    if not snapshot:
        return "(workspace snapshot unavailable)"
    top_level = snapshot.get("top_level", [])
    sample_paths = snapshot.get("sample_paths", [])
    lines = [
        f"- Workspace root: {snapshot.get('root', '(unknown)')}",
        f"- Files: {snapshot.get('total_files', 0)} total ({snapshot.get('user_file_count', 0)} user-visible)",
        f"- Directories: {snapshot.get('total_dirs', 0)}",
    ]
    if top_level:
        lines.append("- Top level: " + ", ".join(top_level[:10]))
    if sample_paths:
        lines.append("- Sample paths: " + ", ".join(sample_paths[:10]))
    return "\n".join(lines)


def workspace_is_effectively_empty(snapshot: Dict[str, Any]) -> bool:
    """Treat workspaces with only task-board state as empty for planning purposes."""
    return int(snapshot.get("user_file_count", 0) or 0) == 0


def apply_deep_plan_guardrails(session: DeepSession, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Ground deep-mode plans in the inspected workspace and required audits."""
    normalized = normalize_deep_plan(plan)
    verifier_checks = list(normalized.get("verifier_checks", []))
    audit_check = "Compare the requested deliverable against the files changed and verification evidence."
    if audit_check not in verifier_checks:
        verifier_checks.append(audit_check)
    normalized["verifier_checks"] = verifier_checks[:5]

    if session.workspace_enabled and workspace_is_effectively_empty(session.workspace_snapshot):
        intent = classify_workspace_intent(session.message)
        if intent in {"focused_write", "broad_write"} or is_explicit_plan_execution_request(session.message):
            normalized["strategy"] = "Inspect the empty workspace, scaffold the smallest useful starting point, then verify before claiming broader scope."
            normalized["builder_steps"] = [
                "Align the requested deliverable with the current empty workspace and update the task board accordingly.",
                "Create the minimum file structure needed for the first working slice.",
                "Implement the first working slice and tighten obvious gaps before verification.",
            ]
            normalized["verifier_checks"] = [
                "Confirm the new files now exist in the workspace and match the revised plan.",
                "Run the narrowest useful startup, syntax, or smoke check for the created slice.",
                audit_check,
            ]
    return normalize_deep_plan(normalized)


def build_heuristic_deep_plan(session: DeepSession) -> Dict[str, Any]:
    """Build a short deterministic plan for fast profiles and fallback cases."""
    if session.workspace_enabled:
        plan = {
            "strategy": "Inspect the relevant files, make the smallest useful change, then verify it.",
            "deliverable": "A concise final answer grounded in the workspace and any verified file changes.",
            "builder_steps": [
                "Inspect the files or artifacts most relevant to the request.",
                "Make the smallest useful workspace change or create the needed artifact.",
                "Re-read the result and tighten obvious gaps before final verification.",
            ],
            "verifier_checks": [
                "Inspect the changed files or artifacts directly.",
                "Run the narrowest useful local verification command if one applies.",
            ],
            "agent_a": {
                "role": "builder",
                "prompt": "Carry out the change carefully, using workspace files as the main working surface.",
            },
            "agent_b": {
                "role": "verifier",
                "prompt": "Pressure-test the result for correctness, regressions, and missing validation.",
            },
        }
    else:
        plan = {
            "strategy": "Answer directly, then check the draft against the request.",
            "deliverable": "A concise, correct final response.",
            "builder_steps": [
                "Extract the exact task and key constraints from the conversation.",
                "Draft the answer or implementation details at the requested level of detail.",
                "Tighten the draft for correctness, completeness, and brevity.",
            ],
            "verifier_checks": [
                "Check that the answer actually addresses the user's request.",
            ],
            "agent_a": {
                "role": "responder",
                "prompt": "Produce the main answer directly and keep it practical.",
            },
            "agent_b": {
                "role": "reviewer",
                "prompt": "Look for missing steps, contradictions, or weak claims.",
            },
        }
    return apply_deep_plan_guardrails(session, plan)


def render_deep_confirmation(message: str, workspace_enabled: bool) -> str:
    """Create a brief local confirmation note without spending a model call."""
    request = " ".join((message or "").strip().split())
    if len(request) > 180:
        request = request[:177].rstrip() + "..."
    next_step = "inspect the workspace first" if workspace_enabled else "work from the conversation context first"
    return f"I’m working on: {request or 'the current request'}. I’ll {next_step} and then respond with the best verified answer I can."


def is_explicit_plan_execution_request(message: str) -> bool:
    """Return whether the user is clearly asking to execute an already proposed plan."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    markers = (
        "execute approved plan",
        "execute this approved plan",
        "execute the approved plan",
        "execute plan now",
        "execute this plan",
        "run this plan",
        "run the plan",
    )
    return any(marker in text for marker in markers)


def should_preview_deep_plan(session: "DeepSession") -> bool:
    """Use a separate plan/approval step only for deep workspace execution requests."""
    if not session.workspace_enabled:
        return False
    if is_explicit_plan_execution_request(session.message):
        return False
    return classify_workspace_intent(session.message) in {"focused_write", "broad_write"}


def format_deep_execution_prompt(plan: Dict[str, Any]) -> str:
    """Create an editable execution draft for the composer."""
    strategy = plan.get("strategy", "Inspect, build, verify, and synthesize.")
    deliverable = plan.get("deliverable", "A strong final response grounded in the workspace.")
    steps = plan.get("builder_steps", [])
    checks = plan.get("verifier_checks", [])

    lines = [
        "Execute approved plan:",
        "",
        f"Strategy: {strategy}",
        f"Deliverable: {deliverable}",
        "",
        "Steps:",
    ]
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(steps, start=1))
    if checks:
        lines.extend(["", "Verification:"])
        lines.extend(f"{idx}. {step}" for idx, step in enumerate(checks, start=1))
    lines.extend([
        "",
        "You can adjust the plan if needed while executing it, but keep the scope aligned with this request.",
    ])
    return "\n".join(lines)


def render_deep_plan_preview(plan: Dict[str, Any]) -> str:
    """Render a short plan-preview answer for approval-first deep mode."""
    lines = [
        "I mapped out an execution plan for this request.",
        "",
        format_deep_plan_note(plan),
        "",
        "Edit the execution draft in the composer if you want to adjust scope or wording, then send it to approve and run.",
    ]
    return "\n".join(lines)


def write_workspace_text(conversation_id: str, rel_path: str, content: str) -> str:
    """Write a UTF-8 text file into the conversation workspace."""
    target = resolve_workspace_relative_path(conversation_id, rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if len(encoded) > WORKSPACE_WRITE_SIZE_LIMIT:
        raise ValueError("Workspace text too large")
    target.write_text(content, encoding="utf-8")
    workspace = get_workspace_path(conversation_id)
    return format_workspace_path(target, workspace)


def read_workspace_text(conversation_id: str, rel_path: str) -> Optional[str]:
    """Read a UTF-8 text file from the conversation workspace if it exists."""
    target = resolve_workspace_relative_path(conversation_id, rel_path)
    if not target.exists() or not target.is_file():
        return None
    if target.stat().st_size > WORKSPACE_FILE_SIZE_LIMIT:
        return None
    return target.read_text(encoding="utf-8")


def format_task_board(session: DeepSession, active_build_step: Optional[int] = None) -> str:
    """Render a durable workspace checklist for the current deep-mode turn."""
    builder_steps = session.plan.get("builder_steps", [])
    verifier_checks = session.plan.get("verifier_checks", [])

    def render_items(items: List[str], completed_count: int = 0, active_index: Optional[int] = None) -> List[str]:
        rendered: List[str] = []
        for idx, item in enumerate(items):
            if idx < completed_count:
                marker = "[x]"
            elif active_index is not None and idx == active_index:
                marker = "[>]"
            else:
                marker = "[ ]"
            rendered.append(f"{marker} {idx + 1}. {item}")
        return rendered or ["(none)"]

    build_lines = render_items(
        builder_steps,
        completed_count=len(session.build_step_summaries),
        active_index=active_build_step,
    )
    verify_lines = render_items(
        verifier_checks,
        completed_count=1 if session.verification_summary else 0,
        active_index=None,
    )

    lines = [
        "# Task Board",
        "",
        "## Request",
        session.message or "(none)",
        "",
        "## Strategy",
        session.plan.get("strategy", "(none)"),
        "",
        "## Deliverable",
        session.plan.get("deliverable", "(none)"),
        "",
        "## Build Checklist",
        *build_lines,
        "",
        "## Verification Checklist",
        *verify_lines,
        "",
        "## Workspace Facts",
        session.workspace_facts or "(pending)",
        "",
        "## Workspace Snapshot",
        format_workspace_snapshot(session.workspace_snapshot),
        "",
        "## Step Notes",
    ]
    if session.build_step_summaries:
        lines.extend(f"- {summary}" for summary in session.build_step_summaries)
    else:
        lines.append("- No build steps completed yet.")

    lines.extend([
        "",
        "## Verification Notes",
        session.verification_summary or "(pending)",
        "",
        "## Scope Audit",
        session.scope_audit_summary or "(pending)",
    ])
    return "\n".join(lines)


async def persist_task_board(
    session: DeepSession,
    active_build_step: Optional[int] = None,
    announce: bool = False,
) -> str:
    """Persist the current task board into the workspace."""
    if not session.workspace_enabled:
        return session.task_board_path
    path = write_workspace_text(
        session.conversation_id,
        session.task_board_path,
        format_task_board(session, active_build_step=active_build_step),
    )
    if announce:
        await send_assistant_note(session.websocket, f"Task board saved to `[[artifact:{path}]]`.")
    return path


def format_task_state_payload(session: DeepSession) -> Dict[str, Any]:
    """Return a machine-readable snapshot of the current deep-mode task state."""
    return {
        "request": session.message,
        "workspace_facts": session.workspace_facts,
        "workspace_snapshot": session.workspace_snapshot,
        "plan": session.plan,
        "task_board_path": session.task_board_path,
        "build_step_summaries": session.build_step_summaries,
        "step_reports": session.step_reports,
        "build_summary": session.build_summary,
        "changed_files": session.changed_files,
        "agent_outputs": session.agent_outputs,
        "verification_summary": session.verification_summary,
        "scope_audit": session.scope_audit,
        "scope_audit_summary": session.scope_audit_summary,
        "draft_response": session.draft_response,
    }


async def persist_task_state(session: DeepSession) -> str:
    """Persist the machine-readable deep-mode state into the workspace."""
    if not session.workspace_enabled:
        return session.task_state_path
    path = write_workspace_text(
        session.conversation_id,
        session.task_state_path,
        json.dumps(format_task_state_payload(session), ensure_ascii=False, indent=2),
    )
    return path


def load_task_state(conversation_id: str, rel_path: str = ".ai/task-state.json") -> Optional[Dict[str, Any]]:
    """Load a previously persisted deep-mode task state if present."""
    raw = read_workspace_text(conversation_id, rel_path)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def build_scope_audit(session: DeepSession) -> Dict[str, Any]:
    """Compare claimed progress against concrete workspace evidence."""
    step_reports = session.step_reports or []
    successful_tool_steps = sum(1 for report in step_reports if int(report.get("successful_tools", 0)) > 0)
    successful_command_checks = sum(1 for report in step_reports if int(report.get("successful_commands", 0)) > 0)
    changed_files = [path for path in session.changed_files if path]
    verification_present = bool((session.verification_summary or "").strip())
    empty_workspace = workspace_is_effectively_empty(session.workspace_snapshot)

    items = [
        {
            "name": "workspace_inspected",
            "complete": bool(session.workspace_snapshot) and bool(session.workspace_facts),
            "notes": "Inspection combines tool findings with a deterministic snapshot.",
        },
        {
            "name": "plan_grounded",
            "complete": bool(session.plan),
            "notes": "Deep plan was normalized against the inspected workspace state.",
        },
        {
            "name": "workspace_actions",
            "complete": bool(changed_files) or successful_tool_steps > 0 or not session.workspace_enabled,
            "notes": (
                f"{len(changed_files)} changed file(s), {successful_tool_steps} build step(s) with successful tool actions."
                if session.workspace_enabled
                else "Workspace mode was not required."
            ),
        },
        {
            "name": "verification",
            "complete": verification_present,
            "notes": (
                f"Verification summary present; {successful_command_checks} build step(s) included successful command checks."
                if verification_present
                else "No verification summary was captured."
            ),
        },
    ]

    gaps: List[str] = []
    if session.workspace_enabled and not changed_files and is_explicit_plan_execution_request(session.message):
        gaps.append("No workspace files were changed during an execution request.")
    if session.workspace_enabled and successful_tool_steps == 0 and is_explicit_plan_execution_request(session.message):
        gaps.append("No successful tool actions were recorded for the build steps.")
    if not verification_present and session.workspace_enabled:
        gaps.append("Verification did not produce a summary grounded in the workspace.")
    if empty_workspace and not changed_files and session.workspace_enabled:
        gaps.append("The workspace still appears effectively empty after the build phase.")

    status = "complete" if not gaps and all(item["complete"] for item in items) else "partial"
    if session.workspace_enabled and not changed_files and not verification_present:
        status = "blocked"

    summary_lines = [
        f"- Status: {status}",
        f"- Changed files: {', '.join(changed_files) if changed_files else '(none)'}",
        f"- Successful build steps with tool evidence: {successful_tool_steps}/{len(step_reports) if step_reports else 0}",
        f"- Verification summary captured: {'yes' if verification_present else 'no'}",
    ]
    if gaps:
        summary_lines.append("- Gaps: " + " | ".join(gaps[:4]))
    else:
        summary_lines.append("- Gaps: none detected from the current audit.")

    return {
        "status": status,
        "items": items,
        "gaps": gaps,
        "summary": "\n".join(summary_lines),
    }


RESUME_HINTS = {
    "continue", "resume", "keep going", "pick up", "carry on", "go on", "finish that",
    "finish it", "proceed", "keep working",
}


def should_resume_task_state(message: str) -> bool:
    """Return whether the user likely wants to continue the existing task board."""
    text = (message or "").strip().lower()
    if not text:
        return False
    return any(hint in text for hint in RESUME_HINTS)


async def maybe_resume_task_state(session: DeepSession) -> bool:
    """Restore prior task state when the user asks to continue the existing work."""
    if not session.workspace_enabled or not should_resume_task_state(session.message):
        return False
    payload = load_task_state(session.conversation_id, session.task_state_path)
    if not payload:
        return False

    plan = payload.get("plan")
    workspace_snapshot = payload.get("workspace_snapshot")
    if isinstance(workspace_snapshot, dict):
        session.workspace_snapshot = workspace_snapshot
    if isinstance(plan, dict):
        session.plan = apply_deep_plan_guardrails(session, plan)
    session.task_board_path = str(payload.get("task_board_path") or session.task_board_path)
    session.build_step_summaries = [
        str(item).strip() for item in payload.get("build_step_summaries", []) if str(item).strip()
    ]
    step_reports = payload.get("step_reports")
    if isinstance(step_reports, list):
        session.step_reports = [item for item in step_reports if isinstance(item, dict)]
    session.build_summary = str(payload.get("build_summary", "")).strip()
    session.changed_files = [
        str(item).strip() for item in payload.get("changed_files", []) if str(item).strip()
    ]
    agent_outputs = payload.get("agent_outputs")
    if isinstance(agent_outputs, dict):
        session.agent_outputs = {str(k): str(v) for k, v in agent_outputs.items()}
    session.verification_summary = str(payload.get("verification_summary", "")).strip()
    scope_audit = payload.get("scope_audit")
    if isinstance(scope_audit, dict):
        session.scope_audit = scope_audit
    session.scope_audit_summary = str(payload.get("scope_audit_summary", "")).strip()
    session.draft_response = str(payload.get("draft_response", "")).strip()
    previous_request = str(payload.get("request", "")).strip()
    session.resumed = True
    await send_assistant_note(
        session.websocket,
        "Resuming from the existing task board and saved workspace state."
        + (f" Original request: {previous_request}" if previous_request else ""),
    )
    return bool(session.plan)

# ==================== vLLM Client (httpx) ====================

def get_agent_llm_params() -> Dict[str, Any]:
    """Return LLM inference parameters from the agent profile."""
    profile = load_pet_profile()
    if not profile:
        return {
            "temperature": 0.25,
            "top_p": 0.95,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.15,
            "max_tokens": 4096,
        }
    return {
        "temperature": profile.get("llm_temperature", 0.25),
        "top_p": profile.get("llm_top_p", 0.95),
        "frequency_penalty": profile.get("llm_frequency_penalty", 0.2),
        "presence_penalty": profile.get("llm_presence_penalty", 0.15),
        "max_tokens": profile.get("llm_max_tokens", 4096),
    }

async def vllm_chat_stream(messages: list, max_tokens: int = None, temperature: float = None):
    """Stream chat completions from vLLM using httpx"""
    agent_params = get_agent_llm_params()
    if max_tokens is None:
        max_tokens = agent_params["max_tokens"]
    if temperature is None:
        temperature = agent_params["temperature"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        async with client.stream(
            "POST",
            f"{VLLM_HOST}/chat/completions",
            json={
                "model": get_active_model_name(),
                "messages": messages,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": agent_params["top_p"],
                "frequency_penalty": agent_params["frequency_penalty"],
                "presence_penalty": agent_params["presence_penalty"],
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

async def vllm_chat_complete(messages: list, max_tokens: int = None, temperature: float = None) -> str:
    """Non-streaming chat completion from vLLM."""
    agent_params = get_agent_llm_params()
    if max_tokens is None:
        max_tokens = agent_params["max_tokens"]
    if temperature is None:
        temperature = agent_params["temperature"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        resp = await client.post(
            f"{VLLM_HOST}/chat/completions",
            json={
                "model": get_active_model_name(),
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": agent_params["top_p"],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def parse_json_object(raw: str) -> Dict:
    """Parse a JSON object, allowing surrounding non-JSON text."""
    cleaned = strip_stream_special_tokens(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("Expected a JSON object")
        return json.loads(match.group())


async def critique_response(message: str, draft: str) -> Dict:
    """Run a lightweight quality check on a draft answer."""
    critique_messages = [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": f"User's question:\n{message}\n\nDraft response:\n{draft}"},
    ]
    raw = await vllm_chat_complete(critique_messages, max_tokens=256, temperature=0.05)
    parsed = parse_json_object(raw)
    return {
        "pass": bool(parsed.get("pass")),
        "issues": str(parsed.get("issues", "")).strip(),
    }


async def send_final_replacement(websocket: WebSocket, content: str):
    """Replace the currently displayed assistant answer with a refined version."""
    await websocket.send_json({"type": "final_replace", "content": content})


async def send_assistant_note(websocket: WebSocket, content: str):
    """Show a transient assistant note while work continues."""
    await websocket.send_json({"type": "assistant_note", "content": content})


async def send_reasoning_note(websocket: WebSocket, content: str):
    """Emit a short reasoning-only note that should not appear as final output."""
    await websocket.send_json({"type": "reasoning_note", "content": content})


async def send_activity_event(
    websocket: WebSocket,
    phase: str,
    label: str,
    content: str,
    *,
    step_label: Optional[str] = None,
):
    """Emit a structured harness activity event for the UI."""
    await websocket.send_json({
        "type": "activity",
        "phase": (phase or "status").strip() or "status",
        "label": (label or "Activity").strip() or "Activity",
        "content": content,
        "step_label": step_label or "",
    })


async def send_scope_audit_event(websocket: WebSocket, audit: Dict[str, Any]):
    """Emit the scope audit through the shared harness activity channel."""
    summary = str(audit.get("summary", "")).strip()
    gaps = audit.get("gaps", [])
    if not summary and isinstance(gaps, list) and gaps:
        summary = "Gaps: " + " | ".join(str(item) for item in gaps[:3])
    await send_activity_event(
        websocket,
        "audit",
        "Audit",
        summary or "Recorded a scope audit for the current run.",
    )


async def send_plan_ready(websocket: WebSocket, plan: Dict[str, Any], execute_prompt: str):
    """Push an editable execution draft to the web composer."""
    await websocket.send_json({
        "type": "plan_ready",
        "plan": format_deep_plan_note(plan),
        "execute_prompt": execute_prompt,
    })


async def send_build_steps(
    websocket: WebSocket,
    steps: List[str],
    completed_count: int = 0,
    active_index: Optional[int] = None,
):
    """Push the current build checklist state to the UI."""
    await websocket.send_json({
        "type": "build_steps",
        "steps": steps,
        "completed_count": completed_count,
        "active_index": active_index,
    })


async def stream_chat_response(
    websocket: WebSocket,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = 0.25,
) -> str:
    """Stream a plain assistant response to the UI and return the final saved text."""
    splitter = ThinkingStreamSplitter()
    await websocket.send_json({"type": "start"})
    async for raw_token in vllm_chat_stream(messages, max_tokens=max_tokens, temperature=temperature):
        for event in splitter.feed(raw_token):
            await websocket.send_json(event)

    for event in splitter.finalize():
        await websocket.send_json(event)

    return strip_stream_special_tokens(splitter.full_response)

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


async def fetch_loaded_model_name() -> Optional[str]:
    """Return the actual model name served by the current vLLM container."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{VLLM_HOST}/models")
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or []
            if not data:
                return None
            first = data[0]
            model_id = first.get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
    except Exception:
        return None
    return None


def sync_active_profile_from_model_name(model_name: Optional[str]) -> str:
    """Update the selected profile when the served model reveals itself."""
    if not model_name:
        return ACTIVE_MODEL_PROFILE
    for key, profile in MODEL_PROFILES.items():
        if profile["name"] == model_name:
            if ACTIVE_MODEL_PROFILE != key or ACTIVE_CUSTOM_MODEL_NAME:
                try:
                    persist_active_model_selection(key, None)
                except Exception:
                    logger.warning("Failed to persist model profile %s", key, exc_info=True)
            return key
    if ACTIVE_MODEL_PROFILE != "custom" or ACTIVE_CUSTOM_MODEL_NAME != model_name:
        try:
            persist_active_model_selection("custom", model_name)
        except Exception:
            logger.warning("Failed to persist custom model profile %s", model_name, exc_info=True)
    return "custom"


async def docker_api_request(method: str, path: str, **kwargs) -> httpx.Response:
    """Send a request to the local Docker API over the mounted socket."""
    ensure_docker_control_enabled()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
        return await client.request(method, f"http://localhost{path}", **kwargs)


async def inspect_vllm_container() -> Dict[str, Any]:
    """Inspect the current vLLM container definition."""
    resp = await docker_api_request("GET", "/containers/vllm/json")
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Unable to inspect vLLM container (HTTP {resp.status_code})")
    return resp.json()


def build_vllm_create_payload(
    container_info: Dict[str, Any],
    profile_key: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Clone the current vLLM container config and swap in the selected model command."""
    config = container_info.get("Config", {})
    host_config = container_info.get("HostConfig", {})
    endpoint_config: Dict[str, Any] = {}
    for network_name, network_data in (container_info.get("NetworkSettings", {}).get("Networks") or {}).items():
        endpoint_entry: Dict[str, Any] = {}
        aliases = network_data.get("Aliases")
        if aliases:
            endpoint_entry["Aliases"] = aliases
        endpoint_config[network_name] = endpoint_entry

    return {
        "Image": container_info.get("Image") or config.get("Image"),
        "Cmd": build_vllm_command(profile_key, model_name=model_name),
        "Env": config.get("Env") or [],
        "ExposedPorts": config.get("ExposedPorts") or {},
        "Labels": config.get("Labels") or {},
        "HostConfig": {
            "Binds": host_config.get("Binds") or [],
            "PortBindings": host_config.get("PortBindings") or {},
            "RestartPolicy": host_config.get("RestartPolicy") or {},
            "DeviceRequests": host_config.get("DeviceRequests") or [],
            "AutoRemove": bool(host_config.get("AutoRemove")),
            "ShmSize": host_config.get("ShmSize") or 0,
        },
        "NetworkingConfig": {
            "EndpointsConfig": endpoint_config,
        },
    }


async def recreate_vllm_container(profile_key: Optional[str] = None, *, model_name: Optional[str] = None):
    """Hard-switch the vLLM container to the selected model profile or custom model."""
    container_info = await inspect_vllm_container()

    stop_resp = await docker_api_request("POST", "/containers/vllm/stop", params={"t": 10})
    if stop_resp.status_code not in {204, 304}:
        raise HTTPException(status_code=500, detail=f"Failed to stop vLLM (HTTP {stop_resp.status_code})")

    delete_resp = await docker_api_request("DELETE", "/containers/vllm", params={"force": "true"})
    if delete_resp.status_code not in {204, 404}:
        raise HTTPException(status_code=500, detail=f"Failed to remove vLLM (HTTP {delete_resp.status_code})")

    create_payload = build_vllm_create_payload(container_info, profile_key, model_name=model_name)
    create_resp = await docker_api_request("POST", "/containers/create", params={"name": "vllm"}, json=create_payload)
    if create_resp.status_code != 201:
        detail = create_resp.text.strip() or f"HTTP {create_resp.status_code}"
        raise HTTPException(status_code=500, detail=f"Failed to create switched vLLM container: {detail}")

    container_id = create_resp.json().get("Id")
    if not container_id:
        raise HTTPException(status_code=500, detail="Docker did not return a container id for the switched vLLM")

    start_resp = await docker_api_request("POST", f"/containers/{container_id}/start")
    if start_resp.status_code != 204:
        detail = start_resp.text.strip() or f"HTTP {start_resp.status_code}"
        raise HTTPException(status_code=500, detail=f"Failed to start switched vLLM container: {detail}")

# ==================== Database (SQLite file at DB_PATH) ====================

FTS_TABLE = "messages_fts"
SUMMARY_TRIGGER_MESSAGE_COUNT = 16
SUMMARY_KEEP_RECENT_MESSAGES = 8
SUMMARY_MAX_SOURCE_MESSAGES = 40
SUMMARY_MAX_CHARS = 12000
SUMMARY_RELATED_HISTORY_LIMIT = 2
SUMMARY_RELATED_HISTORY_MIN_SCORE = 0.55


def normalize_fts_query(query: str) -> str:
    """Convert free text into a conservative FTS query."""
    tokens = re.findall(r"[A-Za-z0-9_]+", (query or "").lower())
    if not tokens:
        return ""
    return " AND ".join(f'"{token}"' for token in tokens[:8])


def sqlite_has_fts(conn: sqlite3.Connection) -> bool:
    """Return whether this SQLite build supports FTS virtual tables."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.fts_probe USING fts5(content)")
        conn.execute("DROP TABLE IF EXISTS temp.fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


def setup_message_fts(conn: sqlite3.Connection):
    """Create and backfill the message FTS index when supported."""
    c = conn.cursor()
    c.execute(f'''CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
                 content,
                 conversation_id UNINDEXED,
                 role UNINDEXED,
                 timestamp UNINDEXED,
                 content='messages',
                 content_rowid='id'
                 )''')
    c.executescript(f'''
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO {FTS_TABLE}(rowid, content, conversation_id, role, timestamp)
            VALUES (new.id, new.content, new.conversation_id, new.role, new.timestamp);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO {FTS_TABLE}({FTS_TABLE}, rowid, content, conversation_id, role, timestamp)
            VALUES ('delete', old.id, old.content, old.conversation_id, old.role, old.timestamp);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO {FTS_TABLE}({FTS_TABLE}, rowid, content, conversation_id, role, timestamp)
            VALUES ('delete', old.id, old.content, old.conversation_id, old.role, old.timestamp);
            INSERT INTO {FTS_TABLE}(rowid, content, conversation_id, role, timestamp)
            VALUES (new.id, new.content, new.conversation_id, new.role, new.timestamp);
        END;
    ''')

    count = c.execute(f"SELECT COUNT(*) FROM {FTS_TABLE}").fetchone()[0]
    if count == 0:
        c.execute(f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES ('rebuild')")


def init_db():
    """Create tables and indexes if missing; safe to call on every startup."""
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

        c.execute('''CREATE TABLE IF NOT EXISTS conversation_summaries
                     (conversation_id TEXT PRIMARY KEY,
                      summary TEXT,
                      source_message_count INTEGER,
                      updated_at TEXT,
                      FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')

        c.execute('''CREATE TABLE IF NOT EXISTS pet_profile
                     (id INTEGER PRIMARY KEY CHECK (id = 1),
                      name TEXT NOT NULL,
                      species TEXT NOT NULL DEFAULT 'agent',
                      status TEXT NOT NULL DEFAULT 'active',
                      theme_primary TEXT NOT NULL,
                      theme_secondary TEXT NOT NULL,
                      theme_accent TEXT NOT NULL,
                      avatar_style TEXT NOT NULL DEFAULT 'agent',
                      curious INTEGER NOT NULL DEFAULT 60,
                      cautious INTEGER NOT NULL DEFAULT 60,
                      playful INTEGER NOT NULL DEFAULT 40,
                      talkative INTEGER NOT NULL DEFAULT 45,
                      independent INTEGER NOT NULL DEFAULT 55,
                      custom_backstory TEXT NOT NULL DEFAULT '',
                      system_prompt_suffix TEXT NOT NULL DEFAULT '',
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL,
                      last_active_at TEXT,
                      hunger INTEGER NOT NULL DEFAULT 35,
                      energy INTEGER NOT NULL DEFAULT 65,
                      last_fed_at TEXT,
                      llm_temperature REAL NOT NULL DEFAULT 0.25,
                      llm_top_p REAL NOT NULL DEFAULT 0.95,
                      llm_frequency_penalty REAL NOT NULL DEFAULT 0.2,
                      llm_presence_penalty REAL NOT NULL DEFAULT 0.15,
                      llm_max_tokens INTEGER NOT NULL DEFAULT 4096)''')

        c.execute('''CREATE TABLE IF NOT EXISTS runs
                     (id TEXT PRIMARY KEY,
                      conversation_id TEXT UNIQUE NOT NULL,
                      title TEXT,
                      status TEXT NOT NULL DEFAULT 'active',
                      sandbox_path TEXT NOT NULL,
                      started_at TEXT NOT NULL,
                      ended_at TEXT,
                      summary TEXT NOT NULL DEFAULT '',
                      promoted_count INTEGER NOT NULL DEFAULT 0)''')

        c.execute('''CREATE TABLE IF NOT EXISTS pet_memories
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      kind TEXT NOT NULL,
                      title TEXT NOT NULL,
                      content TEXT NOT NULL,
                      importance INTEGER NOT NULL DEFAULT 50,
                      source_run_id TEXT,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL)''')

        c.execute('''CREATE TABLE IF NOT EXISTS pet_capabilities
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      kind TEXT NOT NULL,
                      path TEXT NOT NULL,
                      description TEXT NOT NULL DEFAULT '',
                      source_run_id TEXT,
                      status TEXT NOT NULL DEFAULT 'active',
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL)''')

        c.execute('''CREATE TABLE IF NOT EXISTS pet_events
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      type TEXT NOT NULL,
                      payload_json TEXT NOT NULL DEFAULT '{}',
                      created_at TEXT NOT NULL)''')

        try:
            c.execute('ALTER TABLE messages ADD COLUMN feedback TEXT')
        except sqlite3.OperationalError:
            pass

        try:
            c.execute('ALTER TABLE conversations ADD COLUMN run_id TEXT')
        except sqlite3.OperationalError:
            pass

        for statement in (
            'ALTER TABLE pet_profile ADD COLUMN hunger INTEGER NOT NULL DEFAULT 35',
            'ALTER TABLE pet_profile ADD COLUMN energy INTEGER NOT NULL DEFAULT 65',
            'ALTER TABLE pet_profile ADD COLUMN last_fed_at TEXT',
            'ALTER TABLE pet_profile ADD COLUMN llm_temperature REAL NOT NULL DEFAULT 0.25',
            'ALTER TABLE pet_profile ADD COLUMN llm_top_p REAL NOT NULL DEFAULT 0.95',
            'ALTER TABLE pet_profile ADD COLUMN llm_frequency_penalty REAL NOT NULL DEFAULT 0.2',
            'ALTER TABLE pet_profile ADD COLUMN llm_presence_penalty REAL NOT NULL DEFAULT 0.15',
            'ALTER TABLE pet_profile ADD COLUMN llm_max_tokens INTEGER NOT NULL DEFAULT 4096',
            'ALTER TABLE pet_profile ADD COLUMN affection INTEGER NOT NULL DEFAULT 50',
            'ALTER TABLE pet_profile ADD COLUMN pets_today INTEGER NOT NULL DEFAULT 0',
            'ALTER TABLE pet_profile ADD COLUMN last_pet_date TEXT',
            'ALTER TABLE pet_profile ADD COLUMN streak INTEGER NOT NULL DEFAULT 0',
            'ALTER TABLE pet_profile ADD COLUMN last_visit_date TEXT',
        ):
            try:
                c.execute(statement)
            except sqlite3.OperationalError:
                pass

        try:
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conv_timestamp ON messages(conversation_id, timestamp)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_runs_conversation_id ON runs(conversation_id)')
        except sqlite3.OperationalError:
            pass

        c.execute('SELECT id, title, created_at FROM conversations')
        for conv_id, title, created_at in c.fetchall():
            run_id = build_run_id(conv_id)
            sandbox_path = str(get_run_workspace_root(run_id, create=True))
            started_at = created_at or utcnow_iso()
            c.execute(
                '''INSERT OR IGNORE INTO runs
                   (id, conversation_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (run_id, conv_id, title or "", "active", sandbox_path, started_at, None, "", 0),
            )
            try:
                c.execute('UPDATE conversations SET run_id = ? WHERE id = ?', (run_id, conv_id))
            except sqlite3.OperationalError:
                pass

        try:
            if sqlite_has_fts(conn):
                setup_message_fts(conn)
                logger.info("SQLite FTS enabled for messages")
            else:
                logger.warning("SQLite FTS unavailable; falling back to LIKE search")
        except sqlite3.OperationalError as exc:
            logger.warning("SQLite FTS setup failed, falling back to LIKE search: %s", exc)

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


class PetAdoptRequest(BaseModel):
    name: str
    theme_primary: str
    theme_secondary: str
    theme_accent: str
    avatar_style: str = "agent"
    curious: int = 60
    cautious: int = 60
    playful: int = 40
    talkative: int = 45
    independent: int = 55
    custom_backstory: str = ""
    system_prompt_suffix: str = ""
    llm_temperature: float = 0.25
    llm_top_p: float = 0.95
    llm_frequency_penalty: float = 0.2
    llm_presence_penalty: float = 0.15
    llm_max_tokens: int = 4096


class PetUpdateRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    theme_primary: Optional[str] = None
    theme_secondary: Optional[str] = None
    theme_accent: Optional[str] = None
    avatar_style: Optional[str] = None
    curious: Optional[int] = None
    cautious: Optional[int] = None
    playful: Optional[int] = None
    talkative: Optional[int] = None
    independent: Optional[int] = None
    custom_backstory: Optional[str] = None
    system_prompt_suffix: Optional[str] = None
    llm_temperature: Optional[float] = None
    llm_top_p: Optional[float] = None
    llm_frequency_penalty: Optional[float] = None
    llm_presence_penalty: Optional[float] = None
    llm_max_tokens: Optional[int] = None


class PetFeedRequest(BaseModel):
    kind: str = "snack"
    note: str = ""
    source_run_id: Optional[str] = None


class PetMemoryCreateRequest(BaseModel):
    kind: str
    title: str
    content: str
    importance: int = 50
    source_run_id: Optional[str] = None


class PetMemoryUpdateRequest(BaseModel):
    kind: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    importance: Optional[int] = None


class CapabilityPromoteRequest(BaseModel):
    run_id: str
    source_path: str
    name: str
    kind: str
    description: str = ""


class CapabilityUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: str


class WorkspaceFileUpdateRequest(BaseModel):
    path: str
    content: str


class RecoverAssistantMessageRequest(BaseModel):
    message_id: Optional[int] = None
    content: str = ""
    target_dir: str = ""


class SwitchModelRequest(BaseModel):
    profile: str


class ModelDownloadRequest(BaseModel):
    source: str


class ModelDeleteRequest(BaseModel):
    model_id: str


class ModelActivateRequest(BaseModel):
    model_id: str


class UiModelReadyRequest(BaseModel):
    model_name: str
    profile: str
    composer_available: bool = False
    websocket_connected: bool = False


class VoiceSynthesisRequest(BaseModel):
    text: str

# ==================== History selection (token budget for vLLM context) ====================

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


def rerank_search_rows(rows: List[Dict[str, Any]], current_query: str, limit: int) -> List[Dict[str, Any]]:
    """Blend lexical match strength with recency and message quality."""
    if not rows:
        return []

    recent_order = {
        row["timestamp"]: index
        for index, row in enumerate(sorted(rows, key=lambda item: item.get("timestamp", "")))
    }
    lexical_rows = [row for row in rows if row.get("fts_rank") is not None]
    lexical_positions = {
        id(row): index
        for index, row in enumerate(sorted(lexical_rows, key=lambda item: item.get("fts_rank", 0.0)))
    }

    scored_rows = []
    total_rows = len(rows)
    total_lexical = max(len(lexical_rows), 1)

    for row in rows:
        recency_score = calculate_message_relevance_score(
            row,
            current_query,
            recent_order.get(row.get("timestamp", ""), 0),
            total_rows,
        )
        lexical_score = 0.0
        if row.get("fts_rank") is not None:
            lexical_pos = lexical_positions.get(id(row), total_lexical - 1)
            lexical_score = 0.7 * (1.0 - (lexical_pos / total_lexical))
        scored_rows.append((lexical_score + recency_score, row))

    scored_rows.sort(key=lambda item: (item[0], item[1].get("timestamp", "")), reverse=True)
    return [row for _, row in scored_rows[:limit]]


def get_conversation_summary_entry(conv_id: str) -> Optional[Dict[str, Any]]:
    """Return the cached summary for a conversation, if present."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''SELECT summary, source_message_count, updated_at
           FROM conversation_summaries
           WHERE conversation_id = ?''',
        (conv_id,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "summary": row[0],
        "source_message_count": int(row[1] or 0),
        "updated_at": row[2],
    }


def format_messages_for_summary(messages: List[Dict[str, str]], char_limit: int = SUMMARY_MAX_CHARS) -> str:
    """Flatten messages into a bounded transcript for summary generation."""
    parts: List[str] = []
    used = 0
    for msg in messages[-SUMMARY_MAX_SOURCE_MESSAGES:]:
        line = f"{msg['role']}: {msg['content'].strip()}\n"
        if used + len(line) > char_limit:
            remaining = max(char_limit - used, 0)
            if remaining > 0:
                parts.append(line[:remaining])
            break
        parts.append(line)
        used += len(line)
    return "".join(parts).strip()

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
    recent_window = min(SUMMARY_KEEP_RECENT_MESSAGES, total_messages)
    summary_entry = get_conversation_summary_entry(conv_id)
    summary_is_fresh = bool(
        summary_entry
        and summary_entry.get("summary")
        and int(summary_entry.get("source_message_count", 0)) >= max(total_messages - recent_window, 0)
    )

    recent_messages = all_messages[-recent_window:]
    recent_tokens = sum((len(msg['content']) // 4) + 10 for msg in recent_messages)

    older_messages = all_messages[:-recent_window] if recent_window < total_messages else []

    scored_messages = []
    for idx, msg in enumerate(older_messages):
        score = calculate_message_relevance_score(msg, current_query or '', idx, len(older_messages))
        scored_messages.append((score, msg))

    scored_messages.sort(key=lambda x: x[0], reverse=True)

    selected_messages = list(recent_messages)
    total_tokens = recent_tokens
    remaining_tokens = max_tokens - total_tokens
    related_history_count = 0

    for score, msg in scored_messages:
        if summary_is_fresh:
            if related_history_count >= SUMMARY_RELATED_HISTORY_LIMIT:
                break
            if score < SUMMARY_RELATED_HISTORY_MIN_SCORE:
                continue
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
            related_history_count += 1

    for msg in selected_messages:
        msg.pop('id', None)
        msg.pop('feedback', None)

    if summary_is_fresh:
        selected_messages.insert(0, {
            'role': 'system',
            'content': f"Conversation summary:\n{summary_entry['summary']}",
            'timestamp': summary_entry.get('updated_at', ''),
        })

    if limit and len(selected_messages) > limit:
        selected_messages = selected_messages[-limit:]

    return selected_messages

def search_messages(query: str, limit: int = 20) -> List[Dict]:
    """Search through all messages"""
    if not query or len(query.strip()) < 2:
        return []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    safe_limit = max(1, min(int(limit), 100))
    cleaned_query = query.strip()
    fts_query = normalize_fts_query(cleaned_query)

    try:
        if not fts_query:
            raise sqlite3.OperationalError("No valid FTS tokens")

        c.execute(f'''SELECT m.conversation_id, m.role, m.content, m.timestamp, c.title, m.feedback,
                             bm25({FTS_TABLE}) as fts_rank
                      FROM {FTS_TABLE} f
                      JOIN messages m ON m.id = f.rowid
                      LEFT JOIN conversations c ON m.conversation_id = c.id
                      WHERE {FTS_TABLE} MATCH ?
                      ORDER BY bm25({FTS_TABLE}), m.timestamp DESC
                      LIMIT ?''', (fts_query, max(safe_limit * 4, 20)))
        fetched_rows = [
            {
                'conversation_id': row[0], 'role': row[1], 'content': row[2],
                'timestamp': row[3], 'conversation_title': row[4] or 'Untitled',
                'feedback': row[5] or '', 'fts_rank': row[6],
            }
            for row in c.fetchall()
        ]
        results = rerank_search_rows(fetched_rows, cleaned_query, safe_limit)
    except sqlite3.OperationalError:
        search_term = f"%{cleaned_query}%"
        c.execute('''SELECT m.conversation_id, m.role, m.content, m.timestamp, c.title
                     FROM messages m
                     LEFT JOIN conversations c ON m.conversation_id = c.id
                     WHERE m.content LIKE ?
                     ORDER BY m.timestamp DESC
                     LIMIT ?''', (search_term, safe_limit))
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
    ensure_run_for_conversation(conv_id, title=content[:50] + "..." if len(content) > 50 else content)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('SELECT id FROM conversations WHERE id = ?', (conv_id,))
    if not c.fetchone():
        title = content[:50] + "..." if len(content) > 50 else content
        c.execute('''INSERT INTO conversations (id, title, created_at, updated_at)
                     VALUES (?, ?, ?, ?)''',
                  (conv_id, title, datetime.now().isoformat(), datetime.now().isoformat()))
        try:
            c.execute('UPDATE conversations SET run_id = ? WHERE id = ?', (build_run_id(conv_id), conv_id))
        except sqlite3.OperationalError:
            pass

    c.execute('''INSERT INTO messages (conversation_id, role, content, timestamp)
                 VALUES (?, ?, ?, ?)''',
              (conv_id, role, content, datetime.now().isoformat()))

    message_id = c.lastrowid

    c.execute('UPDATE conversations SET updated_at = ? WHERE id = ?',
              (datetime.now().isoformat(), conv_id))

    conn.commit()
    conn.close()
    touch_pet_activity()
    if role == 'assistant':
        feed_pet("task", note="completed_turn", source_run_id=build_run_id(conv_id))
    return message_id


async def refresh_conversation_summary(conv_id: str):
    """Refresh the cached summary for older conversation context."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''SELECT role, content, timestamp
           FROM messages
           WHERE conversation_id = ?
           ORDER BY timestamp ASC''',
        (conv_id,),
    )
    rows = c.fetchall()
    conn.close()

    if len(rows) < SUMMARY_TRIGGER_MESSAGE_COUNT:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('DELETE FROM conversation_summaries WHERE conversation_id = ?', (conv_id,))
        conn.commit()
        conn.close()
        return

    source_rows = rows[:-SUMMARY_KEEP_RECENT_MESSAGES] if len(rows) > SUMMARY_KEEP_RECENT_MESSAGES else []
    if not source_rows:
        return

    transcript = format_messages_for_summary([
        {"role": row[0], "content": row[1], "timestamp": row[2]}
        for row in source_rows
    ])
    if not transcript:
        return

    summary_messages = [
        {"role": "system", "content": CONVERSATION_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    raw_summary = await vllm_chat_complete(summary_messages, max_tokens=256, temperature=0.1)
    summary = strip_stream_special_tokens(raw_summary).strip()
    if not summary:
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''INSERT INTO conversation_summaries (conversation_id, summary, source_message_count, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(conversation_id) DO UPDATE SET
             summary = excluded.summary,
             source_message_count = excluded.source_message_count,
             updated_at = excluded.updated_at''',
        (conv_id, summary, len(source_rows), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def schedule_conversation_summary_refresh(conv_id: str):
    """Kick off a background summary refresh without blocking the current turn."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def runner():
        try:
            await refresh_conversation_summary(conv_id)
        except Exception as exc:
            logger.warning("Conversation summary refresh failed for %s: %s", conv_id, exc)

    loop.create_task(runner())


def parse_feature_flags(raw: Any) -> FeatureFlags:
    """Decode feature flags from WebSocket payloads with sensible defaults."""
    payload = raw if isinstance(raw, dict) else {}

    def flag(name: str, default: bool) -> bool:
        value = payload.get(name, default)
        return value if isinstance(value, bool) else default

    raw_allowed = payload.get("allowed_commands", [])
    allowed_commands = [
        normalize_allowed_command_key(item)
        for item in (raw_allowed if isinstance(raw_allowed, list) else [])
        if isinstance(item, str) and normalize_allowed_command_key(item)
    ]

    return FeatureFlags(
        agent_tools=(
            flag("agent_tools", True)
            if "agent_tools" in payload
            else flag("workspace", True)
        ),
        workspace_write=flag("workspace_write", False),
        workspace_run_commands=flag("workspace_run_commands", False),
        local_rag=flag("local_rag", True),
        web_search=flag("web_search", False),
        allowed_commands=sorted(set(allowed_commands)),
    )


def allowed_workspace_tools(features: FeatureFlags, include_write: bool = False) -> List[str]:
    """Return workspace tools allowed by the current per-turn approvals."""
    allowed = ["workspace.list_files", "workspace.grep", "workspace.read_file", "spreadsheet.describe"]
    if features.workspace_write and include_write:
        allowed.append("workspace.patch_file")
    if features.workspace_run_commands:
        allowed.append("workspace.run_command")
    return allowed

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

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_base_dir = pathlib.Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_base_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(_base_dir / "static"))

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
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "themes_json": json.dumps({"light": COLORS_LIGHT, "dark": COLORS_DARK}),
                "model_name": get_active_model_name(),
                "app_title": APP_TITLE,
            },
        )
    except Exception as e:
        logger.error(f"Error generating home page: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error loading page: {str(e)}")


@app.get("/api/voice/status")
async def voice_status():
    return get_voice_runtime_summary()


@app.post("/api/voice/transcribe")
async def transcribe_voice_audio(file: UploadFile = File(...)):
    runtime = get_voice_runtime_summary()
    if not runtime["stt_available"]:
        raise HTTPException(
            status_code=503,
            detail="Server transcription is unavailable. Install a native STT tool or set VOICE_STT_COMMAND.",
        )

    suffix = pathlib.Path(file.filename or "").suffix.lower()[:16] or ".webm"
    session_id = uuid.uuid4().hex
    input_dir = get_voice_dir("input")
    output_dir = get_voice_dir("transcripts")
    input_path = input_dir / f"{session_id}{suffix}"
    transcript_path = output_dir / f"{session_id}.txt"
    bytes_written = 0

    try:
        with input_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > VOICE_INPUT_SIZE_LIMIT:
                    raise HTTPException(status_code=400, detail="Audio file is too large")
                handle.write(chunk)
    finally:
        await file.close()

    replacements = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "transcript": str(transcript_path),
    }
    command = build_voice_command(
        VOICE_STT_COMMAND,
        replacements,
        fallback=default_stt_command(input_path, output_dir),
    )
    result = await run_native_voice_command(command)
    transcript = read_transcript_output(output_dir, input_path, transcript_path)
    return {
        "transcript": transcript,
        "bytes": bytes_written,
        "content_type": file.content_type or "application/octet-stream",
        "backend": runtime["stt_backend"],
        "command": result["command"][0],
    }


@app.post("/api/voice/speak")
async def synthesize_voice_audio(request: VoiceSynthesisRequest):
    runtime = get_voice_runtime_summary()
    if not runtime["tts_available"]:
        raise HTTPException(
            status_code=503,
            detail="Server speech synthesis is unavailable. Install a native TTS tool or set VOICE_TTS_COMMAND.",
        )

    text = str(request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    speech_id = uuid.uuid4().hex
    text_dir = get_voice_dir("tts-text")
    output_dir = get_voice_dir("tts-audio")
    text_path = text_dir / f"{speech_id}.txt"
    output_ext = infer_tts_output_suffix(VOICE_TTS_COMMAND)
    output_path = output_dir / f"{speech_id}{output_ext}"
    text_path.write_text(text, encoding="utf-8")

    replacements = {
        "input": "",
        "output": str(output_path),
        "output_dir": str(output_dir),
        "transcript": "",
        "textfile": str(text_path),
    }
    result = await run_native_voice_command(
        build_voice_command(
            VOICE_TTS_COMMAND,
            replacements,
            fallback=default_tts_command(output_path, text_path),
        )
    )
    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Speech synthesis finished but no audio file was created")
    media_type = guess_audio_media_type(output_path)
    return {
        "audio_url": f"/api/voice/file/{output_path.name}",
        "media_type": media_type,
        "backend": runtime["tts_backend"],
        "voice": runtime["tts_voice"],
        "command": result["command"][0],
    }


@app.get("/api/voice/file/{filename}")
async def get_voice_file(filename: str):
    safe_name = pathlib.Path(filename).name
    target = (get_voice_dir("tts-audio", create=False) / safe_name).resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    if get_voice_dir("tts-audio", create=False) not in target.parents:
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(target, media_type=guess_audio_media_type(target), filename=target.name)


@app.get("/api/pet")
async def get_pet():
    profile = ensure_default_agent_profile()
    return {"exists": True, "pet": profile, "stats": get_pet_stats()}


@app.post("/api/pet/adopt")
async def adopt_pet(request: PetAdoptRequest):
    profile = upsert_pet_profile(request.model_dump(), create=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO pet_events (type, payload_json, created_at) VALUES (?, ?, ?)',
        ("adopted", json.dumps({"name": profile["name"]}), utcnow_iso()),
    )
    conn.commit()
    conn.close()
    return {"status": "success", "pet": profile, "stats": get_pet_stats()}


@app.patch("/api/pet")
async def update_pet(request: PetUpdateRequest):
    payload = {key: value for key, value in request.model_dump().items() if value is not None}
    profile = upsert_pet_profile(payload, create=False)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO pet_events (type, payload_json, created_at) VALUES (?, ?, ?)',
        ("updated", json.dumps(payload), utcnow_iso()),
    )
    conn.commit()
    conn.close()
    return {"status": "success", "pet": profile, "stats": get_pet_stats()}


@app.post("/api/pet/restart")
async def restart_pet():
    profile = ensure_default_agent_profile()
    ensure_pet_dirs()
    runtime_path = get_pet_root() / "state" / "runtime.json"
    payload = {"status": "active", "last_restart_at": utcnow_iso(), "mode": "restart"}
    with runtime_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE pet_profile SET status = ?, updated_at = ? WHERE id = 1', ("active", utcnow_iso()))
    conn.execute(
        'INSERT INTO pet_events (type, payload_json, created_at) VALUES (?, ?, ?)',
        ("restart", json.dumps(payload), utcnow_iso()),
    )
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"{profile['name']} restarted."}


@app.post("/api/pet/respawn")
async def respawn_pet():
    profile = ensure_default_agent_profile()
    ensure_pet_dirs()
    runtime_path = get_pet_root() / "state" / "runtime.json"
    payload = {"status": "active", "last_respawn_at": utcnow_iso(), "mode": "respawn"}
    with runtime_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE pet_profile SET status = ?, updated_at = ? WHERE id = 1', ("active", utcnow_iso()))
    conn.execute(
        'INSERT INTO pet_events (type, payload_json, created_at) VALUES (?, ?, ?)',
        ("respawn", json.dumps(payload), utcnow_iso()),
    )
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"{profile['name']} runtime reset."}


@app.post("/api/pet/feed")
async def feed_pet_route(request: PetFeedRequest):
    profile = feed_pet(request.kind, request.note, request.source_run_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Agent profile unavailable")
    return {
        "status": "success",
        "pet": profile,
        "stats": get_pet_stats(),
        "message": f"{profile['name']} enjoyed a {request.kind}.",
    }


@app.get("/api/pet/memories")
async def get_pet_memories():
    return {"memories": list_pet_memories()}


@app.post("/api/pet/memories")
async def create_memory(request: PetMemoryCreateRequest):
    memory = create_pet_memory(request.model_dump())
    profile = feed_pet("knowledge", note=memory.get("title", ""), source_run_id=memory.get("source_run_id"))
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO pet_events (type, payload_json, created_at) VALUES (?, ?, ?)',
        ("memory_added", json.dumps({"id": memory.get("id"), "title": memory.get("title")}), utcnow_iso()),
    )
    conn.commit()
    conn.close()
    return {"status": "success", "memory": memory, "pet": profile, "stats": get_pet_stats()}


@app.patch("/api/pet/memories/{memory_id}")
async def update_memory(memory_id: int, request: PetMemoryUpdateRequest):
    memory = update_pet_memory_record(memory_id, request.model_dump())
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "success", "memory": memory}


@app.delete("/api/pet/memories/{memory_id}")
async def delete_memory(memory_id: int):
    if not delete_pet_memory_record(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "success"}


@app.get("/api/pet/capabilities")
async def get_pet_capability_list():
    return {"capabilities": list_pet_capabilities()}


@app.post("/api/pet/capabilities/promote")
async def promote_capability(request: CapabilityPromoteRequest):
    capability = promote_run_artifact(request.model_dump())
    profile = feed_pet("capability", note=capability.get("name", ""), source_run_id=capability.get("source_run_id"))
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO pet_events (type, payload_json, created_at) VALUES (?, ?, ?)',
        ("capability_promoted", json.dumps({"id": capability.get("id"), "name": capability.get("name")}), utcnow_iso()),
    )
    conn.commit()
    conn.close()
    return {"status": "success", "capability": capability, "pet": profile, "stats": get_pet_stats()}


@app.patch("/api/pet/capabilities/{capability_id}")
async def update_capability(capability_id: int, request: CapabilityUpdateRequest):
    capability = update_pet_capability_record(capability_id, request.model_dump())
    if not capability:
        raise HTTPException(status_code=404, detail="Capability not found")
    return {"status": "success", "capability": capability}


@app.delete("/api/pet/capabilities/{capability_id}")
async def delete_capability(capability_id: int):
    if not delete_pet_capability_record(capability_id):
        raise HTTPException(status_code=404, detail="Capability not found")
    return {"status": "success"}

@app.get("/api/pet/bond")
async def get_bond():
    ensure_default_agent_profile()
    return get_pet_bond()


@app.post("/api/pet/bond/pet")
async def pet_the_dog():
    ensure_default_agent_profile()
    return register_pet_action()


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
        delete_run_workspace(conversation_id)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM conversation_summaries WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM runs WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        conn.commit()
        conn.close()
        return {'status': 'success'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset-all")
async def reset_all_application_data():
    try:
        await reset_application_state()
        return {"status": "success"}
    except Exception as e:
        logger.error("Error resetting application data: %s", e)
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
    if not EXECUTE_CODE_ENABLED:
        raise HTTPException(status_code=403, detail="Ad-hoc code execution is disabled on this server")

    code = request.get('code', '').strip()
    language = request.get('language', 'python').lower()
    conversation_id = request.get('conversation_id', '').strip()
    workspace_path_raw = request.get('path', '').strip()

    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    if len(code) > 10000:
        raise HTTPException(status_code=400, detail="Code too long (max 10KB)")

    try:
        if language == 'python':
            import tempfile

            workspace_dir: Optional[pathlib.Path] = None
            temp_dir = None
            if conversation_id:
                workspace_dir = resolve_workspace_relative_path(conversation_id, workspace_path_raw)
                if workspace_dir.is_file():
                    workspace_dir = workspace_dir.parent
                workspace_dir.mkdir(parents=True, exist_ok=True)
                temp_dir = str(workspace_dir)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=temp_dir) as f:
                f.write(code)
                temp_file = f.name

            try:
                process = await asyncio.create_subprocess_exec(
                    'python3', temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace_dir) if workspace_dir else None,
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


@app.get("/api/workspace/{conversation_id}")
async def get_workspace_info(conversation_id: str):
    run = get_run_record(conversation_id)
    workspace = get_workspace_path(conversation_id, create=False)
    return {
        "conversation_id": conversation_id,
        "run_id": run["id"] if run else None,
        "workspace_path": str(workspace),
        "workspace_label": workspace.name,
    }


@app.post("/api/workspace/{conversation_id}/upload")
async def upload_workspace_files(
    conversation_id: str,
    files: List[UploadFile] = File(...),
    target_path: str = Form("."),
):
    workspace = get_workspace_path(conversation_id)
    target_dir = resolve_workspace_relative_path(conversation_id, target_path or ".")

    if target_dir.exists() and not target_dir.is_dir():
        raise HTTPException(status_code=400, detail="target_path must be a directory")
    target_dir.mkdir(parents=True, exist_ok=True)

    uploads = files[:MAX_ATTACHMENTS_PER_MESSAGE]
    saved_files = []
    for upload in uploads:
        filename = sanitize_uploaded_filename(upload.filename or "attachment")
        unique_name = ensure_unique_workspace_filename(target_dir, filename)
        destination = target_dir / unique_name
        size = 0

        try:
            with destination.open("wb") as f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > UPLOAD_FILE_SIZE_LIMIT:
                        raise HTTPException(status_code=400, detail=f"{filename} is too large (max 10MB)")
                    f.write(chunk)
        except Exception:
            try:
                destination.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        finally:
            await upload.close()

        saved_files.append({
            "name": destination.name,
            "path": format_workspace_path(destination, workspace),
            "size": size,
            "content_type": upload.content_type or "application/octet-stream",
            "kind": "spreadsheet" if is_spreadsheet_path(destination.name) else "file",
        })

    return {
        "conversation_id": conversation_id,
        "target_path": format_workspace_path(target_dir, workspace),
        "files": saved_files,
        "count": len(saved_files),
    }


@app.get("/api/workspace/{conversation_id}/files")
async def list_workspace_files(conversation_id: str, path: str = ""):
    run = get_run_record(conversation_id)
    workspace = get_workspace_path(conversation_id, create=False)
    if not run or not workspace.exists():
        return {
            "conversation_id": conversation_id,
            "run_id": run["id"] if run else None,
            "workspace_path": str(workspace),
            "path": ".",
            "items": [],
        }
    target = resolve_workspace_relative_path(conversation_id, path)

    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    items = []
    for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            stat = item.stat()
            items.append({
                "name": item.name,
                "path": format_workspace_path(item, workspace),
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else None,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except (OSError, PermissionError):
            continue

    return {
        "conversation_id": conversation_id,
        "run_id": run["id"] if run else None,
        "workspace_path": str(workspace),
        "path": format_workspace_path(target, workspace),
        "items": items,
    }


@app.get("/api/workspace/{conversation_id}/file")
async def read_workspace_file(conversation_id: str, path: str):
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, path)

    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.stat().st_size > 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 1MB)")

    with target.open('r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    return {
        "conversation_id": conversation_id,
        "workspace_path": str(workspace),
        "path": format_workspace_path(target, workspace),
        "content": content,
        "size": len(content),
        "lines": content.count('\n') + 1,
    }


@app.post("/api/workspace/{conversation_id}/file")
async def write_workspace_file(conversation_id: str, request: WorkspaceFileUpdateRequest):
    workspace = get_workspace_path(conversation_id)
    rel_path = (request.path or "").strip()
    if not rel_path:
        raise HTTPException(status_code=400, detail="Path is required")

    target = resolve_workspace_relative_path(conversation_id, rel_path)
    if target.exists() and not target.is_file():
        raise HTTPException(status_code=400, detail="Path must point to a file")

    encoded = request.content.encode("utf-8")
    if len(encoded) > WORKSPACE_WRITE_SIZE_LIMIT:
        raise HTTPException(status_code=400, detail="File content too large (max 1MB)")

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write(request.content)

    return {
        "conversation_id": conversation_id,
        "workspace_path": str(workspace),
        "path": format_workspace_path(target, workspace),
        "bytes_written": len(encoded),
        "lines": request.content.count("\n") + 1,
    }


@app.post("/api/workspace/{conversation_id}/recover")
async def recover_assistant_message_to_workspace(conversation_id: str, request: RecoverAssistantMessageRequest):
    workspace = get_workspace_path(conversation_id)
    source_message = None

    if request.message_id is not None:
        source_message = get_message_by_id(request.message_id)
        if not source_message:
            raise HTTPException(status_code=404, detail="Message not found")
        if source_message["conversation_id"] != conversation_id:
            raise HTTPException(status_code=400, detail="Message does not belong to this conversation")
        if source_message["role"] != "assistant":
            raise HTTPException(status_code=400, detail="Only assistant messages can be recovered")

    message_content = (
        source_message["content"]
        if source_message is not None
        else str(request.content or "")
    ).strip()
    if not message_content:
        raise HTTPException(status_code=400, detail="No assistant content was provided")

    now = datetime.now()
    source_label = f"assistant message #{source_message['id']}" if source_message else "assistant draft"
    default_target = f"recovered/{now.strftime('%Y%m%d-%H%M%S')}"
    base_rel = sanitize_relative_workspace_path(request.target_dir or default_target, fallback=default_target)
    base_dir = resolve_workspace_relative_path(conversation_id, base_rel)
    if base_dir.exists():
        base_rel = sanitize_relative_workspace_path(f"{base_rel}-{uuid.uuid4().hex[:6]}", fallback=default_target)
        base_dir = resolve_workspace_relative_path(conversation_id, base_rel)
    base_dir.mkdir(parents=True, exist_ok=True)

    files_dir = base_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    matches = list(RECOVERED_CODE_BLOCK_RE.finditer(message_content))
    created_paths: List[str] = []

    for index, match in enumerate(matches, start=1):
        info = (match.group(1) or "").strip()
        code = match.group(2) or ""
        context_before = message_content[max(0, match.start() - 500):match.start()]
        inferred_name = infer_filename_for_code_block(info, context_before, message_content, index, len(matches))
        safe_name = sanitize_relative_workspace_path(inferred_name, fallback=f"snippet-{index}.txt")
        destination = files_dir / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            replacement = ensure_unique_workspace_filename(destination.parent, destination.name)
            destination = destination.parent / replacement
        destination.write_text(code.rstrip() + "\n", encoding="utf-8")
        created_paths.append(format_workspace_path(destination, workspace))

    readme_path = base_dir / "README.md"
    readme_path.write_text(
        build_recovered_notes_markdown(
            source_label=source_label,
            created_paths=created_paths,
            message_content=message_content,
            recovered_at=now.isoformat(),
        ),
        encoding="utf-8",
    )

    all_paths = [format_workspace_path(readme_path, workspace), *created_paths]
    return {
        "conversation_id": conversation_id,
        "workspace_path": str(workspace),
        "source": source_label,
        "target_dir": format_workspace_path(base_dir, workspace),
        "readme_path": format_workspace_path(readme_path, workspace),
        "created_files": created_paths,
        "paths": all_paths,
        "count": len(created_paths),
    }


@app.get("/api/workspace/{conversation_id}/spreadsheet")
async def read_workspace_spreadsheet(conversation_id: str, path: str, sheet: Optional[str] = None):
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, path)

    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        summary = load_spreadsheet_summary(target, sheet=sheet)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "conversation_id": conversation_id,
        "workspace_path": str(workspace),
        "path": format_workspace_path(target, workspace),
        **summary,
    }


@app.get("/api/workspace/{conversation_id}/download")
async def download_workspace(conversation_id: str):
    workspace = get_workspace_path(conversation_id)
    archive = io.BytesIO()

    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in workspace.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(workspace).as_posix())

    headers = {
        "Content-Disposition": f'attachment; filename="{sanitize_conversation_id(conversation_id)}-workspace.zip"'
    }
    return Response(content=archive.getvalue(), media_type="application/zip", headers=headers)


def _validate_tool_call_payload(payload: Any) -> Dict[str, Any]:
    """Validate a parsed tool-call payload and normalize the result."""
    if not isinstance(payload, dict):
        raise ValueError("Tool call payload must be a JSON object")

    call_id = str(payload.get("id", "")).strip()
    name = str(payload.get("name", "")).strip()
    arguments = payload.get("arguments", {})

    if not call_id:
        raise ValueError("Tool call is missing id")
    if not name:
        raise ValueError("Tool call is missing name")
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be an object")

    return {"id": call_id, "name": name, "arguments": arguments}


def parse_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid tool call from model output."""
    cleaned = strip_stream_special_tokens(raw).strip()
    if not cleaned:
        return None

    decoder = json.JSONDecoder()

    wrapped_match = re.search(r"<tool_call>(?P<body>.*?)</tool_call>", cleaned, re.DOTALL)
    if wrapped_match:
        wrapped_body = wrapped_match.group("body").strip()
        for match in re.finditer(r"\{", wrapped_body):
            try:
                payload, _ = decoder.raw_decode(wrapped_body[match.start():])
            except json.JSONDecodeError:
                continue
            return _validate_tool_call_payload(payload)

    for match in re.finditer(r"\{", cleaned):
        try:
            payload, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        try:
            return _validate_tool_call_payload(payload)
        except ValueError:
            continue

    return None


def workspace_list_files_result(conversation_id: str, rel_path: str = "") -> Dict[str, Any]:
    """List files inside the conversation workspace."""
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, rel_path)

    if not target.exists():
        raise ValueError("Path not found")
    if not target.is_dir():
        raise ValueError("Path is not a directory")

    items = []
    for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            items.append({
                "name": item.name,
                "path": format_workspace_path(item, workspace),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        except (OSError, PermissionError):
            continue

    return {
        "path": format_workspace_path(target, workspace),
        "items": items,
    }


def is_probably_binary_file(target: pathlib.Path) -> bool:
    """Cheap binary-file check so grep stays focused on readable text."""
    try:
        with target.open("rb") as f:
            sample = f.read(4096)
    except OSError:
        return True
    return b"\x00" in sample


def workspace_grep_result(
    conversation_id: str,
    query: str,
    rel_path: str = ".",
    glob: str = "*",
    limit: int = 20,
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """Search workspace text files for a query and return line-level matches."""
    cleaned_query = str(query or "").strip()
    if len(cleaned_query) < 2:
        raise ValueError("query must be at least 2 characters")

    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, rel_path or ".")
    if not target.exists():
        raise ValueError("Path not found")

    glob_pattern = str(glob or "*").strip() or "*"
    safe_limit = max(1, min(int(limit), 100))
    file_candidates: List[pathlib.Path]

    if target.is_file():
        file_candidates = [target]
    elif target.is_dir():
        file_candidates = sorted(
            (candidate for candidate in target.rglob(glob_pattern) if candidate.is_file()),
            key=lambda item: format_workspace_path(item, workspace),
        )
    else:
        raise ValueError("path must be a file or directory")

    matches: List[Dict[str, Any]] = []
    files_scanned = 0
    files_with_matches: set[str] = set()
    skipped_files = 0
    needle = cleaned_query if case_sensitive else cleaned_query.casefold()

    for file_path in file_candidates:
        try:
            file_size = file_path.stat().st_size
        except OSError:
            skipped_files += 1
            continue

        if file_size > WORKSPACE_FILE_SIZE_LIMIT or is_probably_binary_file(file_path):
            skipped_files += 1
            continue

        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            skipped_files += 1
            continue

        files_scanned += 1
        rel_file_path = format_workspace_path(file_path, workspace)
        for line_number, line in enumerate(lines, start=1):
            haystack = line if case_sensitive else line.casefold()
            column_index = haystack.find(needle)
            if column_index < 0:
                continue
            match_text = line.rstrip("\n\r")
            before_line = lines[line_number - 2].rstrip("\n\r") if line_number > 1 else ""
            after_line = lines[line_number].rstrip("\n\r") if line_number < len(lines) else ""
            matches.append({
                "path": rel_file_path,
                "line": line_number,
                "column": column_index + 1,
                "text": truncate_output(match_text, limit=240),
                "before": truncate_output(before_line, limit=160) if before_line else "",
                "after": truncate_output(after_line, limit=160) if after_line else "",
            })
            files_with_matches.add(rel_file_path)
            if len(matches) >= safe_limit:
                return {
                    "query": cleaned_query,
                    "path": format_workspace_path(target, workspace),
                    "glob": glob_pattern,
                    "count": len(matches),
                    "files_scanned": files_scanned,
                    "matched_files": len(files_with_matches),
                    "skipped_files": skipped_files,
                    "truncated": True,
                    "paths": sorted(files_with_matches),
                    "grep_matches": matches,
                }

    return {
        "query": cleaned_query,
        "path": format_workspace_path(target, workspace),
        "glob": glob_pattern,
        "count": len(matches),
        "files_scanned": files_scanned,
        "matched_files": len(files_with_matches),
        "skipped_files": skipped_files,
        "truncated": False,
        "paths": sorted(files_with_matches),
        "grep_matches": matches,
    }


def workspace_read_file_result(conversation_id: str, rel_path: str) -> Dict[str, Any]:
    """Read a text file from the conversation workspace."""
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, rel_path)

    if not target.is_file():
        raise ValueError("File not found")
    if target.stat().st_size > WORKSPACE_FILE_SIZE_LIMIT:
        raise ValueError("File too large (max 1MB)")

    with target.open("r", encoding="utf-8", errors="replace") as f:
        content = truncate_output(f.read(), limit=TOOL_RESULT_TEXT_LIMIT)

    return {
        "path": format_workspace_path(target, workspace),
        "content": content,
        "size": len(content),
        "lines": content.count("\n") + 1,
    }


def spreadsheet_describe_result(conversation_id: str, rel_path: str, sheet: Optional[str] = None) -> Dict[str, Any]:
    """Inspect a spreadsheet-like file inside the conversation workspace."""
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, rel_path)

    if not target.is_file():
        raise ValueError("File not found")

    summary = load_spreadsheet_summary(target, sheet=sheet)
    return {
        "path": format_workspace_path(target, workspace),
        **summary,
    }


def build_patch_suggestions(content: str, old_text: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Return a few nearby snippets that may help the model recover from a patch miss."""
    snippets: List[Dict[str, Any]] = []
    search_lines = [line for line in old_text.splitlines() if line.strip()]
    content_lines = content.splitlines()

    for needle in search_lines[:3]:
        for index, line in enumerate(content_lines, start=1):
            if needle.strip() in line:
                snippets.append({
                    "line": index,
                    "snippet": line[:240],
                    "reason": "substring_match",
                })
                if len(snippets) >= limit:
                    return snippets

    if content_lines and len(snippets) < limit:
        ranked = sorted(
            (
                (
                    SequenceMatcher(None, old_text.strip(), line.strip()).ratio(),
                    index,
                    line,
                )
                for index, line in enumerate(content_lines, start=1)
                if line.strip()
            ),
            reverse=True,
        )
        for score, index, line in ranked:
            if score < 0.35:
                break
            candidate = {
                "line": index,
                "snippet": line[:240],
                "reason": "similar_line",
                "score": round(score, 2),
            }
            if candidate not in snippets:
                snippets.append(candidate)
            if len(snippets) >= limit:
                break

    return snippets


def apply_exact_patch_edits(content: str, edits: Any, rel_path: str) -> tuple[str, int]:
    """Apply exact-match text edits to an in-memory file."""
    if not isinstance(edits, list) or not edits:
        raise ValueError("edits must be a non-empty array")

    updated = content
    applied = 0
    for index, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            raise ValueError(f"edit {index} must be an object")

        old_text = edit.get("old_text")
        new_text = edit.get("new_text")
        expected_count = edit.get("expected_count", 1)

        if not isinstance(old_text, str) or not old_text:
            raise ValueError(f"edit {index} old_text must be a non-empty string")
        if not isinstance(new_text, str):
            raise ValueError(f"edit {index} new_text must be a string")
        if not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError(f"edit {index} expected_count must be a positive integer")

        match_count = updated.count(old_text)
        if match_count != expected_count:
            raise PatchApplicationError(
                f"edit {index} expected {expected_count} match(es) for old_text, found {match_count}",
                {
                    "type": "patch_mismatch",
                    "path": rel_path,
                    "edit_index": index,
                    "expected_count": expected_count,
                    "actual_count": match_count,
                    "old_text_excerpt": old_text[:240],
                    "suggestions": build_patch_suggestions(updated, old_text),
                },
            )

        updated = updated.replace(old_text, new_text, expected_count)
        applied += expected_count

    return updated, applied


def workspace_patch_file_result(
    conversation_id: str,
    rel_path: str,
    edits: Any = None,
    create: Any = False,
    new_content: Any = None,
) -> Dict[str, Any]:
    """Patch a text file inside the conversation workspace using exact-match edits."""
    target = resolve_workspace_relative_path(conversation_id, rel_path)
    target_exists = target.exists()

    if target_exists and not target.is_file():
        raise ValueError("path must point to a file")

    if target_exists:
        if create:
            raise ValueError("create can only be used for missing files")
        if new_content is not None:
            raise ValueError("new_content is only allowed when create is true")
        if target.stat().st_size > WORKSPACE_FILE_SIZE_LIMIT:
            raise ValueError("File too large (max 1MB)")

        with target.open("r", encoding="utf-8", errors="replace") as f:
            original = f.read()
        updated, edits_applied = apply_exact_patch_edits(original, edits, rel_path)
    else:
        if create is not True:
            raise ValueError("File not found; set create=true with new_content to create it")
        if edits not in (None, []):
            raise ValueError("edits must be omitted when create is true")
        if not isinstance(new_content, str):
            raise ValueError("new_content must be a string when create is true")
        updated = new_content
        edits_applied = 0

    encoded = updated.encode("utf-8")
    if len(encoded) > WORKSPACE_WRITE_SIZE_LIMIT:
        raise ValueError("File content too large (max 1MB)")

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write(updated)

    workspace = get_workspace_path(conversation_id)
    return {
        "path": format_workspace_path(target, workspace),
        "edits_applied": edits_applied,
        "bytes_written": len(encoded),
        "created": not target_exists,
    }


async def workspace_run_command_result(
    conversation_id: str,
    command: Any,
    cwd: str = ".",
    features: Optional[FeatureFlags] = None,
) -> Dict[str, Any]:
    """Run a short-lived command inside the conversation workspace."""
    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty array of strings")
    if not all(isinstance(part, str) and part.strip() for part in command):
        raise ValueError("command entries must be non-empty strings")

    cwd_path = resolve_workspace_relative_path(conversation_id, cwd or ".")
    if not cwd_path.exists():
        raise ValueError("cwd not found")
    if not cwd_path.is_dir():
        raise ValueError("cwd must be a directory")

    validate_workspace_command(conversation_id, command, cwd_path, features)

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd_path),
            env={**os.environ, "PYTHONPATH": ""},
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Command not found: {command[0]}") from exc

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=COMMAND_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ValueError(f"Command timed out after {COMMAND_TIMEOUT_SECONDS:g}s")

    workspace = get_workspace_path(conversation_id)
    return {
        "stdout": truncate_output(stdout.decode("utf-8", errors="replace")),
        "stderr": truncate_output(stderr.decode("utf-8", errors="replace")),
        "returncode": process.returncode,
        "cwd": format_workspace_path(cwd_path, workspace),
    }


async def _send_terminal_json(websocket: WebSocket, payload: Dict[str, Any]) -> None:
    """Best-effort terminal websocket send."""
    try:
        await websocket.send_json(payload)
    except Exception:
        pass


async def _stream_terminal_output(websocket: WebSocket, master_fd: int) -> None:
    """Forward PTY output into the browser terminal."""
    while True:
        try:
            chunk = await asyncio.to_thread(os.read, master_fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        await _send_terminal_json(
            websocket,
            {
                "type": "terminal_output",
                "content": chunk.decode("utf-8", errors="replace"),
            },
        )


async def _close_terminal_session(session: Optional[TerminalSession]) -> None:
    """Terminate and dispose a PTY-backed terminal session."""
    if not session:
        return

    if session.reader_task and not session.reader_task.done():
        session.reader_task.cancel()
        try:
            await session.reader_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    try:
        os.close(session.master_fd)
    except OSError:
        pass

    if session.process.returncode is None:
        try:
            session.process.terminate()
            await asyncio.wait_for(session.process.wait(), timeout=1.0)
        except Exception:
            try:
                session.process.kill()
                await session.process.wait()
            except Exception:
                pass


def _resize_terminal_session(session: TerminalSession, cols: int, rows: int) -> None:
    """Propagate browser terminal dimensions into the PTY."""
    safe_cols = max(20, min(int(cols), 500))
    safe_rows = max(5, min(int(rows), 200))
    winsize = struct.pack("HHHH", safe_rows, safe_cols, 0, 0)
    fcntl.ioctl(session.master_fd, termios.TIOCSWINSZ, winsize)


async def _start_terminal_session(conversation_id: str, websocket: WebSocket) -> TerminalSession:
    """Create a new interactive shell session rooted in the workspace."""
    workspace_path = get_workspace_path(conversation_id)
    shell_path = os.getenv("SHELL", "").strip() or "/bin/bash"
    if not pathlib.Path(shell_path).exists():
        shell_path = "/bin/sh"

    master_fd, slave_fd = pty.openpty()
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            shell_path,
            "-i",
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=str(workspace_path),
            env={
                **os.environ,
                "TERM": "xterm-256color",
                "PYTHONPATH": "",
            },
            start_new_session=True,
        )
    finally:
        try:
            os.close(slave_fd)
        except OSError:
            pass

    reader_task = asyncio.create_task(_stream_terminal_output(websocket, master_fd))
    session = TerminalSession(
        conversation_id=conversation_id,
        workspace_path=workspace_path,
        process=process,
        master_fd=master_fd,
        reader_task=reader_task,
    )
    await _send_terminal_json(
        websocket,
        {
            "type": "terminal_status",
            "content": f"Shell connected in {workspace_path.name or '.'}",
        },
    )
    return session


def conversation_search_history_result(
    conversation_id: str,
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """Search the current conversation for lightweight retrieval."""
    cleaned_query = (query or "").strip()
    if len(cleaned_query) < 2:
        raise ValueError("query must be at least 2 characters")

    safe_limit = max(1, min(int(limit), 8))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    fts_query = normalize_fts_query(cleaned_query)

    try:
        if not fts_query:
            raise sqlite3.OperationalError("No valid FTS tokens")
        c.execute(
            f'''SELECT m.id, m.role, m.content, m.timestamp, m.feedback, bm25({FTS_TABLE}) as fts_rank
                FROM {FTS_TABLE} f
                JOIN messages m ON m.id = f.rowid
                WHERE {FTS_TABLE} MATCH ? AND m.conversation_id = ?
                ORDER BY bm25({FTS_TABLE}), m.timestamp DESC
                LIMIT ?''',
            (fts_query, conversation_id, max(safe_limit * 4, 12)),
        )
        ranked_rows = rerank_search_rows([
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "timestamp": row[3],
                "feedback": row[4] or "",
                "fts_rank": row[5],
            }
            for row in c.fetchall()
        ], cleaned_query, safe_limit)
    except sqlite3.OperationalError:
        search_term = f"%{cleaned_query}%"
        c.execute(
            '''SELECT id, role, content, timestamp
               FROM messages
               WHERE conversation_id = ? AND content LIKE ?
               ORDER BY timestamp DESC
               LIMIT ?''',
            (conversation_id, search_term, safe_limit),
        )
        ranked_rows = [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "timestamp": row[3],
            }
            for row in c.fetchall()
        ]

    matches = []
    for row in ranked_rows:
        message_id = row.get("id")
        context_before = ""
        context_after = ""
        if message_id is not None:
            previous_row = c.execute(
                '''SELECT role, content
                   FROM messages
                   WHERE conversation_id = ? AND id < ?
                   ORDER BY id DESC
                   LIMIT 1''',
                (conversation_id, message_id),
            ).fetchone()
            next_row = c.execute(
                '''SELECT role, content
                   FROM messages
                   WHERE conversation_id = ? AND id > ?
                   ORDER BY id ASC
                   LIMIT 1''',
                (conversation_id, message_id),
            ).fetchone()
            if previous_row:
                context_before = f"{previous_row[0]}: {build_query_excerpt(previous_row[1], cleaned_query, window=140)}"
            if next_row:
                context_after = f"{next_row[0]}: {build_query_excerpt(next_row[1], cleaned_query, window=140)}"

        snippet = build_query_excerpt(row["content"], cleaned_query, window=260)
        matches.append({
            "role": row["role"],
            "content": truncate_output(row["content"], limit=800),
            "snippet": truncate_output(snippet, limit=320),
            "timestamp": row["timestamp"],
            "context_before": truncate_output(context_before, limit=180) if context_before else "",
            "context_after": truncate_output(context_after, limit=180) if context_after else "",
        })
    conn.close()

    return {
        "query": cleaned_query,
        "matches": matches,
        "count": len(matches),
    }


def strip_html(text: str) -> str:
    """Collapse HTML snippets into plain text."""
    no_tags = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", unescape(no_tags)).strip()


def normalize_search_result_url(url: str) -> str:
    """Resolve DuckDuckGo redirect links to the underlying URL when possible."""
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return target
    if url.startswith("//"):
        return "https:" + url
    return url


def sanitize_web_domain(domain: str) -> Optional[str]:
    """Normalize a user-supplied domain filter into a host name."""
    cleaned = str(domain or "").strip().lower()
    if not cleaned:
        return None
    if "://" in cleaned:
        cleaned = urlparse(cleaned).netloc.lower()
    else:
        cleaned = cleaned.split("/", 1)[0].lower()
    cleaned = cleaned.strip(".")
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]
    if not cleaned or not re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", cleaned):
        return None
    return cleaned


def normalize_web_domain(url_or_domain: str) -> str:
    """Extract a stable normalized domain from a URL or host string."""
    if "://" in str(url_or_domain or ""):
        domain = sanitize_web_domain(url_or_domain)
        return domain or ""
    return sanitize_web_domain(url_or_domain) or ""


def canonicalize_http_url(url: str) -> str:
    """Normalize a URL for dedupe and citation matching."""
    normalized = normalize_search_result_url(str(url or "").strip())
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return normalized
    clean_path = parsed.path or "/"
    if clean_path != "/" and clean_path.endswith("/"):
        clean_path = clean_path.rstrip("/")
    clean_netloc = parsed.netloc.lower()
    if clean_netloc.startswith("www."):
        clean_netloc = clean_netloc[4:]
    return urlunparse((parsed.scheme.lower(), clean_netloc, clean_path, "", parsed.query, ""))


def apply_web_domain_filters(query: str, domains: Any) -> str:
    """Append site filters to a web query when domains are provided."""
    if not isinstance(domains, list):
        return query
    cleaned_domains: List[str] = []
    for domain in domains[:5]:
        normalized = sanitize_web_domain(str(domain))
        if normalized and normalized not in cleaned_domains:
            cleaned_domains.append(normalized)
    if not cleaned_domains:
        return query
    if len(cleaned_domains) == 1:
        return f"{query} site:{cleaned_domains[0]}"
    filters = " OR ".join(f"site:{domain}" for domain in cleaned_domains)
    return f"{query} ({filters})"


def extract_html_title(html: str) -> str:
    """Return the HTML <title> text when present."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    return strip_html(match.group(1)) if match else ""


def normalize_search_title(text: str) -> str:
    """Normalize a title or query into a compact comparable token string."""
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def compute_title_match_score(query: str, title: str) -> float:
    """Estimate how closely a search result title matches the query."""
    normalized_query = normalize_search_title(query)
    normalized_title = normalize_search_title(title)
    if not normalized_query or not normalized_title:
        return 0.0
    if normalized_query == normalized_title:
        return 1.0
    if normalized_query in normalized_title or normalized_title in normalized_query:
        return 0.92
    query_tokens = set(normalized_query.split())
    title_tokens = set(normalized_title.split())
    if not query_tokens or not title_tokens:
        return 0.0
    overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
    return round(min(overlap, 0.9), 3)


def extract_preferred_html_content(html: str) -> str:
    """Prefer article-like content regions when available."""
    for pattern in (
        r"(?is)<article[^>]*>(.*?)</article>",
        r"(?is)<main[^>]*>(.*?)</main>",
        r'(?is)<div[^>]+role=["\']main["\'][^>]*>(.*?)</div>',
    ):
        match = re.search(pattern, html or "")
        if match and len(match.group(1).strip()) > 400:
            return match.group(1)

    body_match = re.search(r"(?is)<body[^>]*>(.*?)</body>", html or "")
    if body_match:
        return body_match.group(1)
    return html or ""


def html_to_text_content(html: str) -> str:
    """Collapse a full HTML document into readable plain text."""
    preferred = extract_preferred_html_content(html)
    cleaned = re.sub(r"(?is)<(script|style|noscript|svg|canvas|nav|footer|aside|form)[^>]*>.*?</\1>", " ", preferred or "")
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</(p|div|section|article|li|ul|ol|h1|h2|h3|h4|h5|h6|tr|table|blockquote)>", "\n", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def build_query_excerpt(content: str, query: str, window: int = 220) -> str:
    """Extract a focused snippet around the first query-token match."""
    text = re.sub(r"\s+", " ", str(content or "")).strip()
    if not text:
        return ""
    tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", str(query or "").lower()) if len(token) >= 2]
    lowered = text.lower()
    start = -1
    for token in tokens:
        index = lowered.find(token)
        if index >= 0 and (start < 0 or index < start):
            start = index
    if start < 0:
        return truncate_output(text, limit=window)
    snippet_start = max(start - (window // 3), 0)
    snippet_end = min(snippet_start + window, len(text))
    snippet = text[snippet_start:snippet_end].strip()
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(text):
        snippet = snippet + "..."
    return snippet


def find_nearest_workspace_file(start: pathlib.Path, workspace: pathlib.Path, filename: str) -> Optional[pathlib.Path]:
    """Find a file by walking upward from a path inside the workspace."""
    current = start if start.is_dir() else start.parent
    workspace_resolved = workspace.resolve()
    while True:
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current == workspace_resolved or current.parent == current:
            return None
        current = current.parent


def find_nearest_workspace_file_any(start: pathlib.Path, workspace: pathlib.Path, filenames: List[str]) -> Optional[pathlib.Path]:
    """Find the nearest matching file from a candidate list."""
    for filename in filenames:
        match = find_nearest_workspace_file(start, workspace, filename)
        if match:
            return match
    return None


def load_package_json_scripts(package_json_path: pathlib.Path) -> Dict[str, str]:
    """Read package.json scripts without failing the wider tool loop."""
    try:
        with package_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    scripts = payload.get("scripts", {})
    if not isinstance(scripts, dict):
        return {}
    return {
        str(name): str(command)
        for name, command in scripts.items()
        if isinstance(name, str) and isinstance(command, str)
    }


def path_has_segment(path: pathlib.Path, name: str) -> bool:
    """Return whether a path contains a given directory segment."""
    lowered = {part.lower() for part in path.parts}
    return name.lower() in lowered


def infer_auto_verify_command(conversation_id: str, rel_path: str) -> Optional[Dict[str, Any]]:
    """Guess a focused verification command after a file patch."""
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, rel_path)
    rel_target = format_workspace_path(target, workspace)
    ext = target.suffix.lower()
    name = target.name.lower()

    package_json = find_nearest_workspace_file(target, workspace, "package.json")
    if package_json:
        scripts = load_package_json_scripts(package_json)
        package_dir = format_workspace_path(package_json.parent, workspace)
        preferred_scripts: List[str] = []

        if ext in {".ts", ".tsx"}:
            preferred_scripts.extend(["typecheck", "lint", "test", "build"])
        elif ext in {".jsx", ".js", ".mjs", ".cjs", ".css", ".scss", ".sass", ".less", ".html", ".vue", ".svelte"}:
            preferred_scripts.extend(["lint", "test", "build"])
        elif name == "package.json":
            preferred_scripts.extend(["lint", "test", "build"])

        if path_has_segment(target, "__tests__") or name.endswith(".test.ts") or name.endswith(".test.tsx") or name.endswith(".spec.ts") or name.endswith(".spec.tsx") or name.endswith(".test.js") or name.endswith(".spec.js"):
            preferred_scripts = ["test", *[script for script in preferred_scripts if script != "test"]]

        for script_name in preferred_scripts:
            if script_name in scripts:
                return {
                    "command": ["npm", "run", script_name],
                    "cwd": package_dir,
                    "label": f"Auto-verify with npm run {script_name}",
                }

    if ext == ".py":
        python_config = find_nearest_workspace_file_any(
            target,
            workspace,
            ["pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini"],
        )
        if python_config and (
            path_has_segment(target, "tests")
            or name.startswith("test_")
            or name.endswith("_test.py")
        ):
            return {
                "command": ["pytest", rel_target],
                "cwd": format_workspace_path(python_config.parent, workspace),
                "label": f"Auto-verify with pytest {rel_target}",
            }
        return {
            "command": ["python3", "-m", "py_compile", rel_target],
            "cwd": ".",
            "label": f"Auto-verify syntax for {rel_target}",
        }

    if ext in {".js", ".mjs", ".cjs"}:
        return {
            "command": ["node", "--check", rel_target],
            "cwd": ".",
            "label": f"Auto-verify syntax for {rel_target}",
        }

    cargo_toml = find_nearest_workspace_file(target, workspace, "Cargo.toml")
    if cargo_toml and ext == ".rs":
        cargo_dir = format_workspace_path(cargo_toml.parent, workspace)
        if path_has_segment(target, "tests") or name.endswith("_test.rs"):
            return {
                "command": ["cargo", "test"],
                "cwd": cargo_dir,
                "label": "Auto-verify with cargo test",
            }
        return {
            "command": ["cargo", "check"],
            "cwd": cargo_dir,
            "label": "Auto-verify with cargo check",
        }

    go_mod = find_nearest_workspace_file(target, workspace, "go.mod")
    if go_mod and ext == ".go":
        go_dir = format_workspace_path(target.parent, workspace)
        if name.endswith("_test.go"):
            return {
                "command": ["go", "test", "."],
                "cwd": go_dir,
                "label": "Auto-verify with go test .",
            }
        return {
            "command": ["go", "test", "."],
            "cwd": go_dir,
            "label": "Auto-verify with go test .",
        }

    return None


async def web_search_result(query: str, limit: int = 5, domains: Any = None) -> Dict[str, Any]:
    """Fetch a small web result set for freshness-sensitive questions."""
    cleaned_query = (query or "").strip()
    if len(cleaned_query) < 2:
        raise ValueError("query must be at least 2 characters")

    safe_limit = max(1, min(int(limit), 8))
    effective_query = apply_web_domain_filters(cleaned_query, domains)
    domain_filters = []
    if isinstance(domains, list):
        for domain in domains[:5]:
            normalized = sanitize_web_domain(str(domain))
            if normalized and normalized not in domain_filters:
                domain_filters.append(normalized)

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(effective_query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ai-chat/1.0; +https://localhost)",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0), follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text

    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="[^"]+"[^>]*>.*?</a>(?P<rest>.*?)(?:<a[^>]+class="result__snippet"[^>]*>|<div[^>]+class="result__snippet"[^>]*>)(?P<snippet>.*?)(?:</a>|</div>)',
        re.IGNORECASE | re.DOTALL,
    )

    results = []
    seen_urls: set[str] = set()
    cursor = 0
    while len(results) < safe_limit:
        match = pattern.search(html, cursor)
        if not match:
            break
        raw_url = canonicalize_http_url(unescape(match.group("url")))
        title = strip_html(match.group("title"))
        snippet = ""
        snippet_match = snippet_pattern.search(html, match.start())
        if snippet_match and snippet_match.start() == match.start():
            snippet = strip_html(snippet_match.group("snippet"))
        if not raw_url or raw_url in seen_urls:
            cursor = match.end()
            continue
        seen_urls.add(raw_url)
        domain = normalize_web_domain(raw_url)
        results.append({
            "search_rank": len(results) + 1,
            "title": title,
            "url": raw_url,
            "domain": domain,
            "title_match_score": compute_title_match_score(cleaned_query, title),
            "snippet": snippet,
        })
        cursor = match.end()

    requested_domains = set(domain_filters)
    results.sort(
        key=lambda item: (
            -float(item.get("title_match_score", 0.0)),
            -int(bool(item.get("domain")) and item.get("domain") in requested_domains),
            int(item.get("search_rank", 9999)),
        )
    )
    for index, item in enumerate(results, start=1):
        item["position"] = index

    return {
        "query": cleaned_query,
        "effective_query": effective_query,
        "domains": domain_filters,
        "results": results,
        "count": len(results),
        "provider": "duckduckgo_html",
    }


async def web_fetch_page_result(url: str) -> Dict[str, Any]:
    """Fetch and normalize the readable content of a web page."""
    raw_url = canonicalize_http_url(str(url or "").strip())
    if not raw_url:
        raise ValueError("url is required")

    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) URL")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ai-chat/1.0; +https://localhost)",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=True) as client:
        resp = await client.get(raw_url, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        body = resp.text

    page_title = ""
    page_text = ""
    if content_type in {"", "text/html", "application/xhtml+xml"}:
        page_title = extract_html_title(body)
        page_text = html_to_text_content(body)
    elif content_type.startswith("text/"):
        page_text = body.strip()
    else:
        raise ValueError(f"Unsupported page content type: {content_type or 'unknown'}")

    page_text = truncate_output(page_text, limit=WEB_PAGE_TEXT_LIMIT)
    final_url = canonicalize_http_url(str(resp.url))
    page_domain = normalize_web_domain(final_url or raw_url)
    return {
        "url": raw_url,
        "final_url": final_url or str(resp.url),
        "title": page_title,
        "domain": page_domain,
        "content_type": content_type or "text/html",
        "status_code": resp.status_code,
        "content_length": len(page_text),
        "content": page_text,
    }


def tool_status_summary(name: str, arguments: Dict[str, Any]) -> str:
    """Human-readable tool activity for the UI."""
    if name == "workspace.list_files":
        return f"Listing {arguments.get('path', '.')}"
    if name == "workspace.grep":
        target = arguments.get("path", ".")
        query = arguments.get("query", "")
        return f"Searching {target} for {query}"
    if name == "workspace.read_file":
        return f"Reading {arguments.get('path', '')}"
    if name == "workspace.patch_file":
        return f"Patching {arguments.get('path', '')}"
    if name == "workspace.run_command":
        command = arguments.get("command", [])
        if isinstance(command, list):
            command_preview = " ".join(str(part) for part in command[:4])
        else:
            command_preview = str(command)
        return f"Running {command_preview or 'command'}"
    if name == "spreadsheet.describe":
        target = arguments.get("path", "")
        sheet = arguments.get("sheet", "")
        return f"Inspecting {target}{f' [{sheet}]' if sheet else ''}"
    if name == "conversation.search_history":
        return f"Searching chat history for {arguments.get('query', '')}"
    if name == "web.search":
        domains = arguments.get("domains", [])
        if isinstance(domains, list) and domains:
            return f"Searching the web for {arguments.get('query', '')} on {', '.join(str(item) for item in domains[:3])}"
        return f"Searching the web for {arguments.get('query', '')}"
    if name == "web.fetch_page":
        return f"Fetching {arguments.get('url', '')}"
    return name


def tool_result_preview(result: Dict[str, Any]) -> str:
    """Compact preview of a tool result for the UI."""
    if not result.get("ok"):
        details = result.get("details", {})
        if isinstance(details, dict) and details.get("type") == "patch_mismatch":
            return (
                f"Patch mismatch in {details.get('path', '')}: "
                f"expected {details.get('expected_count', '?')} match(es), "
                f"found {details.get('actual_count', '?')}"
            )
        return str(result.get("error", "Tool failed"))

    payload = result.get("result", {})
    if not isinstance(payload, dict):
        return "Completed"

    if "final_url" in payload:
        title = payload.get("title", "") or payload.get("final_url", "")
        return f"Fetched {title}"
    if "items" in payload:
        items = payload.get("items", [])
        return f"Found {len(items)} item(s) in {payload.get('path', '.')}"
    if "grep_matches" in payload:
        return (
            f"Found {payload.get('count', 0)} match(es) in "
            f"{payload.get('matched_files', 0)} file(s)"
        )
    if "content" in payload:
        return f"Read {payload.get('lines', 0)} line(s) from {payload.get('path', '')}"
    if "edits_applied" in payload:
        if payload.get("created"):
            return f"Created {payload.get('path', '')} ({payload.get('bytes_written', 0)} bytes)"
        return (
            f"Patched {payload.get('path', '')} "
            f"with {payload.get('edits_applied', 0)} edit(s)"
        )
    if "returncode" in payload:
        return f"Exit {payload.get('returncode', 1)} in {payload.get('cwd', '.')}"
    if "sheet_names" in payload:
        sheet = payload.get("sheet", "")
        return (
            f"Loaded {payload.get('path', '')}"
            f"{f' [{sheet}]' if sheet else ''}: "
            f"{payload.get('row_count', 0)} row(s), {payload.get('column_count', 0)} column(s)"
        )
    if "matches" in payload:
        return f"Found {payload.get('count', 0)} chat match(es) for {payload.get('query', '')}"
    if "results" in payload:
        return f"Found {payload.get('count', 0)} web result(s) for {payload.get('query', '')}"
    return "Completed"


async def execute_tool_call(
    conversation_id: str,
    call: Dict[str, Any],
    features: Optional[FeatureFlags] = None,
) -> Dict[str, Any]:
    """Validate and execute a supported tool call."""
    call_id = call["id"]
    name = call["name"]
    arguments = call.get("arguments", {})

    try:
        if name == "workspace.list_files":
            result = workspace_list_files_result(conversation_id, str(arguments.get("path", "")))
        elif name == "workspace.grep":
            result = workspace_grep_result(
                conversation_id,
                str(arguments.get("query", "")),
                str(arguments.get("path", ".")),
                str(arguments.get("glob", "*")),
                int(arguments.get("limit", 20)),
                bool(arguments.get("case_sensitive", False)),
            )
        elif name == "workspace.read_file":
            path = str(arguments.get("path", "")).strip()
            if not path:
                raise ValueError("path is required")
            result = workspace_read_file_result(conversation_id, path)
        elif name == "workspace.patch_file":
            path = str(arguments.get("path", "")).strip()
            if not path:
                raise ValueError("path is required")
            result = workspace_patch_file_result(
                conversation_id,
                path,
                arguments.get("edits"),
                arguments.get("create", False),
                arguments.get("new_content"),
            )
        elif name == "workspace.run_command":
            result = await workspace_run_command_result(
                conversation_id,
                arguments.get("command"),
                str(arguments.get("cwd", ".")),
                features,
            )
        elif name == "spreadsheet.describe":
            path = str(arguments.get("path", "")).strip()
            if not path:
                raise ValueError("path is required")
            sheet = arguments.get("sheet")
            result = spreadsheet_describe_result(
                conversation_id,
                path,
                str(sheet).strip() if isinstance(sheet, str) and sheet.strip() else None,
            )
        elif name == "conversation.search_history":
            result = conversation_search_history_result(
                conversation_id,
                str(arguments.get("query", "")),
                int(arguments.get("limit", 5)),
            )
        elif name == "web.search":
            result = await web_search_result(
                str(arguments.get("query", "")),
                int(arguments.get("limit", 5)),
                arguments.get("domains"),
            )
        elif name == "web.fetch_page":
            result = await web_fetch_page_result(str(arguments.get("url", "")))
        else:
            raise ValueError(f"Unsupported tool: {name}")
    except PatchApplicationError as exc:
        return {"id": call_id, "ok": False, "error": str(exc), "details": exc.details}
    except HTTPException as exc:
        return {"id": call_id, "ok": False, "error": exc.detail}
    except Exception as exc:
        return {"id": call_id, "ok": False, "error": str(exc)}

    return {"id": call_id, "ok": True, "result": result}


async def run_tool_loop(
    websocket: WebSocket,
    conversation_id: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: Optional[FeatureFlags] = None,
    allowed_tools: Optional[List[str]] = None,
    status_prefix: str = "",
    max_steps: Optional[int] = None,
    activity_phase: str = "tool",
    activity_step_label: Optional[str] = None,
) -> ToolLoopOutcome:
    """Run a small tool protocol until the model returns a final answer."""
    tool_prompt = (
        build_filtered_tool_system_prompt(system_prompt, allowed_tools)
        if allowed_tools else build_tool_system_prompt(system_prompt)
    )
    messages = [{"role": "system", "content": tool_prompt}]
    tool_results: List[Dict[str, Any]] = []
    auto_verify_runs = 0
    auto_verify_signatures: set[str] = set()
    fetched_page_cache: Dict[str, Dict[str, Any]] = {}
    fetched_unique_pages: set[str] = set()
    step_limit = max_steps or TOOL_LOOP_MAX_STEPS
    if is_fast_profile_active():
        step_limit = min(step_limit, 4)
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    for _ in range(step_limit):
        raw = await vllm_chat_complete(messages, max_tokens=max_tokens, temperature=0.2)
        cleaned = strip_stream_special_tokens(raw).strip()

        try:
            call = parse_tool_call(cleaned)
        except Exception as exc:
            logger.warning("Tool call parse error, treating as final answer: %s", exc)
            return await finalize_tool_loop_answer(cleaned, tool_results, max_tokens)

        if not call:
            return await finalize_tool_loop_answer(cleaned, tool_results, max_tokens)

        if allowed_tools and call["name"] not in allowed_tools:
            return ToolLoopOutcome(
                final_text=(
                    f"The requested tool {call['name']} is not allowed in this phase. "
                    "Continue using only the permitted tools or provide the phase summary."
                ),
                tool_results=tool_results,
            )

        if call["name"] == "workspace.run_command":
            arguments = call.get("arguments", {})
            command = arguments.get("command")
            command_list = command if isinstance(command, list) else []
            cwd_value = str(arguments.get("cwd", "."))
            cwd_path = resolve_workspace_relative_path(conversation_id, cwd_value or ".")
            command_key = command_permission_key(conversation_id, command_list, cwd_path) if command_list else ""
            if features is not None and command_list and not is_command_allowlisted(conversation_id, command_list, cwd_path, features):
                approved = await wait_for_command_approval(
                    websocket,
                    conversation_id,
                    command_list,
                    command_key or "command",
                    cwd=cwd_value,
                    step_label=activity_step_label,
                )
                if not approved:
                    denied_result = {
                        "id": call.get("id", ""),
                        "ok": False,
                        "error": f"Command '{command_key or command_list[0]}' was not approved for this chat",
                    }
                    await websocket.send_json({
                        "type": "tool_result",
                        "name": call["name"],
                        "ok": False,
                        "content": f"{status_prefix}{tool_result_preview(denied_result)}",
                        "arguments": call.get("arguments", {}),
                        "payload": {"error": denied_result["error"]},
                    })
                    await send_activity_event(
                        websocket,
                        "error",
                        "Tool Error",
                        f"{status_prefix}{tool_result_preview(denied_result)}",
                        step_label=activity_step_label,
                    )
                    tool_results.append({
                        "call": call,
                        "result": denied_result,
                    })
                    messages.append({"role": "assistant", "content": cleaned})
                    messages.append({
                        "role": "user",
                        "content": "<tool_result>\n" + json.dumps(denied_result, ensure_ascii=False) + "\n</tool_result>",
                    })
                    continue

        await websocket.send_json({
            "type": "tool_start",
            "name": call["name"],
            "content": f"{status_prefix}{tool_status_summary(call['name'], call.get('arguments', {}))}",
            "arguments": call.get("arguments", {}),
        })
        await send_activity_event(
            websocket,
            activity_phase,
            "Tool",
            f"{status_prefix}{tool_status_summary(call['name'], call.get('arguments', {}))}",
            step_label=activity_step_label,
        )

        if call["name"] == "web.fetch_page":
            requested_url = canonicalize_http_url(str(call.get("arguments", {}).get("url", "")).strip())
            if requested_url and requested_url in fetched_page_cache:
                cached_result = dict(fetched_page_cache[requested_url])
                cached_result["id"] = call["id"]
                cached_result["result"] = dict(cached_result.get("result", {}))
                if isinstance(cached_result["result"], dict):
                    cached_result["result"]["cached"] = True
                result = cached_result
            elif requested_url and len(fetched_unique_pages) >= WEB_FETCH_PAGE_MAX_PER_TURN:
                result = {
                    "id": call["id"],
                    "ok": False,
                    "error": (
                        f"Reached the per-turn web page fetch limit ({WEB_FETCH_PAGE_MAX_PER_TURN}). "
                        "Use the pages already fetched to answer."
                    ),
                }
            else:
                result = await execute_tool_call(conversation_id, call, features)
                if result.get("ok"):
                    payload = result.get("result", {})
                    if isinstance(payload, dict):
                        cached_payload = {
                            "id": call["id"],
                            "ok": True,
                            "result": dict(payload),
                        }
                        for cache_key in {
                            requested_url,
                            canonicalize_http_url(str(payload.get("url", "")).strip()),
                            canonicalize_http_url(str(payload.get("final_url", "")).strip()),
                        }:
                            if cache_key:
                                fetched_page_cache[cache_key] = dict(cached_payload)
                        final_key = canonicalize_http_url(str(payload.get("final_url") or payload.get("url") or requested_url).strip())
                        if final_key:
                            fetched_unique_pages.add(final_key)
        else:
            result = await execute_tool_call(conversation_id, call, features)
        await websocket.send_json({
            "type": "tool_result",
            "name": call["name"],
            "ok": result.get("ok", False),
            "content": f"{status_prefix}{tool_result_preview(result)}",
            "arguments": call.get("arguments", {}),
            "payload": result.get("result", {}) if result.get("ok") else {
                "error": result.get("error", "Tool failed"),
            },
        })
        await send_activity_event(
            websocket,
            activity_phase if result.get("ok") else "error",
            "Tool Result" if result.get("ok") else "Tool Error",
            f"{status_prefix}{tool_result_preview(result)}",
            step_label=activity_step_label,
        )
        tool_results.append({
            "call": call,
            "result": result,
        })

        messages.append({"role": "assistant", "content": cleaned})
        messages.append({
            "role": "user",
            "content": "<tool_result>\n" + json.dumps(result, ensure_ascii=False) + "\n</tool_result>",
        })

        if (
            AUTO_VERIFY_AFTER_PATCH
            and auto_verify_runs < AUTO_VERIFY_MAX_RUNS
            and result.get("ok")
            and call["name"] == "workspace.patch_file"
            and (not allowed_tools or "workspace.run_command" in allowed_tools)
        ):
            patched_path = str(result.get("result", {}).get("path", "")).strip()
            auto_verify = infer_auto_verify_command(conversation_id, patched_path) if patched_path else None
            if auto_verify:
                signature = json.dumps(
                    [auto_verify.get("cwd", "."), auto_verify.get("command", [])],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if signature not in auto_verify_signatures:
                    auto_verify_signatures.add(signature)
                    auto_verify_runs += 1
                    auto_call = {
                        "id": f"{call['id']}_auto_verify",
                        "name": "workspace.run_command",
                        "arguments": {
                            "command": auto_verify["command"],
                            "cwd": auto_verify.get("cwd", "."),
                        },
                    }
                    await websocket.send_json({
                        "type": "tool_start",
                        "name": auto_call["name"],
                        "content": f"{status_prefix}{auto_verify.get('label', tool_status_summary(auto_call['name'], auto_call['arguments']))}",
                        "arguments": auto_call["arguments"],
                    })
                    await send_activity_event(
                        websocket,
                        "verify",
                        "Auto Verify",
                        f"{status_prefix}{auto_verify.get('label', tool_status_summary(auto_call['name'], auto_call['arguments']))}",
                        step_label=activity_step_label,
                    )
                    auto_result = await execute_tool_call(conversation_id, auto_call, features)
                    await websocket.send_json({
                        "type": "tool_result",
                        "name": auto_call["name"],
                        "ok": auto_result.get("ok", False),
                        "content": f"{status_prefix}{tool_result_preview(auto_result)}",
                        "arguments": auto_call["arguments"],
                        "payload": auto_result.get("result", {}) if auto_result.get("ok") else {
                            "error": auto_result.get("error", "Tool failed"),
                        },
                    })
                    await send_activity_event(
                        websocket,
                        "verify" if auto_result.get("ok") else "error",
                        "Auto Verify Result" if auto_result.get("ok") else "Auto Verify Error",
                        f"{status_prefix}{tool_result_preview(auto_result)}",
                        step_label=activity_step_label,
                    )
                    tool_results.append({
                        "call": auto_call,
                        "result": auto_result,
                        "auto_generated": True,
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "The system ran an automatic verification command after the patch.\n"
                            "<tool_result>\n"
                            + json.dumps(auto_result, ensure_ascii=False)
                            + "\n</tool_result>"
                        ),
                    })

    return ToolLoopOutcome(
        final_text=(
            "I hit the tool-use limit before I could finish. "
            "Please ask me to continue, or narrow the task to one specific file or command."
        ),
        tool_results=tool_results,
    )


def build_recent_context(history: List[Dict[str, str]], limit: int = 6) -> str:
    """Format recent history into a compact context block."""
    if not history:
        return ""
    recent = history[-limit:]
    return "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)


def classify_workspace_intent(message: str) -> str:
    """Classify whether a workspace request is read-only, focused edit, or broad exploration."""
    text = (message or "").strip().lower()
    if not text:
        return "none"

    words = set(re.findall(r"[a-z0-9_+-]+", text))
    read_verbs = {"inspect", "read", "open", "show", "list", "search", "find", "grep", "explain", "review"}
    write_verbs = {
        "edit", "change", "update", "patch", "refactor", "create", "write", "add",
        "delete", "remove", "rename", "run", "execute", "test", "build", "compile",
        "debug", "fix", "implement", "generate", "scaffold",
    }

    path_refs = extract_workspace_path_references(message)
    repo_scope_terms = {"repo", "repository", "codebase", "workspace", "project", "app", "service", "api"}
    broad_scope = bool(words & repo_scope_terms) or bool(words & WORKSPACE_TEMPLATE_TERMS) or len(path_refs) > 1

    if words & WORKSPACE_TEMPLATE_TERMS and (words & repo_scope_terms or "python" in words):
        return "broad_write"

    if words & write_verbs:
        return "broad_write" if broad_scope else "focused_write"
    if words & read_verbs or path_refs:
        return "broad_read" if broad_scope else "focused_read"
    return "none"


def tool_loop_step_limit_for_request(message: str, allowed_tools: List[str]) -> int:
    """Choose a small tool-loop budget based on request scope."""
    intent = classify_workspace_intent(message)
    steps = TOOL_LOOP_MAX_STEPS
    if intent in {"focused_read", "focused_write"}:
        steps = min(steps, 3)
    elif intent in {"broad_read", "broad_write"}:
        steps = min(steps, 8)
    else:
        steps = min(steps, 2 if allowed_tools else TOOL_LOOP_MAX_STEPS)
    if is_fast_profile_active():
        steps = min(steps, 6 if intent == "broad_write" else 4)
    return max(1, steps)


def tool_loop_token_budget(max_tokens: int, message: str, allowed_tools: List[str]) -> int:
    """Use a smaller planning budget for tool-oriented normal-mode turns."""
    intent = classify_workspace_intent(message)
    if not allowed_tools:
        return max_tokens
    if is_fast_profile_active():
        if intent in {"focused_read", "focused_write"}:
            return min(max_tokens, 1024)
        if intent == "broad_write":
            return min(max_tokens, 2048)
        return min(max_tokens, 1536)
    if intent in {"focused_read", "focused_write"}:
        return min(max_tokens, 1536)
    if intent == "broad_write":
        return min(max_tokens, 3072)
    return min(max_tokens, 2048)


def select_enabled_tools(conversation_id: str, message: str, features: FeatureFlags) -> List[str]:
    """Return the small set of tools worth exposing for this request."""
    allowed: List[str] = []
    intent = classify_workspace_intent(message)
    if should_use_workspace_tools(conversation_id, message, features):
        if intent == "focused_read":
            path_refs = extract_workspace_path_references(message)
            if path_refs and any(pathlib.Path(path).suffix.lower() in SPREADSHEET_SUPPORTED_EXTENSIONS for path in path_refs):
                allowed.append("spreadsheet.describe")
            allowed.extend(allowed_workspace_tools(features, include_write=False))
        elif intent == "focused_write":
            allowed.extend(allowed_workspace_tools(features, include_write=True))
        else:
            allowed.extend(allowed_workspace_tools(features, include_write=True))
    if should_offer_local_rag(message, features):
        allowed.append("conversation.search_history")
    if should_offer_web_search(message, features):
        allowed.append("web.search")
        allowed.append("web.fetch_page")
    deduped: List[str] = []
    for tool_name in allowed:
        if tool_name not in deduped:
            deduped.append(tool_name)
    return deduped


def workspace_has_content(conversation_id: str) -> bool:
    """Return whether the conversation workspace already contains any files."""
    workspace = get_workspace_path(conversation_id, create=False)
    if not workspace.exists():
        return False
    try:
        next(workspace.iterdir())
        return True
    except StopIteration:
        return False


def should_use_workspace_tools(conversation_id: str, message: str, features: FeatureFlags) -> bool:
    """Heuristic: only enter workspace mode when the user is clearly asking for local file work."""
    if not features.agent_tools:
        return False

    text = (message or "").strip().lower()
    if not text:
        return False

    if extract_artifact_references(message):
        return True

    if any(token in text for token in ("workspace", "repo", "repository", "codebase", "project folder")):
        return True

    if re.search(r"(`[^`]+`|[\w./-]+\.[A-Za-z0-9]+)", text):
        return True

    words = set(re.findall(r"[a-z0-9_+-]+", text))
    if words & WORKSPACE_TEMPLATE_TERMS and (
        words & WORKSPACE_SIGNAL_NOUNS
        or {"python", "fastapi", "flask", "django", "saas", "api", "backend", "frontend"} & words
    ):
        return True

    if words & WORKSPACE_SIGNAL_VERBS and words & WORKSPACE_SIGNAL_NOUNS:
        return True

    if workspace_has_content(conversation_id):
        if any(phrase in text for phrase in ("this code", "this app", "this project", "the code here", "the app here")):
            return True
        if any(text.startswith(prefix) for prefix in ("fix ", "debug ", "run ", "test ", "build ", "refactor ")):
            return True

    return False


LOCAL_RAG_HINTS = {
    "remember", "earlier", "before", "previous", "past", "history", "search", "find",
    "recall", "conversation", "discussed", "docs", "documentation", "notes", "decision",
}
WEB_SEARCH_HINTS = {
    "latest", "today", "current", "recent", "news", "release", "released", "version",
    "price", "weather", "score", "web", "website", "online", "search", "google", "browse",
}


def should_offer_local_rag(message: str, features: FeatureFlags) -> bool:
    """Use lightweight retrieval only for prompts that imply recall/search."""
    if not features.local_rag:
        return False
    text = (message or "").strip().lower()
    if len(text) < 3:
        return False
    words = set(re.findall(r"[a-z0-9_+-]+", text))
    return bool(words & LOCAL_RAG_HINTS)


def should_offer_web_search(message: str, features: FeatureFlags) -> bool:
    """Only offer web search when explicitly useful."""
    if not features.web_search:
        return False
    text = (message or "").strip().lower()
    if len(text) < 3:
        return False
    if "http://" in text or "https://" in text:
        return True
    if re.search(r"\b[a-z0-9.-]+\.[a-z]{2,}\b", text):
        return True
    words = set(re.findall(r"[a-z0-9_+-]+", text))
    return bool(words & WEB_SEARCH_HINTS)


async def deep_inspect_workspace(session: DeepSession) -> str:
    """Use read-only tools to gather workspace facts for deep mode."""
    if not session.workspace_enabled:
        session.workspace_facts = ""
        return session.workspace_facts
    session.workspace_snapshot = capture_workspace_snapshot(session.conversation_id)
    await send_activity_event(
        session.websocket,
        "inspect",
        "Inspect",
        "Inspecting the workspace to ground planning in observed files and artifacts.",
    )
    inspect_history = session.history_messages() + [{
        "role": "user",
        "content": (
            f"Conversation context:\n{session.context or '(none)'}\n\n"
            f"Deterministic workspace snapshot:\n{format_workspace_snapshot(session.workspace_snapshot)}\n\n"
            f"User request:\n{session.message}\n\n"
            "Inspect the workspace and gather the most relevant facts before planning. "
            "Use tools when needed, then return a concise fact summary."
        ),
    }]
    outcome = await run_tool_loop(
        session.websocket,
        session.conversation_id,
        inspect_history,
        f"{session.system_prompt}\n\n{DEEP_INSPECT_SYSTEM_PROMPT}",
        min(session.max_tokens, 1536),
        features=session.features,
        allowed_tools=allowed_workspace_tools(session.features, include_write=False),
        status_prefix="Inspect: ",
        max_steps=4,
        activity_phase="inspect",
    )
    facts = outcome.final_text.strip()
    snapshot_facts = format_workspace_snapshot(session.workspace_snapshot)
    session.workspace_facts = (
        f"{facts}\n\nGrounded workspace snapshot:\n{snapshot_facts}"
        if facts
        else f"Grounded workspace snapshot:\n{snapshot_facts}"
    )
    return session.workspace_facts


async def deep_confirm_understanding(session: DeepSession) -> str:
    """Send a brief confirmation of the assistant's current understanding."""
    confirmation = render_deep_confirmation(session.message, session.workspace_enabled)
    if confirmation:
        await send_reasoning_note(session.websocket, confirmation)
    return confirmation


async def deep_build_workspace(session: DeepSession) -> str:
    """Execute the builder checklist one step at a time with workspace offloading."""
    if not session.workspace_enabled:
        session.build_summary = ""
        session.build_step_summaries = []
        session.changed_files = []
        return session.build_summary
    if not session.features.workspace_write:
        session.build_summary = "Build skipped because workspace write permission was not granted for this turn."
        session.build_step_summaries = [session.build_summary]
        session.changed_files = []
        await send_activity_event(
            session.websocket,
            "blocked",
            "Blocked",
            "Write permission required before I can create or edit workspace files.",
        )
        await persist_task_state(session)
        return session.build_summary
    await send_activity_event(
        session.websocket,
        "execute",
        "Execute",
        "Executing the approved plan from the task board.",
    )
    agent_a = session.plan["agent_a"]
    changed_files: List[str] = []
    if session.changed_files:
        changed_files.extend(path for path in session.changed_files if path not in changed_files)
    if not session.resumed:
        session.build_step_summaries = []
        session.step_reports = []
    steps = session.plan.get("builder_steps", [])
    start_index = min(len(session.build_step_summaries), len(steps))
    await send_build_steps(
        session.websocket,
        steps,
        completed_count=start_index,
        active_index=(start_index if start_index < len(steps) else None),
    )

    if start_index >= len(steps):
        await send_activity_event(
            session.websocket,
            "execute",
            "Execute",
            "Build checklist already completed.",
        )
        await send_build_steps(session.websocket, steps, completed_count=len(steps), active_index=None)
        await persist_task_board(session, active_build_step=None)
        await persist_task_state(session)
        session.build_summary = "\n".join(session.build_step_summaries) or "(build already completed)"
        session.changed_files = changed_files
        return session.build_summary

    for idx in range(start_index, len(steps)):
        step = steps[idx]
        await send_build_steps(session.websocket, steps, completed_count=idx, active_index=idx)
        await persist_task_board(session, active_build_step=idx)
        step_label = f"Step {idx + 1}: {step}"
        await send_activity_event(
            session.websocket,
            "execute",
            "Execute",
            f"Working on build step {idx + 1} of {len(steps)}.",
            step_label=step_label,
        )
        build_history = session.history_messages() + [{
            "role": "user",
            "content": (
                f"Conversation context:\n{session.context or '(none)'}\n\n"
                f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
                f"User request:\n{session.message}\n\n"
                f"Planned deliverable:\n{session.plan.get('deliverable', '(none)')}\n\n"
                f"Task board artifact:\n[[artifact:{session.task_board_path}]]\n\n"
                "Completed build step notes:\n"
                + ("\n".join(f"{i + 1}. {summary}" for i, summary in enumerate(session.build_step_summaries)) or "(none)")
                + "\n\n"
                f"Current build step ({idx + 1}/{len(steps)}):\n{step}\n\n"
                f"Build role: {agent_a.get('role', 'builder')}\n"
                f"Build task:\n{agent_a.get('prompt', '')}\n\n"
                "Focus on the current step only. Use the task board as external memory instead of trying to carry the whole plan in-context. "
                "Use tools to inspect, patch, and validate as needed for this step. "
                "When done, return a concise step result covering what changed, any artifact paths, and remaining caveats for the next step."
            ),
        }]
        outcome = await run_tool_loop(
            session.websocket,
            session.conversation_id,
            build_history,
            f"{session.system_prompt}\n\n{DEEP_BUILD_SYSTEM_PROMPT}",
            min(session.max_tokens, 1536),
            features=session.features,
            allowed_tools=allowed_workspace_tools(session.features, include_write=True),
            status_prefix=f"Build {idx + 1}/{len(steps)}: ",
            max_steps=4,
            activity_phase="execute",
            activity_step_label=step_label,
        )
        successful_tools = sum(1 for entry in outcome.tool_results if entry.get("result", {}).get("ok"))
        successful_commands = sum(
            1
            for entry in outcome.tool_results
            if entry.get("call", {}).get("name") == "workspace.run_command"
            and entry.get("result", {}).get("ok")
        )
        touched_paths: List[str] = []
        for entry in outcome.tool_results:
            call = entry.get("call", {})
            result = entry.get("result", {})
            payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
            path = payload.get("path")
            if isinstance(path, str) and path and path not in touched_paths:
                touched_paths.append(path)
        report = {
            "step_index": idx,
            "step": step,
            "tool_calls": len(outcome.tool_results),
            "successful_tools": successful_tools,
            "successful_commands": successful_commands,
            "paths": touched_paths,
            "summary": outcome.final_text.strip(),
        }
        session.step_reports.append(report)
        step_summary = outcome.final_text.strip() or f"Completed build step {idx + 1}."
        if successful_tools == 0:
            step_summary = f"{step_summary} No successful tool actions were recorded for this step."
        session.build_step_summaries.append(f"Step {idx + 1}: {step_summary}")

        for entry in outcome.tool_results:
            call = entry.get("call", {})
            result = entry.get("result", {})
            if call.get("name") != "workspace.patch_file" or not result.get("ok"):
                continue
            payload = result.get("result", {})
            path = payload.get("path")
            if isinstance(path, str) and path not in changed_files:
                changed_files.append(path)

        session.changed_files = changed_files
        session.build_summary = "\n".join(session.build_step_summaries) or "(no build work completed)"
        await send_build_steps(
            session.websocket,
            steps,
            completed_count=idx + 1,
            active_index=(idx + 1 if idx + 1 < len(steps) else None),
        )
        await persist_task_board(session, active_build_step=(idx + 1 if idx + 1 < len(steps) else None))
        await persist_task_state(session)

    session.changed_files = changed_files
    session.build_summary = "\n".join(session.build_step_summaries) or "(no build work completed)"
    await send_build_steps(session.websocket, steps, completed_count=len(steps), active_index=None)
    await persist_task_board(session, active_build_step=None)
    await persist_task_state(session)
    return session.build_summary


async def deep_decompose(session: DeepSession, preview_only: bool = False) -> Dict[str, Any]:
    """Create a sequential execution plan using message context plus observed workspace facts."""
    if session.plan:
        await send_assistant_note(session.websocket, "Reusing the saved execution plan from the existing task board.")
        if not preview_only:
            await send_build_steps(
                session.websocket,
                session.plan.get("builder_steps", []),
                completed_count=len(session.build_step_summaries),
                active_index=len(session.build_step_summaries) if len(session.build_step_summaries) < len(session.plan.get("builder_steps", [])) else None,
            )
            await persist_task_board(session, active_build_step=len(session.build_step_summaries))
            await persist_task_state(session)
        return session.plan
    await send_activity_event(
        session.websocket,
        "plan",
        "Plan",
        "Turning the inspected request into a grounded execution plan.",
    )
    if is_fast_profile_active():
        session.plan = build_heuristic_deep_plan(session)
    else:
        decompose_messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation context:\n{session.context or '(none)'}\n\n"
                    f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
                    f"User's new query:\n{session.message}"
                ),
            },
        ]
        raw = await vllm_chat_complete(decompose_messages, max_tokens=768, temperature=0.1)
        try:
            session.plan = apply_deep_plan_guardrails(
                session,
                parse_json_object(strip_stream_special_tokens(raw)),
            )
        except Exception:
            logger.warning("Deep planner returned invalid JSON; using heuristic plan instead.")
            session.plan = build_heuristic_deep_plan(session)
    session.plan = apply_deep_plan_guardrails(session, session.plan)
    await send_assistant_note(session.websocket, format_deep_plan_note(session.plan))
    if not preview_only:
        await send_build_steps(session.websocket, session.plan.get("builder_steps", []), completed_count=0, active_index=0)
        await persist_task_board(session, active_build_step=0, announce=True)
        await persist_task_state(session)
    return session.plan


async def deep_answer_directly(session: DeepSession) -> str:
    """Use deep mode as a thoughtful answer path without forcing execution planning."""
    await send_activity_event(
        session.websocket,
        "respond",
        "Respond",
        "Answering directly from the inspected context without a workspace execution plan.",
    )
    allowed_tools = select_enabled_tools(session.conversation_id, session.message, session.features)
    allowed_tools = [
        tool_name for tool_name in allowed_tools
        if tool_name not in {"workspace.patch_file", "workspace.run_command"}
    ]
    direct_history = session.history_messages() + [{
        "role": "user",
        "content": (
            f"Conversation context:\n{session.context or '(none)'}\n\n"
            f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
            f"User request:\n{session.message}\n\n"
            "Answer directly unless execution is clearly required. "
            "Use read-only tools if they materially improve the answer."
        ),
    }]
    if allowed_tools:
        outcome = await run_tool_loop(
            session.websocket,
            session.conversation_id,
            direct_history,
            f"{session.system_prompt}\n\n{DEEP_DIRECT_SYSTEM_PROMPT}",
            min(session.max_tokens, 2048),
            features=session.features,
            allowed_tools=allowed_tools,
            status_prefix="Deep: ",
            max_steps=4,
            activity_phase="respond",
        )
        return outcome.final_text

    messages = [{"role": "system", "content": f"{session.system_prompt}\n\n{DEEP_DIRECT_SYSTEM_PROMPT}"}]
    messages.extend(direct_history)
    raw = await vllm_chat_complete(messages, max_tokens=min(session.max_tokens, 2048), temperature=0.15)
    return strip_stream_special_tokens(raw)


async def deep_parallel_solve(session: DeepSession) -> Dict[str, str]:
    """Capture build/review state without spending an extra model round-trip."""
    session.agent_outputs = {
        "agent_a_role": session.plan["agent_a"]["role"],
        "agent_b_role": session.plan["agent_b"]["role"],
        "output_a": session.build_summary,
        "output_b": "Verification will validate the built result directly from the workspace and current summaries.",
    }
    await persist_task_state(session)
    return session.agent_outputs


async def deep_verify(session: DeepSession) -> str:
    """Run a read-only verification pass after the parallel solve."""
    if not session.workspace_enabled:
        session.verification_summary = ""
        return session.verification_summary
    await send_activity_event(
        session.websocket,
        "verify",
        "Verify",
        "Checking the built result against the workspace and requested deliverable.",
    )
    verify_history = session.history_messages() + [{
        "role": "user",
        "content": (
            f"User request:\n{session.message}\n\n"
            f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
            f"Task board artifact:\n[[artifact:{session.task_board_path}]]\n\n"
            f"Planned deliverable:\n{session.plan.get('deliverable', '(none)')}\n\n"
            "Verifier checks:\n"
            + "\n".join(f"{idx}. {step}" for idx, step in enumerate(session.plan.get("verifier_checks", []), start=1))
            + "\n\n"
            f"Build summary:\n{session.build_summary or '(none)'}\n\n"
            f"Files changed:\n{', '.join(session.changed_files) if session.changed_files else '(none)'}\n\n"
            f"Proposed solution A ({session.agent_outputs.get('agent_a_role', 'builder')}):\n"
            f"{session.agent_outputs.get('output_a', '')}\n\n"
            f"Proposed solution B ({session.agent_outputs.get('agent_b_role', 'verifier')}):\n"
            f"{session.agent_outputs.get('output_b', '')}\n\n"
            "Use read-only tools and commands to verify likely assumptions. "
            "Return a concise verification summary."
        ),
    }]
    outcome = await run_tool_loop(
        session.websocket,
        session.conversation_id,
        verify_history,
        f"{session.system_prompt}\n\n{DEEP_VERIFY_SYSTEM_PROMPT}",
        min(session.max_tokens, 1536),
        features=session.features,
        allowed_tools=allowed_workspace_tools(session.features, include_write=False),
        status_prefix="Verify: ",
        max_steps=4,
        activity_phase="verify",
    )
    session.verification_summary = outcome.final_text
    session.scope_audit = build_scope_audit(session)
    session.scope_audit_summary = str(session.scope_audit.get("summary", "")).strip()
    await send_scope_audit_event(session.websocket, session.scope_audit)
    await persist_task_board(session, active_build_step=None)
    await persist_task_state(session)
    return session.verification_summary


async def deep_review(session: DeepSession) -> str:
    """Inspect built artifacts and synthesize the final draft from workspace state."""
    await send_activity_event(
        session.websocket,
        "synthesize",
        "Synthesize",
        "Preparing the final answer from verified artifacts, notes, and audit results.",
    )
    artifact_refs = [f"[[artifact:{session.task_board_path}]]"]
    artifact_refs.extend(f"[[artifact:{path}]]" for path in session.changed_files[:8])
    synth_history = session.history_messages() + [{
        "role": "user",
        "content": (
            f"User's question:\n{session.message}\n\n"
            f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
            f"Planned deliverable:\n{session.plan.get('deliverable', '(none)')}\n\n"
            f"Artifacts to inspect:\n" + "\n".join(artifact_refs) + "\n\n"
            f"Build summary:\n{session.build_summary or '(none)'}\n\n"
            f"Verification summary:\n{session.verification_summary or '(none)'}\n\n"
            f"Scope audit:\n{session.scope_audit_summary or '(none)'}\n\n"
            f"Implementation pass ({session.agent_outputs.get('agent_a_role', 'builder')}):\n{session.agent_outputs.get('output_a', '')}\n\n"
            f"Review pass ({session.agent_outputs.get('agent_b_role', 'verifier')}):\n{session.agent_outputs.get('output_b', '')}\n\n"
            "Read the task board and the most relevant built artifacts before answering. "
            "Base the final answer on what the workspace actually contains, what verification established, and which scope gaps remain."
        ),
    }]
    outcome = await run_tool_loop(
        session.websocket,
        session.conversation_id,
        synth_history,
        f"{session.system_prompt}\n\n{DEEP_SYNTHESIZE_SYSTEM_PROMPT}",
        min(session.max_tokens, 2048),
        features=session.features,
        allowed_tools=allowed_workspace_tools(session.features, include_write=False),
        status_prefix="Synthesize: ",
        max_steps=6,
        activity_phase="synthesize",
    )
    session.draft_response = outcome.final_text
    await persist_task_state(session)
    return session.draft_response

# ==================== WebSocket Handlers ====================

async def orchestrated_chat(
    websocket: WebSocket,
    conversation_id: str,
    message: str,
    history: list,
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
) -> str:
    """Deep-mode pipeline using explicit shared session state."""
    await send_activity_event(
        websocket,
        "analyze",
        "Analyze",
        "Evaluating the prompt to choose the right execution path.",
    )

    session = DeepSession(
        websocket=websocket,
        conversation_id=conversation_id,
        message=message,
        history=history,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        features=features,
        context=build_recent_context(history),
        workspace_enabled=should_use_workspace_tools(conversation_id, message, features),
    )

    if session.workspace_enabled:
        logger.info("Deep mode using workspace flow for conv %s", conversation_id)
    else:
        logger.info("Deep mode using text-only flow for conv %s", conversation_id)
    await send_activity_event(
        websocket,
        "evaluate",
        "Evaluate",
        (
            "Workspace-assisted deep mode selected."
            if session.workspace_enabled
            else "Text-first deep mode selected."
        ),
    )

    try:
        await deep_confirm_understanding(session)
    except Exception as e:
        logger.warning("Deep mode confirmation failed, continuing: %s", e)

    try:
        await maybe_resume_task_state(session)
    except Exception as e:
        logger.warning("Deep mode resume failed, continuing fresh: %s", e)

    await deep_inspect_workspace(session)

    if should_preview_deep_plan(session):
        await deep_decompose(session, preview_only=True)
        await send_plan_ready(websocket, session.plan, format_deep_execution_prompt(session.plan))
        return render_deep_plan_preview(session.plan)

    if is_explicit_plan_execution_request(session.message):
        await deep_decompose(session)
        logger.info("Deep mode execution strategy: %s", session.plan.get("strategy", "parallel subtasks"))
        await deep_build_workspace(session)
        await deep_parallel_solve(session)
        await deep_verify(session)
        draft_response = await deep_review(session)
    else:
        draft_response = await deep_answer_directly(session)

    if not DEEP_CRITIQUE_ENABLED or is_fast_profile_active():
        return draft_response

    await send_activity_event(
        websocket,
        "verify",
        "Quality Check",
        "Reviewing the drafted answer before final delivery.",
    )
    critique = await critique_response(message, draft_response)
    if critique["pass"]:
        return draft_response

    issues = critique["issues"] or "Tighten correctness, completeness, and structure."
    logger.info("Deep mode critique requested refinement: %s", issues)
    await send_activity_event(
        websocket,
        "synthesize",
        "Refine",
        "Refining the draft after quality review.",
    )

    refine_messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User's question:\n{message}\n\n"
                f"Current draft:\n{draft_response}\n\n"
                f"Issues to fix:\n{issues}"
            ),
        },
    ]
    refined = await vllm_chat_complete(refine_messages, max_tokens=max_tokens, temperature=0.15)
    refined_response = strip_stream_special_tokens(refined)

    follow_up = await critique_response(message, refined_response)
    if follow_up["pass"]:
        return refined_response

    logger.info("Deep mode refined draft still failed critique, returning refined version: %s", follow_up["issues"])
    return refined_response


async def process_chat_turn(websocket: WebSocket, data: Dict[str, Any]) -> None:
    """Process a single chat turn so the websocket loop can also handle stop requests."""
    message = data.get('message', '').strip()
    conv_id = data.get('conversation_id')
    attachments_raw = data.get('attachments', [])
    attachments = [
        str(item).strip()
        for item in attachments_raw[:MAX_ATTACHMENTS_PER_MESSAGE]
        if isinstance(item, str) and str(item).strip()
    ]

    if not message:
        await websocket.send_json({'type': 'error', 'content': 'Empty message'})
        return

    custom_system_prompt = data.get('system_prompt')
    mode = data.get('mode', 'normal')
    features = parse_feature_flags(data.get('features'))

    try:
        attachment_context = format_attachment_context(attachments)
        effective_message = (
            f"{message}\n\n{attachment_context}" if attachment_context else message
        )

        save_message(conv_id, 'user', effective_message)
        history = get_conversation_history(conv_id, current_query=effective_message)
        system_prompt = build_effective_system_prompt(
            custom_system_prompt or DEFAULT_SYSTEM_PROMPT,
            effective_message,
        )
        agent_params = get_agent_llm_params()
        max_tokens = agent_params["max_tokens"]
        deep_succeeded = False
        workspace_intent = classify_workspace_intent(effective_message)
        enabled_tools = select_enabled_tools(conv_id, effective_message, features) if TOOL_LOOP_ENABLED else []

        await send_activity_event(
            websocket,
            "evaluate",
            "Evaluate",
            (
                f"Mode request: {mode}. Workspace intent: {workspace_intent}. "
                f"Candidate tools: {', '.join(enabled_tools) if enabled_tools else 'none'}."
            ),
        )

        if mode == 'deep':
            try:
                full_response = await orchestrated_chat(
                    websocket, conv_id, effective_message, history, system_prompt, max_tokens, features
                )
                assistant_message_id = save_message(conv_id, 'assistant', full_response)
                schedule_conversation_summary_refresh(conv_id)
                await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
                await websocket.send_json({'type': 'done'})
                deep_succeeded = True
            except Exception as e:
                logger.warning("Deep mode failed, falling back: %s", e)
                await websocket.send_json({
                    'type': 'status',
                    'content': 'Deep mode failed, using normal mode...',
                })

        if not deep_succeeded:
            await send_activity_event(
                websocket,
                "respond",
                "Respond",
                (
                    "Using the shared tool loop to answer."
                    if enabled_tools
                    else "Answering directly without tools."
                ),
            )
            tool_step_limit = tool_loop_step_limit_for_request(effective_message, enabled_tools)
            tool_max_tokens = tool_loop_token_budget(max_tokens, effective_message, enabled_tools)
            logger.info(
                "Processing message for conv %s, max_tokens=%s, tool_loop=%s, tools=%s, tool_max_tokens=%s, tool_steps=%s, workspace_intent=%s",
                conv_id,
                max_tokens,
                TOOL_LOOP_ENABLED,
                enabled_tools,
                tool_max_tokens,
                tool_step_limit,
                classify_workspace_intent(effective_message),
            )

            if enabled_tools:
                tool_outcome = await run_tool_loop(
                    websocket,
                    conv_id,
                    history,
                    system_prompt,
                    tool_max_tokens,
                    features=features,
                    allowed_tools=enabled_tools,
                    max_steps=tool_step_limit,
                    activity_phase="respond",
                )
                full_response = tool_outcome.final_text
                await websocket.send_json({'type': 'start'})
                await send_final_replacement(websocket, full_response)
            else:
                messages = [{'role': 'system', 'content': system_prompt}]
                for msg in history:
                    messages.append({'role': msg['role'], 'content': msg['content']})
                full_response = await stream_chat_response(
                    websocket,
                    messages,
                    max_tokens,
                )
            assistant_message_id = save_message(conv_id, 'assistant', full_response)
            schedule_conversation_summary_refresh(conv_id)
            await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
            await websocket.send_json({'type': 'done'})

    except asyncio.CancelledError:
        logger.info("Canceled active chat turn for conversation %s", conv_id)
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM API error: {e}")
        await websocket.send_json({'type': 'error', 'content': f'vLLM error: {e.response.status_code}. Is the model loaded?'})
    except httpx.ConnectError:
        await websocket.send_json({'type': 'error', 'content': 'Cannot connect to vLLM. Is it running?'})
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        await websocket.send_json({'type': 'error', 'content': f'Error: {str(e)}'})

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    active_task: Optional[asyncio.Task] = None

    try:
        while True:
            data = await websocket.receive_json()
            if data.get('type') == 'stop':
                if active_task and not active_task.done():
                    active_task.cancel()
                    try:
                        await active_task
                    except asyncio.CancelledError:
                        pass
                    await websocket.send_json({'type': 'canceled', 'content': 'Stopped'})
                    await websocket.send_json({'type': 'done'})
                else:
                    await websocket.send_json({'type': 'idle'})
                active_task = None
                continue

            if data.get('type') == 'command_approval':
                conversation_id = str(data.get("conversation_id", "")).strip()
                command_key = normalize_allowed_command_key(str(data.get("command_key", "")))
                approved = bool(data.get("approved"))
                waiter = COMMAND_APPROVAL_WAITERS.get(conversation_id)
                if not waiter:
                    await websocket.send_json({'type': 'error', 'content': 'No command approval is pending for this chat.'})
                    continue
                expected_key = normalize_allowed_command_key(str(waiter.get("command_key", "")))
                future = waiter.get("future")
                if command_key != expected_key:
                    await websocket.send_json({'type': 'error', 'content': 'Command approval response did not match the pending command.'})
                    continue
                if isinstance(future, asyncio.Future) and not future.done():
                    future.set_result(approved)
                continue

            if data.get('type') == 'interrupt':
                if active_task and not active_task.done():
                    active_task.cancel()
                    try:
                        await active_task
                    except asyncio.CancelledError:
                        pass
                    await websocket.send_json({'type': 'canceled', 'content': 'Interrupted and reprompting'})
                interrupt_payload = dict(data)
                interrupt_payload.pop('type', None)
                active_task = asyncio.create_task(process_chat_turn(websocket, interrupt_payload))
                continue

            if active_task and not active_task.done():
                await websocket.send_json({'type': 'error', 'content': 'Please wait for the current response to finish or stop it first.'})
                continue

            active_task = asyncio.create_task(process_chat_turn(websocket, data))

    except WebSocketDisconnect:
        for conversation_id, waiter in list(COMMAND_APPROVAL_WAITERS.items()):
            future = waiter.get("future")
            if isinstance(future, asyncio.Future) and not future.done():
                future.set_result(False)
        if active_task and not active_task.done():
            active_task.cancel()
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


@app.websocket("/ws/terminal/{conversation_id}")
async def terminal_websocket(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    session: Optional[TerminalSession] = None

    try:
        if not INTERACTIVE_TERMINAL_ENABLED:
            await _send_terminal_json(
                websocket,
                {
                    "type": "terminal_unavailable",
                    "content": "Interactive terminal is disabled on this server",
                },
            )
            return

        session = await _start_terminal_session(conversation_id, websocket)
        while True:
            data = await websocket.receive_json()
            msg_type = str(data.get("type", "")).strip().lower()

            if msg_type == "input":
                content = str(data.get("content", ""))
                if content:
                    try:
                        os.write(session.master_fd, content.encode("utf-8", errors="replace"))
                    except OSError:
                        await _send_terminal_json(websocket, {"type": "terminal_status", "content": "Terminal write failed"})
                continue

            if msg_type == "signal":
                if str(data.get("signal", "")).strip().lower() == "interrupt" and session.process.returncode is None:
                    try:
                        os.killpg(session.process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    await _send_terminal_json(websocket, {"type": "terminal_status", "content": "Sent Ctrl+C"})
                continue

            if msg_type == "restart":
                await _close_terminal_session(session)
                session = await _start_terminal_session(conversation_id, websocket)
                await _send_terminal_json(websocket, {"type": "terminal_cleared"})
                continue

            if msg_type == "resize":
                try:
                    _resize_terminal_session(
                        session,
                        int(data.get("cols", 80)),
                        int(data.get("rows", 24)),
                    )
                except Exception:
                    pass
                continue

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("Terminal websocket error: %s", exc)
        await _send_terminal_json(websocket, {"type": "terminal_status", "content": f"Terminal error: {exc}"})
    finally:
        await _close_terminal_session(session)


async def get_model_runtime_summary() -> Dict[str, Any]:
    """Collect model state for health and dashboard views."""
    loaded_model_name = await fetch_loaded_model_name()
    active_profile_key = sync_active_profile_from_model_name(loaded_model_name)
    selected_profile = get_active_model_profile()
    active_profile = (
        MODEL_PROFILES[active_profile_key]
        if active_profile_key in MODEL_PROFILES
        else build_model_profile("custom", loaded_model_name or selected_profile["name"], CUSTOM_MODEL_ARGS, label="Custom")
    )
    model_ok = await vllm_health_check()
    if model_ok:
        mark_model_load_completed(loaded_model_name or selected_profile["name"])
    else:
        target_name = selected_profile["name"]
        loading = MODEL_LOADING_STATUS if isinstance(MODEL_LOADING_STATUS, dict) else {}
        if loading.get("status") not in {"loading"} or loading.get("model_name") != target_name:
            mark_model_load_started(target_name, reason="startup", profile_key=selected_profile["key"])
    available_profiles = [
        {
            "key": profile["key"],
            "label": profile["label"],
            "name": profile["name"],
            "active": profile["key"] == active_profile_key,
            "selected": profile["key"] == selected_profile["key"],
        }
        for profile in MODEL_PROFILES.values()
    ]
    if selected_profile["key"] == "custom":
        available_profiles.append(
            {
                "key": selected_profile["key"],
                "label": selected_profile["label"],
                "name": selected_profile["name"],
                "active": active_profile_key == "custom",
                "selected": True,
            }
        )
    return {
        "model_ok": model_ok,
        "loaded_model_name": loaded_model_name or selected_profile["name"],
        "active_profile": active_profile,
        "selected_profile": selected_profile,
        "available_profiles": available_profiles,
        "loading": get_model_loading_stats(loaded_model_name or selected_profile["name"], model_ok),
    }

@app.get("/health")
async def health():
    runtime = await get_model_runtime_summary()
    voice = get_voice_runtime_summary()
    return {
        "status": "healthy" if runtime["model_ok"] else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model": runtime["loaded_model_name"],
        "model_available": runtime["model_ok"],
        "model_profile": runtime["active_profile"]["key"],
        "loading": runtime["loading"],
        "voice": voice,
        "message": (
            f"Model {runtime['loaded_model_name']} is ready"
            if runtime["model_ok"]
            else f"Model {runtime['selected_profile']['name']} is loading or unavailable"
        ),
    }


@app.post("/api/ui/model-ready")
async def log_ui_model_ready(request: UiModelReadyRequest):
    logger.info(
        "Model %s is ready in web UI (profile=%s, websocket_connected=%s, composer_available=%s)",
        request.model_name,
        request.profile,
        request.websocket_connected,
        request.composer_available,
    )
    return {"status": "ok"}

# ==================== Dashboard ====================

def get_cache_info(model_name: str):
    """Get HuggingFace model cache info"""
    model_dir = model_name.replace("/", "--")
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
        "model_id": model_name,
        "cache_dir": cache_dir,
        "status": "valid" if has_snapshots else "incomplete",
        "size_bytes": total_size,
        "size_display": size_display,
        "file_count": file_count,
        "last_modified": last_modified,
    }


def normalize_model_source(source: str) -> str:
    """Accept a Hugging Face model id or model URL and normalize it to repo_id."""
    raw = (source or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Enter a Hugging Face model id or URL")

    if raw.startswith("https://") or raw.startswith("http://"):
        parsed = urlparse(raw)
        host = (parsed.netloc or "").lower()
        if host not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            raise HTTPException(status_code=400, detail="Only Hugging Face model links are supported")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Could not determine the model id from that Hugging Face URL")
        if parts[0] in {"models", "datasets", "spaces"} and len(parts) >= 3:
            if parts[0] != "models":
                raise HTTPException(status_code=400, detail="Only Hugging Face model repositories are supported")
            model_id = "/".join(parts[1:3])
        else:
            model_id = "/".join(parts[:2])
    else:
        model_id = raw

    model_id = model_id.strip("/")
    if not re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?", model_id):
        raise HTTPException(status_code=400, detail="Model id must look like 'owner/name' or 'name'")
    return model_id


def iter_model_cache_dirs() -> List[pathlib.Path]:
    """Return all Hugging Face model cache directories."""
    hub_root = pathlib.Path(HF_CACHE_PATH) / "hub"
    if not hub_root.exists():
        return []
    return sorted(
        [path for path in hub_root.iterdir() if path.is_dir() and path.name.startswith("models--")],
        key=lambda path: path.name.lower(),
    )


def cache_dir_to_model_id(cache_dir: pathlib.Path) -> str:
    """Convert a Hugging Face cache directory name into a repo id."""
    name = cache_dir.name
    if name.startswith("models--"):
        name = name[len("models--"):]
    return name.replace("--", "/")


def list_cached_models() -> List[Dict[str, Any]]:
    """List model repositories currently present in the Hugging Face cache."""
    runtime_model_names = {profile["name"] for profile in MODEL_PROFILES.values()}
    active_name = get_active_model_name()
    jobs_by_model = {job.get("model_id"): job for job in MODEL_DOWNLOAD_JOBS.values()}
    items: List[Dict[str, Any]] = []
    for cache_dir in iter_model_cache_dirs():
        model_id = cache_dir_to_model_id(cache_dir)
        info = get_cache_info(model_id)
        info.update(
            {
                "download_url": f"https://huggingface.co/{quote_plus(model_id, safe='/')}",
                "managed_by_profile": model_id in runtime_model_names,
                "active_profile": model_id == active_name,
                "download_job": jobs_by_model.get(model_id),
            }
        )
        items.append(info)
    items.sort(key=lambda item: ((not item["active_profile"]), item["model_id"].lower()))
    return items


def current_model_download_jobs() -> List[Dict[str, Any]]:
    """Return current background model download jobs."""
    jobs = [dict(job) for job in MODEL_DOWNLOAD_JOBS.values()]
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return jobs


async def run_model_download_job(job_id: str, model_id: str) -> None:
    """Download a Hugging Face model snapshot into the shared cache."""
    job = MODEL_DOWNLOAD_JOBS[job_id]
    job["status"] = "downloading"
    job["started_at"] = datetime.now().isoformat()
    job["error"] = None
    try:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is not installed in the chat-app image")
        await asyncio.to_thread(
            snapshot_download,
            repo_id=model_id,
            repo_type="model",
            cache_dir=HF_CACHE_PATH,
            local_files_only=False,
            resume_download=True,
        )
        info = get_cache_info(model_id)
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        job["cache"] = info
    except Exception as exc:
        logger.error("Model download failed for %s", model_id, exc_info=True)
        job["status"] = "error"
        job["completed_at"] = datetime.now().isoformat()
        job["error"] = str(exc)


def model_is_managed_by_profile(model_id: str) -> bool:
    """Return whether the model is referenced by a configured runtime profile."""
    return any(profile["name"] == model_id for profile in MODEL_PROFILES.values())


def model_cache_dir(model_id: str) -> pathlib.Path:
    """Return the cache directory for a model id."""
    model_dir = model_id.replace("/", "--")
    return pathlib.Path(HF_CACHE_PATH) / "hub" / f"models--{model_dir}"


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse a stored ISO timestamp safely."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def model_load_history_summary(model_name: str) -> Dict[str, Any]:
    """Summarize prior successful load times for a model."""
    entry = MODEL_LOAD_HISTORY.get(model_name) or {}
    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []
    durations = [float(item) for item in samples if isinstance(item, (int, float))]
    if not durations:
        return {
            "sample_count": 0,
            "average_seconds": None,
            "last_seconds": None,
            "last_completed_at": entry.get("last_completed_at"),
        }
    return {
        "sample_count": len(durations),
        "average_seconds": round(sum(durations) / len(durations), 1),
        "last_seconds": round(durations[-1], 1),
        "last_completed_at": entry.get("last_completed_at"),
    }


def mark_model_load_started(model_name: str, reason: str, profile_key: Optional[str] = None):
    """Record that a model load has started so the UI can estimate progress."""
    global MODEL_LOADING_STATUS
    now = datetime.now().isoformat()
    MODEL_LOADING_STATUS = {
        "status": "loading",
        "phase": "restarting" if reason in {"switch", "activate", "redownload", "restart"} else "loading",
        "reason": reason,
        "model_name": model_name,
        "profile_key": profile_key or ACTIVE_MODEL_PROFILE,
        "started_at": now,
        "updated_at": now,
    }
    persist_model_state()


def mark_model_load_completed(model_name: str):
    """Record a successful model load and update historical timing."""
    global MODEL_LOADING_STATUS, MODEL_LOAD_HISTORY
    now_dt = datetime.now()
    loading = MODEL_LOADING_STATUS if isinstance(MODEL_LOADING_STATUS, dict) else {}
    started_at = _parse_iso_datetime(loading.get("started_at"))
    duration = None
    if started_at:
        duration = max(0.0, (now_dt - started_at).total_seconds())
        entry = MODEL_LOAD_HISTORY.get(model_name) or {}
        samples = entry.get("samples")
        if not isinstance(samples, list):
            samples = []
        samples = [float(item) for item in samples if isinstance(item, (int, float))]
        samples.append(round(duration, 1))
        MODEL_LOAD_HISTORY[model_name] = {
            "samples": samples[-10:],
            "last_completed_at": now_dt.isoformat(),
        }

    MODEL_LOADING_STATUS = {
        "status": "ready",
        "phase": "ready",
        "model_name": model_name,
        "completed_at": now_dt.isoformat(),
        "last_duration_seconds": round(duration, 1) if duration is not None else None,
    }
    persist_model_state()


def get_model_loading_stats(model_name: str, model_ok: bool) -> Dict[str, Any]:
    """Return current loading state with historical ETA estimates."""
    history = model_load_history_summary(model_name)
    loading = MODEL_LOADING_STATUS if isinstance(MODEL_LOADING_STATUS, dict) else {}
    target_name = str(loading.get("model_name") or model_name)
    target_matches = target_name == model_name
    started_at = _parse_iso_datetime(loading.get("started_at")) if target_matches else None
    elapsed_seconds = max(0.0, (datetime.now() - started_at).total_seconds()) if started_at else None
    estimated_total = history.get("average_seconds")
    eta_seconds = None
    progress = None
    if elapsed_seconds is not None and estimated_total:
        eta_seconds = round(max(estimated_total - elapsed_seconds, 0.0), 1)
        progress = min(max(elapsed_seconds / max(estimated_total, 1.0), 0.0), 0.99)

    if model_ok:
        phase = "ready"
        status = "ready"
    else:
        phase = str(loading.get("phase") or "loading")
        status = str(loading.get("status") or "loading")

    return {
        "status": status,
        "phase": phase,
        "model_name": target_name if not model_ok else model_name,
        "elapsed_seconds": round(elapsed_seconds, 1) if elapsed_seconds is not None else None,
        "estimated_total_seconds": estimated_total,
        "eta_seconds": eta_seconds,
        "progress": round(progress, 3) if progress is not None else None,
        "history": history,
        "started_at": loading.get("started_at") if target_matches else None,
        "updated_at": loading.get("updated_at"),
        "reason": loading.get("reason"),
    }

@app.get("/api/dashboard")
async def get_dashboard():
    runtime = await get_model_runtime_summary()

    # Get vLLM container status via Docker socket
    container_status = None
    docker_control_available = DOCKER_CONTROL_ENABLED and os.path.exists("/var/run/docker.sock")
    if docker_control_available:
        try:
            resp = await docker_api_request("GET", "/containers/vllm/json")
            if resp.status_code == 200:
                info = resp.json()
                container_status = {
                    "status": info.get("State", {}).get("Status"),
                    "running": info.get("State", {}).get("Running"),
                    "started_at": info.get("State", {}).get("StartedAt"),
                }
        except Exception:
            pass

    cache_info = get_cache_info(runtime["selected_profile"]["name"])

    return {
        "model_name": runtime["loaded_model_name"],
        "selected_model_name": runtime["selected_profile"]["name"],
        "selected_profile": runtime["selected_profile"],
        "selected_profile_key": runtime["selected_profile"]["key"],
        "active_profile": runtime["active_profile"],
        "active_profile_key": runtime["active_profile"]["key"],
        "available_profiles": runtime["available_profiles"],
        "vllm_host": VLLM_HOST,
        "model_available": runtime["model_ok"],
        "loading": runtime["loading"],
        "container": container_status,
        "cache": cache_info,
        "docker_control_available": docker_control_available,
        "interactive_terminal_enabled": INTERACTIVE_TERMINAL_ENABLED,
        "execute_code_enabled": EXECUTE_CODE_ENABLED,
    }

@app.get("/api/models/library")
async def get_model_library():
    return {
        "cache_root": str(pathlib.Path(HF_CACHE_PATH) / "hub"),
        "models": list_cached_models(),
        "jobs": current_model_download_jobs(),
        "profiles": list(MODEL_PROFILES.values()),
        "active_model_name": get_active_model_name(),
        "loading": get_model_loading_stats(get_active_model_name(), False),
    }

@app.post("/api/models/library/download")
async def download_model_to_library(request: ModelDownloadRequest):
    global MODEL_DOWNLOAD_LOCK

    model_id = normalize_model_source(request.source)
    if MODEL_DOWNLOAD_LOCK is None:
        MODEL_DOWNLOAD_LOCK = asyncio.Lock()

    async with MODEL_DOWNLOAD_LOCK:
        for job in MODEL_DOWNLOAD_JOBS.values():
            if job.get("model_id") == model_id and job.get("status") in {"queued", "downloading"}:
                return {
                    "status": "accepted",
                    "message": f"{model_id} is already downloading.",
                    "job": job,
                }

        existing_cache = get_cache_info(model_id)
        if existing_cache.get("status") == "valid":
            return {
                "status": "success",
                "message": f"{model_id} is already cached.",
                "cache": existing_cache,
            }

        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "model_id": model_id,
            "source": request.source.strip(),
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
        MODEL_DOWNLOAD_JOBS[job_id] = job
        asyncio.create_task(run_model_download_job(job_id, model_id))

    return {
        "status": "accepted",
        "message": f"Started downloading {model_id} into the shared Hugging Face cache.",
        "job": job,
    }

@app.post("/api/models/library/delete")
async def delete_model_from_library(request: ModelDeleteRequest):
    model_id = normalize_model_source(request.model_id)

    active_job = next(
        (
            job for job in MODEL_DOWNLOAD_JOBS.values()
            if job.get("model_id") == model_id and job.get("status") in {"queued", "downloading"}
        ),
        None,
    )
    if active_job:
        raise HTTPException(status_code=409, detail=f"{model_id} is currently downloading")

    if model_is_managed_by_profile(model_id):
        raise HTTPException(
            status_code=409,
            detail=f"{model_id} is configured as one of the runtime model profiles. Change the profile config before deleting it.",
        )

    cache_dir = model_cache_dir(model_id)
    if not cache_dir.exists():
        raise HTTPException(status_code=404, detail=f"No cached files found for {model_id}")

    shutil.rmtree(cache_dir)
    logger.info("Deleted model cache: %s", cache_dir)
    return {
        "status": "success",
        "message": f"Deleted cached files for {model_id}.",
        "model_id": model_id,
    }

@app.post("/api/models/library/activate")
async def activate_model_from_library(request: ModelActivateRequest):
    global ACTIVE_MODEL_LOCK

    ensure_docker_control_enabled()
    model_id = normalize_model_source(request.model_id)
    if get_cache_info(model_id).get("status") == "not_downloaded":
        raise HTTPException(status_code=404, detail=f"{model_id} is not cached yet")

    if ACTIVE_MODEL_LOCK is None:
        ACTIVE_MODEL_LOCK = asyncio.Lock()

    async with ACTIVE_MODEL_LOCK:
        if get_active_model_name() == model_id:
            return {
                "status": "success",
                "message": f"{model_id} is already selected.",
                "model_name": model_id,
                "profile": ACTIVE_MODEL_PROFILE,
            }

        mark_model_load_started(model_id, reason="activate", profile_key="custom")
        await recreate_vllm_container("custom", model_name=model_id)
        persist_active_model_selection("custom", model_id)
        return {
            "status": "success",
            "message": f"Switching to {model_id}. vLLM is reloading now.",
            "model_name": model_id,
            "profile": "custom",
        }

@app.post("/api/vllm/restart")
async def restart_vllm():
    try:
        ensure_docker_control_enabled()
        mark_model_load_started(get_active_model_name(), reason="restart", profile_key=get_active_model_profile()["key"])
        resp = await docker_api_request("POST", "/containers/vllm/restart", params={"t": 10})
        if resp.status_code == 204:
            return {"status": "success", "message": "vLLM is restarting. Model will reload in a few minutes."}
        return {"status": "error", "message": f"Restart failed (HTTP {resp.status_code})"}
    except Exception as e:
        logger.error(f"Failed to restart vLLM: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/model/switch")
async def switch_model(request: SwitchModelRequest):
    global ACTIVE_MODEL_PROFILE, ACTIVE_MODEL_LOCK

    ensure_docker_control_enabled()
    target_key = request.profile.strip().lower()
    if target_key not in MODEL_PROFILES:
        raise HTTPException(status_code=400, detail=f"Unknown model profile: {request.profile}")

    if ACTIVE_MODEL_LOCK is None:
        ACTIVE_MODEL_LOCK = asyncio.Lock()

    async with ACTIVE_MODEL_LOCK:
        current_key = ACTIVE_MODEL_PROFILE
        if current_key == target_key:
            profile = MODEL_PROFILES[target_key]
            return {
                "status": "success",
                "message": f"{profile['label']} is already selected.",
                "model_name": profile["name"],
                "profile": target_key,
            }

        logger.info("Switching vLLM profile from %s to %s", current_key, target_key)
        mark_model_load_started(MODEL_PROFILES[target_key]["name"], reason="switch", profile_key=target_key)
        await recreate_vllm_container(target_key)
        persist_active_model_selection(target_key, None)
        target_profile = MODEL_PROFILES[target_key]
        return {
            "status": "success",
            "message": f"Switching to {target_profile['label']} ({target_profile['name']}). vLLM is reloading now.",
            "model_name": target_profile["name"],
            "profile": target_key,
        }

@app.post("/api/model/redownload")
async def redownload_model():
    try:
        ensure_docker_control_enabled()
        mark_model_load_started(get_active_model_name(), reason="redownload", profile_key=get_active_model_profile()["key"])
        # Stop vLLM first
        await docker_api_request("POST", "/containers/vllm/stop", params={"t": 10})

        # Delete model cache
        model_dir = get_active_model_name().replace("/", "--")
        cache_dir = os.path.join(HF_CACHE_PATH, "hub", f"models--{model_dir}")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"Deleted model cache: {cache_dir}")

        # Start vLLM (will re-download on boot)
        await docker_api_request("POST", "/containers/vllm/start")

        return {"status": "success", "message": "Cache cleared. vLLM is restarting and will re-download the model."}
    except Exception as e:
        logger.error(f"Failed to redownload model: {e}")
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    global ACTIVE_MODEL_LOCK, MODEL_DOWNLOAD_LOCK
    ACTIVE_MODEL_LOCK = asyncio.Lock()
    MODEL_DOWNLOAD_LOCK = asyncio.Lock()
    logger.info("=" * 60)
    logger.info("AI Chat Application Starting...")
    logger.info(f"Selected model profile: {ACTIVE_MODEL_PROFILE} ({get_active_model_name()})")
    logger.info(f"vLLM: {VLLM_HOST}")
    logger.info("Docker control enabled: %s", DOCKER_CONTROL_ENABLED)
    logger.info("Interactive terminal enabled: %s", INTERACTIVE_TERMINAL_ENABLED)
    logger.info("Execute-code endpoint enabled: %s", EXECUTE_CODE_ENABLED)
    logger.info("=" * 60)

    # Check model availability in background
    async def wait_for_model():
        for i in range(120):  # Wait up to 10 minutes
            if await vllm_health_check():
                loaded_model_name = await fetch_loaded_model_name()
                sync_active_profile_from_model_name(loaded_model_name)
                logger.info(
                    "Model %s passed backend health check; waiting for clients to report UI readiness.",
                    loaded_model_name or get_active_model_name(),
                )
                return
            if i % 6 == 0:
                logger.info(f"Waiting for model to load... ({i*5}s)")
            await asyncio.sleep(5)
        logger.warning("Model did not become available within 10 minutes")

    asyncio.create_task(wait_for_model())

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting AI Chat with vLLM ({get_active_model_name()})...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
