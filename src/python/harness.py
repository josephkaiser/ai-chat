#!/usr/bin/env python3
"""
Coding companion with vLLM — short technical answers with code.

Repo map (start here if you are new):
  src/python/harness.py — FastAPI routes, WebSockets, SQLite helpers, vLLM httpx client
  src/python/src/python/ai_chat/   — Internal helper modules (prompts, themes, routing, readers)
  src/web/              — index.html, app.js, style.css (vanilla front end)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import mimetypes
import os
import pathlib
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import venv
import zipfile
from html import unescape
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse, urlunparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.python.ai_chat.deep_flow import DeepRouteRequest, decide_deep_route
try:
    import numpy as np
except Exception:
    np = None
try:
    import pandas as pd
except Exception:
    pd = None

from src.python.ai_chat.embeddings import configured_embedding_model_name, embed_passages, embed_queries, embeddings_available
from src.python.ai_chat.prompts import (
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
    STEP_DECOMPOSE_SYSTEM_PROMPT,
    TOOL_USE_SYSTEM_PROMPT,
)
from src.python.ai_chat.themes import COLORS_DARK, COLORS_LIGHT
from src.python.ai_chat.thinking_stream import ThinkingStreamSplitter, strip_stream_special_tokens
from src.python.ai_chat.turn_strategy import (
    TurnAssessment,
    build_turn_assessment,
    format_turn_assessment_summary,
    infer_explicit_planning_request,
)
from src.python.ai_chat.workspace_reader import (
    build_text_file_result as build_text_file_result_helper,
    build_tool_loop_hard_limit_message as build_tool_loop_hard_limit_message_helper,
    build_workspace_file_result as build_workspace_file_result_helper,
    normalize_pause_reason as normalize_pause_reason_helper,
    workspace_file_content_kind as workspace_file_content_kind_helper,
    workspace_file_default_view as workspace_file_default_view_helper,
    workspace_file_is_editable as workspace_file_is_editable_helper,
    workspace_file_live_reader_mode as workspace_file_live_reader_mode_helper,
)

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
DB_PATH = os.getenv("DB_PATH", "/app/data/chat.db")
VLLM_HOST = os.getenv("VLLM_HOST", "http://vllm:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-14B-AWQ")
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/cache/huggingface")
WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT", str(pathlib.Path("/app/workspaces")))
RUNS_ROOT = os.getenv("RUNS_ROOT", str(pathlib.Path("/app/runs")))
MANAGED_PYTHON_ENVS_ROOT = os.getenv("MANAGED_PYTHON_ENVS_ROOT", str(pathlib.Path("/app/python-envs")))
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
MODEL_GEMMA_4_NAME = os.getenv("MODEL_GEMMA_4_NAME", "google/gemma-4-E4B-it")
MODEL_GEMMA_4_ARGS = os.getenv(
    "MODEL_GEMMA_4_ARGS",
    "--gpu-memory-utilization 0.85 --max-model-len 8192 --enable-prefix-caching "
    "--max-num-seqs 12 --enable-chunked-prefill --enforce-eager",
)
CUSTOM_MODEL_ARGS = os.getenv("CUSTOM_MODEL_ARGS", MODEL_14B_ARGS)
DEFAULT_MODEL_PROFILE = os.getenv("DEFAULT_MODEL_PROFILE", "14b").strip().lower()
DEEP_CRITIQUE_ENABLED = os.getenv("DEEP_CRITIQUE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
TOOL_LOOP_ENABLED = os.getenv("TOOL_LOOP_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
TOOL_LOOP_MAX_STEPS = int(os.getenv("TOOL_LOOP_MAX_STEPS", "8"))
TOOL_LOOP_MAX_CONTINUATIONS = max(0, int(os.getenv("TOOL_LOOP_MAX_CONTINUATIONS", "3")))
AUTO_VERIFY_AFTER_PATCH = os.getenv("AUTO_VERIFY_AFTER_PATCH", "1").strip().lower() not in {"0", "false", "no"}
AUTO_VERIFY_MAX_RUNS = max(0, int(os.getenv("AUTO_VERIFY_MAX_RUNS", "2")))
WORKSPACE_FILE_SIZE_LIMIT = 1024 * 1024
WORKSPACE_WRITE_SIZE_LIMIT = 1024 * 1024
COMMAND_TIMEOUT_SECONDS = float(os.getenv("COMMAND_TIMEOUT_SECONDS", "8"))
COMMAND_OUTPUT_LIMIT = int(os.getenv("COMMAND_OUTPUT_LIMIT", "12000"))
TOOL_RESULT_TEXT_LIMIT = int(os.getenv("TOOL_RESULT_TEXT_LIMIT", "40000"))
WORKSPACE_COMMAND_ARTIFACT_LIMIT = max(1, int(os.getenv("WORKSPACE_COMMAND_ARTIFACT_LIMIT", "8")))
WEB_PAGE_TEXT_LIMIT = int(os.getenv("WEB_PAGE_TEXT_LIMIT", "16000"))
WEB_FETCH_PAGE_MAX_PER_TURN = max(1, int(os.getenv("WEB_FETCH_PAGE_MAX_PER_TURN", "3")))
CURATED_SOURCE_FAILURE_THRESHOLD = max(1, int(os.getenv("CURATED_SOURCE_FAILURE_THRESHOLD", "2")))
CURATED_SOURCE_DISABLE_MINUTES = max(1, int(os.getenv("CURATED_SOURCE_DISABLE_MINUTES", "240")))
UPLOAD_FILE_SIZE_LIMIT = 10 * 1024 * 1024
VOICE_INPUT_SIZE_LIMIT = int(os.getenv("VOICE_INPUT_SIZE_LIMIT", str(15 * 1024 * 1024)))
VOICE_COMMAND_TIMEOUT_SECONDS = float(os.getenv("VOICE_COMMAND_TIMEOUT_SECONDS", "180"))
VOICE_TTS_COMMAND = os.getenv("VOICE_TTS_COMMAND", "").strip()
VOICE_STT_COMMAND = os.getenv("VOICE_STT_COMMAND", "").strip()
VOICE_STT_LANGUAGE = os.getenv("VOICE_STT_LANGUAGE", "en").strip() or "en"
VOICE_TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "Samantha").strip() or "Samantha"
VOICE_STORAGE_LIMIT_BYTES = max(0, int(os.getenv("VOICE_STORAGE_LIMIT_BYTES", str(3 * 1024 * 1024 * 1024))))
VOICE_EPHEMERAL_KINDS = ("input", "transcripts", "tts-text", "tts-audio")
MAX_ATTACHMENTS_PER_MESSAGE = 8
SPREADSHEET_PREVIEW_ROWS = 8
SPREADSHEET_MAX_COLUMNS = 40
SPREADSHEET_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm"}
TEXT_DOCUMENT_EXTENSIONS = {
    ".c", ".cc", ".cfg", ".conf", ".cpp", ".css", ".csv", ".env", ".go", ".h", ".hpp", ".html",
    ".ini", ".java", ".js", ".json", ".jsx", ".log", ".md", ".py", ".rb", ".rs", ".rst", ".sh",
    ".sql", ".svg", ".tex", ".toml", ".ts", ".tsx", ".txt", ".xml", ".yaml", ".yml",
}
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".rst"}
HTML_EXTENSIONS = {".htm", ".html"}
DELIMITED_TEXT_EXTENSIONS = {".csv", ".tsv"}
BINARY_SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
DOCUMENT_INDEX_SIZE_LIMIT = int(os.getenv("DOCUMENT_INDEX_SIZE_LIMIT", str(25 * 1024 * 1024)))
DOCUMENT_TEXT_READ_LIMIT = int(os.getenv("DOCUMENT_TEXT_READ_LIMIT", str(8 * 1024 * 1024)))
DOCUMENT_CHUNK_TARGET_CHARS = int(os.getenv("DOCUMENT_CHUNK_TARGET_CHARS", "900"))
DOCUMENT_CHUNK_OVERLAP_CHARS = int(os.getenv("DOCUMENT_CHUNK_OVERLAP_CHARS", "120"))
DOCUMENT_RETRIEVAL_CONTEXT_BUDGET = int(os.getenv("DOCUMENT_RETRIEVAL_CONTEXT_BUDGET", "9000"))
DOCUMENT_RETRIEVAL_MAX_WINDOWS = int(os.getenv("DOCUMENT_RETRIEVAL_MAX_WINDOWS", "4"))
DOCUMENT_RETRIEVAL_FTS_LIMIT = int(os.getenv("DOCUMENT_RETRIEVAL_FTS_LIMIT", "18"))
DOCUMENT_RETRIEVAL_SEMANTIC_LIMIT = int(os.getenv("DOCUMENT_RETRIEVAL_SEMANTIC_LIMIT", "32"))
MESSAGE_RETRIEVAL_SEMANTIC_LIMIT = int(os.getenv("MESSAGE_RETRIEVAL_SEMANTIC_LIMIT", "24"))
EMBEDDING_TEXT_CHAR_LIMIT = int(os.getenv("EMBEDDING_TEXT_CHAR_LIMIT", "2400"))
RETRIEVAL_RRF_K = int(os.getenv("RETRIEVAL_RRF_K", "60"))
DOCUMENT_COMMAND_TIMEOUT_SECONDS = float(os.getenv("DOCUMENT_COMMAND_TIMEOUT_SECONDS", "30"))
PDFTOTEXT_BIN = shutil.which("pdftotext") or ""
PDFINFO_BIN = shutil.which("pdfinfo") or ""
WORKSPACE_SIGNAL_VERBS = {
    "inspect", "read", "open", "show", "list", "search", "find", "grep",
    "edit", "change", "update", "patch", "refactor", "create", "write", "add",
    "delete", "remove", "rename", "run", "execute", "test", "build", "compile",
    "debug", "fix", "implement", "make", "making", "start", "starting", "draft",
    "wire", "wiring", "setup", "set", "tweak", "adjust", "modify", "improve", "revise",
}
WORKSPACE_SIGNAL_NOUNS = {
    "workspace", "repo", "repository", "codebase", "project", "folder", "directory",
    "file", "files", "code", "app", "application", "program", "module", "script", "source", "test", "tests",
    "model", "models", "tracker", "trackers", "tracking", "workflow", "workflows",
    "pipeline", "pipelines", "automation", "automations", "monitor", "monitors",
    "job", "jobs", "dashboard", "dashboards",
}
WORKSPACE_TEMPLATE_TERMS = {
    "template", "starter", "scaffold", "boilerplate", "example", "sample",
    "saas", "mvp", "skeleton", "seed", "bootstrap", "generate",
}
SERVER_REPO_BOOTSTRAP_EXCLUDE_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "data",
    "feeds",
    "logs",
    "node_modules",
    "python-envs",
    "runs",
    "workspaces",
}
WORKSPACE_HIDDEN_RUNTIME_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "python-env",
    "python-envs",
}
CURRENT_REPO_REFERENCE_PHRASES = (
    "this repo",
    "this repository",
    "this codebase",
    "repo here",
    "repository here",
    "codebase here",
    "current repo",
    "current repository",
    "current codebase",
    "the repo here",
    "the repository here",
    "in this repo",
    "in this repository",
    "in this codebase",
)


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}

STRICT_WORKSPACE_COMMAND_PATHS = env_flag("STRICT_WORKSPACE_COMMAND_PATHS", True)
def infer_model_provider(model_name: str) -> str:
    """Infer a friendly provider/vendor label from a model identifier."""
    model_name = (model_name or "").strip()
    namespace = model_name.split("/", 1)[0].strip().lower()
    namespace_map = {
        "qwen": "Qwen",
        "qwenlm": "Qwen",
        "meta-llama": "Meta",
        "google": "Google",
        "mistralai": "Mistral",
        "microsoft": "Microsoft",
        "deepseek-ai": "DeepSeek",
        "moonshotai": "Moonshot",
        "tiiuae": "TII",
    }
    if namespace:
        return namespace_map.get(namespace, namespace.replace("-", " ").title())

    lowered = model_name.lower()
    if "gemma" in lowered:
        return "Google"
    if "llama" in lowered:
        return "Meta"
    if "qwen" in lowered:
        return "Qwen"
    return "Unknown"


def build_profile_display_label(label: str, provider: str) -> str:
    """Build a concise profile label that keeps the provider visible."""
    normalized_label = (label or "").strip()
    normalized_provider = (provider or "").strip()
    if not normalized_label:
        return normalized_provider or "Model"
    if not normalized_provider:
        return normalized_label
    if normalized_provider.lower() in normalized_label.lower():
        return normalized_label
    return f"{normalized_label} ({normalized_provider})"


def build_model_profile(
    key: str,
    name: str,
    args: str,
    label: Optional[str] = None,
    *,
    fast_path: bool = False,
) -> Dict[str, Any]:
    """Create a model profile dictionary."""
    base_label = label or key.upper()
    provider = infer_model_provider(name)
    return {
        "key": key,
        "label": base_label,
        "display_label": build_profile_display_label(base_label, provider),
        "provider": provider,
        "name": name,
        "args": args,
        "fast_path": fast_path,
    }


def build_configured_model_profiles() -> Dict[str, Dict[str, Any]]:
    """Build the configured model profile table from environment defaults."""
    configured = [
        ("14b", MODEL_14B_NAME, MODEL_14B_ARGS, "14B", False),
        ("8b", MODEL_8B_NAME, MODEL_8B_ARGS, "8B", True),
        ("gemma4", MODEL_GEMMA_4_NAME, MODEL_GEMMA_4_ARGS, "Gemma 4 E4B", True),
    ]
    profiles: Dict[str, Dict[str, Any]] = {}
    for key, name, args, label, fast_path in configured:
        normalized_name = (name or "").strip()
        if not normalized_name:
            continue
        profiles[key] = build_model_profile(key, normalized_name, args, label=label, fast_path=fast_path)
    return profiles


MODEL_PROFILES: Dict[str, Dict[str, Any]] = build_configured_model_profiles()
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
PERMISSION_APPROVAL_WAITERS: Dict[str, Dict[str, Any]] = {}
CURATED_SOURCE_HEALTH: Dict[str, Dict[str, Any]] = {}


def serialize_model_profile(profile: Dict[str, Any], *, active: bool = False, selected: bool = False) -> Dict[str, Any]:
    """Return the UI-safe subset of a model profile."""
    return {
        "key": profile["key"],
        "label": profile["label"],
        "display_label": profile.get("display_label") or profile["label"],
        "provider": profile.get("provider") or "",
        "name": profile["name"],
        "active": active,
        "selected": selected,
    }


def get_active_model_profile() -> Dict[str, Any]:
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
    return bool(get_active_model_profile().get("fast_path"))


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
APP_TITLE = "AI Chat"

# Completion budget: ceiling only. The model still ends the stream when it predicts EOS
# (end of sequence); it does not "fill" unused max_tokens. Tune via env if replies truncate.
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "4096"))

WORKSPACE_ROOT_PATH = pathlib.Path(WORKSPACE_ROOT).resolve()
WORKSPACE_ROOT_PATH.mkdir(parents=True, exist_ok=True)
RUNS_ROOT_PATH = pathlib.Path(RUNS_ROOT).resolve()
RUNS_ROOT_PATH.mkdir(parents=True, exist_ok=True)
MANAGED_PYTHON_ENVS_ROOT_PATH = pathlib.Path(MANAGED_PYTHON_ENVS_ROOT).resolve()
MANAGED_PYTHON_ENVS_ROOT_PATH.mkdir(parents=True, exist_ok=True)
VOICE_ROOT_PATH = pathlib.Path(VOICE_ROOT).resolve()
VOICE_ROOT_PATH.mkdir(parents=True, exist_ok=True)
PIPER_DEFAULT_MODEL = pathlib.Path(
    os.getenv("PIPER_MODEL", str(VOICE_ROOT_PATH / "models" / "en_US-lessac-high.onnx"))
)


def sanitize_conversation_id(conversation_id: str) -> str:
    """Restrict conversation IDs to a safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", (conversation_id or "").strip())
    return cleaned[:120] or "default"


def sanitize_workspace_slug(value: str) -> str:
    """Normalize a display name into a filesystem-friendly slug."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip().lower()).strip("._-")
    return cleaned[:80] or "workspace"


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


def utcnow_iso() -> str:
    """Return a stable timestamp string for persisted records."""
    return datetime.now().isoformat()


def workspace_display_name_from_path(path: pathlib.Path) -> str:
    """Choose a stable user-facing name for one workspace root."""
    name = str(path.name or "").strip()
    if name:
        return name
    return str(path).strip() or "Workspace"


def canonicalize_workspace_root_path(root_path: str, *, create: bool = False) -> pathlib.Path:
    """Validate and normalize a workspace root path supplied by the user or DB."""
    raw = str(root_path or "").strip()
    if not raw:
        raise ValueError("Workspace path is required")
    candidate = pathlib.Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ValueError("Workspace path must be absolute")
    resolved = candidate.resolve(strict=False)
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("Workspace path must point to a directory")
    if not create and not resolved.exists():
        raise ValueError("Workspace path does not exist")
    return resolved


def allocate_managed_workspace_root(display_name: str) -> pathlib.Path:
    """Allocate a new direct-root workspace under the configured workspace root."""
    slug = sanitize_workspace_slug(display_name)
    for index in range(1, 1000):
        suffix = "" if index == 1 else f"-{index}"
        candidate = (WORKSPACE_ROOT_PATH / f"{slug}{suffix}").resolve()
        if WORKSPACE_ROOT_PATH not in candidate.parents and candidate != WORKSPACE_ROOT_PATH:
            raise ValueError("Workspace path escaped workspace root")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise ValueError("Unable to allocate a workspace directory")


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


def build_workspace_run_id(workspace_id: str) -> str:
    """Derive a stable run id from a workspace id."""
    return f"workspace-{sanitize_conversation_id(workspace_id)}"


def workspace_row_to_record(row: Optional[tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    """Normalize a workspace row into a dict payload."""
    if not row:
        return None
    return {
        "id": str(row[0] or "").strip(),
        "display_name": str(row[1] or "").strip() or "Workspace",
        "root_path": str(row[2] or "").strip(),
        "created_at": row[3],
        "updated_at": row[4],
    }


def get_workspace_record(workspace_id: str) -> Optional[Dict[str, Any]]:
    """Return one workspace catalog entry by id."""
    safe_workspace_id = str(workspace_id or "").strip()
    if not safe_workspace_id:
        return None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            '''SELECT id, display_name, root_path, created_at, updated_at
               FROM workspaces
               WHERE id = ?''',
            (safe_workspace_id,),
        )
        row = c.fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()
    return workspace_row_to_record(row)


def get_workspace_record_by_path(root_path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Return one workspace row for a canonical root path."""
    resolved = canonicalize_workspace_root_path(str(root_path), create=False)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            '''SELECT id, display_name, root_path, created_at, updated_at
               FROM workspaces
               WHERE root_path = ?''',
            (str(resolved),),
        )
        row = c.fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()
    return workspace_row_to_record(row)


def count_conversations_for_workspace(workspace_id: str) -> int:
    """Return how many chats currently point at one workspace."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('SELECT COUNT(*) FROM conversations WHERE workspace_id = ?', (workspace_id,))
        row = c.fetchone()
    except sqlite3.OperationalError:
        row = (0,)
    conn.close()
    return int(row[0] or 0) if row else 0


def list_conversation_ids_for_workspace(workspace_id: str) -> List[str]:
    """Return every conversation currently attached to one workspace."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('SELECT id FROM conversations WHERE workspace_id = ? ORDER BY updated_at DESC, created_at DESC', (workspace_id,))
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return [str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()]


def create_workspace_record(
    *,
    display_name: Optional[str] = None,
    root_path: Optional[str] = None,
    create_if_missing: bool = True,
) -> Dict[str, Any]:
    """Create and persist one workspace catalog entry."""
    now = utcnow_iso()
    resolved_path = (
        canonicalize_workspace_root_path(root_path, create=create_if_missing)
        if str(root_path or "").strip()
        else allocate_managed_workspace_root(display_name or "Workspace")
    )
    existing = get_workspace_record_by_path(resolved_path) if resolved_path.exists() else None
    if existing:
        raise ValueError("A workspace already exists for that path")

    workspace_id = uuid.uuid4().hex
    final_name = str(display_name or "").strip() or workspace_display_name_from_path(resolved_path)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''INSERT INTO workspaces (id, display_name, root_path, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)''',
        (workspace_id, final_name, str(resolved_path), now, now),
    )
    conn.commit()
    conn.close()
    return {
        "id": workspace_id,
        "display_name": final_name,
        "root_path": str(resolved_path),
        "created_at": now,
        "updated_at": now,
    }


def list_workspace_records(*, ensure_default: bool = False) -> List[Dict[str, Any]]:
    """Return the shared workspace catalog."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            '''SELECT id, display_name, root_path, created_at, updated_at
               FROM workspaces
               ORDER BY LOWER(display_name), created_at'''
        )
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    if not rows and ensure_default:
        create_workspace_record(display_name="Workspace")
        return list_workspace_records(ensure_default=False)
    workspaces: List[Dict[str, Any]] = []
    for row in rows:
        record = workspace_row_to_record(row)
        if record:
            workspaces.append(record)
    return workspaces


def ensure_default_workspace() -> Dict[str, Any]:
    """Return the fallback workspace used when the UI has not picked one yet."""
    workspaces = list_workspace_records(ensure_default=True)
    if not workspaces:
        raise ValueError("Unable to create a default workspace")
    return workspaces[0]


def get_conversation_record(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Return one conversation row including workspace linkage."""
    safe_conversation_id = str(conversation_id or "").strip()
    if not safe_conversation_id:
        return None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            '''SELECT id, title, created_at, updated_at, run_id, workspace_id
               FROM conversations
               WHERE id = ?''',
            (safe_conversation_id,),
        )
        row = c.fetchone()
    except sqlite3.OperationalError:
        c.execute(
            '''SELECT id, title, created_at, updated_at, run_id
               FROM conversations
               WHERE id = ?''',
            (safe_conversation_id,),
        )
        legacy_row = c.fetchone()
        row = (*legacy_row, "") if legacy_row else None
    conn.close()
    if not row:
        return None
    return {
        "id": str(row[0] or "").strip(),
        "title": str(row[1] or "").strip(),
        "created_at": row[2],
        "updated_at": row[3],
        "run_id": str(row[4] or "").strip(),
        "workspace_id": str(row[5] or "").strip(),
    }


def ensure_conversation_record(
    conversation_id: str,
    *,
    title: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a conversation row if needed and make sure it points at one workspace."""
    safe_conversation_id = str(conversation_id or "").strip()
    if not safe_conversation_id:
        raise ValueError("Conversation id is required")
    existing = get_conversation_record(safe_conversation_id)
    now = utcnow_iso()
    resolved_workspace = get_workspace_record(workspace_id) if str(workspace_id or "").strip() else None
    if not resolved_workspace:
        if existing and existing.get("workspace_id"):
            resolved_workspace = get_workspace_record(existing["workspace_id"])
        if not resolved_workspace:
            try:
                resolved_workspace = ensure_default_workspace()
            except Exception:
                resolved_workspace = {
                    "id": "",
                    "display_name": "",
                    "root_path": "",
                    "created_at": now,
                    "updated_at": now,
                }

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if not existing:
        try:
            c.execute(
                '''INSERT INTO conversations (id, title, created_at, updated_at, workspace_id)
                   VALUES (?, ?, ?, ?, ?)''',
                (
                    safe_conversation_id,
                    str(title or "").strip(),
                    now,
                    now,
                    resolved_workspace["id"],
                ),
            )
        except sqlite3.OperationalError:
            c.execute(
                '''INSERT INTO conversations (id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?)''',
                (
                    safe_conversation_id,
                    str(title or "").strip(),
                    now,
                    now,
                ),
            )
    else:
        updates = []
        params: List[Any] = []
        if resolved_workspace["id"] and existing.get("workspace_id") != resolved_workspace["id"]:
            updates.append("workspace_id = ?")
            params.append(resolved_workspace["id"])
        if str(title or "").strip() and not str(existing.get("title") or "").strip():
            updates.append("title = ?")
            params.append(str(title or "").strip())
        if updates:
            updates.append("updated_at = ?")
            params.append(now)
            params.append(safe_conversation_id)
            try:
                c.execute(f'''UPDATE conversations SET {", ".join(updates)} WHERE id = ?''', params)
            except sqlite3.OperationalError:
                fallback_updates = []
                fallback_params: List[Any] = []
                if str(title or "").strip() and not str(existing.get("title") or "").strip():
                    fallback_updates.append("title = ?")
                    fallback_params.append(str(title or "").strip())
                if fallback_updates:
                    fallback_updates.append("updated_at = ?")
                    fallback_params.append(now)
                    fallback_params.append(safe_conversation_id)
                    c.execute(f'''UPDATE conversations SET {", ".join(fallback_updates)} WHERE id = ?''', fallback_params)
    conn.commit()
    conn.close()
    return get_conversation_record(safe_conversation_id) or {
        "id": safe_conversation_id,
        "title": str(title or "").strip(),
        "created_at": now,
        "updated_at": now,
        "run_id": "",
        "workspace_id": resolved_workspace["id"],
    }


def get_workspace_record_for_conversation(
    conversation_id: str,
    *,
    create: bool = True,
    requested_workspace_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve the workspace row for one conversation."""
    conversation = get_conversation_record(conversation_id)
    workspace_id = str(conversation.get("workspace_id") or "").strip() if conversation else ""
    if not workspace_id and str(requested_workspace_id or "").strip():
        workspace_id = str(requested_workspace_id or "").strip()
    workspace = get_workspace_record(workspace_id) if workspace_id else None
    if workspace:
        return workspace
    if not create:
        return None
    ensure_conversation_record(conversation_id, workspace_id=requested_workspace_id)
    conversation = get_conversation_record(conversation_id)
    workspace_id = str(conversation.get("workspace_id") or "").strip() if conversation else ""
    return get_workspace_record(workspace_id) if workspace_id else None


def get_workspace_id_for_conversation(
    conversation_id: str,
    *,
    create: bool = True,
    requested_workspace_id: Optional[str] = None,
) -> str:
    """Resolve the stable workspace id for one conversation."""
    workspace = get_workspace_record_for_conversation(
        conversation_id,
        create=create,
        requested_workspace_id=requested_workspace_id,
    )
    return str(workspace.get("id") or "").strip() if workspace else ""


def get_run_record_by_workspace_id(workspace_id: str) -> Optional[Dict[str, Any]]:
    """Return the run metadata for a workspace, if any."""
    safe_workspace_id = str(workspace_id or "").strip()
    if not safe_workspace_id:
        return None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            '''SELECT id, conversation_id, workspace_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count
               FROM runs
               WHERE workspace_id = ?''',
            (safe_workspace_id,),
        )
        row = c.fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "conversation_id": row[1],
        "workspace_id": row[2],
        "title": row[3] or "",
        "status": row[4] or "active",
        "sandbox_path": row[5],
        "started_at": row[6],
        "ended_at": row[7],
        "summary": row[8] or "",
        "promoted_count": int(row[9] or 0),
    }


def get_run_record(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Return the run metadata for a conversation, if any."""
    workspace_id = get_workspace_id_for_conversation(conversation_id, create=False)
    if workspace_id:
        workspace_run = get_run_record_by_workspace_id(workspace_id)
        if workspace_run:
            return workspace_run

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            '''SELECT id, conversation_id, workspace_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count
               FROM runs
               WHERE conversation_id = ?''',
            (conversation_id,),
        )
        row = c.fetchone()
    except sqlite3.OperationalError:
        c.execute(
            '''SELECT id, conversation_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count
               FROM runs
               WHERE conversation_id = ?''',
            (conversation_id,),
        )
        legacy_row = c.fetchone()
        row = (legacy_row[0], legacy_row[1], "", *legacy_row[2:]) if legacy_row else None
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "conversation_id": row[1],
        "workspace_id": row[2],
        "title": row[3] or "",
        "status": row[4] or "active",
        "sandbox_path": row[5],
        "started_at": row[6],
        "ended_at": row[7],
        "summary": row[8] or "",
        "promoted_count": int(row[9] or 0),
    }


def ensure_run_for_workspace(
    workspace_id: str,
    *,
    conversation_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or fetch the run record that owns one workspace."""
    workspace = get_workspace_record(workspace_id)
    if not workspace:
        raise ValueError("Workspace not found")

    existing = get_run_record_by_workspace_id(workspace_id)
    workspace_path = canonicalize_workspace_root_path(workspace["root_path"], create=True)
    if existing:
        if existing["sandbox_path"] != str(workspace_path):
            conn = sqlite3.connect(DB_PATH)
            conn.execute('UPDATE runs SET sandbox_path = ? WHERE id = ?', (str(workspace_path), existing["id"]))
            conn.commit()
            conn.close()
            existing["sandbox_path"] = str(workspace_path)
        return existing

    run_id = build_workspace_run_id(workspace_id)
    now = utcnow_iso()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            '''INSERT OR REPLACE INTO runs
               (id, conversation_id, workspace_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                run_id,
                str(conversation_id or "").strip() or None,
                workspace_id,
                title or workspace.get("display_name") or "",
                "active",
                str(workspace_path),
                now,
                None,
                "",
                0,
            ),
        )
    except sqlite3.OperationalError:
        conn.execute(
            '''INSERT OR REPLACE INTO runs
               (id, conversation_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                run_id,
                str(conversation_id or "").strip() or "",
                title or workspace.get("display_name") or "",
                "active",
                str(workspace_path),
                now,
                None,
                "",
                0,
            ),
        )
    if str(conversation_id or "").strip():
        try:
            conn.execute('UPDATE conversations SET run_id = ? WHERE id = ?', (run_id, conversation_id))
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()
    return get_run_record_by_workspace_id(workspace_id) or {
        "id": run_id,
        "conversation_id": str(conversation_id or "").strip(),
        "workspace_id": workspace_id,
        "title": title or workspace.get("display_name") or "",
        "status": "active",
        "sandbox_path": str(workspace_path),
        "started_at": now,
        "ended_at": None,
        "summary": "",
        "promoted_count": 0,
    }


def ensure_run_for_conversation(
    conversation_id: str,
    title: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or fetch the run record that owns a conversation's workspace."""
    conversation = ensure_conversation_record(conversation_id, title=title, workspace_id=workspace_id)
    resolved_workspace_id = str(conversation.get("workspace_id") or "").strip()
    if not resolved_workspace_id:
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
            "workspace_id": "",
            "title": title or "",
            "status": "active",
            "sandbox_path": str(workspace),
            "started_at": now,
            "ended_at": None,
            "summary": "",
            "promoted_count": 0,
        }
    run = ensure_run_for_workspace(resolved_workspace_id, conversation_id=conversation_id, title=title)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('UPDATE conversations SET run_id = ? WHERE id = ?', (run["id"], conversation_id))
        conn.commit()
        conn.close()
    except sqlite3.OperationalError:
        pass
    return run


def get_workspace_path_for_workspace_id(workspace_id: str, create: bool = True) -> pathlib.Path:
    """Return the persistent root path for a workspace id."""
    workspace = get_workspace_record(workspace_id)
    if not workspace:
        raise ValueError("Workspace not found")
    return canonicalize_workspace_root_path(workspace["root_path"], create=create)


def get_workspace_path(conversation_id: str, create: bool = True) -> pathlib.Path:
    """Return the persistent workspace path for a conversation."""
    workspace = get_workspace_record_for_conversation(conversation_id, create=create)
    if workspace:
        return canonicalize_workspace_root_path(workspace["root_path"], create=create)
    if not create:
        legacy = (RUNS_ROOT_PATH / sanitize_conversation_id(build_run_id(conversation_id)) / "workspace").resolve()
        if RUNS_ROOT_PATH not in legacy.parents:
            raise ValueError("Workspace path escaped runs root")
        return legacy
    ensure_run_for_conversation(conversation_id)
    workspace = get_workspace_record_for_conversation(conversation_id, create=False)
    if not workspace:
        raise ValueError("Workspace not found")
    return canonicalize_workspace_root_path(workspace["root_path"], create=True)


def get_legacy_managed_python_env_path(conversation_id: str, create: bool = False) -> pathlib.Path:
    """Return the historical per-run managed Python environment path for one conversation."""
    run = get_run_record(conversation_id)
    run_id = str(run.get("id", "")).strip() if isinstance(run, dict) else ""
    if not run_id:
        run_id = build_run_id(conversation_id)
    env_root = (get_run_root(run_id, create=create) / "python-env").resolve()
    if create:
        env_root.parent.mkdir(parents=True, exist_ok=True)
    return env_root


def get_managed_python_env_path_for_workspace(workspace_id: str, create: bool = False) -> pathlib.Path:
    """Return the server-owned Python environment path for one workspace."""
    env_root = (MANAGED_PYTHON_ENVS_ROOT_PATH / sanitize_conversation_id(workspace_id)).resolve()
    if MANAGED_PYTHON_ENVS_ROOT_PATH not in env_root.parents and env_root != MANAGED_PYTHON_ENVS_ROOT_PATH:
        raise ValueError("Managed Python environment path escaped root")
    if create:
        env_root.parent.mkdir(parents=True, exist_ok=True)
    return env_root


def get_managed_python_env_path(conversation_id: str, create: bool = False) -> pathlib.Path:
    """Return the server-owned Python environment path for one conversation's workspace."""
    workspace_id = get_workspace_id_for_conversation(conversation_id, create=create)
    if workspace_id:
        return get_managed_python_env_path_for_workspace(workspace_id, create=create)
    env_root = (MANAGED_PYTHON_ENVS_ROOT_PATH / sanitize_conversation_id(conversation_id)).resolve()
    if MANAGED_PYTHON_ENVS_ROOT_PATH not in env_root.parents and env_root != MANAGED_PYTHON_ENVS_ROOT_PATH:
        raise ValueError("Managed Python environment path escaped root")
    if create:
        env_root.parent.mkdir(parents=True, exist_ok=True)
    return env_root


def resolve_existing_managed_python_env_path(conversation_id: str) -> pathlib.Path:
    """Return the active managed environment path, falling back to the legacy run-local location."""
    preferred = get_managed_python_env_path(conversation_id, create=False)
    executable = "python.exe" if os.name == "nt" else "python"
    preferred_python = preferred / ("Scripts" if os.name == "nt" else "bin") / executable
    if preferred_python.exists():
        return preferred
    legacy = get_legacy_managed_python_env_path(conversation_id, create=False)
    legacy_python = legacy / ("Scripts" if os.name == "nt" else "bin") / executable
    if legacy_python.exists():
        return legacy
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def delete_managed_python_env_for_workspace(workspace_id: str) -> None:
    """Delete the managed environment for a workspace plus any legacy per-chat fallbacks."""
    candidate_paths = [get_managed_python_env_path_for_workspace(workspace_id, create=False)]
    candidate_paths.extend(
        get_legacy_managed_python_env_path(conversation_id, create=False)
        for conversation_id in list_conversation_ids_for_workspace(workspace_id)
    )
    seen: set[str] = set()
    for env_root in candidate_paths:
        env_key = str(env_root)
        if env_key in seen:
            continue
        seen.add(env_key)
        if env_root.exists():
            shutil.rmtree(env_root, ignore_errors=True)


def delete_managed_python_env(conversation_id: str) -> None:
    """Delete any managed Python environment paths for a conversation's workspace."""
    workspace_id = get_workspace_id_for_conversation(conversation_id, create=False)
    if workspace_id:
        delete_managed_python_env_for_workspace(workspace_id)
        return
    legacy = get_legacy_managed_python_env_path(conversation_id, create=False)
    if legacy.exists():
        shutil.rmtree(legacy, ignore_errors=True)


def migrate_legacy_managed_python_env(conversation_id: str) -> pathlib.Path:
    """Move a legacy run-local managed Python environment into the dedicated server-owned root."""
    preferred = get_managed_python_env_path(conversation_id, create=False)
    if preferred.exists():
        return preferred

    legacy = get_legacy_managed_python_env_path(conversation_id, create=False)
    if not legacy.exists():
        return preferred

    preferred.parent.mkdir(parents=True, exist_ok=True)
    try:
        legacy.rename(preferred)
    except OSError:
        shutil.copytree(legacy, preferred, dirs_exist_ok=False)
        shutil.rmtree(legacy, ignore_errors=True)
    return preferred


def managed_python_bin_dir(conversation_id: str, create: bool = False) -> pathlib.Path:
    """Return the binary/scripts directory for the managed Python environment."""
    if create:
        env_root = get_managed_python_env_path(conversation_id, create=True)
    else:
        env_root = resolve_existing_managed_python_env_path(conversation_id)
    return env_root / ("Scripts" if os.name == "nt" else "bin")


def managed_python_python_path(conversation_id: str, create: bool = False) -> pathlib.Path:
    """Return the Python executable inside the managed workspace environment."""
    executable = "python.exe" if os.name == "nt" else "python"
    return managed_python_bin_dir(conversation_id, create=create) / executable


def managed_python_pip_path(conversation_id: str, create: bool = False) -> pathlib.Path:
    """Return the pip executable inside the managed workspace environment."""
    executable = "pip.exe" if os.name == "nt" else "pip"
    return managed_python_bin_dir(conversation_id, create=create) / executable


def managed_python_env_exists(conversation_id: str) -> bool:
    """Return whether the managed Python environment is already provisioned."""
    return managed_python_python_path(conversation_id, create=False).exists()


def _create_managed_python_env_sync(conversation_id: str) -> pathlib.Path:
    """Create the server-owned managed Python environment for one conversation."""
    existing = resolve_existing_managed_python_env_path(conversation_id)
    existing_python = managed_python_python_path(conversation_id, create=False)
    if existing.exists() and existing_python.exists():
        return existing

    env_root = get_managed_python_env_path(conversation_id, create=True)
    python_path = managed_python_python_path(conversation_id, create=False)
    if python_path.exists():
        return env_root
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(str(env_root))
    return env_root


async def ensure_managed_python_env(conversation_id: str) -> pathlib.Path:
    """Create the managed Python environment on demand."""
    if managed_python_env_exists(conversation_id):
        return migrate_legacy_managed_python_env(conversation_id)
    return await asyncio.to_thread(_create_managed_python_env_sync, conversation_id)


def get_voice_dir(kind: str, create: bool = True) -> pathlib.Path:
    """Return a server-owned directory for transient voice artifacts."""
    safe_kind = re.sub(r"[^A-Za-z0-9._-]+", "_", (kind or "misc").strip()) or "misc"
    target = (VOICE_ROOT_PATH / safe_kind).resolve()
    if VOICE_ROOT_PATH not in target.parents and target != VOICE_ROOT_PATH:
        raise ValueError("Voice path escaped voice root")
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def build_voice_artifact_prefix(conversation_id: Optional[str] = None) -> str:
    """Return a stable filename prefix for chat-scoped voice artifacts."""
    raw_conversation_id = (conversation_id or "").strip()
    if not raw_conversation_id:
        return ""
    cleaned = sanitize_conversation_id(raw_conversation_id)
    return f"conv-{cleaned}-"


def build_voice_artifact_id(conversation_id: Optional[str] = None) -> str:
    """Generate a unique artifact id that optionally ties the file to a conversation."""
    return f"{build_voice_artifact_prefix(conversation_id)}{uuid.uuid4().hex}"


def delete_voice_file(path: pathlib.Path) -> bool:
    """Best-effort delete for voice artifacts."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except Exception as exc:
        logger.warning("Failed to delete voice artifact %s: %s", path, exc)
        return False
    return True


def list_voice_cache_files() -> List[Dict[str, Any]]:
    """Collect removable files inside the app-owned voice cache."""
    files: List[Dict[str, Any]] = []
    for kind in VOICE_EPHEMERAL_KINDS:
        root = get_voice_dir(kind, create=False)
        if not root.exists():
            continue
        for child in root.iterdir():
            try:
                if child.is_dir() and not child.is_symlink():
                    continue
                stat = child.stat()
            except FileNotFoundError:
                continue
            files.append({
                "path": child,
                "size": max(0, int(stat.st_size)),
                "mtime": float(stat.st_mtime),
            })
    return files


def prune_voice_storage_if_needed(protected_paths: Optional[List[pathlib.Path]] = None) -> Dict[str, int]:
    """Prune the oldest cached voice artifacts when the cache exceeds its size budget."""
    cache_files = list_voice_cache_files()
    total_bytes = sum(item["size"] for item in cache_files)
    if VOICE_STORAGE_LIMIT_BYTES <= 0 or total_bytes <= VOICE_STORAGE_LIMIT_BYTES:
        return {
            "total_bytes": total_bytes,
            "limit_bytes": VOICE_STORAGE_LIMIT_BYTES,
            "removed_files": 0,
            "removed_bytes": 0,
        }

    protected = {
        path.resolve(strict=False)
        for path in (protected_paths or [])
    }
    removed_files = 0
    removed_bytes = 0

    for item in sorted(cache_files, key=lambda entry: (entry["mtime"], entry["path"].name)):
        if total_bytes - removed_bytes <= VOICE_STORAGE_LIMIT_BYTES:
            break
        target = item["path"].resolve(strict=False)
        if target in protected:
            continue
        if delete_voice_file(item["path"]):
            removed_files += 1
            removed_bytes += item["size"]

    if removed_files:
        logger.info(
            "Pruned %s voice artifacts (%s bytes) to enforce the %s-byte cache limit",
            removed_files,
            removed_bytes,
            VOICE_STORAGE_LIMIT_BYTES,
        )

    return {
        "total_bytes": max(0, total_bytes - removed_bytes),
        "limit_bytes": VOICE_STORAGE_LIMIT_BYTES,
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
    }


def delete_voice_artifacts_for_conversation(conversation_id: str) -> int:
    """Remove any cached voice artifacts previously generated for a conversation."""
    prefix = build_voice_artifact_prefix(conversation_id)
    if not prefix:
        return 0

    removed_files = 0
    for kind in VOICE_EPHEMERAL_KINDS:
        root = get_voice_dir(kind, create=False)
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.name.startswith(prefix):
                continue
            if child.is_dir() and not child.is_symlink():
                continue
            if delete_voice_file(child):
                removed_files += 1

    if removed_files:
        logger.info("Deleted %s voice artifacts for conversation %s", removed_files, conversation_id)
    return removed_files


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


@dataclass
class WorkflowExecutionContext:
    """Transient per-turn execution bookkeeping kept only in memory."""
    workflow_name: str
    route_metadata: Dict[str, Any] = field(default_factory=dict)
    tool_count: int = 0
    artifact_paths: List[str] = field(default_factory=list)


def create_workflow_execution(
    conversation_id: str,
    user_message_id: int,
    workflow_name: str,
    route_metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowExecutionContext:
    """Create an in-memory workflow context for a single turn."""
    del conversation_id, user_message_id
    return WorkflowExecutionContext(
        workflow_name=str(workflow_name or "direct_answer"),
        route_metadata=dict(route_metadata or {}),
    )


def persist_workflow_execution_route(execution: WorkflowExecutionContext) -> None:
    """Compatibility shim for the simplified runtime."""
    del execution


def record_workflow_step(
    execution: Optional[WorkflowExecutionContext],
    *,
    step_name: str,
    call: Dict[str, Any],
    result: Dict[str, Any],
    latency_ms: int = 0,
    auto_generated: bool = False,
) -> None:
    """Track successful tool outputs for the current turn only."""
    del step_name, latency_ms, auto_generated
    if not execution:
        return
    execution.tool_count += 1
    payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
    path = str(payload.get("path", "")).strip()
    if result.get("ok") and path and path not in execution.artifact_paths:
        execution.artifact_paths.append(path)
    if result.get("ok") and isinstance(payload.get("items"), list):
        for item in payload.get("items", []):
            item_path = str((item or {}).get("path", "")).strip() if isinstance(item, dict) else ""
            if item_path and item_path not in execution.artifact_paths:
                execution.artifact_paths.append(item_path)


def finalize_workflow_execution(
    execution: Optional[WorkflowExecutionContext],
    *,
    assistant_message_id: Optional[int] = None,
    final_outcome: str = "",
    status: str = "completed",
    error_text: str = "",
) -> None:
    """Compatibility shim for removed workflow persistence."""
    del execution, assistant_message_id, final_outcome, status, error_text


def sync_workflow_feedback_for_message(message_id: int, feedback: str) -> None:
    """Compatibility shim for feedback-aware workflow hooks."""
    del feedback
    message = get_message_by_id(message_id)
    if not message:
        return
    conversation_id = str(message.get("conversation_id") or "").strip()
    if conversation_id:
        schedule_conversation_summary_refresh(conversation_id)


def resolve_workspace_relative_path(conversation_id: str, rel_path: str = "") -> pathlib.Path:
    """Resolve a user/model path inside the conversation workspace."""
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path_from_root(workspace, rel_path)
    return target


def resolve_workspace_relative_path_from_root(workspace: pathlib.Path, rel_path: str = "") -> pathlib.Path:
    """Resolve a user/model path inside a specific workspace root."""
    target = (workspace / (rel_path or "")).resolve()
    if target != workspace and workspace not in target.parents:
        raise HTTPException(status_code=403, detail="Access denied")
    return target


def path_is_within_root(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Return whether a path resolves inside an allowed root."""
    try:
        resolved_path = path.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
    except Exception:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def workspace_rel_path_is_hidden(rel_path: str) -> bool:
    """Return whether a workspace-relative path points at a dot-prefixed entry."""
    raw = str(rel_path or "").strip().replace("\\", "/")
    if not raw or raw == ".":
        return False
    parts = [part for part in pathlib.PurePosixPath(raw).parts if part not in {"", "."}]
    return any(part.startswith(".") or part.lower() in WORKSPACE_HIDDEN_RUNTIME_NAMES for part in parts)


def workspace_command_allows_argument(
    argument: str,
    workspace: pathlib.Path,
    extra_roots: Optional[List[pathlib.Path]] = None,
) -> bool:
    """Reject command arguments that reference paths outside the workspace."""
    token = str(argument or "").strip()
    if not token or token == "-":
        return True
    if "\x00" in token:
        return False
    if "://" in token or token.startswith("-"):
        return True

    normalized = token.replace("\\", "/")
    allowed_roots = [workspace, *[root for root in (extra_roots or []) if isinstance(root, pathlib.Path)]]
    if normalized.startswith("/"):
        try:
            resolved = pathlib.Path(normalized).resolve(strict=False)
        except Exception:
            return False
        return any(path_is_within_root(resolved, root) for root in allowed_roots)

    path_like = "/" in normalized or normalized in {".", ".."} or normalized.startswith("./") or normalized.startswith("../")
    if not path_like:
        return True

    try:
        resolved = (workspace / normalized).resolve(strict=False)
    except Exception:
        return False
    return any(path_is_within_root(resolved, root) for root in allowed_roots)


def normalize_allowed_command_key(value: str) -> str:
    """Normalize a persisted command-approval token."""
    return str(value or "").strip().lower()


def normalize_allowed_tool_permission_key(value: str) -> str:
    """Normalize a persisted tool-approval token."""
    return str(value or "").strip().lower()


def parse_pip_install_command(command: List[str]) -> Optional[Dict[str, Any]]:
    """Return normalized metadata when argv represents a pip install command."""
    if not isinstance(command, list) or not command:
        return None

    parts = [str(part).strip() for part in command if isinstance(part, str) and str(part).strip()]
    if not parts:
        return None

    executable_name = pathlib.Path(parts[0]).name.lower()
    lowered = [part.lower() for part in parts]
    install_index: Optional[int] = None

    if executable_name in {"pip", "pip3", "pip.exe", "pip3.exe"}:
        if len(parts) >= 2 and lowered[1] == "install":
            install_index = 1
    elif executable_name in {"python", "python3", "python.exe", "python3.exe"}:
        if len(parts) >= 4 and lowered[1] == "-m" and lowered[2] == "pip" and lowered[3] == "install":
            install_index = 3

    if install_index is None:
        return None

    requested_args = parts[install_index + 1:]
    return {
        "requested_args": requested_args,
        "packages_preview": compact_tool_text(" ".join(requested_args), limit=72) or "the requested Python packages",
    }


def parse_python_venv_command(command: List[str]) -> Optional[Dict[str, Any]]:
    """Return metadata when argv creates a Python virtual environment."""
    if not isinstance(command, list) or not command:
        return None

    parts = [str(part).strip() for part in command if isinstance(part, str) and str(part).strip()]
    if len(parts) < 4:
        return None

    executable_name = pathlib.Path(parts[0]).name.lower()
    lowered = [part.lower() for part in parts]
    if executable_name not in {"python", "python3", "python.exe", "python3.exe"}:
        return None
    if lowered[1] != "-m" or lowered[2] != "venv":
        return None

    target = next((part for part in parts[3:] if not part.startswith("-")), ".venv")
    return {"target": target}


def command_runtime_timeout_seconds(command: List[str]) -> Optional[float]:
    """Choose the timeout for a workspace command, or None to wait until completion."""
    if parse_pip_install_command(command) or parse_python_venv_command(command):
        return None
    return COMMAND_TIMEOUT_SECONDS


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

    if parse_pip_install_command(command):
        return "exec:pip.install"

    if parse_python_venv_command(command):
        return "exec:python.venv"

    normalized = executable.replace("\\", "/")
    if "/" not in normalized:
        return f"exec:{pathlib.Path(normalized).name.lower()}"

    workspace = get_workspace_path(conversation_id)
    managed_env_root = resolve_existing_managed_python_env_path(conversation_id)
    resolved_exec = pathlib.Path(executable).expanduser()
    if not resolved_exec.is_absolute():
        resolved_exec = (cwd_path / resolved_exec).resolve(strict=False)
    else:
        resolved_exec = resolved_exec.resolve(strict=False)

    if path_is_within_root(resolved_exec, managed_env_root):
        return f"exec:{pathlib.Path(resolved_exec).name.lower()}"

    if not path_is_within_root(resolved_exec, workspace):
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


def is_tool_permission_allowlisted(features: FeatureFlags, permission_key: str) -> bool:
    """Return whether a non-command tool permission was already approved for this chat."""
    normalized = normalize_allowed_tool_permission_key(permission_key)
    if not normalized:
        return False
    allowed = {
        normalize_allowed_tool_permission_key(item)
        for item in (features.allowed_tool_permissions or [])
        if isinstance(item, str) and normalize_allowed_tool_permission_key(item)
    }
    return normalized in allowed


def remember_approved_tool_permission(features: FeatureFlags, permission_key: str) -> None:
    """Persist an approved tool permission onto the in-flight feature flags."""
    normalized = normalize_allowed_tool_permission_key(permission_key)
    if not normalized:
        return
    current = [
        normalize_allowed_tool_permission_key(item)
        for item in (features.allowed_tool_permissions or [])
        if isinstance(item, str) and normalize_allowed_tool_permission_key(item)
    ]
    if normalized not in current:
        current.append(normalized)
    features.allowed_tool_permissions = sorted(set(current))


def remember_approved_command(features: FeatureFlags, command_key: str) -> None:
    """Persist an approved command executable onto the in-flight feature flags."""
    normalized = normalize_allowed_command_key(command_key)
    if not normalized:
        return
    current = [
        normalize_allowed_command_key(item)
        for item in (features.allowed_commands or [])
        if isinstance(item, str) and normalize_allowed_command_key(item)
    ]
    if normalized not in current:
        current.append(normalized)
    features.allowed_commands = sorted(set(current))


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
    managed_env_root = resolve_existing_managed_python_env_path(conversation_id)
    executable = command[0].strip()

    if "/" in executable:
        resolved_exec = pathlib.Path(executable).expanduser()
        if not resolved_exec.is_absolute():
            resolved_exec = (cwd_path / resolved_exec).resolve(strict=False)
        else:
            resolved_exec = resolved_exec.resolve(strict=False)
        if not path_is_within_root(resolved_exec, workspace) and not path_is_within_root(resolved_exec, managed_env_root):
            raise ValueError("Executable path must stay inside the workspace")

    if not STRICT_WORKSPACE_COMMAND_PATHS:
        return

    for index, part in enumerate(command[1:], start=1):
        if not workspace_command_allows_argument(part, workspace, extra_roots=[managed_env_root]):
            raise ValueError(f"command argument {index} references a path outside the workspace")


def delete_run_workspace(conversation_id: str):
    """Delete hosted run artifacts and the managed env for a conversation's workspace."""
    workspace_id = get_workspace_id_for_conversation(conversation_id, create=False)
    run = get_run_record(conversation_id)
    if run:
        sandbox_path = pathlib.Path(run["sandbox_path"]).resolve()
        if sandbox_path.exists() and (RUNS_ROOT_PATH == sandbox_path or RUNS_ROOT_PATH in sandbox_path.parents):
            hosted_root = sandbox_path.parent if sandbox_path.name == "workspace" else sandbox_path
            shutil.rmtree(hosted_root, ignore_errors=True)
    if workspace_id:
        delete_managed_python_env_for_workspace(workspace_id)
        return
    legacy = get_legacy_managed_python_env_path(conversation_id, create=False)
    if legacy.exists():
        shutil.rmtree(legacy, ignore_errors=True)


def build_workspace_command_env(conversation_id: str) -> Dict[str, str]:
    """Build the subprocess environment for workspace commands."""
    env = {**os.environ, "PYTHONPATH": "", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    python_path = managed_python_python_path(conversation_id, create=False)
    if python_path.exists():
        env_root = resolve_existing_managed_python_env_path(conversation_id)
        bin_dir = python_path.parent
        existing_path = env.get("PATH", "")
        env["PATH"] = str(bin_dir) if not existing_path else f"{bin_dir}{os.pathsep}{existing_path}"
        env["VIRTUAL_ENV"] = str(env_root)
        env.pop("PYTHONHOME", None)
    else:
        env.pop("VIRTUAL_ENV", None)
    return env


async def normalize_command_for_managed_python(
    conversation_id: str,
    command: List[str],
) -> List[str]:
    """Rewrite Python package installs to the managed chat environment."""
    pip_install = parse_pip_install_command(command)
    if not pip_install:
        return list(command)
    await ensure_managed_python_env(conversation_id)
    pip_path = managed_python_pip_path(conversation_id, create=False)
    return [str(pip_path), "install", *list(pip_install.get("requested_args", []))]


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


RESET_DB_REBUILD_MARKERS = (
    "database disk image is malformed",
    "file is not a database",
    "file is encrypted or is not a database",
    "malformed database schema",
    "database schema is corrupt",
    "no such table",
)


def reset_error_requires_database_rebuild(error: BaseException) -> bool:
    """Return whether a reset failure should recreate the SQLite file from scratch."""
    if not isinstance(error, sqlite3.DatabaseError):
        return False
    message = str(error).strip().lower()
    if not message:
        return False
    return any(marker in message for marker in RESET_DB_REBUILD_MARKERS)


def recreate_database_file_for_reset(reason: BaseException) -> None:
    """Delete the SQLite database and sidecars, then rebuild an empty schema."""
    logger.warning("Reset is recreating the SQLite database after error: %s", reason)
    db_path = pathlib.Path(DB_PATH)
    os.makedirs(db_path.parent, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            pathlib.Path(f"{DB_PATH}{suffix}").unlink(missing_ok=True)
        except FileNotFoundError:
            continue
    init_db()


async def reset_application_state() -> None:
    """Wipe persisted chat data, workspaces, and transient runtime state."""
    for waiter in PERMISSION_APPROVAL_WAITERS.values():
        future = waiter.get("future")
        if future and not future.done():
            future.cancel()
    PERMISSION_APPROVAL_WAITERS.clear()

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM conversation_summaries')
        c.execute('DELETE FROM messages')
        c.execute('DELETE FROM document_chunks')
        c.execute('DELETE FROM document_sources')
        c.execute('DELETE FROM conversations')
        c.execute('DELETE FROM runs')
        try:
            c.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('messages', 'document_chunks', 'document_sources')"
            )
        except sqlite3.OperationalError:
            pass
        if sqlite_has_fts(conn):
            try:
                c.execute(f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES ('rebuild')")
            except sqlite3.OperationalError:
                pass
            try:
                c.execute(f'DELETE FROM {DOCUMENT_FTS_TABLE}')
            except sqlite3.OperationalError:
                pass
        conn.commit()
    except sqlite3.DatabaseError as exc:
        if conn is not None:
            conn.close()
            conn = None
        if not reset_error_requires_database_rebuild(exc):
            raise
        recreate_database_file_for_reset(exc)
    finally:
        if conn is not None:
            conn.close()

    reset_directory_contents(RUNS_ROOT_PATH)
    reset_directory_contents(MANAGED_PYTHON_ENVS_ROOT_PATH)
    reset_directory_contents(WORKSPACE_ROOT_PATH)
    for kind in VOICE_EPHEMERAL_KINDS:
        reset_directory_contents(get_voice_dir(kind, create=True))


def format_workspace_path(path: pathlib.Path, workspace: pathlib.Path) -> str:
    """Render a workspace path using workspace-relative POSIX separators."""
    if path == workspace:
        return "."
    return path.relative_to(workspace).as_posix()


def is_spreadsheet_path(path: str) -> bool:
    """Return whether a workspace-relative path looks like a spreadsheet."""
    return pathlib.Path(path or "").suffix.lower() in SPREADSHEET_SUPPORTED_EXTENSIONS


def classify_workspace_file_kind(path: str) -> str:
    """Return a lightweight file category for UI attachment chips."""
    if is_spreadsheet_path(path):
        return "spreadsheet"
    if is_pdf_path(path):
        return "pdf"
    if workspace_file_content_kind(path) == "image":
        return "image"
    if is_text_document_path(path):
        return "document"
    return "file"


def build_workspace_listing_item(path: pathlib.Path, workspace: pathlib.Path) -> Dict[str, Any]:
    """Return the shared workspace file metadata used by listings and artifact surfacing."""
    stat = path.stat()
    rel_path = format_workspace_path(path, workspace)
    content_kind = workspace_file_content_kind(rel_path)
    return {
        "name": path.name,
        "path": rel_path,
        "type": "directory" if path.is_dir() else "file",
        "size": stat.st_size if path.is_file() else None,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "content_kind": content_kind,
        "kind": classify_workspace_file_kind(rel_path),
    }


def is_previewable_workspace_kind(kind: str) -> bool:
    """Return whether the inline viewer can preview this workspace content kind."""
    return str(kind or "").strip().lower() in {"image", "html", "markdown", "pdf", "csv", "spreadsheet"}


def preview_priority_for_workspace_kind(kind: str) -> int:
    """Prefer media and viewer-friendly files when choosing a primary artifact to surface."""
    normalized = str(kind or "").strip().lower()
    if normalized == "image":
        return 0
    if normalized == "html":
        return 1
    if normalized == "pdf":
        return 2
    if normalized == "markdown":
        return 3
    if normalized in {"csv", "spreadsheet"}:
        return 4
    if normalized == "document":
        return 5
    if normalized == "data":
        return 6
    if normalized == "code":
        return 7
    return 8


def capture_workspace_file_snapshot(conversation_id: str) -> Dict[str, Dict[str, Any]]:
    """Capture the current visible workspace files so command-created artifacts can be detected."""
    workspace = get_workspace_path(conversation_id, create=False)
    if not workspace.exists():
        return {}

    snapshot: Dict[str, Dict[str, Any]] = {}
    for candidate in workspace.rglob("*"):
        if not candidate.is_file():
            continue
        try:
            item = build_workspace_listing_item(candidate, workspace)
        except (OSError, PermissionError, ValueError):
            continue
        rel_path = str(item.get("path", "")).strip()
        if not rel_path or workspace_rel_path_is_hidden(rel_path):
            continue
        snapshot[rel_path] = item
    return snapshot


def detect_workspace_artifact_changes(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    *,
    limit: int = WORKSPACE_COMMAND_ARTIFACT_LIMIT,
) -> List[Dict[str, Any]]:
    """Return recently changed workspace files after a command run."""
    changes: List[Dict[str, Any]] = []
    for path, item in after.items():
        previous = before.get(path)
        if previous == item:
            continue
        changes.append(item)

    changes.sort(
        key=lambda item: (
            preview_priority_for_workspace_kind(item.get("content_kind", "")),
            -(datetime.fromisoformat(item.get("modified_at", "")).timestamp() if item.get("modified_at") else 0.0),
            str(item.get("path", "")).lower(),
        )
    )
    return changes[:max(1, int(limit or WORKSPACE_COMMAND_ARTIFACT_LIMIT))]


def choose_primary_workspace_artifact(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the most useful artifact to surface from a command run."""
    cleaned = [item for item in items if isinstance(item, dict) and str(item.get("path", "")).strip()]
    if not cleaned:
        return None
    return cleaned[0]


def indexable_for_retrieval(path: str | pathlib.Path) -> bool:
    """Return whether the file should still be indexed for retrieval."""
    return is_supported_document_path(path)


def workspace_file_content_kind(path: str | pathlib.Path) -> str:
    """Return the frontend content kind used by the live workspace reader."""
    return workspace_file_content_kind_helper(path)


def workspace_file_live_reader_mode(path: str | pathlib.Path) -> str:
    """Return how the live reader should open this file."""
    return workspace_file_live_reader_mode_helper(path)


def workspace_file_is_editable(path: str | pathlib.Path) -> bool:
    """Return whether the live workspace reader should allow editing."""
    return workspace_file_is_editable_helper(path)


def workspace_file_default_view(path: str | pathlib.Path) -> str:
    """Default the reader to preview mode so files open read-first."""
    return workspace_file_default_view_helper(path)


def build_text_file_result(
    target: pathlib.Path,
    rel_path: str,
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Read a workspace text file directly for the live reader or tool layer."""
    return build_text_file_result_helper(
        target,
        rel_path,
        max_bytes=WORKSPACE_FILE_SIZE_LIMIT,
        limit=limit,
        truncate_output_func=truncate_output,
    )


def build_workspace_file_result(
    target: pathlib.Path,
    *,
    conversation_id: Optional[str] = None,
    rel_path: Optional[str] = None,
    text_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Return one consistent payload shape for workspace file reads."""
    preview_limit = TOOL_RESULT_TEXT_LIMIT if text_limit is None else text_limit
    return build_workspace_file_result_helper(
        target,
        rel_path=rel_path,
        max_bytes=WORKSPACE_FILE_SIZE_LIMIT,
        document_preview_builder=build_document_preview_result,
        text_limit=preview_limit if workspace_file_live_reader_mode(target) == "document_preview" else text_limit,
        truncate_output_func=truncate_output,
        conversation_id=conversation_id,
    )


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


def is_pdf_path(path: str | pathlib.Path) -> bool:
    """Return whether the given path looks like a PDF."""
    return pathlib.Path(path).suffix.lower() == ".pdf"


def is_text_document_path(path: str | pathlib.Path) -> bool:
    """Return whether the path extension is commonly safe to parse as text."""
    return pathlib.Path(path).suffix.lower() in TEXT_DOCUMENT_EXTENSIONS


def is_supported_document_path(path: str | pathlib.Path) -> bool:
    """Return whether the app can index this file as a text-bearing document."""
    return is_pdf_path(path) or is_text_document_path(path)


def build_document_fingerprint(target: pathlib.Path) -> str:
    """Summarize file identity using cheap stat information."""
    stats = target.stat()
    return f"{int(stats.st_size)}:{int(stats.st_mtime_ns)}"


def run_document_command(command: List[str], timeout: float = DOCUMENT_COMMAND_TIMEOUT_SECONDS) -> tuple[int, str, str]:
    """Run a local document utility and decode stdout/stderr safely."""
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    return completed.returncode, stdout, stderr


def clean_document_text(text: str) -> str:
    """Normalize extracted text while keeping paragraph boundaries intact."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in str(text or "").replace("\r", "").split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def embedding_backend_active() -> bool:
    """Return whether the optional semantic retriever is ready to use."""
    return bool(np is not None and embeddings_available())


def active_embedding_model_name() -> str:
    """Return the configured semantic embedding model label."""
    return configured_embedding_model_name().strip() or "semantic-default"


def current_document_chunk_settings() -> Dict[str, int]:
    """Return the chunking settings that define the stored document index shape."""
    return {
        "chunk_target_chars": int(DOCUMENT_CHUNK_TARGET_CHARS),
        "chunk_overlap_chars": int(DOCUMENT_CHUNK_OVERLAP_CHARS),
    }


def cached_document_index_matches_settings(source: Optional[Dict[str, Any]]) -> bool:
    """Return whether a cached document index matches the current chunk settings."""
    metadata = (source or {}).get("metadata") or {}
    return (
        int(metadata.get("chunk_target_chars", 0) or 0) == DOCUMENT_CHUNK_TARGET_CHARS
        and int(metadata.get("chunk_overlap_chars", 0) or 0) == DOCUMENT_CHUNK_OVERLAP_CHARS
    )


def trim_embedding_text(text: str, limit: int = EMBEDDING_TEXT_CHAR_LIMIT) -> str:
    """Normalize and cap text before sending it to the embedding model."""
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)].rstrip(' ,.;:')}..."


def build_message_embedding_text(role: str, content: str) -> str:
    """Build a semantic-search-friendly representation of one message."""
    cleaned_content = trim_embedding_text(content)
    cleaned_role = str(role or "").strip().lower() or "message"
    return f"{cleaned_role}: {cleaned_content}" if cleaned_content else cleaned_role


def build_document_chunk_embedding_text(path: str, section_title: str, content: str) -> str:
    """Build a semantic-search-friendly representation of one stored chunk."""
    parts = []
    cleaned_path = trim_embedding_text(path, limit=220)
    cleaned_section = trim_embedding_text(section_title, limit=220)
    cleaned_content = trim_embedding_text(content)
    if cleaned_path:
        parts.append(f"path: {cleaned_path}")
    if cleaned_section:
        parts.append(f"section: {cleaned_section}")
    if cleaned_content:
        parts.append(cleaned_content)
    return "\n".join(parts)


def normalize_embedding_vector(vector: Any) -> Optional["np.ndarray"]:
    """Normalize a dense embedding so cosine similarity becomes a dot product."""
    if np is None or vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32)
    if array.size == 0:
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return None
    return array / norm


def serialize_embedding_vector(vector: Any) -> Optional[bytes]:
    """Store normalized embeddings compactly in SQLite."""
    normalized = normalize_embedding_vector(vector)
    if normalized is None:
        return None
    return normalized.astype(np.float16).tobytes()


def deserialize_embedding_vector(blob: Any) -> Optional["np.ndarray"]:
    """Decode one normalized embedding vector from SQLite."""
    if np is None or not blob:
        return None
    try:
        vector = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    except Exception:
        return None
    if vector.size == 0:
        return None
    return vector


def parse_pdfinfo_output(stdout: str) -> Dict[str, Any]:
    """Convert `pdfinfo` key-value output into structured metadata."""
    metadata: Dict[str, Any] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        cleaned_key = key.strip().lower().replace(" ", "_")
        cleaned_value = value.strip()
        if not cleaned_key:
            continue
        metadata[cleaned_key] = cleaned_value
    pages = metadata.get("pages")
    try:
        metadata["pages"] = int(str(pages).strip())
    except Exception:
        metadata["pages"] = None
    return metadata


def file_sample_looks_textual(target: pathlib.Path) -> bool:
    """Best-effort check for non-binary files without depending on extensions alone."""
    try:
        with target.open("rb") as f:
            sample = f.read(4096)
    except OSError:
        return False
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    return True


def extract_pdf_document(target: pathlib.Path) -> Dict[str, Any]:
    """Extract text and metadata from a PDF using local command-line tools."""
    if not PDFTOTEXT_BIN:
        raise ValueError("PDF text extraction is unavailable on this server")

    pdfinfo_meta: Dict[str, Any] = {}
    if PDFINFO_BIN:
        code, stdout, stderr = run_document_command([PDFINFO_BIN, str(target)])
        if code == 0:
            pdfinfo_meta = parse_pdfinfo_output(stdout)
        elif stderr.strip():
            logger.warning("pdfinfo failed for %s: %s", target, stderr.strip())

    code, stdout, stderr = run_document_command([PDFTOTEXT_BIN, "-layout", str(target), "-"])
    if code != 0:
        detail = stderr.strip() or "pdftotext failed"
        raise ValueError(detail)

    raw_pages = stdout.split("\f")
    pages = [clean_document_text(page) for page in raw_pages]
    while pages and not pages[-1]:
        pages.pop()
    full_text = "\n\n".join(page for page in pages if page).strip()
    return {
        "file_type": "pdf",
        "extractor": "pdftotext",
        "title": str(pdfinfo_meta.get("title") or "").strip(),
        "page_count": pdfinfo_meta.get("pages"),
        "pages": pages,
        "full_text": full_text,
        "metadata": pdfinfo_meta,
    }


def extract_text_document(target: pathlib.Path) -> Dict[str, Any]:
    """Read a text-like file and normalize it for downstream chunking."""
    if target.stat().st_size > DOCUMENT_TEXT_READ_LIMIT:
        raise ValueError(f"Text file too large to index (max {DOCUMENT_TEXT_READ_LIMIT // (1024 * 1024)}MB)")
    if not file_sample_looks_textual(target):
        raise ValueError("File does not look text-decodable")

    raw = target.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_document_text(raw)
    return {
        "file_type": "text",
        "extractor": "utf-8",
        "title": target.name,
        "page_count": None,
        "pages": [cleaned] if cleaned else [],
        "full_text": cleaned,
        "metadata": {
            "line_count": raw.count("\n") + (1 if raw else 0),
        },
    }


def extract_document_payload(target: pathlib.Path) -> Dict[str, Any]:
    """Extract a supported document into normalized text plus metadata."""
    if target.stat().st_size > DOCUMENT_INDEX_SIZE_LIMIT:
        raise ValueError(f"File too large to index (max {DOCUMENT_INDEX_SIZE_LIMIT // (1024 * 1024)}MB)")
    if is_pdf_path(target):
        return extract_pdf_document(target)
    if is_text_document_path(target) or file_sample_looks_textual(target):
        return extract_text_document(target)
    raise ValueError("Unsupported document type for text extraction")


def is_section_heading(text: str) -> bool:
    """Heuristic heading detector for chunking semi-structured text."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return False
    if len(cleaned) > 120:
        return False
    if cleaned.endswith((".", ";", ":", ",")) and len(cleaned.split()) > 8:
        return False
    words = cleaned.split()
    if len(words) > 14:
        return False
    if re.match(r"^(\d+(\.\d+){0,3}|[IVXLC]+|[A-Z])[\).\s-]+\S", cleaned):
        return True
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if not alpha_words:
        return False
    if cleaned.upper() == cleaned and len(alpha_words) <= 10:
        return True
    titleish = sum(1 for word in alpha_words if word[:1].isupper())
    return titleish >= max(1, int(len(alpha_words) * 0.7))


def split_chunk_text(text: str, target_chars: int = DOCUMENT_CHUNK_TARGET_CHARS) -> List[str]:
    """Split long text into smaller retrieval chunks using paragraph/sentence boundaries."""
    cleaned = clean_document_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= target_chars:
        return [cleaned]

    pieces: List[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", cleaned) if part.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    current = ""
    for paragraph in paragraphs:
        paragraph_pieces = [paragraph]
        if len(paragraph) > target_chars * 1.35:
            paragraph_pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", paragraph) if piece.strip()]
        for piece in paragraph_pieces:
            separator = "\n\n" if current else ""
            if current and len(current) + len(separator) + len(piece) > target_chars:
                pieces.append(current.strip())
                overlap = current[-DOCUMENT_CHUNK_OVERLAP_CHARS:].strip() if DOCUMENT_CHUNK_OVERLAP_CHARS > 0 else ""
                current = overlap if overlap and overlap != current else ""
                separator = "\n\n" if current else ""
            current = f"{current}{separator}{piece}".strip()

    if current:
        pieces.append(current.strip())
    return [piece for piece in pieces if piece]


def build_document_chunks(payload: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Create retrieval chunks with section-aware grouping and a safe fallback."""
    pages = payload.get("pages") or []
    full_text = clean_document_text(payload.get("full_text") or "")
    if not pages and full_text:
        pages = [full_text]
    if not pages:
        return [], []

    chunks: List[Dict[str, Any]] = []
    section_titles: List[str] = []
    active_section = ""
    chunk_index = 0

    for page_number, page_text in enumerate(pages, start=1):
        page_clean = clean_document_text(page_text)
        if not page_clean:
            continue
        blocks = [block.strip() for block in re.split(r"\n\s*\n+", page_clean) if block.strip()]
        if not blocks:
            blocks = [page_clean]

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            first_line = lines[0]
            block_section = active_section
            content = block
            if is_section_heading(first_line):
                block_section = first_line
                if block_section not in section_titles:
                    section_titles.append(block_section)
                remainder = clean_document_text("\n".join(lines[1:]))
                if remainder:
                    content = remainder
                    active_section = block_section
                else:
                    active_section = block_section
                    continue

            content_parts = split_chunk_text(content)
            for part in content_parts:
                section_name = block_section or active_section or f"Page {page_number}"
                chunks.append({
                    "chunk_index": chunk_index,
                    "page_start": page_number,
                    "page_end": page_number,
                    "section_title": section_name,
                    "content": part,
                    "metadata": {
                        "page": page_number,
                    },
                })
                chunk_index += 1

    if not chunks and full_text:
        for part in split_chunk_text(full_text):
            chunks.append({
                "chunk_index": chunk_index,
                "page_start": 1,
                "page_end": payload.get("page_count") or 1,
                "section_title": payload.get("title") or "Document",
                "content": part,
                "metadata": {},
            })
            chunk_index += 1

    return chunks, section_titles[:12]


def delete_document_index(conn: sqlite3.Connection, conversation_id: str, rel_path: str):
    """Remove all persisted chunks for a document before reindexing."""
    c = conn.cursor()
    chunk_ids = [
        row[0]
        for row in c.execute(
            'SELECT id FROM document_chunks WHERE conversation_id = ? AND path = ?',
            (conversation_id, rel_path),
        ).fetchall()
    ]
    if chunk_ids:
        try:
            c.executemany(f'DELETE FROM {DOCUMENT_FTS_TABLE} WHERE rowid = ?', [(chunk_id,) for chunk_id in chunk_ids])
        except sqlite3.OperationalError:
            pass
    c.execute('DELETE FROM document_chunks WHERE conversation_id = ? AND path = ?', (conversation_id, rel_path))
    c.execute('DELETE FROM document_sources WHERE conversation_id = ? AND path = ?', (conversation_id, rel_path))


def store_document_index(
    conn: sqlite3.Connection,
    conversation_id: str,
    rel_path: str,
    fingerprint: str,
    target: pathlib.Path,
    payload: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    section_titles: List[str],
    status: str,
    error: str = "",
):
    """Persist extracted document metadata and its retrieval chunks."""
    metadata = dict(payload.get("metadata") or {})
    metadata["section_titles"] = section_titles[:12]
    metadata["chunk_count"] = len(chunks)
    metadata.update(current_document_chunk_settings())
    indexed_at = utcnow_iso()
    c = conn.cursor()
    delete_document_index(conn, conversation_id, rel_path)
    c.execute(
        '''INSERT INTO document_sources
           (conversation_id, path, fingerprint, file_size, modified_at, file_type, extractor, page_count, title,
            metadata_json, status, error, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            conversation_id,
            rel_path,
            fingerprint,
            int(target.stat().st_size),
            datetime.fromtimestamp(target.stat().st_mtime).isoformat(),
            str(payload.get("file_type") or "file"),
            str(payload.get("extractor") or ""),
            payload.get("page_count"),
            str(payload.get("title") or ""),
            json.dumps(metadata),
            status,
            error,
            indexed_at,
        ),
    )
    chunk_embeddings: List[Optional[bytes]] = []
    if chunks and embedding_backend_active():
        vectors = embed_passages([
            build_document_chunk_embedding_text(
                rel_path,
                str(chunk.get("section_title") or ""),
                str(chunk.get("content") or ""),
            )
            for chunk in chunks
        ])
        chunk_embeddings = [None] * len(chunks)
        for index, vector in enumerate(vectors[: len(chunks)]):
            chunk_embeddings[index] = serialize_embedding_vector(vector)
    active_model = active_embedding_model_name() if embedding_backend_active() else ""
    for chunk in chunks:
        chunk_blob = chunk_embeddings[int(chunk.get("chunk_index", 0))] if chunk_embeddings else None
        c.execute(
            '''INSERT INTO document_chunks
               (conversation_id, path, chunk_index, page_start, page_end, section_title, content, char_count, metadata_json,
                embedding, embedding_model, embedded_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                conversation_id,
                rel_path,
                int(chunk.get("chunk_index", 0)),
                chunk.get("page_start"),
                chunk.get("page_end"),
                str(chunk.get("section_title") or ""),
                str(chunk.get("content") or ""),
                len(str(chunk.get("content") or "")),
                json.dumps(chunk.get("metadata") or {}),
                chunk_blob,
                active_model if chunk_blob else "",
                indexed_at if chunk_blob else None,
                indexed_at,
            ),
        )
        chunk_id = c.lastrowid
        try:
            c.execute(
                f'''INSERT INTO {DOCUMENT_FTS_TABLE}(rowid, content, section_title, path, conversation_id)
                    VALUES (?, ?, ?, ?, ?)''',
                (
                    chunk_id,
                    str(chunk.get("content") or ""),
                    str(chunk.get("section_title") or ""),
                    rel_path,
                    conversation_id,
                ),
            )
        except sqlite3.OperationalError:
            pass


def get_document_source_record(conversation_id: str, rel_path: str) -> Optional[Dict[str, Any]]:
    """Return persisted document-source metadata when available."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            '''SELECT path, fingerprint, file_size, modified_at, file_type, extractor, page_count, title,
                      metadata_json, status, error, indexed_at
               FROM document_sources
               WHERE conversation_id = ? AND path = ?''',
            (conversation_id, rel_path),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    metadata: Dict[str, Any]
    try:
        metadata = json.loads(row[8] or "{}")
    except Exception:
        metadata = {}
    return {
        "path": row[0],
        "fingerprint": row[1],
        "file_size": int(row[2] or 0),
        "modified_at": row[3] or "",
        "file_type": row[4] or "file",
        "extractor": row[5] or "",
        "page_count": row[6],
        "title": row[7] or "",
        "metadata": metadata,
        "status": row[9] or "ready",
        "error": row[10] or "",
        "indexed_at": row[11] or "",
    }


def ensure_document_index(conversation_id: str, rel_path: str) -> Dict[str, Any]:
    """Index a supported document on demand and reuse cached chunks when unchanged."""
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, rel_path)
    if not target.is_file():
        raise ValueError("File not found")
    if not is_supported_document_path(target) and not file_sample_looks_textual(target):
        raise ValueError("Unsupported document type")

    fingerprint = build_document_fingerprint(target)
    cached = get_document_source_record(conversation_id, format_workspace_path(target, workspace))
    if (
        cached
        and cached.get("fingerprint") == fingerprint
        and cached.get("status") in {"ready", "empty"}
        and cached_document_index_matches_settings(cached)
    ):
        return cached

    rel_file_path = format_workspace_path(target, workspace)
    payload = {
        "file_type": "file",
        "extractor": "",
        "title": target.name,
        "page_count": None,
        "metadata": {},
    }

    try:
        payload = extract_document_payload(target)
        chunks, section_titles = build_document_chunks(payload)
        status = "ready" if chunks else "empty"
        conn = sqlite3.connect(DB_PATH)
        try:
            store_document_index(
                conn,
                conversation_id,
                rel_file_path,
                fingerprint,
                target,
                payload,
                chunks,
                section_titles,
                status=status,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        conn = sqlite3.connect(DB_PATH)
        try:
            store_document_index(
                conn,
                conversation_id,
                rel_file_path,
                fingerprint,
                target,
                payload,
                [],
                [],
                status="error",
                error=str(exc),
            )
            conn.commit()
        finally:
            conn.close()
        raise

    return get_document_source_record(conversation_id, rel_file_path) or {
        "path": rel_file_path,
        "fingerprint": fingerprint,
        "file_size": int(target.stat().st_size),
        "modified_at": datetime.fromtimestamp(target.stat().st_mtime).isoformat(),
        "file_type": payload.get("file_type") or "file",
        "extractor": payload.get("extractor") or "",
        "page_count": payload.get("page_count"),
        "title": payload.get("title") or target.name,
        "metadata": {
            **(payload.get("metadata") or {}),
            **current_document_chunk_settings(),
        },
        "status": "ready",
        "error": "",
        "indexed_at": utcnow_iso(),
    }


def reciprocal_rank_fusion_bonus(rank: Optional[int], k: int = RETRIEVAL_RRF_K) -> float:
    """Convert a 0-based rank into a small, stable fusion bonus."""
    if rank is None:
        return 0.0
    return 1.0 / (max(1, int(k)) + max(0, int(rank)) + 1)


def exact_text_hit_bonus(query: str, *fields: str) -> float:
    """Reward literal matches so filenames, headings, and error strings stay strong."""
    lowered_query = " ".join(str(query or "").strip().lower().split())
    if len(lowered_query) < 2:
        return 0.0
    for field in fields:
        if lowered_query and lowered_query in str(field or "").lower():
            return 0.45
    return 0.0


def embed_message_rows(conn: sqlite3.Connection, rows: List[tuple]) -> int:
    """Embed persisted messages in batches and store the normalized vectors."""
    if not rows or not embedding_backend_active():
        return 0
    texts = [build_message_embedding_text(row[1], row[2]) for row in rows]
    vectors = embed_passages(texts)
    active_model = active_embedding_model_name()
    embedded_at = utcnow_iso()
    updates = []
    for row, vector in zip(rows, vectors):
        blob = serialize_embedding_vector(vector)
        if blob is None:
            continue
        updates.append((blob, active_model, embedded_at, int(row[0])))
    if updates:
        conn.executemany(
            'UPDATE messages SET embedding = ?, embedding_model = ?, embedded_at = ? WHERE id = ?',
            updates,
        )
    return len(updates)


def ensure_message_embeddings(conversation_id: str) -> int:
    """Backfill semantic embeddings for one conversation on demand."""
    if not embedding_backend_active():
        return 0
    active_model = active_embedding_model_name()
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            '''SELECT id, role, content
               FROM messages
               WHERE conversation_id = ?
                 AND TRIM(COALESCE(content, '')) != ''
                 AND (embedding IS NULL OR COALESCE(embedding_model, '') != ?)
               ORDER BY id ASC''',
            (conversation_id, active_model),
        ).fetchall()
        if not rows:
            return 0
        updated = embed_message_rows(conn, rows)
        if updated:
            conn.commit()
        return updated
    finally:
        conn.close()


def ensure_document_chunk_embeddings(conversation_id: str, rel_paths: List[str]) -> int:
    """Backfill semantic embeddings for indexed chunks on demand."""
    if not embedding_backend_active() or not rel_paths:
        return 0
    active_model = active_embedding_model_name()
    placeholders = ", ".join("?" for _ in rel_paths)
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            f'''SELECT id, path, section_title, content
                FROM document_chunks
                WHERE conversation_id = ?
                  AND path IN ({placeholders})
                  AND TRIM(COALESCE(content, '')) != ''
                  AND (embedding IS NULL OR COALESCE(embedding_model, '') != ?)
                ORDER BY path ASC, chunk_index ASC''',
            [conversation_id, *rel_paths, active_model],
        ).fetchall()
        if not rows:
            return 0
        vectors = embed_passages([
            build_document_chunk_embedding_text(row[1], row[2], row[3])
            for row in rows
        ])
        embedded_at = utcnow_iso()
        updates = []
        for row, vector in zip(rows, vectors):
            blob = serialize_embedding_vector(vector)
            if blob is None:
                continue
            updates.append((blob, active_model, embedded_at, int(row[0])))
        if updates:
            conn.executemany(
                'UPDATE document_chunks SET embedding = ?, embedding_model = ?, embedded_at = ? WHERE id = ?',
                updates,
            )
            conn.commit()
        return len(updates)
    finally:
        conn.close()


def fetch_semantic_message_candidates(
    conversation_id: str,
    query: str,
    limit: int = MESSAGE_RETRIEVAL_SEMANTIC_LIMIT,
) -> List[Dict[str, Any]]:
    """Return semantic nearest-neighbor candidates from one conversation."""
    if not embedding_backend_active():
        return []
    vectors = embed_queries([trim_embedding_text(query)])
    query_vector = normalize_embedding_vector(vectors[0]) if vectors else None
    if query_vector is None:
        return []
    ensure_message_embeddings(conversation_id)
    active_model = active_embedding_model_name()
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            '''SELECT id, role, content, timestamp, feedback, embedding
               FROM messages
               WHERE conversation_id = ?
                 AND embedding IS NOT NULL
                 AND COALESCE(embedding_model, '') = ?''',
            (conversation_id, active_model),
        ).fetchall()
    finally:
        conn.close()

    ranked = []
    for row in rows:
        vector = deserialize_embedding_vector(row[5])
        if vector is None or vector.shape != query_vector.shape:
            continue
        similarity = float(np.dot(query_vector, vector))
        if similarity <= 0.0:
            continue
        ranked.append({
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "timestamp": row[3],
            "feedback": message_feedback_value(row[1], row[4]),
            "semantic_similarity": round(similarity, 4),
        })
    ranked.sort(key=lambda item: (item.get("semantic_similarity", 0.0), item.get("timestamp", "")), reverse=True)
    return ranked[: max(1, int(limit))]


def fetch_semantic_document_candidates(
    conversation_id: str,
    rel_paths: List[str],
    query: str,
    limit: int = DOCUMENT_RETRIEVAL_SEMANTIC_LIMIT,
) -> List[Dict[str, Any]]:
    """Return semantic nearest-neighbor candidates from indexed attachment chunks."""
    if not embedding_backend_active() or not rel_paths:
        return []
    vectors = embed_queries([trim_embedding_text(query)])
    query_vector = normalize_embedding_vector(vectors[0]) if vectors else None
    if query_vector is None:
        return []
    ensure_document_chunk_embeddings(conversation_id, rel_paths)
    active_model = active_embedding_model_name()
    placeholders = ", ".join("?" for _ in rel_paths)
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            f'''SELECT id, path, chunk_index, page_start, page_end, section_title, content, embedding
                FROM document_chunks
                WHERE conversation_id = ?
                  AND path IN ({placeholders})
                  AND embedding IS NOT NULL
                  AND COALESCE(embedding_model, '') = ?''',
            [conversation_id, *rel_paths, active_model],
        ).fetchall()
    finally:
        conn.close()

    ranked = []
    for row in rows:
        vector = deserialize_embedding_vector(row[7])
        if vector is None or vector.shape != query_vector.shape:
            continue
        similarity = float(np.dot(query_vector, vector))
        if similarity <= 0.0:
            continue
        ranked.append({
            "id": row[0],
            "path": row[1],
            "chunk_index": int(row[2] or 0),
            "page_start": row[3],
            "page_end": row[4],
            "section_title": row[5] or "",
            "content": row[6] or "",
            "semantic_similarity": round(similarity, 4),
        })
    ranked.sort(key=lambda item: (item.get("semantic_similarity", 0.0), -(item.get("page_start") or 0)), reverse=True)
    return ranked[: max(1, int(limit))]


def fetch_document_chunk_rows(
    conversation_id: str,
    rel_path: str,
    start_index: int,
    end_index: int,
) -> List[Dict[str, Any]]:
    """Fetch an inclusive chunk window from the persisted document index."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            '''SELECT id, path, chunk_index, page_start, page_end, section_title, content
               FROM document_chunks
               WHERE conversation_id = ? AND path = ? AND chunk_index BETWEEN ? AND ?
               ORDER BY chunk_index ASC''',
            (conversation_id, rel_path, start_index, end_index),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": row[0],
            "path": row[1],
            "chunk_index": int(row[2] or 0),
            "page_start": row[3],
            "page_end": row[4],
            "section_title": row[5] or "",
            "content": row[6] or "",
        }
        for row in rows
    ]


def tokenize_retrieval_text(text: str) -> List[str]:
    """Normalize free text into lowercase retrieval tokens."""
    return [token for token in re.findall(r"[A-Za-z0-9_]+", str(text or "").lower()) if len(token) >= 2]


def compute_overlap_score(query_tokens: List[str], content: str) -> float:
    """Estimate retrieval relevance using token overlap."""
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize_retrieval_text(content))
    if not content_tokens:
        return 0.0
    query_token_set = set(query_tokens)
    overlap = len(query_token_set & content_tokens)
    return overlap / max(len(query_token_set), 1)


def fetch_ranked_document_candidates(
    conversation_id: str,
    rel_paths: List[str],
    query: str,
    hyde_query: str = "",
    limit: int = DOCUMENT_RETRIEVAL_FTS_LIMIT,
) -> List[Dict[str, Any]]:
    """Retrieve and rerank candidate chunks across indexed attachments."""
    if not rel_paths:
        return []

    conn = sqlite3.connect(DB_PATH)
    candidates: Dict[int, Dict[str, Any]] = {}
    query_tokens = tokenize_retrieval_text(query)
    hyde_tokens = tokenize_retrieval_text(hyde_query)
    placeholders = ", ".join("?" for _ in rel_paths)

    try:
        search_terms = []
        normalized_query = normalize_fts_query(query)
        normalized_hyde = normalize_fts_query(hyde_query)
        if normalized_query:
            search_terms.append(("query", normalized_query))
        if normalized_hyde and normalized_hyde != normalized_query:
            search_terms.append(("hyde", normalized_hyde))

        for label, search_term in search_terms:
            try:
                rows = conn.execute(
                    f'''SELECT dc.id, dc.path, dc.chunk_index, dc.page_start, dc.page_end, dc.section_title, dc.content,
                               bm25({DOCUMENT_FTS_TABLE}) AS fts_rank
                        FROM {DOCUMENT_FTS_TABLE}
                        JOIN document_chunks dc ON dc.id = {DOCUMENT_FTS_TABLE}.rowid
                        WHERE {DOCUMENT_FTS_TABLE} MATCH ?
                          AND dc.conversation_id = ?
                          AND dc.path IN ({placeholders})
                        ORDER BY fts_rank ASC
                        LIMIT ?''',
                    [search_term, conversation_id, *rel_paths, max(4, limit)],
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

            for position, row in enumerate(rows):
                item = candidates.setdefault(
                    row[0],
                    {
                        "id": row[0],
                        "path": row[1],
                        "chunk_index": int(row[2] or 0),
                        "page_start": row[3],
                        "page_end": row[4],
                        "section_title": row[5] or "",
                        "content": row[6] or "",
                        "query_rank": None,
                        "hyde_rank": None,
                        "semantic_query_rank": None,
                        "semantic_hyde_rank": None,
                        "semantic_query_similarity": None,
                        "semantic_hyde_similarity": None,
                        "fts_rank": row[7],
                    },
                )
                item[f"{label}_rank"] = position

        if not candidates:
            rows = conn.execute(
                f'''SELECT id, path, chunk_index, page_start, page_end, section_title, content
                    FROM document_chunks
                    WHERE conversation_id = ?
                      AND path IN ({placeholders})
                    ORDER BY path ASC, chunk_index ASC
                    LIMIT ?''',
                [conversation_id, *rel_paths, max(50, limit * 5)],
            ).fetchall()
            for row in rows:
                candidates[row[0]] = {
                    "id": row[0],
                    "path": row[1],
                    "chunk_index": int(row[2] or 0),
                    "page_start": row[3],
                    "page_end": row[4],
                    "section_title": row[5] or "",
                    "content": row[6] or "",
                    "query_rank": None,
                    "hyde_rank": None,
                    "semantic_query_rank": None,
                    "semantic_hyde_rank": None,
                    "semantic_query_similarity": None,
                    "semantic_hyde_similarity": None,
                    "fts_rank": None,
                }
    finally:
        conn.close()

    semantic_query_rows = fetch_semantic_document_candidates(
        conversation_id,
        rel_paths,
        query,
        limit=max(DOCUMENT_RETRIEVAL_SEMANTIC_LIMIT, limit * 2),
    ) if query else []
    semantic_hyde_rows = fetch_semantic_document_candidates(
        conversation_id,
        rel_paths,
        hyde_query,
        limit=max(DOCUMENT_RETRIEVAL_SEMANTIC_LIMIT // 2, limit),
    ) if hyde_query else []

    for position, row in enumerate(semantic_query_rows):
        item = candidates.setdefault(
            row["id"],
            {
                "id": row["id"],
                "path": row["path"],
                "chunk_index": int(row.get("chunk_index", 0)),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "section_title": row.get("section_title", ""),
                "content": row.get("content", ""),
                "query_rank": None,
                "hyde_rank": None,
                "semantic_query_rank": None,
                "semantic_hyde_rank": None,
                "semantic_query_similarity": None,
                "semantic_hyde_similarity": None,
                "fts_rank": None,
            },
        )
        item["semantic_query_rank"] = position
        item["semantic_query_similarity"] = row.get("semantic_similarity")

    for position, row in enumerate(semantic_hyde_rows):
        item = candidates.setdefault(
            row["id"],
            {
                "id": row["id"],
                "path": row["path"],
                "chunk_index": int(row.get("chunk_index", 0)),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "section_title": row.get("section_title", ""),
                "content": row.get("content", ""),
                "query_rank": None,
                "hyde_rank": None,
                "semantic_query_rank": None,
                "semantic_hyde_rank": None,
                "semantic_query_similarity": None,
                "semantic_hyde_similarity": None,
                "fts_rank": None,
            },
        )
        item["semantic_hyde_rank"] = position
        item["semantic_hyde_similarity"] = row.get("semantic_similarity")

    ranked: List[Dict[str, Any]] = []
    normalized_query_text = " ".join(query_tokens)
    normalized_hyde_text = " ".join(hyde_tokens)
    for item in candidates.values():
        content = item.get("content", "")
        path = item.get("path", "")
        section = item.get("section_title", "")
        score = 0.0
        score += 160.0 * reciprocal_rank_fusion_bonus(item.get("query_rank"))
        score += 110.0 * reciprocal_rank_fusion_bonus(item.get("hyde_rank"))
        score += 180.0 * reciprocal_rank_fusion_bonus(item.get("semantic_query_rank"))
        score += 120.0 * reciprocal_rank_fusion_bonus(item.get("semantic_hyde_rank"))
        score += 1.4 * max(0.0, float(item.get("semantic_query_similarity") or 0.0))
        score += 0.8 * max(0.0, float(item.get("semantic_hyde_similarity") or 0.0))
        score += 0.9 * compute_overlap_score(query_tokens, f"{section}\n{content}")
        score += 0.55 * compute_overlap_score(hyde_tokens, f"{section}\n{content}")
        score += 0.35 * compute_overlap_score(query_tokens, path)
        if normalized_query_text and normalized_query_text in " ".join(tokenize_retrieval_text(content)):
            score += 0.35
        if normalized_query_text and normalized_query_text in " ".join(tokenize_retrieval_text(section)):
            score += 0.2
        score += exact_text_hit_bonus(query, path, section, content)
        if query:
            score += 0.2 * SequenceMatcher(None, query[:180].lower(), content[:260].lower()).ratio()
        if hyde_query:
            score += 0.1 * SequenceMatcher(None, normalized_hyde_text[:180], content[:260].lower()).ratio()
        if len(content) < 120:
            score -= 0.12
        ranked.append({**item, "score": round(score, 4)})

    ranked.sort(key=lambda row: (row.get("score", 0.0), -(row.get("page_start") or 0)), reverse=True)
    return ranked[: max(1, limit)]


def build_document_overview_entries(conversation_id: str, rel_paths: List[str], char_budget: int) -> List[Dict[str, Any]]:
    """Build a compact fallback overview for indexed attachments."""
    conn = sqlite3.connect(DB_PATH)
    entries: List[Dict[str, Any]] = []
    remaining = char_budget
    try:
        placeholders = ", ".join("?" for _ in rel_paths)
        source_rows = conn.execute(
            f'''SELECT path, page_count, title, metadata_json
                FROM document_sources
                WHERE conversation_id = ?
                  AND path IN ({placeholders})
                  AND status = 'ready'
                ORDER BY path ASC''',
            [conversation_id, *rel_paths],
        ).fetchall()
        for path, page_count, title, metadata_json in source_rows:
            try:
                metadata = json.loads(metadata_json or "{}")
            except Exception:
                metadata = {}
            section_titles = [str(item).strip() for item in metadata.get("section_titles", []) if str(item).strip()]
            first_chunk = conn.execute(
                '''SELECT page_start, page_end, section_title, content
                   FROM document_chunks
                   WHERE conversation_id = ? AND path = ?
                   ORDER BY chunk_index ASC
                   LIMIT 1''',
                (conversation_id, path),
            ).fetchone()
            preview = ""
            preview_section = ""
            preview_page_start = None
            preview_page_end = None
            if first_chunk:
                preview_page_start, preview_page_end, preview_section, preview = first_chunk
            header_parts = [path]
            if title and title != path:
                header_parts.append(f"title: {title}")
            if page_count:
                header_parts.append(f"pages: {page_count}")
            if section_titles:
                header_parts.append(f"sections: {', '.join(section_titles[:5])}")
            block_lines = [" | ".join(header_parts)]
            if preview:
                label = preview_section or "Opening passage"
                if preview_page_start:
                    page_label = f"pages {preview_page_start}" if preview_page_start == preview_page_end else f"pages {preview_page_start}-{preview_page_end}"
                    label = f"{label} ({page_label})"
                block_lines.extend([label, truncate_output(preview, limit=1400)])
            block = "\n".join(block_lines)
            if len(block) > remaining and entries:
                break
            entries.append({"path": path, "text": truncate_output(block, limit=max(remaining, 400))})
            remaining -= len(entries[-1]["text"]) + 2
            if remaining < 300:
                break
    finally:
        conn.close()
    return entries


def pack_document_context_windows(
    conversation_id: str,
    ranked_candidates: List[Dict[str, Any]],
    char_budget: int = DOCUMENT_RETRIEVAL_CONTEXT_BUDGET,
    max_windows: int = DOCUMENT_RETRIEVAL_MAX_WINDOWS,
) -> List[Dict[str, Any]]:
    """Expand top-ranked chunks with local neighbors and fit them into a prompt budget."""
    windows: List[Dict[str, Any]] = []
    used_chunk_ids: set[int] = set()
    remaining = char_budget

    for candidate in ranked_candidates:
        if len(windows) >= max_windows or remaining < 400:
            break
        center_index = int(candidate.get("chunk_index", 0))
        rows = fetch_document_chunk_rows(
            conversation_id,
            str(candidate.get("path") or ""),
            max(center_index - 1, 0),
            center_index + 1,
        )
        rows = [row for row in rows if row.get("id") not in used_chunk_ids]
        if not rows:
            continue

        used_chunk_ids.update(int(row["id"]) for row in rows)
        page_start = rows[0].get("page_start")
        page_end = rows[-1].get("page_end")
        section_label = candidate.get("section_title") or rows[0].get("section_title") or "Relevant excerpt"
        page_label = ""
        if page_start:
            page_label = f"pages {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
        header = f"{candidate.get('path')} | {section_label}"
        if page_label:
            header = f"{header} | {page_label}"
        combined_text = "\n\n".join(str(row.get("content") or "").strip() for row in rows if str(row.get("content") or "").strip())
        block = f"{header}\n{truncate_output(combined_text, limit=min(2600, max(remaining - len(header) - 1, 600)))}"
        if len(block) > remaining and windows:
            continue
        windows.append({
            "path": candidate.get("path"),
            "header": header,
            "text": block,
            "score": candidate.get("score", 0.0),
        })
        remaining -= len(block) + 2

    return windows


def is_overview_attachment_request(message: str) -> bool:
    """Detect high-level requests that benefit from overview snippets even without strong matches."""
    text = str(message or "").lower()
    signals = (
        "summarize", "summary", "overview", "what is this", "what's this",
        "main point", "main points", "high level", "high-level", "tl;dr", "key takeaways",
    )
    return any(signal in text for signal in signals)


async def generate_hyde_query(message: str, paths: List[str]) -> str:
    """Create a concise hypothetical answer passage to improve retrieval recall."""
    question = str(message or "").strip()
    if not question:
        return ""
    prompt = (
        "Write a short hypothetical excerpt from the attached documents that would likely answer the user's question. "
        "Stay factual and concise, 2-4 sentences max, and do not mention that this is hypothetical.\n\n"
        f"Files: {', '.join(paths[:4])}\n"
        f"Question: {question}"
    )
    messages = [
        {"role": "system", "content": "You write compact retrieval probes for document search."},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = await vllm_chat_complete(messages, max_tokens=160, temperature=0.1)
    except Exception as exc:
        logger.warning("HyDE generation failed: %s", exc)
        return ""
    return truncate_output(strip_stream_special_tokens(raw).strip(), limit=500)


def format_document_source_line(source: Dict[str, Any]) -> str:
    """Render concise source metadata for prompt grounding."""
    path = str(source.get("path") or "")
    metadata = source.get("metadata") or {}
    parts = [path]
    file_type = str(source.get("file_type") or "").strip()
    if file_type:
        parts.append(file_type)
    page_count = source.get("page_count")
    if page_count:
        parts.append(f"{page_count} pages")
    section_titles = [str(item).strip() for item in metadata.get("section_titles", []) if str(item).strip()]
    if section_titles:
        parts.append(f"sections: {', '.join(section_titles[:4])}")
    return " | ".join(parts)


async def build_attachment_context(conversation_id: str, paths: List[str], message: str) -> str:
    """Index attachments, retrieve the most relevant chunks, and render a compact prompt block."""
    cleaned = []
    for path in paths[:MAX_ATTACHMENTS_PER_MESSAGE]:
        candidate = str(path or "").strip()
        if candidate and candidate not in cleaned:
            cleaned.append(candidate)
    if not cleaned:
        return ""

    indexed_sources: List[Dict[str, Any]] = []
    notes: List[str] = []
    for rel_path in cleaned:
        try:
            source = ensure_document_index(conversation_id, rel_path)
        except Exception as exc:
            notes.append(f"- {rel_path}: {exc}")
            continue
        indexed_sources.append(source)
        if source.get("status") == "empty":
            notes.append(f"- {rel_path}: no extractable text was found")
        elif source.get("status") == "error":
            notes.append(f"- {rel_path}: {source.get('error') or 'indexing failed'}")

    ready_paths = [str(source.get("path") or "") for source in indexed_sources if source.get("status") == "ready"]
    lines = ["Attached files are available in the workspace:"]
    if indexed_sources:
        lines.extend(f"- {format_document_source_line(source)}" for source in indexed_sources[:MAX_ATTACHMENTS_PER_MESSAGE])
    else:
        lines.extend(f"- {path}" for path in cleaned)

    if ready_paths:
        hyde_query = "" if embedding_backend_active() else await generate_hyde_query(message, ready_paths)
        ranked = fetch_ranked_document_candidates(conversation_id, ready_paths, message, hyde_query=hyde_query)
        windows = pack_document_context_windows(conversation_id, ranked)
        if windows:
            lines.extend(["", "Relevant extracted context from attachments:"])
            lines.extend(f"---\n{window['text']}" for window in windows)
        if (not windows or is_overview_attachment_request(message)) and ready_paths:
            overview_entries = build_document_overview_entries(
                conversation_id,
                ready_paths,
                char_budget=max(1800, DOCUMENT_RETRIEVAL_CONTEXT_BUDGET // 2),
            )
            if overview_entries:
                lines.extend(["", "Attachment overview:"])
                lines.extend(f"---\n{entry['text']}" for entry in overview_entries)

    if notes:
        lines.extend(["", "Attachment parsing notes:"])
        lines.extend(notes[:MAX_ATTACHMENTS_PER_MESSAGE])

    return "\n".join(lines)


def build_document_preview_result(
    target: pathlib.Path,
    conversation_id: Optional[str] = None,
    rel_path: Optional[str] = None,
    limit: int = TOOL_RESULT_TEXT_LIMIT,
) -> Dict[str, Any]:
    """Return a text preview for a supported document, using the index when possible."""
    if conversation_id and rel_path:
        source = ensure_document_index(conversation_id, rel_path)
        if source.get("status") == "error":
            raise ValueError(source.get("error") or "Document indexing failed")
        chunk_rows = fetch_document_chunk_rows(conversation_id, rel_path, 0, 2)
        preview_text = "\n\n".join(str(row.get("content") or "").strip() for row in chunk_rows if str(row.get("content") or "").strip())
        preview_text = truncate_output(preview_text, limit=limit)
        metadata = source.get("metadata") or {}
        return {
            "path": rel_path,
            "content": preview_text,
            "size": len(preview_text),
            "lines": preview_text.count("\n") + 1 if preview_text else 0,
            "file_type": source.get("file_type") or "file",
            "extractor": source.get("extractor") or "",
            "page_count": source.get("page_count"),
            "title": source.get("title") or target.name,
            "metadata": metadata,
        }

    payload = extract_document_payload(target)
    preview_text = truncate_output(payload.get("full_text") or "", limit=limit)
    return {
        "path": target.name,
        "content": preview_text,
        "size": len(preview_text),
        "lines": preview_text.count("\n") + 1 if preview_text else 0,
        "file_type": payload.get("file_type") or "file",
        "extractor": payload.get("extractor") or "",
        "page_count": payload.get("page_count"),
        "title": payload.get("title") or target.name,
        "metadata": payload.get("metadata") or {},
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


WORKSPACE_RESPONSE_TRUTHFULNESS_RULES = """Workspace and tool truthfulness rules:
- Never claim a workspace file was created, saved, or updated unless this turn actually completed a successful `workspace.patch_file` call.
- If no workspace file was written, provide the result inline and optionally offer to save it.
- Never claim you searched the web, looked something up, or added sourced citations unless this turn actually used the web tools.
- If the prompt already includes extracted attachment context, treat that content as available instead of claiming you cannot read the attachment.
- If command execution is available for this turn, do not claim you cannot execute code, install packages, convert files, or inspect command output here. Use `workspace.run_command`, or explain that approval was denied only when that actually happened.
- If the user asks to render or preview HTML and a render tool is available, prefer using it over claiming you cannot display pages.
- If the user asks about earlier chat context and conversation search is available, prefer using it over guessing from memory."""


WORKSPACE_WRITE_CLAIM_PATTERN = re.compile(
    r"\b(created?|saved?|wrote|written|updated?|patched?|added|put|placed)\b",
    re.IGNORECASE,
)
WORKSPACE_WRITE_CLAIM_HINT_PATTERN = re.compile(
    r"\b(workspace|file|folder|directory|artifact|draft)\b",
    re.IGNORECASE,
)


def normalize_workspace_reference_path(path_value: str) -> str:
    """Normalize a workspace-relative path so user-visible claims can be checked safely."""
    return str(path_value or "").strip().replace("\\", "/").strip("./").lower()


def successful_workspace_write_paths(tool_results: List[Dict[str, Any]]) -> set[str]:
    """Collect workspace paths that were actually written during the current turn."""
    written_paths: set[str] = set()
    for entry in tool_results:
        call = entry.get("call", {})
        result = entry.get("result", {})
        if call.get("name") != "workspace.patch_file" or not result.get("ok"):
            continue
        payload = result.get("result", {})
        if not isinstance(payload, dict):
            continue
        normalized = normalize_workspace_reference_path(str(payload.get("path", "")))
        if normalized:
            written_paths.add(normalized)
    return written_paths


def strip_unverified_workspace_write_claims(message: str, tool_results: Optional[List[Dict[str, Any]]] = None) -> str:
    """Remove file-write claims that are not backed by a successful workspace write tool result."""
    cleaned = str(message or "").strip()
    if not cleaned:
        return ""

    written_paths = successful_workspace_write_paths(tool_results or [])
    filtered_lines: List[str] = []
    removed_any = False

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or not WORKSPACE_WRITE_CLAIM_PATTERN.search(line):
            filtered_lines.append(raw_line)
            continue

        line_paths = {
            normalize_workspace_reference_path(path_ref)
            for path_ref in extract_workspace_path_references(line)
        }
        line_paths.discard("")
        has_workspace_claim_hint = bool(line_paths) or bool(WORKSPACE_WRITE_CLAIM_HINT_PATTERN.search(line))

        if not has_workspace_claim_hint:
            filtered_lines.append(raw_line)
            continue

        if line_paths:
            if written_paths.intersection(line_paths):
                filtered_lines.append(raw_line)
                continue
            removed_any = True
            continue

        if written_paths:
            filtered_lines.append(raw_line)
            continue

        removed_any = True

    if not removed_any:
        return cleaned

    repaired = "\n".join(filtered_lines)
    repaired = re.sub(r"\n{3,}", "\n\n", repaired).strip()
    if repaired:
        return repaired
    return (
        "I didn't actually create a workspace file in this turn. "
        "If you want, I can save the result once workspace editing is enabled."
    )


def truncate_output(text: str, limit: int = COMMAND_OUTPUT_LIMIT) -> str:
    """Trim long command output while making truncation explicit."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} chars]"


def normalize_pause_reason(value: Any) -> str:
    """Normalize persisted pause reasons into a small stable set."""
    return normalize_pause_reason_helper(value)


def build_tool_loop_hard_limit_message(progress_summary: str) -> str:
    """Render the user-facing resume note after exhausting the automatic tool budget."""
    return build_tool_loop_hard_limit_message_helper(progress_summary)


def build_tool_system_prompt(base_system_prompt: str) -> str:
    """Combine the base assistant prompt with the tool protocol and supported tools."""
    tool_schema = """Available tools:
- workspace.list_files {"path":"."}
- workspace.grep {"query":"FeatureFlags","path":".","glob":"*.py","limit":20}
- workspace.read_file {"path":"src/python/harness.py"}
- workspace.patch_file {"path":"src/python/harness.py","edits":[{"old_text":"before","new_text":"after","expected_count":1}]}
- workspace.patch_file {"path":"notes/todo.txt","create":true,"new_content":"hello"}
- workspace.run_command {"command":["python3","main.py"],"cwd":"."}
- workspace.run_command {"command":["pip","install","pandas"],"cwd":"."}
- workspace.render {"html":"<html><body><h1>Dashboard</h1></body></html>","title":"dashboard"}
- spreadsheet.describe {"path":"model.xlsx","sheet":"Assumptions"}
- conversation.search_history {"query":"retry logic","limit":5}
- web.search {"query":"python async docs","limit":5}
- web.search {"query":"history of C programming language","domains":["wikipedia.org"],"limit":5}
- web.fetch_page {"url":"https://en.wikipedia.org/wiki/Python_(programming_language)"}

Constraints:
- Paths are always relative to the current conversation workspace.
- workspace.list_files hides dot-prefixed files unless you explicitly target a hidden path.
- workspace.grep searches text files in the workspace and returns matching file paths and lines.
- workspace.patch_file uses exact-match replacements; prefer small edits.
- workspace.run_command takes an argv array, never a shell string.
- For Python dependencies, use pip installs normally; the server will place them into a managed chat-scoped Python environment outside the workspace and make later Python commands use it when available.
- When choosing Python packages or troubleshooting installs, use web.search or web.fetch_page to verify current package names, docs, and examples when helpful.
- If workspace.run_command is available for this turn, use it instead of claiming you cannot run code, install packages, convert files, or inspect real command output.
- workspace.render displays HTML in the workspace viewer panel; use it when the user asks to preview, render, or display HTML content.
- When you generate a chart, plot, screenshot, PDF, or other visual result, save it as a workspace file such as PNG, SVG, HTML, or PDF so the UI can surface it as an artifact.
- spreadsheet.describe is for CSV, TSV, and Excel inspection.
- conversation.search_history searches the current conversation only.
- web.search is for fresh or explicitly web-based questions, supports optional domain filters, and by default checks the general web plus Wikipedia and Reddit result sets.
- web.fetch_page reads a web page after search and returns normalized domain/content metadata for citation-ready summaries.
- The context window is limited; use workspace files, artifacts, and task boards as durable memory when they help.
- A single tool call can be enough if it creates durable progress, retrieval-ready context, or verification evidence.
- Some tools may pause for inline user approval. If approval is denied, treat that capability as blocked for this turn and wait for the user to approve and resume.
- When the user asks for a concrete output shape such as a PDF, chart, rendered page, runnable app, mobile-ready fix, or real command output, keep going until that exact shape is delivered or a blocker is verified.
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
    """Build the active system prompt for a turn."""
    del user_message
    return "\n\n".join(part for part in (base_system_prompt.strip(), WORKSPACE_RESPONSE_TRUTHFULNESS_RULES) if part)


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
    task_request: str = ""
    context: str = ""
    workspace_enabled: bool = False
    workspace_facts: str = ""
    workspace_snapshot: Dict[str, Any] = field(default_factory=dict)
    recent_product_feedback_entries: List[Dict[str, Any]] = field(default_factory=list)
    recent_product_feedback_summary: str = ""
    recent_product_feedback_artifact_path: str = ".ai/recent-feedback.md"
    plan: Dict[str, Any] = field(default_factory=dict)
    plan_preview_pending: bool = False
    task_board_path: str = ".ai/task-board.md"
    task_state_path: str = ".ai/task-state.json"
    build_step_summaries: List[str] = field(default_factory=list)
    step_subplans: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step_substep_summaries: Dict[str, List[str]] = field(default_factory=dict)
    step_substep_reports: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    step_reports: List[Dict[str, Any]] = field(default_factory=list)
    build_summary: str = ""
    changed_files: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    verification_summary: str = ""
    scope_audit: Dict[str, Any] = field(default_factory=dict)
    scope_audit_summary: str = ""
    draft_response: str = ""
    final_response: str = ""
    execution_requested: bool = False
    auto_execute: bool = False
    plan_override_builder_steps: List[str] = field(default_factory=list)
    resumed: bool = False
    pause_reason: str = ""
    workflow_execution: Optional[WorkflowExecutionContext] = None

    def history_messages(self) -> List[Dict[str, str]]:
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.history]


@dataclass
class ToolLoopOutcome:
    """Result of a server-side tool loop."""
    final_text: str
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    hit_limit: bool = False
    requested_phase_upgrade: str = ""
    requested_tool_name: str = ""
    blocked_on_permission: bool = False
    blocked_permission_key: str = ""


@dataclass
class DeepBuildResult:
    """State returned after attempting the planned build steps."""
    summary: str
    needs_user_confirmation: bool = False
    build_complete: bool = False
    pause_reason: str = ""


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
    features: Optional["FeatureFlags"] = None,
) -> ToolLoopOutcome:
    """Apply citation repair when fetched web pages informed the final answer."""
    sources = fetched_web_sources(tool_results)
    cleaned = str(final_text or "").strip()
    if sources and not answer_has_required_source_grounding(cleaned, sources):
        cleaned = await repair_answer_with_citations(cleaned, sources, max_tokens)
    leaked_call = extract_leaked_tool_call(cleaned)
    if leaked_call:
        cleaned = format_leaked_tool_call_message(leaked_call, features or FeatureFlags())
    cleaned = strip_unverified_workspace_write_claims(cleaned, tool_results)
    return ToolLoopOutcome(final_text=cleaned, tool_results=tool_results)

@dataclass
class FeatureFlags:
    """Per-request feature switches from the UI."""
    agent_tools: bool = True
    workspace_write: bool = False
    workspace_run_commands: bool = False
    local_rag: bool = True
    web_search: bool = False
    auto_approve_tool_permissions: bool = False
    allowed_commands: List[str] = field(default_factory=list)
    allowed_tool_permissions: List[str] = field(default_factory=list)


@dataclass
class PreparedTurnRequest:
    """Normalized routing state for one user turn before execution starts."""
    conversation_id: str
    user_message_id: int
    saved_user_message: str
    effective_message: str
    history: List[Dict[str, str]]
    system_prompt: str
    requested_mode: str
    resolved_mode: str
    features: FeatureFlags
    slash_command: Optional[Dict[str, str]]
    max_tokens: int
    workspace_intent: str
    enabled_tools: List[str]
    auto_execute_workspace: bool
    resume_saved_workspace: bool
    plan_override_builder_steps: List[str]
    promoted_to_planning: bool
    repo_bootstrapped: bool
    repo_bootstrap_summary: str
    assessment: TurnAssessment


DIRECT_SLASH_COMMAND_ALIASES = {
    "search": "search",
    "web": "search",
    "grep": "grep",
    "plan": "plan",
    "code": "code",
    "edit": "code",
    "pip": "pip",
}


def normalize_direct_slash_command(name: str) -> str:
    """Map supported slash aliases to their canonical command name."""
    return DIRECT_SLASH_COMMAND_ALIASES.get(str(name or "").strip().lower(), "")


def parse_direct_slash_command_payload(payload: Any) -> Optional[Dict[str, str]]:
    """Validate a structured slash command payload from the client."""
    if not isinstance(payload, dict):
        return None
    name = normalize_direct_slash_command(payload.get("name") or payload.get("raw_name") or "")
    if not name:
        return None
    return {
        "name": name,
        "raw_name": str(payload.get("raw_name") or payload.get("name") or "").strip().lower(),
        "args": str(payload.get("args") or "").strip(),
    }


def infer_direct_slash_command_from_message(message: str) -> Optional[Dict[str, str]]:
    """Fallback parser so typed slash commands still work without explicit client metadata."""
    match = re.match(r"^\s*/([a-z0-9_-]+)(?:\s+([\s\S]*\S))?\s*$", str(message or ""), re.IGNORECASE)
    if not match:
        return None
    name = normalize_direct_slash_command(match.group(1))
    if not name:
        return None
    return {
        "name": name,
        "raw_name": str(match.group(1) or "").strip().lower(),
        "args": str(match.group(2) or "").strip(),
    }


async def wait_for_command_approval(
    websocket: WebSocket,
    conversation_id: str,
    command: List[str],
    command_key: str,
    cwd: str = ".",
    step_label: Optional[str] = None,
) -> bool:
    """Compatibility wrapper for command-specific approvals."""
    request = PermissionApprovalRequest(
        key=command_key or "command",
        approval_target="command",
        title=f"Allow {pathlib.Path(str(command[0] or 'command')).name}?",
        content=f"The assistant wants to run {command_preview_text(command)} in {tool_path_preview(cwd or '.', '.')}.",
        preview=command_preview_text(command),
    )
    return await wait_for_permission_approval(
        websocket,
        conversation_id,
        request,
        step_label=step_label,
    )


@dataclass
class PermissionApprovalRequest:
    """One resumable inline approval request for a tool capability."""
    key: str
    approval_target: str
    title: str
    content: str
    preview: str = ""
    allow_label: str = "Approve and continue"
    deny_label: str = "Pause task"


def build_tool_permission_request(
    conversation_id: str,
    call: Dict[str, Any],
) -> Optional[PermissionApprovalRequest]:
    """Return the approval request needed before executing a gated tool."""
    name = str(call.get("name", "")).strip()
    arguments = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}

    if name == "workspace.run_command":
        command = arguments.get("command")
        command_list = command if isinstance(command, list) else []
        if not command_list:
            return None
        cwd_value = str(arguments.get("cwd", ".") or ".")
        try:
            cwd_path = resolve_workspace_relative_path(conversation_id, cwd_value)
            key = command_permission_key(conversation_id, command_list, cwd_path)
        except Exception:
            return None
        pip_install = parse_pip_install_command(command_list)
        if pip_install:
            preview = command_preview_text(command_list)
            return PermissionApprovalRequest(
                key=key,
                approval_target="command",
                title="Allow pip install?",
                content=(
                    f"The assistant wants to install {pip_install.get('packages_preview', 'the requested Python packages')} "
                    "into this chat's managed Python environment."
                ),
                preview=preview,
            )
        python_venv = parse_python_venv_command(command_list)
        if python_venv:
            preview = command_preview_text(command_list)
            venv_target = tool_path_preview(python_venv.get("target", ".venv"), ".venv")
            return PermissionApprovalRequest(
                key=key,
                approval_target="command",
                title="Allow Python venv setup?",
                content=f"The assistant wants to create a Python virtual environment at {venv_target} for this chat.",
                preview=preview,
            )
        executable = pathlib.Path(str(command_list[0] or "command")).name or "command"
        preview = command_preview_text(command_list)
        return PermissionApprovalRequest(
            key=key,
            approval_target="command",
            title=f"Allow {executable}?",
            content=f"The assistant wants to run {preview} in {tool_path_preview(cwd_value, '.')} for this chat.",
            preview=preview,
        )

    if name in {"workspace.list_files", "workspace.read_file", "spreadsheet.describe"}:
        path_preview = tool_path_preview(arguments.get("path", "."), ".")
        return PermissionApprovalRequest(
            key="tool:workspace",
            approval_target="tool",
            title="Use workspace files?",
            content=f"The assistant wants to inspect {path_preview} in this chat's workspace to ground the answer.",
            preview=path_preview,
        )

    if name == "workspace.grep":
        query = compact_tool_text(arguments.get("query", ""), limit=72) or "text"
        target = tool_path_preview(arguments.get("path", "."), ".")
        return PermissionApprovalRequest(
            key="tool:workspace.grep",
            approval_target="tool",
            title="Search the workspace?",
            content=f"The assistant wants to search {target} for {query}.",
            preview=query,
        )

    if name in {"workspace.patch_file", "workspace.render"}:
        path_preview = tool_path_preview(arguments.get("path", arguments.get("title", "workspace file")), "workspace file")
        return PermissionApprovalRequest(
            key="tool:workspace.write",
            approval_target="tool",
            title="Edit workspace files?",
            content=(
                f"The assistant wants to update {path_preview} in this chat's workspace."
                if name == "workspace.patch_file"
                else f"The assistant wants to write an HTML preview for {path_preview} in this chat's workspace."
            ),
            preview=path_preview,
        )

    if name in {"web.search", "web.fetch_page"}:
        preview = (
            compact_tool_text(arguments.get("query", ""), limit=96)
            if name == "web.search"
            else compact_tool_text(arguments.get("url", ""), limit=96)
        ) or ("the requested query" if name == "web.search" else "the requested page")
        return PermissionApprovalRequest(
            key="tool:web.search",
            approval_target="tool",
            title="Use web search?",
            content=(
                f'The assistant wants to search the web for "{preview}".'
                if name == "web.search"
                else f"The assistant wants to open {preview} to verify the answer against a live source."
            ),
            preview=preview,
        )

    return None


def render_permission_blocked_message(request: PermissionApprovalRequest) -> str:
    """Return the user-facing pause message after a required approval is not granted."""
    approval_kind = "command" if request.approval_target == "command" else "tool"
    title = str(request.title or "Approval required").strip().rstrip("?")
    content = str(request.content or "").strip().rstrip(".")
    lines = [
        "I paused here because this task needs an approval before I can continue.",
        "",
        f"Needed {approval_kind} approval: {title}.",
    ]
    if content:
        lines.append(content + ".")
    if request.preview:
        lines.extend([
            "",
            f"Request details: {request.preview}",
        ])
    lines.extend([
        "",
        "Approve it for this chat and then say continue to resume the task.",
    ])
    return "\n".join(lines)


async def wait_for_permission_approval(
    websocket: WebSocket,
    conversation_id: str,
    request: PermissionApprovalRequest,
    *,
    step_label: Optional[str] = None,
) -> bool:
    """Pause the active tool loop until the user approves or denies a gated tool."""
    client_turn_id = str(getattr(websocket, "active_client_turn_id", "") or "").strip()
    if getattr(websocket, "supports_command_approval", True) is False:
        await websocket.send_json({
            "type": "permission_required",
            "conversation_id": conversation_id,
            "client_turn_id": client_turn_id,
            "permission_key": request.key,
            "approval_target": request.approval_target,
            "title": request.title,
            "content": request.content,
            "preview": request.preview,
            "allow_label": request.allow_label,
            "deny_label": request.deny_label,
        })
        await send_activity_event(
            websocket,
            "blocked",
            "Blocked",
            f"Cannot request approval for '{request.key or 'tool'}' over HTTP fallback; denying it.",
            step_label=step_label,
        )
        return False

    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    PERMISSION_APPROVAL_WAITERS[conversation_id] = {
        "future": future,
        "permission_key": request.key,
        "approval_target": request.approval_target,
        "client_turn_id": client_turn_id,
        "websocket_id": id(websocket),
    }
    try:
        await websocket.send_json({
            "type": "permission_required",
            "conversation_id": conversation_id,
            "client_turn_id": client_turn_id,
            "permission_key": request.key,
            "approval_target": request.approval_target,
            "title": request.title,
            "content": request.content,
            "preview": request.preview,
            "allow_label": request.allow_label,
            "deny_label": request.deny_label,
        })
        await send_activity_event(
            websocket,
            "blocked",
            "Blocked",
            f"Waiting for approval: {request.content}",
            step_label=step_label,
        )
        return await future
    finally:
        current = PERMISSION_APPROVAL_WAITERS.get(conversation_id)
        if current and current.get("future") is future:
            PERMISSION_APPROVAL_WAITERS.pop(conversation_id, None)


class PatchApplicationError(ValueError):
    """Structured patch failure that can be surfaced back to the model."""

    def __init__(self, message: str, details: Dict[str, Any]):
        super().__init__(message)
        self.details = details


PLAN_REQUEST_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "always", "because", "been",
    "before", "being", "below", "between", "brief", "build", "change", "changes",
    "check", "could", "deliverable", "does", "done", "each", "every", "exact", "feature",
    "final", "first", "from", "have", "into", "just", "keep", "look", "looks", "make",
    "next", "only", "other", "plan", "please", "relevant", "request", "result", "should",
    "show", "specific", "specifics", "still", "surface", "surfaces", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "turn", "using", "what",
    "when", "where", "which", "with", "within", "work", "would",
}

PLAN_REQUEST_SHORT_TOKENS = {
    "ai", "api", "app", "bug", "ci", "css", "db", "dx", "fix", "go", "js", "md", "ml",
    "qa", "ts", "ui", "ux",
}

GENERIC_PLAN_PATTERNS = (
    "inspect the most relevant",
    "inspect the relevant",
    "inspect relevant",
    "build or revise",
    "build the main deliverable",
    "make the highest-leverage",
    "re-read the result",
    "run a focused check",
    "extract the exact task",
    "draft the answer",
    "tighten the draft",
    "do the most useful concrete work",
    "implement the next concrete slice",
    "validate the slice",
    "grounded in the workspace",
    "grounded in inspected evidence",
    "inspect, build, verify",
)


def summarize_request_for_plan(message: str, limit: int = 96) -> str:
    """Return a short request-focused snippet to ground planner text."""
    cleaned = " ".join((message or "").strip().split()).strip(" .")
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)].rstrip(' ,.;:')}..."


def extract_request_terms_for_plan(message: str, limit: int = 12) -> List[str]:
    """Pick durable request terms that help detect boilerplate planner text."""
    lowered = " ".join((message or "").strip().lower().split())
    if not lowered:
        return []
    terms: List[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9._/-]*", lowered):
        if token.isdigit():
            continue
        if token in PLAN_REQUEST_STOPWORDS:
            continue
        if len(token) < 4 and token not in PLAN_REQUEST_SHORT_TOKENS and not any(ch in token for ch in "./_-"):
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def plan_text_is_generic(text: str, request_terms: List[str]) -> bool:
    """Return whether planner text still reads like reusable boilerplate."""
    lowered = " ".join((text or "").strip().lower().split())
    if not lowered:
        return True
    if request_terms and any(term in lowered for term in request_terms):
        return False
    return any(pattern in lowered for pattern in GENERIC_PLAN_PATTERNS)


def request_prefers_illustrative_output(message: str) -> bool:
    """Return whether the request wants a visible proof artifact, not just a minimal scalar output."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    phrases = (
        "keep going until the artifact is real",
        "show the result",
        "show the output",
        "show me the result",
        "show me the output",
        "visible result",
        "prove it works",
        "proof it works",
        "show the graph",
        "show the chart",
        "chart showing",
        "graph showing",
    )
    if any(phrase in text for phrase in phrases):
        return True

    words = set(re.findall(r"[a-z0-9_+-]+", text))
    demo_terms = {"demo", "prove", "proof", "showcase"}
    output_terms = {
        "result", "results", "output", "artifact", "artifacts", "chart", "graph",
        "plot", "sequence", "table", "viewer", "render", "preview", "visual",
    }
    display_terms = {"show", "display", "render", "preview", "graph", "chart", "plot"}
    work_terms = {"run", "execute", "works", "working", "real", "actual"}

    if words & demo_terms and words & output_terms:
        return True
    if words & {"chart", "graph", "plot", "sequence", "table"} and words & {"show", "display", "render", "artifact", "output", "result"}:
        return True
    if words & display_terms and words & {"result", "output", "artifact"} and words & work_terms:
        return True
    return False


def build_request_specific_step(step_index: int, total_steps: int, request_focus: str) -> str:
    """Rewrite a generic build step so it stays anchored to the current request."""
    if request_prefers_illustrative_output(request_focus):
        if total_steps <= 1:
            return f"Build, run, and surface the clearest proof artifact for: {request_focus}"
        if step_index <= 0:
            return f"Inspect the relevant surfaces and build the main demo slice for: {request_focus}"
        if step_index == total_steps - 2:
            return f"Run the demo and capture real output or a visible artifact for: {request_focus}"
        if step_index >= total_steps - 1:
            return f"Polish the output so the proof is clear and verify it for: {request_focus}"
        return f"Implement the next concrete demo slice needed for: {request_focus}"
    if total_steps <= 1:
        return f"Implement the targeted change and verify it for: {request_focus}"
    if step_index <= 0:
        return f"Inspect the concrete code paths, UI surfaces, or artifacts involved in: {request_focus}"
    if total_steps == 2:
        return f"Implement the targeted change and verify it for: {request_focus}"
    if total_steps >= 3 and step_index == total_steps - 2:
        return f"Tighten the wiring, copy, or edge cases needed to finish: {request_focus}"
    if step_index >= total_steps - 1:
        return f"Verify the result against this request and close obvious gaps: {request_focus}"
    return f"Implement the targeted change needed for: {request_focus}"


def apply_plan_request_specificity_guardrails(message: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite overly generic planner output so the visible plan reflects the actual request."""
    normalized = normalize_deep_plan(plan)
    request_focus = summarize_request_for_plan(message)
    if not request_focus:
        return normalized

    request_terms = extract_request_terms_for_plan(message)
    if plan_text_is_generic(normalized.get("strategy", ""), request_terms):
        normalized["strategy"] = (
            "Inspect the exact surfaces implicated by the request, make the targeted change, "
            f"and verify the outcome: {request_focus}"
        )
    if plan_text_is_generic(normalized.get("deliverable", ""), request_terms):
        normalized["deliverable"] = (
            "A concrete result, grounded in inspected evidence, that directly addresses: "
            f"{request_focus}"
        )

    builder_steps = list(normalized.get("builder_steps", []))
    normalized["builder_steps"] = [
        build_request_specific_step(idx, len(builder_steps), request_focus)
        if plan_text_is_generic(step, request_terms)
        else step
        for idx, step in enumerate(builder_steps)
    ]
    return normalize_deep_plan(normalized)


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
    ][:4]
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


def normalize_plan_override_steps(payload: Any) -> List[str]:
    """Normalize a user-edited build-step override payload from the web UI."""
    if not isinstance(payload, list):
        return []
    steps = [
        str(step).strip()
        for step in payload
        if str(step).strip()
    ]
    return steps[:4]


def apply_plan_builder_step_override(plan: Dict[str, Any], builder_steps: List[str]) -> Dict[str, Any]:
    """Apply a UI-edited build checklist onto an already-saved deep-mode plan."""
    normalized = normalize_deep_plan(plan)
    override_steps = normalize_plan_override_steps(builder_steps)
    if override_steps:
        normalized["builder_steps"] = override_steps
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


def step_plan_cache_key(step_index: int) -> str:
    """Return a stable string key for persisted nested step plans."""
    return str(max(0, int(step_index or 0)))


def normalize_step_subplan(plan: Dict[str, Any], fallback_step: str) -> Dict[str, Any]:
    """Coerce a nested step plan into the saved micro-plan structure."""
    normalized = dict(plan or {})
    goal = str(normalized.get("goal", "")).strip()
    if not goal:
        goal = f"Complete this build step: {fallback_step}"

    substeps = normalized.get("substeps")
    if not isinstance(substeps, list):
        substeps = []
    internal_only_markers = (
        "__pycache__", ".pyc", ".ai/task-state", "task-state.json",
        ".ai/task-board", "task board artifact", "task-state artifact",
    )
    normalized_substeps = []
    for item in substeps:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(marker in lowered for marker in internal_only_markers):
            continue
        normalized_substeps.append(cleaned)
        if len(normalized_substeps) >= 4:
            break
    if not normalized_substeps:
        normalized_substeps = [
            f"Inspect the files, artifacts, or gaps most relevant to: {fallback_step}",
            f"Implement the next useful slice needed for: {fallback_step}",
            "Validate this slice and tighten the obvious gaps before moving on.",
        ]

    success_signal = str(normalized.get("success_signal", "")).strip()
    if not success_signal:
        success_signal = f"The build step is complete and reviewable: {fallback_step}"

    progress_note = str(normalized.get("progress_note", "")).strip()
    return {
        "goal": goal,
        "substeps": normalized_substeps,
        "success_signal": success_signal,
        "progress_note": progress_note,
    }


def build_heuristic_step_subplan(
    session: DeepSession,
    step: str,
    step_index: int,
    total_steps: int,
) -> Dict[str, Any]:
    """Build a deterministic nested subplan for a single top-level build step."""
    lowered = str(step or "").strip().lower()
    if request_prefers_illustrative_output(step):
        substeps = [
            f"Inspect the current demo output or artifact gaps tied to: {step}",
            f"Generate or improve a more illustrative output artifact or real command result for: {step}",
            f"Re-run or re-open the result and confirm the surfaced output now proves: {step}",
        ]
    elif any(term in lowered for term in ("create", "scaffold", "structure", "top-level", "starter repo")):
        substeps = [
            f"Lay down the concrete files, folders, and wiring needed for: {step}",
            f"Fill in the first working implementation slice required by: {step}",
            f"Tighten the surrounding docs or config so this slice is runnable and reviewable for: {step}",
        ]
    elif any(term in lowered for term in ("tighten", "polish", "docs", "wiring", "gap")):
        substeps = [
            f"Inspect the current output for missing wiring, docs, or sharp edges tied to: {step}",
            f"Apply the highest-leverage edits that close the most obvious gaps in: {step}",
            f"Re-read the touched files and check that this step now feels cohesive: {step}",
        ]
    elif any(term in lowered for term in ("verify", "check", "test", "validate")):
        substeps = [
            f"Inspect the files or commands most relevant to the verification target in: {step}",
            f"Run the narrowest useful command or direct file check for: {step}",
            f"Summarize what passed, what failed, and what still needs tightening for: {step}",
        ]
    else:
        substeps = [
            f"Inspect the specific files, artifacts, or gaps tied to this build step: {step}",
            f"Implement the next concrete slice needed to move this step forward: {step}",
            f"Validate the slice and tighten any obvious follow-up issues inside: {step}",
        ]

    return normalize_step_subplan(
        {
            "goal": f"Complete build step {step_index + 1} of {total_steps}: {step}",
            "substeps": substeps,
            "success_signal": f"Step {step_index + 1} is complete, grounded in workspace evidence, and ready for the next build step.",
        },
        step,
    )


def step_substep_summaries_for_index(session: DeepSession, step_index: int) -> List[str]:
    """Return persisted completed substep summaries for one build step."""
    raw = session.step_substep_summaries.get(step_plan_cache_key(step_index), [])
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def store_step_substep_summaries(session: DeepSession, step_index: int, summaries: List[str]) -> None:
    """Persist completed substep summaries for one build step."""
    key = step_plan_cache_key(step_index)
    normalized = [str(item).strip() for item in summaries if str(item).strip()]
    if normalized:
        session.step_substep_summaries[key] = normalized
    else:
        session.step_substep_summaries.pop(key, None)


def step_substep_reports_for_index(session: DeepSession, step_index: int) -> List[Dict[str, Any]]:
    """Return persisted completed substep reports for one build step."""
    raw = session.step_substep_reports.get(step_plan_cache_key(step_index), [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def store_step_substep_reports(session: DeepSession, step_index: int, reports: List[Dict[str, Any]]) -> None:
    """Persist completed substep evidence reports for one build step."""
    key = step_plan_cache_key(step_index)
    normalized = [item for item in reports if isinstance(item, dict)]
    if normalized:
        session.step_substep_reports[key] = normalized
    else:
        session.step_substep_reports.pop(key, None)


def render_nested_checklist(
    items: List[str],
    completed_count: int = 0,
    active_index: Optional[int] = None,
    prefix: str = "",
) -> List[str]:
    """Render a markdown checklist for nested build substeps."""
    rendered: List[str] = []
    for idx, item in enumerate(items):
        if idx < completed_count:
            marker = "[x]"
        elif active_index is not None and idx == active_index:
            marker = "[>]"
        else:
            marker = "[ ]"
        label = f"{prefix}{idx + 1}." if prefix else f"{idx + 1}."
        rendered.append(f"{marker} {label} {item}")
    return rendered or ["(none)"]


def format_step_subplan_progress(
    subplan: Dict[str, Any],
    completed_summaries: List[str],
    *,
    active_substep_index: Optional[int] = None,
    prefix: str = "",
) -> str:
    """Render a saved nested subplan plus progress markers for prompts/task boards."""
    substeps = [
        str(item).strip()
        for item in subplan.get("substeps", [])
        if str(item).strip()
    ]
    progress_lines = [
        f"Goal: {str(subplan.get('goal', '')).strip() or '(none)'}",
        "Substeps:",
        *render_nested_checklist(
            substeps,
            completed_count=len(completed_summaries),
            active_index=active_substep_index,
            prefix=prefix,
        ),
    ]
    if completed_summaries:
        progress_lines.append("Completed substep notes:")
        progress_lines.extend(f"- {summary}" for summary in completed_summaries)
    progress_note = str(subplan.get("progress_note", "")).strip()
    if progress_note:
        progress_lines.extend([
            "Current progress note:",
            progress_note,
        ])
    success_signal = str(subplan.get("success_signal", "")).strip()
    if success_signal:
        progress_lines.extend([
            "Done looks like:",
            success_signal,
        ])
    return "\n".join(progress_lines)


def summarize_completed_step_from_substeps(step: str, substep_summaries: List[str]) -> str:
    """Condense nested substep summaries into the parent build-step summary."""
    cleaned = [str(item).strip() for item in substep_summaries if str(item).strip()]
    if not cleaned:
        return f"Completed build step: {step}"
    joined = " | ".join(cleaned)
    return truncate_output(joined, limit=900)


def build_step_details_payload(
    session: DeepSession,
    steps: List[str],
    *,
    completed_count: int = 0,
    active_index: Optional[int] = None,
    active_substep_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a structured UI payload for top-level build steps and nested substeps."""
    details: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps or []):
        if idx < completed_count:
            state = "complete"
        elif active_index is not None and idx == active_index:
            state = "active"
        else:
            state = "pending"

        subplan = session.step_subplans.get(step_plan_cache_key(idx), {})
        completed_notes = step_substep_summaries_for_index(session, idx)
        completed_reports = step_substep_reports_for_index(session, idx)
        substeps = [
            str(item).strip()
            for item in subplan.get("substeps", [])
            if str(item).strip()
        ] if isinstance(subplan, dict) else []

        if active_index == idx and active_substep_index is not None:
            nested_active_index = active_substep_index
        elif state == "active" and len(completed_notes) < len(substeps):
            nested_active_index = len(completed_notes)
        else:
            nested_active_index = None

        nested_items: List[Dict[str, str]] = []
        for sub_idx, substep in enumerate(substeps):
            if sub_idx < len(completed_notes):
                nested_state = "complete"
            elif nested_active_index is not None and sub_idx == nested_active_index:
                nested_state = "active"
            else:
                nested_state = "pending"
            nested_items.append({
                "text": substep,
                "state": nested_state,
                "report": completed_reports[sub_idx] if sub_idx < len(completed_reports) and isinstance(completed_reports[sub_idx], dict) else {},
            })

        details.append({
            "text": str(step).strip(),
            "state": state,
            "goal": str(subplan.get("goal", "")).strip() if isinstance(subplan, dict) else "",
            "success_signal": str(subplan.get("success_signal", "")).strip() if isinstance(subplan, dict) else "",
            "progress_note": str(subplan.get("progress_note", "")).strip() if isinstance(subplan, dict) else "",
            "substeps": nested_items,
            "completed_notes": completed_notes,
            "completed_reports": completed_reports,
        })
    return details


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
            if not workspace_rel_path_is_hidden(rel_text):
                user_file_count += 1
            if not workspace_rel_path_is_hidden(rel_text) and len(sample_paths) < max_entries:
                sample_paths.append(rel_text)

    top_level: List[str] = []
    if workspace.exists():
        for item in sorted(workspace.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if workspace_rel_path_is_hidden(item.name):
                continue
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


INSPECT_CLARIFICATION_MARKERS = (
    "would you like to proceed",
    "different issue to address",
    "need more context",
    "need additional context",
    "there is no existing repository to inspect or modify",
    "there is no existing repo to inspect or modify",
    "workspace currently contains no files or directories",
    "workspace contains no files or directories",
    "which file",
    "which part",
    "which issue",
)


def extract_context_clarification_from_workspace_facts(facts: str) -> str:
    """Return a blocking clarification question when inspect explicitly says context is missing."""
    cleaned = str(facts or "").strip()
    if not cleaned:
        return ""
    primary = cleaned.split("\n\nGrounded workspace snapshot:", 1)[0].strip()
    if not primary or "?" not in primary:
        return ""
    lowered = primary.lower()
    if any(marker in lowered for marker in INSPECT_CLARIFICATION_MARKERS):
        return primary
    return ""


def should_pause_for_workspace_clarification(
    request_text: str,
    workspace_facts: str,
    workspace_snapshot: Dict[str, Any],
    *,
    has_plan: bool = False,
    execution_requested: bool = False,
) -> str:
    """Return a clarification question when inspect found the workspace lacks enough grounding to proceed safely."""
    if has_plan or execution_requested:
        return ""
    clarification = extract_context_clarification_from_workspace_facts(workspace_facts)
    if not clarification:
        return ""
    if workspace_is_effectively_empty(workspace_snapshot):
        return clarification
    if request_targets_current_repo(request_text) and int(workspace_snapshot.get("user_file_count", 0) or 0) <= 1:
        return clarification
    return ""


def request_targets_current_repo(message: str) -> bool:
    """Return whether the user is referring to the server's current repo rather than asking for a new scaffold."""
    text = " ".join((message or "").strip().lower().split())
    if not text or request_is_repo_scaffold(message):
        return False
    if any(phrase in text for phrase in CURRENT_REPO_REFERENCE_PHRASES):
        return True

    words = set(re.findall(r"[a-z0-9_+-]+", text))
    repo_terms = {"repo", "repository", "codebase", "project", "app"}
    reference_terms = {"this", "here", "current", "existing", "local"}
    repo_task_terms = {
        "review", "inspect", "analyze", "explain", "find", "grep", "search", "read",
        "fix", "patch", "change", "update", "edit", "refactor", "improve", "debug",
        "run", "test", "build",
    }
    return bool(words & repo_terms) and bool(words & reference_terms) and bool(words & repo_task_terms)


def should_exclude_repo_bootstrap_name(name: str) -> bool:
    """Return whether one repo entry should stay out of a seeded conversation workspace."""
    lowered = str(name or "").strip().lower()
    if not lowered:
        return True
    if lowered in SERVER_REPO_BOOTSTRAP_EXCLUDE_NAMES:
        return True
    if lowered == ".ds_store":
        return True
    if lowered == ".env" or lowered.startswith(".env."):
        return True
    return False


def repo_bootstrap_ignore(source_root: pathlib.Path, current_dir: str, names: List[str]) -> List[str]:
    """Ignore runtime-heavy repo entries when snapshotting the current repo into a chat workspace."""
    del source_root, current_dir
    return [name for name in names if should_exclude_repo_bootstrap_name(name)]


def bootstrap_workspace_from_current_repo(conversation_id: str) -> Dict[str, Any]:
    """Seed an empty conversation workspace with a filtered snapshot of the current server repo."""
    source_root = _repo_root.resolve()
    workspace = get_workspace_path(conversation_id)
    if not source_root.exists() or not source_root.is_dir():
        return {}
    if not path_is_within_root(workspace, WORKSPACE_ROOT_PATH):
        return {}

    copied_entries = 0
    try:
        for item in sorted(source_root.iterdir(), key=lambda path: path.name.lower()):
            if should_exclude_repo_bootstrap_name(item.name):
                continue
            destination = workspace / item.name
            if item.is_dir():
                shutil.copytree(
                    item,
                    destination,
                    ignore=lambda current_dir, names: repo_bootstrap_ignore(source_root, current_dir, list(names)),
                    dirs_exist_ok=False,
                )
                copied_entries += 1
            elif item.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)
                copied_entries += 1
    except Exception:
        reset_directory_contents(workspace)
        raise

    snapshot = capture_workspace_snapshot(conversation_id)
    return {
        "source_root": str(source_root),
        "copied_entries": copied_entries,
        "snapshot": snapshot,
    }


def maybe_bootstrap_workspace_from_current_repo(conversation_id: str, message: str) -> Dict[str, Any]:
    """Populate a fresh conversation workspace from the current repo when the user clearly asks about this repo."""
    if not request_targets_current_repo(message):
        return {}
    snapshot = capture_workspace_snapshot(conversation_id)
    if not workspace_is_effectively_empty(snapshot):
        return {}
    if not (_repo_root / "app.py").exists():
        return {}
    return bootstrap_workspace_from_current_repo(conversation_id)


def request_is_repo_scaffold(message: str) -> bool:
    """Return whether the user is asking for a starter repo or scaffold to be created."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    explicit_repo_phrases = (
        "scaffold a repo",
        "scaffold the repo",
        "generate a repo",
        "generate the repo",
        "create a repo",
        "create the repo",
        "repo scaffold",
        "repository scaffold",
        "starter repo",
        "project scaffold",
        "project structure",
        "repo structure",
        "repository structure",
        "folder structure",
        "directory structure",
        "file tree",
        "multi-file project",
        "multi file project",
    )
    if any(phrase in text for phrase in explicit_repo_phrases):
        return True
    if any(phrase in text for phrase in CURRENT_REPO_REFERENCE_PHRASES):
        return False
    words = set(re.findall(r"[a-z0-9_+-]+", text))
    repo_shape_terms = {
        "folders", "folder", "directories", "directory",
        "tree", "layout", "structure", "scaffold", "boilerplate", "template", "skeleton",
    }
    build_terms = {"build", "bootstrap", "create", "generate", "make", "scaffold", "seed", "starter", "template"}
    return bool(words & build_terms) and bool(words & repo_shape_terms)


def build_empty_workspace_steps(message: str) -> List[str]:
    """Choose concrete builder steps when starting from an empty workspace."""
    if request_is_repo_scaffold(message):
        return [
            "Create the requested repo structure and essential top-level files directly in the workspace.",
            "Implement the first useful working slice so the scaffold is immediately usable and reviewable.",
            "Tighten obvious gaps, wiring, and docs before verification.",
        ]
    if request_prefers_illustrative_output(message):
        return [
            "Create the main demo artifact or code slice directly in the workspace.",
            "Run it and capture real output or a surfaced artifact that proves it works.",
            "Polish the output, add a richer visual or sequence when it helps, and verify the final result.",
        ]
    return [
        "Create the first durable workspace artifact or code slice that best fits the request, even if it spans more than one file.",
        "Add supporting files, notes, or structure wherever they materially improve usability or make the result work.",
        "Tighten obvious gaps and verify the result before finishing.",
    ]


def build_empty_workspace_strategy(message: str) -> str:
    """Describe the execution approach for empty-workspace build requests."""
    if request_is_repo_scaffold(message):
        return "Scaffold the requested repo directly into the empty workspace, then implement the first useful slice and verify it."
    if request_prefers_illustrative_output(message):
        return "Build the requested demo directly in the workspace, run it, and refine the visible proof artifact until it clearly shows the result."
    return "Start by creating the most valuable durable artifact or code slice for the request in the empty workspace, expand as needed, then verify the result."


def build_empty_workspace_deliverable(message: str) -> str:
    """Describe the expected deliverable for empty-workspace build requests."""
    if request_is_repo_scaffold(message):
        return "A usable starter repo scaffold written into the workspace and ready to inspect or download."
    if request_prefers_illustrative_output(message):
        return "A runnable workspace demo plus a real output artifact or command result that clearly proves it works."
    return "A useful workspace result written into the empty workspace, shaped to fit the request rather than forced into a single-file deliverable."


def build_empty_workspace_verifier_checks(message: str, audit_check: str) -> List[str]:
    """Choose concrete verification checks when starting from an empty workspace."""
    if request_is_repo_scaffold(message):
        return [
            "Confirm the workspace now contains the requested scaffold and key entry-point files.",
            "Run the narrowest useful startup, syntax, or smoke check for the created repo.",
            audit_check,
        ]
    if request_prefers_illustrative_output(message):
        return [
            "Confirm the demo files or surfaced artifacts now exist and match the request.",
            "Re-run or inspect the real output artifact to verify the visible result is genuine.",
            audit_check,
        ]
    return [
        "Confirm the main artifact now exists in the workspace and matches the request.",
        "Run the narrowest useful syntax or smoke check for the artifact and any minimal supporting files.",
        audit_check,
    ]


def apply_deep_plan_guardrails(session: DeepSession, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Ground deep-mode plans in the inspected workspace and required audits."""
    normalized = normalize_deep_plan(plan)
    request_text = session.task_request or session.message
    if plan_looks_like_refusal(normalized, session.message):
        normalized["strategy"] = (
            "Use the inspected context to do the most useful work available, then call out any remaining blocker only if it materially limits the result."
        )
        normalized["deliverable"] = (
            "A useful answer or workspace result grounded in inspected evidence, not a plan for refusing the request."
        )
        normalized["builder_steps"] = [
            "Inspect the most relevant available files, attachments, or artifacts before deciding the next move.",
            "Do the most useful concrete work that is still possible from the inspected context.",
            "Tighten the result and isolate any true blocker instead of turning the blocker into the deliverable.",
        ]
        normalized["verifier_checks"] = [
            "Confirm the output still addresses the user's actual request.",
            "Call out any remaining blocker only if it is real, verified, and still affects the result.",
        ]

    if session.workspace_enabled and workspace_is_effectively_empty(session.workspace_snapshot):
        intent = classify_workspace_intent(session.message)
        if intent in {"focused_write", "broad_write"} or is_explicit_plan_execution_request(session.message):
            normalized["strategy"] = build_empty_workspace_strategy(request_text)
            normalized["deliverable"] = build_empty_workspace_deliverable(request_text)
            normalized["builder_steps"] = build_empty_workspace_steps(request_text)
            normalized["verifier_checks"] = build_empty_workspace_verifier_checks(request_text, "")
    elif session.workspace_enabled and request_prefers_illustrative_output(request_text):
        normalized["strategy"] = (
            "Inspect the relevant workspace surfaces, build the requested demo or proof slice, "
            "run it, and refine the visible output until it clearly demonstrates the result."
        )
        normalized["deliverable"] = (
            "A real workspace result with runnable output or a surfaced artifact that convincingly proves the requested behavior."
        )
        normalized["builder_steps"] = [
            "Inspect the relevant files or surfaces and build the main demo slice that best proves the request.",
            "Run the demo and capture real output or a surfaced artifact that shows the result.",
            "Polish the proof so the output is more illustrative and verify the final result against the request.",
        ]
        normalized["verifier_checks"] = [
            "Confirm the main demo files or surfaced artifacts now exist and match the request.",
            "Inspect or re-run the real output artifact to verify the claimed behavior.",
        ]

    verifier_checks = [str(step).strip() for step in normalized.get("verifier_checks", []) if str(step).strip()]
    audit_check = "Compare the requested deliverable against the files changed and verification evidence."
    if audit_check not in verifier_checks:
        verifier_checks.append(audit_check)
    normalized["verifier_checks"] = verifier_checks[:5]
    normalized = apply_plan_request_specificity_guardrails(request_text, normalized)
    return normalize_deep_plan(normalized)


def build_heuristic_deep_plan(session: DeepSession) -> Dict[str, Any]:
    """Build a short deterministic plan for fast profiles and fallback cases."""
    if session.workspace_enabled:
        plan = {
            "strategy": "Inspect the relevant files, make the highest-leverage change that fits the request, then verify it.",
            "deliverable": "A concise final answer grounded in the workspace and any verified file changes.",
            "builder_steps": [
                "Inspect the files or artifacts most relevant to the request.",
                "Make the workspace change or create the artifact that most advances the request.",
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
    return f"I'm working on: {request or 'the current request'}. I'll {next_step} and then respond with the best verified answer I can."


PLAN_EXECUTION_MARKERS = (
    "execute approved plan",
    "execute this approved plan",
    "execute the approved plan",
    "execute plan now",
    "execute this plan",
    "run this plan",
    "run the plan",
)

PLAN_APPROVAL_REPLY_MARKERS = {
    "yes",
    "yes please",
    "yes do it",
    "yes do that",
    "yes go ahead",
    "yes start",
    "yep",
    "sure",
    "ok",
    "okay",
    "approve",
    "approve this plan",
    "approve the plan",
    "approve and run",
    "run",
    "run it",
    "run this plan",
    "execute",
    "execute it",
    "execute this plan",
    "go ahead",
    "go ahead please",
    "go ahead and run it",
    "do it",
    "please do",
    "sounds good",
    "looks good",
    "that works",
    "lets do it",
    "let's do it",
    "start",
    "start it",
    "start with step 1",
    "start with the first step",
}


def normalize_turn_message(message: str) -> str:
    """Normalize a user message for short command-style intent checks."""
    return " ".join((message or "").strip().lower().split())


def normalize_approval_reply_text(message: str) -> str:
    """Normalize a short approval reply while tolerating light punctuation."""
    text = normalize_turn_message(message)
    if text.startswith("/plan "):
        text = text[6:].strip()
    text = re.sub(r"[.!?,;:]+", " ", text)
    return " ".join(text.split())


def is_explicit_plan_execution_request(message: str) -> bool:
    """Return whether the user is clearly asking to execute an already proposed plan."""
    text = normalize_turn_message(message)
    if not text:
        return False
    return any(marker in text for marker in PLAN_EXECUTION_MARKERS)


def is_plan_approval_reply(message: str) -> bool:
    """Return whether the user is approving a pending saved plan in short-form text."""
    text = normalize_approval_reply_text(message)
    if not text:
        return False
    return text in PLAN_APPROVAL_REPLY_MARKERS


def is_bare_plan_execution_request(message: str) -> bool:
    """Detect short approval-style plan execution requests without an edited inline draft."""
    if not is_explicit_plan_execution_request(message):
        return False
    text = normalize_turn_message(message)
    if not text:
        return False
    structural_markers = (
        "strategy:",
        "deliverable:",
        "steps:",
        "verification:",
        "1.",
        "2.",
    )
    return len(text) <= 120 and not any(marker in text for marker in structural_markers)


def request_is_about_limitations(message: str) -> bool:
    """Return whether the user is explicitly asking about support, errors, or blockers."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    markers = (
        "why can't",
        "why cannot",
        "can't do",
        "cannot do",
        "unable to",
        "not able to",
        "limitation",
        "limitations",
        "constraint",
        "constraints",
        "support",
        "supported",
        "unsupported",
        "capability",
        "capabilities",
        "does this work",
        "is this available",
        "why did this fail",
        "error",
        "broken",
        "workaround",
    )
    return any(marker in text for marker in markers)


def plan_looks_like_refusal(plan: Dict[str, Any], message: str) -> bool:
    """Flag plans that mainly script a refusal instead of useful work."""
    if request_is_about_limitations(message):
        return False

    fields: List[str] = [
        str(plan.get("strategy") or ""),
        str(plan.get("deliverable") or ""),
    ]
    fields.extend(str(step) for step in plan.get("builder_steps", []) if str(step).strip())
    fields.extend(str(step) for step in plan.get("verifier_checks", []) if str(step).strip())
    haystack = " ".join(part.strip().lower() for part in fields if part and part.strip())
    if not haystack:
        return False

    strong_markers = (
        "notify user",
        "processing constraints",
        "explanation of processing constraints",
        "alternative steps",
        "suggest external",
        "external pdf",
        "workaround",
        "unavailable on this server",
        "disabled server-side",
    )
    weak_markers = (
        "cannot",
        "can't",
        "unable to",
        "limitation",
        "constraint",
        "constraints",
        "blocked",
        "not possible",
        "unsupported",
        "unavailable",
        "redirect",
    )
    strong_hits = sum(1 for marker in strong_markers if marker in haystack)
    weak_hits = sum(1 for marker in weak_markers if marker in haystack)
    return strong_hits >= 1 or weak_hits >= 3


def should_preview_deep_plan(session: "DeepSession") -> bool:
    """Use a separate plan/approval step only for deep workspace execution requests."""
    if not session.workspace_enabled:
        return False
    if session.execution_requested or session.auto_execute:
        return False
    if infer_explicit_planning_request(session.message):
        return True
    return classify_workspace_intent(session.message) in {"focused_write", "broad_write"}


def should_auto_execute_workspace_task(
    conversation_id: str,
    message: str,
    features: "FeatureFlags",
) -> bool:
    """Auto-upgrade concrete workspace build requests into the full inspect/plan/build/verify flow."""
    if not features.agent_tools or not features.workspace_write:
        return False
    if request_is_about_limitations(message):
        return False
    if is_explicit_plan_execution_request(message) or is_plan_approval_reply(message):
        return False
    intent = classify_workspace_intent(message)
    if intent not in {"focused_write", "broad_write"}:
        return False
    return should_use_workspace_tools(conversation_id, message, features)


def should_resume_saved_workspace_task(
    conversation_id: str,
    message: str,
    features: "FeatureFlags",
) -> bool:
    """Auto-route short resume replies back into an existing saved task board."""
    if not features.agent_tools:
        return False
    if not (
        should_resume_task_state(message)
        or is_plan_approval_reply(message)
        or is_bare_plan_execution_request(message)
    ):
        return False
    payload = load_task_state(conversation_id)
    if not payload:
        return False
    if bool(payload.get("plan_preview_pending")):
        return True
    if task_state_has_pending_follow_up(payload):
        return True
    return bool(payload.get("plan"))


def format_deep_execution_prompt(plan: Dict[str, Any]) -> str:
    """Create an editable execution draft for plan approval and execution."""
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
        "Run the plan end-to-end in this turn when possible. Update the task board as you go, verify important checkpoints, and stop early only if you are blocked or waiting on a permission gate.",
        "",
        "You can adjust the plan if needed while executing it, but keep the scope aligned with this request.",
    ])
    return "\n".join(lines)


def build_pending_execution_plan_payload(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured pending-plan payload for UI hydration and approval."""
    normalized = normalize_deep_plan(plan)
    return {
        "summary": format_deep_plan_note(normalized),
        "execute_prompt": format_deep_execution_prompt(normalized),
        "builder_steps": list(normalized.get("builder_steps", [])),
    }


def render_deep_plan_preview(plan: Dict[str, Any]) -> str:
    """Render a short plan-preview answer for approval-first deep mode."""
    return (
        "I drafted the execution plan and put it in the approval panel below.\n\n"
        "Review or edit the steps there, then press Approve And Run to authorize workspace execution for this request."
    )


def render_saved_plan_write_access_message(plan: Dict[str, Any]) -> str:
    """Explain that a saved plan exists but this turn was not approved to edit the workspace."""
    return (
        "I found the saved plan, but write access was not granted for this turn.\n\n"
        f"{format_deep_plan_note(plan)}\n\n"
        "Approve workspace edits for this chat, then say continue and I'll resume the saved plan in the workspace. If you want to reshape the plan first, tell me what to change."
    )


def write_workspace_text(conversation_id: str, rel_path: str, content: str) -> str:
    """Write a UTF-8 text file into the conversation workspace."""
    target = resolve_workspace_relative_path(conversation_id, rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if len(encoded) > WORKSPACE_WRITE_SIZE_LIMIT:
        raise ValueError("Workspace text too large")
    validate_workspace_text_content(target, content)
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


def validate_workspace_text_content(target: pathlib.Path, content: str) -> Optional[Dict[str, Any]]:
    """Validate structured text formats before writing them into the workspace."""
    suffix = target.suffix.lower()
    if suffix != ".json":
        return None
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        location = f"line {exc.lineno} column {exc.colno}"
        raise ValueError(f"Invalid JSON in {target.name} at {location}: {exc.msg}") from exc
    return {
        "type": "json",
        "status": "valid",
    }


def format_task_board(
    session: DeepSession,
    active_build_step: Optional[int] = None,
    active_substep_index: Optional[int] = None,
) -> str:
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
        session.task_request or session.message or "(none)",
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
    ]
    if session.recent_product_feedback_summary:
        lines.extend([
            "",
            "## Recent Product Feedback",
            session.recent_product_feedback_summary,
            "",
            f"[[artifact:{session.recent_product_feedback_artifact_path}]]",
        ])
    lines.extend([
        "",
        "## Step Notes",
    ])
    if session.build_step_summaries:
        lines.extend(f"- {summary}" for summary in session.build_step_summaries)
    else:
        lines.append("- No build steps completed yet.")

    if session.step_subplans:
        lines.extend([
            "",
            "## Nested Step Plans",
        ])
        for idx, step in enumerate(builder_steps):
            subplan = session.step_subplans.get(step_plan_cache_key(idx))
            if not isinstance(subplan, dict) or not subplan:
                continue
            completed_substeps = step_substep_summaries_for_index(session, idx)
            sub_active_index = None
            if active_build_step == idx:
                if active_substep_index is not None:
                    sub_active_index = active_substep_index
                else:
                    substeps = [
                        str(item).strip()
                        for item in subplan.get("substeps", [])
                        if str(item).strip()
                    ]
                    if len(completed_substeps) < len(substeps):
                        sub_active_index = len(completed_substeps)
            lines.extend([
                "",
                f"### Step {idx + 1}: {step}",
                format_step_subplan_progress(
                    subplan,
                    completed_substeps,
                    active_substep_index=sub_active_index,
                    prefix=f"{idx + 1}.",
                ),
            ])

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
    active_substep_index: Optional[int] = None,
    announce: bool = False,
) -> str:
    """Persist the current task board into the workspace."""
    if not session.workspace_enabled:
        return session.task_board_path
    path = write_workspace_text(
        session.conversation_id,
        session.task_board_path,
        format_task_board(
            session,
            active_build_step=active_build_step,
            active_substep_index=active_substep_index,
        ),
    )
    if announce:
        await send_assistant_note(session.websocket, f"[[artifact:{path}]]")
    return path


def format_task_state_payload(session: DeepSession) -> Dict[str, Any]:
    """Return a machine-readable snapshot of the current deep-mode task state."""
    return {
        "request": session.task_request or session.message,
        "pause_reason": normalize_pause_reason(session.pause_reason),
        "workspace_facts": session.workspace_facts,
        "workspace_snapshot": session.workspace_snapshot,
        "recent_product_feedback_entries": session.recent_product_feedback_entries,
        "recent_product_feedback_summary": session.recent_product_feedback_summary,
        "recent_product_feedback_artifact_path": session.recent_product_feedback_artifact_path,
        "plan": session.plan,
        "plan_preview_pending": session.plan_preview_pending,
        "task_board_path": session.task_board_path,
        "build_step_summaries": session.build_step_summaries,
        "step_subplans": session.step_subplans,
        "step_substep_summaries": session.step_substep_summaries,
        "step_substep_reports": session.step_substep_reports,
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


def load_pending_execution_plan(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Return a pending plan preview for UI rehydration when one is saved in the workspace."""
    payload = load_task_state(conversation_id)
    if not payload or not bool(payload.get("plan_preview_pending")):
        return None
    plan = payload.get("plan")
    if not isinstance(plan, dict) or not plan:
        return None
    return build_pending_execution_plan_payload(plan)


def task_state_has_remaining_build_steps(payload: Dict[str, Any]) -> bool:
    """Return whether a saved task state still has unchecked build steps."""
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        return False
    steps = [
        str(step).strip()
        for step in plan.get("builder_steps", [])
        if str(step).strip()
    ]
    completed = [
        str(item).strip()
        for item in payload.get("build_step_summaries", [])
        if str(item).strip()
    ]
    return bool(steps) and len(completed) < len(steps)


def task_state_has_pending_follow_up(payload: Dict[str, Any]) -> bool:
    """Return whether a saved task state still represents resumable work."""
    pause_reason = normalize_pause_reason(payload.get("pause_reason"))
    if pause_reason:
        return True
    if task_state_has_remaining_build_steps(payload):
        return True
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        return False
    steps = [
        str(step).strip()
        for step in plan.get("builder_steps", [])
        if str(step).strip()
    ]
    completed = [
        str(item).strip()
        for item in payload.get("build_step_summaries", [])
        if str(item).strip()
    ]
    verification_summary = str(payload.get("verification_summary", "")).strip()
    return bool(steps) and len(completed) >= len(steps) and not verification_summary


def build_saved_progress_fallback_response(
    conversation_id: str,
    request_text: str = "",
    *,
    error_text: str = "",
) -> str:
    """Build a short, honest fallback reply when a workspace turn saved state but no clean final answer."""
    payload = load_task_state(conversation_id) or {}
    if not payload:
        if error_text:
            return (
                "I hit an unexpected internal error before I could finish the reply. "
                "Please say continue and I'll retry from the current workspace state."
            )
        return ""

    request = str(payload.get("request", "")).strip() or str(request_text or "").strip()
    snapshot = payload.get("workspace_snapshot") if isinstance(payload.get("workspace_snapshot"), dict) else {}
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    task_board_path = str(payload.get("task_board_path", "")).strip()
    build_summaries = [
        str(item).strip()
        for item in payload.get("build_step_summaries", [])
        if str(item).strip()
    ]
    changed_files = [
        str(item).strip()
        for item in payload.get("changed_files", [])
        if str(item).strip()
    ]
    verification_summary = str(payload.get("verification_summary", "")).strip()
    builder_steps = [
        str(step).strip()
        for step in plan.get("builder_steps", [])
        if str(step).strip()
    ]
    completed_count = len(build_summaries)
    pending_follow_up = task_state_has_pending_follow_up(payload)

    lines: List[str] = []
    if error_text:
        lines.append("I hit an unexpected internal error before I could finish the chat reply, but I saved the current workspace state.")
    else:
        lines.append("I didn't finish a clean chat reply for this turn, but I did save the current workspace state.")

    if request and request_targets_current_repo(request) and int(snapshot.get("user_file_count", 0) or 0) > 0 and not changed_files:
        lines.append("The files in the workspace panel are the repo snapshot for context, not newly generated output from this turn.")

    if builder_steps:
        lines.append(f"Progress: completed {completed_count} of {len(builder_steps)} build steps.")
        if completed_count < len(builder_steps):
            lines.append(f"Next step: {builder_steps[completed_count]}")

    if build_summaries:
        lines.append("Latest saved note: " + truncate_output(build_summaries[-1], limit=280))

    if changed_files:
        preview = ", ".join(f"`{path}`" for path in changed_files[:5])
        if len(changed_files) > 5:
            preview += f", and {len(changed_files) - 5} more"
        lines.append(f"Touched files: {preview}")

    if verification_summary:
        lines.append("Verification: " + truncate_output(verification_summary, limit=220))

    if task_board_path:
        lines.append(f"[[artifact:{task_board_path}]]")

    if pending_follow_up:
        lines.append("Say continue to resume from the saved workspace state.")

    return "\n\n".join(line for line in lines if line).strip()


def ensure_nonempty_turn_response(
    response: str,
    conversation_id: str,
    request_text: str = "",
    *,
    error_text: str = "",
) -> str:
    """Guarantee that the user gets a visible reply even when the rich pipeline returns nothing."""
    cleaned = str(response or "").strip()
    if cleaned:
        return cleaned
    fallback = build_saved_progress_fallback_response(
        conversation_id,
        request_text,
        error_text=error_text,
    )
    if fallback:
        return fallback
    if error_text:
        return (
            "I hit an unexpected internal error before I could finish the reply. "
            "Please say continue and I'll retry from the current workspace state."
        )
    return "I finished the turn without producing a visible reply. Please say continue and I'll resume from the current workspace state."


def render_step_checkpoint_message(session: DeepSession, step_index: int, step_summary: str) -> str:
    """Create a neutral pause message after a build step finishes."""
    steps = [
        str(step).strip()
        for step in session.plan.get("builder_steps", [])
        if str(step).strip()
    ]
    total_steps = len(steps)
    current_step = steps[step_index] if 0 <= step_index < total_steps else ""
    summary = step_summary.strip() or f"Completed build step {step_index + 1}."

    lines = [
        f"Completed step {step_index + 1} of {total_steps}: {current_step or 'Current plan step'}",
        "",
        summary,
        "",
    ]
    if step_index + 1 < total_steps:
        lines.extend([
            f"Next step: {steps[step_index + 1]}",
            "",
            "Saved progress is ready to resume from the next step.",
        ])
    else:
        lines.extend([
            "Build steps are complete.",
            "",
            "Next step: run verification and prepare the final answer.",
            "",
            "Saved progress is ready to resume from verification.",
        ])
    return "\n".join(lines)


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
    if session.workspace_enabled and not changed_files and session.execution_requested:
        gaps.append("No workspace files were changed during an execution request.")
    if session.workspace_enabled and successful_tool_steps == 0 and session.execution_requested:
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
    "finish it", "proceed", "keep working", "next", "next step", "do the next step",
    "keep going with the plan", "move to the next step",
}
PAUSE_REASON_COMMAND_APPROVAL = "command_approval"
PAUSE_REASON_WRITE_BLOCKED = "write_blocked"
PAUSE_REASON_USER_DECISION = "user_decision"
PAUSE_REASON_HARD_LIMIT = "hard_limit"
KNOWN_PAUSE_REASONS = {
    PAUSE_REASON_COMMAND_APPROVAL,
    PAUSE_REASON_WRITE_BLOCKED,
    PAUSE_REASON_USER_DECISION,
    PAUSE_REASON_HARD_LIMIT,
}


def should_resume_task_state(message: str) -> bool:
    """Return whether the user likely wants to continue the existing task board."""
    text = (message or "").strip().lower()
    if not text:
        return False
    return any(hint in text for hint in RESUME_HINTS)


async def maybe_resume_task_state(session: DeepSession) -> bool:
    """Restore prior task state when the user asks to continue the existing work."""
    if not session.workspace_enabled:
        return False

    resume_requested = should_resume_task_state(session.message)
    execute_saved_plan = is_bare_plan_execution_request(session.message)
    approval_reply = is_plan_approval_reply(session.message)
    if not resume_requested and not execute_saved_plan and not approval_reply:
        return False
    payload = load_task_state(session.conversation_id, session.task_state_path)
    if not payload:
        return False
    pending_follow_up = task_state_has_pending_follow_up(payload)
    if approval_reply:
        if bool(payload.get("plan_preview_pending")):
            execute_saved_plan = True
        elif pending_follow_up:
            resume_requested = True
        else:
            return False
    if execute_saved_plan and not bool(payload.get("plan_preview_pending")):
        return False

    plan = payload.get("plan")
    workspace_snapshot = payload.get("workspace_snapshot")
    if isinstance(workspace_snapshot, dict):
        session.workspace_snapshot = workspace_snapshot
    recent_feedback_entries = payload.get("recent_product_feedback_entries")
    if isinstance(recent_feedback_entries, list):
        session.recent_product_feedback_entries = [item for item in recent_feedback_entries if isinstance(item, dict)]
    session.recent_product_feedback_summary = str(payload.get("recent_product_feedback_summary", "")).strip()
    session.recent_product_feedback_artifact_path = str(
        payload.get("recent_product_feedback_artifact_path") or session.recent_product_feedback_artifact_path
    )
    if isinstance(plan, dict):
        session.plan = apply_deep_plan_guardrails(session, plan)
    session.plan_preview_pending = bool(payload.get("plan_preview_pending"))
    session.pause_reason = normalize_pause_reason(payload.get("pause_reason"))
    session.task_board_path = str(payload.get("task_board_path") or session.task_board_path)
    session.build_step_summaries = [
        str(item).strip() for item in payload.get("build_step_summaries", []) if str(item).strip()
    ]
    step_subplans = payload.get("step_subplans")
    if isinstance(step_subplans, dict):
        builder_steps = [
            str(item).strip()
            for item in session.plan.get("builder_steps", [])
            if str(item).strip()
        ]
        session.step_subplans = {
            str(key): normalize_step_subplan(
                value,
                builder_steps[int(str(key))] if str(key).isdigit() and int(str(key)) < len(builder_steps) else str(key),
            )
            for key, value in step_subplans.items()
            if isinstance(value, dict)
        }
    step_substep_summaries = payload.get("step_substep_summaries")
    if isinstance(step_substep_summaries, dict):
        session.step_substep_summaries = {
            str(key): [str(item).strip() for item in value if str(item).strip()]
            for key, value in step_substep_summaries.items()
            if isinstance(value, list)
        }
    step_substep_reports = payload.get("step_substep_reports")
    if isinstance(step_substep_reports, dict):
        session.step_substep_reports = {
            str(key): [item for item in value if isinstance(item, dict)]
            for key, value in step_substep_reports.items()
            if isinstance(value, list)
        }
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
    if previous_request and (execute_saved_plan or pending_follow_up or resume_requested or not session.task_request):
        session.task_request = previous_request
    session.execution_requested = bool(session.execution_requested or execute_saved_plan or pending_follow_up)
    session.resumed = True
    await send_assistant_note(
        session.websocket,
        (
            "Loaded the saved approved plan from the existing task board."
            if execute_saved_plan else
            "Resuming from the existing task board and saved workspace state."
        )
        + (f" Original request: {previous_request}" if previous_request else ""),
    )
    return bool(session.plan)

# ==================== vLLM Client (httpx) ====================

def get_agent_llm_params() -> Dict[str, Any]:
    """Return fixed default inference parameters for the local runtime."""
    return {
        "temperature": 0.25,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.15,
        "max_tokens": 4096,
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


async def send_reasoning_note(
    websocket: WebSocket,
    content: str,
    *,
    step_label: Optional[str] = None,
    phase: str = "think",
):
    """Emit a short reasoning-only note that should not appear as final output."""
    await websocket.send_json({
        "type": "reasoning_note",
        "content": content,
        "step_label": step_label or "",
        "phase": (phase or "think").strip() or "think",
    })


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
        summary or "Scope audit recorded.",
    )


async def send_plan_ready(websocket: WebSocket, plan: Dict[str, Any], execute_prompt: str):
    """Push a plan preview plus execution draft metadata to the web UI."""
    payload = build_pending_execution_plan_payload(plan)
    payload["execute_prompt"] = execute_prompt
    await websocket.send_json({
        "type": "plan_ready",
        "plan": payload.get("summary", ""),
        "execute_prompt": payload.get("execute_prompt", ""),
        "builder_steps": payload.get("builder_steps", []),
    })


async def send_build_steps(
    websocket: WebSocket,
    steps: List[str],
    completed_count: int = 0,
    active_index: Optional[int] = None,
    step_details: Optional[List[Dict[str, Any]]] = None,
):
    """Push the current build checklist state to the UI."""
    await websocket.send_json({
        "type": "build_steps",
        "steps": steps,
        "completed_count": completed_count,
        "active_index": active_index,
        "step_details": step_details or [],
    })


async def stream_chat_response(
    websocket: WebSocket,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = 0.25,
    *,
    start_output: bool = True,
) -> str:
    """Stream a plain assistant response to the UI and return the final saved text."""
    splitter = ThinkingStreamSplitter()
    if start_output:
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

# ==================== Database (SQLite file at DB_PATH) ====================

FTS_TABLE = "messages_fts"
DOCUMENT_FTS_TABLE = "document_chunks_fts"
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


def setup_document_tables(conn: sqlite3.Connection):
    """Create document-source metadata and chunk indexes when supported."""
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS document_sources
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            path TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            file_size INTEGER NOT NULL DEFAULT 0,
            modified_at TEXT NOT NULL DEFAULT '',
            file_type TEXT NOT NULL DEFAULT 'file',
            extractor TEXT NOT NULL DEFAULT '',
            page_count INTEGER,
            title TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'ready',
            error TEXT NOT NULL DEFAULT '',
            indexed_at TEXT NOT NULL,
            UNIQUE(conversation_id, path))'''
    )
    c.execute(
        '''CREATE TABLE IF NOT EXISTS document_chunks
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            section_title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            char_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            embedding BLOB,
            embedding_model TEXT,
            embedded_at TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(conversation_id, path, chunk_index))'''
    )
    c.execute('CREATE INDEX IF NOT EXISTS idx_document_sources_lookup ON document_sources(conversation_id, path)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_document_chunks_lookup ON document_chunks(conversation_id, path, chunk_index)')
    if sqlite_has_fts(conn):
        c.execute(
            f'''CREATE VIRTUAL TABLE IF NOT EXISTS {DOCUMENT_FTS_TABLE} USING fts5(
                 content,
                 section_title,
                 path UNINDEXED,
                 conversation_id UNINDEXED
                 )'''
        )


def ensure_embedding_columns(conn: sqlite3.Connection):
    """Add optional semantic-retrieval columns when running against an older database."""
    c = conn.cursor()
    for statement in (
        'ALTER TABLE messages ADD COLUMN embedding BLOB',
        'ALTER TABLE messages ADD COLUMN embedding_model TEXT',
        'ALTER TABLE messages ADD COLUMN embedded_at TEXT',
        'ALTER TABLE document_chunks ADD COLUMN embedding BLOB',
        'ALTER TABLE document_chunks ADD COLUMN embedding_model TEXT',
        'ALTER TABLE document_chunks ADD COLUMN embedded_at TEXT',
    ):
        try:
            c.execute(statement)
        except sqlite3.OperationalError:
            pass


def init_db():
    """Create tables and indexes if missing; safe to call on every startup."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS workspaces
                     (id TEXT PRIMARY KEY,
                      display_name TEXT NOT NULL,
                      root_path TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL)''')

        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id TEXT PRIMARY KEY,
                      title TEXT,
                      created_at TEXT,
                      updated_at TEXT,
                      workspace_id TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS messages
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      conversation_id TEXT,
                      role TEXT,
                      content TEXT,
                      timestamp TEXT,
                      feedback TEXT,
                      embedding BLOB,
                      embedding_model TEXT,
                      embedded_at TEXT,
                      FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')

        c.execute('''CREATE TABLE IF NOT EXISTS conversation_summaries
                     (conversation_id TEXT PRIMARY KEY,
                      summary TEXT,
                      source_message_count INTEGER,
                      updated_at TEXT,
                      FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')

        c.execute('''CREATE TABLE IF NOT EXISTS runs
                     (id TEXT PRIMARY KEY,
                      conversation_id TEXT UNIQUE NOT NULL,
                      workspace_id TEXT,
                      title TEXT,
                      status TEXT NOT NULL DEFAULT 'active',
                      sandbox_path TEXT NOT NULL,
                      started_at TEXT NOT NULL,
                      ended_at TEXT,
                      summary TEXT NOT NULL DEFAULT '',
                      promoted_count INTEGER NOT NULL DEFAULT 0)''')

        try:
            c.execute('ALTER TABLE messages ADD COLUMN feedback TEXT')
        except sqlite3.OperationalError:
            pass

        c.execute(
            '''UPDATE messages
               SET feedback = CASE
                   WHEN LOWER(TRIM(feedback)) IN ('positive', 'negative', 'neutral') THEN LOWER(TRIM(feedback))
                   ELSE 'neutral'
               END
               WHERE role = 'assistant' AND (
                   feedback IS NULL
                   OR TRIM(feedback) = ''
                   OR feedback != LOWER(TRIM(feedback))
                   OR LOWER(TRIM(feedback)) NOT IN ('positive', 'negative', 'neutral')
               )'''
        )

        try:
            c.execute('ALTER TABLE conversations ADD COLUMN run_id TEXT')
        except sqlite3.OperationalError:
            pass

        try:
            c.execute('ALTER TABLE conversations ADD COLUMN workspace_id TEXT')
        except sqlite3.OperationalError:
            pass

        try:
            c.execute('ALTER TABLE runs ADD COLUMN workspace_id TEXT')
        except sqlite3.OperationalError:
            pass

        try:
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conv_timestamp ON messages(conversation_id, timestamp)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_runs_conversation_id ON runs(conversation_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_workspace_id ON conversations(workspace_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_runs_workspace_id ON runs(workspace_id)')
        except sqlite3.OperationalError:
            pass

        c.execute('SELECT id, display_name, root_path, created_at, updated_at FROM workspaces')
        workspace_by_path: Dict[str, Dict[str, Any]] = {}
        for row in c.fetchall():
            record = workspace_row_to_record(row)
            if record:
                workspace_by_path[record["root_path"]] = record

        c.execute('''SELECT id, conversation_id, workspace_id, sandbox_path, started_at, title
                     FROM runs''')
        run_rows = c.fetchall()
        runs_by_conversation: Dict[str, Dict[str, Any]] = {}
        for run_id, conversation_id, workspace_id, sandbox_path, started_at, title in run_rows:
            safe_conversation_id = str(conversation_id or "").strip()
            if safe_conversation_id:
                runs_by_conversation[safe_conversation_id] = {
                    "id": str(run_id or "").strip(),
                    "workspace_id": str(workspace_id or "").strip(),
                    "sandbox_path": str(sandbox_path or "").strip(),
                    "started_at": started_at,
                    "title": str(title or "").strip(),
                }

        c.execute('SELECT id, title, created_at, updated_at, workspace_id FROM conversations')
        for conv_id, title, created_at, updated_at, workspace_id in c.fetchall():
            existing_run = runs_by_conversation.get(conv_id) or {}
            sandbox_path_value = str(existing_run.get("sandbox_path") or "").strip()
            if sandbox_path_value:
                sandbox_path = pathlib.Path(sandbox_path_value).expanduser().resolve(strict=False)
                sandbox_path.mkdir(parents=True, exist_ok=True)
            else:
                run_id = build_run_id(conv_id)
                sandbox_path = get_run_workspace_root(run_id, create=True)
            root_path = str(sandbox_path)

            workspace_record = workspace_by_path.get(root_path)
            resolved_workspace_id = str(workspace_id or "").strip()
            if resolved_workspace_id and not workspace_record:
                workspace_record = get_workspace_record(resolved_workspace_id)
            if not workspace_record:
                resolved_workspace_id = uuid.uuid4().hex
                workspace_record = {
                    "id": resolved_workspace_id,
                    "display_name": str(title or "").strip() or workspace_display_name_from_path(sandbox_path),
                    "root_path": root_path,
                    "created_at": created_at or utcnow_iso(),
                    "updated_at": updated_at or created_at or utcnow_iso(),
                }
                c.execute(
                    '''INSERT OR IGNORE INTO workspaces (id, display_name, root_path, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)''',
                    (
                        workspace_record["id"],
                        workspace_record["display_name"],
                        workspace_record["root_path"],
                        workspace_record["created_at"],
                        workspace_record["updated_at"],
                    ),
                )
                workspace_by_path[root_path] = workspace_record

            resolved_workspace_id = workspace_record["id"]
            c.execute('UPDATE conversations SET workspace_id = ? WHERE id = ?', (resolved_workspace_id, conv_id))

            run_id = str(existing_run.get("id") or "").strip() or build_workspace_run_id(resolved_workspace_id)
            started_at = existing_run.get("started_at") or created_at or utcnow_iso()
            run_title = str(existing_run.get("title") or "").strip() or str(title or "").strip()
            c.execute(
                '''INSERT OR IGNORE INTO runs
                   (id, conversation_id, workspace_id, title, status, sandbox_path, started_at, ended_at, summary, promoted_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    run_id,
                    conv_id,
                    resolved_workspace_id,
                    run_title,
                    "active",
                    root_path,
                    started_at,
                    None,
                    "",
                    0,
                ),
            )
            c.execute(
                '''UPDATE runs
                   SET workspace_id = ?, sandbox_path = ?, title = COALESCE(NULLIF(title, ''), ?)
                   WHERE id = ?''',
                (resolved_workspace_id, root_path, run_title, run_id),
            )
            try:
                c.execute('UPDATE conversations SET run_id = ? WHERE id = ?', (run_id, conv_id))
            except sqlite3.OperationalError:
                pass

        try:
            c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_workspaces_root_path ON workspaces(root_path)')
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

        setup_document_tables(conn)
        ensure_embedding_columns(conn)

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


class WorkspaceCreateRequest(BaseModel):
    display_name: Optional[str] = None
    root_path: Optional[str] = None
    create_if_missing: bool = True


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    workspace_id: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    mode: str = "deep"
    features: Dict[str, Any] = Field(default_factory=dict)
    slash_command: Optional[Dict[str, Any]] = None
    plan_override_steps: List[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    feedback: Optional[str] = None


class WorkspaceFileUpdateRequest(BaseModel):
    path: str
    content: str


class VoiceSynthesisRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None


class BufferedChatTransport:
    """Collect chat events for HTTP fallback requests."""

    supports_command_approval = False

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.events.append(json.loads(json.dumps(payload, default=str)))

# ==================== History selection (token budget for vLLM context) ====================

VALID_FEEDBACK_LABELS = {"positive", "negative", "neutral"}


def normalize_feedback_label(value: Optional[str], *, strict: bool = False) -> str:
    """Normalize a persisted feedback label."""
    feedback = str(value or "").strip().lower()
    if not feedback:
        return "neutral"
    if feedback in VALID_FEEDBACK_LABELS:
        return feedback
    if strict:
        raise ValueError("Feedback must be 'positive', 'negative', or 'neutral'")
    return "neutral"


def message_feedback_value(role: Optional[str], feedback: Optional[str]) -> str:
    """Return a normalized feedback label for assistant messages only."""
    return normalize_feedback_label(feedback) if str(role or "").strip().lower() == "assistant" else ""


IMPLICIT_FEEDBACK_REPLY_PREFIXES = (
    "this ",
    "it ",
    "that ",
    "the ",
    "also ",
    "still ",
    "i see ",
    "you ",
    "your ",
    "we ",
    "but ",
)
IMPLICIT_NEGATIVE_FEEDBACK_CATEGORY_PATTERNS: List[tuple[str, tuple[str, ...]]] = [
    (
        "artifact_visibility",
        (
            r"\b(?:plot|image|png|pdf|artifact|viewer)\b.*\b(?:doesn't show|didn't show|not show|not showing|won't show|broken|missing)\b",
            r"\bi see\b.*\bbut not\b",
        ),
    ),
    (
        "missing_context",
        (
            r"\bnot enough context\b",
            r"\bshould have asked\b",
            r"\bshould've asked\b",
            r"\bno context\b",
            r"\bmissing context\b",
        ),
    ),
    (
        "non_interactive_artifact",
        (
            r"\bnot interactive\b",
            r"\bcan't pick\b",
            r"\bcannot pick\b",
            r"\bnot a real artifact\b",
        ),
    ),
    (
        "silent_turn",
        (
            r"\bno chat response\b",
            r"\bdidn't have enough context\b.*\bshould have asked\b",
            r"\bgave files that i have no context for\b",
        ),
    ),
    (
        "approval_resume",
        (
            r"\bforgot\b.*\bapprov",
            r"\bacted as if\b.*\bapprov",
            r"\bdidn't\b.*\bapprov",
            r"\bdid not\b.*\bapprov",
        ),
    ),
    (
        "workspace_clutter",
        (
            r"\boverwhelming\b",
            r"\b\.venv\b",
            r"\bdotfiles should be hidden\b",
        ),
    ),
    (
        "output_quality",
        (
            r"\bunsatisfying\b",
            r"\bcould be much better\b",
            r"\bnot based on the actual\b",
            r"\bnot the actual\b",
            r"\bthird check\b.*\bdoesn't get checked\b",
            r"\bshould have\b",
        ),
    ),
]
IMPLICIT_NEGATIVE_FEEDBACK_GENERAL_CUES = (
    "doesn't",
    "didn't",
    "isn't",
    "doesnt",
    "didnt",
    "isnt",
    "forgot",
    "confusing",
    "overwhelming",
    "unsatisfying",
    "broken",
    "broke",
    "missing",
    "not interactive",
    "not enough",
    "should have",
    "no chat response",
    "not a real artifact",
    "not based on",
)


def detect_implicit_failure_feedback(message: str) -> Dict[str, str]:
    """Classify short corrective user replies that imply the previous assistant turn failed."""
    raw = str(message or "").strip()
    if not raw:
        return {}
    normalized = " ".join(raw.lower().split())
    if not normalized:
        return {}

    category = ""
    for name, patterns in IMPLICIT_NEGATIVE_FEEDBACK_CATEGORY_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            category = name
            break

    if not category:
        starts_like_feedback = normalized.startswith(IMPLICIT_FEEDBACK_REPLY_PREFIXES)
        has_general_cue = any(cue in normalized for cue in IMPLICIT_NEGATIVE_FEEDBACK_GENERAL_CUES)
        if not (starts_like_feedback and has_general_cue):
            return {}
        category = "general_failure"

    return {
        "label": "negative",
        "category": category,
        "excerpt": truncate_output(raw, limit=220),
    }


def apply_implicit_feedback_from_user_reply(
    conn: sqlite3.Connection,
    conversation_id: str,
    user_message_id: int,
    content: str,
) -> Dict[str, str]:
    """Interpret corrective user text as negative feedback on the immediately previous assistant message."""
    signal = detect_implicit_failure_feedback(content)
    if signal.get("label") != "negative":
        return {}

    cursor = conn.cursor()
    cursor.execute(
        '''SELECT id, content
           FROM messages
           WHERE conversation_id = ? AND role = 'assistant' AND id < ?
           ORDER BY id DESC
           LIMIT 1''',
        (conversation_id, user_message_id),
    )
    row = cursor.fetchone()
    if not row:
        return {}

    assistant_message_id = int(row[0])
    cursor.execute('UPDATE messages SET feedback = ? WHERE id = ?', ('negative', assistant_message_id))
    logger.info(
        "Implicit negative feedback detected for assistant message %s in conversation %s (%s)",
        assistant_message_id,
        conversation_id,
        signal.get("category", "general_failure"),
    )
    return {
        **signal,
        "assistant_message_id": str(assistant_message_id),
        "assistant_excerpt": truncate_output(str(row[1] or ""), limit=180),
    }


def collect_recent_product_feedback_entries(
    limit: int = 6,
    *,
    per_conversation_limit: int = 2,
) -> List[Dict[str, str]]:
    """Collect recent corrective user replies that should inform repo-level product improvements."""
    safe_limit = max(1, min(int(limit or 6), 12))
    safe_per_conversation = max(1, min(int(per_conversation_limit or 2), 3))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT m.id,
                  m.conversation_id,
                  m.content,
                  m.timestamp,
                  prev.id,
                  prev.content,
                  prev.feedback,
                  c.title,
                  c.updated_at
           FROM messages m
           JOIN conversations c ON c.id = m.conversation_id
           JOIN messages prev ON prev.id = (
               SELECT p.id
               FROM messages p
               WHERE p.conversation_id = m.conversation_id AND p.id < m.id
               ORDER BY p.id DESC
               LIMIT 1
           )
           WHERE m.role = 'user' AND prev.role = 'assistant'
           ORDER BY c.updated_at DESC, m.id DESC
           LIMIT ?''',
        (max(safe_limit * 12, 48),),
    )
    rows = cursor.fetchall()
    conn.close()

    counts_by_conversation: Dict[str, int] = {}
    entries: List[Dict[str, str]] = []
    for row in rows:
        signal = detect_implicit_failure_feedback(str(row[2] or ""))
        previous_feedback = normalize_feedback_label(row[6])
        if signal.get("label") != "negative" and previous_feedback != "negative":
            continue
        conversation_id = str(row[1] or "").strip()
        if not conversation_id:
            continue
        if counts_by_conversation.get(conversation_id, 0) >= safe_per_conversation:
            continue
        counts_by_conversation[conversation_id] = counts_by_conversation.get(conversation_id, 0) + 1
        entries.append({
            "conversation_id": conversation_id,
            "conversation_title": str(row[7] or "Untitled").strip() or "Untitled",
            "updated_at": str(row[8] or row[3] or "").strip(),
            "user_message_id": str(row[0]),
            "assistant_message_id": str(row[4]),
            "category": signal.get("category", "general_failure" if previous_feedback == "negative" else ""),
            "user_feedback": truncate_output(str(row[2] or ""), limit=220),
            "assistant_excerpt": truncate_output(str(row[5] or ""), limit=180),
        })
        if len(entries) >= safe_limit:
            break
    return entries


def format_recent_product_feedback_summary(entries: List[Dict[str, str]]) -> str:
    """Format recent product feedback as compact plain text for prompts and task boards."""
    if not entries:
        return ""
    lines: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        category = str(entry.get("category") or "general_failure").replace("_", " ")
        lines.append(
            f"{idx}. [{category}] User: {entry.get('user_feedback', '(none)')}"
        )
        lines.append(
            f"   Prior assistant: {entry.get('assistant_excerpt', '(none)')}"
        )
    return "\n".join(lines)


def format_recent_product_feedback_markdown(entries: List[Dict[str, str]]) -> str:
    """Render recent corrective feedback into a durable workspace artifact."""
    if not entries:
        return "# Recent Product Feedback\n\nNo recent corrective feedback was captured."
    lines = [
        "# Recent Product Feedback",
        "",
        "Treat these as recent failure signals from corrective user replies or explicit negative feedback.",
    ]
    for idx, entry in enumerate(entries, start=1):
        category = str(entry.get("category") or "general_failure").replace("_", " ").title()
        lines.extend([
            "",
            f"## {idx}. {category}",
            f"- Conversation: {entry.get('conversation_title', 'Untitled')} ({entry.get('conversation_id', '')})",
            f"- Updated: {entry.get('updated_at', '(unknown)')}",
            f"- User feedback: {entry.get('user_feedback', '(none)')}",
            f"- Prior assistant: {entry.get('assistant_excerpt', '(none)')}",
        ])
    return "\n".join(lines)


def request_wants_recent_product_feedback(message: str) -> bool:
    """Return whether a repo-improvement request should be grounded in recent corrective chat feedback."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    repoish = (
        request_targets_current_repo(message)
        or "chat.db" in text
        or "ai-chat" in text
        or "developer of this ai-chat" in text
    )
    if not repoish:
        return False
    feedback_terms = (
        "feedback",
        "recent chats",
        "last chat",
        "last few messages",
        "chat.db",
        "dev loop",
        "under-deliver",
        "failure",
    )
    improvement_terms = (
        "improve",
        "fix",
        "review",
        "patch",
        "developer",
        "app",
        "agent",
        "software",
        "prompt",
    )
    return any(term in text for term in feedback_terms) and any(term in text for term in improvement_terms)

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


def get_conversation_messages_for_ui(conv_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Return recent raw conversation messages for the UI, preserving ids and feedback."""
    safe_limit = max(1, min(int(limit or 100), 500))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''SELECT id, role, content, timestamp, feedback
           FROM (
               SELECT id, role, content, timestamp, feedback
               FROM messages
               WHERE conversation_id = ?
               ORDER BY timestamp DESC, id DESC
               LIMIT ?
           )
           ORDER BY timestamp ASC, id ASC''',
        (conv_id, safe_limit),
    )
    rows = c.fetchall()
    conn.close()

    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "timestamp": row[3],
            "feedback": message_feedback_value(row[1], row[4]),
        }
        for row in rows
    ]

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
            'timestamp': row[3], 'feedback': message_feedback_value(row[1], row[4])
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
                'feedback': message_feedback_value(row[1], row[5]), 'fts_rank': row[6],
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

def save_message(conv_id: str, role: str, content: str, workspace_id: Optional[str] = None) -> int:
    """Save message to database and return message ID"""
    title = content[:50] + "..." if len(content) > 50 else content
    ensure_run_for_conversation(conv_id, title=title, workspace_id=workspace_id)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('SELECT id FROM conversations WHERE id = ?', (conv_id,))
    if not c.fetchone():
        created = ensure_conversation_record(conv_id, title=title, workspace_id=workspace_id)
        try:
            c.execute('UPDATE conversations SET run_id = ? WHERE id = ?', (created.get("run_id", ""), conv_id))
        except sqlite3.OperationalError:
            pass

    stored_feedback = "neutral" if role == "assistant" else None
    c.execute('''INSERT INTO messages (conversation_id, role, content, timestamp, feedback)
                 VALUES (?, ?, ?, ?, ?)''',
              (conv_id, role, content, datetime.now().isoformat(), stored_feedback))

    message_id = c.lastrowid

    if role == 'user':
        apply_implicit_feedback_from_user_reply(conn, conv_id, message_id, content)

    c.execute('UPDATE conversations SET updated_at = ? WHERE id = ?',
              (datetime.now().isoformat(), conv_id))

    conn.commit()
    conn.close()
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
    raw_allowed_tool_permissions = payload.get("allowed_tool_permissions", [])
    allowed_tool_permissions = [
        normalize_allowed_tool_permission_key(item)
        for item in (
            raw_allowed_tool_permissions
            if isinstance(raw_allowed_tool_permissions, list)
            else []
        )
        if isinstance(item, str) and normalize_allowed_tool_permission_key(item)
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
        web_search=flag("web_search", True),
        auto_approve_tool_permissions=flag("auto_approve_tool_permissions", False),
        allowed_commands=sorted(set(allowed_commands)),
        allowed_tool_permissions=sorted(set(allowed_tool_permissions)),
    )


def allowed_workspace_tools(
    features: FeatureFlags,
    include_write: bool = False,
    include_render: bool = False,
) -> List[str]:
    """Return workspace tools allowed by the current per-turn approvals."""
    allowed = ["workspace.list_files", "workspace.grep", "workspace.read_file", "spreadsheet.describe"]
    if include_render:
        allowed.append("workspace.render")
    if features.workspace_write and include_write:
        allowed.append("workspace.patch_file")
    if features.workspace_run_commands:
        allowed.append("workspace.run_command")
    return allowed

def update_message_feedback(message_id: int, feedback: str) -> bool:
    """Update feedback for a message and report whether it exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    exists = c.execute('SELECT 1 FROM messages WHERE id = ?', (message_id,)).fetchone()
    if not exists:
        conn.close()
        return False
    c.execute('UPDATE messages SET feedback = ? WHERE id = ?', (normalize_feedback_label(feedback), message_id))
    conn.commit()
    conn.close()
    return True

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
            'content': row[3], 'timestamp': row[4], 'feedback': message_feedback_value(row[2], row[5])
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

_repo_root = pathlib.Path(__file__).resolve().parents[2]
_web_root = _repo_root / "src" / "web"


def compute_static_asset_version(web_root: pathlib.Path) -> str:
    """Build a cheap cache-busting token from local static asset mtimes."""
    hasher = hashlib.sha1()
    for rel_path in ("index.html", "app.js", "style.css"):
        target = web_root / rel_path
        try:
            stat = target.stat()
        except OSError:
            continue
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    digest = hasher.hexdigest()[:12]
    return digest or str(int(time.time()))


app.mount("/static", StaticFiles(directory=str(_web_root)), name="static")
templates = Jinja2Templates(directory=str(_web_root))

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
        response = templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "themes_json": json.dumps({"light": COLORS_LIGHT, "dark": COLORS_DARK}),
                "model_name": get_active_model_name(),
                "app_title": APP_TITLE,
                "static_asset_version": compute_static_asset_version(_web_root),
            },
        )
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as e:
        logger.error(f"Error generating home page: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error loading page: {str(e)}")


@app.get("/api/voice/status")
async def voice_status():
    return get_voice_runtime_summary()


@app.post("/api/voice/transcribe")
async def transcribe_voice_audio(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
):
    runtime = get_voice_runtime_summary()
    if not runtime["stt_available"]:
        raise HTTPException(
            status_code=503,
            detail="Server transcription is unavailable. Install a native STT tool or set VOICE_STT_COMMAND.",
        )

    suffix = pathlib.Path(file.filename or "").suffix.lower()[:16] or ".webm"
    session_id = build_voice_artifact_id(conversation_id)
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
    try:
        result = await run_native_voice_command(command)
        transcript = read_transcript_output(output_dir, input_path, transcript_path)
        return {
            "transcript": transcript,
            "bytes": bytes_written,
            "content_type": file.content_type or "application/octet-stream",
            "backend": runtime["stt_backend"],
            "command": result["command"][0],
        }
    finally:
        delete_voice_file(input_path)
        delete_voice_file(transcript_path)
        prune_voice_storage_if_needed()


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

    speech_id = build_voice_artifact_id(request.conversation_id)
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
    response_payload: Optional[Dict[str, Any]] = None
    keep_output = False
    try:
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
        response_payload = {
            "audio_url": f"/api/voice/file/{output_path.name}",
            "media_type": media_type,
            "backend": runtime["tts_backend"],
            "voice": runtime["tts_voice"],
            "command": result["command"][0],
        }
        keep_output = True
    finally:
        delete_voice_file(text_path)
        if not keep_output:
            delete_voice_file(output_path)
        prune_voice_storage_if_needed(
            protected_paths=[output_path] if keep_output and output_path.exists() else None
        )

    return response_payload


@app.get("/api/voice/file/{filename}")
async def get_voice_file(filename: str):
    safe_name = pathlib.Path(filename).name
    target = (get_voice_dir("tts-audio", create=False) / safe_name).resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    if get_voice_dir("tts-audio", create=False) not in target.parents:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        os.utime(target, None)
    except OSError:
        pass
    return FileResponse(target, media_type=guess_audio_media_type(target), filename=target.name)


def build_workspace_catalog_payload(workspace: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize one workspace row for the shared catalog UI."""
    workspace_id = str(workspace.get("id") or "").strip()
    root_path = str(workspace.get("root_path") or "").strip()
    run = get_run_record_by_workspace_id(workspace_id) if workspace_id else None
    return {
        "id": workspace_id,
        "display_name": str(workspace.get("display_name") or "").strip() or "Workspace",
        "root_path": root_path,
        "created_at": workspace.get("created_at"),
        "updated_at": workspace.get("updated_at"),
        "conversation_count": count_conversations_for_workspace(workspace_id) if workspace_id else 0,
        "run_id": run["id"] if run else None,
        "root_exists": pathlib.Path(root_path).exists() if root_path else False,
    }


def get_workspace_route_target(workspace_id: str, *, create: bool = False) -> tuple[Dict[str, Any], pathlib.Path, Optional[Dict[str, Any]]]:
    """Resolve one workspace id into its record, path, and run metadata."""
    workspace = get_workspace_record(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    try:
        workspace_path = get_workspace_path_for_workspace_id(workspace_id, create=create)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run = get_run_record_by_workspace_id(workspace_id)
    return workspace, workspace_path, run


@app.get("/api/workspaces")
async def get_workspaces():
    try:
        workspaces = [build_workspace_catalog_payload(item) for item in list_workspace_records(ensure_default=True)]
        default_workspace = workspaces[0]["id"] if workspaces else None
        return {"workspaces": workspaces, "default_workspace_id": default_workspace}
    except Exception as e:
        logger.error("Error getting workspaces: %s", e)
        return {"workspaces": [], "default_workspace_id": None}


@app.post("/api/workspaces")
async def create_workspace(request: WorkspaceCreateRequest):
    try:
        workspace = create_workspace_record(
            display_name=request.display_name,
            root_path=request.root_path,
            create_if_missing=bool(request.create_if_missing),
        )
        return build_workspace_catalog_payload(workspace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="A workspace already exists for that path") from exc


@app.get("/api/workspaces/{workspace_id}")
async def get_workspace_catalog_entry(workspace_id: str):
    workspace = get_workspace_record(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return build_workspace_catalog_payload(workspace)


@app.post("/api/workspaces/{workspace_id}/rename")
async def rename_workspace(workspace_id: str, request: RenameRequest):
    workspace = get_workspace_record(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    title = str(request.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Workspace name is required")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'UPDATE workspaces SET display_name = ?, updated_at = ? WHERE id = ?',
        (title, utcnow_iso(), workspace_id),
    )
    conn.commit()
    conn.close()
    updated = get_workspace_record(workspace_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return build_workspace_catalog_payload(updated)


@app.delete("/api/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str, delete_files: bool = False):
    workspace = get_workspace_record(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    conversation_count = count_conversations_for_workspace(workspace_id)
    if conversation_count:
        raise HTTPException(status_code=409, detail="Delete or relink chats in this workspace before removing it")

    workspace_path = pathlib.Path(workspace["root_path"]).expanduser().resolve(strict=False)
    run = get_run_record_by_workspace_id(workspace_id)
    if run:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('DELETE FROM runs WHERE id = ?', (run["id"],))
        conn.commit()
        conn.close()
        sandbox_path = pathlib.Path(run["sandbox_path"]).resolve(strict=False)
        if sandbox_path.exists() and (RUNS_ROOT_PATH == sandbox_path or RUNS_ROOT_PATH in sandbox_path.parents):
            hosted_root = sandbox_path.parent if sandbox_path.name == "workspace" else sandbox_path
            shutil.rmtree(hosted_root, ignore_errors=True)
    delete_managed_python_env_for_workspace(workspace_id)

    if delete_files and workspace_path.exists():
        shutil.rmtree(workspace_path, ignore_errors=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM workspaces WHERE id = ?', (workspace_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "workspace_id": workspace_id, "deleted_files": bool(delete_files)}


@app.get("/api/conversations")
async def get_conversations():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT c.id, c.title, c.created_at, c.updated_at,
                     c.workspace_id, w.display_name, w.root_path,
                     m.content, m.timestamp
                     FROM conversations c
                     LEFT JOIN workspaces w ON w.id = c.workspace_id
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
                'workspace_id': str(row[4] or '').strip(),
                'workspace_display_name': str(row[5] or '').strip(),
                'workspace_root_path': str(row[6] or '').strip(),
                'last_message': row[7] if row[7] else '',
                'last_message_timestamp': row[8] if row[8] else row[3]
            })

        conn.close()
        return {'conversations': conversations}
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return {'conversations': []}

@app.post("/api/message/{message_id}/feedback")
async def submit_feedback(message_id: int, request: FeedbackRequest):
    try:
        feedback = normalize_feedback_label(request.feedback, strict=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        message = get_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        if message.get("role") != "assistant":
            raise HTTPException(status_code=400, detail="Feedback can only be recorded for assistant messages")
        update_message_feedback(message_id, feedback)
        sync_workflow_feedback_for_message(message_id, feedback)
        return {"status": "success", "message": f"Feedback recorded: {feedback}", "feedback": feedback}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
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

@app.post("/api/chat")
async def chat_http(request: ChatRequest):
    transport = BufferedChatTransport()
    payload = request.model_dump()
    features = payload.get("features")
    if isinstance(features, dict):
        payload["features"] = {
            **features,
            "workspace_run_commands": False,
            "allowed_commands": [],
        }
    await process_chat_turn(transport, payload)
    return {"events": transport.events}

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        conversation = get_conversation_record(conversation_id)
        workspace_id = str(conversation.get("workspace_id") or "").strip() if conversation else ""
        workspace = get_workspace_record(workspace_id) if workspace_id else None
        history = get_conversation_messages_for_ui(conversation_id, limit=100)
        return {
            'messages': history,
            'pending_plan': load_pending_execution_plan(conversation_id),
            'workspace_id': workspace_id,
            'workspace': build_workspace_catalog_payload(workspace) if workspace else None,
        }
    except Exception as e:
        return {'messages': [], 'pending_plan': None, 'workspace_id': '', 'workspace': None}

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
        delete_voice_artifacts_for_conversation(conversation_id)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM conversation_summaries WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        conn.commit()
        conn.close()
        prune_voice_storage_if_needed()
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
        target = pathlib.Path(abs_path)
        return build_workspace_file_result(target, rel_path=path, text_limit=None)
    except (HTTPException):
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def build_workspace_info_payload(
    workspace_id: str,
    *,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return shared workspace metadata for either route shape."""
    workspace, workspace_path, run = get_workspace_route_target(workspace_id, create=False)
    payload = {
        "workspace_id": workspace_id,
        "run_id": run["id"] if run else None,
        "workspace_path": str(workspace_path),
        "workspace_label": workspace.get("display_name") or workspace_path.name,
        "display_name": workspace.get("display_name") or workspace_path.name,
        "root_path": str(workspace_path),
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return payload


async def upload_workspace_files_for_workspace(
    workspace_id: str,
    files: List[UploadFile],
    target_path: str = ".",
    *,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload files into a workspace addressed directly by workspace id."""
    _workspace_record, workspace, _run = get_workspace_route_target(workspace_id, create=True)
    target_dir = resolve_workspace_relative_path_from_root(workspace, target_path or ".")

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
            "kind": classify_workspace_file_kind(destination.name),
        })

    payload = {
        "workspace_id": workspace_id,
        "workspace_path": str(workspace),
        "target_path": format_workspace_path(target_dir, workspace),
        "files": saved_files,
        "count": len(saved_files),
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return payload


def list_workspace_files_for_workspace(
    workspace_id: str,
    path: str = "",
    *,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List files in a workspace addressed directly by workspace id."""
    _workspace_record, workspace, run = get_workspace_route_target(workspace_id, create=False)
    if not workspace.exists():
        payload = {
            "workspace_id": workspace_id,
            "run_id": run["id"] if run else None,
            "workspace_path": str(workspace),
            "path": ".",
            "items": [],
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        return payload

    target = resolve_workspace_relative_path_from_root(workspace, path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    target_rel_path = format_workspace_path(target, workspace)
    target_is_hidden = workspace_rel_path_is_hidden(target_rel_path)
    items = []
    for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            item_rel_path = format_workspace_path(item, workspace)
            if workspace_rel_path_is_hidden(item_rel_path) and not target_is_hidden:
                continue
            items.append(build_workspace_listing_item(item, workspace))
        except (OSError, PermissionError):
            continue

    payload = {
        "workspace_id": workspace_id,
        "run_id": run["id"] if run else None,
        "workspace_path": str(workspace),
        "path": format_workspace_path(target, workspace),
        "items": items,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return payload


def read_workspace_file_for_workspace(
    workspace_id: str,
    path: str,
    *,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Read one file from a workspace addressed directly by workspace id."""
    _workspace_record, workspace, _run = get_workspace_route_target(workspace_id, create=True)
    target = resolve_workspace_relative_path_from_root(workspace, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    rel_path = format_workspace_path(target, workspace)
    try:
        preview = build_workspace_file_result(
            target,
            conversation_id=conversation_id,
            rel_path=rel_path,
            text_limit=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "workspace_id": workspace_id,
        "workspace_path": str(workspace),
        **preview,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return payload


def write_workspace_file_for_workspace(
    workspace_id: str,
    request: WorkspaceFileUpdateRequest,
    *,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Write one text file into a workspace addressed directly by workspace id."""
    _workspace_record, workspace, _run = get_workspace_route_target(workspace_id, create=True)
    rel_path = (request.path or "").strip()
    if not rel_path:
        raise HTTPException(status_code=400, detail="Path is required")

    target = resolve_workspace_relative_path_from_root(workspace, rel_path)
    if target.exists() and not target.is_file():
        raise HTTPException(status_code=400, detail="Path must point to a file")

    encoded = request.content.encode("utf-8")
    if len(encoded) > WORKSPACE_WRITE_SIZE_LIMIT:
        raise HTTPException(status_code=400, detail="File content too large (max 1MB)")

    try:
        validation = validate_workspace_text_content(target, request.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write(request.content)

    payload = {
        "workspace_id": workspace_id,
        "workspace_path": str(workspace),
        "path": format_workspace_path(target, workspace),
        "bytes_written": len(encoded),
        "lines": request.content.count("\n") + 1,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if validation:
        payload["validation"] = validation
    return payload


def read_workspace_spreadsheet_for_workspace(
    workspace_id: str,
    path: str,
    sheet: Optional[str] = None,
    *,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Read spreadsheet summary data from a workspace addressed directly by workspace id."""
    _workspace_record, workspace, _run = get_workspace_route_target(workspace_id, create=True)
    target = resolve_workspace_relative_path_from_root(workspace, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        summary = load_spreadsheet_summary(target, sheet=sheet)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "workspace_id": workspace_id,
        "workspace_path": str(workspace),
        "path": format_workspace_path(target, workspace),
        **summary,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return payload


def build_workspace_archive_response(workspace_id: str) -> Response:
    """Return a zip archive response for one workspace."""
    _workspace_record, workspace, _run = get_workspace_route_target(workspace_id, create=True)
    archive_entries: List[tuple[pathlib.Path, str]] = []
    for file_path in workspace.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(workspace).as_posix()
        if rel_path == ".ai" or rel_path.startswith(".ai/"):
            continue
        archive_entries.append((file_path, rel_path))

    if not archive_entries:
        return Response(status_code=204)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path, rel_path in archive_entries:
            zf.write(file_path, arcname=rel_path)

    headers = {"Content-Disposition": 'attachment; filename="workspace.zip"'}
    return Response(content=archive.getvalue(), media_type="application/zip", headers=headers)


@app.get("/api/workspaces/{workspace_id}/files")
async def list_workspace_files_by_workspace(workspace_id: str, path: str = ""):
    return list_workspace_files_for_workspace(workspace_id, path)


@app.get("/api/workspaces/{workspace_id}/file")
async def read_workspace_file_by_workspace(workspace_id: str, path: str):
    return read_workspace_file_for_workspace(workspace_id, path)


@app.get("/api/workspaces/{workspace_id}/file/view")
async def view_workspace_file_by_workspace(workspace_id: str, path: str):
    _workspace_record, workspace, _run = get_workspace_route_target(workspace_id, create=True)
    target = resolve_workspace_relative_path_from_root(workspace, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type)


@app.get("/api/workspaces/{workspace_id}/file/download")
async def download_workspace_file_by_workspace(workspace_id: str, path: str):
    _workspace_record, workspace, _run = get_workspace_route_target(workspace_id, create=True)
    target = resolve_workspace_relative_path_from_root(workspace, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.post("/api/workspaces/{workspace_id}/file")
async def write_workspace_file_by_workspace(workspace_id: str, request: WorkspaceFileUpdateRequest):
    return write_workspace_file_for_workspace(workspace_id, request)


@app.get("/api/workspaces/{workspace_id}/spreadsheet")
async def read_workspace_spreadsheet_by_workspace(workspace_id: str, path: str, sheet: Optional[str] = None):
    return read_workspace_spreadsheet_for_workspace(workspace_id, path, sheet=sheet)


@app.post("/api/workspaces/{workspace_id}/upload")
async def upload_workspace_files_by_workspace(
    workspace_id: str,
    files: List[UploadFile] = File(...),
    target_path: str = Form("."),
):
    return await upload_workspace_files_for_workspace(workspace_id, files, target_path)


@app.get("/api/workspaces/{workspace_id}/download")
async def download_workspace_by_workspace(workspace_id: str):
    return build_workspace_archive_response(workspace_id)


@app.get("/api/workspace/{conversation_id}")
async def get_workspace_info(conversation_id: str):
    workspace_id = get_workspace_id_for_conversation(conversation_id, create=False)
    if not workspace_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return build_workspace_info_payload(workspace_id, conversation_id=conversation_id)


@app.post("/api/workspace/{conversation_id}/upload")
async def upload_workspace_files(
    conversation_id: str,
    files: List[UploadFile] = File(...),
    target_path: str = Form("."),
):
    workspace_id = get_workspace_id_for_conversation(conversation_id)
    return await upload_workspace_files_for_workspace(workspace_id, files, target_path, conversation_id=conversation_id)


@app.get("/api/workspace/{conversation_id}/files")
async def list_workspace_files(conversation_id: str, path: str = ""):
    workspace_id = get_workspace_id_for_conversation(conversation_id, create=False)
    if not workspace_id:
        workspace = get_workspace_path(conversation_id, create=False)
        return {
            "conversation_id": conversation_id,
            "workspace_id": "",
            "run_id": None,
            "workspace_path": str(workspace),
            "path": ".",
            "items": [],
        }
    return list_workspace_files_for_workspace(workspace_id, path, conversation_id=conversation_id)


@app.get("/api/workspace/{conversation_id}/file")
async def read_workspace_file(conversation_id: str, path: str):
    workspace_id = get_workspace_id_for_conversation(conversation_id)
    return read_workspace_file_for_workspace(workspace_id, path, conversation_id=conversation_id)


@app.get("/api/workspace/{conversation_id}/file/view")
async def view_workspace_file(conversation_id: str, path: str):
    workspace_id = get_workspace_id_for_conversation(conversation_id)
    return await view_workspace_file_by_workspace(workspace_id, path)


@app.get("/api/workspace/{conversation_id}/file/download")
async def download_workspace_file(conversation_id: str, path: str):
    workspace_id = get_workspace_id_for_conversation(conversation_id)
    return await download_workspace_file_by_workspace(workspace_id, path)


@app.post("/api/workspace/{conversation_id}/file")
async def write_workspace_file(conversation_id: str, request: WorkspaceFileUpdateRequest):
    workspace_id = get_workspace_id_for_conversation(conversation_id)
    return write_workspace_file_for_workspace(workspace_id, request, conversation_id=conversation_id)


@app.get("/api/workspace/{conversation_id}/spreadsheet")
async def read_workspace_spreadsheet(conversation_id: str, path: str, sheet: Optional[str] = None):
    workspace_id = get_workspace_id_for_conversation(conversation_id)
    return read_workspace_spreadsheet_for_workspace(workspace_id, path, sheet=sheet, conversation_id=conversation_id)


@app.get("/api/workspace/{conversation_id}/download")
async def download_workspace(conversation_id: str):
    workspace_id = get_workspace_id_for_conversation(conversation_id)
    return build_workspace_archive_response(workspace_id)


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


SUPPORTED_TOOL_NAMES = {
    "workspace.list_files",
    "workspace.grep",
    "workspace.read_file",
    "workspace.patch_file",
    "workspace.run_command",
    "workspace.render",
    "spreadsheet.describe",
    "conversation.search_history",
    "web.search",
    "web.fetch_page",
}

SUPPORTED_TOOL_NAME_PATTERN = re.compile(
    r'"name"\s*:\s*"(?P<name>'
    r'workspace\.(?:list_files|grep|read_file|patch_file|run_command|render)'
    r'|spreadsheet\.describe'
    r'|conversation\.search_history'
    r'|web\.(?:search|fetch_page)'
    r')"'
)


def extract_leaked_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    """Detect an internal tool payload that leaked into visible assistant text."""
    cleaned = strip_stream_special_tokens(raw).strip()
    if not cleaned:
        return None

    looks_like_tool_payload = (
        cleaned.startswith("<tool_call>")
        or (cleaned.startswith("{") and '"name"' in cleaned and '"arguments"' in cleaned)
    )
    if not looks_like_tool_payload:
        return None

    try:
        call = parse_tool_call(cleaned)
    except Exception:
        call = None

    if call and call.get("name") in SUPPORTED_TOOL_NAMES:
        return call

    name_match = SUPPORTED_TOOL_NAME_PATTERN.search(cleaned)
    if not name_match:
        return None

    arguments: Dict[str, Any] = {}
    path_match = re.search(r'"path"\s*:\s*"(?P<path>[^"]+)"', cleaned)
    if path_match:
        arguments["path"] = path_match.group("path")

    return {
        "id": "leaked_tool_call",
        "name": name_match.group("name"),
        "arguments": arguments,
    }


def format_leaked_tool_call_message(call: Dict[str, Any], features: "FeatureFlags") -> str:
    """Turn an internal tool payload leak into a short, user-facing explanation."""
    name = str(call.get("name", "")).strip()
    arguments = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
    path = str(arguments.get("path", "")).strip()
    quoted_path = f"`{path}`" if path else "the workspace"

    if name == "workspace.patch_file":
        if not features.workspace_write:
            return (
                f"I tried to use an internal file-editing tool to write {quoted_path}, "
                "but workspace editing was not available for this turn. Retry with a coding/edit request "
                "if you want me to create it directly, or ask me to paste the code inline instead."
            )
        if not is_tool_permission_allowlisted(features, "tool:workspace.write"):
            return (
                f"I tried to edit {quoted_path}, but file-edit permission was not approved for this chat. "
                "Retry and approve workspace edits if you want me to change files directly."
            )
        return (
            f"I tried to create or edit {quoted_path}, but an internal tool payload leaked into the reply "
            "instead of a normal answer. Please retry the request."
        )

    if name == "workspace.run_command":
        if not features.workspace_run_commands:
            return (
                "I tried to use an internal command runner, but command execution was not available for this turn. "
                "Retry with a verification-oriented request if you want me to run checks automatically."
            )
        return (
            "I tried to run an internal verification command, but the tool payload leaked into the visible reply. "
            "Please retry the request."
        )

    if name.startswith("workspace.") or name == "spreadsheet.describe":
        return (
            "I accidentally exposed an internal workspace tool call instead of a normal reply. "
            "Please retry the request."
        )

    if name.startswith("web."):
        if not features.web_search:
            return (
                "I tried to use an internal web tool, but web search was not available for this turn. "
                "Retry with a current-events or citation-oriented request if you want live sources."
            )
        if not is_tool_permission_allowlisted(features, "tool:web.search"):
            return (
                "I tried to use an internal web tool, but web access was not approved for this chat. "
                "Retry and approve web search if you want current sources."
            )
        return (
            "I accidentally exposed an internal web tool call instead of a normal reply. "
            "Please retry the request."
        )

    return (
        "I accidentally exposed an internal tool call instead of a normal reply. "
        "Please retry the request."
    )


def workspace_list_files_result(conversation_id: str, rel_path: str = "") -> Dict[str, Any]:
    """List files inside the conversation workspace."""
    workspace = get_workspace_path(conversation_id)
    target = resolve_workspace_relative_path(conversation_id, rel_path)

    if not target.exists():
        raise ValueError("Path not found")
    if not target.is_dir():
        raise ValueError("Path is not a directory")

    target_rel_path = format_workspace_path(target, workspace)
    target_is_hidden = workspace_rel_path_is_hidden(target_rel_path)

    items = []
    for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            item_rel_path = format_workspace_path(item, workspace)
            if workspace_rel_path_is_hidden(item_rel_path) and not target_is_hidden:
                continue
            items.append(build_workspace_listing_item(item, workspace))
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

        rel_file_path = format_workspace_path(file_path, workspace)
        if is_pdf_path(file_path):
            try:
                source = ensure_document_index(conversation_id, rel_file_path)
            except Exception:
                skipped_files += 1
                continue
            if source.get("status") != "ready":
                skipped_files += 1
                continue
            pdf_matches = fetch_ranked_document_candidates(
                conversation_id,
                [rel_file_path],
                cleaned_query,
                hyde_query="",
                limit=max(1, safe_limit - len(matches)),
            )
            if pdf_matches:
                files_scanned += 1
            for match in pdf_matches:
                excerpt = build_query_excerpt(match.get("content", ""), cleaned_query, window=240)
                page_start = match.get("page_start")
                page_end = match.get("page_end")
                page_label = ""
                if page_start:
                    page_label = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
                matches.append({
                    "path": rel_file_path,
                    "line": page_start or 1,
                    "column": 1,
                    "text": truncate_output(
                        f"{match.get('section_title') or 'PDF excerpt'}{f' ({page_label})' if page_label else ''}: {excerpt}".strip(),
                        limit=240,
                    ),
                    "before": "",
                    "after": "",
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
    rel_file_path = format_workspace_path(target, workspace)
    return build_workspace_file_result(
        target,
        conversation_id=conversation_id,
        rel_path=rel_file_path,
        text_limit=TOOL_RESULT_TEXT_LIMIT,
    )


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

    validation = validate_workspace_text_content(target, updated)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write(updated)

    workspace = get_workspace_path(conversation_id)
    result = {
        "path": format_workspace_path(target, workspace),
        "edits_applied": edits_applied,
        "bytes_written": len(encoded),
        "created": not target_exists,
    }
    if validation:
        result["validation"] = validation
    return result


def workspace_render_result(conversation_id: str, html: str, title: str = "") -> Dict[str, Any]:
    """Write HTML content to a preview file and return the path for the viewer."""
    if not isinstance(html, str) or not html.strip():
        raise ValueError("html is required and must be a non-empty string")
    encoded = html.encode("utf-8")
    if len(encoded) > WORKSPACE_WRITE_SIZE_LIMIT:
        raise ValueError("HTML content too large (max 1MB)")
    filename = (title.strip().replace("/", "_").replace(" ", "_")[:60] + ".html") if title.strip() else "_preview.html"
    target = resolve_workspace_relative_path(conversation_id, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write(html)
    workspace = get_workspace_path(conversation_id)
    return {
        "path": format_workspace_path(target, workspace),
        "bytes_written": len(encoded),
        "render": True,
    }


async def workspace_run_command_result(
    conversation_id: str,
    command: Any,
    cwd: str = ".",
    features: Optional[FeatureFlags] = None,
) -> Dict[str, Any]:
    """Run a command inside the conversation workspace."""
    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty array of strings")
    if not all(isinstance(part, str) and part.strip() for part in command):
        raise ValueError("command entries must be non-empty strings")

    cwd_path = resolve_workspace_relative_path(conversation_id, cwd or ".")
    if not cwd_path.exists():
        raise ValueError("cwd not found")
    if not cwd_path.is_dir():
        raise ValueError("cwd must be a directory")

    workspace = get_workspace_path(conversation_id)
    before_snapshot = capture_workspace_file_snapshot(conversation_id)

    try:
        normalized_command = await normalize_command_for_managed_python(conversation_id, command)
    except Exception as exc:
        raise ValueError(f"Managed Python environment setup failed: {exc}") from exc

    validate_workspace_command(conversation_id, normalized_command, cwd_path, features)

    try:
        process = await asyncio.create_subprocess_exec(
            *normalized_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd_path),
            env=build_workspace_command_env(conversation_id),
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Command not found: {normalized_command[0]}") from exc

    timeout_seconds = command_runtime_timeout_seconds(normalized_command)
    try:
        if timeout_seconds is None:
            stdout, stderr = await process.communicate()
        else:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            await process.wait()
        raise ValueError(f"Command timed out after {timeout_seconds:g}s")
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            await process.wait()
        raise

    after_snapshot = capture_workspace_file_snapshot(conversation_id)
    artifact_items = detect_workspace_artifact_changes(before_snapshot, after_snapshot)
    primary_artifact = choose_primary_workspace_artifact(artifact_items)
    primary_path = str((primary_artifact or {}).get("path", "")).strip()
    primary_kind = str((primary_artifact or {}).get("content_kind", "")).strip().lower()
    open_path = primary_path if primary_path and is_previewable_workspace_kind(primary_kind) else ""
    return {
        "stdout": truncate_output(stdout.decode("utf-8", errors="replace")),
        "stderr": truncate_output(stderr.decode("utf-8", errors="replace")),
        "returncode": process.returncode,
        "cwd": format_workspace_path(cwd_path, workspace),
        "path": primary_path,
        "open_path": open_path,
        "items": artifact_items,
        "artifacts_detected": len(artifact_items),
    }

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
    lexical_rows: List[Dict[str, Any]]

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
                "feedback": message_feedback_value(row[1], row[4]),
                "fts_rank": row[5],
            }
            for row in c.fetchall()
        ], cleaned_query, max(safe_limit * 4, 12))
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

    semantic_rows = fetch_semantic_message_candidates(
        conversation_id,
        cleaned_query,
        limit=max(MESSAGE_RETRIEVAL_SEMANTIC_LIMIT, safe_limit * 4),
    )

    candidate_rows: Dict[int, Dict[str, Any]] = {}
    for position, row in enumerate(lexical_rows):
        candidate_rows[int(row["id"])] = {
            **row,
            "lexical_rank": position,
            "semantic_rank": None,
            "semantic_similarity": None,
        }
    for position, row in enumerate(semantic_rows):
        item = candidate_rows.setdefault(
            int(row["id"]),
            {
                **row,
                "lexical_rank": None,
                "semantic_rank": None,
                "semantic_similarity": None,
            },
        )
        item["semantic_rank"] = position
        item["semantic_similarity"] = row.get("semantic_similarity")
        item.setdefault("feedback", row.get("feedback", ""))

    if candidate_rows:
        recency_order = {
            row_id: index
            for index, row_id in enumerate(
                item["id"] for item in sorted(candidate_rows.values(), key=lambda item: item.get("timestamp", ""))
            )
        }
        total_candidates = max(1, len(candidate_rows))
        ranked_rows = []
        for item in candidate_rows.values():
            score = 0.0
            score += 160.0 * reciprocal_rank_fusion_bonus(item.get("lexical_rank"))
            score += 185.0 * reciprocal_rank_fusion_bonus(item.get("semantic_rank"))
            score += 1.1 * max(0.0, float(item.get("semantic_similarity") or 0.0))
            score += exact_text_hit_bonus(cleaned_query, item.get("content", ""))
            score += calculate_message_relevance_score(
                item,
                cleaned_query,
                recency_order.get(int(item["id"]), 0),
                total_candidates,
            )
            ranked_rows.append({**item, "score": round(score, 4)})
        ranked_rows.sort(key=lambda item: (item.get("score", 0.0), item.get("timestamp", "")), reverse=True)
        ranked_rows = ranked_rows[:safe_limit]
    else:
        ranked_rows = []

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


CURATED_WEB_SEARCH_PRESETS = [
    {
        "scope": "philosophy_reference",
        "domains": ["plato.stanford.edu", "iep.utm.edu"],
        "keywords": {
            "philosophy", "philosopher", "ethics", "epistemology", "metaphysics",
            "existentialism", "stoicism", "nihilism", "phenomenology", "logic",
            "consciousness", "ontology", "aesthetics", "kant", "aristotle",
            "plato", "descartes", "nietzsche", "hegel", "wittgenstein",
        },
    },
]


def curated_source_domains() -> set[str]:
    """Return the set of domains managed by curated search presets."""
    domains: set[str] = set()
    for preset in CURATED_WEB_SEARCH_PRESETS:
        for domain in preset.get("domains", []):
            normalized = sanitize_web_domain(str(domain))
            if normalized:
                domains.add(normalized)
    return domains


def is_curated_source_active(domain: str) -> bool:
    """Return whether a curated source domain is currently eligible for auto-fan-out."""
    normalized = sanitize_web_domain(domain)
    if not normalized or normalized not in curated_source_domains():
        return True
    state = CURATED_SOURCE_HEALTH.get(normalized, {})
    disabled_until = _parse_iso_datetime(state.get("disabled_until"))
    if disabled_until and disabled_until > datetime.now():
        return False
    return True


def record_curated_source_failure(domain: str, error: str = "") -> None:
    """Temporarily disable repeatedly failing curated sources."""
    normalized = sanitize_web_domain(domain)
    if not normalized or normalized not in curated_source_domains():
        return
    state = dict(CURATED_SOURCE_HEALTH.get(normalized, {}))
    failures = int(state.get("failures", 0)) + 1
    state["failures"] = failures
    state["last_error"] = truncate_output(str(error or "").strip(), limit=240)
    state["last_failure_at"] = utcnow_iso()
    if failures >= CURATED_SOURCE_FAILURE_THRESHOLD:
        state["disabled_until"] = (
            datetime.now() + timedelta(minutes=CURATED_SOURCE_DISABLE_MINUTES)
        ).isoformat()
        logger.warning("Temporarily disabled curated source %s after %s failures", normalized, failures)
    CURATED_SOURCE_HEALTH[normalized] = state


def record_curated_source_success(domain: str) -> None:
    """Clear failure state for curated sources that recover."""
    normalized = sanitize_web_domain(domain)
    if not normalized or normalized not in curated_source_domains():
        return
    CURATED_SOURCE_HEALTH[normalized] = {
        "failures": 0,
        "last_success_at": utcnow_iso(),
    }


def curated_search_plans_for_query(query: str, limit: int) -> List[Dict[str, Any]]:
    """Return extra authoritative search plans for queries matching curated domains."""
    words = set(re.findall(r"[a-z0-9_+-]+", str(query or "").lower()))
    plans: List[Dict[str, Any]] = []
    for preset in CURATED_WEB_SEARCH_PRESETS:
        if not words.intersection(set(preset.get("keywords", set()))):
            continue
        active_domains = [
            domain for domain in list(preset.get("domains", []))
            if is_curated_source_active(domain)
        ]
        if not active_domains:
            continue
        plans.append({
            "scope": str(preset["scope"]),
            "query": apply_web_domain_filters(str(query or "").strip(), active_domains),
            "limit": min(4, limit),
            "domains": active_domains,
        })
    return plans


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


async def duckduckgo_search_results(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    search_scope: str,
) -> List[Dict[str, Any]]:
    """Run one DuckDuckGo HTML search and normalize the result set."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ai-chat/1.0; +https://localhost)",
    }
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
    while len(results) < limit:
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
            "search_scope": search_scope,
            "search_rank": len(results) + 1,
            "title": title,
            "url": raw_url,
            "domain": domain,
            "snippet": snippet,
        })
        cursor = match.end()

    return results


async def run_scoped_web_search(
    client: httpx.AsyncClient,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one scoped search without letting a single source failure break the whole fan-out."""
    try:
        results = await duckduckgo_search_results(client, plan["query"], int(plan["limit"]), str(plan["scope"]))
        return {
            "scope": str(plan["scope"]),
            "domains": list(plan.get("domains", [])),
            "query": str(plan["query"]),
            "results": results,
            "error": "",
        }
    except Exception as exc:
        for domain in plan.get("domains", []):
            record_curated_source_failure(str(domain), str(exc))
        logger.warning("Scoped web search failed for %s: %s", plan.get("scope"), exc)
        return {
            "scope": str(plan["scope"]),
            "domains": list(plan.get("domains", [])),
            "query": str(plan["query"]),
            "results": [],
            "error": str(exc),
        }


async def web_search_result(query: str, limit: int = 5, domains: Any = None) -> Dict[str, Any]:
    """Fetch a web result set, plus Wikipedia and Reddit context when helpful."""
    cleaned_query = (query or "").strip()
    if len(cleaned_query) < 2:
        raise ValueError("query must be at least 2 characters")

    safe_limit = max(1, min(int(limit), 8))
    domain_filters = []
    if isinstance(domains, list):
        for domain in domains[:5]:
            normalized = sanitize_web_domain(str(domain))
            if normalized and normalized not in domain_filters:
                domain_filters.append(normalized)

    search_plans = []
    effective_query = apply_web_domain_filters(cleaned_query, domain_filters)
    if domain_filters:
        search_plans.append({
            "scope": "domain_filtered",
            "query": effective_query,
            "limit": safe_limit,
            "domains": domain_filters,
        })
    else:
        search_plans.extend([
            {
                "scope": "web",
                "query": cleaned_query,
                "limit": safe_limit,
                "domains": [],
            },
            {
                "scope": "wikipedia",
                "query": apply_web_domain_filters(cleaned_query, ["wikipedia.org"]),
                "limit": min(3, safe_limit),
                "domains": ["wikipedia.org"],
            },
            {
                "scope": "reddit",
                "query": apply_web_domain_filters(cleaned_query, ["reddit.com"]),
                "limit": min(3, safe_limit),
                "domains": ["reddit.com"],
            },
        ])
        search_plans.extend(curated_search_plans_for_query(cleaned_query, safe_limit))

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0), follow_redirects=True) as client:
        scoped_runs = await asyncio.gather(*[
            run_scoped_web_search(client, plan)
            for plan in search_plans
        ])

    requested_domains = set(domain_filters)
    result_sets = []
    results: List[Dict[str, Any]] = []
    overall_position = 1
    skipped_searches = []
    for plan, scoped_run in zip(search_plans, scoped_runs):
        scope_results = list(scoped_run.get("results", []))
        if scoped_run.get("error"):
            skipped_searches.append({
                "scope": scoped_run.get("scope", plan["scope"]),
                "domains": list(scoped_run.get("domains", plan.get("domains", []))),
                "error": truncate_output(str(scoped_run.get("error", "")), limit=240),
            })
        for item in scope_results:
            item["title_match_score"] = compute_title_match_score(cleaned_query, str(item.get("title", "")))
        scope_results.sort(
            key=lambda item: (
                -float(item.get("title_match_score", 0.0)),
                -int(bool(item.get("domain")) and item.get("domain") in requested_domains),
                int(item.get("search_rank", 9999)),
            )
        )
        for scope_index, item in enumerate(scope_results, start=1):
            item["scope_position"] = scope_index
            item["position"] = overall_position
            overall_position += 1
        result_sets.append({
            "scope": plan["scope"],
            "query": plan["query"],
            "domains": list(plan.get("domains", [])),
            "count": len(scope_results),
            "results": scope_results,
        })
        results.extend(scope_results)

        if not scoped_run.get("error"):
            for domain in plan.get("domains", []):
                record_curated_source_success(str(domain))

    return {
        "query": cleaned_query,
        "effective_query": effective_query,
        "domains": domain_filters,
        "searches_run": [plan["scope"] for plan in search_plans],
        "skipped_searches": skipped_searches,
        "result_sets": result_sets,
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
    source_domain = normalize_web_domain(raw_url)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=True) as client:
            resp = await client.get(raw_url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            body = resp.text
    except Exception as exc:
        record_curated_source_failure(source_domain, str(exc))
        raise

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
    record_curated_source_success(page_domain)
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


def compact_tool_text(value: Any, limit: int = 96) -> str:
    """Normalize tool-facing text for concise activity summaries."""
    return truncate_output(re.sub(r"\s+", " ", str(value or "")).strip(), limit=limit)


def pluralize_tool_count(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Render a small count phrase like '3 files'."""
    safe_count = max(0, int(count or 0))
    noun = singular if safe_count == 1 else (plural or f"{singular}s")
    return f"{safe_count} {noun}"


def tool_path_preview(value: Any, fallback: str = ".", limit: int = 72) -> str:
    """Return a readable path-like value for activity lines."""
    preview = compact_tool_text(value, limit=limit)
    return preview or fallback


def command_preview_text(command: Any, limit: int = 96) -> str:
    """Render a stable shell-like preview of a structured argv command."""
    if isinstance(command, list):
        parts = [str(part).strip() for part in command if str(part).strip()]
        if not parts:
            return "command"
        preview = shlex.join(parts)
    else:
        preview = str(command or "").strip()
    preview = preview.replace("\n", " ")
    return compact_tool_text(preview, limit=limit) or "command"


def tool_activity_label(name: str) -> str:
    """Map tool names to compact UI activity labels."""
    if name == "workspace.run_command":
        return "Command"
    if name in {"workspace.list_files", "workspace.grep", "workspace.read_file", "spreadsheet.describe"}:
        return "Explore"
    if name == "workspace.patch_file":
        return "Edit"
    if name == "workspace.render":
        return "Render"
    if name == "conversation.search_history":
        return "History"
    if name in {"web.search", "web.fetch_page"}:
        return "Web"
    return "Tool"


def tool_status_summary(name: str, arguments: Dict[str, Any]) -> str:
    """Human-readable tool activity for the UI."""
    if name == "workspace.list_files":
        return f"Exploring {tool_path_preview(arguments.get('path', '.'), '.')}"
    if name == "workspace.grep":
        target = tool_path_preview(arguments.get("path", "."), ".")
        query = compact_tool_text(arguments.get("query", ""), limit=60)
        return f"Searching {target} for {query or 'text'}"
    if name == "workspace.read_file":
        return f"Opening {tool_path_preview(arguments.get('path', ''), 'file')}"
    if name == "workspace.patch_file":
        return f"Editing {tool_path_preview(arguments.get('path', ''), 'file')}"
    if name == "workspace.run_command":
        return f"Running {command_preview_text(arguments.get('command', []))}"
    if name == "workspace.render":
        return f"Rendering {tool_path_preview(arguments.get('title', 'HTML preview'), 'HTML preview')}"
    if name == "spreadsheet.describe":
        target = tool_path_preview(arguments.get("path", ""), "spreadsheet")
        sheet = compact_tool_text(arguments.get("sheet", ""), limit=40)
        return f"Inspecting {target}{f' [{sheet}]' if sheet else ''}"
    if name == "conversation.search_history":
        query = compact_tool_text(arguments.get("query", ""), limit=60)
        return f"Searching chat history for {query or 'prior messages'}"
    if name == "web.search":
        query = compact_tool_text(arguments.get("query", ""), limit=60)
        domains = arguments.get("domains", [])
        if isinstance(domains, list) and domains:
            domain_preview = ", ".join(compact_tool_text(item, limit=30) for item in domains[:3] if compact_tool_text(item, limit=30))
            return f"Searching the web for {query or 'sources'} on {domain_preview}" if domain_preview else f"Searching the web for {query or 'sources'}"
        return f"Searching the web for {query or 'sources'}"
    if name == "web.fetch_page":
        return f"Fetching {tool_path_preview(arguments.get('url', ''), 'page', limit=88)}"
    return name


def tool_result_preview(
    result: Dict[str, Any],
    name: str = "",
    arguments: Optional[Dict[str, Any]] = None,
) -> str:
    """Compact preview of a tool result for the UI."""
    arguments = arguments or {}
    if not result.get("ok"):
        details = result.get("details", {})
        if isinstance(details, dict) and details.get("type") == "patch_mismatch":
            return (
                f"Patch mismatch in {details.get('path', '')}: "
                f"expected {details.get('expected_count', '?')} match(es), "
                f"found {details.get('actual_count', '?')}"
            )
        if name == "workspace.run_command":
            return f"Command failed: {command_preview_text(arguments.get('command', []))}"
        return compact_tool_text(result.get("error", "Tool failed"), limit=200) or "Tool failed"

    payload = result.get("result", {})
    if not isinstance(payload, dict):
        return "Completed"

    if name == "workspace.run_command" or "returncode" in payload:
        returncode = int(payload.get("returncode", 0) or 0)
        cwd = tool_path_preview(payload.get("cwd", arguments.get("cwd", ".")), ".")
        command = command_preview_text(arguments.get("command", []))
        artifact_items = [item for item in payload.get("items", []) if isinstance(item, dict)]
        artifact_preview = ""
        if artifact_items:
            preview_paths = [
                tool_path_preview(item.get("path", ""), "")
                for item in artifact_items[:2]
                if tool_path_preview(item.get("path", ""), "")
            ]
            if preview_paths:
                if len(artifact_items) == 1:
                    artifact_preview = f" and created {preview_paths[0]}"
                elif len(artifact_items) == 2:
                    artifact_preview = f" and created {preview_paths[0]} and {preview_paths[1]}"
                else:
                    artifact_preview = (
                        f" and created {len(artifact_items)} artifacts "
                        f"including {preview_paths[0]} and {preview_paths[1]}"
                    )
        if returncode == 0:
            return f"Ran {command} in {cwd}{artifact_preview}"
        return f"Ran {command} in {cwd}{artifact_preview} (exit {returncode})"

    if name == "workspace.list_files" or "items" in payload:
        items = payload.get("items", [])
        files = sum(1 for item in items if isinstance(item, dict) and item.get("type") == "file")
        directories = sum(1 for item in items if isinstance(item, dict) and item.get("type") == "directory")
        target = tool_path_preview(payload.get("path", arguments.get("path", ".")), ".")
        if files and directories:
            return f"Explored {pluralize_tool_count(files, 'file')} and {pluralize_tool_count(directories, 'folder')} in {target}"
        if files:
            return f"Explored {pluralize_tool_count(files, 'file')} in {target}"
        if directories:
            return f"Explored {pluralize_tool_count(directories, 'folder')} in {target}"
        return f"Explored {target}"

    if name == "workspace.grep" or "grep_matches" in payload:
        matched_files = int(payload.get("matched_files", 0) or 0)
        files_scanned = int(payload.get("files_scanned", 0) or 0)
        match_count = int(payload.get("count", 0) or 0)
        query = compact_tool_text(payload.get("query", arguments.get("query", "")), limit=60) or "text"
        target = tool_path_preview(payload.get("path", arguments.get("path", ".")), ".")
        if matched_files:
            match_note = f" ({pluralize_tool_count(match_count, 'match', 'matches')})" if match_count else ""
            return f"Explored {pluralize_tool_count(matched_files, 'file')} for {query} in {target}{match_note}"
        scanned = pluralize_tool_count(files_scanned, "file")
        return f"Searched {scanned} in {target} for {query}"

    if name == "workspace.read_file" or "content" in payload:
        path = tool_path_preview(payload.get("path", arguments.get("path", "")), "file")
        lines = int(payload.get("lines", 0) or 0)
        line_note = f" ({pluralize_tool_count(lines, 'line')})" if lines else ""
        return f"Opened {path}{line_note}"

    if name == "workspace.render" or payload.get("render"):
        return f"Rendered {tool_path_preview(payload.get('path', 'preview'), 'preview')} in the workspace viewer"

    if name == "workspace.patch_file" or "edits_applied" in payload:
        path = tool_path_preview(payload.get("path", arguments.get("path", "")), "file")
        validation = payload.get("validation", {})
        validation_note = ""
        if isinstance(validation, dict) and validation.get("type") == "json" and validation.get("status") == "valid":
            validation_note = ", valid JSON"
        if payload.get("created"):
            return f"Created {path} ({payload.get('bytes_written', 0)} bytes{validation_note})"
        return f"Edited {path} ({pluralize_tool_count(payload.get('edits_applied', 0), 'change')}{validation_note})"

    if name == "spreadsheet.describe" or "sheet_names" in payload:
        sheet = compact_tool_text(payload.get("sheet", ""), limit=40)
        path = tool_path_preview(payload.get("path", arguments.get("path", "")), "spreadsheet")
        sheet_note = f" [{sheet}]" if sheet else ""
        return (
            f"Explored {path}{sheet_note}: "
            f"{pluralize_tool_count(payload.get('row_count', 0), 'row')}, "
            f"{pluralize_tool_count(payload.get('column_count', 0), 'column')}"
        )

    if name == "conversation.search_history" or "matches" in payload:
        query = compact_tool_text(payload.get("query", arguments.get("query", "")), limit=60) or "history"
        return f"Searched chat history for {query} ({pluralize_tool_count(payload.get('count', 0), 'match', 'matches')})"

    if name == "web.search" or "results" in payload:
        query = compact_tool_text(payload.get("query", arguments.get("query", "")), limit=60) or "sources"
        return f"Searched the web for {query} ({pluralize_tool_count(payload.get('count', 0), 'result')})"

    if name == "web.fetch_page" or "final_url" in payload:
        title = compact_tool_text(payload.get("title", ""), limit=80)
        target = title or tool_path_preview(payload.get("final_url", payload.get("url", "")), "page", limit=88)
        return f"Fetched {target}"

    return "Completed"


def tool_loop_progressed(tool_results: List[Dict[str, Any]]) -> bool:
    """Return whether the current tool batch produced any successful work."""
    return any(entry.get("result", {}).get("ok") for entry in tool_results)


def summarize_tool_loop_progress(tool_results: List[Dict[str, Any]], limit: int = 8) -> str:
    """Create a compact progress summary that can seed a continuation batch."""
    if not tool_results:
        return "No tool actions were recorded yet."

    successful = sum(1 for entry in tool_results if entry.get("result", {}).get("ok"))
    touched_paths: List[str] = []
    commands: List[str] = []
    recent: List[str] = []
    for entry in tool_results:
        call = entry.get("call", {})
        result = entry.get("result", {})
        payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
        path = str(payload.get("path", "")).strip()
        if path and path not in touched_paths:
            touched_paths.append(path)
        if call.get("name") == "workspace.run_command":
            preview = command_preview_text(call.get("arguments", {}).get("command", []))
            if preview and preview not in commands:
                commands.append(preview)
        preview = tool_result_preview(result, str(call.get("name", "")).strip(), call.get("arguments", {}))
        if preview:
            recent.append(preview)

    lines = [
        f"Completed {successful} of {len(tool_results)} tool actions so far.",
    ]
    if touched_paths:
        lines.append("Touched paths: " + ", ".join(touched_paths[:6]))
    if commands:
        lines.append("Commands run: " + ", ".join(commands[:4]))
    if recent:
        lines.append("Recent tool results:")
        lines.extend(f"- {item}" for item in recent[-limit:])
    return "\n".join(lines)


def build_tool_loop_resume_message(progress_summary: str, batch_index: int) -> str:
    """Prompt the model to continue from saved progress instead of restarting."""
    return (
        f"Continue the same task from the saved workspace state (resume batch {batch_index + 1}).\n\n"
        f"Progress so far:\n{progress_summary}\n\n"
        "Do not restart from scratch or repeat completed edits unless re-checking is necessary. "
        "Use tools only for the remaining work, then finish with the best grounded answer."
    )


def should_attempt_capability_recovery(
    response: str,
    *,
    request_text: str = "",
    allowed_tools: Optional[List[str]] = None,
    blocked_on_permission: bool = False,
) -> bool:
    """Return whether a draft likely refused or handed off a capability that should trigger one more tool pass."""
    if blocked_on_permission:
        return False
    text = " ".join(str(response or "").strip().split())
    if not text:
        return False
    if any(marker in text.lower() for marker in CAPABILITY_RECOVERY_SKIP_MARKERS):
        return False
    if DIRECT_TOOL_RECOVERY_REFUSAL_PATTERN.search(text):
        return True
    allowed = {tool_name for tool_name in (allowed_tools or []) if isinstance(tool_name, str)}
    if not allowed.intersection({"workspace.run_command", "workspace.render"}):
        return False
    return request_demands_agent_execution(request_text) and response_hands_execution_back_to_user(text)


def build_capability_recovery_message(
    draft_response: str,
    allowed_tools: List[str],
    progress_summary: str = "",
) -> str:
    """Prompt the model to continue instead of stopping at a false limitation or hand-off."""
    lines = [
        "Continue the same task from the current workspace state instead of refusing or handing off a capability that is available in this turn.",
        "Available tools for this turn: " + ", ".join(allowed_tools),
    ]
    if progress_summary.strip():
        lines.extend([
            "",
            "Progress so far:",
            progress_summary.strip(),
        ])
    lines.extend([
        "",
        "Draft that needs correction:",
        draft_response.strip() or "(empty draft)",
        "",
        "Use the available tools when they help. Do not claim a capability is unavailable unless approval was denied or a tool result in this conversation proved the blocker. "
        "If the request asked for a concrete output shape such as a PDF, chart, rendered page, runnable app, mobile-ready fix, or real command output, keep going until that exact shape is delivered or a blocker is verified.",
    ])
    return "\n".join(lines)


async def maybe_recover_tool_outcome_from_capability_refusal(
    websocket: WebSocket,
    conversation_id: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: Optional[FeatureFlags],
    allowed_tools: Optional[List[str]],
    outcome: ToolLoopOutcome,
    *,
    status_prefix: str = "",
    max_steps: Optional[int] = None,
    activity_phase: str = "tool",
    activity_step_label: Optional[str] = None,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
    continuation_limit: int = 0,
) -> Optional[ToolLoopOutcome]:
    """Give a tool-enabled draft one recovery pass when it falsely refuses an available capability."""
    request_text = str(history[-1].get("content", "")) if history else ""
    if not allowed_tools or not should_attempt_capability_recovery(
        outcome.final_text,
        request_text=request_text,
        allowed_tools=list(allowed_tools),
        blocked_on_permission=outcome.blocked_on_permission,
    ):
        return None

    progress_summary = summarize_tool_loop_progress(outcome.tool_results)
    await send_activity_event(
        websocket,
        "evaluate",
        "Recover",
        "The draft either claimed a limitation or handed work back even though this turn still has relevant tools. Continuing from saved state.",
        step_label=activity_step_label,
    )
    recovery_history = list(history)
    recovery_history.append({
        "role": "user",
        "content": build_capability_recovery_message(
            outcome.final_text,
            list(allowed_tools),
            progress_summary,
        ),
    })
    return await run_resumable_tool_loop(
        websocket,
        conversation_id,
        recovery_history,
        system_prompt,
        max_tokens,
        features=features,
        allowed_tools=allowed_tools,
        status_prefix=status_prefix,
        max_steps=max_steps,
        activity_phase=activity_phase,
        activity_step_label=activity_step_label,
        workflow_execution=workflow_execution,
        continuation_limit=max(0, int(continuation_limit or 0)),
        allow_capability_recovery=False,
    )


def tool_loop_continuation_limit_for_request(
    message: str,
    allowed_tools: List[str],
    activity_phase: str = "respond",
) -> int:
    """Choose how many extra tool batches are worth attempting automatically."""
    if not allowed_tools or TOOL_LOOP_MAX_CONTINUATIONS <= 0:
        return 0

    intent = classify_workspace_intent(message)
    wants_illustrative_output = request_prefers_illustrative_output(message)
    if activity_phase == "execute":
        return TOOL_LOOP_MAX_CONTINUATIONS
    if intent == "broad_write":
        return TOOL_LOOP_MAX_CONTINUATIONS
    if intent == "focused_write":
        return min(2, TOOL_LOOP_MAX_CONTINUATIONS)
    if wants_illustrative_output and any(name.startswith("workspace.") for name in allowed_tools):
        return min(2, TOOL_LOOP_MAX_CONTINUATIONS)
    if activity_phase in {"inspect", "verify", "synthesize"}:
        return min(1, TOOL_LOOP_MAX_CONTINUATIONS)
    if any(name.startswith("workspace.") for name in allowed_tools):
        return min(1, TOOL_LOOP_MAX_CONTINUATIONS)
    return 0


async def run_resumable_tool_loop(
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
    workflow_execution: Optional[WorkflowExecutionContext] = None,
    continuation_limit: int = 0,
    allow_capability_recovery: bool = True,
) -> ToolLoopOutcome:
    """Run the tool loop in a few resumable batches before asking the user to continue."""
    base_history = list(history)
    combined_results: List[Dict[str, Any]] = []
    progress_summary = ""

    for batch_index in range(max(0, int(continuation_limit or 0)) + 1):
        batch_history = list(base_history)
        if batch_index > 0 and progress_summary:
            batch_history.append({
                "role": "user",
                "content": build_tool_loop_resume_message(progress_summary, batch_index),
            })

        outcome = await run_tool_loop(
            websocket,
            conversation_id,
            batch_history,
            system_prompt,
            max_tokens,
            features=features,
            allowed_tools=allowed_tools,
            status_prefix=status_prefix,
            max_steps=max_steps,
            activity_phase=activity_phase,
            activity_step_label=activity_step_label,
            workflow_execution=workflow_execution,
        )
        combined_results.extend(outcome.tool_results)

        if outcome.blocked_on_permission:
            outcome.tool_results = combined_results
            return outcome

        if allow_capability_recovery and allowed_tools and not outcome.hit_limit:
            recovery_outcome = await maybe_recover_tool_outcome_from_capability_refusal(
                websocket,
                conversation_id,
                batch_history,
                system_prompt,
                max_tokens,
                features,
                allowed_tools,
                outcome,
                status_prefix=status_prefix,
                max_steps=max_steps,
                activity_phase=activity_phase,
                activity_step_label=activity_step_label,
                workflow_execution=workflow_execution,
                continuation_limit=max(0, int(continuation_limit or 0)) - batch_index,
            )
            if recovery_outcome is not None:
                combined_results.extend(recovery_outcome.tool_results)
                outcome = recovery_outcome
                if outcome.blocked_on_permission:
                    outcome.tool_results = combined_results
                    return outcome

        if not outcome.hit_limit:
            outcome.tool_results = combined_results
            return outcome

        progress_summary = summarize_tool_loop_progress(combined_results)
        if batch_index >= max(0, int(continuation_limit or 0)) or not tool_loop_progressed(outcome.tool_results):
            outcome.tool_results = combined_results
            if combined_results:
                outcome.final_text = build_tool_loop_hard_limit_message(progress_summary)
            return outcome

        await send_activity_event(
            websocket,
            activity_phase,
            "Continue",
            "Continuing automatically from the saved workspace state.",
            step_label=activity_step_label,
        )

    return ToolLoopOutcome(
        final_text="I couldn't continue the tool run automatically.",
        tool_results=combined_results,
        hit_limit=True,
    )


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
        elif name == "workspace.render":
            result = workspace_render_result(
                conversation_id,
                str(arguments.get("html", "")),
                str(arguments.get("title", "")),
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


def build_permission_denied_result(
    call: Dict[str, Any],
    request: PermissionApprovalRequest,
) -> Dict[str, Any]:
    """Format a denied approval with a user-facing pause message."""
    label = "command" if request.approval_target == "command" else "tool"
    return {
        "id": call.get("id", ""),
        "ok": False,
        "error_code": "permission_denied",
        "error": f"{label.capitalize()} permission '{request.key}' was denied for this chat",
        "message_to_user": render_permission_blocked_message(request),
    }


async def ensure_tool_permission(
    websocket: WebSocket,
    conversation_id: str,
    call: Dict[str, Any],
    features: Optional[FeatureFlags],
    *,
    step_label: Optional[str] = None,
) -> tuple[bool, Optional[PermissionApprovalRequest]]:
    """Request runtime approval for a gated tool when the chat hasn't approved it yet."""
    if features is None:
        return True, None

    request = build_tool_permission_request(conversation_id, call)
    if request is None:
        return True, None

    if features.auto_approve_tool_permissions:
        if request.approval_target == "command":
            remember_approved_command(features, request.key)
        else:
            remember_approved_tool_permission(features, request.key)
        return True, request

    if request.approval_target == "command":
        arguments = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
        command = arguments.get("command")
        command_list = command if isinstance(command, list) else []
        cwd_value = str(arguments.get("cwd", ".") or ".")
        try:
            cwd_path = resolve_workspace_relative_path(conversation_id, cwd_value)
        except Exception:
            return True, None
        if is_command_allowlisted(conversation_id, command_list, cwd_path, features):
            return True, request
    elif is_tool_permission_allowlisted(features, request.key):
        return True, request

    approved = await wait_for_permission_approval(
        websocket,
        conversation_id,
        request,
        step_label=step_label,
    )
    if approved:
        if request.approval_target == "command":
            remember_approved_command(features, request.key)
        else:
            remember_approved_tool_permission(features, request.key)
    return approved, request


async def emit_direct_tool_call(
    websocket: WebSocket,
    conversation_id: str,
    call: Dict[str, Any],
    *,
    features: Optional[FeatureFlags] = None,
    status_prefix: str = "",
    activity_phase: str = "respond",
    activity_step_label: Optional[str] = None,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> Dict[str, Any]:
    """Execute one deterministic tool call while mirroring the normal tool-loop UI events."""
    tool_name = call["name"]
    tool_arguments = call.get("arguments", {})
    approved, permission_request = await ensure_tool_permission(
        websocket,
        conversation_id,
        call,
        features,
        step_label=activity_step_label,
    )
    if not approved and permission_request is not None:
        result = build_permission_denied_result(call, permission_request)
        result_summary = tool_result_preview(result, tool_name, tool_arguments)
        await websocket.send_json({
            "type": "tool_result",
            "name": tool_name,
            "ok": False,
            "content": f"{status_prefix}{result_summary}",
            "arguments": tool_arguments,
            "payload": {"error": result.get("error", "Tool failed")},
        })
        await send_activity_event(
            websocket,
            "error",
            tool_activity_label(tool_name),
            f"{status_prefix}{result_summary}",
            step_label=activity_step_label,
        )
        record_workflow_step(
            workflow_execution,
            step_name=activity_step_label or activity_phase or call.get("name", ""),
            call=call,
            result=result,
            latency_ms=0,
        )
        return result

    activity_label = tool_activity_label(tool_name)
    start_summary = tool_status_summary(tool_name, tool_arguments)
    await websocket.send_json({
        "type": "tool_start",
        "name": tool_name,
        "content": f"{status_prefix}{start_summary}",
        "arguments": tool_arguments,
    })
    await send_activity_event(
        websocket,
        activity_phase,
        activity_label,
        f"{status_prefix}{start_summary}",
        step_label=activity_step_label,
    )
    started = time.perf_counter()
    result = await execute_tool_call(conversation_id, call, features)
    latency_ms = int((time.perf_counter() - started) * 1000)
    result_summary = tool_result_preview(result, tool_name, tool_arguments)
    await websocket.send_json({
        "type": "tool_result",
        "name": tool_name,
        "ok": result.get("ok", False),
        "content": f"{status_prefix}{result_summary}",
        "arguments": tool_arguments,
        "payload": result.get("result", {}) if result.get("ok") else {
            "error": result.get("error", "Tool failed"),
        },
    })
    await send_activity_event(
        websocket,
        activity_phase if result.get("ok") else "error",
        activity_label,
        f"{status_prefix}{result_summary}",
        step_label=activity_step_label,
    )
    record_workflow_step(
        workflow_execution,
        step_name=activity_step_label or activity_phase or call.get("name", ""),
        call=call,
        result=result,
        latency_ms=latency_ms,
    )
    return result


def build_disallowed_phase_tool_result(
    call: Dict[str, Any],
    allowed_tools: Optional[List[str]],
    activity_phase: str,
) -> Dict[str, Any]:
    """Tell the model a requested tool is out of scope for the current phase without leaking it to the user."""
    allowed = [tool_name for tool_name in (allowed_tools or []) if isinstance(tool_name, str) and tool_name.strip()]
    allowed_summary = ", ".join(allowed) if allowed else "(none)"
    return {
        "id": call.get("id", ""),
        "ok": False,
        "error_code": "tool_not_allowed_in_phase",
        "error": (
            f"Tool {call.get('name', '')} is not available during the {activity_phase or 'current'} phase. "
            f"Allowed tools: {allowed_summary}"
        ),
        "allowed_tools": allowed,
        "phase": activity_phase or "",
    }


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
    workflow_execution: Optional[WorkflowExecutionContext] = None,
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
            return await finalize_tool_loop_answer(cleaned, tool_results, max_tokens, features)

        if not call:
            return await finalize_tool_loop_answer(cleaned, tool_results, max_tokens, features)

        if allowed_tools and call["name"] not in allowed_tools:
            if should_upgrade_to_workspace_execution(call, features, activity_phase):
                return ToolLoopOutcome(
                    final_text=(
                        f"The next step needs workspace execution because it requested {call['name']}. "
                        "Switch into the build flow and continue from the current task."
                    ),
                    tool_results=tool_results,
                    requested_phase_upgrade="workspace_execution",
                    requested_tool_name=call["name"],
                )
            disallowed_result = build_disallowed_phase_tool_result(call, allowed_tools, activity_phase)
            tool_results.append({
                "call": call,
                "result": disallowed_result,
                "internal_only": True,
            })
            record_workflow_step(
                workflow_execution,
                step_name=activity_step_label or activity_phase or call.get("name", ""),
                call=call,
                result=disallowed_result,
                latency_ms=0,
            )
            messages.append({"role": "assistant", "content": cleaned})
            messages.append({
                "role": "user",
                "content": (
                    "<tool_result>\n"
                    + json.dumps(disallowed_result, ensure_ascii=False)
                    + "\n</tool_result>\n"
                    + "Use only the allowed tools for this phase, or answer directly if no more tools are needed."
                ),
            })
            continue

        approved, permission_request = await ensure_tool_permission(
            websocket,
            conversation_id,
            call,
            features,
            step_label=activity_step_label,
        )
        if not approved and permission_request is not None:
            denied_result = build_permission_denied_result(call, permission_request)
            denied_summary = tool_result_preview(denied_result, call["name"], call.get("arguments", {}))
            await websocket.send_json({
                "type": "tool_result",
                "name": call["name"],
                "ok": False,
                "content": f"{status_prefix}{denied_summary}",
                "arguments": call.get("arguments", {}),
                "payload": {"error": denied_result["error"]},
            })
            await send_activity_event(
                websocket,
                "error",
                tool_activity_label(call["name"]),
                f"{status_prefix}{denied_summary}",
                step_label=activity_step_label,
            )
            tool_results.append({
                "call": call,
                "result": denied_result,
            })
            record_workflow_step(
                workflow_execution,
                step_name=activity_step_label or activity_phase or call.get("name", ""),
                call=call,
                result=denied_result,
                latency_ms=0,
            )
            return ToolLoopOutcome(
                final_text=render_permission_blocked_message(permission_request),
                tool_results=tool_results,
                blocked_on_permission=True,
                blocked_permission_key=permission_request.key,
            )

        start_summary = tool_status_summary(call["name"], call.get("arguments", {}))
        await websocket.send_json({
            "type": "tool_start",
            "name": call["name"],
            "content": f"{status_prefix}{start_summary}",
            "arguments": call.get("arguments", {}),
        })
        await send_activity_event(
            websocket,
            activity_phase,
            tool_activity_label(call["name"]),
            f"{status_prefix}{start_summary}",
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
                started = time.perf_counter()
                result = await execute_tool_call(conversation_id, call, features)
                latency_ms = int((time.perf_counter() - started) * 1000)
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
            started = time.perf_counter()
            result = await execute_tool_call(conversation_id, call, features)
            latency_ms = int((time.perf_counter() - started) * 1000)
        result_summary = tool_result_preview(result, call["name"], call.get("arguments", {}))
        await websocket.send_json({
            "type": "tool_result",
            "name": call["name"],
            "ok": result.get("ok", False),
            "content": f"{status_prefix}{result_summary}",
            "arguments": call.get("arguments", {}),
            "payload": result.get("result", {}) if result.get("ok") else {
                "error": result.get("error", "Tool failed"),
            },
        })
        await send_activity_event(
            websocket,
            activity_phase if result.get("ok") else "error",
            tool_activity_label(call["name"]),
            f"{status_prefix}{result_summary}",
            step_label=activity_step_label,
        )
        tool_results.append({
            "call": call,
            "result": result,
        })
        record_workflow_step(
            workflow_execution,
            step_name=activity_step_label or activity_phase or call.get("name", ""),
            call=call,
            result=result,
            latency_ms=latency_ms,
        )

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
                        tool_activity_label(auto_call["name"]),
                        f"{status_prefix}{auto_verify.get('label', tool_status_summary(auto_call['name'], auto_call['arguments']))}",
                        step_label=activity_step_label,
                    )
                    auto_started = time.perf_counter()
                    auto_result = await execute_tool_call(conversation_id, auto_call, features)
                    auto_latency_ms = int((time.perf_counter() - auto_started) * 1000)
                    auto_result_summary = tool_result_preview(auto_result, auto_call["name"], auto_call.get("arguments", {}))
                    await websocket.send_json({
                        "type": "tool_result",
                        "name": auto_call["name"],
                        "ok": auto_result.get("ok", False),
                        "content": f"{status_prefix}{auto_result_summary}",
                        "arguments": auto_call["arguments"],
                        "payload": auto_result.get("result", {}) if auto_result.get("ok") else {
                            "error": auto_result.get("error", "Tool failed"),
                        },
                    })
                    await send_activity_event(
                        websocket,
                        "verify" if auto_result.get("ok") else "error",
                        tool_activity_label(auto_call["name"]),
                        f"{status_prefix}{auto_result_summary}",
                        step_label=activity_step_label,
                    )
                    tool_results.append({
                        "call": auto_call,
                        "result": auto_result,
                        "auto_generated": True,
                    })
                    record_workflow_step(
                        workflow_execution,
                        step_name=activity_step_label or "auto_verify",
                        call=auto_call,
                        result=auto_result,
                        latency_ms=auto_latency_ms,
                        auto_generated=True,
                    )
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
        hit_limit=True,
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
    read_verbs = {"inspect", "read", "open", "show", "list", "search", "find", "grep", "explain", "review", "render", "preview", "display", "view"}
    write_verbs = {
        "edit", "change", "update", "patch", "refactor", "create", "write", "add",
        "delete", "remove", "rename", "run", "execute", "test", "build", "compile",
        "debug", "fix", "implement", "generate", "scaffold", "make", "making",
        "start", "starting", "draft", "wire", "wiring", "setup", "set",
        "tweak", "adjust", "modify", "improve", "revise",
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


def should_upgrade_to_workspace_execution(
    call: Dict[str, Any],
    features: Optional["FeatureFlags"],
    activity_phase: str,
) -> bool:
    """Allow direct-answer turns to pivot into execution when the next step clearly needs it."""
    if activity_phase != "respond" or not features or not features.agent_tools:
        return False

    name = str(call.get("name", "")).strip()
    if name in {"workspace.patch_file", "workspace.render"}:
        return bool(features.workspace_write)
    if name == "workspace.run_command":
        return bool(features.workspace_write and features.workspace_run_commands)
    return False


def tool_loop_step_limit_for_request(message: str, allowed_tools: List[str]) -> int:
    """Choose a small tool-loop budget based on request scope."""
    intent = classify_workspace_intent(message)
    repo_scale = request_targets_current_repo(message)
    wants_illustrative_output = request_prefers_illustrative_output(message)
    steps = TOOL_LOOP_MAX_STEPS
    if intent == "focused_read":
        steps = min(steps, 3)
    elif intent == "focused_write":
        steps = min(steps, 5)
    elif intent == "broad_read":
        steps = min(steps, 6)
    elif intent == "broad_write":
        steps = min(steps, 8)
    else:
        steps = min(steps, 2 if allowed_tools else TOOL_LOOP_MAX_STEPS)
    if is_fast_profile_active():
        if intent == "focused_write":
            steps = min(steps, 4)
        else:
            steps = min(steps, 6 if intent == "broad_write" else 4)
    if wants_illustrative_output and any(name.startswith("workspace.") for name in allowed_tools):
        steps = max(steps, 7 if is_fast_profile_active() else 10)
    if repo_scale and any(name.startswith("workspace.") for name in allowed_tools):
        steps = max(steps, 10 if is_fast_profile_active() else 12)
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


def deep_execute_step_limit_for_request(message: str) -> int:
    """Give build-phase workspace work a bit more room before pausing."""
    intent = classify_workspace_intent(message)
    wants_illustrative_output = request_prefers_illustrative_output(message)
    if is_fast_profile_active():
        base = 6 if intent == "broad_write" else 5
        if wants_illustrative_output:
            base = max(base, 7)
        return base
    if intent == "broad_write":
        base = 9
    elif intent == "focused_write":
        base = 8
    else:
        base = 6
    if wants_illustrative_output:
        base = max(base, 11)
    return base


def deep_execute_token_budget(max_tokens: int, message: str) -> int:
    """Reserve a slightly larger response window for inspect/patch/verify substeps."""
    intent = classify_workspace_intent(message)
    if is_fast_profile_active():
        if intent == "broad_write":
            return min(max_tokens, 1792)
        if intent == "focused_write":
            return min(max_tokens, 1536)
        return min(max_tokens, 1280)
    if intent == "broad_write":
        return min(max_tokens, 2560)
    if intent == "focused_write":
        return min(max_tokens, 2048)
    return min(max_tokens, 1536)


def select_enabled_tools(
    conversation_id: str,
    message: str,
    features: FeatureFlags,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[str]:
    """Return the small set of tools worth exposing for this request."""
    allowed: List[str] = []
    intent = classify_workspace_intent(message)
    render_requested = should_offer_workspace_render(message, history, features)
    workspace_requested = render_requested or should_use_workspace_tools(conversation_id, message, features) or (
        features.agent_tools
        and features.workspace_write
        and intent in {"focused_write", "broad_write"}
    )
    if workspace_requested:
        if intent == "focused_read":
            path_refs = extract_workspace_path_references(message)
            if path_refs and any(pathlib.Path(path).suffix.lower() in SPREADSHEET_SUPPORTED_EXTENSIONS for path in path_refs):
                allowed.append("spreadsheet.describe")
            allowed.extend(allowed_workspace_tools(features, include_write=False, include_render=render_requested))
        elif intent == "focused_write":
            allowed.extend(allowed_workspace_tools(features, include_write=True, include_render=render_requested))
        else:
            allowed.extend(allowed_workspace_tools(features, include_write=True, include_render=render_requested))
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
    """Heuristic: enter workspace mode when local files are likely to ground the answer or change."""
    if not features.agent_tools:
        return False

    text = (message or "").strip().lower()
    if not text:
        return False

    if extract_artifact_references(message):
        return True

    if any(marker in text for marker in ATTACHMENT_CONTEXT_MARKERS):
        return True

    if message_requests_workspace_render(message):
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
        repo_change_terms = {
            "prompt", "prompts", "behavior", "flow", "workflow", "feature", "features",
            "logic", "planner", "plan", "harness", "tool", "tools", "approval",
        }
        if words & WORKSPACE_SIGNAL_VERBS and words & repo_change_terms:
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
ATTACHMENT_CONTEXT_MARKERS = (
    "attached files are available in the workspace:",
    "relevant extracted context from attachments:",
    "attachment overview:",
)
EXPLICIT_WEB_SEARCH_PHRASES = (
    "search the web",
    "search web",
    "search online",
    "use search",
    "look up",
    "browse",
    "google",
    "find sources",
    "add citations",
    "cite sources",
    "with citations",
    "with sources",
)
EXPLICIT_HISTORY_SEARCH_PHRASES = (
    "search history",
    "search the history",
    "search our chat",
    "search this chat",
    "conversation history",
    "earlier in this chat",
    "what did we say",
    "what did i say",
)
RENDER_CONTEXT_HINTS = {
    "html", "page", "site", "dashboard", "report", "visualization", "viewer", "preview", "render",
}
HTML_SNIPPET_PATTERN = re.compile(
    r"<!doctype html|<html\b|<head\b|<body\b|<main\b|<section\b|<article\b|<div\b|<style\b|```html",
    re.IGNORECASE,
)
DIRECT_TOOL_RECOVERY_REFUSAL_PATTERN = re.compile(
    r"\b("
    r"can't|cannot|don't have|do not have|unable to|limited by|no built-in|"
    r"not available|prevent me from|server-side restriction|don't support|do not support|"
    r"not accessible|would require|toolset"
    r")\b",
    re.IGNORECASE,
)
EXECUTION_HANDOFF_PATTERN = re.compile(
    r"("
    r"to run (?:the app|it) locally:"
    r"|install dependencies:"
    r"|visit `?http://localhost"
    r"|start the server:"
    r"|to play:"
    r"|download the file to your local machine"
    r"|run it with `python"
    r"|run it with python"
    r")",
    re.IGNORECASE,
)

CAPABILITY_RECOVERY_SKIP_MARKERS = (
    "approve it for this chat",
    "permission was denied",
    "approval was denied",
    "paused here while waiting for approval",
)
AGENT_EXECUTION_EXPECTATION_PHRASES = (
    "run it yourself",
    "show me the real command output",
    "show me the actual output",
    "show me the rendered chart",
    "show me the chart artifact",
    "keep going until the artifact is real",
    "show the result without handing execution back to me",
    "render it in the viewer",
    "render the homepage in the viewer",
    "keep iterating until",
    "do not stop at just writing html",
    "do not stop at html",
    "convert it to pdf",
    "give me the final artifact path",
    "actually works on mobile",
)
LOCAL_INSTRUCTIONS_REQUEST_PHRASES = (
    "how do i run",
    "how to run",
    "run locally",
    "run this locally",
    "tell me how to run",
    "show me how to run",
)


def explicit_history_lookup_requested(message: str) -> bool:
    """Return whether the user explicitly asked to search earlier chat context."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    return any(phrase in text for phrase in EXPLICIT_HISTORY_SEARCH_PHRASES)


def explicit_web_search_requested(message: str) -> bool:
    """Return whether the user explicitly asked for web research or sourced citations."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    if any(phrase in text for phrase in EXPLICIT_WEB_SEARCH_PHRASES):
        return True
    words = set(re.findall(r"[a-z0-9_+-]+", text))
    return bool({"citations", "citation", "sources", "source", "references"} & words)


def message_requests_workspace_render(message: str) -> bool:
    """Return whether the user is asking to preview or render HTML-like content."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    words = set(re.findall(r"[a-z0-9_+-]+", text))
    if any(phrase in text for phrase in ("render this page", "show this page", "open in the viewer", "using your viewer")):
        return True
    if {"render", "preview", "viewer"} & words and (RENDER_CONTEXT_HINTS & words or "viewer tools" in text):
        return True
    if {"display", "show", "view", "open"} & words and {"html", "page", "site", "dashboard", "report"} & words:
        return True
    return False


def history_contains_renderable_html(history: Optional[List[Dict[str, str]]], limit: int = 6) -> bool:
    """Check recent chat history for inline HTML that could be rendered."""
    if not history:
        return False
    for item in history[-limit:]:
        if str(item.get("role", "")) != "assistant":
            continue
        if HTML_SNIPPET_PATTERN.search(str(item.get("content", ""))):
            return True
    return False


def should_offer_workspace_render(
    message: str,
    history: Optional[List[Dict[str, str]]],
    features: FeatureFlags,
) -> bool:
    """Expose the HTML render tool when the prompt strongly implies it."""
    if not features.agent_tools:
        return False
    if not message_requests_workspace_render(message):
        return False
    text = " ".join((message or "").strip().lower().split())
    if HTML_SNIPPET_PATTERN.search(text):
        return True
    return history_contains_renderable_html(history) or bool(RENDER_CONTEXT_HINTS & set(re.findall(r"[a-z0-9_+-]+", text)))


def should_offer_local_rag(message: str, features: FeatureFlags) -> bool:
    """Use lightweight retrieval only for prompts that imply recall/search."""
    if not features.local_rag and not explicit_history_lookup_requested(message):
        return False
    text = (message or "").strip().lower()
    if len(text) < 3:
        return False
    words = set(re.findall(r"[a-z0-9_+-]+", text))
    return bool(words & LOCAL_RAG_HINTS)


def should_offer_web_search(message: str, features: FeatureFlags) -> bool:
    """Only offer web search when explicitly useful."""
    text = (message or "").strip().lower()
    if len(text) < 3:
        return False
    if explicit_web_search_requested(message):
        return True
    if not features.web_search:
        return False
    if "http://" in text or "https://" in text:
        return True
    if re.search(r"\b[a-z0-9.-]+\.[a-z]{2,}\b", text):
        return True
    words = set(re.findall(r"[a-z0-9_+-]+", text))
    return bool(words & WEB_SEARCH_HINTS)


def request_demands_agent_execution(message: str) -> bool:
    """Return whether the user asked the assistant to run, render, or verify the result itself."""
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return False
    if any(phrase in text for phrase in LOCAL_INSTRUCTIONS_REQUEST_PHRASES):
        return False
    if any(phrase in text for phrase in AGENT_EXECUTION_EXPECTATION_PHRASES):
        return True

    words = set(re.findall(r"[a-z0-9_+-]+", text))
    if request_prefers_illustrative_output(text) and words & {"run", "execute", "show", "prove", "render"}:
        return True
    if {"run", "execute"} & words and {"yourself", "actual", "real", "output"} & words:
        return True
    if {"render", "viewer", "preview"} & words and {"html", "homepage", "page", "chart", "artifact", "report"} & words:
        return True
    if {"chart", "artifact", "output"} & words and {"show", "actual", "real"} & words:
        return True
    if "mobile" in words and {"work", "works", "working"} & words:
        return True
    if "pdf" in words and {"convert", "final", "artifact", "path"} & words:
        return True
    return False


def response_hands_execution_back_to_user(response: str) -> bool:
    """Return whether a draft shifts runnable work back onto the user."""
    text = str(response or "").strip()
    if not text:
        return False
    return bool(EXECUTION_HANDOFF_PATTERN.search(text))


def direct_response_tool_recovery_candidates(
    conversation_id: str,
    message: str,
    history: Optional[List[Dict[str, str]]],
    response: str,
    features: FeatureFlags,
) -> List[str]:
    """Give direct-mode replies one recovery pass when they falsely refuse an available capability."""
    if (
        not DIRECT_TOOL_RECOVERY_REFUSAL_PATTERN.search(str(response or ""))
        and not (
            request_demands_agent_execution(message)
            and response_hands_execution_back_to_user(response)
        )
    ):
        return []

    allowed: List[str] = []
    render_requested = should_offer_workspace_render(message, history, features)
    workspace_requested = render_requested or should_use_workspace_tools(conversation_id, message, features)
    if workspace_requested:
        allowed.extend(allowed_workspace_tools(features, include_write=False, include_render=render_requested))
    if should_offer_local_rag(message, features):
        allowed.append("conversation.search_history")
    if should_offer_web_search(message, features):
        allowed.extend(["web.search", "web.fetch_page"])

    deduped: List[str] = []
    for tool_name in allowed:
        if tool_name not in deduped:
            deduped.append(tool_name)
    return deduped


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
    recent_feedback_block = ""
    if session.recent_product_feedback_summary:
        recent_feedback_block = (
            "Recent product feedback to treat as failure signals for this pass:\n"
            f"{session.recent_product_feedback_summary}\n\n"
            f"Feedback digest artifact:\n[[artifact:{session.recent_product_feedback_artifact_path}]]\n\n"
        )
    inspect_history = session.history_messages() + [{
        "role": "user",
        "content": (
            f"Conversation context:\n{session.context or '(none)'}\n\n"
            f"{recent_feedback_block}"
            f"Deterministic workspace snapshot:\n{format_workspace_snapshot(session.workspace_snapshot)}\n\n"
            f"User request:\n{session.task_request or session.message}\n\n"
            "Inspect the workspace and gather the most relevant facts before planning. "
            "Use tools when needed, then return a concise fact summary."
        ),
    }]
    inspect_tools = allowed_workspace_tools(session.features, include_write=False)
    outcome = await run_resumable_tool_loop(
        session.websocket,
        session.conversation_id,
        inspect_history,
        f"{session.system_prompt}\n\n{DEEP_INSPECT_SYSTEM_PROMPT}",
        min(session.max_tokens, 1536),
        features=session.features,
        allowed_tools=inspect_tools,
        status_prefix="Inspect: ",
        max_steps=4,
        activity_phase="inspect",
        workflow_execution=session.workflow_execution,
        continuation_limit=tool_loop_continuation_limit_for_request(
            session.task_request or session.message,
            inspect_tools,
            activity_phase="inspect",
        ),
    )
    if outcome.blocked_on_permission:
        snapshot_facts = format_workspace_snapshot(session.workspace_snapshot)
        session.workspace_facts = (
            f"Grounded workspace snapshot:\n{snapshot_facts}"
            if snapshot_facts else ""
        )
        session.pause_reason = PAUSE_REASON_COMMAND_APPROVAL
        session.draft_response = outcome.final_text
        await persist_task_state(session)
        return session.workspace_facts
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
    confirmation = render_deep_confirmation(session.task_request or session.message, session.workspace_enabled)
    if confirmation:
        await send_reasoning_note(session.websocket, confirmation, phase="inspect")
    return confirmation


async def deep_plan_step_subtasks(
    session: DeepSession,
    step: str,
    step_index: int,
    total_steps: int,
) -> Dict[str, Any]:
    """Generate or reuse a nested micro-plan for one top-level build step."""
    key = step_plan_cache_key(step_index)
    cached = session.step_subplans.get(key)
    if isinstance(cached, dict) and cached:
        normalized = normalize_step_subplan(cached, step)
        session.step_subplans[key] = normalized
        return normalized

    step_label = f"Step {step_index + 1}: {step}"
    await send_reasoning_note(
        session.websocket,
        f"Planning inside {step_label}. I'm only decomposing this step so execution stays scoped.",
        step_label=step_label,
        phase="plan",
    )
    await send_activity_event(
        session.websocket,
        "plan",
        "Subplan",
        f"Breaking build step {step_index + 1} into smaller executable substeps.",
        step_label=step_label,
    )

    if is_fast_profile_active():
        subplan = build_heuristic_step_subplan(session, step, step_index, total_steps)
    else:
        messages = [
            {"role": "system", "content": STEP_DECOMPOSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request:\n{session.task_request or session.message}\n\n"
                    f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
                    f"Workspace snapshot:\n{format_workspace_snapshot(session.workspace_snapshot)}\n\n"
                    f"Top-level plan strategy:\n{session.plan.get('strategy', '(none)')}\n\n"
                    "Completed top-level build notes:\n"
                    + ("\n".join(f"{idx + 1}. {summary}" for idx, summary in enumerate(session.build_step_summaries)) or "(none)")
                    + "\n\n"
                    f"Current build step ({step_index + 1}/{total_steps}):\n{step}"
                ),
            },
        ]
        raw = await vllm_chat_complete(messages, max_tokens=384, temperature=0.1)
        try:
            subplan = normalize_step_subplan(parse_json_object(strip_stream_special_tokens(raw)), step)
        except Exception:
            logger.warning("Nested step planner returned invalid JSON; using heuristic subplan instead.")
            subplan = build_heuristic_step_subplan(session, step, step_index, total_steps)

    normalized = normalize_step_subplan(subplan, step)
    session.step_subplans[key] = normalized
    await persist_task_state(session)
    return normalized


async def deep_build_workspace(session: DeepSession) -> DeepBuildResult:
    """Execute the remaining planned build steps and persist progress after each one."""
    if not session.workspace_enabled:
        session.pause_reason = ""
        session.build_summary = ""
        session.build_step_summaries = []
        session.changed_files = []
        return DeepBuildResult(summary=session.build_summary, build_complete=True)
    if not session.features.workspace_write:
        session.pause_reason = PAUSE_REASON_WRITE_BLOCKED
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
        return DeepBuildResult(
            summary=session.build_summary,
            build_complete=False,
            pause_reason=session.pause_reason,
        )
    await send_activity_event(
        session.websocket,
        "execute",
        "Execute",
        "Executing the approved plan from the task board.",
    )
    session.pause_reason = ""
    agent_a = session.plan["agent_a"]
    changed_files: List[str] = []
    if session.changed_files:
        changed_files.extend(path for path in session.changed_files if path not in changed_files)
    if not session.resumed:
        session.build_step_summaries = []
        session.step_subplans = {}
        session.step_substep_summaries = {}
        session.step_substep_reports = {}
        session.step_reports = []
    session.workspace_snapshot = capture_workspace_snapshot(session.conversation_id)
    steps = session.plan.get("builder_steps", [])
    build_tools = allowed_workspace_tools(session.features, include_write=True)
    start_index = min(len(session.build_step_summaries), len(steps))
    await send_build_steps(
        session.websocket,
        steps,
        completed_count=start_index,
        active_index=(start_index if start_index < len(steps) else None),
        step_details=build_step_details_payload(
            session,
            steps,
            completed_count=start_index,
            active_index=(start_index if start_index < len(steps) else None),
        ),
    )

    if start_index >= len(steps):
        await send_activity_event(
            session.websocket,
            "execute",
            "Execute",
            "Build checklist already completed.",
        )
        await send_build_steps(
            session.websocket,
            steps,
            completed_count=len(steps),
            active_index=None,
            step_details=build_step_details_payload(
                session,
                steps,
                completed_count=len(steps),
                active_index=None,
            ),
        )
        await persist_task_board(session, active_build_step=None)
        await persist_task_state(session)
        session.build_summary = "\n".join(session.build_step_summaries) or "(build already completed)"
        session.changed_files = changed_files
        return DeepBuildResult(summary=session.build_summary, build_complete=True)

    for idx in range(start_index, len(steps)):
        step = steps[idx]
        step_label = f"Step {idx + 1}: {step}"
        step_subplan = await deep_plan_step_subtasks(session, step, idx, len(steps))
        step_key = step_plan_cache_key(idx)
        completed_substeps = step_substep_summaries_for_index(session, idx)
        completed_substep_reports = step_substep_reports_for_index(session, idx)
        substeps = [
            str(item).strip()
            for item in step_subplan.get("substeps", [])
            if str(item).strip()
        ]
        next_substep_index = len(completed_substeps) if len(completed_substeps) < len(substeps) else None

        await send_build_steps(
            session.websocket,
            steps,
            completed_count=idx,
            active_index=idx,
            step_details=build_step_details_payload(
                session,
                steps,
                completed_count=idx,
                active_index=idx,
                active_substep_index=next_substep_index,
            ),
        )
        await persist_task_board(
            session,
            active_build_step=idx,
            active_substep_index=next_substep_index,
        )

        step_tool_calls = 0
        step_successful_tools = 0
        step_successful_commands = 0
        touched_paths: List[str] = []

        for sub_idx in range(len(completed_substeps), len(substeps)):
            current_substep = substeps[sub_idx]
            substep_label = f"Step {idx + 1}.{sub_idx + 1}: {current_substep}"
            await send_reasoning_note(
                session.websocket,
                f"Current focus is {substep_label}. I'm staying inside this substep until it is complete or blocked.",
                step_label=substep_label,
                phase="execute",
            )
            await send_activity_event(
                session.websocket,
                "execute",
                "Execute",
                (
                    f"Working on substep {sub_idx + 1} of {len(substeps)} "
                    f"inside build step {idx + 1} of {len(steps)}."
                ),
                step_label=substep_label,
            )
            await persist_task_board(
                session,
                active_build_step=idx,
                active_substep_index=sub_idx,
            )

            build_history = session.history_messages() + [{
                "role": "user",
                "content": (
                    f"Conversation context:\n{session.context or '(none)'}\n\n"
                    f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
                    f"Current workspace snapshot:\n{format_workspace_snapshot(session.workspace_snapshot)}\n\n"
                    f"User request:\n{session.task_request or session.message}\n\n"
                    f"Planned deliverable:\n{session.plan.get('deliverable', '(none)')}\n\n"
                    f"Task board artifact:\n[[artifact:{session.task_board_path}]]\n\n"
                    "Completed build step notes:\n"
                    + ("\n".join(f"{i + 1}. {summary}" for i, summary in enumerate(session.build_step_summaries)) or "(none)")
                    + "\n\n"
                    f"Current build step ({idx + 1}/{len(steps)}):\n{step}\n\n"
                    "Nested step plan:\n"
                    + format_step_subplan_progress(
                        step_subplan,
                        completed_substeps,
                        active_substep_index=sub_idx,
                        prefix=f"{idx + 1}.",
                    )
                    + "\n\n"
                    f"Current substep ({sub_idx + 1}/{len(substeps)}):\n{current_substep}\n\n"
                    f"Build role: {agent_a.get('role', 'builder')}\n"
                    f"Build task:\n{agent_a.get('prompt', '')}\n\n"
                    "Focus on the current substep only. Treat any later substeps as future work inside this same top-level build step. "
                    "Use the task board as external memory instead of trying to carry the whole plan in-context. "
                    "On each loop, either gather missing evidence, update a durable artifact, or verify the current substep. "
                    "Use tools to inspect, patch, and validate as needed for this substep. "
                    "When done, return a concise substep result covering what changed, any artifact paths, and caveats that matter for later verification."
                ),
            }]
            outcome = await run_resumable_tool_loop(
                session.websocket,
                session.conversation_id,
                build_history,
                f"{session.system_prompt}\n\n{DEEP_BUILD_SYSTEM_PROMPT}",
                deep_execute_token_budget(session.max_tokens, session.task_request or session.message),
                features=session.features,
                allowed_tools=build_tools,
                status_prefix=f"Build {idx + 1}.{sub_idx + 1}: ",
                max_steps=deep_execute_step_limit_for_request(session.task_request or session.message),
                activity_phase="execute",
                activity_step_label=substep_label,
                workflow_execution=session.workflow_execution,
                continuation_limit=tool_loop_continuation_limit_for_request(
                    session.task_request or session.message,
                    build_tools,
                    activity_phase="execute",
                ),
            )

            substep_successful_tools = sum(
                1 for entry in outcome.tool_results if entry.get("result", {}).get("ok")
            )
            substep_successful_commands = sum(
                1
                for entry in outcome.tool_results
                if entry.get("call", {}).get("name") == "workspace.run_command"
                and entry.get("result", {}).get("ok")
            )
            step_tool_calls += len(outcome.tool_results)
            step_successful_tools += substep_successful_tools
            step_successful_commands += substep_successful_commands

            substep_touched_paths: List[str] = []
            for entry in outcome.tool_results:
                call = entry.get("call", {})
                result = entry.get("result", {})
                payload = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
                path = payload.get("path")
                if isinstance(path, str) and path and path not in substep_touched_paths:
                    substep_touched_paths.append(path)
                if isinstance(path, str) and path and path not in touched_paths:
                    touched_paths.append(path)
                if call.get("name") != "workspace.patch_file" or not result.get("ok"):
                    continue
                if isinstance(path, str) and path not in changed_files:
                    changed_files.append(path)

            substep_summary = outcome.final_text.strip() or f"Completed substep {idx + 1}.{sub_idx + 1}."
            if substep_successful_tools == 0:
                substep_summary = f"{substep_summary} No successful tool actions were recorded for this substep."

            substep_report = {
                "substep_index": sub_idx,
                "substep": current_substep,
                "tool_calls": len(outcome.tool_results),
                "successful_tools": substep_successful_tools,
                "successful_commands": substep_successful_commands,
                "paths": list(substep_touched_paths),
                "summary": substep_summary,
            }

            if outcome.blocked_on_permission:
                step_subplan["progress_note"] = ""
                session.step_subplans[step_key] = normalize_step_subplan(step_subplan, step)
                session.workspace_snapshot = capture_workspace_snapshot(session.conversation_id)
                session.changed_files = changed_files
                session.build_summary = "\n".join(session.build_step_summaries) or "(build paused awaiting approval)"
                session.pause_reason = PAUSE_REASON_COMMAND_APPROVAL
                await send_build_steps(
                    session.websocket,
                    steps,
                    completed_count=idx,
                    active_index=idx,
                    step_details=build_step_details_payload(
                        session,
                        steps,
                        completed_count=idx,
                        active_index=idx,
                        active_substep_index=sub_idx,
                    ),
                )
                await persist_task_board(
                    session,
                    active_build_step=idx,
                    active_substep_index=sub_idx,
                )
                await persist_task_state(session)
                return DeepBuildResult(
                    summary=outcome.final_text,
                    needs_user_confirmation=True,
                    build_complete=False,
                    pause_reason=session.pause_reason,
                )

            if outcome.hit_limit:
                step_subplan["progress_note"] = substep_summary
                session.step_subplans[step_key] = normalize_step_subplan(step_subplan, step)
                session.workspace_snapshot = capture_workspace_snapshot(session.conversation_id)
                session.changed_files = changed_files
                session.build_summary = "\n".join(session.build_step_summaries) or "(build in progress)"
                session.pause_reason = PAUSE_REASON_HARD_LIMIT
                await send_build_steps(
                    session.websocket,
                    steps,
                    completed_count=idx,
                    active_index=idx,
                    step_details=build_step_details_payload(
                        session,
                        steps,
                        completed_count=idx,
                        active_index=idx,
                        active_substep_index=sub_idx,
                    ),
                )
                await persist_task_board(
                    session,
                    active_build_step=idx,
                    active_substep_index=sub_idx,
                )
                await persist_task_state(session)
                return DeepBuildResult(
                    summary=(
                        f"Paused during step {idx + 1}.{sub_idx + 1}: {current_substep}\n\n"
                        f"{substep_summary}"
                    ),
                    needs_user_confirmation=True,
                    build_complete=False,
                    pause_reason=session.pause_reason,
                )

            completed_substeps.append(substep_summary)
            store_step_substep_summaries(session, idx, completed_substeps)
            completed_substep_reports.append(substep_report)
            store_step_substep_reports(session, idx, completed_substep_reports)
            step_subplan["progress_note"] = ""
            session.step_subplans[step_key] = normalize_step_subplan(step_subplan, step)
            session.workspace_snapshot = capture_workspace_snapshot(session.conversation_id)
            session.changed_files = changed_files
            session.build_summary = "\n".join(session.build_step_summaries) or "(build in progress)"
            next_nested_index = sub_idx + 1 if sub_idx + 1 < len(substeps) else None
            await send_build_steps(
                session.websocket,
                steps,
                completed_count=idx,
                active_index=idx,
                step_details=build_step_details_payload(
                    session,
                    steps,
                    completed_count=idx,
                    active_index=idx,
                    active_substep_index=next_nested_index,
                ),
            )
            await persist_task_board(
                session,
                active_build_step=idx,
                active_substep_index=next_nested_index,
            )
            await persist_task_state(session)

        report = {
            "step_index": idx,
            "step": step,
            "tool_calls": step_tool_calls,
            "successful_tools": step_successful_tools,
            "successful_commands": step_successful_commands,
            "paths": touched_paths,
            "subplan": session.step_subplans.get(step_key, {}),
            "substep_summaries": list(completed_substeps),
            "substep_reports": list(completed_substep_reports),
            "summary": summarize_completed_step_from_substeps(step, completed_substeps),
        }
        session.step_reports.append(report)
        step_summary = report["summary"].strip() or f"Completed build step {idx + 1}."
        if step_successful_tools == 0 and not completed_substeps:
            step_summary = f"{step_summary} No successful tool actions were recorded for this step."
        session.build_step_summaries.append(f"Step {idx + 1}: {step_summary}")

        session.workspace_snapshot = capture_workspace_snapshot(session.conversation_id)
        session.changed_files = changed_files
        session.build_summary = "\n".join(session.build_step_summaries) or "(no build work completed)"
        next_active_index = idx + 1 if idx + 1 < len(steps) else None
        await send_build_steps(
            session.websocket,
            steps,
            completed_count=idx + 1,
            active_index=next_active_index,
            step_details=build_step_details_payload(
                session,
                steps,
                completed_count=idx + 1,
                active_index=next_active_index,
            ),
        )
        await persist_task_board(
            session,
            active_build_step=next_active_index,
            active_substep_index=None,
        )
        await persist_task_state(session)

    await send_activity_event(
        session.websocket,
        "execute",
        "Execute",
        "Completed all planned build steps.",
    )
    session.pause_reason = ""
    return DeepBuildResult(summary=session.build_summary, build_complete=True)


async def deep_decompose(session: DeepSession, preview_only: bool = False) -> Dict[str, Any]:
    """Create a sequential execution plan using message context plus observed workspace facts."""
    if session.plan:
        await send_assistant_note(session.websocket, "Reusing the saved execution plan from the existing task board.")
        session.plan_preview_pending = bool(preview_only)
        session.pause_reason = PAUSE_REASON_USER_DECISION if preview_only else ""
        if not preview_only:
            await send_build_steps(
                session.websocket,
                session.plan.get("builder_steps", []),
                completed_count=len(session.build_step_summaries),
                active_index=len(session.build_step_summaries) if len(session.build_step_summaries) < len(session.plan.get("builder_steps", [])) else None,
                step_details=build_step_details_payload(
                    session,
                    session.plan.get("builder_steps", []),
                    completed_count=len(session.build_step_summaries),
                    active_index=(
                        len(session.build_step_summaries)
                        if len(session.build_step_summaries) < len(session.plan.get("builder_steps", []))
                        else None
                    ),
                ),
            )
            await persist_task_board(session, active_build_step=len(session.build_step_summaries))
        else:
            await persist_task_board(session, active_build_step=None)
        await persist_task_state(session)
        return session.plan
    await send_activity_event(
        session.websocket,
        "plan",
        "Plan",
        "Turning the inspected request into a grounded execution plan.",
    )
    request_focus = summarize_request_for_plan(session.task_request or session.message)
    if request_focus:
        await send_reasoning_note(
            session.websocket,
            f"Planning specifically around this request: {request_focus}",
            phase="plan",
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
                    f"User's new query:\n{session.task_request or session.message}"
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
    session.plan_preview_pending = bool(preview_only)
    session.pause_reason = PAUSE_REASON_USER_DECISION if preview_only else ""
    await send_assistant_note(
        session.websocket,
        "Plan draft ready in the approval panel."
        if preview_only else format_deep_plan_note(session.plan),
    )
    if not preview_only:
        await send_build_steps(
            session.websocket,
            session.plan.get("builder_steps", []),
            completed_count=0,
            active_index=0,
            step_details=build_step_details_payload(
                session,
                session.plan.get("builder_steps", []),
                completed_count=0,
                active_index=0,
            ),
        )
        await persist_task_board(session, active_build_step=0, announce=True)
    else:
        await persist_task_board(session, active_build_step=None)
    await persist_task_state(session)
    return session.plan


async def deep_answer_directly(session: DeepSession) -> str:
    """Use deep mode as a thoughtful answer path without forcing execution planning."""
    await send_activity_event(
        session.websocket,
        "respond",
        "Respond",
        "Answering from inspected context.",
    )
    allowed_tools = select_enabled_tools(
        session.conversation_id,
        session.message,
        session.features,
        history=session.history,
    )
    allowed_tools = [
        tool_name for tool_name in allowed_tools
        if tool_name not in {"workspace.patch_file", "workspace.run_command"}
    ]
    direct_history = session.history_messages() + [{
        "role": "user",
        "content": (
            f"Conversation context:\n{session.context or '(none)'}\n\n"
            f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
            f"User request:\n{session.task_request or session.message}\n\n"
            "Answer directly unless execution is clearly required. "
            "Use read-only tools if they materially improve the answer."
        ),
    }]
    if allowed_tools:
        outcome = await run_resumable_tool_loop(
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
            workflow_execution=session.workflow_execution,
            continuation_limit=tool_loop_continuation_limit_for_request(
                session.task_request or session.message,
                allowed_tools,
                activity_phase="respond",
            ),
        )
        if outcome.requested_phase_upgrade == "workspace_execution":
            await send_activity_event(
                session.websocket,
                "evaluate",
                "Escalate",
                "Switching from answer mode into workspace execution because the next step needs file changes.",
            )
            if not session.workspace_enabled:
                session.workspace_enabled = True
                await deep_inspect_workspace(session)
            session.auto_execute = True
            return await run_deep_execution_flow(session, requires_existing_plan=False)
        if outcome.blocked_on_permission:
            session.pause_reason = PAUSE_REASON_COMMAND_APPROVAL
            session.draft_response = outcome.final_text
            await persist_task_state(session)
            return session.draft_response
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
        session.pause_reason = ""
        session.verification_summary = ""
        return session.verification_summary
    await send_activity_event(
        session.websocket,
        "verify",
        "Verify",
        "Checking the result against the workspace.",
    )
    verify_history = session.history_messages() + [{
        "role": "user",
        "content": (
            f"User request:\n{session.task_request or session.message}\n\n"
            f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
            + (
                f"Recent product feedback to verify against:\n{session.recent_product_feedback_summary}\n\n"
                if session.recent_product_feedback_summary else ""
            )
            +
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
    verify_tools = allowed_workspace_tools(session.features, include_write=False)
    outcome = await run_resumable_tool_loop(
        session.websocket,
        session.conversation_id,
        verify_history,
        f"{session.system_prompt}\n\n{DEEP_VERIFY_SYSTEM_PROMPT}",
        min(session.max_tokens, 1536),
        features=session.features,
        allowed_tools=verify_tools,
        status_prefix="Verify: ",
        max_steps=4,
        activity_phase="verify",
        workflow_execution=session.workflow_execution,
        continuation_limit=tool_loop_continuation_limit_for_request(
            session.task_request or session.message,
            verify_tools,
            activity_phase="verify",
        ),
    )
    if outcome.blocked_on_permission:
        session.pause_reason = PAUSE_REASON_COMMAND_APPROVAL
        session.verification_summary = outcome.final_text
        await persist_task_board(session, active_build_step=None)
        await persist_task_state(session)
        return session.verification_summary
    session.pause_reason = PAUSE_REASON_HARD_LIMIT if outcome.hit_limit else ""
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
        "Preparing the final answer from verified artifacts.",
    )
    artifact_refs = [f"[[artifact:{session.task_board_path}]]"]
    if session.recent_product_feedback_summary:
        artifact_refs.append(f"[[artifact:{session.recent_product_feedback_artifact_path}]]")
    artifact_refs.extend(f"[[artifact:{path}]]" for path in session.changed_files[:8])
    synth_history = session.history_messages() + [{
        "role": "user",
        "content": (
            f"User's question:\n{session.task_request or session.message}\n\n"
            f"Observed workspace facts:\n{session.workspace_facts or '(none)'}\n\n"
            + (
                f"Recent product feedback:\n{session.recent_product_feedback_summary}\n\n"
                if session.recent_product_feedback_summary else ""
            )
            +
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
    synth_tools = allowed_workspace_tools(session.features, include_write=False)
    outcome = await run_resumable_tool_loop(
        session.websocket,
        session.conversation_id,
        synth_history,
        f"{session.system_prompt}\n\n{DEEP_SYNTHESIZE_SYSTEM_PROMPT}",
        min(session.max_tokens, 2048),
        features=session.features,
        allowed_tools=synth_tools,
        status_prefix="Synthesize: ",
        max_steps=6,
        activity_phase="synthesize",
        workflow_execution=session.workflow_execution,
        continuation_limit=tool_loop_continuation_limit_for_request(
            session.task_request or session.message,
            synth_tools,
            activity_phase="synthesize",
        ),
    )
    if outcome.blocked_on_permission:
        session.pause_reason = PAUSE_REASON_COMMAND_APPROVAL
        session.draft_response = outcome.final_text
        await persist_task_state(session)
        return session.draft_response
    session.pause_reason = PAUSE_REASON_HARD_LIMIT if outcome.hit_limit else ""
    session.draft_response = outcome.final_text
    await persist_task_state(session)
    return session.draft_response


def create_deep_session(
    *,
    websocket: WebSocket,
    conversation_id: str,
    message: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
    workspace_enabled: bool,
    task_request: Optional[str] = None,
    execution_requested: bool = False,
    auto_execute: bool = False,
    plan_override_builder_steps: Optional[List[str]] = None,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> DeepSession:
    """Create a deep-session object with the shared defaults used across entry points."""
    return DeepSession(
        websocket=websocket,
        conversation_id=conversation_id,
        message=message,
        task_request=task_request or message,
        history=history,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        features=features,
        context=build_recent_context(history),
        workspace_enabled=workspace_enabled,
        execution_requested=execution_requested,
        auto_execute=auto_execute,
        plan_override_builder_steps=normalize_plan_override_steps(plan_override_builder_steps),
        workflow_execution=workflow_execution,
    )


async def collect_recent_product_feedback_for_session(session: DeepSession) -> str:
    """Hydrate a deep session with recent corrective product feedback when the request calls for it."""
    request_text = session.task_request or session.message
    if not request_wants_recent_product_feedback(request_text):
        session.recent_product_feedback_entries = []
        session.recent_product_feedback_summary = ""
        return ""

    entries = collect_recent_product_feedback_entries(limit=6)
    if not entries:
        session.recent_product_feedback_entries = []
        session.recent_product_feedback_summary = ""
        return ""

    session.recent_product_feedback_entries = entries
    session.recent_product_feedback_summary = format_recent_product_feedback_summary(entries)
    if session.workspace_enabled:
        write_workspace_text(
            session.conversation_id,
            session.recent_product_feedback_artifact_path,
            format_recent_product_feedback_markdown(entries),
        )
    await send_activity_event(
        session.websocket,
        "inspect",
        "Feedback",
        f"Loaded {len(entries)} recent corrective feedback signal(s) from chat history for this repo-improvement pass.",
    )
    await persist_task_state(session)
    return session.recent_product_feedback_summary


async def apply_deep_session_plan_override(session: DeepSession) -> bool:
    """Apply any edited builder-step overrides onto a restored deep plan."""
    if not session.plan or not session.plan_override_builder_steps:
        return False
    previous_steps = [
        str(step).strip()
        for step in session.plan.get("builder_steps", [])
        if str(step).strip()
    ]
    session.plan = apply_plan_builder_step_override(session.plan, session.plan_override_builder_steps)
    if previous_steps != session.plan.get("builder_steps", []):
        await send_assistant_note(
            session.websocket,
            "Updated the saved build steps from the plan card before execution.",
        )
        return True
    return False


async def bootstrap_deep_session(
    session: DeepSession,
    *,
    resume_task_state_allowed: bool = True,
) -> bool:
    """Run the shared confirmation/resume/inspect bootstrap for deep sessions."""
    try:
        await deep_confirm_understanding(session)
    except Exception as exc:
        logger.warning("Deep mode confirmation failed, continuing: %s", exc)

    if resume_task_state_allowed:
        try:
            await maybe_resume_task_state(session)
        except Exception as exc:
            logger.warning("Deep mode resume failed, continuing fresh: %s", exc)

    try:
        await collect_recent_product_feedback_for_session(session)
    except Exception as exc:
        logger.warning("Recent product feedback collection failed, continuing: %s", exc)

    await apply_deep_session_plan_override(session)
    await deep_inspect_workspace(session)
    return not (
        session.pause_reason == PAUSE_REASON_COMMAND_APPROVAL
        and bool(session.draft_response)
    )


async def run_deep_plan_preview_flow(session: DeepSession) -> str:
    """Build and publish the current deep plan preview."""
    await deep_decompose(session, preview_only=True)
    await send_plan_ready(session.websocket, session.plan, format_deep_execution_prompt(session.plan))
    return render_deep_plan_preview(session.plan)


async def run_deep_execution_flow(
    session: DeepSession,
    *,
    requires_existing_plan: bool = False,
    missing_plan_message: str = "I couldn't find a saved plan to execute in this chat. Ask me to generate the plan again first.",
    blocked_write_renderer: Optional[Callable[[DeepSession], str]] = None,
) -> str:
    """Run the shared deep plan/build/verify/review pipeline."""
    decision = decide_deep_route(
        DeepRouteRequest(
            requires_existing_plan=requires_existing_plan,
            has_plan=bool(session.plan),
            should_preview_plan=False,
            execution_requested=session.execution_requested,
            auto_execute=session.auto_execute,
            workspace_write=session.features.workspace_write,
        )
    )
    if decision.action == "missing_plan":
        return missing_plan_message

    await deep_decompose(session)
    logger.info("Deep mode execution strategy: %s", session.plan.get("strategy", "parallel subtasks"))

    decision = decide_deep_route(
        DeepRouteRequest(
            requires_existing_plan=False,
            has_plan=bool(session.plan),
            should_preview_plan=False,
            execution_requested=session.execution_requested,
            auto_execute=session.auto_execute,
            workspace_write=session.features.workspace_write,
        )
    )
    if decision.action == "blocked_write":
        session.pause_reason = PAUSE_REASON_WRITE_BLOCKED
        await persist_task_state(session)
        await send_plan_ready(session.websocket, session.plan, format_deep_execution_prompt(session.plan))
        renderer = blocked_write_renderer or (lambda current_session: render_saved_plan_write_access_message(current_session.plan))
        return renderer(session)

    build_result = await deep_build_workspace(session)
    if build_result.needs_user_confirmation:
        session.pause_reason = normalize_pause_reason(build_result.pause_reason)
        return build_result.summary
    await deep_parallel_solve(session)
    await deep_verify(session)
    if session.pause_reason == PAUSE_REASON_HARD_LIMIT:
        return session.verification_summary
    final_review = await deep_review(session)
    if session.pause_reason == PAUSE_REASON_HARD_LIMIT:
        return session.draft_response
    return final_review


async def maybe_refine_deep_response(
    session: DeepSession,
    draft_response: str,
) -> str:
    """Run the optional critique/refinement pass for deep-mode responses."""
    if not DEEP_CRITIQUE_ENABLED or is_fast_profile_active():
        return draft_response

    await send_activity_event(
        session.websocket,
        "verify",
        "Review",
        "Reviewing the draft.",
    )
    critique = await critique_response(session.task_request or session.message, draft_response)
    if critique["pass"]:
        return draft_response

    issues = critique["issues"] or "Tighten correctness, completeness, and structure."
    logger.info("Deep mode critique requested refinement: %s", issues)
    await send_activity_event(
        session.websocket,
        "synthesize",
        "Refine",
        "Refining the draft.",
    )

    refine_messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User's question:\n{session.task_request or session.message}\n\n"
                f"Current draft:\n{draft_response}\n\n"
                f"Issues to fix:\n{issues}"
            ),
        },
    ]
    refined = await vllm_chat_complete(refine_messages, max_tokens=session.max_tokens, temperature=0.15)
    refined_response = strip_stream_special_tokens(refined)

    follow_up = await critique_response(session.task_request or session.message, refined_response)
    if follow_up["pass"]:
        return refined_response

    logger.info("Deep mode refined draft still failed critique, returning refined version: %s", follow_up["issues"])
    return refined_response


# ==================== WebSocket Handlers ====================

async def orchestrated_chat(
    websocket: WebSocket,
    conversation_id: str,
    message: str,
    history: list,
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
    auto_execute: bool = False,
    plan_override_builder_steps: Optional[List[str]] = None,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> str:
    """Deep-mode pipeline using explicit shared session state."""
    await send_activity_event(
        websocket,
        "analyze",
        "Analyze",
        "Choosing the execution path.",
    )

    session = create_deep_session(
        websocket=websocket,
        conversation_id=conversation_id,
        message=message,
        history=history,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        features=features,
        workspace_enabled=(
            auto_execute
            or should_resume_saved_workspace_task(conversation_id, message, features)
            or should_use_workspace_tools(conversation_id, message, features)
        ),
        execution_requested=is_explicit_plan_execution_request(message),
        auto_execute=auto_execute,
        plan_override_builder_steps=normalize_plan_override_steps(plan_override_builder_steps),
        workflow_execution=workflow_execution,
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
            "Workspace path selected."
            if session.workspace_enabled
            else "Text path selected."
        ),
    )

    bootstrap_ready = await bootstrap_deep_session(session, resume_task_state_allowed=True)
    if not bootstrap_ready and session.draft_response:
        return session.draft_response

    clarification = should_pause_for_workspace_clarification(
        session.task_request or session.message,
        session.workspace_facts,
        session.workspace_snapshot,
        has_plan=bool(session.plan),
        execution_requested=session.execution_requested,
    )
    if clarification:
        session.draft_response = clarification
        await persist_task_state(session)
        return clarification

    decision = decide_deep_route(
        DeepRouteRequest(
            requires_existing_plan=session.execution_requested,
            has_plan=bool(session.plan),
            should_preview_plan=should_preview_deep_plan(session),
            execution_requested=session.execution_requested,
            auto_execute=session.auto_execute,
            workspace_write=session.features.workspace_write,
        )
    )

    if decision.action == "preview_plan":
        draft_response = await run_deep_plan_preview_flow(session)
    elif decision.action in {"execute_plan", "blocked_write", "missing_plan"}:
        draft_response = await run_deep_execution_flow(
            session,
            requires_existing_plan=session.execution_requested,
        )
    else:
        draft_response = await deep_answer_directly(session)

    return await maybe_refine_deep_response(session, draft_response)


def build_seeded_tool_history(
    history: List[Dict[str, str]],
    user_message: str,
    call: Dict[str, Any],
    result: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Append a deterministic first tool call/result before handing control back to the tool loop."""
    seeded = list(history)
    seeded.append({"role": "user", "content": user_message})
    seeded.append({"role": "assistant", "content": json.dumps(call, ensure_ascii=False)})
    seeded.append({
        "role": "user",
        "content": "<tool_result>\n" + json.dumps(result, ensure_ascii=False) + "\n</tool_result>",
    })
    return seeded


async def handle_direct_search_command(
    websocket: WebSocket,
    conversation_id: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
    query: str,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> str:
    """Run a deterministic web search first, then let the model fetch pages only if needed."""
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return "Use `/search <query>` to run a direct web search."

    command_features = replace(features, web_search=True)
    search_call = {
        "id": "direct_search",
        "name": "web.search",
        "arguments": {"query": cleaned_query, "limit": 5},
    }
    search_result = await emit_direct_tool_call(
        websocket,
        conversation_id,
        search_call,
        features=command_features,
        status_prefix="Slash /search: ",
        activity_phase="respond",
        workflow_execution=workflow_execution,
    )
    if not search_result.get("ok"):
        if search_result.get("error_code") == "permission_denied":
            return str(search_result.get("message_to_user") or search_result.get("error") or "").strip()
        return f"Search failed: {search_result.get('error', 'unknown error')}"

    seeded_history = build_seeded_tool_history(
        history,
        (
            f"The user invoked /search with query:\n{cleaned_query}\n\n"
            "Start from the provided web search results. Fetch a result page only if it materially improves the answer, then respond clearly with citations."
        ),
        search_call,
        search_result,
    )
    outcome = await run_resumable_tool_loop(
        websocket,
        conversation_id,
        seeded_history,
        system_prompt,
        min(max_tokens, 2048),
        features=command_features,
        allowed_tools=["web.fetch_page"],
        status_prefix="Slash /search: ",
        max_steps=3,
        activity_phase="respond",
        workflow_execution=workflow_execution,
        continuation_limit=tool_loop_continuation_limit_for_request(
            cleaned_query,
            ["web.fetch_page"],
            activity_phase="respond",
        ),
    )
    return outcome.final_text


async def handle_direct_grep_command(
    websocket: WebSocket,
    conversation_id: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
    query: str,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> str:
    """Run grep first, then let the model inspect matching files if needed."""
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return "Use `/grep <text>` to search the workspace first."

    command_features = replace(features, agent_tools=True)
    grep_call = {
        "id": "direct_grep",
        "name": "workspace.grep",
        "arguments": {"query": cleaned_query, "path": ".", "glob": "*", "limit": 20},
    }
    grep_result = await emit_direct_tool_call(
        websocket,
        conversation_id,
        grep_call,
        features=command_features,
        status_prefix="Slash /grep: ",
        activity_phase="respond",
        workflow_execution=workflow_execution,
    )
    if not grep_result.get("ok"):
        if grep_result.get("error_code") == "permission_denied":
            return str(grep_result.get("message_to_user") or grep_result.get("error") or "").strip()
        return f"Grep failed: {grep_result.get('error', 'unknown error')}"

    seeded_history = build_seeded_tool_history(
        history,
        (
            f"The user invoked /grep with query:\n{cleaned_query}\n\n"
            "Start from the provided grep matches. Read files only when needed to explain the relevant hits, and keep the answer grounded in the observed matches."
        ),
        grep_call,
        grep_result,
    )
    grep_tools = ["workspace.read_file", "workspace.list_files", "spreadsheet.describe"]
    outcome = await run_resumable_tool_loop(
        websocket,
        conversation_id,
        seeded_history,
        system_prompt,
        min(max_tokens, 2048),
        features=command_features,
        allowed_tools=grep_tools,
        status_prefix="Slash /grep: ",
        max_steps=3,
        activity_phase="respond",
        workflow_execution=workflow_execution,
        continuation_limit=tool_loop_continuation_limit_for_request(
            cleaned_query,
            grep_tools,
            activity_phase="respond",
        ),
    )
    return outcome.final_text


async def handle_direct_plan_command(
    websocket: WebSocket,
    conversation_id: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
    request_text: str,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> str:
    """Inspect first and always return a concrete execution plan draft."""
    cleaned_request = str(request_text or "").strip()
    if not cleaned_request:
        return "Use `/plan <task>` to generate an execution plan."

    command_features = replace(features, agent_tools=True)
    execution_attempt = is_explicit_plan_execution_request(cleaned_request) or is_plan_approval_reply(cleaned_request)
    if execution_attempt:
        payload = load_task_state(conversation_id)
        if not payload or (
            not bool(payload.get("plan_preview_pending"))
            and not task_state_has_pending_follow_up(payload)
        ):
            return "I couldn't find a saved plan to execute in this chat. Generate the plan again with `/plan <task>` first."
        session = create_deep_session(
            websocket=websocket,
            conversation_id=conversation_id,
            message=cleaned_request,
            history=history,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            features=command_features,
            workspace_enabled=True,
            execution_requested=True,
            workflow_execution=workflow_execution,
        )
        await send_activity_event(
            websocket,
            "analyze",
            "Analyze",
            "Slash /plan is resuming the saved execution plan.",
        )
        bootstrap_ready = await bootstrap_deep_session(session, resume_task_state_allowed=True)
        if not bootstrap_ready and session.draft_response:
            return session.draft_response
        return await run_deep_execution_flow(
            session,
            requires_existing_plan=True,
            missing_plan_message="I couldn't find a saved plan to execute in this chat. Generate the plan again with `/plan <task>` first.",
        )

    session = create_deep_session(
        websocket=websocket,
        conversation_id=conversation_id,
        message=cleaned_request,
        history=history,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        features=command_features,
        workspace_enabled=True,
        execution_requested=True,
        workflow_execution=workflow_execution,
    )
    await send_activity_event(
        websocket,
        "analyze",
        "Analyze",
        "Slash /plan selected the planning flow.",
    )
    bootstrap_ready = await bootstrap_deep_session(session, resume_task_state_allowed=False)
    if not bootstrap_ready and session.draft_response:
        return session.draft_response
    return await run_deep_plan_preview_flow(session)


async def handle_direct_code_command(
    websocket: WebSocket,
    conversation_id: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
    request_text: str,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> str:
    """Run the inspect-plan-build-verify pipeline directly for code work."""
    cleaned_request = str(request_text or "").strip()
    if not cleaned_request:
        return "Use `/code <task>` to inspect the workspace and implement a change."

    command_features = replace(features, agent_tools=True)
    session = create_deep_session(
        websocket=websocket,
        conversation_id=conversation_id,
        message=cleaned_request,
        history=history,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        features=command_features,
        workspace_enabled=True,
        execution_requested=is_explicit_plan_execution_request(cleaned_request),
        workflow_execution=workflow_execution,
    )
    await send_activity_event(
        websocket,
        "analyze",
        "Analyze",
        "Slash /code selected the direct code workflow.",
    )
    bootstrap_ready = await bootstrap_deep_session(session, resume_task_state_allowed=False)
    if not bootstrap_ready and session.draft_response:
        return session.draft_response
    return await run_deep_execution_flow(
        session,
        requires_existing_plan=False,
        blocked_write_renderer=lambda current_session: (
            "I planned the code change, but write access was not granted for this turn.\n\n"
            f"{format_deep_plan_note(current_session.plan)}\n\n"
            "Approve workspace edits for this chat, then say continue to resume the saved plan, or edit the build steps in the plan card first."
        ),
    )


async def handle_direct_pip_command(
    websocket: WebSocket,
    conversation_id: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
    request_text: str,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> str:
    """Install Python packages into the managed chat environment."""
    cleaned_request = str(request_text or "").strip()
    if not cleaned_request:
        return "Use `/pip <package ...>` to install Python packages into the managed environment for this chat."

    try:
        pip_args = shlex.split(cleaned_request)
    except ValueError as exc:
        return f"Pip arguments could not be parsed: {exc}"

    if pip_args and pip_args[0].lower() == "install":
        pip_args = pip_args[1:]
    if not pip_args:
        return "Use `/pip <package ...>` to install one or more Python packages into the managed environment for this chat."

    command_features = replace(features, workspace_run_commands=True)
    install_call = {
        "id": "direct_pip_install",
        "name": "workspace.run_command",
        "arguments": {"command": ["pip", "install", *pip_args], "cwd": "."},
    }
    install_result = await emit_direct_tool_call(
        websocket,
        conversation_id,
        install_call,
        features=command_features,
        status_prefix="Slash /pip: ",
        activity_phase="respond",
        workflow_execution=workflow_execution,
    )
    if not install_result.get("ok"):
        if install_result.get("error_code") == "permission_denied":
            return str(install_result.get("message_to_user") or install_result.get("error") or "").strip()
        return f"Pip install failed: {install_result.get('error', 'unknown error')}"

    installed_preview = compact_tool_text(" ".join(pip_args), limit=96)
    return (
        f"Installed {installed_preview or 'the requested Python packages'} into the managed Python environment for this chat. "
        "I can use that environment in follow-up commands without cluttering the workspace."
    )


async def handle_direct_slash_command(
    websocket: WebSocket,
    conversation_id: str,
    slash_command: Dict[str, str],
    history: List[Dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    features: FeatureFlags,
    workflow_execution: Optional[WorkflowExecutionContext] = None,
) -> str:
    """Dispatch supported direct slash commands to their deterministic workflow."""
    name = slash_command.get("name", "")
    args = slash_command.get("args", "")
    if name == "search":
        return await handle_direct_search_command(
            websocket, conversation_id, history, system_prompt, max_tokens, features, args, workflow_execution
        )
    if name == "grep":
        return await handle_direct_grep_command(
            websocket, conversation_id, history, system_prompt, max_tokens, features, args, workflow_execution
        )
    if name == "plan":
        return await handle_direct_plan_command(
            websocket, conversation_id, history, system_prompt, max_tokens, features, args, workflow_execution
        )
    if name == "code":
        return await handle_direct_code_command(
            websocket, conversation_id, history, system_prompt, max_tokens, features, args, workflow_execution
        )
    if name == "pip":
        return await handle_direct_pip_command(
            websocket, conversation_id, history, system_prompt, max_tokens, features, args, workflow_execution
        )
    return f"Unsupported slash command: /{slash_command.get('raw_name') or name}"


def format_turn_route_activity(
    assessment: TurnAssessment,
    *,
    mode: str,
    promoted_to_planning: bool,
) -> str:
    """Render the high-level skill-loop routing summary for the activity log."""
    lines = [format_turn_assessment_summary(assessment), f"Mode {mode}. Intent {assessment.workspace_intent}."]
    if promoted_to_planning:
        lines.append("Explicit planning language promoted this turn into the planning loop.")
    return " ".join(line for line in lines if line)


async def prepare_turn_request(data: Dict[str, Any]) -> PreparedTurnRequest:
    """Normalize one inbound chat turn into a simpler routing payload."""
    message = data.get('message', '').strip()
    conv_id = data.get('conversation_id')
    requested_workspace_id = str(data.get("workspace_id") or "").strip()
    attachments_raw = data.get('attachments', [])
    attachments = [
        str(item).strip()
        for item in attachments_raw[:MAX_ATTACHMENTS_PER_MESSAGE]
        if isinstance(item, str) and str(item).strip()
    ]
    custom_system_prompt = data.get('system_prompt')
    requested_mode = data.get('mode', 'deep')
    features = parse_feature_flags(data.get('features'))
    slash_command = (
        parse_direct_slash_command_payload(data.get("slash_command"))
        or infer_direct_slash_command_from_message(message)
    )

    conversation_title = message[:50] + "..." if len(message) > 50 else message
    ensure_conversation_record(conv_id, title=conversation_title, workspace_id=requested_workspace_id or None)

    attachment_context = await build_attachment_context(conv_id, attachments, message)
    saved_user_message = f"{message}\n\n{attachment_context}" if attachment_context else message
    if slash_command:
        slash_request = (
            f"{slash_command.get('args', '')}\n\n{attachment_context}".strip()
            if attachment_context else
            str(slash_command.get("args", "")).strip()
        )
        history = get_conversation_history(conv_id, current_query=slash_request or saved_user_message)
        user_message_id = save_message(conv_id, 'user', saved_user_message, workspace_id=requested_workspace_id or None)
        effective_message = slash_request or saved_user_message
    else:
        effective_message = saved_user_message
        user_message_id = save_message(conv_id, 'user', effective_message, workspace_id=requested_workspace_id or None)
        history = get_conversation_history(conv_id, current_query=effective_message)

    repo_bootstrap = maybe_bootstrap_workspace_from_current_repo(conv_id, effective_message)
    repo_bootstrapped = bool(repo_bootstrap)
    repo_bootstrap_summary = ""
    if repo_bootstrapped:
        snapshot = repo_bootstrap.get("snapshot", {}) if isinstance(repo_bootstrap.get("snapshot"), dict) else {}
        repo_bootstrap_summary = (
            "Loaded a snapshot of the current repo into this conversation workspace "
            f"({int(snapshot.get('user_file_count', 0) or 0)} user-visible files, "
            f"{int(snapshot.get('total_dirs', 0) or 0)} directories)."
        )

    system_prompt = build_effective_system_prompt(
        custom_system_prompt or DEFAULT_SYSTEM_PROMPT,
        effective_message,
    )
    mode = str(requested_mode or "deep").strip().lower() or "deep"
    agent_params = get_agent_llm_params()
    max_tokens = agent_params["max_tokens"]
    workspace_intent = classify_workspace_intent(effective_message)
    enabled_tools = (
        select_enabled_tools(conv_id, effective_message, features, history=history)
        if TOOL_LOOP_ENABLED else []
    )

    resume_saved_workspace = (
        not slash_command
        and should_resume_saved_workspace_task(conv_id, effective_message, features)
    )
    auto_execute_workspace = (
        not slash_command
        and (
            should_auto_execute_workspace_task(conv_id, effective_message, features)
            or resume_saved_workspace
        )
    )
    workspace_requested = (
        should_use_workspace_tools(conv_id, effective_message, features)
        or auto_execute_workspace
        or resume_saved_workspace
    )
    local_rag_requested = should_offer_local_rag(effective_message, features)
    web_search_requested = should_offer_web_search(effective_message, features)
    assessment = build_turn_assessment(
        message=effective_message,
        requested_mode=requested_mode,
        resolved_mode=mode,
        workspace_intent=workspace_intent,
        enabled_tools=enabled_tools,
        workspace_requested=workspace_requested,
        has_attachment_context=bool(attachment_context),
        slash_command_name=(slash_command or {}).get("name", ""),
        local_rag_requested=local_rag_requested,
        web_search_requested=web_search_requested,
        auto_execute_workspace=auto_execute_workspace,
        resume_saved_workspace=resume_saved_workspace,
        execution_requested=is_explicit_plan_execution_request(effective_message),
        workspace_run_commands_enabled=features.workspace_run_commands,
    )
    promoted_to_planning = False
    if (
        not slash_command
        and mode != "deep"
        and assessment.explicit_planning_request
        and workspace_requested
    ):
        mode = "deep"
        promoted_to_planning = True
        assessment = build_turn_assessment(
            message=effective_message,
            requested_mode=requested_mode,
            resolved_mode=mode,
            workspace_intent=workspace_intent,
            enabled_tools=enabled_tools,
            workspace_requested=workspace_requested,
            has_attachment_context=bool(attachment_context),
            slash_command_name="",
            local_rag_requested=local_rag_requested,
            web_search_requested=web_search_requested,
            auto_execute_workspace=auto_execute_workspace,
            resume_saved_workspace=resume_saved_workspace,
            execution_requested=is_explicit_plan_execution_request(effective_message),
            workspace_run_commands_enabled=features.workspace_run_commands,
        )

    return PreparedTurnRequest(
        conversation_id=conv_id,
        user_message_id=user_message_id,
        saved_user_message=saved_user_message,
        effective_message=effective_message,
        history=history,
        system_prompt=system_prompt,
        requested_mode=requested_mode,
        resolved_mode=mode,
        features=features,
        slash_command=slash_command,
        max_tokens=max_tokens,
        workspace_intent=workspace_intent,
        enabled_tools=enabled_tools,
        auto_execute_workspace=auto_execute_workspace,
        resume_saved_workspace=resume_saved_workspace,
        plan_override_builder_steps=normalize_plan_override_steps(data.get("plan_override_steps")),
        promoted_to_planning=promoted_to_planning,
        repo_bootstrapped=repo_bootstrapped,
        repo_bootstrap_summary=repo_bootstrap_summary,
        assessment=assessment,
    )


async def process_chat_turn(websocket: WebSocket, data: Dict[str, Any]) -> None:
    """Process a single chat turn so the websocket loop can also handle stop requests."""
    message = data.get('message', '').strip()
    conv_id = data.get('conversation_id')
    client_turn_id = str(data.get("client_turn_id", "") or "").strip()

    if not message:
        await websocket.send_json({'type': 'error', 'content': 'Empty message'})
        return

    workflow_execution: Optional[WorkflowExecutionContext] = None
    response_started = False

    try:
        if client_turn_id:
            setattr(websocket, "active_client_turn_id", client_turn_id)
        prepared = await prepare_turn_request(data)
        mode = prepared.resolved_mode
        slash_command = prepared.slash_command
        features = prepared.features
        history = prepared.history
        system_prompt = prepared.system_prompt
        max_tokens = prepared.max_tokens
        enabled_tools = list(prepared.enabled_tools)
        effective_message = prepared.effective_message
        deep_succeeded = False
        workflow_execution = create_workflow_execution(
            conv_id,
            prepared.user_message_id,
            "chat_turn",
            {"assessment": prepared.assessment.as_metadata()},
        )

        if prepared.repo_bootstrapped and prepared.repo_bootstrap_summary:
            await send_activity_event(
                websocket,
                "inspect",
                "Repo Snapshot",
                prepared.repo_bootstrap_summary,
            )

        await send_activity_event(
            websocket,
            "evaluate",
            "Skill Loop",
            (
                format_turn_route_activity(
                    prepared.assessment,
                    mode=mode,
                    promoted_to_planning=prepared.promoted_to_planning,
                )
            ),
        )

        if slash_command:
            full_response = await handle_direct_slash_command(
                websocket,
                conv_id,
                slash_command,
                history,
                system_prompt,
                max_tokens,
                features,
                workflow_execution=workflow_execution,
            )
            await websocket.send_json({'type': 'start'})
            response_started = True
            await send_final_replacement(websocket, full_response)
            assistant_message_id = save_message(conv_id, 'assistant', full_response)
            finalize_workflow_execution(
                workflow_execution,
                assistant_message_id=assistant_message_id,
                final_outcome="completed_slash",
            )
            schedule_conversation_summary_refresh(conv_id)
            await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
            await websocket.send_json({'type': 'done'})
            return

        if mode == 'deep' or prepared.auto_execute_workspace:
            try:
                await websocket.send_json({'type': 'start'})
                response_started = True
                full_response = await orchestrated_chat(
                    websocket,
                    conv_id,
                    effective_message,
                    history,
                    system_prompt,
                    max_tokens,
                    features,
                    auto_execute=prepared.auto_execute_workspace,
                    plan_override_builder_steps=prepared.plan_override_builder_steps,
                    workflow_execution=workflow_execution,
                )
                full_response = ensure_nonempty_turn_response(full_response, conv_id, effective_message)
                await send_final_replacement(websocket, full_response)
                assistant_message_id = save_message(conv_id, 'assistant', full_response)
                finalize_workflow_execution(
                    workflow_execution,
                    assistant_message_id=assistant_message_id,
                    final_outcome="completed_deep",
                )
                schedule_conversation_summary_refresh(conv_id)
                await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
                await websocket.send_json({'type': 'done'})
                deep_succeeded = True
            except Exception as e:
                logger.warning("Deep/auto workspace mode failed, falling back: %s", e)
                if response_started:
                    await send_assistant_note(
                        websocket,
                        build_saved_progress_fallback_response(
                            conv_id,
                            effective_message,
                            error_text=str(e),
                        ),
                    )
                await websocket.send_json({
                    'type': 'status',
                    'content': 'Workspace execution path failed, using normal mode...',
                })

        if not deep_succeeded:
            await send_activity_event(
                websocket,
                "respond",
                "Respond",
                (
                    "Using tools to answer."
                    if enabled_tools
                    else "Answering directly."
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
                prepared.workspace_intent,
            )

            if enabled_tools:
                tool_outcome = await run_resumable_tool_loop(
                    websocket,
                    conv_id,
                    history,
                    system_prompt,
                    tool_max_tokens,
                    features=features,
                    allowed_tools=enabled_tools,
                    max_steps=tool_step_limit,
                    activity_phase="respond",
                    workflow_execution=workflow_execution,
                    continuation_limit=tool_loop_continuation_limit_for_request(
                        effective_message,
                        enabled_tools,
                        activity_phase="respond",
                    ),
                )
                full_response = tool_outcome.final_text
                full_response = ensure_nonempty_turn_response(full_response, conv_id, effective_message)
                if not response_started:
                    await websocket.send_json({'type': 'start'})
                    response_started = True
                await send_final_replacement(websocket, full_response)
            else:
                messages = [{'role': 'system', 'content': system_prompt}]
                for msg in history:
                    messages.append({'role': msg['role'], 'content': msg['content']})
                full_response = await stream_chat_response(
                    websocket,
                    messages,
                    max_tokens,
                    start_output=not response_started,
                )
                response_started = True
                recovery_tools = direct_response_tool_recovery_candidates(
                    conv_id,
                    effective_message,
                    history,
                    full_response,
                    features,
                )
                if recovery_tools:
                    await send_activity_event(
                        websocket,
                        "evaluate",
                        "Recover",
                        "The draft either refused a capability or handed the work back even though tools are available. Retrying with tools.",
                    )
                    recovery_history = list(history)
                    recovery_history.append({
                        "role": "user",
                        "content": build_capability_recovery_message(full_response, recovery_tools),
                    })
                    tool_outcome = await run_resumable_tool_loop(
                        websocket,
                        conv_id,
                        recovery_history,
                        system_prompt,
                        tool_loop_token_budget(max_tokens, effective_message, recovery_tools),
                        features=features,
                        allowed_tools=recovery_tools,
                        max_steps=max(2, tool_loop_step_limit_for_request(effective_message, recovery_tools)),
                        activity_phase="respond",
                        workflow_execution=workflow_execution,
                        continuation_limit=tool_loop_continuation_limit_for_request(
                            effective_message,
                            recovery_tools,
                            activity_phase="respond",
                        ),
                    )
                    full_response = tool_outcome.final_text
                leaked_call = extract_leaked_tool_call(full_response)
                if leaked_call:
                    logger.warning(
                        "Recovered leaked tool payload in direct response for conv %s: %s",
                        conv_id,
                        leaked_call.get("name", ""),
                    )
                    full_response = format_leaked_tool_call_message(leaked_call, features)
                full_response = strip_unverified_workspace_write_claims(full_response)
                full_response = ensure_nonempty_turn_response(full_response, conv_id, effective_message)
                await send_final_replacement(websocket, full_response)
            assistant_message_id = save_message(conv_id, 'assistant', full_response)
            finalize_workflow_execution(
                workflow_execution,
                assistant_message_id=assistant_message_id,
                final_outcome=(
                    "completed_with_tools"
                    if workflow_execution and workflow_execution.tool_count > 0
                    else "completed_direct"
                ),
            )
            schedule_conversation_summary_refresh(conv_id)
            await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
            await websocket.send_json({'type': 'done'})

    except asyncio.CancelledError:
        finalize_workflow_execution(workflow_execution, status="canceled", final_outcome="canceled")
        logger.info("Canceled active chat turn for conversation %s", conv_id)
        raise
    except httpx.HTTPStatusError as e:
        finalize_workflow_execution(
            workflow_execution,
            status="failed",
            final_outcome="http_error",
            error_text=f"vLLM HTTP {e.response.status_code}",
        )
        logger.error(f"vLLM API error: {e}")
        await websocket.send_json({'type': 'error', 'content': f'vLLM error: {e.response.status_code}. Is the model loaded?'})
    except httpx.ConnectError:
        finalize_workflow_execution(
            workflow_execution,
            status="failed",
            final_outcome="connect_error",
            error_text="Cannot connect to vLLM",
        )
        await websocket.send_json({'type': 'error', 'content': 'Cannot connect to vLLM. Is it running?'})
    except Exception as e:
        finalize_workflow_execution(
            workflow_execution,
            status="failed",
            final_outcome="exception",
            error_text=str(e),
        )
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        fallback_response = ensure_nonempty_turn_response(
            "",
            conv_id,
            message,
            error_text=str(e),
        )
        if fallback_response:
            if not response_started:
                await websocket.send_json({'type': 'start'})
            await send_final_replacement(websocket, fallback_response)
            assistant_message_id = save_message(conv_id, 'assistant', fallback_response)
            schedule_conversation_summary_refresh(conv_id)
            await websocket.send_json({'type': 'message_id', 'message_id': assistant_message_id})
            await websocket.send_json({'type': 'done'})
        else:
            await websocket.send_json({'type': 'error', 'content': f'Error: {str(e)}'})
    finally:
        if client_turn_id and str(getattr(websocket, "active_client_turn_id", "") or "").strip() == client_turn_id:
            setattr(websocket, "active_client_turn_id", "")

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    active_task: Optional[asyncio.Task] = None

    async def _keepalive():
        """Send periodic pings so idle connections are not dropped."""
        try:
            while True:
                await asyncio.sleep(15)
                await websocket.send_json({"type": "ping"})
        except Exception:
            pass

    keepalive_task = asyncio.create_task(_keepalive())

    try:
        while True:
            data = await websocket.receive_json()
            if data.get('type') == 'pong':
                continue
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

            if data.get('type') in {'command_approval', 'permission_response'}:
                conversation_id = str(data.get("conversation_id", "")).strip()
                approval_target = str(data.get("approval_target", "") or "").strip().lower()
                client_turn_id = str(data.get("client_turn_id", "") or "").strip()
                raw_permission_key = (
                    data.get("permission_key")
                    if data.get('type') == 'permission_response'
                    else data.get("command_key", "")
                )
                permission_key = (
                    normalize_allowed_command_key(str(raw_permission_key))
                    if approval_target == "command" or data.get('type') == 'command_approval'
                    else normalize_allowed_tool_permission_key(str(raw_permission_key))
                )
                approved = bool(data.get("approved"))
                waiter = PERMISSION_APPROVAL_WAITERS.get(conversation_id)
                if not waiter:
                    await websocket.send_json({'type': 'error', 'content': 'No permission approval is pending for this chat.'})
                    continue
                expected_target = str(waiter.get("approval_target", "") or "").strip().lower()
                expected_key = (
                    normalize_allowed_command_key(str(waiter.get("permission_key", "")))
                    if expected_target == "command"
                    else normalize_allowed_tool_permission_key(str(waiter.get("permission_key", "")))
                )
                expected_client_turn_id = str(waiter.get("client_turn_id", "") or "").strip()
                future = waiter.get("future")
                if approval_target and expected_target and approval_target != expected_target:
                    await websocket.send_json({'type': 'error', 'content': 'Permission approval target did not match the pending request.'})
                    continue
                if permission_key != expected_key:
                    await websocket.send_json({'type': 'error', 'content': 'Permission approval response did not match the pending request.'})
                    continue
                if expected_client_turn_id and client_turn_id != expected_client_turn_id:
                    await websocket.send_json({'type': 'error', 'content': 'Permission approval response did not match the active turn.'})
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
        websocket_id = id(websocket)
        for conversation_id, waiter in list(PERMISSION_APPROVAL_WAITERS.items()):
            if waiter.get("websocket_id") != websocket_id:
                continue
            future = waiter.get("future")
            if isinstance(future, asyncio.Future) and not future.done():
                future.set_result(False)
        if active_task and not active_task.done():
            active_task.cancel()
        keepalive_task.cancel()
        logger.info("WebSocket disconnected")
    except Exception as e:
        websocket_id = id(websocket)
        for conversation_id, waiter in list(PERMISSION_APPROVAL_WAITERS.items()):
            if waiter.get("websocket_id") != websocket_id:
                continue
            future = waiter.get("future")
            if isinstance(future, asyncio.Future) and not future.done():
                future.set_result(False)
        keepalive_task.cancel()
        logger.error(f"WebSocket error: {e}")

async def get_model_runtime_summary() -> Dict[str, Any]:
    """Collect model state for health checks and lightweight UI status."""
    loaded_model_name = await fetch_loaded_model_name()
    selected_profile = get_active_model_profile()
    loading = MODEL_LOADING_STATUS if isinstance(MODEL_LOADING_STATUS, dict) else {}
    tracked_model_name = str(loading.get("model_name") or selected_profile["name"])
    tracked_profile_key = str(loading.get("profile_key") or selected_profile["key"])
    active_profile_key = sync_active_profile_from_model_name(loaded_model_name)
    active_profile = (
        MODEL_PROFILES[active_profile_key]
        if active_profile_key in MODEL_PROFILES
        else build_model_profile("custom", loaded_model_name or selected_profile["name"], CUSTOM_MODEL_ARGS, label="Custom")
    )
    model_ok = await vllm_health_check()
    if model_ok:
        mark_model_load_completed(loaded_model_name or selected_profile["name"])
    elif loading.get("status") not in {"loading", "failed"} or loading.get("model_name") != tracked_model_name:
        mark_model_load_started(tracked_model_name, reason="startup", profile_key=tracked_profile_key)
    available_profiles = [
        serialize_model_profile(
            profile,
            active=profile["key"] == active_profile_key,
            selected=profile["key"] == selected_profile["key"],
        )
        for profile in MODEL_PROFILES.values()
    ]
    if selected_profile["key"] == "custom":
        available_profiles.append(
            serialize_model_profile(selected_profile, active=active_profile_key == "custom", selected=True)
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
    loading = runtime["loading"]
    if runtime["model_ok"]:
        message = f"Model {runtime['loaded_model_name']} is ready"
    elif loading.get("status") == "failed":
        detail = str(loading.get("detail") or "vLLM failed to start.")
        message = f"Model {loading.get('model_name') or runtime['selected_profile']['name']} failed to start: {detail}"
    else:
        message = f"Model {runtime['selected_profile']['name']} is loading or unavailable"
    return {
        "status": "healthy" if runtime["model_ok"] else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model": runtime["loaded_model_name"],
        "model_available": runtime["model_ok"],
        "model_profile": runtime["active_profile"]["key"],
        "loading": loading,
        "voice": voice,
        "message": message,
    }

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
        "phase": "restarting" if reason in {"switch", "activate", "redownload", "restart", "rollback"} else "loading",
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
        if status == "failed":
            eta_seconds = None
            progress = None

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
        "failed_at": loading.get("failed_at"),
        "detail": loading.get("detail"),
        "container": loading.get("container"),
    }

@app.on_event("startup")
async def startup_event():
    global ACTIVE_MODEL_LOCK
    ACTIVE_MODEL_LOCK = asyncio.Lock()
    logger.info("=" * 60)
    logger.info("AI Chat Application Starting...")
    logger.info(f"Selected model profile: {ACTIVE_MODEL_PROFILE} ({get_active_model_name()})")
    logger.info(f"vLLM: {VLLM_HOST}")
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


def build_uvicorn_run_kwargs() -> Dict[str, Any]:
    """Build bind/TLS settings for local or container startup."""
    host = str(os.getenv("APP_HOST", "0.0.0.0")).strip() or "0.0.0.0"
    raw_port = str(os.getenv("APP_PORT") or os.getenv("PORT") or "8000").strip()
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError(f"APP_PORT/PORT must be an integer, got {raw_port!r}") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"APP_PORT/PORT must be between 1 and 65535, got {port}")

    certfile = str(os.getenv("SSL_CERTFILE", "")).strip()
    keyfile = str(os.getenv("SSL_KEYFILE", "")).strip()
    if bool(certfile) != bool(keyfile):
        raise RuntimeError("SSL_CERTFILE and SSL_KEYFILE must be set together")

    kwargs: Dict[str, Any] = {"host": host, "port": port}
    if certfile and keyfile:
        cert_path = pathlib.Path(certfile)
        key_path = pathlib.Path(keyfile)
        if not cert_path.is_file():
            raise RuntimeError(f"SSL_CERTFILE does not exist: {cert_path}")
        if not key_path.is_file():
            raise RuntimeError(f"SSL_KEYFILE does not exist: {key_path}")
        kwargs["ssl_certfile"] = str(cert_path)
        kwargs["ssl_keyfile"] = str(key_path)
    return kwargs

if __name__ == "__main__":
    import uvicorn
    run_kwargs = build_uvicorn_run_kwargs()
    scheme = "https" if run_kwargs.get("ssl_certfile") and run_kwargs.get("ssl_keyfile") else "http"
    logger.info(
        "Starting AI Chat with vLLM (%s) on %s://%s:%s...",
        get_active_model_name(),
        scheme,
        run_kwargs["host"],
        run_kwargs["port"],
    )
    uvicorn.run(app, **run_kwargs)
