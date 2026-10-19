"""Runtime settings for the helper app (all overridable via env)."""

from __future__ import annotations

import os
import pathlib


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


VLLM_HOST = os.getenv("VLLM_HOST", "http://vllm:8000/v1").rstrip("/")
# /tokenize and /v1/models live at the server root, not under /v1.
VLLM_ROOT = VLLM_HOST[:-3] if VLLM_HOST.endswith("/v1") else VLLM_HOST
MODEL_NAME = os.getenv("MODEL_NAME", "").strip()

DATA_ROOT = pathlib.Path(os.getenv("HELPER_DATA_ROOT", "/app/data/helper"))
WEB_ROOT = pathlib.Path(__file__).resolve().parents[2] / "helper_web"

# Context policy. MAX_MODEL_LEN is discovered from vLLM at runtime; this is only a fallback.
FALLBACK_MAX_MODEL_LEN = _int("HELPER_FALLBACK_MAX_MODEL_LEN", 32768)
# Tokens reserved for the reply (thinking + answer). Clamped to a share of the window.
REPLY_RESERVE = _int("HELPER_REPLY_RESERVE", 8192)
# Upper bound on a single reading pass when iterating through oversized material.
CHUNK_TOKENS_MAX = _int("HELPER_CHUNK_TOKENS_MAX", 64000)
# Share of the input budget the recent conversation may occupy before older turns are compacted.
HISTORY_SHARE = float(os.getenv("HELPER_HISTORY_SHARE", "0.35"))
# memory.md gets its own budget; beyond this it is condensed.
MEMORY_TOKENS_MAX = _int("HELPER_MEMORY_TOKENS_MAX", 6000)

UPLOAD_MAX_BYTES = _int("HELPER_UPLOAD_MAX_BYTES", 200 * 1024 * 1024)
EXTRACT_MAX_CHARS = _int("HELPER_EXTRACT_MAX_CHARS", 8_000_000)
VISION_ENABLED = _flag("HELPER_VISION", True)
MAX_IMAGES_IN_CONTEXT = _int("HELPER_MAX_IMAGES", 4)

TEMPERATURE = float(os.getenv("HELPER_TEMPERATURE", "1.0"))
TOP_P = float(os.getenv("HELPER_TOP_P", "0.95"))
TOP_K = _int("HELPER_TOP_K", 64)
