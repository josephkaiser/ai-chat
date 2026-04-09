from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

try:
    import numpy as np
except Exception:
    np = None

try:
    from fastembed import TextEmbedding
except Exception:
    TextEmbedding = None

logger = logging.getLogger(__name__)

EMBEDDING_ENABLED = str(os.getenv("EMBEDDING_ENABLED", "true")).strip().lower() not in {
    "0", "false", "no", "off",
}
EMBEDDING_MODEL_NAME = str(os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")).strip()
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
EMBEDDING_MAX_LENGTH = max(64, int(os.getenv("EMBEDDING_MAX_LENGTH", "512")))
EMBEDDING_CACHE_DIR = (
    str(os.getenv("EMBEDDING_CACHE_DIR", "")).strip()
    or str(os.getenv("HF_CACHE_PATH", "")).strip()
    or None
)

_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_ERROR: Optional[str] = None


def configured_embedding_model_name() -> str:
    """Return the configured semantic embedding model label."""
    return EMBEDDING_MODEL_NAME or "fastembed-default"


def embeddings_available() -> bool:
    """Return whether the optional embedding runtime can be used."""
    return bool(EMBEDDING_ENABLED and TextEmbedding is not None and np is not None)


def embedding_runtime_error() -> str:
    """Return the last model-load failure, if any."""
    return _MODEL_ERROR or ""


def _build_model_kwargs() -> List[dict]:
    base_kwargs = {}
    if EMBEDDING_MODEL_NAME:
        base_kwargs["model_name"] = EMBEDDING_MODEL_NAME

    candidates = []
    if EMBEDDING_CACHE_DIR:
        candidates.append({**base_kwargs, "cache_dir": EMBEDDING_CACHE_DIR, "max_length": EMBEDDING_MAX_LENGTH})
        candidates.append({**base_kwargs, "cache_dir": EMBEDDING_CACHE_DIR})
    candidates.append({**base_kwargs, "max_length": EMBEDDING_MAX_LENGTH})
    candidates.append(dict(base_kwargs))
    return candidates


def _get_model():
    global _MODEL, _MODEL_ERROR
    if not embeddings_available():
        return None
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        last_error = None
        for kwargs in _build_model_kwargs():
            try:
                _MODEL = TextEmbedding(**kwargs)
                _MODEL_ERROR = None
                logger.info("Semantic retrieval enabled with fastembed model %s", configured_embedding_model_name())
                return _MODEL
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                _MODEL_ERROR = str(exc)
                logger.warning("Semantic retrieval model load failed: %s", exc)
                return None
        if last_error is not None:
            _MODEL_ERROR = str(last_error)
            logger.warning("Semantic retrieval model load failed: %s", last_error)
        return None


def _prepare_text(text: str, *, prefix: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        cleaned = "(empty)"
    return f"{prefix}: {cleaned}"


def _embed(texts: List[str], *, prefix: str) -> List["np.ndarray"]:
    if not texts or np is None:
        return []
    model = _get_model()
    if model is None:
        return []

    prepared = [_prepare_text(text, prefix=prefix) for text in texts]
    try:
        vectors = list(model.embed(prepared, batch_size=EMBEDDING_BATCH_SIZE))
    except TypeError:
        vectors = list(model.embed(prepared))
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


def embed_passages(texts: List[str]) -> List["np.ndarray"]:
    """Embed passage-style texts for storage and retrieval."""
    return _embed(texts, prefix="passage")


def embed_queries(texts: List[str]) -> List["np.ndarray"]:
    """Embed query-style texts for semantic search."""
    return _embed(texts, prefix="query")
