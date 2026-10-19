"""Thin async client for the vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from . import config

_client: Optional[httpx.AsyncClient] = None
_model_info: Dict[str, Any] = {}


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=900, write=60, pool=30))
    return _client


async def model_info(refresh: bool = False) -> Dict[str, Any]:
    """Return {id, max_model_len, ready}. Cached once the server answers."""
    if _model_info.get("ready") and not refresh:
        return _model_info
    try:
        resp = await client().get(f"{config.VLLM_HOST}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        entry = data[0] if data else {}
        _model_info.update(
            id=entry.get("id") or config.MODEL_NAME,
            max_model_len=int(entry.get("max_model_len") or config.FALLBACK_MAX_MODEL_LEN),
            ready=True,
        )
    except Exception as exc:  # server still loading
        _model_info.update(
            id=config.MODEL_NAME,
            max_model_len=config.FALLBACK_MAX_MODEL_LEN,
            ready=False,
            error=str(exc),
        )
    return _model_info


def estimate_tokens(text: str) -> int:
    # Gemma's tokenizer averages ~3.6 chars/token on mixed prose+code.
    return max(1, int(len(text) / 3.6)) if text else 0


async def count_tokens(text: str) -> int:
    if not text:
        return 0
    info = await model_info()
    if not info.get("ready"):
        return estimate_tokens(text)
    try:
        resp = await client().post(
            f"{config.VLLM_ROOT}/tokenize",
            json={"model": info["id"], "prompt": text, "add_special_tokens": False},
            timeout=120,
        )
        resp.raise_for_status()
        return int(resp.json()["count"])
    except Exception:
        return estimate_tokens(text)


async def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Exact prompt size for a chat request (falls back to an estimate)."""
    info = await model_info()
    if info.get("ready"):
        try:
            resp = await client().post(
                f"{config.VLLM_ROOT}/tokenize",
                json={"model": info["id"], "messages": messages, "add_generation_prompt": True},
                timeout=120,
            )
            resp.raise_for_status()
            return int(resp.json()["count"])
        except Exception:
            pass
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content) + 4
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                else:
                    total += 300
    return total


async def stream_chat(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    thinking: bool,
    temperature: Optional[float] = None,
) -> AsyncIterator[Dict[str, str]]:
    """Yield {"type": "reasoning"|"content", "text": ...} deltas, then {"type": "done", "finish_reason"}."""
    info = await model_info()
    body = {
        "model": info.get("id") or config.MODEL_NAME,
        "messages": messages,
        "stream": True,
        "max_tokens": max(16, max_tokens),
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "top_p": config.TOP_P,
        "top_k": config.TOP_K,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        "skip_special_tokens": True,
    }
    finish_reason = None
    async with client().stream("POST", f"{config.VLLM_HOST}/chat/completions", json=body) as resp:
        if resp.status_code >= 400:
            detail = (await resp.aread()).decode("utf-8", "replace")
            raise RuntimeError(f"vLLM error {resp.status_code}: {detail[:800]}")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    yield {"type": "reasoning", "text": reasoning}
                if delta.get("content"):
                    yield {"type": "content", "text": delta["content"]}
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    yield {"type": "done", "finish_reason": finish_reason or "stop"}


async def complete(messages: List[Dict[str, Any]], *, max_tokens: int, temperature: float = 0.3) -> str:
    """Non-thinking, low-temperature completion used for internal passes (notes, compaction)."""
    parts: List[str] = []
    async for event in stream_chat(messages, max_tokens=max_tokens, thinking=False, temperature=temperature):
        if event["type"] == "content":
            parts.append(event["text"])
    return "".join(parts).strip()
