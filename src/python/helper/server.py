"""FastAPI app for the helper UI."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, engine, ingest, llm, store

app = FastAPI(title="AI Chat Helper")
app.mount("/static", StaticFiles(directory=str(config.WEB_ROOT)), name="static")

# One generation at a time: a single GPU is fastest when a turn has the whole KV cache to itself.
_gpu_lock = asyncio.Lock()
_active: Dict[str, engine.Turn] = {}
_SAFE_NAME = re.compile(r"[^\w.\- ()+@,]")


def _chat_or_404(chat_id: str) -> Dict[str, Any]:
    try:
        return store.load_chat(chat_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, "chat not found")


def _clean_name(name: str) -> str:
    name = Path(name or "file").name
    return _SAFE_NAME.sub("_", name)[:160] or "file"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(config.WEB_ROOT / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
async def status() -> Dict[str, Any]:
    info = await llm.model_info(refresh=True)
    return {
        "model": info.get("id"),
        "ready": bool(info.get("ready")),
        "max_model_len": info.get("max_model_len"),
        "vision": config.VISION_ENABLED,
        "max_images": config.MAX_IMAGES_IN_CONTEXT,
        "busy": _gpu_lock.locked(),
        "upload_max_bytes": config.UPLOAD_MAX_BYTES,
    }


# --- chats -------------------------------------------------------------------

@app.get("/api/chats")
async def chats() -> List[Dict[str, Any]]:
    return store.list_chats()


@app.post("/api/chats")
async def create_chat() -> Dict[str, Any]:
    return store.create_chat()


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str) -> Dict[str, Any]:
    chat = _chat_or_404(chat_id)
    return {
        **chat,
        "files": store.list_files(chat_id),
        "memory": store.read_memory(chat_id),
        "generating": chat_id in _active,
    }


class ChatPatch(BaseModel):
    title: Optional[str] = None


@app.patch("/api/chats/{chat_id}")
async def patch_chat(chat_id: str, body: ChatPatch) -> Dict[str, Any]:
    chat = _chat_or_404(chat_id)
    if body.title is not None:
        chat["title"] = body.title.strip()[:120] or chat["title"]
    store.save_chat(chat)
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str) -> Dict[str, Any]:
    _chat_or_404(chat_id)
    if chat_id in _active:
        _active[chat_id].cancelled = True
    store.delete_chat(chat_id)
    return {"ok": True}


@app.delete("/api/chats/{chat_id}/messages/{message_id}")
async def truncate_messages(chat_id: str, message_id: str) -> Dict[str, Any]:
    """Delete a message and everything after it (used by edit / regenerate)."""
    if chat_id in _active:
        raise HTTPException(409, "chat is generating")
    chat = _chat_or_404(chat_id)
    idx = next((i for i, m in enumerate(chat["messages"]) if m["id"] == message_id), None)
    if idx is None:
        raise HTTPException(404, "message not found")
    chat["messages"] = chat["messages"][:idx]
    chat["compacted_upto"] = min(chat.get("compacted_upto", 0), idx)
    store.save_chat(chat)
    return {"ok": True}


# --- memory ------------------------------------------------------------------

@app.get("/api/chats/{chat_id}/memory", response_class=PlainTextResponse)
async def get_memory(chat_id: str) -> str:
    _chat_or_404(chat_id)
    return store.read_memory(chat_id)


class MemoryBody(BaseModel):
    text: str


@app.put("/api/chats/{chat_id}/memory")
async def put_memory(chat_id: str, body: MemoryBody) -> Dict[str, Any]:
    _chat_or_404(chat_id)
    store.write_memory(chat_id, body.text or store.MEMORY_TEMPLATE)
    return {"ok": True}


# --- files -------------------------------------------------------------------

async def _ingest(chat_id: str, name: str, data: bytes, mime: str) -> Dict[str, Any]:
    name = _clean_name(name)
    text, kind = await asyncio.to_thread(ingest.extract_bytes, data, name, mime)
    file_id = store.new_file_id()
    folder = store.files_dir(chat_id) / file_id
    folder.mkdir(parents=True)
    suffix = Path(name).suffix.lower()[:12]
    (folder / f"original{suffix}").write_bytes(data)
    if kind != "image":
        (folder / "text.txt").write_text(text, encoding="utf-8")
    tokens = 280 if kind == "image" else await llm.count_tokens(text)
    meta = {
        "id": file_id,
        "name": name,
        "kind": kind,
        "mime": mime or mimetypes.guess_type(name)[0] or "application/octet-stream",
        "bytes": len(data),
        "chars": len(text),
        "tokens": tokens,
        "enabled": True,
        "added": time.time(),
    }
    store.save_file_meta(chat_id, meta)
    return meta


@app.post("/api/chats/{chat_id}/files")
async def upload_files(chat_id: str, files: List[UploadFile] = File(...)) -> List[Dict[str, Any]]:
    _chat_or_404(chat_id)
    out = []
    for upload in files:
        data = await upload.read()
        if len(data) > config.UPLOAD_MAX_BYTES:
            raise HTTPException(413, f"{upload.filename} exceeds {config.UPLOAD_MAX_BYTES // (1024 * 1024)} MB")
        if ingest.kind_for(upload.filename or "", upload.content_type or "") == "image" and not config.VISION_ENABLED:
            raise HTTPException(415, "Image input is disabled (HELPER_VISION=0)")
        out.append(await _ingest(chat_id, upload.filename or "file", data, upload.content_type or ""))
    return out


class PasteBody(BaseModel):
    text: str
    name: Optional[str] = None


@app.post("/api/chats/{chat_id}/paste")
async def paste_as_file(chat_id: str, body: PasteBody) -> Dict[str, Any]:
    _chat_or_404(chat_id)
    name = body.name or f"pasted-{time.strftime('%H%M%S')}.txt"
    return await _ingest(chat_id, name, body.text.encode("utf-8"), "text/plain")


class FilePatch(BaseModel):
    enabled: Optional[bool] = None


@app.patch("/api/chats/{chat_id}/files/{file_id}")
async def patch_file(chat_id: str, file_id: str, body: FilePatch) -> Dict[str, Any]:
    _chat_or_404(chat_id)
    try:
        meta = store.get_file_meta(chat_id, file_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, "file not found")
    if body.enabled is not None:
        meta["enabled"] = body.enabled
    store.save_file_meta(chat_id, meta)
    return meta


@app.delete("/api/chats/{chat_id}/files/{file_id}")
async def delete_file(chat_id: str, file_id: str) -> Dict[str, Any]:
    _chat_or_404(chat_id)
    try:
        store.delete_file(chat_id, file_id)
    except KeyError:
        raise HTTPException(404, "file not found")
    return {"ok": True}


@app.get("/api/chats/{chat_id}/files/{file_id}/raw")
async def file_raw(chat_id: str, file_id: str) -> FileResponse:
    try:
        meta = store.get_file_meta(chat_id, file_id)
        path = store.find_original(chat_id, file_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, "file not found")
    if not path:
        raise HTTPException(404, "file not found")
    return FileResponse(path, media_type=meta.get("mime"), filename=meta["name"], content_disposition_type="inline")


@app.get("/api/chats/{chat_id}/files/{file_id}/text", response_class=PlainTextResponse)
async def file_text(chat_id: str, file_id: str) -> str:
    try:
        return store.read_file_text(chat_id, file_id)
    except KeyError:
        raise HTTPException(404, "file not found")


# --- turns -------------------------------------------------------------------

class TurnBody(BaseModel):
    text: str = ""
    attachments: List[str] = []
    thinking: bool = True


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _close_dangling_turn(chat_id: str, text: str) -> None:
    """Keep user/assistant alternation intact (Gemma's template requires it) after errors or stops."""
    try:
        chat = store.load_chat(chat_id)
    except (KeyError, FileNotFoundError):
        return
    if chat["messages"] and chat["messages"][-1]["role"] == "user":
        chat["messages"].append(store.new_message("assistant", text, error=True))
        store.save_chat(chat)


@app.post("/api/chats/{chat_id}/turn")
async def turn(chat_id: str, body: TurnBody, request: Request) -> StreamingResponse:
    _chat_or_404(chat_id)
    if chat_id in _active:
        raise HTTPException(409, "already generating in this chat")
    if not body.text.strip() and not body.attachments:
        raise HTTPException(400, "empty message")
    worker = engine.Turn(chat_id, thinking=body.thinking)
    _active[chat_id] = worker

    async def events():
        try:
            if _gpu_lock.locked():
                yield _sse({"type": "status", "text": "Waiting for another chat to finish…"})
            async with _gpu_lock:
                async for event in worker.run(body.text, body.attachments):
                    if await request.is_disconnected():
                        worker.cancelled = True
                    yield _sse(event)
        except Exception as exc:
            _close_dangling_turn(chat_id, f"⚠️ {exc or exc.__class__.__name__}")
            yield _sse({"type": "error", "text": str(exc) or exc.__class__.__name__})
        else:
            _close_dangling_turn(chat_id, "_(stopped)_")
        finally:
            _active.pop(chat_id, None)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chats/{chat_id}/cancel")
async def cancel(chat_id: str) -> Dict[str, Any]:
    worker = _active.get(chat_id)
    if worker:
        worker.cancelled = True
    return {"ok": bool(worker)}


@app.exception_handler(KeyError)
async def _key_error(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)


def build_uvicorn_run_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "host": os.getenv("APP_HOST", "0.0.0.0"),
        "port": int(os.getenv("APP_PORT") or os.getenv("PORT") or 8000),
    }
    cert, key = os.getenv("SSL_CERTFILE", "").strip(), os.getenv("SSL_KEYFILE", "").strip()
    if cert and key:
        kwargs.update(ssl_certfile=cert, ssl_keyfile=key)
    return kwargs
