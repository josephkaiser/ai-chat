"""File-backed chat storage.

Layout per chat:
    data/helper/chats/<chat_id>/chat.json     metadata + messages
    data/helper/chats/<chat_id>/memory.md     scratch memory; starts empty for every new chat
    data/helper/chats/<chat_id>/files/<fid>/  original upload + extracted text + meta.json
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

_lock = threading.RLock()
_ID_RE = re.compile(r"^[a-z0-9-]{6,40}$")

MEMORY_TEMPLATE = "# Memory\n\n"


def _now() -> float:
    return time.time()


def chats_root() -> Path:
    root = config.DATA_ROOT / "chats"
    root.mkdir(parents=True, exist_ok=True)
    return root


def chat_dir(chat_id: str) -> Path:
    if not _ID_RE.match(chat_id or ""):
        raise KeyError(chat_id)
    path = chats_root() / chat_id
    if not path.is_dir():
        raise KeyError(chat_id)
    return path


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def create_chat(title: str = "New chat") -> Dict[str, Any]:
    with _lock:
        chat_id = uuid.uuid4().hex[:12]
        path = chats_root() / chat_id
        (path / "files").mkdir(parents=True)
        chat = {
            "id": chat_id,
            "title": title,
            "created": _now(),
            "updated": _now(),
            "messages": [],
            # Index of the first message still sent verbatim; earlier turns live in memory.md.
            "compacted_upto": 0,
        }
        _write_json(path / "chat.json", chat)
        (path / "memory.md").write_text(MEMORY_TEMPLATE, encoding="utf-8")
        return chat


def load_chat(chat_id: str) -> Dict[str, Any]:
    with _lock:
        return json.loads((chat_dir(chat_id) / "chat.json").read_text(encoding="utf-8"))


def save_chat(chat: Dict[str, Any]) -> None:
    with _lock:
        chat["updated"] = _now()
        _write_json(chat_dir(chat["id"]) / "chat.json", chat)


def list_chats() -> List[Dict[str, Any]]:
    out = []
    for path in chats_root().iterdir():
        meta = path / "chat.json"
        if not meta.is_file():
            continue
        try:
            chat = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"id": chat["id"], "title": chat.get("title") or "Chat", "updated": chat.get("updated", 0)})
    out.sort(key=lambda c: c["updated"], reverse=True)
    return out


def delete_chat(chat_id: str) -> None:
    with _lock:
        shutil.rmtree(chat_dir(chat_id), ignore_errors=True)


def new_message(role: str, content: str, **extra: Any) -> Dict[str, Any]:
    msg = {"id": uuid.uuid4().hex[:10], "role": role, "content": content, "ts": _now()}
    msg.update(extra)
    return msg


# --- memory.md ---------------------------------------------------------------

def read_memory(chat_id: str) -> str:
    path = chat_dir(chat_id) / "memory.md"
    return path.read_text(encoding="utf-8") if path.is_file() else MEMORY_TEMPLATE


def write_memory(chat_id: str, text: str) -> None:
    with _lock:
        path = chat_dir(chat_id) / "memory.md"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        os.replace(tmp, path)


def append_memory(chat_id: str, section: str, body: str) -> None:
    body = body.strip()
    if not body:
        return
    with _lock:
        current = read_memory(chat_id).rstrip() + "\n\n"
        write_memory(chat_id, f"{current}## {section}\n{body}\n")


# --- files -------------------------------------------------------------------

def files_dir(chat_id: str) -> Path:
    path = chat_dir(chat_id) / "files"
    path.mkdir(exist_ok=True)
    return path


def new_file_id() -> str:
    return uuid.uuid4().hex[:10]


def file_dir(chat_id: str, file_id: str) -> Path:
    if not _ID_RE.match(file_id or ""):
        raise KeyError(file_id)
    path = files_dir(chat_id) / file_id
    if not path.is_dir():
        raise KeyError(file_id)
    return path


def save_file_meta(chat_id: str, meta: Dict[str, Any]) -> None:
    with _lock:
        _write_json(files_dir(chat_id) / meta["id"] / "meta.json", meta)


def list_files(chat_id: str) -> List[Dict[str, Any]]:
    out = []
    for path in files_dir(chat_id).iterdir():
        meta = path / "meta.json"
        if meta.is_file():
            try:
                out.append(json.loads(meta.read_text(encoding="utf-8")))
            except Exception:
                continue
    out.sort(key=lambda m: m.get("added", 0))
    return out


def get_file_meta(chat_id: str, file_id: str) -> Dict[str, Any]:
    return json.loads((file_dir(chat_id, file_id) / "meta.json").read_text(encoding="utf-8"))


def read_file_text(chat_id: str, file_id: str) -> str:
    path = file_dir(chat_id, file_id) / "text.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def delete_file(chat_id: str, file_id: str) -> None:
    with _lock:
        shutil.rmtree(file_dir(chat_id, file_id), ignore_errors=True)


def find_original(chat_id: str, file_id: str) -> Optional[Path]:
    folder = file_dir(chat_id, file_id)
    for child in folder.iterdir():
        if child.name.startswith("original"):
            return child
    return None
