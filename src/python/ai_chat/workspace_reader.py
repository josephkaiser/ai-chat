from __future__ import annotations

import pathlib
import mimetypes
import zipfile
from typing import Any, Callable, Dict, Optional


MARKDOWN_EXTENSIONS = {".md", ".markdown", ".rst"}
HTML_EXTENSIONS = {".htm", ".html"}
DELIMITED_TEXT_EXTENSIONS = {".csv", ".tsv"}
BINARY_SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
ARCHIVE_EXTENSIONS = {".zip"}
RTF_EXTENSIONS = {".rtf"}

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


def is_pdf_path(path: str | pathlib.Path) -> bool:
    return pathlib.Path(path).suffix.lower() == ".pdf"


def normalize_pause_reason(value: Any) -> str:
    cleaned = str(value or "").strip().lower()
    return cleaned if cleaned in KNOWN_PAUSE_REASONS else ""


def build_tool_loop_hard_limit_message(progress_summary: str) -> str:
    summary = str(progress_summary or "").strip()
    lines = [
        "Paused after reaching the current tool budget.",
        "",
    ]
    if summary:
        lines.extend([summary, ""])
    lines.append("Saved progress is ready to resume. Say continue to pick up from the current workspace state.")
    return "\n".join(lines)


def workspace_file_content_kind(path: str | pathlib.Path) -> str:
    suffix = pathlib.Path(path).suffix.lower()
    if is_pdf_path(path):
        return "pdf"
    if suffix in ARCHIVE_EXTENSIONS:
        return "archive"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in MARKDOWN_EXTENSIONS:
        return "markdown"
    if suffix in HTML_EXTENSIONS:
        return "html"
    if suffix in DELIMITED_TEXT_EXTENSIONS:
        return "csv"
    if suffix in BINARY_SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    return "text"


def workspace_file_live_reader_mode(path: str | pathlib.Path) -> str:
    suffix = pathlib.Path(path).suffix.lower()
    kind = workspace_file_content_kind(path)
    if kind == "pdf":
        return "document_preview"
    if kind == "archive":
        return "archive_preview"
    if suffix in RTF_EXTENSIONS:
        return "document_preview"
    if kind == "spreadsheet":
        return "spreadsheet"
    if kind == "image":
        return "binary_preview"
    return "text"


def workspace_file_is_editable(path: str | pathlib.Path) -> bool:
    return workspace_file_live_reader_mode(path) == "text"


def workspace_file_default_view(path: str | pathlib.Path) -> str:
    return "preview"


def build_pdf_inline_preview_result(
    target: pathlib.Path,
    rel_path: str,
    *,
    error_message: str = "",
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if str(error_message or "").strip():
        metadata["preview_error"] = str(error_message).strip()
    return {
        "path": rel_path,
        "content": "",
        "size": target.stat().st_size,
        "lines": 0,
        "file_type": "pdf",
        "extractor": "",
        "page_count": None,
        "title": target.name,
        "metadata": metadata,
        "truncated": False,
    }


def build_binary_preview_result(
    target: pathlib.Path,
    rel_path: str,
) -> Dict[str, Any]:
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return {
        "path": rel_path,
        "content": "",
        "size": target.stat().st_size,
        "lines": 0,
        "file_type": "binary",
        "media_type": media_type,
        "truncated": False,
    }


def build_archive_preview_result(
    target: pathlib.Path,
    rel_path: str,
    *,
    max_entries: int = 200,
) -> Dict[str, Any]:
    entries = []
    truncated = False
    with zipfile.ZipFile(target) as archive:
        members = archive.infolist()
        truncated = len(members) > max_entries
        for info in members[:max_entries]:
            entries.append({
                "path": info.filename,
                "is_dir": info.is_dir(),
                "size": int(info.file_size),
                "compressed_size": int(info.compress_size),
            })
    listing = "\n".join(
        f"{entry['path']}{'/' if entry['is_dir'] and not str(entry['path']).endswith('/') else ''}"
        for entry in entries
    )
    return {
        "path": rel_path,
        "content": listing,
        "size": target.stat().st_size,
        "lines": listing.count("\n") + 1 if listing else 0,
        "file_type": "zip",
        "entry_count": len(entries),
        "entries": entries,
        "truncated": truncated,
    }


def build_text_file_result(
    target: pathlib.Path,
    rel_path: str,
    *,
    max_bytes: int,
    limit: Optional[int] = None,
    truncate_output_func: Callable[[str, int], str],
) -> Dict[str, Any]:
    if target.stat().st_size > max_bytes:
        raise ValueError("File too large (max 1MB)")

    with target.open("r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    truncated = False
    if limit is not None and limit > 0 and len(content) > limit:
        content = truncate_output_func(content, limit)
        truncated = True

    return {
        "path": rel_path,
        "content": content,
        "size": target.stat().st_size,
        "lines": content.count("\n") + 1 if content else 0,
        "file_type": "text",
        "truncated": truncated,
    }


def merge_document_outline(
    payload: Dict[str, Any],
    outline_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(outline_payload, dict):
        return payload
    for key in ("title", "extractor", "page_count", "section_titles", "chunk_count", "line_count", "opening_preview", "preview_chunks", "summary"):
        value = outline_payload.get(key)
        if value in (None, "", [], {}):
            continue
        payload[key] = value
    metadata = outline_payload.get("metadata")
    if isinstance(metadata, dict) and metadata:
        payload["metadata"] = metadata
    return payload


def build_workspace_file_result(
    target: pathlib.Path,
    *,
    rel_path: Optional[str],
    max_bytes: int,
    document_preview_builder: Callable[..., Dict[str, Any]],
    text_limit: Optional[int] = None,
    truncate_output_func: Callable[[str, int], str],
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_rel_path = rel_path or target.name
    live_mode = workspace_file_live_reader_mode(target)
    if live_mode == "spreadsheet":
        raise ValueError("Spreadsheet files should be opened with the spreadsheet preview tool")
    if live_mode == "document_preview":
        try:
            payload = document_preview_builder(
                target,
                conversation_id=conversation_id,
                rel_path=resolved_rel_path,
                limit=text_limit,
            )
            payload["truncated"] = bool(payload.get("truncated", False))
        except Exception as exc:
            if workspace_file_content_kind(target) != "pdf":
                raise
            payload = build_pdf_inline_preview_result(
                target,
                resolved_rel_path,
                error_message=str(exc),
            )
    elif live_mode == "binary_preview":
        payload = build_binary_preview_result(target, resolved_rel_path)
    elif live_mode == "archive_preview":
        payload = build_archive_preview_result(target, resolved_rel_path)
    else:
        payload = build_text_file_result(
            target,
            resolved_rel_path,
            max_bytes=max_bytes,
            limit=text_limit,
            truncate_output_func=truncate_output_func,
        )
        if conversation_id:
            try:
                payload = merge_document_outline(
                    payload,
                    document_preview_builder(
                        target,
                        conversation_id=conversation_id,
                        rel_path=resolved_rel_path,
                        limit=text_limit or len(str(payload.get("content") or "")) or 1,
                    ),
                )
            except Exception:
                pass

    payload.update({
        "content_kind": workspace_file_content_kind(target),
        "editable": workspace_file_is_editable(target),
        "default_view": workspace_file_default_view(target),
    })
    return payload
