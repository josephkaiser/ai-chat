"""Runtime-layer composition helpers for visible chat and hidden file turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

TurnKind = Literal["visible_chat", "runtime_file"]
LayerVisibility = Literal["visible", "model_only"]

TURN_KIND_VISIBLE_CHAT: TurnKind = "visible_chat"
TURN_KIND_RUNTIME_FILE: TurnKind = "runtime_file"


@dataclass(frozen=True)
class RuntimeLayer:
    """One bounded context layer attached to a turn."""

    name: str
    content: str
    visibility: LayerVisibility = "model_only"


@dataclass(frozen=True)
class RuntimeTurnEnvelope:
    """Normalized visible/model messages for one inbound turn."""

    turn_kind: TurnKind
    raw_message: str
    saved_user_message: str
    effective_message: str
    model_message: str
    attachment_context: str = ""
    runtime_context: str = ""
    slash_request: str = ""
    layers: List[RuntimeLayer] = field(default_factory=list)


def normalize_turn_kind(value: Any) -> TurnKind:
    """Normalize untrusted turn-kind input into a supported value."""
    normalized = str(value or "").strip().lower()
    if normalized == TURN_KIND_RUNTIME_FILE:
        return TURN_KIND_RUNTIME_FILE
    return TURN_KIND_VISIBLE_CHAT


def visible_message_kind_sql(column: str = "kind") -> str:
    """Return the SQL clause used to exclude hidden runtime turns from user-facing queries."""
    safe_column = str(column or "kind").strip() or "kind"
    return f"COALESCE({safe_column}, '{TURN_KIND_VISIBLE_CHAT}') = '{TURN_KIND_VISIBLE_CHAT}'"


def compose_runtime_turn(
    *,
    raw_message: str,
    attachment_context: str = "",
    runtime_context: str = "",
    slash_request: str = "",
    turn_kind: Any = TURN_KIND_VISIBLE_CHAT,
) -> RuntimeTurnEnvelope:
    """Build the visible and model-facing message payloads for one turn."""
    normalized_turn_kind = normalize_turn_kind(turn_kind)
    cleaned_message = str(raw_message or "").strip()
    cleaned_attachment_context = str(attachment_context or "").strip()
    cleaned_runtime_context = str(runtime_context or "").strip()
    cleaned_slash_request = str(slash_request or "").strip()

    saved_user_message = cleaned_message
    if cleaned_attachment_context:
        saved_user_message = f"{cleaned_message}\n\n{cleaned_attachment_context}" if cleaned_message else cleaned_attachment_context

    effective_message = cleaned_slash_request or saved_user_message
    layers: List[RuntimeLayer] = []
    if cleaned_attachment_context:
        layers.append(RuntimeLayer(name="attachments", content=cleaned_attachment_context, visibility="visible"))
    if cleaned_runtime_context:
        layers.append(RuntimeLayer(name="runtime_context", content=cleaned_runtime_context, visibility="model_only"))
    model_message = compose_model_message(effective_message, layers=layers)

    return RuntimeTurnEnvelope(
        turn_kind=normalized_turn_kind,
        raw_message=cleaned_message,
        saved_user_message=saved_user_message,
        effective_message=effective_message,
        model_message=model_message,
        attachment_context=cleaned_attachment_context,
        runtime_context=cleaned_runtime_context,
        slash_request=cleaned_slash_request,
        layers=layers,
    )


def compose_model_message(
    effective_message: str,
    *,
    runtime_context: str = "",
    layers: Optional[List[RuntimeLayer]] = None,
) -> str:
    """Rebuild the model-facing prompt from the latest effective message plus model-only layers."""
    base = str(effective_message or "").strip()
    suffixes: List[str] = []
    if layers is not None:
        for layer in layers:
            if getattr(layer, "visibility", "") != "model_only":
                continue
            cleaned = str(getattr(layer, "content", "") or "").strip()
            if cleaned:
                suffixes.append(cleaned)
    else:
        cleaned_runtime_context = str(runtime_context or "").strip()
        if cleaned_runtime_context:
            suffixes.append(cleaned_runtime_context)

    parts = [base] if base else []
    parts.extend(suffixes)
    return "\n\n".join(parts)


def build_model_history(
    history: List[Dict[str, str]],
    *,
    effective_message: str,
    model_message: str,
) -> List[Dict[str, str]]:
    """Return a transient history copy with model-only layers applied to the latest user turn."""
    cloned_history = [dict(item) for item in (history or [])]
    cleaned_effective = str(effective_message or "").strip()
    cleaned_model = str(model_message or "").strip()
    if not cleaned_model or cleaned_model == cleaned_effective:
        return cloned_history
    if cloned_history and str(cloned_history[-1].get("role") or "") == "user":
        cloned_history[-1]["content"] = cleaned_model
        return cloned_history
    cloned_history.append({"role": "user", "content": cleaned_model})
    return cloned_history
