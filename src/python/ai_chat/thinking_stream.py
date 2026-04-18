"""
Split model output into "thinking" vs visible answer for the WebSocket UI.

Must stay in sync with THINK_TAG_PAIRS in src/web/app.js (same open/close tags).
Qwen-style models may wrap reasoning in redacted_thinking or think tags; boundaries can span stream chunks.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Pairs (open, close) — order matters when multiple opens appear in one buffer (earliest wins).
THINK_TAG_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("<redacted_thinking>", "</redacted_thinking>"),
    ("<think>", "</think>"),
)


def _strip_incomplete_tag_suffix(buffer: str, *needles: str) -> Tuple[str, str]:
    """If buffer might end mid-tag, return (safe_to_emit, carry)."""
    if not buffer or not needles:
        return buffer, ""
    for needle in needles:
        max_k = min(len(buffer), len(needle) - 1)
        for k in range(max_k, 0, -1):
            if needle.startswith(buffer[-k:]):
                return buffer[:-k], buffer[-k:]
    return buffer, ""


def strip_stream_special_tokens(text: str) -> str:
    """Remove chat-template special tokens and thinking blocks from saved text."""
    if not text:
        return text
    # Remove <|xxx|> template tokens
    text = re.sub(r"<\|[^|]*\|>", "", text)
    # Remove <think>...</think> and <redacted_thinking>...</redacted_thinking> blocks
    text = re.sub(r"<redacted_thinking>.*?</redacted_thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _strip_template_noise(raw: str) -> str:
    return re.sub(r"<\|[^|]*\|>", "", raw) if raw else ""


class ThinkingStreamSplitter:
    """
    Consumes raw vLLM token strings and returns WebSocket JSON payloads to send.

    full_response accumulates the raw assistant text (before final strip_stream_special_tokens).
    """

    def __init__(self, pairs: Tuple[Tuple[str, str], ...] = THINK_TAG_PAIRS) -> None:
        self._pairs = pairs
        self.full_response = ""
        self._buf = ""
        self._in_think = False
        self._think_close = ""

    def feed(self, raw_token: str) -> List[Dict]:
        token = _strip_template_noise(raw_token)
        if not token:
            return []

        out: List[Dict] = []
        self.full_response += token
        self._buf += token

        while self._buf:
            if not self._in_think:
                best_o = None
                best_pair = None
                for o_tag, c_tag in self._pairs:
                    idx = self._buf.find(o_tag)
                    if idx != -1 and (best_o is None or idx < best_o):
                        best_o, best_pair = idx, (o_tag, c_tag)
                if best_o is not None:
                    o_tag, c_tag = best_pair
                    before = self._buf[:best_o]
                    if before:
                        out.append({"type": "token", "content": before})
                    out.append({"type": "think_start"})
                    self._think_close = c_tag
                    self._buf = self._buf[best_o + len(o_tag) :]
                    self._in_think = True
                    continue
                open_needles = tuple(p[0] for p in self._pairs)
                safe, carry = _strip_incomplete_tag_suffix(self._buf, *open_needles)
                if safe:
                    out.append({"type": "token", "content": safe})
                self._buf = carry
                break

            c = self._buf.find(self._think_close)
            if c != -1:
                think_chunk = self._buf[:c]
                if think_chunk:
                    out.append({"type": "think_token", "content": think_chunk})
                out.append({"type": "think_end"})
                self._buf = self._buf[c + len(self._think_close) :]
                self._in_think = False
                self._think_close = ""
                continue
            safe, carry = _strip_incomplete_tag_suffix(self._buf, self._think_close)
            if safe:
                out.append({"type": "think_token", "content": safe})
            self._buf = carry
            break

        return out

    def finalize(self) -> List[Dict]:
        out: List[Dict] = []
        if self._in_think:
            if self._buf:
                out.append({"type": "think_token", "content": self._buf})
            out.append({"type": "think_end"})
        elif self._buf:
            out.append({"type": "token", "content": self._buf})
        return out
