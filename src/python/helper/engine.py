"""Turn engine: fits files + history + memory.md into the context window, iterating when it can't.

Prompt layout (ordered for vLLM prefix-cache reuse):
    system   : instructions + every file that fits verbatim (new files append at the end)
    history  : turns since `compacted_upto` (older turns were folded into memory.md)
    user     : <memory.md> ... </memory.md> + reading notes + the new request (+ images)

When the material doesn't fit:
    * old turns are summarised into memory.md and dropped from the prompt
    * files that don't fit are read in chunks; each pass extracts notes relevant to the request,
      the notes are saved to memory.md, and the final answer is written from those notes
    * memory.md itself is condensed when it outgrows its budget
"""

from __future__ import annotations

import base64
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from . import config, llm, store

SYSTEM_PROMPT = """You are a sharp, practical all-purpose assistant running locally. The user pastes text and drops files in for you to read, review, summarise, transform, debug, or use as reference, and copies your output back out.

How to work:
- Ground answers in the attached files and memory when they are relevant. Cite where things are (file name, page, section, function, heading) so the user can find them; only give line numbers you can actually see.
- When asked to transform or produce text/code, output the complete result ready to copy — no elisions like "rest unchanged" unless the user asks for a diff.
- Put code, configs, commands and anything meant to be copied verbatim in fenced code blocks with a language tag.
- Be direct. Skip preamble and restating the question. Use headings/lists only when they help.
- If something you need is missing from the files (truncated, unreadable, not attached), say so plainly instead of guessing.

Memory: the user message carries the current memory.md for this chat. It holds facts, decisions, summaries of earlier turns and reading notes from large files. Treat it as reliable context. To save something worth keeping for later turns (user preferences, decisions, key facts, open TODOs), end your reply with a block:
<remember>
- one concise fact per line
</remember>
Only remember things that will matter later; never repeat what memory already says. The block is hidden from the user."""

READ_PASS_PROMPT = """You are reading one part of a large body of material on behalf of an assistant that cannot see it directly. The assistant must answer the user's request below using only your notes.

Extract everything from THIS PART that is relevant or plausibly useful for the request: facts, figures, names, definitions, decisions, errors, code (verbatim when short), and exact short quotes. Always note where it came from (file name plus page/section/line/function as available). If the request is broad (summarise / review / explain), capture the structure and key points of this part instead.

Be dense and factual. No preamble, no commentary about the task. Record only what IS in this part — never write that something is absent or "not mentioned". Lines carry ⟨L123⟩ markers every 10 lines (markers are not part of the file): cite line numbers by counting from the nearest marker. If this part has nothing useful, reply with exactly: NONE"""

COMPACT_PROMPT = """Summarise the following earlier conversation turns for long-term memory. Keep: what the user asked for, decisions made, important facts/numbers/names, code or text that was produced (describe it; keep short key snippets), preferences the user expressed, and anything left unresolved. Drop pleasantries and redundancy. Write terse markdown bullets."""

CONDENSE_MEMORY_PROMPT = """Rewrite this memory.md to be shorter while keeping every fact that could matter later (decisions, preferences, key facts, figures, names, unresolved items, where things are in which file). Merge duplicates, drop stale or superseded items. Keep markdown with '## ' sections. Output only the new memory.md."""

LINE_MARK_RE = re.compile(r"⟨L(\d+)⟩")
REMEMBER_RE = re.compile(r"<remember>(.*?)(?:</remember>|$)", re.S | re.I)
SAFETY_TOKENS = 256


# --- helpers -----------------------------------------------------------------

def _file_block(meta: Dict[str, Any], text: str) -> str:
    return f'<file name="{meta["name"]}" kind="{meta["kind"]}">\n{text}\n</file>'


def _history_text(msg: Dict[str, Any]) -> str:
    content = msg.get("content") or ""
    if msg["role"] == "user" and msg.get("attachments"):
        names = ", ".join(a["name"] for a in msg["attachments"])
        content = f"{content}\n\n[attached: {names}]"
    return content


async def _message_tokens(msg: Dict[str, Any]) -> int:
    if not msg.get("tokens"):
        msg["tokens"] = await llm.count_tokens(_history_text(msg)) + 6
    return int(msg["tokens"])


def _image_part(chat_id: str, file_id: str) -> Optional[Dict[str, Any]]:
    try:
        meta = store.get_file_meta(chat_id, file_id)
        path = store.find_original(chat_id, file_id)
    except KeyError:
        return None
    if not path:
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = meta.get("mime") or "image/png"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _line_marked(text: str, first_line: int, every: int = 10) -> str:
    """Prefix every Nth line with ⟨L123⟩ so reading passes can cite real line numbers cheaply (~5% tokens)."""
    lines = text.split("\n")
    return "\n".join(
        f"⟨L{n}⟩{ln}" if (n % every == 0 or i == 0) else ln for i, (n, ln) in enumerate(zip(range(first_line, first_line + len(lines)), lines))
    )


def _split_by_tokens(text: str, approx_tokens: int, chunk_tokens: int) -> List[str]:
    """Split text into ~chunk_tokens pieces on line boundaries using the file's chars/token ratio."""
    if approx_tokens <= chunk_tokens:
        return [text]
    chars_per_token = max(1.5, len(text) / max(1, approx_tokens))
    limit = int(chunk_tokens * chars_per_token)
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            cut = text.rfind("\n", start + limit // 2, end)
            if cut > start:
                end = cut + 1
        chunks.append(text[start:end])
        start = end
    return chunks


_ABSENCE_RE = re.compile(
    r"\b(not (mentioned|defined|found|present|included|specified|provided|stated|contained)|"
    r"no (mention|information|reference)|does not (contain|mention|include|define|specify))\b",
    re.I,
)


def _drop_absences(note: str) -> str:
    """Remove note lines that only report something is missing from a chunk (noise in later passes)."""
    kept = [ln for ln in (note or "").splitlines() if not _ABSENCE_RE.search(ln)]
    kept = [ln for ln in kept if not re.match(r"^\s*(based on|here are|from) (the )?(provided|this|these)\b.*:\s*$", ln, re.I)]
    text = "\n".join(kept).strip()
    return text if re.search(r"\w", text) else "NONE"


def extract_remember(content: str) -> Tuple[str, List[str]]:
    notes = [m.strip() for m in REMEMBER_RE.findall(content) if m.strip()]
    clean = LINE_MARK_RE.sub(r"line \1", REMEMBER_RE.sub("", content)).rstrip()
    return clean, notes


def _short(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --- the turn ----------------------------------------------------------------

class Turn:
    def __init__(self, chat_id: str, thinking: bool):
        self.chat_id = chat_id
        self.thinking = thinking
        self.cancelled = False

    async def run(self, user_text: str, attachment_ids: List[str]) -> AsyncIterator[Dict[str, Any]]:
        chat = store.load_chat(self.chat_id)
        info = await llm.model_info(refresh=True)
        if not info.get("ready"):
            raise RuntimeError("The model server is still loading. Give it a minute and try again.")
        max_len = int(info["max_model_len"])
        reply_reserve = min(config.REPLY_RESERVE if self.thinking else config.REPLY_RESERVE // 2, max_len // 4)
        budget = max_len - reply_reserve - SAFETY_TOKENS

        files = [f for f in store.list_files(self.chat_id) if f.get("enabled", True)]
        by_id = {f["id"]: f for f in store.list_files(self.chat_id)}
        attachments = [
            {"id": fid, "name": by_id[fid]["name"], "kind": by_id[fid]["kind"]}
            for fid in attachment_ids if fid in by_id
        ]
        user_msg = store.new_message("user", user_text, attachments=attachments)
        chat["messages"].append(user_msg)
        if len(chat["messages"]) == 1 or chat.get("title") in {"", "New chat"}:
            chat["title"] = _short(user_text or (attachments[0]["name"] if attachments else "Chat"), 60)
        store.save_chat(chat)
        yield {"type": "user_saved", "message": user_msg, "title": chat["title"]}

        started = time.time()
        system_tokens = await llm.count_tokens(SYSTEM_PROMPT) + 8

        # 1. memory.md must stay within its own budget.
        memory = store.read_memory(self.chat_id)
        mem_tokens = await llm.count_tokens(memory)
        mem_cap = min(config.MEMORY_TOKENS_MAX, budget // 6)
        if mem_tokens > mem_cap:
            yield {"type": "status", "text": "Condensing memory.md…"}
            memory = await self._condense_memory(memory, mem_cap)
            mem_tokens = await llm.count_tokens(memory)
            yield {"type": "memory"}

        # 2. Keep recent history within its share; fold older turns into memory.md.
        history = chat["messages"][chat.get("compacted_upto", 0):-1]
        hist_tokens = [await self._safe_tokens(m) for m in history]
        hist_cap = int(budget * config.HISTORY_SHARE)
        if sum(hist_tokens) > hist_cap and len(history) > 2:
            drop = 0
            while sum(hist_tokens[drop:]) > hist_cap and drop < len(history) - 2:
                drop += 2  # whole user/assistant pairs
            yield {"type": "status", "text": f"Folding {drop} older messages into memory.md…"}
            summary = await self._compact(history[:drop])
            first = chat.get("compacted_upto", 0)
            store.append_memory(self.chat_id, f"Earlier conversation (messages {first + 1}–{first + drop})", summary)
            chat["compacted_upto"] = first + drop
            store.save_chat(chat)
            history, hist_tokens = history[drop:], hist_tokens[drop:]
            memory = store.read_memory(self.chat_id)
            mem_tokens = await llm.count_tokens(memory)
            yield {"type": "memory"}
        user_tokens = await self._safe_tokens(user_msg)

        # 3. Decide which files go in verbatim. Images ride along as pixels on their own turn.
        text_files = [f for f in files if f["kind"] != "image"]
        fixed = system_tokens + mem_tokens + sum(hist_tokens) + user_tokens + 64
        file_budget = max(0, budget - fixed)
        verbatim, overflow = self._choose_files(text_files, attachment_ids, file_budget)
        file_tokens = sum(int(f.get("tokens") or 0) + 12 for f in verbatim)

        # 4. Iterate through whatever doesn't fit and collect notes.
        notes = ""
        if overflow:
            room = max_len - (system_tokens + mem_tokens + user_tokens + 2048 + SAFETY_TOKENS + 600)
            chunk_tokens = max(2048, min(config.CHUNK_TOKENS_MAX, room))
            async for event in self._read_through(overflow, user_text, memory, chunk_tokens):
                if event["type"] == "notes":
                    notes = event["text"]
                else:
                    yield event
                if self.cancelled:
                    return
            notes_cap = max(1024, budget - fixed - file_tokens)
            if await llm.count_tokens(notes) > notes_cap:
                yield {"type": "status", "text": "Condensing reading notes…"}
                notes = await llm.complete(
                    [
                        {"role": "system", "content": "Condense these reading notes, keeping every detail relevant to the request and all source locations. Output only the notes."},
                        {"role": "user", "content": f"Request: {user_text}\n\n{notes}"},
                    ],
                    max_tokens=min(notes_cap, 8000),
                )
            store.append_memory(
                self.chat_id,
                f"Reading notes for: {_short(user_text, 70)} ({', '.join(f['name'] for f in overflow)})",
                notes,
            )
            memory = store.read_memory(self.chat_id)
            yield {"type": "memory"}

        # 5. Build the final prompt.
        system = SYSTEM_PROMPT
        manifest = []
        for f in files:
            status = "image" if f["kind"] == "image" else ("full text below" if f in verbatim else "too large — see reading notes in memory.md")
            manifest.append(f"- {f['name']} ({f['kind']}, ~{f.get('tokens', 0)} tokens): {status}")
        if manifest:
            system += "\n\n# Attached files\n" + "\n".join(manifest)
        for f in verbatim:
            system += "\n\n" + _file_block(f, store.read_file_text(self.chat_id, f["id"]))

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        images_left = config.MAX_IMAGES_IN_CONTEXT if config.VISION_ENABLED else 0
        rendered: List[Dict[str, Any]] = []
        for msg in reversed(history + [user_msg]):
            text = _history_text(msg)
            if msg is user_msg:
                text = f"<memory.md>\n{memory.strip()}\n</memory.md>\n\n{text}"
            parts: List[Dict[str, Any]] = []
            for att in msg.get("attachments") or []:
                if att["kind"] == "image" and images_left > 0 and att["id"] in by_id:
                    part = _image_part(self.chat_id, att["id"])
                    if part:
                        parts.append(part)
                        images_left -= 1
            if parts and msg["role"] == "user":
                rendered.append({"role": "user", "content": parts + [{"type": "text", "text": text}]})
            else:
                rendered.append({"role": msg["role"], "content": text})
        messages.extend(reversed(rendered))

        prompt_tokens = await llm.count_message_tokens(messages)
        yield {
            "type": "context",
            "prompt_tokens": prompt_tokens,
            "max_model_len": max_len,
            "verbatim": [f["name"] for f in verbatim],
            "iterated": [f["name"] for f in overflow],
            "history_messages": len(history),
            "compacted_upto": chat.get("compacted_upto", 0),
        }
        if prompt_tokens > max_len - 512:
            raise RuntimeError(
                f"Prompt is {prompt_tokens} tokens but the window is {max_len}. Disable some files or start a new chat."
            )

        # 6. Answer.
        yield {"type": "status", "text": "Thinking…" if self.thinking else "Writing…"}
        content, reasoning, finish = [], [], "stop"
        first_token_at = None
        async for event in llm.stream_chat(messages, max_tokens=max_len - prompt_tokens - 64, thinking=self.thinking):
            if self.cancelled:
                finish = "cancelled"
                break
            if event["type"] == "done":
                finish = event["finish_reason"]
                continue
            if first_token_at is None:
                first_token_at = time.time()
            (content if event["type"] == "content" else reasoning).append(event["text"])
            yield event

        full = "".join(content)
        clean, remembered = extract_remember(full)
        if remembered:
            store.append_memory(self.chat_id, f"Noted (turn {len(chat['messages'])})", "\n".join(remembered))
            yield {"type": "memory"}
        elapsed = time.time() - started
        assistant = store.new_message(
            "assistant",
            clean,
            reasoning="".join(reasoning),
            finish_reason=finish,
            stats={
                "prompt_tokens": prompt_tokens,
                "seconds": round(elapsed, 1),
                "ttft": round((first_token_at or time.time()) - started, 1),
                "iterated": [f["name"] for f in overflow],
            },
        )
        chat = store.load_chat(self.chat_id)
        chat["messages"].append(assistant)
        store.save_chat(chat)
        yield {"type": "done", "message": assistant}

    # -- pieces --

    async def _safe_tokens(self, msg: Dict[str, Any]) -> int:
        return await _message_tokens(msg)

    def _choose_files(
        self, files: List[Dict[str, Any]], fresh_ids: List[str], budget: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        total = sum(int(f.get("tokens") or 0) + 12 for f in files)
        if total <= budget:
            return files, []
        # Prefer files attached this turn, then small files (more of them fit verbatim).
        ranked = sorted(files, key=lambda f: (f["id"] not in fresh_ids, int(f.get("tokens") or 0)))
        chosen, used = set(), 0
        for f in ranked:
            cost = int(f.get("tokens") or 0) + 12
            if used + cost <= budget:
                chosen.add(f["id"])
                used += cost
        # Keep original order for verbatim files so the system prompt prefix stays cache-friendly.
        verbatim = [f for f in files if f["id"] in chosen]
        overflow = [f for f in files if f["id"] not in chosen]
        return verbatim, overflow

    async def _read_through(
        self, files: List[Dict[str, Any]], request: str, memory: str, chunk_tokens: int
    ) -> AsyncIterator[Dict[str, Any]]:
        pieces: List[Tuple[str, str]] = []
        for f in files:
            text = store.read_file_text(self.chat_id, f["id"])
            parts = _split_by_tokens(text, int(f.get("tokens") or llm.estimate_tokens(text)), chunk_tokens)
            line = 1
            for i, part in enumerate(parts, 1):
                last = line + part.count("\n")
                label = f["name"] if len(parts) == 1 else f"{f['name']} (part {i}/{len(parts)}, lines {line}–{last})"
                pieces.append((label, f"[{label}]\n" + _file_block(f, _line_marked(part, line))))
                line = last
        collected: List[str] = []
        for n, (label, block) in enumerate(pieces, 1):
            if self.cancelled:
                return
            yield {"type": "status", "text": f"Reading {label} — pass {n}/{len(pieces)}", "progress": [n, len(pieces)]}
            note = await llm.complete(
                [
                    {"role": "system", "content": READ_PASS_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"<memory.md>\n{memory.strip()}\n</memory.md>\n\nUser request: {request}\n\n{block}\n\n"
                            f"Now write extraction notes for {label} only. Bullet points of what this part DOES contain "
                            "that bears on the request, each with its location. Do not answer the request, and do not list "
                            "things this part lacks. If nothing here is relevant, reply exactly NONE."
                        ),
                    },
                ],
                max_tokens=2048,
            )
            note = _drop_absences(note)
            if note and note.strip().upper() != "NONE":
                collected.append(f"### {label}\n{note.strip()}")
                yield {"type": "note", "label": label, "text": note.strip()}
        yield {"type": "notes", "text": "\n\n".join(collected) or "(nothing relevant found)"}

    async def _compact(self, messages: List[Dict[str, Any]]) -> str:
        transcript = "\n\n".join(f"{m['role'].upper()}: {_history_text(m)}" for m in messages)
        info = await llm.model_info()
        cap_chars = int((int(info["max_model_len"]) - 4096) * 3.2)
        if len(transcript) > cap_chars:
            transcript = transcript[:cap_chars // 2] + "\n\n[...]\n\n" + transcript[-cap_chars // 2:]
        return await llm.complete(
            [{"role": "system", "content": COMPACT_PROMPT}, {"role": "user", "content": transcript}],
            max_tokens=1500,
        )

    async def _condense_memory(self, memory: str, cap: int) -> str:
        condensed = await llm.complete(
            [
                {"role": "system", "content": CONDENSE_MEMORY_PROMPT + f" Target: under {int(cap * 0.6)} tokens."},
                {"role": "user", "content": memory},
            ],
            max_tokens=int(cap * 0.8),
        )
        if condensed.strip():
            if not condensed.lstrip().startswith("#"):
                condensed = "# Memory\n\n" + condensed
            store.write_memory(self.chat_id, condensed)
            return condensed
        return memory
