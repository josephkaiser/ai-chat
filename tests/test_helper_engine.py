"""Offline tests for the helper turn engine using a fake model with a tiny context window."""

import asyncio
import importlib

import pytest


@pytest.fixture()
def helper(tmp_path, monkeypatch):
    monkeypatch.setenv("HELPER_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("HELPER_REPLY_RESERVE", "512")
    monkeypatch.setenv("HELPER_CHUNK_TOKENS_MAX", "1500")
    from src.python.helper import config, engine, llm, store

    importlib.reload(config)
    for mod in (llm, store, engine):
        importlib.reload(mod)

    calls = []

    async def model_info(refresh=False):
        return {"id": "fake", "max_model_len": 4096, "ready": True}

    async def count_tokens(text):
        return llm.estimate_tokens(text)

    async def count_message_tokens(messages):
        return sum(
            llm.estimate_tokens(m["content"] if isinstance(m["content"], str) else str(m["content"]))
            for m in messages
        )

    async def stream_chat(messages, *, max_tokens, thinking, temperature=None):
        system = messages[0]["content"]
        calls.append({"system": system, "messages": messages, "thinking": thinking})
        if system.startswith("You are reading one part"):
            reply = "- found: the secret word is BANANA (big.txt)" if "BANANA" in messages[-1]["content"] else "NONE"
        elif system.startswith("Summarise the following earlier"):
            reply = "- earlier: user discussed apples"
        else:
            reply = "Answer text.\n<remember>\n- user likes terse answers\n</remember>"
        if thinking:
            yield {"type": "reasoning", "text": "hmm"}
        yield {"type": "content", "text": reply}
        yield {"type": "done", "finish_reason": "stop"}

    monkeypatch.setattr(llm, "model_info", model_info)
    monkeypatch.setattr(llm, "count_tokens", count_tokens)
    monkeypatch.setattr(llm, "count_message_tokens", count_message_tokens)
    monkeypatch.setattr(llm, "stream_chat", stream_chat)
    return config, engine, store, calls


def _add_text_file(store, chat_id, name, text):
    fid = store.new_file_id()
    folder = store.files_dir(chat_id) / fid
    folder.mkdir()
    (folder / "original.txt").write_text(text)
    (folder / "text.txt").write_text(text)
    store.save_file_meta(chat_id, {
        "id": fid, "name": name, "kind": "text", "mime": "text/plain",
        "bytes": len(text), "chars": len(text), "tokens": int(len(text) / 3.6),
        "enabled": True, "added": 0,
    })
    return fid


def _run(engine, chat_id, text, attachments=(), thinking=False):
    async def go():
        return [e async for e in engine.Turn(chat_id, thinking=thinking).run(text, list(attachments))]
    return asyncio.run(go())


def test_small_file_goes_verbatim_and_remember_is_saved(helper):
    config, engine, store, calls = helper
    chat = store.create_chat()
    fid = _add_text_file(store, chat["id"], "notes.txt", "hello world\n" * 20)
    events = _run(engine, chat["id"], "what does it say?", [fid])

    ctx = next(e for e in events if e["type"] == "context")
    assert ctx["verbatim"] == ["notes.txt"] and ctx["iterated"] == []
    assert "hello world" in calls[-1]["system"]
    done = next(e for e in events if e["type"] == "done")
    assert "<remember>" not in done["message"]["content"]
    assert "user likes terse answers" in store.read_memory(chat["id"])
    saved = store.load_chat(chat["id"])["messages"]
    assert [m["role"] for m in saved] == ["user", "assistant"]


def test_oversized_file_is_read_in_passes_and_noted(helper):
    config, engine, store, calls = helper
    chat = store.create_chat()
    big = ("filler line about nothing in particular\n" * 400) + "the secret word is BANANA\n" + ("more filler\n" * 400)
    fid = _add_text_file(store, chat["id"], "big.txt", big)
    events = _run(engine, chat["id"], "what is the secret word?", [fid])

    passes = [c for c in calls if c["system"].startswith("You are reading one part")]
    assert len(passes) >= 2
    ctx = next(e for e in events if e["type"] == "context")
    assert ctx["iterated"] == ["big.txt"]
    assert "BANANA" not in calls[-1]["system"]  # not verbatim
    assert "BANANA" in calls[-1]["messages"][-1]["content"]  # via notes in memory.md
    assert "Reading notes" in store.read_memory(chat["id"])
    assert ctx["prompt_tokens"] < 4096


def test_long_history_is_folded_into_memory(helper):
    config, engine, store, calls = helper
    chat = store.create_chat()
    for i in range(12):
        _run(engine, chat["id"], f"question {i} " + "about apples " * 60)
    saved = store.load_chat(chat["id"])
    assert saved["compacted_upto"] > 0 and saved["compacted_upto"] % 2 == 0
    assert "Earlier conversation" in store.read_memory(chat["id"])
    final = calls[-1]["messages"]
    assert final[1]["role"] == "user"  # history still starts on a user turn
    roles = [m["role"] for m in final[1:]]
    assert all(a != b for a, b in zip(roles, roles[1:]))  # strict alternation


def test_new_chat_starts_with_blank_memory(helper):
    config, engine, store, calls = helper
    first = store.create_chat()
    store.append_memory(first["id"], "Note", "- something")
    second = store.create_chat()
    assert "something" not in store.read_memory(second["id"])
    assert store.read_memory(second["id"]) == store.MEMORY_TEMPLATE


def test_extract_remember_strips_unterminated_block(helper):
    config, engine, store, calls = helper
    clean, notes = engine.extract_remember("Hi\n<remember>\n- a\n")
    assert clean == "Hi" and notes == ["- a"]
