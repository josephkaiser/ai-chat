"use strict";

// ---------------------------------------------------------------------------
// state

const S = {
  status: null,
  chatId: null,
  chat: null,            // {id,title,messages,files,memory,...}
  pending: [],           // file metas attached to the next message
  uploading: 0,
  generating: false,
  abort: null,
  memoryDirty: false,
  lastContext: null,
};

const PASTE_FILE_CHARS = 6000;
const PASTE_FILE_LINES = 120;

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
};

const fmtTokens = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n || 0));

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch { return null; }
}

function toast(text, ms = 2200) {
  const t = $("#toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), ms);
}

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    init.body = JSON.stringify(opts.json);
    delete init.json;
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  const type = res.headers.get("content-type") || "";
  return type.includes("json") ? res.json() : res.text();
}

// ---------------------------------------------------------------------------
// markdown

marked.setOptions({ gfm: true, breaks: false });

function renderMarkdown(text) {
  const html = DOMPurify.sanitize(marked.parse(text || ""), { ADD_ATTR: ["target"] });
  const tpl = document.createElement("template");
  tpl.innerHTML = html;
  tpl.content.querySelectorAll("pre > code").forEach((code) => {
    const pre = code.parentElement;
    const lang = (code.className.match(/language-([\w+#-]+)/) || [])[1] || "";
    try {
      if (lang && hljs.getLanguage(lang)) hljs.highlightElement(code);
      else if (code.textContent.length < 20000) hljs.highlightElement(code);
    } catch {}
    const wrap = el("div", { class: "code-wrap" });
    const tools = el("div", { class: "code-tools" },
      lang ? el("span", {}, lang) : null,
      el("button", { type: "button", "data-copy": "code" }, "Copy"),
    );
    pre.replaceWith(wrap);
    wrap.append(pre, tools);
  });
  tpl.content.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
  return tpl.content;
}

const REMEMBER_RE = /<remember>[\s\S]*?(<\/remember>|$)/gi;
const visibleText = (text) => (text || "").replace(REMEMBER_RE, "").replace(/⟨L(\d+)⟩/g, "line $1").trimEnd();

async function copyText(text, label = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = el("textarea", { style: "position:fixed;opacity:0" });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  toast(label);
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-copy='code']");
  if (!btn) return;
  const code = btn.closest(".code-wrap").querySelector("code");
  copyText(code.textContent, "Code copied");
});

// ---------------------------------------------------------------------------
// status + chat list

async function refreshStatus() {
  try {
    S.status = await api("/api/status");
  } catch {
    S.status = { ready: false };
  }
  const box = $("#model-status");
  box.classList.toggle("ready", !!S.status.ready);
  box.innerHTML = "";
  box.append(
    el("span", { class: "dot" }),
    S.status.ready
      ? `${(S.status.model || "").split("/").pop()} · ${fmtTokens(S.status.max_model_len)} ctx`
      : "model loading…",
  );
  if (!S.status.ready) setTimeout(refreshStatus, 5000);
}

async function refreshChatList() {
  const chats = await api("/api/chats");
  const list = $("#chat-list");
  list.innerHTML = "";
  for (const c of chats) {
    list.append(el("div", {
      class: `chat-item${c.id === S.chatId ? " active" : ""}`,
      title: c.title,
      onclick: () => openChat(c.id),
    },
      el("span", {}, c.title || "Chat"),
      el("button", {
        title: "Delete chat",
        onclick: async (e) => {
          e.stopPropagation();
          if (!e.shiftKey && !window.confirm(`Delete "${c.title}"? (Shift-click skips this prompt)`)) return;
          await api(`/api/chats/${c.id}`, { method: "DELETE" });
          if (c.id === S.chatId) newChat(false);
          refreshChatList();
        },
      }, "✕"),
    ));
  }
}

function newChat(focus = true) {
  if (S.generating) stopGeneration();
  S.chatId = null;
  S.chat = null;
  S.pending = [];
  S.lastContext = null;
  store("lastChat", "");
  renderAll();
  refreshChatList();
  if (focus) $("#input").focus();
}

async function ensureChat() {
  if (S.chatId) return S.chatId;
  const chat = await api("/api/chats", { method: "POST" });
  S.chatId = chat.id;
  S.chat = { ...chat, files: [], memory: "# Memory\n\n" };
  store("lastChat", chat.id);
  refreshChatList();
  return chat.id;
}

async function openChat(id) {
  if (S.generating && id !== S.chatId) stopGeneration();
  try {
    S.chat = await api(`/api/chats/${id}`);
  } catch {
    return newChat(false);
  }
  S.chatId = id;
  S.pending = [];
  S.memoryDirty = false;
  const lastAssistant = [...S.chat.messages].reverse().find((m) => m.role === "assistant" && m.stats);
  S.lastContext = lastAssistant ? { prompt_tokens: lastAssistant.stats.prompt_tokens } : null;
  store("lastChat", id);
  renderAll();
  refreshChatList();
  document.getElementById("app").classList.remove("show-sidebar");
}

async function reloadChat({ messages = false } = {}) {
  if (!S.chatId) return;
  const fresh = await api(`/api/chats/${S.chatId}`);
  if (!S.memoryDirty) S.chat.memory = fresh.memory;
  S.chat.files = fresh.files;
  S.chat.title = fresh.title;
  if (messages) S.chat.messages = fresh.messages;
  renderFiles();
  renderMemory();
  $("#chat-title").textContent = S.chat.title || "New chat";
  if (messages) renderMessages();
}

// ---------------------------------------------------------------------------
// rendering

function renderAll() {
  $("#chat-title").textContent = S.chat?.title || "New chat";
  renderMessages();
  renderFiles();
  renderMemory();
  renderPending();
  renderMeter();
}

function renderMessages() {
  const box = $("#messages");
  box.innerHTML = "";
  const msgs = S.chat?.messages || [];
  if (!msgs.length) {
    box.append(el("div", { id: "empty-state" },
      el("h2", {}, "Drop anything here"),
      el("p", { class: "muted" },
        "Files, zipped code, PDFs, docs, spreadsheets, logs, screenshots — or paste text. ",
        "Everything stays in this chat's context; big material gets read through in passes and noted in memory.md."),
    ));
    return;
  }
  const compacted = S.chat.compacted_upto || 0;
  msgs.forEach((m, i) => {
    if (i === compacted && compacted > 0) {
      box.append(el("div", { class: "small muted", style: "text-align:center;margin:-8px 0 18px" },
        `↑ ${compacted} earlier messages are summarised in memory.md (not sent verbatim)`));
    }
    box.append(renderMessage(m, i === msgs.length - 1));
  });
  box.scrollTop = box.scrollHeight;
}

function attachmentChips(attachments) {
  if (!attachments?.length) return null;
  return el("div", { class: "chips" }, attachments.map((a) => {
    const chip = el("span", { class: "chip", title: a.name });
    if (a.kind === "image" && S.chatId) chip.append(el("img", { src: `/api/chats/${S.chatId}/files/${a.id}/raw`, alt: "" }));
    chip.append(el("span", { class: "name" }, a.name));
    return chip;
  }));
}

function renderMessage(m, isLast) {
  const node = el("div", { class: `msg ${m.role}${m.error ? " error" : ""}${isLast ? " last" : ""}`, "data-id": m.id });
  if (m.role === "user") {
    node.append(attachmentChips(m.attachments) || "", el("div", { class: "body" }, m.content));
  } else {
    if (m.reasoning) node.append(reasoningBlock(m.reasoning, false));
    const body = el("div", { class: "body" });
    body.append(renderMarkdown(m.content));
    node.append(body);
  }
  node.append(messageActions(m));
  return node;
}

function reasoningBlock(text, open) {
  const d = el("details", { class: "reasoning", open },
    el("summary", {}, "Reasoning"),
    el("div", { class: "r-body" }, text));
  return d;
}

function messageActions(m) {
  const bar = el("div", { class: "actions" });
  bar.append(el("button", { type: "button", title: "Copy as markdown", onclick: () => copyText(m.content) }, "Copy"));
  if (m.role === "assistant") {
    bar.append(
      el("button", { type: "button", title: "Copy without markdown formatting", onclick: (e) => {
        const body = e.target.closest(".msg").querySelector(".body");
        copyText(body.innerText, "Copied as plain text");
      } }, "Copy text"),
      el("button", { type: "button", title: "Save as .md", onclick: () => downloadText(m.content, `${slug(S.chat?.title)}.md`) }, "Save"),
      el("button", { type: "button", title: "Regenerate this reply", onclick: () => regenerate(m) }, "Retry"),
    );
    if (m.finish_reason === "length") {
      bar.append(el("button", { type: "button", onclick: () => send("Continue exactly where you left off.") }, "Continue"));
    }
    if (m.stats) {
      const s = m.stats;
      const extra = s.iterated?.length ? ` · read through ${s.iterated.join(", ")}` : "";
      bar.append(el("span", { class: "stats" }, `${fmtTokens(s.prompt_tokens)} in · ${s.seconds}s${extra}`));
    }
  } else {
    bar.append(el("button", { type: "button", title: "Edit and resend", onclick: () => editMessage(m) }, "Edit"));
  }
  return bar;
}

function slug(t) {
  return (t || "reply").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 50) || "reply";
}

function downloadText(text, name) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  const a = el("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderFiles() {
  const list = $("#file-list");
  list.innerHTML = "";
  const files = S.chat?.files || [];
  const max = S.status?.max_model_len || 0;
  let running = 0;
  let enabledTotal = 0;
  for (const f of files) {
    const on = f.enabled !== false;
    if (on && f.kind !== "image") enabledTotal += f.tokens || 0;
    running += on && f.kind !== "image" ? f.tokens || 0 : 0;
    const over = max && on && running > max * 0.7;
    const cb = el("input", { type: "checkbox", title: "Include in context" });
    cb.checked = on;
    cb.addEventListener("change", async () => {
      await api(`/api/chats/${S.chatId}/files/${f.id}`, { method: "PATCH", json: { enabled: cb.checked } });
      reloadChat();
    });
    list.append(el("div", { class: `file-row${on ? "" : " off"}${over ? " over" : ""}` },
      cb,
      el("span", { class: "fname", title: `${f.name} — click to view`, onclick: () => viewFile(f) }, f.name),
      el("span", { class: "ftok", title: over ? "Won't fit verbatim — will be read in passes" : "tokens" },
        f.kind === "image" ? "img" : fmtTokens(f.tokens)),
      el("button", { title: "Remove", onclick: async () => {
        await api(`/api/chats/${S.chatId}/files/${f.id}`, { method: "DELETE" });
        S.pending = S.pending.filter((p) => p.id !== f.id);
        renderPending();
        reloadChat();
      } }, "✕"),
    ));
  }
  if (!files.length) list.append(el("div", { class: "small muted" }, "Nothing yet — drop files anywhere."));
  $("#files-total").textContent = files.length
    ? `${fmtTokens(enabledTotal)}${max ? ` / ${fmtTokens(max)}` : ""} tokens`
    : "";
}

async function viewFile(f) {
  const dlg = $("#viewer");
  $("#viewer-title").textContent = f.name;
  const body = $("#viewer-body");
  body.innerHTML = "";
  if (f.kind === "image") {
    body.append(el("img", { src: `/api/chats/${S.chatId}/files/${f.id}/raw`, style: "max-width:100%" }));
  } else {
    body.textContent = "Loading…";
    const text = await api(`/api/chats/${S.chatId}/files/${f.id}/text`);
    body.textContent = text.length > 400000 ? text.slice(0, 400000) + "\n\n[… preview truncated …]" : text;
  }
  dlg.showModal();
}
$("#viewer-close").onclick = () => $("#viewer").close();
$("#viewer-copy").onclick = () => copyText($("#viewer-body").innerText);

function renderMemory() {
  const ta = $("#memory");
  if (document.activeElement !== ta || !S.memoryDirty) ta.value = S.chat?.memory || "# Memory\n\n";
  ta.disabled = !S.chatId;
  $("#memory-save").hidden = !S.memoryDirty;
}

$("#memory").addEventListener("input", () => {
  S.memoryDirty = true;
  $("#memory-save").hidden = false;
});
$("#memory-save").onclick = async () => {
  if (!S.chatId) return;
  await api(`/api/chats/${S.chatId}/memory`, { method: "PUT", json: { text: $("#memory").value } });
  S.memoryDirty = false;
  S.chat.memory = $("#memory").value;
  renderMemory();
  toast("memory.md saved");
};

function renderPending() {
  const box = $("#pending");
  box.innerHTML = "";
  for (const f of S.pending) {
    const chip = el("span", { class: "chip", title: `${f.name} · ${fmtTokens(f.tokens)} tokens` });
    if (f.kind === "image") chip.append(el("img", { src: `/api/chats/${S.chatId}/files/${f.id}/raw`, alt: "" }));
    chip.append(
      el("span", { class: "name" }, f.name),
      el("span", { class: "muted" }, f.kind === "image" ? "" : fmtTokens(f.tokens)),
      el("button", { type: "button", title: "Detach (stays in Files panel)", onclick: () => {
        S.pending = S.pending.filter((p) => p !== f);
        renderPending();
      } }, "✕"),
    );
    box.append(chip);
  }
  if (S.uploading) box.append(el("span", { class: "chip busy" }, `Uploading ${S.uploading}…`));
  updateComposerHint();
}

function renderMeter() {
  const max = S.status?.max_model_len || 0;
  const used = S.lastContext?.prompt_tokens || 0;
  const pct = max ? Math.min(100, (used / max) * 100) : 0;
  $("#ctx-meter .bar span").style.width = `${pct}%`;
  $("#ctx-meter .label").textContent = max ? `${fmtTokens(used)} / ${fmtTokens(max)}` : "";
}

function updateComposerHint() {
  const text = $("#input").value;
  $("#composer-tokens").textContent = text.length > 200 ? `~${fmtTokens(Math.round(text.length / 3.6))} tokens` : "";
}

function setStatus(text) {
  const line = $("#status-line");
  line.hidden = !text;
  line.textContent = text || "";
}

function setGenerating(on) {
  S.generating = on;
  const btn = $("#send");
  btn.textContent = on ? "Stop" : "Send";
  btn.classList.toggle("stop", on);
  btn.classList.toggle("primary", !on);
  if (!on) setStatus("");
}

// ---------------------------------------------------------------------------
// uploads: drop, picker, paste

async function uploadFiles(fileList) {
  const files = [...fileList].filter((f) => f && (f.size > 0 || f.type));
  if (!files.length) return;
  const limit = S.status?.upload_max_bytes || 200 * 1024 * 1024;
  const tooBig = files.filter((f) => f.size > limit);
  if (tooBig.length) toast(`Skipped (too large): ${tooBig.map((f) => f.name).join(", ")}`, 4000);
  const ok = files.filter((f) => f.size <= limit);
  if (!ok.length) return;
  await ensureChat();
  S.uploading += ok.length;
  renderPending();
  try {
    // Upload in small batches so one huge file doesn't hold up the rest.
    for (let i = 0; i < ok.length; i += 4) {
      const form = new FormData();
      ok.slice(i, i + 4).forEach((f) => form.append("files", f, f.name || "pasted.png"));
      const metas = await api(`/api/chats/${S.chatId}/files`, { method: "POST", body: form });
      S.pending.push(...metas);
      S.uploading -= Math.min(4, ok.length - i);
      renderPending();
    }
  } catch (err) {
    toast(`Upload failed: ${err.message}`, 5000);
  } finally {
    S.uploading = 0;
    renderPending();
    reloadChat();
  }
}

async function pasteAsFile(text) {
  await ensureChat();
  S.uploading += 1;
  renderPending();
  try {
    const name = guessPasteName(text);
    const meta = await api(`/api/chats/${S.chatId}/paste`, { json: { text, name } });
    S.pending.push(meta);
    toast(`Pasted ${text.split("\n").length} lines as ${meta.name}`);
  } catch (err) {
    toast(`Paste failed: ${err.message}`, 5000);
  } finally {
    S.uploading = Math.max(0, S.uploading - 1);
    renderPending();
    reloadChat();
  }
}

function guessPasteName(text) {
  const t = text.trimStart();
  const stamp = new Date().toTimeString().slice(0, 8).replace(/:/g, "");
  let ext = "txt";
  if (/^[\[{]/.test(t)) { try { JSON.parse(t); ext = "json"; } catch {} }
  if (ext === "txt") {
    if (/^(diff --git|--- a\/|\+\+\+ b\/|@@ )/m.test(t)) ext = "diff";
    else if (/^\s*(def |class |import |from \S+ import )/m.test(t)) ext = "py";
    else if (/^\s*(function |const |let |import .* from |export )/m.test(t)) ext = "js";
    else if (/^\s*<(!doctype|html|div|\?xml)/i.test(t)) ext = "html";
    else if (/^#{1,6} /m.test(t)) ext = "md";
    else if (/^(\d{4}-\d\d-\d\d[ T]|\[\d|\w+ \d+ \d\d:\d\d:\d\d|(ERROR|WARN|INFO|DEBUG)\b)/m.test(t)) ext = "log";
  }
  return `pasted-${stamp}.${ext}`;
}

$("#input").addEventListener("paste", (e) => {
  const cd = e.clipboardData;
  if (!cd) return;
  const files = [...cd.files];
  if (files.length) {
    e.preventDefault();
    uploadFiles(files.map((f, i) => (f.name && f.name !== "image.png")
      ? f
      : new File([f], `pasted-${Date.now()}-${i}.${(f.type.split("/")[1] || "png")}`, { type: f.type })));
    return;
  }
  const text = cd.getData("text/plain");
  if ($("#paste-as-file").checked && text &&
      (text.length >= PASTE_FILE_CHARS || text.split("\n").length >= PASTE_FILE_LINES)) {
    e.preventDefault();
    pasteAsFile(text);
  }
});

$("#attach").onclick = () => $("#file-input").click();
$("#file-input").addEventListener("change", (e) => {
  uploadFiles(e.target.files);
  e.target.value = "";
});

let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  $("#drop-overlay").hidden = false;
});
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("dragleave", (e) => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("#drop-overlay").hidden = true;
});
window.addEventListener("drop", async (e) => {
  if (!hasFiles(e)) {
    // Dropped plain text (e.g. a selection dragged from another window).
    const text = e.dataTransfer?.getData("text/plain");
    if (text && e.target !== $("#input")) {
      e.preventDefault();
      if (text.length >= PASTE_FILE_CHARS) pasteAsFile(text);
      else insertAtCursor(text);
    }
    return;
  }
  e.preventDefault();
  dragDepth = 0;
  $("#drop-overlay").hidden = true;
  const files = await collectDropped(e.dataTransfer);
  uploadFiles(files);
});

// Folders dropped from a file manager: walk them and upload every file with its relative path.
async function collectDropped(dt) {
  const entries = [...(dt.items || [])].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length || entries.every((en) => en.isFile)) return [...dt.files];
  const out = [];
  const SKIP = new Set(["node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", ".next"]);
  async function walk(entry, prefix) {
    if (out.length >= 500) return;
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      out.push(new File([file], prefix + file.name, { type: file.type }));
    } else if (entry.isDirectory && !SKIP.has(entry.name)) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      } while (batch.length);
    }
  }
  for (const entry of entries) await walk(entry, "");
  if (out.length >= 500) toast("Folder truncated at 500 files — zip it for bigger trees", 4000);
  return out;
}

function insertAtCursor(text) {
  const ta = $("#input");
  const { selectionStart: a, selectionEnd: b, value } = ta;
  ta.value = value.slice(0, a) + text + value.slice(b);
  ta.selectionStart = ta.selectionEnd = a + text.length;
  autosize();
  ta.focus();
}

// ---------------------------------------------------------------------------
// sending + streaming

function autosize() {
  const ta = $("#input");
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, window.innerHeight * 0.4) + "px";
  updateComposerHint();
}
$("#input").addEventListener("input", autosize);
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!S.generating) send();
  }
});
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (S.generating) stopGeneration();
  else send();
});

function stopGeneration() {
  if (S.chatId) api(`/api/chats/${S.chatId}/cancel`, { method: "POST" }).catch(() => {});
  S.abort?.abort();
}

async function send(override) {
  const ta = $("#input");
  const text = (override ?? ta.value).trim();
  if (S.uploading) return toast("Wait for uploads to finish");
  if (!text && !S.pending.length) return;
  if (S.status && !S.status.ready) {
    await refreshStatus();
    if (!S.status.ready) return toast("Model is still loading…");
  }
  const chatId = await ensureChat();
  const attachments = S.pending.map((f) => f.id);
  const attachMeta = S.pending.map((f) => ({ id: f.id, name: f.name, kind: f.kind }));
  S.pending = [];
  if (override === undefined) ta.value = "";
  autosize();
  renderPending();

  // Optimistic render.
  S.chat.messages = S.chat.messages || [];
  const tempUser = { id: "pending", role: "user", content: text, attachments: attachMeta };
  S.chat.messages.push(tempUser);
  renderMessages();
  const box = $("#messages");
  const live = el("div", { class: "msg assistant last" });
  const progress = el("div", { class: "progress" });
  const body = el("div", { class: "body cursor" });
  live.append(progress, body);
  box.append(live);
  box.scrollTop = box.scrollHeight;

  setGenerating(true);
  S.abort = new AbortController();
  let content = "";
  let reasoning = "";
  let reasoningEl = null;
  let finalMessage = null;
  let rafPending = false;

  const paint = () => {
    rafPending = false;
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
    if (reasoningEl) reasoningEl.querySelector(".r-body").textContent = reasoning;
    body.innerHTML = "";
    body.append(renderMarkdown(visibleText(content)));
    if (nearBottom) box.scrollTop = box.scrollHeight;
  };
  const schedule = () => { if (!rafPending) { rafPending = true; requestAnimationFrame(paint); } };

  try {
    const res = await fetch(`/api/chats/${chatId}/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, attachments, thinking: $("#thinking").checked }),
      signal: S.abort.signal,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail; } catch {}
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (!raw.startsWith("data:")) continue;
        const ev = JSON.parse(raw.slice(5));
        switch (ev.type) {
          case "user_saved":
            Object.assign(tempUser, ev.message);
            S.chat.title = ev.title;
            $("#chat-title").textContent = ev.title;
            refreshChatList();
            break;
          case "status":
            setStatus(ev.text);
            if (/^(Reading|Folding|Condensing)/.test(ev.text)) {
              const last = progress.lastElementChild;
              if (last && last.classList.contains("step") && /^Reading/.test(ev.text) && /^Reading/.test(last.textContent)) last.textContent = ev.text;
              else progress.append(el("div", { class: "step" }, ev.text));
            }
            break;
          case "note":
            progress.append(el("details", {}, el("summary", {}, `notes: ${ev.label}`), el("pre", {}, ev.text)));
            break;
          case "memory":
            reloadChat();
            break;
          case "context":
            S.lastContext = ev;
            renderMeter();
            if (ev.iterated?.length) setStatus(`Answering from reading notes on ${ev.iterated.join(", ")}`);
            break;
          case "reasoning":
            if (!reasoningEl) {
              reasoningEl = reasoningBlock("", true);
              live.insertBefore(reasoningEl, body);
            }
            reasoning += ev.text;
            setStatus("Thinking…");
            schedule();
            break;
          case "content":
            if (reasoningEl && reasoningEl.open && !content) reasoningEl.open = false;
            content += ev.text;
            setStatus("Writing…");
            schedule();
            break;
          case "done":
            finalMessage = ev.message;
            break;
          case "error":
            throw new Error(ev.text);
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      body.classList.remove("cursor");
      live.classList.add("error");
      body.append(el("p", {}, `⚠️ ${err.message}`));
    }
  } finally {
    setGenerating(false);
    S.abort = null;
    body.classList.remove("cursor");
    // The server is the source of truth (it strips <remember>, records stats, closes stopped turns).
    await reloadChat({ messages: true }).catch(() => {});
    if (!finalMessage && content) toast("Stopped");
    $("#input").focus();
  }
}

async function regenerate(m) {
  if (S.generating) return;
  const msgs = S.chat.messages;
  const idx = msgs.findIndex((x) => x.id === m.id);
  const user = msgs[idx - 1];
  if (!user || user.role !== "user") return;
  await api(`/api/chats/${S.chatId}/messages/${user.id}`, { method: "DELETE" });
  S.chat.messages = msgs.slice(0, idx - 1);
  S.pending = (user.attachments || []).map((a) => S.chat.files.find((f) => f.id === a.id)).filter(Boolean);
  renderMessages();
  send(user.content);
}

async function editMessage(m) {
  if (S.generating) return;
  await api(`/api/chats/${S.chatId}/messages/${m.id}`, { method: "DELETE" });
  const idx = S.chat.messages.findIndex((x) => x.id === m.id);
  S.chat.messages = S.chat.messages.slice(0, idx);
  S.pending = (m.attachments || []).map((a) => S.chat.files.find((f) => f.id === a.id)).filter(Boolean);
  renderMessages();
  renderPending();
  const ta = $("#input");
  ta.value = m.content;
  autosize();
  ta.focus();
}

// ---------------------------------------------------------------------------
// chrome

$("#new-chat").onclick = () => newChat();
$("#toggle-panel").onclick = () => {
  const app = $("#app");
  if (window.matchMedia("(max-width: 1100px)").matches) app.classList.toggle("show-panel");
  else {
    app.classList.toggle("no-panel");
    store("hidePanel", app.classList.contains("no-panel") ? "1" : "");
  }
};
$("#toggle-sidebar").onclick = () => $("#app").classList.toggle("show-sidebar");
$("#chat-title").onclick = async () => {
  if (!S.chatId) return;
  const title = window.prompt("Rename chat", S.chat.title);
  if (!title) return;
  await api(`/api/chats/${S.chatId}`, { method: "PATCH", json: { title } });
  S.chat.title = title;
  $("#chat-title").textContent = title;
  refreshChatList();
};
$("#thinking").checked = store("thinking") !== "0";
$("#thinking").addEventListener("change", (e) => store("thinking", e.target.checked ? "1" : "0"));
$("#paste-as-file").checked = store("pasteAsFile") !== "0";
$("#paste-as-file").addEventListener("change", (e) => store("pasteAsFile", e.target.checked ? "1" : "0"));
if (store("hidePanel") === "1") $("#app").classList.add("no-panel");

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") {
    e.preventDefault();
    newChat();
  }
  if (e.key === "Escape" && S.generating) stopGeneration();
});

(async function init() {
  await refreshStatus();
  await refreshChatList();
  const last = store("lastChat");
  if (last) await openChat(last);
  else renderAll();
  $("#input").focus();
})();
