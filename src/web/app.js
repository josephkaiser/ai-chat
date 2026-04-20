// Generated from src/web/app.ts by scripts/build_frontend.mjs
const state = {
    appTitle: document.body.dataset.appTitle || "AI Chat",
    ws: null,
    reconnectTimer: 0,
    connectionState: "offline",
    generating: false,
    modelAvailable: false,
    workspaces: [],
    currentWorkspaceId: localStorage.getItem("lastWorkspaceId") || "",
    conversations: [],
    currentConversationId: "",
    currentConversationTitle: "New chat",
    messages: [],
    activeAssistantIndex: -1,
    currentDirectoryPath: ".",
    fileItems: [],
    selectedFilePath: "",
    leftSidebarOpen: false,
    viewerOpen: false,
    contextEvalReport: null,
    contextEvalLoading: false,
    contextEvalError: "",
    selectedContextEvalBucketKey: "",
    selectedContextEvalExampleIndex: 0
};
function query(selector) {
    const element = document.querySelector(selector);
    if (!element) {
        throw new Error(`Missing required element: ${selector}`);
    }
    return element;
}
const workspaceSelect = query("#workspaceSelect");
const sidebarToggle = query("#sidebarToggle");
const viewerToggle = query("#viewerToggle");
const refreshWorkspaceButton = query("#refreshWorkspaceButton");
const refreshContextEvalButton = query("#refreshContextEvalButton");
const newChatButton = query("#newChatButton");
const conversationList = query("#conversationList");
const chatTitle = query("#chatTitle");
const chatMeta = query("#chatMeta");
const connectionBadge = query("#connectionBadge");
const modelBadge = query("#modelBadge");
const chatMessages = query("#chatMessages");
const composerForm = query("#composerForm");
const composerInput = query("#composerInput");
const composerHint = query("#composerHint");
const sendButton = query("#sendButton");
const viewerTitle = query("#viewerTitle");
const viewerMeta = query("#viewerMeta");
const upDirectoryButton = query("#upDirectoryButton");
const directoryPath = query("#directoryPath");
const fileList = query("#fileList");
const filePreview = query("#filePreview");
const contextEvalReport = query("#contextEvalReport");
function generateId() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
function escapeHtml(value) {
    return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function formatRelativeTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    const diffMs = date.getTime() - Date.now();
    const minutes = Math.round(diffMs / 60000);
    const formatter = new Intl.RelativeTimeFormat(undefined, {
        numeric: "auto"
    });
    if (Math.abs(minutes) < 60) return formatter.format(minutes, "minute");
    const hours = Math.round(minutes / 60);
    if (Math.abs(hours) < 48) return formatter.format(hours, "hour");
    const days = Math.round(hours / 24);
    return formatter.format(days, "day");
}
function formatBytes(value) {
    if (!value) return "";
    const units = [
        "B",
        "KB",
        "MB",
        "GB"
    ];
    let size = value;
    let unitIndex = 0;
    while(size >= 1024 && unitIndex < units.length - 1){
        size /= 1024;
        unitIndex += 1;
    }
    return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}
function formatDecimal(value, digits = 2) {
    if (typeof value !== "number" || Number.isNaN(value)) return "0";
    return value.toFixed(digits);
}
function workspaceRelativeCapturePath(sourcePath) {
    const normalized = String(sourcePath || "").replace(/\\/g, "/");
    const marker = "/.ai/context-evals/";
    const markerIndex = normalized.lastIndexOf(marker);
    if (markerIndex === -1) return "";
    return `.ai/context-evals/${normalized.slice(markerIndex + marker.length)}`;
}
function parentDirectory(path) {
    if (!path || path === ".") return ".";
    const parts = path.split("/").filter(Boolean);
    parts.pop();
    return parts.length ? parts.join("/") : ".";
}
function fileViewUrl(path) {
    return `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file/view?path=${encodeURIComponent(path)}`;
}
function currentWorkspaceName() {
    return state.workspaces.find((workspace)=>workspace.id === state.currentWorkspaceId)?.display_name || "workspace";
}
function syncShellLayout() {
    document.body.dataset.leftSidebarOpen = String(state.leftSidebarOpen);
    document.body.dataset.viewerOpen = String(state.viewerOpen && Boolean(state.selectedFilePath));
    sidebarToggle.textContent = state.leftSidebarOpen ? "Close Workspace" : "Workspace";
    viewerToggle.hidden = !state.selectedFilePath;
    viewerToggle.textContent = state.viewerOpen ? "Hide File" : "Selected File";
}
function syncRuntimeSummary() {
    if (state.connectionState === "offline") {
        connectionBadge.className = "status-badge offline";
        connectionBadge.textContent = "Offline";
        return;
    }
    if (state.generating) {
        connectionBadge.className = "status-badge streaming";
        connectionBadge.textContent = "Streaming";
        return;
    }
    if (!state.modelAvailable) {
        connectionBadge.className = "status-badge loading";
        connectionBadge.textContent = "Loading";
        return;
    }
    connectionBadge.className = "status-badge online";
    connectionBadge.textContent = "Ready";
}
function setConnectionState(nextState) {
    state.connectionState = nextState;
    syncRuntimeSummary();
}
function setGenerating(nextValue) {
    state.generating = nextValue;
    sendButton.textContent = nextValue ? "Stop" : "Send";
    setConnectionState(nextValue ? "streaming" : state.ws?.readyState === WebSocket.OPEN ? "online" : "offline");
}
function setComposerHint(text) {
    composerHint.textContent = text;
}
function truncatePreview(value, limit = 88) {
    const flattened = value.replace(/\s+/g, " ").trim();
    if (flattened.length <= limit) return flattened;
    return `${flattened.slice(0, limit - 1)}…`;
}
function titleFromMessage(content) {
    const firstLine = content.split("\n").map((line)=>line.trim()).find(Boolean) || "New chat";
    return firstLine.length > 44 ? `${firstLine.slice(0, 43)}…` : firstLine;
}
function updateChatMeta() {
    const workspaceName = currentWorkspaceName();
    if (!state.messages.length) {
        chatMeta.textContent = `Start a new conversation in ${workspaceName}.`;
        return;
    }
    const countLabel = `${state.messages.length} message${state.messages.length === 1 ? "" : "s"}`;
    chatMeta.textContent = `${countLabel} in ${workspaceName}.`;
}
function renderInlineMarkdown(text) {
    return text.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/\*([^*]+)\*/g, "<em>$1</em>").replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}
function renderRichText(raw) {
    const codeBlocks = [];
    const fenced = raw.replace(/```([\s\S]*?)```/g, (_match, code)=>{
        const token = `@@CODEBLOCK_${codeBlocks.length}@@`;
        codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
        return token;
    });
    const blocks = fenced.split(/\n{2,}/).map((block)=>block.trim()).filter(Boolean).map((block)=>{
        if (/^@@CODEBLOCK_\d+@@$/.test(block)) return block;
        const escaped = escapeHtml(block);
        const lines = escaped.split("\n");
        if (lines.every((line)=>/^- /.test(line))) {
            return `<ul>${lines.map((line)=>`<li>${renderInlineMarkdown(line.replace(/^- /, ""))}</li>`).join("")}</ul>`;
        }
        if (lines.every((line)=>/^\d+\. /.test(line))) {
            return `<ol>${lines.map((line)=>`<li>${renderInlineMarkdown(line.replace(/^\d+\. /, ""))}</li>`).join("")}</ol>`;
        }
        const heading = lines[0].match(/^(#{1,3})\s+(.*)$/);
        if (heading) {
            const level = Math.min(heading[1].length, 3);
            const headingHtml = `<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`;
            const rest = lines.slice(1).join("<br>");
            return rest ? `${headingHtml}<p>${renderInlineMarkdown(rest)}</p>` : headingHtml;
        }
        return `<p>${renderInlineMarkdown(lines.join("<br>"))}</p>`;
    }).join("");
    return blocks.replace(/@@CODEBLOCK_(\d+)@@/g, (_match, index)=>codeBlocks[Number(index)] || "");
}
function renderMessages() {
    if (!state.messages.length) {
        chatMessages.innerHTML = `
            <div class="empty-state">
                <div>
                    <div class="empty-state-title">Start a conversation</div>
                    <p>Ask the assistant to inspect code, make a change, or explain something in the current workspace.</p>
                </div>
            </div>
        `;
        updateChatMeta();
        return;
    }
    chatMessages.innerHTML = state.messages.map((message)=>{
        const role = message.error ? "assistant error" : message.role;
        const meta = [
            message.role === "user" ? "You" : message.role === "assistant" ? "Assistant" : "System",
            formatRelativeTime(message.timestamp)
        ].filter(Boolean).join(" • ");
        return `
            <article class="message ${role}">
                <div class="message-role">${escapeHtml(meta)}</div>
                <div class="message-content">${renderRichText(message.content || "")}</div>
            </article>
        `;
    }).join("");
    chatMessages.scrollTop = chatMessages.scrollHeight;
    updateChatMeta();
}
function renderConversations() {
    const items = [
        ...state.conversations
    ];
    const hasTransientConversation = state.currentConversationId && !items.some((conversation)=>conversation.id === state.currentConversationId);
    if (hasTransientConversation) {
        items.unshift({
            id: state.currentConversationId,
            title: state.currentConversationTitle,
            updated_at: new Date().toISOString(),
            last_message: state.messages.at(-1)?.content || "",
            workspace_id: state.currentWorkspaceId
        });
    }
    if (!items.length) {
        conversationList.innerHTML = `
            <div class="empty-state">
                <div>
                    <div class="empty-state-title">No saved chats</div>
                    <p>Use the composer to create the first one.</p>
                </div>
            </div>
        `;
        return;
    }
    conversationList.innerHTML = items.map((conversation)=>`
        <button
            type="button"
            class="conversation-item${conversation.id === state.currentConversationId ? " active" : ""}"
            data-conversation-id="${escapeHtml(conversation.id)}"
        >
            <div class="conversation-title">${escapeHtml(conversation.title || "Untitled chat")}</div>
            <div class="conversation-preview">${escapeHtml(truncatePreview(conversation.last_message || "No messages yet"))}</div>
            <div class="conversation-preview">${escapeHtml(formatRelativeTime(conversation.updated_at) || "Just now")}</div>
        </button>
    `).join("");
    conversationList.querySelectorAll(".conversation-item").forEach((button)=>{
        button.addEventListener("click", ()=>{
            const id = button.dataset.conversationId || "";
            if (!id) return;
            void loadConversation(id);
        });
    });
}
function renderWorkspaceOptions() {
    workspaceSelect.innerHTML = state.workspaces.map((workspace)=>`
        <option value="${escapeHtml(workspace.id)}">${escapeHtml(workspace.display_name)}</option>
    `).join("");
    if (state.currentWorkspaceId) {
        workspaceSelect.value = state.currentWorkspaceId;
    }
}
function renderFileList() {
    directoryPath.textContent = state.currentDirectoryPath === "." ? "/" : `/${state.currentDirectoryPath}`;
    viewerMeta.textContent = `Browsing ${state.currentDirectoryPath === "." ? "/" : `/${state.currentDirectoryPath}`}`;
    const buttons = [];
    if (state.currentDirectoryPath !== ".") {
        buttons.push(`
            <button type="button" class="file-item up" data-path="..">
                <div class="file-item-name"><span class="file-item-kind">Up</span>Parent directory</div>
            </button>
        `);
    }
    buttons.push(...state.fileItems.map((item)=>`
        <button
            type="button"
            class="file-item${item.path === state.selectedFilePath ? " active" : ""}"
            data-path="${escapeHtml(item.path)}"
            data-kind="${escapeHtml(item.type)}"
        >
            <div class="file-item-name">
                <span class="file-item-kind">${escapeHtml(item.type === "directory" ? "Dir" : item.content_kind || "File")}</span>
                ${escapeHtml(item.name)}
            </div>
            <div class="file-item-meta">${escapeHtml([
            formatBytes(item.size),
            formatRelativeTime(item.modified_at)
        ].filter(Boolean).join(" • "))}</div>
        </button>
    `));
    fileList.innerHTML = buttons.join("") || `
        <div class="empty-state">
            <div>
                <div class="empty-state-title">Empty folder</div>
                <p>No visible files are in this directory yet.</p>
            </div>
        </div>
    `;
    fileList.querySelectorAll(".file-item").forEach((button)=>{
        button.addEventListener("click", ()=>{
            const path = button.dataset.path || "";
            if (!path) return;
            if (path === "..") {
                void loadDirectory(parentDirectory(state.currentDirectoryPath));
                return;
            }
            if (button.dataset.kind === "directory") {
                void loadDirectory(path);
                return;
            }
            void openFile(path);
        });
    });
}
function renderContextEvalReport() {
    if (state.contextEvalLoading) {
        contextEvalReport.innerHTML = `<div class="context-eval-empty">Loading replay report…</div>`;
        return;
    }
    if (state.contextEvalError) {
        contextEvalReport.innerHTML = `<div class="context-eval-empty">${escapeHtml(state.contextEvalError)}</div>`;
        return;
    }
    const report = state.contextEvalReport;
    if (!report || !report.total_cases) {
        contextEvalReport.innerHTML = `
            <div class="context-eval-empty">
                No captured replay cases yet. Negative feedback, retries, and corrective follow-ups will appear here.
            </div>
        `;
        return;
    }
    const recommended = report.recommended_fix;
    const topBuckets = report.top_triage_buckets || [];
    const recentFailures = report.recent_failures || [];
    const selectedBucket = topBuckets.find((bucket)=>bucket.key === state.selectedContextEvalBucketKey) || recommended || topBuckets[0] || null;
    const exampleCases = selectedBucket?.example_cases || [];
    const selectedExample = exampleCases[Math.min(Math.max(state.selectedContextEvalExampleIndex, 0), Math.max(exampleCases.length - 1, 0))] || null;
    const selectedCapturePath = selectedExample ? workspaceRelativeCapturePath(selectedExample.source_path) : "";
    contextEvalReport.innerHTML = `
        <div class="context-eval-card">
            <div class="context-eval-label">Current Signal</div>
            <div class="context-eval-stats">
                <div class="context-eval-stat">
                    <strong>${escapeHtml(String(report.total_cases))}</strong>
                    <span>captures</span>
                </div>
                <div class="context-eval-stat">
                    <strong>${escapeHtml(String(report.failed_cases))}</strong>
                    <span>failing</span>
                </div>
                <div class="context-eval-stat">
                    <strong>${escapeHtml(formatDecimal(report.average_score))}</strong>
                    <span>avg score</span>
                </div>
            </div>
        </div>
        ${recommended ? `
            <div class="context-eval-card">
                <div class="context-eval-label">Recommended Next Fix</div>
                <div class="context-eval-focus">
                    <strong>${escapeHtml(recommended.title)}</strong>
                    <span class="context-eval-severity ${escapeHtml(recommended.severity)}">${escapeHtml(recommended.severity)}</span>
                </div>
                <p>${escapeHtml(recommended.recommendation)}</p>
                <div class="context-eval-meta">
                    <span>${escapeHtml(String(recommended.failure_count))} failures</span>
                    <span>${escapeHtml(String(recommended.case_count))} cases</span>
                    <span>score ${escapeHtml(formatDecimal(recommended.priority_score))}</span>
                </div>
            </div>
        ` : ""}
        <div class="context-eval-card">
            <div class="context-eval-label">Top Fixes</div>
            ${topBuckets.length ? `
                <div class="context-eval-list">
                    ${topBuckets.slice(0, 4).map((bucket)=>`
                        <article class="context-eval-item${selectedBucket?.key === bucket.key ? " active" : ""}">
                            <div class="context-eval-focus">
                                <strong>${escapeHtml(bucket.title)}</strong>
                                <span class="context-eval-severity ${escapeHtml(bucket.severity)}">${escapeHtml(bucket.severity)}</span>
                            </div>
                            <div class="context-eval-meta">
                                <span>${escapeHtml(String(bucket.failure_count))} failures</span>
                                <span>${escapeHtml(String(bucket.case_count))} cases</span>
                            </div>
                            <p>${escapeHtml(bucket.recommendation)}</p>
                            <div class="context-eval-actions">
                                <button
                                    type="button"
                                    class="context-eval-button"
                                    data-context-eval-bucket="${escapeHtml(bucket.key)}"
                                >
                                    ${selectedBucket?.key === bucket.key ? "Viewing example" : "Open example"}
                                </button>
                            </div>
                            ${bucket.example_cases[0] ? `
                                <div class="context-eval-example">
                                    <div class="context-eval-label">Example</div>
                                    <div>${escapeHtml(bucket.example_cases[0].name)}</div>
                                    <div class="context-eval-meta">
                                        <span>${escapeHtml(bucket.example_cases[0].phase)}</span>
                                        <span>${escapeHtml(bucket.example_cases[0].trigger)}</span>
                                    </div>
                                </div>
                            ` : ""}
                        </article>
                    `).join("")}
                </div>
            ` : `<div class="context-eval-empty">No triage buckets yet.</div>`}
        </div>
        ${selectedBucket && selectedExample ? `
            <div class="context-eval-card">
                <div class="context-eval-label">Capture Drill-Down</div>
                <div class="context-eval-focus">
                    <strong>${escapeHtml(selectedBucket.title)}</strong>
                    <span class="context-eval-severity ${escapeHtml(selectedBucket.severity)}">${escapeHtml(selectedBucket.severity)}</span>
                </div>
                <div class="context-eval-meta">
                    <span>${escapeHtml(selectedExample.name)}</span>
                    <span>${escapeHtml(selectedExample.phase)}</span>
                    <span>${escapeHtml(selectedExample.trigger)}</span>
                </div>
                ${exampleCases.length > 1 ? `
                    <div class="context-eval-actions">
                        ${exampleCases.map((exampleCase, index)=>`
                            <button
                                type="button"
                                class="context-eval-button${selectedExample === exampleCase ? " active" : ""}"
                                data-context-eval-bucket-example="${escapeHtml(String(index))}"
                            >
                                Example ${escapeHtml(String(index + 1))}
                            </button>
                        `).join("")}
                    </div>
                ` : ""}
                <div class="context-eval-kv">
                    <span class="context-eval-kv-label">Replay file</span>
                    <code>${escapeHtml(selectedCapturePath || selectedExample.source_path)}</code>
                </div>
                <div class="context-eval-actions">
                    <button
                        type="button"
                        class="context-eval-button"
                        data-context-eval-open-capture="true"
                        ${selectedCapturePath ? "" : "disabled"}
                    >
                        Open Capture File
                    </button>
                </div>
                <div class="context-eval-drill-grid">
                    <div class="context-eval-drill-panel">
                        <div class="context-eval-label">Failed Checks</div>
                        ${(selectedExample.failed_checks || []).length ? `
                            <ul class="context-eval-detail-list">
                                ${(selectedExample.failed_checks || []).map((failedCheck)=>`
                                    <li>${escapeHtml(failedCheck)}</li>
                                `).join("")}
                            </ul>
                        ` : `<div class="context-eval-empty">No failed checks recorded.</div>`}
                    </div>
                    <div class="context-eval-drill-panel">
                        <div class="context-eval-label">Selected Keys</div>
                        ${(selectedExample.selected_keys || []).length ? `
                            <ul class="context-eval-detail-list">
                                ${(selectedExample.selected_keys || []).map((selectedKey)=>`
                                    <li><code>${escapeHtml(selectedKey)}</code></li>
                                `).join("")}
                            </ul>
                        ` : `<div class="context-eval-empty">No selected keys recorded.</div>`}
                    </div>
                </div>
            </div>
        ` : ""}
        <div class="context-eval-card">
            <div class="context-eval-label">Recent Failures</div>
            ${recentFailures.length ? `
                <div class="context-eval-list">
                    ${recentFailures.slice(0, 3).map((failure)=>`
                        <article class="context-eval-item">
                            <strong>${escapeHtml(failure.name)}</strong>
                            <div class="context-eval-meta">
                                <span>${escapeHtml(failure.trigger || "unknown")}</span>
                                <span>score ${escapeHtml(formatDecimal(failure.score))}</span>
                            </div>
                            <p>${escapeHtml((failure.failed_checks || []).slice(0, 2).join(" | "))}</p>
                        </article>
                    `).join("")}
                </div>
            ` : `<div class="context-eval-empty">No recent failures.</div>`}
        </div>
    `;
    contextEvalReport.querySelectorAll("[data-context-eval-bucket]").forEach((button)=>{
        button.addEventListener("click", ()=>{
            const bucketKey = button.dataset.contextEvalBucket || "";
            if (!bucketKey) return;
            state.selectedContextEvalBucketKey = bucketKey;
            state.selectedContextEvalExampleIndex = 0;
            renderContextEvalReport();
        });
    });
    contextEvalReport.querySelectorAll("[data-context-eval-bucket-example]").forEach((button)=>{
        button.addEventListener("click", ()=>{
            const rawIndex = button.dataset.contextEvalBucketExample || "0";
            const nextIndex = Number.parseInt(rawIndex, 10);
            state.selectedContextEvalExampleIndex = Number.isFinite(nextIndex) ? Math.max(0, nextIndex) : 0;
            renderContextEvalReport();
        });
    });
    contextEvalReport.querySelectorAll("[data-context-eval-open-capture]").forEach((button)=>{
        button.addEventListener("click", ()=>{
            if (!selectedCapturePath) return;
            void openFile(selectedCapturePath, {
                reveal: true
            });
        });
    });
}
function renderPreviewEmpty(title, body) {
    viewerTitle.textContent = title;
    viewerMeta.textContent = `Browsing ${state.currentDirectoryPath === "." ? "/" : `/${state.currentDirectoryPath}`}`;
    filePreview.innerHTML = `
        <div class="empty-state">
            <div>
                <div class="empty-state-title">${escapeHtml(title)}</div>
                <p>${escapeHtml(body)}</p>
            </div>
        </div>
    `;
}
function renderPreview(payload) {
    const contentKind = payload.content_kind || "text";
    viewerTitle.textContent = payload.path;
    viewerMeta.textContent = `${contentKind.toUpperCase()} preview`;
    if (contentKind === "image") {
        filePreview.innerHTML = `<img class="file-preview-media" alt="${escapeHtml(payload.path)}" src="${fileViewUrl(payload.path)}">`;
        return;
    }
    if (contentKind === "pdf") {
        filePreview.innerHTML = `<iframe class="file-preview-frame" title="${escapeHtml(payload.path)}" src="${fileViewUrl(payload.path)}"></iframe>`;
        return;
    }
    if (contentKind === "html") {
        const srcdoc = escapeHtml(payload.content || "");
        filePreview.innerHTML = `<iframe class="file-preview-frame" sandbox="" title="${escapeHtml(payload.path)}" srcdoc="${srcdoc}"></iframe>`;
        return;
    }
    if (contentKind === "markdown") {
        filePreview.innerHTML = `<div class="preview-markdown">${renderRichText(payload.content || "")}</div>`;
        return;
    }
    if (contentKind === "csv") {
        const rows = (payload.content || "").split(/\r?\n/).filter(Boolean).slice(0, 40).map((row)=>row.split(",").map((cell)=>escapeHtml(cell.trim())));
        if (!rows.length) {
            renderPreviewEmpty("Empty table", "This file does not contain any rows.");
            return;
        }
        const header = rows[0];
        const bodyRows = rows.slice(1);
        filePreview.innerHTML = `
            <table class="preview-table">
                <thead>
                    <tr>${header.map((cell)=>`<th>${cell}</th>`).join("")}</tr>
                </thead>
                <tbody>
                    ${bodyRows.map((row)=>`<tr>${row.map((cell)=>`<td>${cell}</td>`).join("")}</tr>`).join("")}
                </tbody>
            </table>
        `;
        return;
    }
    if (contentKind === "archive") {
        filePreview.innerHTML = `<pre class="archive-list">${escapeHtml(payload.content || "")}</pre>`;
        return;
    }
    filePreview.innerHTML = `<pre class="preview-code">${escapeHtml(payload.content || "")}</pre>`;
}
function closeViewer() {
    state.viewerOpen = false;
    syncShellLayout();
}
function openViewer() {
    if (!state.selectedFilePath) return;
    state.viewerOpen = true;
    syncShellLayout();
}
async function fetchJson(url, init) {
    const response = await fetch(url, init);
    const data = await response.json().catch(()=>({}));
    if (!response.ok) {
        const message = typeof data.detail === "string" ? data.detail : `Request failed: ${response.status}`;
        throw new Error(message);
    }
    return data;
}
async function fetchHealth() {
    try {
        const health = await fetchJson("/health");
        state.modelAvailable = Boolean(health.model_available);
        modelBadge.textContent = health.model_available ? `Model ready: ${health.model}` : `Model loading: ${health.model}`;
    } catch (error) {
        state.modelAvailable = false;
        modelBadge.textContent = error instanceof Error ? error.message : "Health check failed";
    }
    syncRuntimeSummary();
}
async function loadWorkspaces(preferredId = "") {
    const payload = await fetchJson("/api/workspaces");
    state.workspaces = payload.workspaces || [];
    const nextWorkspaceId = preferredId || state.currentWorkspaceId || payload.default_workspace_id || state.workspaces[0]?.id || "";
    state.currentWorkspaceId = nextWorkspaceId;
    if (nextWorkspaceId) {
        localStorage.setItem("lastWorkspaceId", nextWorkspaceId);
    }
    renderWorkspaceOptions();
    updateChatMeta();
    if (nextWorkspaceId) {
        await loadDirectory(".");
    } else {
        state.fileItems = [];
        renderFileList();
    }
    await loadContextEvalReport();
}
async function loadConversations() {
    const payload = await fetchJson("/api/conversations");
    state.conversations = payload.conversations || [];
    const active = state.conversations.find((conversation)=>conversation.id === state.currentConversationId);
    if (active) {
        state.currentConversationTitle = active.title || state.currentConversationTitle;
        chatTitle.textContent = state.currentConversationTitle;
    }
    renderConversations();
}
async function loadContextEvalReport() {
    if (!state.currentWorkspaceId) {
        state.contextEvalReport = null;
        state.contextEvalError = "";
        state.contextEvalLoading = false;
        renderContextEvalReport();
        return;
    }
    state.contextEvalLoading = true;
    state.contextEvalError = "";
    renderContextEvalReport();
    const conversationId = state.conversations.some((conversation)=>conversation.id === state.currentConversationId) ? state.currentConversationId : "";
    const params = new URLSearchParams({
        limit: "100",
        ...conversationId ? {
            conversation_id: conversationId
        } : {
            workspace_id: state.currentWorkspaceId
        }
    });
    try {
        state.contextEvalReport = await fetchJson(`/api/context-evals/report?${params.toString()}`);
        const topBucket = state.contextEvalReport.top_triage_buckets?.[0];
        const selectedStillExists = state.contextEvalReport.top_triage_buckets?.some((bucket)=>bucket.key === state.selectedContextEvalBucketKey);
        state.selectedContextEvalBucketKey = selectedStillExists ? state.selectedContextEvalBucketKey : topBucket?.key || "";
        state.selectedContextEvalExampleIndex = 0;
    } catch (error) {
        state.contextEvalReport = null;
        state.contextEvalError = error instanceof Error ? error.message : "Could not load replay report.";
        state.selectedContextEvalBucketKey = "";
        state.selectedContextEvalExampleIndex = 0;
    } finally{
        state.contextEvalLoading = false;
        renderContextEvalReport();
    }
}
async function loadConversation(id) {
    const payload = await fetchJson(`/api/conversation/${encodeURIComponent(id)}`);
    state.currentConversationId = id;
    state.messages = payload.messages || [];
    state.activeAssistantIndex = -1;
    state.selectedFilePath = "";
    state.viewerOpen = false;
    const matchingConversation = state.conversations.find((conversation)=>conversation.id === id);
    state.currentConversationTitle = matchingConversation?.title || titleFromMessage(state.messages[0]?.content || "New chat");
    chatTitle.textContent = state.currentConversationTitle;
    if (payload.workspace_id && payload.workspace_id !== state.currentWorkspaceId) {
        await loadWorkspaces(payload.workspace_id);
    } else {
        renderConversations();
    }
    renderMessages();
    syncShellLayout();
    await loadContextEvalReport();
}
function startNewChat() {
    state.currentConversationId = generateId();
    state.currentConversationTitle = "New chat";
    state.messages = [];
    state.activeAssistantIndex = -1;
    state.selectedFilePath = "";
    state.viewerOpen = false;
    chatTitle.textContent = state.currentConversationTitle;
    renderMessages();
    renderConversations();
    syncShellLayout();
    void loadContextEvalReport();
}
function ensureAssistantMessage() {
    const existing = state.messages[state.activeAssistantIndex];
    if (existing && existing.role === "assistant") {
        return existing;
    }
    const assistantMessage = {
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString()
    };
    state.messages.push(assistantMessage);
    state.activeAssistantIndex = state.messages.length - 1;
    renderMessages();
    return assistantMessage;
}
function finishGeneration() {
    state.activeAssistantIndex = -1;
    setGenerating(false);
    setComposerHint("Enter sends. Shift+Enter adds a line break.");
    void loadConversations();
    void loadDirectory(state.currentDirectoryPath);
    void loadContextEvalReport();
}
function pushSystemMessage(content, error = false) {
    state.messages.push({
        role: "system",
        content,
        timestamp: new Date().toISOString(),
        error
    });
    renderMessages();
}
function handleChatEvent(event) {
    if (!event || typeof event !== "object") return;
    if (event.type === "start") {
        ensureAssistantMessage();
        setGenerating(true);
        setComposerHint("Assistant is working...");
        return;
    }
    if (event.type === "assistant_note" || event.type === "final_replace") {
        const assistantMessage = ensureAssistantMessage();
        assistantMessage.content = event.content || "";
        renderMessages();
        return;
    }
    if (event.type === "token") {
        const assistantMessage = ensureAssistantMessage();
        assistantMessage.content += event.content || "";
        renderMessages();
        return;
    }
    if (event.type === "activity" && event.content) {
        setComposerHint(event.content);
        return;
    }
    if (event.type === "tool_result" && event.payload?.open_path) {
        void openFile(event.payload.open_path, {
            reveal: true
        });
        return;
    }
    if (event.type === "permission_required") {
        const socket = state.ws;
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: "permission_response",
                conversation_id: event.conversation_id || state.currentConversationId,
                client_turn_id: event.client_turn_id || "",
                permission_key: event.permission_key || "",
                approval_target: event.approval_target || "tool",
                approved: true
            }));
            setComposerHint("Auto-approved workspace access. Continuing...");
        } else {
            setComposerHint(event.content || "Permission is required to continue.");
        }
        return;
    }
    if (event.type === "error") {
        const assistantMessage = ensureAssistantMessage();
        assistantMessage.content = event.content || "The assistant hit an error.";
        assistantMessage.error = true;
        renderMessages();
        finishGeneration();
        return;
    }
    if (event.type === "canceled" || event.type === "done") {
        finishGeneration();
    }
}
async function dispatchChatPayload(payload) {
    const socket = state.ws;
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(payload));
        return;
    }
    const response = await fetchJson("/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    });
    (response.events || []).forEach((event)=>handleChatEvent(event));
    if (!(response.events || []).some((event)=>event.type === "done" || event.type === "error")) {
        finishGeneration();
    }
}
async function sendCurrentMessage() {
    if (state.generating) {
        state.ws?.send(JSON.stringify({
            type: "stop"
        }));
        return;
    }
    const message = composerInput.value.trim();
    if (!message) return;
    if (!state.currentConversationId) {
        startNewChat();
    }
    state.messages.push({
        role: "user",
        content: message,
        timestamp: new Date().toISOString()
    });
    if (state.currentConversationTitle === "New chat") {
        state.currentConversationTitle = titleFromMessage(message);
        chatTitle.textContent = state.currentConversationTitle;
    }
    composerInput.value = "";
    renderMessages();
    renderConversations();
    setGenerating(true);
    setComposerHint(state.modelAvailable ? "Waiting for the assistant..." : "Model may still be loading. Sending anyway...");
    const payload = {
        message,
        conversation_id: state.currentConversationId,
        workspace_id: state.currentWorkspaceId || null,
        mode: "deep",
        turn_kind: "visible_chat",
        auto_approve_tool_permissions: true
    };
    try {
        await dispatchChatPayload(payload);
    } catch (error) {
        pushSystemMessage(error instanceof Error ? error.message : "Message send failed", true);
        finishGeneration();
    }
}
async function loadDirectory(path) {
    if (!state.currentWorkspaceId) {
        renderPreviewEmpty("No workspace", "Create or select a workspace first.");
        return;
    }
    const payload = await fetchJson(`/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/files?path=${encodeURIComponent(path)}`);
    state.currentDirectoryPath = payload.path || ".";
    state.fileItems = payload.items || [];
    renderFileList();
}
async function openFile(path, options = {}) {
    if (!state.currentWorkspaceId) return;
    state.selectedFilePath = path;
    if (options.reveal !== false) {
        state.viewerOpen = true;
    }
    syncShellLayout();
    renderFileList();
    try {
        const payload = await fetchJson(`/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file?path=${encodeURIComponent(path)}`);
        renderPreview(payload);
    } catch (error) {
        renderPreviewEmpty("Preview unavailable", error instanceof Error ? error.message : "Could not open that file.");
    }
}
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    state.ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
    setConnectionState("offline");
    state.ws.addEventListener("open", ()=>{
        setConnectionState(state.generating ? "streaming" : "online");
        void fetchHealth();
    });
    state.ws.addEventListener("message", (messageEvent)=>{
        const payload = JSON.parse(messageEvent.data);
        if (payload.type === "ping") {
            state.ws?.send(JSON.stringify({
                type: "pong"
            }));
            return;
        }
        handleChatEvent(payload);
    });
    state.ws.addEventListener("close", ()=>{
        setConnectionState("offline");
        if (state.generating) {
            pushSystemMessage("Streaming connection closed. The current turn stopped.", true);
            finishGeneration();
        }
        window.clearTimeout(state.reconnectTimer);
        state.reconnectTimer = window.setTimeout(()=>connectWebSocket(), 2000);
    });
    state.ws.addEventListener("error", ()=>{
        setConnectionState("offline");
    });
}
function attachEvents() {
    newChatButton.addEventListener("click", ()=>startNewChat());
    refreshContextEvalButton.addEventListener("click", ()=>{
        void loadContextEvalReport();
    });
    refreshWorkspaceButton.addEventListener("click", ()=>{
        void loadDirectory(state.currentDirectoryPath);
        if (state.selectedFilePath) void openFile(state.selectedFilePath, {
            reveal: state.viewerOpen
        });
    });
    sidebarToggle.addEventListener("click", ()=>{
        state.leftSidebarOpen = !state.leftSidebarOpen;
        syncShellLayout();
    });
    viewerToggle.addEventListener("click", ()=>{
        if (!state.selectedFilePath) return;
        if (state.viewerOpen) {
            closeViewer();
            return;
        }
        openViewer();
    });
    workspaceSelect.addEventListener("change", async ()=>{
        const nextWorkspaceId = workspaceSelect.value;
        if (!nextWorkspaceId || nextWorkspaceId === state.currentWorkspaceId) return;
        state.currentWorkspaceId = nextWorkspaceId;
        state.selectedFilePath = "";
        localStorage.setItem("lastWorkspaceId", nextWorkspaceId);
        startNewChat();
        await loadDirectory(".");
        renderPreviewEmpty("Open a file", "Select a file from the workspace to preview it here.");
    });
    upDirectoryButton.addEventListener("click", ()=>{
        void loadDirectory(parentDirectory(state.currentDirectoryPath));
    });
    composerForm.addEventListener("submit", (event)=>{
        event.preventDefault();
        void sendCurrentMessage();
    });
    composerInput.addEventListener("keydown", (event)=>{
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void sendCurrentMessage();
        }
    });
}
async function bootstrap() {
    attachEvents();
    syncShellLayout();
    renderMessages();
    renderConversations();
    renderContextEvalReport();
    renderPreviewEmpty("Open a file", "Select a file from the workspace to preview it here.");
    connectWebSocket();
    await fetchHealth();
    await loadWorkspaces();
    await loadConversations();
    if (state.conversations.length) {
        await loadConversation(state.conversations[0].id);
    } else {
        startNewChat();
    }
}
void bootstrap();


//# sourceURL=/Users/joe/dev/ai-chat/src/web/app.ts