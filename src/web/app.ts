interface ConversationSummary {
    id: string;
    title: string;
    updated_at?: string;
    last_message?: string;
    workspace_id?: string;
}

interface WorkspaceSummary {
    id: string;
    display_name: string;
    root_path?: string;
}

interface ChatMessage {
    id?: string;
    role: "user" | "assistant" | "system";
    content: string;
    timestamp?: string;
    error?: boolean;
}

interface WorkspaceItem {
    name: string;
    path: string;
    type: "file" | "directory";
    size?: number | null;
    modified_at?: string;
    content_kind?: string;
    kind?: string;
}

interface WorkspaceFilePayload {
    path: string;
    content: string;
    content_kind: string;
    editable?: boolean;
    default_view?: string;
    entries?: Array<{
        path: string;
        is_dir: boolean;
        size: number;
        compressed_size: number;
    }>;
    media_type?: string;
}

interface ChatEvent {
    type: string;
    content?: string;
    message_id?: string | number;
    conversation_id?: string;
    client_turn_id?: string;
    permission_key?: string;
    approval_target?: string;
    payload?: {
        path?: string;
        open_path?: string;
    };
}

type ConnectionState = "offline" | "online" | "streaming";

const state = {
    appTitle: document.body.dataset.appTitle || "AI Chat",
    ws: null as WebSocket | null,
    reconnectTimer: 0 as number,
    connectionState: "offline" as ConnectionState,
    generating: false,
    modelAvailable: false,
    workspaces: [] as WorkspaceSummary[],
    currentWorkspaceId: localStorage.getItem("lastWorkspaceId") || "",
    conversations: [] as ConversationSummary[],
    currentConversationId: "",
    currentConversationTitle: "New chat",
    messages: [] as ChatMessage[],
    activeAssistantIndex: -1,
    currentDirectoryPath: ".",
    fileItems: [] as WorkspaceItem[],
    selectedFilePath: "",
    leftSidebarOpen: false,
    viewerOpen: false,
};

function query<T extends Element>(selector: string): T {
    const element = document.querySelector<T>(selector);
    if (!element) {
        throw new Error(`Missing required element: ${selector}`);
    }
    return element;
}

const workspaceSelect = query<HTMLSelectElement>("#workspaceSelect");
const sidebarToggle = query<HTMLButtonElement>("#sidebarToggle");
const viewerToggle = query<HTMLButtonElement>("#viewerToggle");
const refreshWorkspaceButton = query<HTMLButtonElement>("#refreshWorkspaceButton");
const newChatButton = query<HTMLButtonElement>("#newChatButton");
const conversationList = query<HTMLDivElement>("#conversationList");
const chatTitle = query<HTMLHeadingElement>("#chatTitle");
const chatMeta = query<HTMLParagraphElement>("#chatMeta");
const connectionBadge = query<HTMLSpanElement>("#connectionBadge");
const modelBadge = query<HTMLSpanElement>("#modelBadge");
const chatMessages = query<HTMLDivElement>("#chatMessages");
const composerForm = query<HTMLFormElement>("#composerForm");
const composerInput = query<HTMLTextAreaElement>("#composerInput");
const composerHint = query<HTMLSpanElement>("#composerHint");
const sendButton = query<HTMLButtonElement>("#sendButton");
const viewerTitle = query<HTMLHeadingElement>("#viewerTitle");
const viewerMeta = query<HTMLParagraphElement>("#viewerMeta");
const upDirectoryButton = query<HTMLButtonElement>("#upDirectoryButton");
const directoryPath = query<HTMLDivElement>("#directoryPath");
const fileList = query<HTMLDivElement>("#fileList");
const filePreview = query<HTMLDivElement>("#filePreview");

function generateId(): string {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function escapeHtml(value: string): string {
    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function formatRelativeTime(value?: string): string {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    const diffMs = date.getTime() - Date.now();
    const minutes = Math.round(diffMs / 60000);
    const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
    if (Math.abs(minutes) < 60) return formatter.format(minutes, "minute");
    const hours = Math.round(minutes / 60);
    if (Math.abs(hours) < 48) return formatter.format(hours, "hour");
    const days = Math.round(hours / 24);
    return formatter.format(days, "day");
}

function formatBytes(value?: number | null): string {
    if (!value) return "";
    const units = ["B", "KB", "MB", "GB"];
    let size = value;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }
    return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function parentDirectory(path: string): string {
    if (!path || path === ".") return ".";
    const parts = path.split("/").filter(Boolean);
    parts.pop();
    return parts.length ? parts.join("/") : ".";
}

function fileViewUrl(path: string): string {
    return `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file/view?path=${encodeURIComponent(path)}`;
}

function currentWorkspaceName(): string {
    return state.workspaces.find((workspace) => workspace.id === state.currentWorkspaceId)?.display_name || "workspace";
}

function syncShellLayout(): void {
    document.body.dataset.leftSidebarOpen = String(state.leftSidebarOpen);
    document.body.dataset.viewerOpen = String(state.viewerOpen && Boolean(state.selectedFilePath));
    sidebarToggle.textContent = state.leftSidebarOpen ? "Close Workspace" : "Workspace";
    viewerToggle.hidden = !state.selectedFilePath;
    viewerToggle.textContent = state.viewerOpen ? "Hide File" : "Selected File";
}

function syncRuntimeSummary(): void {
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

function setConnectionState(nextState: ConnectionState): void {
    state.connectionState = nextState;
    syncRuntimeSummary();
}

function setGenerating(nextValue: boolean): void {
    state.generating = nextValue;
    sendButton.textContent = nextValue ? "Stop" : "Send";
    setConnectionState(nextValue ? "streaming" : (state.ws?.readyState === WebSocket.OPEN ? "online" : "offline"));
}

function setComposerHint(text: string): void {
    composerHint.textContent = text;
}

function truncatePreview(value: string, limit = 88): string {
    const flattened = value.replace(/\s+/g, " ").trim();
    if (flattened.length <= limit) return flattened;
    return `${flattened.slice(0, limit - 1)}…`;
}

function titleFromMessage(content: string): string {
    const firstLine = content.split("\n").map((line) => line.trim()).find(Boolean) || "New chat";
    return firstLine.length > 44 ? `${firstLine.slice(0, 43)}…` : firstLine;
}

function updateChatMeta(): void {
    const workspaceName = currentWorkspaceName();
    if (!state.messages.length) {
        chatMeta.textContent = `Start a new conversation in ${workspaceName}.`;
        return;
    }

    const countLabel = `${state.messages.length} message${state.messages.length === 1 ? "" : "s"}`;
    chatMeta.textContent = `${countLabel} in ${workspaceName}.`;
}

function renderInlineMarkdown(text: string): string {
    return text
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function renderRichText(raw: string): string {
    const codeBlocks: string[] = [];
    const fenced = raw.replace(/```([\s\S]*?)```/g, (_match, code: string) => {
        const token = `@@CODEBLOCK_${codeBlocks.length}@@`;
        codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
        return token;
    });

    const blocks = fenced
        .split(/\n{2,}/)
        .map((block) => block.trim())
        .filter(Boolean)
        .map((block) => {
            if (/^@@CODEBLOCK_\d+@@$/.test(block)) return block;

            const escaped = escapeHtml(block);
            const lines = escaped.split("\n");

            if (lines.every((line) => /^- /.test(line))) {
                return `<ul>${lines.map((line) => `<li>${renderInlineMarkdown(line.replace(/^- /, ""))}</li>`).join("")}</ul>`;
            }

            if (lines.every((line) => /^\d+\. /.test(line))) {
                return `<ol>${lines.map((line) => `<li>${renderInlineMarkdown(line.replace(/^\d+\. /, ""))}</li>`).join("")}</ol>`;
            }

            const heading = lines[0].match(/^(#{1,3})\s+(.*)$/);
            if (heading) {
                const level = Math.min(heading[1].length, 3);
                const headingHtml = `<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`;
                const rest = lines.slice(1).join("<br>");
                return rest ? `${headingHtml}<p>${renderInlineMarkdown(rest)}</p>` : headingHtml;
            }

            return `<p>${renderInlineMarkdown(lines.join("<br>"))}</p>`;
        })
        .join("");

    return blocks.replace(/@@CODEBLOCK_(\d+)@@/g, (_match, index: string) => codeBlocks[Number(index)] || "");
}

function renderMessages(): void {
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

    chatMessages.innerHTML = state.messages.map((message) => {
        const role = message.error ? "assistant error" : message.role;
        const meta = [
            message.role === "user" ? "You" : (message.role === "assistant" ? "Assistant" : "System"),
            formatRelativeTime(message.timestamp),
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

function renderConversations(): void {
    const items = [...state.conversations];
    const hasTransientConversation = state.currentConversationId
        && !items.some((conversation) => conversation.id === state.currentConversationId);

    if (hasTransientConversation) {
        items.unshift({
            id: state.currentConversationId,
            title: state.currentConversationTitle,
            updated_at: new Date().toISOString(),
            last_message: state.messages.at(-1)?.content || "",
            workspace_id: state.currentWorkspaceId,
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

    conversationList.innerHTML = items.map((conversation) => `
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

    conversationList.querySelectorAll<HTMLButtonElement>(".conversation-item").forEach((button) => {
        button.addEventListener("click", () => {
            const id = button.dataset.conversationId || "";
            if (!id) return;
            void loadConversation(id);
        });
    });
}

function renderWorkspaceOptions(): void {
    workspaceSelect.innerHTML = state.workspaces.map((workspace) => `
        <option value="${escapeHtml(workspace.id)}">${escapeHtml(workspace.display_name)}</option>
    `).join("");

    if (state.currentWorkspaceId) {
        workspaceSelect.value = state.currentWorkspaceId;
    }
}

function renderFileList(): void {
    directoryPath.textContent = state.currentDirectoryPath === "." ? "/" : `/${state.currentDirectoryPath}`;
    viewerMeta.textContent = `Browsing ${state.currentDirectoryPath === "." ? "/" : `/${state.currentDirectoryPath}`}`;

    const buttons: string[] = [];
    if (state.currentDirectoryPath !== ".") {
        buttons.push(`
            <button type="button" class="file-item up" data-path="..">
                <div class="file-item-name"><span class="file-item-kind">Up</span>Parent directory</div>
            </button>
        `);
    }

    buttons.push(...state.fileItems.map((item) => `
        <button
            type="button"
            class="file-item${item.path === state.selectedFilePath ? " active" : ""}"
            data-path="${escapeHtml(item.path)}"
            data-kind="${escapeHtml(item.type)}"
        >
            <div class="file-item-name">
                <span class="file-item-kind">${escapeHtml(item.type === "directory" ? "Dir" : (item.content_kind || "File"))}</span>
                ${escapeHtml(item.name)}
            </div>
            <div class="file-item-meta">${escapeHtml([formatBytes(item.size), formatRelativeTime(item.modified_at)].filter(Boolean).join(" • "))}</div>
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

    fileList.querySelectorAll<HTMLButtonElement>(".file-item").forEach((button) => {
        button.addEventListener("click", () => {
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

function renderPreviewEmpty(title: string, body: string): void {
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

function renderPreview(payload: WorkspaceFilePayload): void {
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
        const rows = (payload.content || "")
            .split(/\r?\n/)
            .filter(Boolean)
            .slice(0, 40)
            .map((row) => row.split(",").map((cell) => escapeHtml(cell.trim())));

        if (!rows.length) {
            renderPreviewEmpty("Empty table", "This file does not contain any rows.");
            return;
        }

        const header = rows[0];
        const bodyRows = rows.slice(1);
        filePreview.innerHTML = `
            <table class="preview-table">
                <thead>
                    <tr>${header.map((cell) => `<th>${cell}</th>`).join("")}</tr>
                </thead>
                <tbody>
                    ${bodyRows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}
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

function closeViewer(): void {
    state.viewerOpen = false;
    syncShellLayout();
}

function openViewer(): void {
    if (!state.selectedFilePath) return;
    state.viewerOpen = true;
    syncShellLayout();
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
    const response = await fetch(url, init);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const message = typeof data.detail === "string" ? data.detail : `Request failed: ${response.status}`;
        throw new Error(message);
    }
    return data as T;
}

async function fetchHealth(): Promise<void> {
    try {
        const health = await fetchJson<{
            model_available: boolean;
            model: string;
            message: string;
        }>("/health");
        state.modelAvailable = Boolean(health.model_available);
        modelBadge.textContent = health.model_available
            ? `Model ready: ${health.model}`
            : `Model loading: ${health.model}`;
    } catch (error) {
        state.modelAvailable = false;
        modelBadge.textContent = error instanceof Error ? error.message : "Health check failed";
    }

    syncRuntimeSummary();
}

async function loadWorkspaces(preferredId = ""): Promise<void> {
    const payload = await fetchJson<{
        workspaces: WorkspaceSummary[];
        default_workspace_id?: string | null;
    }>("/api/workspaces");

    state.workspaces = payload.workspaces || [];
    const nextWorkspaceId = preferredId
        || state.currentWorkspaceId
        || payload.default_workspace_id
        || state.workspaces[0]?.id
        || "";

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
}

async function loadConversations(): Promise<void> {
    const payload = await fetchJson<{ conversations: ConversationSummary[] }>("/api/conversations");
    state.conversations = payload.conversations || [];

    const active = state.conversations.find((conversation) => conversation.id === state.currentConversationId);
    if (active) {
        state.currentConversationTitle = active.title || state.currentConversationTitle;
        chatTitle.textContent = state.currentConversationTitle;
    }

    renderConversations();
}

async function loadConversation(id: string): Promise<void> {
    const payload = await fetchJson<{
        messages: ChatMessage[];
        workspace_id?: string;
    }>(`/api/conversation/${encodeURIComponent(id)}`);

    state.currentConversationId = id;
    state.messages = payload.messages || [];
    state.activeAssistantIndex = -1;
    state.selectedFilePath = "";
    state.viewerOpen = false;

    const matchingConversation = state.conversations.find((conversation) => conversation.id === id);
    state.currentConversationTitle = matchingConversation?.title || titleFromMessage(state.messages[0]?.content || "New chat");
    chatTitle.textContent = state.currentConversationTitle;

    if (payload.workspace_id && payload.workspace_id !== state.currentWorkspaceId) {
        await loadWorkspaces(payload.workspace_id);
    } else {
        renderConversations();
    }

    renderMessages();
    syncShellLayout();
}

function startNewChat(): void {
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
}

function ensureAssistantMessage(): ChatMessage {
    const existing = state.messages[state.activeAssistantIndex];
    if (existing && existing.role === "assistant") {
        return existing;
    }

    const assistantMessage: ChatMessage = {
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
    };
    state.messages.push(assistantMessage);
    state.activeAssistantIndex = state.messages.length - 1;
    renderMessages();
    return assistantMessage;
}

function finishGeneration(): void {
    state.activeAssistantIndex = -1;
    setGenerating(false);
    setComposerHint("Enter sends. Shift+Enter adds a line break.");
    void loadConversations();
    void loadDirectory(state.currentDirectoryPath);
}

function pushSystemMessage(content: string, error = false): void {
    state.messages.push({
        role: "system",
        content,
        timestamp: new Date().toISOString(),
        error,
    });
    renderMessages();
}

function handleChatEvent(event: ChatEvent): void {
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
        void openFile(event.payload.open_path, { reveal: true });
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
                approved: true,
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

async function dispatchChatPayload(payload: Record<string, unknown>): Promise<void> {
    const socket = state.ws;
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(payload));
        return;
    }

    const response = await fetchJson<{ events: ChatEvent[] }>("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    (response.events || []).forEach((event) => handleChatEvent(event));
    if (!(response.events || []).some((event) => event.type === "done" || event.type === "error")) {
        finishGeneration();
    }
}

async function sendCurrentMessage(): Promise<void> {
    if (state.generating) {
        state.ws?.send(JSON.stringify({ type: "stop" }));
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
        timestamp: new Date().toISOString(),
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
        auto_approve_tool_permissions: true,
    };

    try {
        await dispatchChatPayload(payload);
    } catch (error) {
        pushSystemMessage(error instanceof Error ? error.message : "Message send failed", true);
        finishGeneration();
    }
}

async function loadDirectory(path: string): Promise<void> {
    if (!state.currentWorkspaceId) {
        renderPreviewEmpty("No workspace", "Create or select a workspace first.");
        return;
    }

    const payload = await fetchJson<{
        path: string;
        items: WorkspaceItem[];
    }>(`/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/files?path=${encodeURIComponent(path)}`);

    state.currentDirectoryPath = payload.path || ".";
    state.fileItems = payload.items || [];
    renderFileList();
}

async function openFile(path: string, options: { reveal?: boolean } = {}): Promise<void> {
    if (!state.currentWorkspaceId) return;

    state.selectedFilePath = path;
    if (options.reveal !== false) {
        state.viewerOpen = true;
    }
    syncShellLayout();
    renderFileList();

    try {
        const payload = await fetchJson<WorkspaceFilePayload>(
            `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file?path=${encodeURIComponent(path)}`
        );
        renderPreview(payload);
    } catch (error) {
        renderPreviewEmpty("Preview unavailable", error instanceof Error ? error.message : "Could not open that file.");
    }
}

function connectWebSocket(): void {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    state.ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
    setConnectionState("offline");

    state.ws.addEventListener("open", () => {
        setConnectionState(state.generating ? "streaming" : "online");
        void fetchHealth();
    });

    state.ws.addEventListener("message", (messageEvent: MessageEvent<string>) => {
        const payload = JSON.parse(messageEvent.data) as ChatEvent;
        if (payload.type === "ping") {
            state.ws?.send(JSON.stringify({ type: "pong" }));
            return;
        }
        handleChatEvent(payload);
    });

    state.ws.addEventListener("close", () => {
        setConnectionState("offline");
        if (state.generating) {
            pushSystemMessage("Streaming connection closed. The current turn stopped.", true);
            finishGeneration();
        }
        window.clearTimeout(state.reconnectTimer);
        state.reconnectTimer = window.setTimeout(() => connectWebSocket(), 2000);
    });

    state.ws.addEventListener("error", () => {
        setConnectionState("offline");
    });
}

function attachEvents(): void {
    newChatButton.addEventListener("click", () => startNewChat());

    refreshWorkspaceButton.addEventListener("click", () => {
        void loadDirectory(state.currentDirectoryPath);
        if (state.selectedFilePath) void openFile(state.selectedFilePath, { reveal: state.viewerOpen });
    });

    sidebarToggle.addEventListener("click", () => {
        state.leftSidebarOpen = !state.leftSidebarOpen;
        syncShellLayout();
    });

    viewerToggle.addEventListener("click", () => {
        if (!state.selectedFilePath) return;
        if (state.viewerOpen) {
            closeViewer();
            return;
        }
        openViewer();
    });

    workspaceSelect.addEventListener("change", async () => {
        const nextWorkspaceId = workspaceSelect.value;
        if (!nextWorkspaceId || nextWorkspaceId === state.currentWorkspaceId) return;
        state.currentWorkspaceId = nextWorkspaceId;
        state.selectedFilePath = "";
        localStorage.setItem("lastWorkspaceId", nextWorkspaceId);
        startNewChat();
        await loadDirectory(".");
        renderPreviewEmpty("Open a file", "Select a file from the workspace to preview it here.");
    });

    upDirectoryButton.addEventListener("click", () => {
        void loadDirectory(parentDirectory(state.currentDirectoryPath));
    });

    composerForm.addEventListener("submit", (event) => {
        event.preventDefault();
        void sendCurrentMessage();
    });

    composerInput.addEventListener("keydown", (event: KeyboardEvent) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void sendCurrentMessage();
        }
    });
}

async function bootstrap(): Promise<void> {
    attachEvents();
    syncShellLayout();
    renderMessages();
    renderConversations();
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
