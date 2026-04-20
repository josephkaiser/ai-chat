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

interface ContextEvalRecentFailure {
    source_path: string;
    name: string;
    score: number;
    trigger: string;
    failed_checks: string[];
    selected_keys: string[];
}

interface ContextEvalExampleCase {
    name: string;
    source_path: string;
    trigger: string;
    phase: string;
    failed_checks: string[];
    selected_keys: string[];
}

interface ContextEvalTriageBucket {
    key: string;
    category: string;
    title: string;
    severity: "high" | "medium" | "low" | string;
    recommendation: string;
    failure_count: number;
    case_count: number;
    priority_score: number;
    trigger_counts: Record<string, number>;
    phase_counts: Record<string, number>;
    sample_failed_checks: string[];
    example_cases: ContextEvalExampleCase[];
}

interface ContextEvalReport {
    total_cases: number;
    passed_cases: number;
    failed_cases: number;
    average_score: number;
    failed_check_counts: Record<string, number>;
    trigger_counts: Record<string, number>;
    phase_counts: Record<string, number>;
    recent_failures: ContextEvalRecentFailure[];
    triage_bucket_count: number;
    top_triage_buckets: ContextEvalTriageBucket[];
    recommended_fix?: ContextEvalTriageBucket | null;
}

interface ContextEvalPromotionResponse {
    status: string;
    fixture_name: string;
    fixture_path: string;
    source_path: string;
    review_status: string;
}

type ConnectionState = "offline" | "online" | "streaming";

const state = {
    appTitle: document.body.dataset.appTitle || "AI Chat",
    ws: null as WebSocket | null,
    reconnectTimer: 0 as number,
    connectionState: "offline" as ConnectionState,
    generating: false,
    modelAvailable: false,
    modelName: "",
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
    contextEvalReport: null as ContextEvalReport | null,
    contextEvalLoading: false,
    contextEvalError: "",
    selectedContextEvalBucketKey: "",
    selectedContextEvalExampleIndex: 0,
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
const workspaceSettingsButton = query<HTMLButtonElement>("#workspaceSettingsButton");
const refreshContextEvalButton = query<HTMLButtonElement>("#refreshContextEvalButton");
const newChatButton = query<HTMLButtonElement>("#newChatButton");
const conversationList = query<HTMLDivElement>("#conversationList");
const connectionBadge = query<HTMLSpanElement>("#connectionBadge");
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
const contextEvalReport = query<HTMLDivElement>("#contextEvalReport");
const settingsOverlay = query<HTMLDivElement>("#settingsOverlay");
const closeSettingsButton = query<HTMLButtonElement>("#closeSettingsButton");
const settingsSummary = query<HTMLDivElement>("#settingsSummary");
const resetAppButton = query<HTMLButtonElement>("#resetAppButton");

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

function formatDecimal(value?: number | null, digits = 2): string {
    if (typeof value !== "number" || Number.isNaN(value)) return "0";
    return value.toFixed(digits);
}

function workspaceRelativeCapturePath(sourcePath: string): string {
    const normalized = String(sourcePath || "").replace(/\\/g, "/");
    const marker = "/.ai/context-evals/";
    const markerIndex = normalized.lastIndexOf(marker);
    if (markerIndex === -1) return "";
    return `.ai/context-evals/${normalized.slice(markerIndex + marker.length)}`;
}

function suggestContextEvalFixtureName(bucket: ContextEvalTriageBucket, exampleCase: ContextEvalExampleCase): string {
    const segments = [
        exampleCase.phase || "capture",
        bucket.category || "context_eval",
        bucket.key || exampleCase.name || "case",
    ];
    return segments
        .join("_")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "")
        || "captured_context_eval_case";
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

function defaultComposerStatus(): string {
    if (state.connectionState === "offline") {
        return "Runtime offline.";
    }
    if (!state.modelAvailable) {
        return state.modelName ? `Model loading: ${state.modelName}` : "Model loading…";
    }
    return state.modelName ? `Model ready: ${state.modelName}` : "Model ready.";
}

function syncShellLayout(): void {
    document.body.dataset.leftSidebarOpen = String(state.leftSidebarOpen);
    document.body.dataset.viewerOpen = String(state.viewerOpen && Boolean(state.selectedFilePath));
    const sidebarToggleLabel = state.leftSidebarOpen ? "Close workspace" : "Open workspace";
    sidebarToggle.setAttribute("aria-label", sidebarToggleLabel);
    sidebarToggle.setAttribute("title", sidebarToggleLabel);
    const viewerToggleLabel = state.viewerOpen ? "Hide selected file" : "Open selected file";
    viewerToggle.setAttribute("aria-label", viewerToggleLabel);
    viewerToggle.setAttribute("title", viewerToggleLabel);
    viewerToggle.hidden = !state.selectedFilePath;
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

function renderInlineMarkdown(text: string): string {
    return text
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function normalizeCodeLanguage(language?: string): string {
    const normalized = String(language || "").trim().toLowerCase();
    if (!normalized) return "text";
    if (["py", "python"].includes(normalized)) return "python";
    if (["js", "jsx", "javascript", "ts", "tsx", "typescript"].includes(normalized)) return "javascript";
    if (["sh", "shell", "bash", "zsh"].includes(normalized)) return "bash";
    if (["c", "h", "cpp", "cxx", "hpp"].includes(normalized)) return "c";
    if (["json"].includes(normalized)) return "json";
    return normalized;
}

function keywordPatternForLanguage(language: string): RegExp | null {
    const patterns: Record<string, string> = {
        python: String.raw`\b(?:and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield)\b`,
        javascript: String.raw`\b(?:async|await|break|case|catch|class|const|continue|default|delete|else|export|extends|false|finally|for|from|function|if|import|in|instanceof|let|new|null|of|return|switch|throw|true|try|typeof|var|while|yield)\b`,
        bash: String.raw`\b(?:if|then|else|elif|fi|for|do|done|case|esac|while|in|function|return|local|export)\b`,
        c: String.raw`\b(?:auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|if|inline|int|long|register|restrict|return|short|signed|sizeof|static|struct|switch|typedef|union|unsigned|void|volatile|while)\b`,
        json: String.raw`\b(?:true|false|null)\b`,
    };
    const pattern = patterns[language];
    return pattern ? new RegExp(pattern, "g") : null;
}

function commentPatternForLanguage(language: string): RegExp | null {
    if (language === "python" || language === "bash") return /#.*$/gm;
    if (language === "javascript" || language === "c") return /\/\/.*$|\/\*[\s\S]*?\*\//gm;
    return null;
}

function highlightCode(rawCode: string, language?: string): string {
    const normalizedLanguage = normalizeCodeLanguage(language);
    const placeholders: string[] = [];
    const stash = (text: string, pattern: RegExp, className: string): string => {
        return text.replace(pattern, (match) => {
            const token = `@@CODETOKEN_${placeholders.length}@@`;
            placeholders.push(`<span class="${className}">${escapeHtml(match)}</span>`);
            return token;
        });
    };

    let highlighted = rawCode.replace(/\r\n?/g, "\n");
    highlighted = stash(highlighted, /"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, "tok-string");

    const commentPattern = commentPatternForLanguage(normalizedLanguage);
    if (commentPattern) {
        highlighted = stash(highlighted, commentPattern, "tok-comment");
    }

    highlighted = escapeHtml(highlighted);

    const keywordPattern = keywordPatternForLanguage(normalizedLanguage);
    if (keywordPattern) {
        highlighted = highlighted.replace(keywordPattern, '<span class="tok-keyword">$&</span>');
    }

    highlighted = highlighted.replace(/\b([A-Za-z_][A-Za-z0-9_]*)(?=\s*\()/g, '<span class="tok-function">$1</span>');
    highlighted = highlighted.replace(/\b(0x[0-9a-fA-F]+|\d+(?:\.\d+)?)\b/g, '<span class="tok-number">$1</span>');

    return highlighted.replace(/@@CODETOKEN_(\d+)@@/g, (_match, index: string) => placeholders[Number(index)] || "");
}

function renderRichText(raw: string): string {
    const codeBlocks: string[] = [];
    const fenced = raw.replace(/```([^\n`]*)\n?([\s\S]*?)```/g, (_match, language: string, code: string) => {
        const token = `@@CODEBLOCK_${codeBlocks.length}@@`;
        const normalizedLanguage = normalizeCodeLanguage(language);
        const displayLanguage = normalizedLanguage === "text" ? "text" : normalizedLanguage;
        const cleanCode = code.replace(/^\n+|\n+$/g, "");
        codeBlocks.push(`
            <div class="message-code-block">
                <div class="message-code-header">${escapeHtml(displayLanguage)}</div>
                <pre><code class="language-${escapeHtml(normalizedLanguage)}">${highlightCode(cleanCode, normalizedLanguage)}</code></pre>
            </div>
        `.trim());
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
        <div class="conversation-row${conversation.id === state.currentConversationId ? " active" : ""}">
            <button
                type="button"
                class="conversation-item${conversation.id === state.currentConversationId ? " active" : ""}"
                data-conversation-id="${escapeHtml(conversation.id)}"
            >
                <div class="conversation-title">${escapeHtml(conversation.title || "Untitled chat")}</div>
                <div class="conversation-preview">${escapeHtml(truncatePreview(conversation.last_message || "No messages yet"))}</div>
                <div class="conversation-preview">${escapeHtml(formatRelativeTime(conversation.updated_at) || "Just now")}</div>
            </button>
            <div class="conversation-actions">
                <button type="button" class="conversation-action-button" data-action="rename" data-conversation-id="${escapeHtml(conversation.id)}" data-title="${escapeHtml(conversation.title || "Untitled chat")}">Rename</button>
                <button type="button" class="conversation-action-button conversation-action-danger" data-action="delete" data-conversation-id="${escapeHtml(conversation.id)}">Delete</button>
            </div>
        </div>
    `).join("");

    conversationList.querySelectorAll<HTMLButtonElement>(".conversation-item").forEach((button) => {
        button.addEventListener("click", () => {
            const id = button.dataset.conversationId || "";
            if (!id) return;
            void loadConversation(id);
        });
    });

    conversationList.querySelectorAll<HTMLButtonElement>(".conversation-action-button").forEach((button) => {
        button.addEventListener("click", (event) => {
            event.stopPropagation();
            const id = button.dataset.conversationId || "";
            if (!id) return;
            if (button.dataset.action === "rename") {
                void renameConversation(id, button.dataset.title || "");
                return;
            }
            if (button.dataset.action === "delete") {
                void deleteConversation(id);
            }
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

function renderContextEvalReport(): void {
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
    const selectedBucket = topBuckets.find((bucket) => bucket.key === state.selectedContextEvalBucketKey)
        || recommended
        || topBuckets[0]
        || null;
    const exampleCases = selectedBucket?.example_cases || [];
    const selectedExample = exampleCases[
        Math.min(Math.max(state.selectedContextEvalExampleIndex, 0), Math.max(exampleCases.length - 1, 0))
    ] || null;
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
                    ${topBuckets.slice(0, 4).map((bucket) => `
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
                        ${exampleCases.map((exampleCase, index) => `
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
                    <button
                        type="button"
                        class="context-eval-button"
                        data-context-eval-promote-capture="true"
                    >
                        Promote to Fixture
                    </button>
                </div>
                <div class="context-eval-drill-grid">
                    <div class="context-eval-drill-panel">
                        <div class="context-eval-label">Failed Checks</div>
                        ${(selectedExample.failed_checks || []).length ? `
                            <ul class="context-eval-detail-list">
                                ${(selectedExample.failed_checks || []).map((failedCheck) => `
                                    <li>${escapeHtml(failedCheck)}</li>
                                `).join("")}
                            </ul>
                        ` : `<div class="context-eval-empty">No failed checks recorded.</div>`}
                    </div>
                    <div class="context-eval-drill-panel">
                        <div class="context-eval-label">Selected Keys</div>
                        ${(selectedExample.selected_keys || []).length ? `
                            <ul class="context-eval-detail-list">
                                ${(selectedExample.selected_keys || []).map((selectedKey) => `
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
                    ${recentFailures.slice(0, 3).map((failure) => `
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

    contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-bucket]").forEach((button) => {
        button.addEventListener("click", () => {
            const bucketKey = button.dataset.contextEvalBucket || "";
            if (!bucketKey) return;
            state.selectedContextEvalBucketKey = bucketKey;
            state.selectedContextEvalExampleIndex = 0;
            renderContextEvalReport();
        });
    });
    contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-bucket-example]").forEach((button) => {
        button.addEventListener("click", () => {
            const rawIndex = button.dataset.contextEvalBucketExample || "0";
            const nextIndex = Number.parseInt(rawIndex, 10);
            state.selectedContextEvalExampleIndex = Number.isFinite(nextIndex) ? Math.max(0, nextIndex) : 0;
            renderContextEvalReport();
        });
    });
    contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-open-capture]").forEach((button) => {
        button.addEventListener("click", () => {
            if (!selectedCapturePath) return;
            void openFile(selectedCapturePath, { reveal: true });
        });
    });
    contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-promote-capture]").forEach((button) => {
        button.addEventListener("click", async () => {
            if (!selectedExample?.source_path) return;
            const suggestedName = selectedBucket
                ? suggestContextEvalFixtureName(selectedBucket, selectedExample)
                : selectedExample.name;
            const fixtureName = window.prompt("Fixture name", suggestedName);
            if (!fixtureName || !fixtureName.trim()) return;
            const reviewStatus = window.prompt("Review status (candidate, accepted, superseded)", "candidate");
            if (!reviewStatus || !reviewStatus.trim()) return;
            try {
                const response = await fetchJson<ContextEvalPromotionResponse>("/api/context-evals/promote", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        source_path: selectedExample.source_path,
                        fixture_name: fixtureName.trim(),
                        review_status: reviewStatus.trim().toLowerCase(),
                    }),
                });
                setComposerHint(`Promoted replay case as ${response.fixture_name} (${response.review_status}).`);
            } catch (error) {
                setComposerHint(error instanceof Error ? error.message : "Could not promote replay case.");
            }
        });
    });
}

function renderSettingsSummary(): void {
    settingsSummary.innerHTML = `
        <div><strong>Workspace:</strong> ${escapeHtml(currentWorkspaceName())}</div>
        <div><strong>Conversation:</strong> ${escapeHtml(state.currentConversationTitle || "New chat")}</div>
        <div><strong>Runtime:</strong> ${escapeHtml(defaultComposerStatus())}</div>
    `;
}

function showSettings(): void {
    renderSettingsSummary();
    settingsOverlay.hidden = false;
}

function closeSettings(): void {
    settingsOverlay.hidden = true;
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
        state.modelName = String(health.model || "").trim();
    } catch (error) {
        state.modelAvailable = false;
        state.modelName = "";
        setComposerHint(error instanceof Error ? error.message : "Health check failed");
    }

    syncRuntimeSummary();
    if (!state.generating) {
        setComposerHint(defaultComposerStatus());
    }
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
    if (nextWorkspaceId) {
        await loadDirectory(".");
    } else {
        state.fileItems = [];
        renderFileList();
    }
    await loadContextEvalReport();
}

async function loadConversations(): Promise<void> {
    const payload = await fetchJson<{ conversations: ConversationSummary[] }>("/api/conversations");
    state.conversations = payload.conversations || [];

    const active = state.conversations.find((conversation) => conversation.id === state.currentConversationId);
    if (active) {
        state.currentConversationTitle = active.title || state.currentConversationTitle;
    }

    renderConversations();
}

async function loadContextEvalReport(): Promise<void> {
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

    const conversationId = state.conversations.some((conversation) => conversation.id === state.currentConversationId)
        ? state.currentConversationId
        : "";
    const params = new URLSearchParams({
        limit: "100",
        ...(conversationId ? { conversation_id: conversationId } : { workspace_id: state.currentWorkspaceId }),
    });

    try {
        state.contextEvalReport = await fetchJson<ContextEvalReport>(`/api/context-evals/report?${params.toString()}`);
        const topBucket = state.contextEvalReport.top_triage_buckets?.[0];
        const selectedStillExists = state.contextEvalReport.top_triage_buckets?.some(
            (bucket) => bucket.key === state.selectedContextEvalBucketKey
        );
        state.selectedContextEvalBucketKey = selectedStillExists
            ? state.selectedContextEvalBucketKey
            : (topBucket?.key || "");
        state.selectedContextEvalExampleIndex = 0;
    } catch (error) {
        state.contextEvalReport = null;
        state.contextEvalError = error instanceof Error ? error.message : "Could not load replay report.";
        state.selectedContextEvalBucketKey = "";
        state.selectedContextEvalExampleIndex = 0;
    } finally {
        state.contextEvalLoading = false;
        renderContextEvalReport();
    }
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

    if (payload.workspace_id && payload.workspace_id !== state.currentWorkspaceId) {
        await loadWorkspaces(payload.workspace_id);
    } else {
        renderConversations();
    }

    renderMessages();
    syncShellLayout();
    await loadContextEvalReport();
}

function startNewChat(): void {
    state.currentConversationId = generateId();
    state.currentConversationTitle = "New chat";
    state.messages = [];
    state.activeAssistantIndex = -1;
    state.selectedFilePath = "";
    state.viewerOpen = false;
    renderMessages();
    renderConversations();
    syncShellLayout();
    void loadContextEvalReport();
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
    setComposerHint(defaultComposerStatus());
    void loadConversations();
    void loadDirectory(state.currentDirectoryPath);
    void loadContextEvalReport();
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

async function renameConversation(conversationId: string, currentTitle: string): Promise<void> {
    const nextTitle = String(prompt("Rename chat", currentTitle) || "").trim();
    if (!nextTitle) return;
    await fetchJson(`/api/conversation/${encodeURIComponent(conversationId)}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: nextTitle }),
    });
    if (conversationId === state.currentConversationId) {
        state.currentConversationTitle = nextTitle;
    }
    await loadConversations();
}

async function deleteConversation(conversationId: string): Promise<void> {
    if (!confirm("Delete this chat?")) return;
    await fetchJson(`/api/conversation/${encodeURIComponent(conversationId)}`, {
        method: "DELETE",
    });
    if (conversationId === state.currentConversationId) {
        startNewChat();
    }
    await loadConversations();
}

async function resetAppData(): Promise<void> {
    if (!confirm("Reset all app data? This deletes chats, workspaces, and related runtime state.")) return;
    await fetchJson("/api/reset-all", { method: "POST" });
    closeSettings();
    state.currentConversationId = "";
    state.currentConversationTitle = "New chat";
    state.messages = [];
    state.selectedFilePath = "";
    state.viewerOpen = false;
    renderMessages();
    renderPreviewEmpty("Open a file", "Select a file from the workspace to preview it here.");
    await loadWorkspaces();
    await loadConversations();
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
    workspaceSettingsButton.addEventListener("click", () => showSettings());
    closeSettingsButton.addEventListener("click", () => closeSettings());
    resetAppButton.addEventListener("click", () => {
        void resetAppData();
    });
    settingsOverlay.addEventListener("click", (event) => {
        if (event.target === settingsOverlay) closeSettings();
    });
    refreshContextEvalButton.addEventListener("click", () => {
        void loadContextEvalReport();
    });

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
