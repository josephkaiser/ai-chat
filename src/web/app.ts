import {
    chatMessages,
    closeSettingsButton,
    composerAttachments,
    composerForm,
    composerHint,
    composerInput,
    connectionBadge,
    conversationList,
    directoryPath,
    downloadFileButton,
    fileList,
    filePreview,
    fixtureReviewFilter,
    isMobileViewport,
    newChatButton,
    openInTabButton,
    refreshContextEvalButton,
    refreshFixtureReviewButton,
    refreshWorkspaceButton,
    resetAppButton,
    scrollToBottomButton,
    sendButton,
    settingsOverlay,
    settingsSummary,
    sidebarToggle,
    viewerCloseButton,
    viewerMeta,
    viewerModeButton,
    viewerTitle,
    viewerToggle,
    workspaceSettingsButton,
} from "./dom.js";
import { createContextEvalController } from "./context_eval.js";
import {
    FIRST_TURN_COMPOSER_MAX_HEIGHT,
    LATER_TURN_COMPOSER_MAX_HEIGHT,
    PASTE_ATTACH_CHAR_THRESHOLD,
    PASTE_ATTACH_LINE_THRESHOLD,
    state,
} from "./state.js";
import type {
    ChatEvent,
    ChatMessage,
    ConversationSummary,
    KatexNamespace,
    PendingComposerAttachment,
    SpreadsheetPreviewPayload,
    ThinkingStatus,
    WorkspaceFilePayload,
    WorkspaceItem,
    WorkspaceUploadResponse,
    WorkspaceSummary,
} from "./types.js";

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

function getKatexRenderer(): KatexNamespace | null {
    const candidate = (globalThis as { katex?: KatexNamespace }).katex;
    if (candidate && typeof candidate.renderToString === "function") {
        return candidate;
    }
    return null;
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

function formatElapsedDuration(startValue?: string, endMs = Date.now()): string {
    if (!startValue) return "";
    const startMs = new Date(startValue).getTime();
    if (Number.isNaN(startMs)) return "";
    const diffMs = Math.max(0, endMs - startMs);
    const totalSeconds = Math.round(diffMs / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (minutes <= 0) return `${seconds}s`;
    return `${minutes}m ${seconds}s`;
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

function parentDirectory(path: string): string {
    if (!path || path === ".") return ".";
    const parts = path.split("/").filter(Boolean);
    parts.pop();
    return parts.length ? parts.join("/") : ".";
}

function withCacheBust(url: string, nonce = ""): string {
    if (!nonce) return url;
    const joiner = url.includes("?") ? "&" : "?";
    return `${url}${joiner}_=${encodeURIComponent(nonce)}`;
}

function fileViewUrl(path: string, nonce = ""): string {
    return withCacheBust(
        `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file/view?path=${encodeURIComponent(path)}`,
        nonce,
    );
}

function workspaceFileRenderUrl(path: string, nonce = ""): string {
    const encodedPath = String(path || "")
        .split("/")
        .filter(Boolean)
        .map((segment) => encodeURIComponent(segment))
        .join("/");
    return withCacheBust(
        `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file/render/${encodedPath}`,
        nonce,
    );
}

function workspaceFileApiUrl(path: string, nonce = ""): string {
    return withCacheBust(
        `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file?path=${encodeURIComponent(path)}`,
        nonce,
    );
}

function downloadSelectedFile(): void {
    if (!state.currentWorkspaceId || !state.selectedFilePath) return;
    const anchor = document.createElement("a");
    anchor.href = fileViewUrl(state.selectedFilePath);
    anchor.download = state.selectedFilePath.split("/").pop() || "download";
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
}

function openSelectedFileInTab(): void {
    if (!state.currentWorkspaceId || !state.selectedFilePath) return;
    window.open(fileViewUrl(state.selectedFilePath, String(Date.now())), "_blank", "noopener");
}

function inferredPreviewLanguage(path: string): string {
    const extension = artifactFileExtension(path);
    return normalizeCodeLanguage(extension || "text");
}

function shouldRenderTextAsMarkdown(path: string, content: string): boolean {
    const extension = artifactFileExtension(path);
    if (["md", "markdown", "mdx"].includes(extension)) return true;
    const trimmed = String(content || "").trim();
    return Boolean(trimmed) && (
        trimmed.includes("[[artifact:")
        || /(^|\n)#{1,4}\s/.test(trimmed)
        || /```[\s\S]*?```/.test(trimmed)
        || /\$\$[\s\S]*?\$\$/.test(trimmed)
        || /\\\[[\s\S]*?\\\]/.test(trimmed)
    );
}

function shouldRenderTextAsCode(path: string): boolean {
    const extension = artifactFileExtension(path);
    return Boolean(extension) && !["md", "markdown", "mdx", "txt"].includes(extension);
}

function composerHasConversationHistory(): boolean {
    return state.messages.some((message) => message.role === "user" || message.role === "assistant");
}

function composerMaxHeight(): number {
    return composerHasConversationHistory() ? LATER_TURN_COMPOSER_MAX_HEIGHT : FIRST_TURN_COMPOSER_MAX_HEIGHT;
}

function syncComposerHeight(): void {
    composerInput.style.height = "auto";
    const nextHeight = Math.min(Math.max(composerInput.scrollHeight, 42), composerMaxHeight());
    composerInput.style.height = `${nextHeight}px`;
    composerInput.style.overflowY = composerInput.scrollHeight > nextHeight ? "auto" : "hidden";
}

function pendingAttachmentPaths(): string[] {
    return state.pendingAttachments.map((attachment) => attachment.path);
}

function defaultComposerStatus(): string {
    if (state.composerUploadingCount > 0) {
        return `Uploading ${state.composerUploadingCount} file${state.composerUploadingCount === 1 ? "" : "s"}…`;
    }
    if (state.connectionState === "offline") {
        return "Runtime offline.";
    }
    if (!state.modelAvailable) {
        return state.modelName ? `Model loading: ${state.modelName}` : "Model loading…";
    }
    return state.modelName ? `Model ready: ${state.modelName}` : "Model ready.";
}

function syncComposerActions(): void {
    sendButton.disabled = state.composerUploadingCount > 0 && !state.generating;
}

function setViewerMode(nextMode: ViewerMode): void {
    state.viewerMode = nextMode;
    syncShellLayout();
}

function syncShellLayout(): void {
    document.body.dataset.leftSidebarOpen = String(state.leftSidebarOpen);
    document.body.dataset.viewerMode = state.viewerMode;
    const sidebarToggleLabel = state.leftSidebarOpen ? "Close sidebar" : "Open sidebar";
    sidebarToggle.setAttribute("aria-label", sidebarToggleLabel);
    sidebarToggle.dataset.tooltip = sidebarToggleLabel;

    const viewerToggleTitle = "Open file browser";
    viewerToggle.setAttribute("aria-label", viewerToggleTitle);
    viewerToggle.dataset.tooltip = viewerToggleTitle;
    viewerToggle.disabled = !state.currentWorkspaceId;
    viewerToggle.hidden = state.viewerMode !== "closed";

    refreshWorkspaceButton.hidden = state.viewerMode === "closed";
    refreshWorkspaceButton.disabled = !state.currentWorkspaceId;
    const showOpenInTabButton = state.viewerMode === "file"
        && Boolean(state.selectedFilePath)
        && state.selectedFileContentKind === "html";
    openInTabButton.hidden = !showOpenInTabButton;
    openInTabButton.disabled = !showOpenInTabButton;
    downloadFileButton.hidden = !(state.viewerMode === "file" && state.selectedFilePath);
    downloadFileButton.disabled = !state.selectedFilePath;
    const showModeButton = Boolean(state.selectedFilePath) && state.viewerMode === "file";
    viewerModeButton.hidden = !showModeButton;
    if (showModeButton) {
        const viewerModeLabel = "Back to file list";
        viewerModeButton.setAttribute("aria-label", viewerModeLabel);
        viewerModeButton.dataset.tooltip = viewerModeLabel;
    }

    viewerCloseButton.hidden = state.viewerMode === "closed";

    window.requestAnimationFrame(() => {
        if (state.stickChatToBottom) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        syncScrollJumpButton();
    });
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
    syncComposerActions();
    setConnectionState(nextValue ? "streaming" : (state.ws?.readyState === WebSocket.OPEN ? "online" : "offline"));
}

function clearVisibleGenerationState(): void {
    state.activeAssistantIndex = -1;
    state.thinkingStatus = null;
    setGenerating(false);
    setComposerHint(defaultComposerStatus());
}

function detachGenerationFromVisibleConversation(conversationId: string): void {
    if (!conversationId) return;
    if (state.activeStreamConversationId !== conversationId) return;
    clearVisibleGenerationState();
}

function setComposerHint(text: string): void {
    composerHint.textContent = text;
}

function normalizeThinkingText(text: string): string {
    return String(text || "")
        .replace(/^(Inspect|Deep|Verify|Synthesize):\s*/i, "")
        .replace(/^Slash \/[a-z0-9_-]+:\s*/i, "")
        .replace(/\s+/g, " ")
        .trim();
}

function findAssistantWorkStartedAt(messages: ChatMessage[], assistantIndex: number): string {
    for (let index = assistantIndex - 1; index >= 0; index -= 1) {
        const candidate = messages[index];
        if (candidate.role === "user" && candidate.timestamp) {
            return candidate.timestamp;
        }
    }
    return messages[assistantIndex]?.timestamp || "";
}

function assistantMetaLabel(message: ChatMessage, index: number): string {
    if (state.generating && index === state.activeAssistantIndex) {
        return "Assistant";
    }
    const startedAt = findAssistantWorkStartedAt(state.messages, index);
    const duration = formatElapsedDuration(startedAt || message.timestamp);
    if (duration) return `Worked for ${duration}`;
    return "Assistant";
}

function thinkingStatusLabel(status: ThinkingStatus): string {
    if (status.error) return "Attention";
    if (status.phase === "permission") return "Permissions";
    if (status.phase === "thinking") return "Working";
    return "Status";
}

function isSuppressedAssistantNote(text: string): boolean {
    const normalized = normalizeThinkingText(text).toLowerCase();
    return normalized.includes("i hit an unexpected internal error before i could finish the reply")
        || normalized.includes("please say continue and i'll retry from the current workspace state")
        || normalized.includes("please say continue and i’ll retry from the current workspace state");
}

function summarizeRuntimeError(text: string): string {
    const normalized = normalizeThinkingText(text);
    if (!normalized) {
        return "Hit a snag before the reply could finish.";
    }
    if (/workspace state|saved progress|saved workspace/i.test(normalized)) {
        return "The reply paused before the final answer, but the current work is still available.";
    }
    if (/model loaded|vllm|runtime/i.test(normalized)) {
        return "The model runtime was not ready to finish that reply.";
    }
    if (/internal error/i.test(normalized)) {
        return "Hit a snag before the reply could finish.";
    }
    return normalized;
}

function buildThinkingSummary(event: ChatEvent): string {
    if (event.type === "start") {
        return "Getting oriented.";
    }
    if (event.type === "status") {
        const normalized = normalizeThinkingText(event.content || "");
        if (/workspace execution path failed/i.test(normalized)) {
            return "Switching to the fallback path and continuing.";
        }
        return normalized || "Continuing.";
    }
    if (event.type === "permission_required") {
        return "Opening workspace access and continuing.";
    }
    if (event.type === "error") {
        return summarizeRuntimeError(event.content || "");
    }
    const normalized = normalizeThinkingText(event.content || "");
    if (!normalized && event.label) {
        return normalizeThinkingText(event.label);
    }
    if (/^Using tools to answer\.?$/i.test(normalized)) {
        return "Checking the workspace and gathering context.";
    }
    return normalized || "Working through the next step.";
}

function setThinkingStatus(text: string, options: Partial<ThinkingStatus> = {}): void {
    const normalized = normalizeThinkingText(text);
    if (!normalized) return;
    state.thinkingStatus = {
        text: normalized,
        phase: options.phase || "thinking",
        error: Boolean(options.error),
        persistent: Boolean(options.persistent),
    };
    setComposerHint(`Thinking: ${normalized}`);
    renderMessages();
}

function clearThinkingStatus(): void {
    state.thinkingStatus = null;
    setComposerHint(defaultComposerStatus());
    renderMessages();
}

function truncatePreview(value: string, limit = 88): string {
    const flattened = value.replace(/\s+/g, " ").trim();
    if (flattened.length <= limit) return flattened;
    return `${flattened.slice(0, limit - 1)}…`;
}

function cleanConversationText(value: string): string {
    return String(value || "")
        .replace(/<\/?think>/gi, " ")
        .replace(/\[\[artifact:([^\]]+)\]\]/gi, "$1")
        .replace(/[`*_#>\[\]]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function conversationTitleLooksLowQuality(title: string): boolean {
    const normalized = cleanConversationText(title).toLowerCase();
    if (!normalized || normalized === "new chat" || normalized === "untitled" || normalized === "untitled chat") {
        return true;
    }
    if (normalized.endsWith("?")) {
        return true;
    }
    return [
        "okay", "ok", "sure", "yes", "no", "here", "heres", "here's",
        "let's", "lets", "i can", "i will", "i'll", "we can", "you can",
        "can you", "could you", "would you", "help me",
    ].some((prefix) => normalized.startsWith(prefix));
}

function displayConversationTitle(title: string, preview: string, seed = ""): string {
    const cleanedTitle = cleanConversationText(title);
    const cleanedSeed = cleanConversationText(seed);
    const cleanedPreview = cleanConversationText(preview);
    const base = cleanedTitle && !conversationTitleLooksLowQuality(cleanedTitle)
        ? cleanedTitle
        : (cleanedSeed || cleanedPreview || "New chat");
    const words = base.split(/\s+/).filter(Boolean);
    if (words.length <= 5) return base;
    return `${words.slice(0, 5).join(" ")}…`;
}

function displayConversationPreview(title: string, preview: string): string {
    const cleanedTitle = cleanConversationText(title).toLowerCase();
    const cleanedPreview = truncatePreview(cleanConversationText(preview), 72);
    if (!cleanedPreview || cleanedPreview.toLowerCase() === cleanedTitle) {
        return "No messages yet";
    }
    return cleanedPreview;
}

function lineCount(text: string): number {
    return text ? text.split(/\r?\n/).length : 0;
}

function inferPastedLanguage(text: string): { language: string; extension: string } {
    const trimmed = text.trim();
    if (!trimmed) return { language: "text", extension: "txt" };
    if (/^#!/.test(trimmed) || /(^|\n)\s*(from\s+\w+\s+import|import\s+\w+|def\s+\w+\(|class\s+\w+[\(:])/m.test(trimmed)) {
        return { language: "python", extension: "py" };
    }
    if (/^\s*<!doctype html>|<html\b|<body\b|<head\b/i.test(trimmed)) {
        return { language: "html", extension: "html" };
    }
    if (/(^|\n)\s*(SELECT|INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM|CREATE TABLE|ALTER TABLE)\b/i.test(trimmed)) {
        return { language: "sql", extension: "sql" };
    }
    if (/^\s*[{[]/.test(trimmed)) {
        try {
            JSON.parse(trimmed);
            return { language: "json", extension: "json" };
        } catch (_error) {
            // Ignore malformed JSON and continue through the remaining heuristics.
        }
    }
    if (/(^|\n)\s*(export\s+default|export\s+(const|function|class)|import\s+.+from\s+['"]|const\s+\w+\s*=|let\s+\w+\s*=|function\s+\w+\()/m.test(trimmed)) {
        return { language: "javascript", extension: "js" };
    }
    if (/(^|\n)\s*#\s+/.test(trimmed) || /(^|\n)\s*##\s+/.test(trimmed)) {
        return { language: "markdown", extension: "md" };
    }
    if (/(^|\n)\s*(body|html|main|div|section|header|footer)\s*\{/.test(trimmed)) {
        return { language: "css", extension: "css" };
    }
    return { language: "text", extension: "txt" };
}

function shouldMaterializePastedText(text: string): boolean {
    const trimmed = text.trim();
    if (!trimmed) return false;
    const chars = trimmed.length;
    const lines = lineCount(trimmed);
    if (chars >= PASTE_ATTACH_CHAR_THRESHOLD || lines >= PASTE_ATTACH_LINE_THRESHOLD) {
        return true;
    }
    const { extension } = inferPastedLanguage(trimmed);
    return extension !== "txt" && (chars > 320 || lines > 12);
}

function suggestPastedFilename(text: string): string {
    const { extension } = inferPastedLanguage(text);
    const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
    return `pasted/paste-${stamp}.${extension}`;
}

async function materializeComposerPaste(text: string): Promise<string> {
    if (!state.currentWorkspaceId) {
        throw new Error("Select a workspace before attaching pasted files.");
    }
    const path = suggestPastedFilename(text);
    setComposerUploadState(1);
    try {
        await fetchJson(`/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                path,
                content: text,
            }),
        });
        const { extension } = inferPastedLanguage(text);
        state.pendingAttachments = mergePendingAttachments([
            ...state.pendingAttachments,
            {
                name: path.split("/").pop() || path,
                path,
                kind: extension,
                contentType: "text/plain",
                size: text.length,
            },
        ]);
        renderComposerAttachments();
    } finally {
        setComposerUploadState(0);
    }
    await loadDirectory(parentDirectory(path));
    await openFile(path);
    return path;
}

function mergePendingAttachments(items: PendingComposerAttachment[]): PendingComposerAttachment[] {
    const deduped = new Map<string, PendingComposerAttachment>();
    items.forEach((item) => {
        if (!item.path) return;
        deduped.set(item.path, item);
    });
    return [...deduped.values()];
}

function setComposerDropActive(active: boolean): void {
    state.composerDropActive = active;
    composerForm.classList.toggle("drag-active", active);
}

function setComposerUploadState(count: number): void {
    state.composerUploadingCount = Math.max(0, count);
    syncComposerActions();
    renderComposerAttachments();
    if (!state.generating) {
        setComposerHint(defaultComposerStatus());
    }
}

function removePendingAttachment(path: string): void {
    state.pendingAttachments = state.pendingAttachments.filter((attachment) => attachment.path !== path);
    renderComposerAttachments();
    setComposerHint(state.pendingAttachments.length ? `${state.pendingAttachments.length} file${state.pendingAttachments.length === 1 ? "" : "s"} attached.` : defaultComposerStatus());
}

function renderComposerAttachments(): void {
    if (!state.pendingAttachments.length && state.composerUploadingCount <= 0) {
        composerAttachments.hidden = true;
        composerAttachments.innerHTML = "";
        return;
    }
    composerAttachments.hidden = false;
    const uploadingMarkup = state.composerUploadingCount > 0 ? `
        <div class="composer-attachment-chip uploading" aria-live="polite">
            <span class="composer-attachment-kind">Upload</span>
            <span class="composer-attachment-name">
                Uploading ${state.composerUploadingCount} file${state.composerUploadingCount === 1 ? "" : "s"}…
            </span>
        </div>
    ` : "";
    const attachmentMarkup = state.pendingAttachments.map((attachment) => {
        const kindLabel = displayFileKindLabel(attachment.path, attachment.kind || "", "file");
        return `
            <div class="composer-attachment-chip" data-path="${escapeHtml(attachment.path)}">
                <span class="composer-attachment-kind">${escapeHtml(kindLabel)}</span>
                <span class="composer-attachment-name">${escapeHtml(attachment.name)}</span>
                <button
                    type="button"
                    class="composer-attachment-remove"
                    data-remove-path="${escapeHtml(attachment.path)}"
                    aria-label="Remove ${escapeHtml(attachment.name)}"
                    title="Remove ${escapeHtml(attachment.name)}"
                >
                    ×
                </button>
            </div>
        `;
    }).join("");
    composerAttachments.innerHTML = `${uploadingMarkup}${attachmentMarkup}`;
    composerAttachments.querySelectorAll<HTMLButtonElement>("[data-remove-path]").forEach((button) => {
        button.addEventListener("click", () => {
            const path = button.dataset.removePath || "";
            if (!path) return;
            removePendingAttachment(path);
        });
    });
}

async function uploadComposerFiles(files: File[]): Promise<PendingComposerAttachment[]> {
    if (!state.currentWorkspaceId) {
        throw new Error("Select a workspace before attaching files.");
    }
    const uploads = files.filter((file) => file && file.size >= 0);
    if (!uploads.length) return [];
    setComposerUploadState(uploads.length);
    let attachments: PendingComposerAttachment[] = [];
    try {
        const formData = new FormData();
        uploads.forEach((file) => formData.append("files", file, file.name));
        formData.append("target_path", ".");
        const payload = await fetchJson<WorkspaceUploadResponse>(
            `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/upload`,
            {
                method: "POST",
                body: formData,
            },
        );
        attachments = (payload.files || []).map((file) => ({
            name: file.name,
            path: file.path,
            size: file.size,
            kind: file.kind,
            contentType: file.content_type,
        }));
        state.pendingAttachments = mergePendingAttachments([...state.pendingAttachments, ...attachments]);
        renderComposerAttachments();
    } finally {
        setComposerUploadState(0);
    }
    await loadDirectory(".");
    return attachments;
}

function mergeComposerText(note: string): void {
    const existing = composerInput.value.trim();
    composerInput.value = existing ? `${existing}\n\n${note}` : note;
    syncComposerHeight();
}

const ARTIFACT_HELPER_TEXT_PATTERNS = [
    /^\s*(?:you can )?(?:open|view|inspect|see|read|preview|find)\b[^.!?]*?(?:here|below)?[:.]?\s*$/i,
    /^\s*(?:saved plan(?: and notes)?|saved progress|task board|feedback digest|full feedback digest|artifact|artifacts?)[:.]?\s*$/i,
];

const MATERIALIZABLE_ARTIFACT_EXTENSIONS = new Set([
    "md", "txt", "py", "js", "ts", "tsx", "jsx", "json", "html", "css", "sql",
    "sh", "bash", "zsh", "yaml", "yml", "toml", "csv", "xml", "svg", "c", "h",
    "cpp", "hpp", "java", "rs", "go", "rb", "php",
]);

function isNearBottom(element: HTMLElement, threshold = 72): boolean {
    return element.scrollHeight - element.scrollTop - element.clientHeight <= threshold;
}

function syncScrollJumpButton(): void {
    scrollToBottomButton.hidden = !state.messages.length || state.stickChatToBottom;
}

function scrollChatToBottom(behavior: ScrollBehavior = "auto"): void {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior });
    state.stickChatToBottom = true;
    syncScrollJumpButton();
}

function extractArtifactReferences(message: string): string[] {
    const refs: string[] = [];
    for (const match of String(message || "").matchAll(/\[\[artifact:([^\]]+)\]\]/gi)) {
        const cleaned = String(match[1] || "").trim();
        if (cleaned && !refs.includes(cleaned)) {
            refs.push(cleaned);
        }
    }
    return refs;
}

function resolveArtifactReferencePath(path: string): string {
    return String(path || "")
        .trim()
        .replace(/^`|`$/g, "")
        .replace(/\\/g, "/")
        .replace(/^\.\//, "")
        .replace(/^\/+/, "");
}

function artifactFileExtension(path: string): string {
    const fileName = resolveArtifactReferencePath(path).split("/").pop() || "";
    const match = fileName.match(/\.([A-Za-z0-9]+)$/);
    return String(match?.[1] || "").toLowerCase();
}

function isSpreadsheetPreviewPath(path: string): boolean {
    const extension = artifactFileExtension(path);
    return ["xlsx", "xls", "xlsm"].includes(extension);
}

function displayFileKindLabel(path: string, contentKind = "", itemType = "file"): string {
    if (itemType === "directory") return "Dir";
    const extension = artifactFileExtension(path);
    if (extension === "c") return "C";
    if (extension === "h") return "Header";
    if (["cpp", "cxx", "cc"].includes(extension)) return "C++";
    if (extension === "hpp") return "C++ Header";
    if (extension === "py") return "Python";
    if (extension === "js") return "JS";
    if (extension === "ts") return "TS";
    if (extension === "tsx") return "TSX";
    if (extension === "jsx") return "JSX";
    if (extension === "rs") return "Rust";
    if (extension === "go") return "Go";
    if (extension === "java") return "Java";
    if (extension === "sql") return "SQL";
    if (extension === "html") return "HTML";
    if (extension === "css") return "CSS";
    if (extension === "json") return "JSON";
    if (extension === "md") return "Markdown";

    const normalized = String(contentKind || "").trim().toLowerCase();
    if (normalized === "code") return "Code";
    if (normalized === "markdown") return "Markdown";
    if (normalized === "html") return "HTML";
    if (normalized === "pdf") return "PDF";
    if (normalized === "image") return "Image";
    if (normalized === "csv") return "CSV";
    if (normalized === "spreadsheet") return "Sheet";
    return "File";
}

function displayFilePreviewLabel(path: string, contentKind = ""): string {
    const extension = artifactFileExtension(path);
    if (extension === "c") return "C source";
    if (extension === "h") return "C header";
    if (["cpp", "cxx", "cc"].includes(extension)) return "C++ source";
    if (extension === "hpp") return "C++ header";
    if (extension === "py") return "Python";
    if (extension === "js") return "JavaScript";
    if (extension === "ts") return "TypeScript";
    if (extension === "tsx") return "TSX";
    if (extension === "jsx") return "JSX";
    if (extension === "rs") return "Rust";
    if (extension === "go") return "Go";
    if (extension === "java") return "Java";
    if (extension === "sql") return "SQL";
    if (extension === "html") return "HTML";
    if (extension === "css") return "CSS";
    if (extension === "json") return "JSON";
    if (extension === "md") return "Markdown";

    const normalized = String(contentKind || "").trim().toLowerCase();
    if (normalized === "text") return "Text";
    if (normalized === "code") return "Code";
    if (normalized) return normalized.toUpperCase();
    return "Preview";
}

function unwrapSingleFence(raw: string): { language: string; body: string } | null {
    const trimmed = String(raw || "").trim();
    const match = trimmed.match(/^```([^\n`]*)\n([\s\S]*)\n```$/);
    if (!match) return null;
    return {
        language: normalizeCodeLanguage(match[1] || ""),
        body: String(match[2] || "").replace(/\r\n?/g, "\n"),
    };
}

function extractSingleFenceFromMessage(raw: string): { language: string; body: string } | null {
    const matches = [...String(raw || "").matchAll(/```([^\n`]*)\n([\s\S]*?)\n```/g)];
    if (matches.length !== 1) return null;
    return {
        language: normalizeCodeLanguage(matches[0][1] || ""),
        body: String(matches[0][2] || "").replace(/\r\n?/g, "\n"),
    };
}

function stripArtifactHelperLines(text: string): string {
    return String(text || "")
        .split(/\r?\n/)
        .filter((line) => !ARTIFACT_HELPER_TEXT_PATTERNS.some((pattern) => pattern.test(line.trim())))
        .join("\n");
}

function stripSingleFenceFromMessage(raw: string): string {
    const matches = [...String(raw || "").matchAll(/```([^\n`]*)\n([\s\S]*?)\n```/g)];
    if (matches.length !== 1) return String(raw || "").trim();
    const match = matches[0];
    const before = String(raw || "").slice(0, match.index || 0).trim();
    const after = String(raw || "").slice((match.index || 0) + match[0].length).trim();
    return [before, after].filter(Boolean).join("\n\n").trim();
}

function parseArtifactMessage(raw: string): {
    artifactPaths: string[];
    displayContent: string;
    materializedBody: string;
} {
    const cleaned = stripArtifactHelperLines(raw);
    const artifactMatch = cleaned.match(/\[\[artifact:([^\]]+)\]\]/i);
    const artifactPaths = extractArtifactReferences(cleaned).map(resolveArtifactReferencePath);
    if (!artifactMatch) {
        return {
            artifactPaths,
            displayContent: cleaned,
            materializedBody: "",
        };
    }

    const before = cleaned.slice(0, artifactMatch.index || 0).trim();
    const after = cleaned.slice((artifactMatch.index || 0) + artifactMatch[0].length).trim();
    const fenceFromAfter = extractSingleFenceFromMessage(after);
    if (after && fenceFromAfter) {
        return {
            artifactPaths,
            displayContent: [stripSingleFenceFromMessage(before), stripSingleFenceFromMessage(after)].filter(Boolean).join("\n\n").trim(),
            materializedBody: after,
        };
    }

    const fenceFromBefore = extractSingleFenceFromMessage(before);
    if (!after && fenceFromBefore) {
        return {
            artifactPaths,
            displayContent: stripSingleFenceFromMessage(before).trim(),
            materializedBody: before,
        };
    }

    const displayParts = [before, after].filter(Boolean);
    return {
        artifactPaths,
        displayContent: displayParts.join("\n\n").trim(),
        materializedBody: after,
    };
}

function shouldMaterializeArtifact(path: string, body: string): boolean {
    if (!resolveArtifactReferencePath(path) || !String(body || "").trim()) return false;
    const extension = artifactFileExtension(path);
    return !extension || MATERIALIZABLE_ARTIFACT_EXTENSIONS.has(extension);
}

function normalizeArtifactBodyForWrite(path: string, raw: string): string {
    const trimmed = String(raw || "").trim();
    const fence = unwrapSingleFence(trimmed) || extractSingleFenceFromMessage(trimmed);
    if (!fence) return trimmed;
    const extension = artifactFileExtension(path);
    if (!extension) return fence.body;
    if (extension === "md" && ["markdown", "md", "text"].includes(fence.language)) {
        return fence.body;
    }
    if (extension === fence.language) {
        return fence.body;
    }
    if (extension === "py" && fence.language === "python") return fence.body;
    if (extension === "js" && fence.language === "javascript") return fence.body;
    if (extension === "ts" && fence.language === "javascript") return fence.body;
    if (extension === "sh" && fence.language === "bash") return fence.body;
    return trimmed;
}

async function materializeArtifactFromAssistant(path: string, content: string): Promise<void> {
    if (!state.currentWorkspaceId) return;
    await fetchJson(`/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            path,
            content: normalizeArtifactBodyForWrite(path, content),
        }),
    });
    await loadDirectory(parentDirectory(path));
    await openFile(path);
}

function renderArtifactReferences(paths: string[]): string {
    if (!paths.length) return "";
    return `
        <div class="artifact-ref-list">
            ${paths.map((path) => `
                <button
                    type="button"
                    class="artifact-ref"
                    data-artifact-path="${escapeHtml(path)}"
                    title="${escapeHtml(path)}"
                >
                    ${escapeHtml(path)}
                </button>
            `).join("")}
        </div>
    `;
}

function summarizeArtifactDisplayText(text: string, limit = 140): string {
    const cleaned = cleanConversationText(text);
    if (!cleaned) return "";
    if (cleaned.length <= limit) return cleaned;
    return `${cleaned.slice(0, Math.max(0, limit - 1)).trimEnd()}…`;
}

function renderArtifactObject(paths: string[], note = "", status = "Saved in viewer"): string {
    if (!paths.length) return "";
    const summary = summarizeArtifactDisplayText(note);
    return `
        <div class="artifact-object">
            <div class="artifact-object-meta">${escapeHtml(status)}</div>
            ${renderArtifactReferences(paths)}
            ${summary ? `<div class="artifact-object-note">${escapeHtml(summary)}</div>` : ""}
        </div>
    `;
}

function bindArtifactLinks(root: ParentNode): void {
    root.querySelectorAll<HTMLElement>("[data-artifact-path]").forEach((element) => {
        element.addEventListener("click", (event) => {
            event.preventDefault();
            const path = resolveArtifactReferencePath(element.dataset.artifactPath || "");
            if (!path) return;
            void openFile(path);
        });
    });
}

function maybeHandleAssistantArtifacts(message: ChatMessage, index: number): void {
    if (message.role !== "assistant" || !state.currentWorkspaceId) return;
    if (state.generating && index === state.activeAssistantIndex) return;
    const parsed = parseArtifactMessage(message.content || "");
    if (!parsed.artifactPaths.length) return;
    const artifactBody = parsed.materializedBody;

    const primaryPath = parsed.artifactPaths[0];
    const artifactKey = [
        state.currentConversationId,
        primaryPath,
        artifactBody.length,
        artifactBody.slice(0, 120),
    ].join(":");
    if (state.handledArtifactKeys.has(artifactKey)) return;
    state.handledArtifactKeys.add(artifactKey);

    void (async () => {
        try {
            setLatestWorkedFile(primaryPath);
            if (shouldMaterializeArtifact(primaryPath, artifactBody)) {
                await materializeArtifactFromAssistant(primaryPath, artifactBody);
                return;
            }
            await openFile(primaryPath);
        } catch (_error) {
            state.handledArtifactKeys.delete(artifactKey);
        }
    })();
}

type MathSegment = {
    placeholder: string;
    html: string;
};

const MATH_COMMAND_HTML: Record<string, string> = {
    alpha: "&alpha;",
    beta: "&beta;",
    gamma: "&gamma;",
    delta: "&delta;",
    epsilon: "&epsilon;",
    theta: "&theta;",
    lambda: "&lambda;",
    mu: "&mu;",
    pi: "&pi;",
    sigma: "&sigma;",
    phi: "&phi;",
    omega: "&omega;",
    Gamma: "&Gamma;",
    Delta: "&Delta;",
    Theta: "&Theta;",
    Lambda: "&Lambda;",
    Pi: "&Pi;",
    Sigma: "&Sigma;",
    Phi: "&Phi;",
    Omega: "&Omega;",
    cdot: "&middot;",
    times: "&times;",
    pm: "&plusmn;",
    mp: "∓",
    le: "&le;",
    leq: "&le;",
    ge: "&ge;",
    geq: "&ge;",
    neq: "&ne;",
    ne: "&ne;",
    infty: "&infin;",
    to: "&rarr;",
    rightarrow: "&rarr;",
    Rightarrow: "&Rightarrow;",
    leftarrow: "&larr;",
    Leftarrow: "&lArr;",
    implies: "&Rightarrow;",
    approx: "&asymp;",
    sim: "&sim;",
    partial: "&part;",
    nabla: "&nabla;",
    int: "&int;",
    sum: "&sum;",
    prod: "&prod;",
    ln: "ln",
    log: "log",
    sin: "sin",
    cos: "cos",
    tan: "tan",
    sec: "sec",
    csc: "csc",
    cot: "cot",
    lim: "lim",
};

const MATH_SPACING_COMMANDS = new Set([
    "quad", "qquad", ",", ";", "!", ":", "enspace", "enskip", "medspace", "thinspace",
]);

function readMathGroup(source: string, start: number): { content: string; end: number } | null {
    if (source[start] !== "{") return null;
    let depth = 0;
    let i = start;
    let content = "";
    while (i < source.length) {
        const char = source[i];
        if (char === "{") {
            if (depth > 0) content += char;
            depth += 1;
            i += 1;
            continue;
        }
        if (char === "}") {
            depth -= 1;
            if (depth === 0) {
                return { content, end: i + 1 };
            }
            if (depth < 0) return null;
            content += char;
            i += 1;
            continue;
        }
        content += char;
        i += 1;
    }
    return null;
}

function readMathEntity(source: string, start: number): { html: string; end: number } | null {
    if (source[start] !== "&") return null;
    const end = source.indexOf(";", start + 1);
    if (end === -1) return null;
    return { html: source.slice(start, end + 1), end: end + 1 };
}

function readMathAtom(source: string, start: number): { html: string; end: number } {
    if (start >= source.length) return { html: "", end: start };
    const char = source[start];

    if (char === " ") return { html: " ", end: start + 1 };
    if (char === "\n") return { html: "<br>", end: start + 1 };

    const entity = readMathEntity(source, start);
    if (entity) return entity;

    if (char === "{") {
        const group = readMathGroup(source, start);
        if (group) {
            return { html: renderMathExpression(group.content), end: group.end };
        }
    }

    if (char === "\\") {
        const commandMatch = source.slice(start + 1).match(/^([A-Za-z]+|.)/);
        if (!commandMatch) return { html: "\\", end: start + 1 };
        const command = commandMatch[1];
        const end = start + 1 + command.length;

        if (command === "\\") return { html: "<br>", end };
        if (command === "left" || command === "right") return readMathAtom(source, end);
        if (command === "frac") {
            const numerator = readMathGroup(source, end);
            const denominator = numerator ? readMathGroup(source, numerator.end) : null;
            if (numerator && denominator) {
                return {
                    html: `<span class="math-frac"><span class="math-frac-num">${renderMathExpression(numerator.content)}</span><span class="math-frac-den">${renderMathExpression(denominator.content)}</span></span>`,
                    end: denominator.end,
                };
            }
        }
        if (command === "sqrt") {
            const radicand = readMathGroup(source, end);
            if (radicand) {
                return {
                    html: `<span class="math-sqrt"><span class="math-sqrt-sign">&radic;</span><span class="math-sqrt-body">${renderMathExpression(radicand.content)}</span></span>`,
                    end: radicand.end,
                };
            }
        }
        if (command === "text") {
            const textGroup = readMathGroup(source, end);
            if (textGroup) {
                return {
                    html: `<span class="math-text">${textGroup.content}</span>`,
                    end: textGroup.end,
                };
            }
        }
        if (MATH_SPACING_COMMANDS.has(command)) return { html: "&nbsp;", end };
        return { html: MATH_COMMAND_HTML[command] || command, end };
    }

    return { html: char, end: start + 1 };
}

function readMathScript(source: string, start: number): { html: string; end: number } {
    if (start >= source.length) return { html: "", end: start };
    if (source[start] === "{") {
        const group = readMathGroup(source, start);
        if (group) {
            return { html: renderMathExpression(group.content), end: group.end };
        }
    }
    return readMathAtom(source, start);
}

function renderMathExpression(source: string): string {
    const katex = getKatexRenderer();
    if (katex) {
        try {
            return katex.renderToString(source, {
                displayMode: false,
                throwOnError: false,
                strict: "ignore",
            });
        } catch {
            // Fall through to the built-in renderer if KaTeX chokes on malformed input.
        }
    }

    let i = 0;
    let html = "";

    while (i < source.length) {
        const atom = readMathAtom(source, i);
        let base = atom.html;
        i = atom.end;

        while (i < source.length && (source[i] === "^" || source[i] === "_")) {
            const operator = source[i];
            const script = readMathScript(source, i + 1);
            const tag = operator === "^" ? "sup" : "sub";
            base += `<${tag}>${script.html}</${tag}>`;
            i = script.end;
        }

        html += base;
    }

    return html.replace(/ {2,}/g, " ");
}

function renderDisplayMathExpression(source: string): string {
    const katex = getKatexRenderer();
    if (katex) {
        try {
            return katex.renderToString(source, {
                displayMode: true,
                throwOnError: false,
                strict: "ignore",
            });
        } catch {
            // Fall through to the built-in renderer if KaTeX chokes on malformed input.
        }
    }
    return `<div class="math-block">${renderMathExpression(source)}</div>`;
}

function extractInlineMathSegments(text: string): { content: string; mathSegments: MathSegment[] } {
    const mathSegments: MathSegment[] = [];
    let content = "";

    const stash = (html: string): string => {
        const placeholder = `@@MATH_${mathSegments.length}@@`;
        mathSegments.push({ placeholder, html });
        return placeholder;
    };

    let i = 0;
    while (i < text.length) {
        if (text[i] === "$" && text[i + 1] !== "$") {
            const end = text.indexOf("$", i + 1);
            if (end > i + 1) {
                content += stash(`<span class="math-inline">${renderMathExpression(text.slice(i + 1, end))}</span>`);
                i = end + 1;
                continue;
            }
        }
        if (text.startsWith("\\(", i)) {
            const end = text.indexOf("\\)", i + 2);
            if (end !== -1) {
                content += stash(`<span class="math-inline">${renderMathExpression(text.slice(i + 2, end))}</span>`);
                i = end + 2;
                continue;
            }
        }
        content += text[i];
        i += 1;
    }

    return { content, mathSegments };
}

function restoreMathSegments(text: string, mathSegments: MathSegment[]): string {
    return mathSegments.reduce(
        (current, segment) => current.replace(segment.placeholder, segment.html),
        text,
    );
}

function renderMathBlock(block: string): string | null {
    const trimmed = block.trim();
    const doubleDollar = trimmed.match(/^\$\$([\s\S]*?)\$\$$/);
    if (doubleDollar) {
        return `<div class="math-block">${renderDisplayMathExpression(doubleDollar[1].trim())}</div>`;
    }
    const bracketed = trimmed.match(/^\\\[([\s\S]*?)\\\]$/);
    if (bracketed) {
        return `<div class="math-block">${renderDisplayMathExpression(bracketed[1].trim())}</div>`;
    }
    return null;
}

function renderInlineMarkdown(text: string): string {
    const extracted = extractInlineMathSegments(text);
    const rendered = extracted.content
        .replace(/\[\[artifact:([^\]]+)\]\]/gi, (_match, rawPath: string) => {
            const path = resolveArtifactReferencePath(rawPath);
            const label = path.split("/").pop() || path;
            return `<button type="button" class="artifact-inline-ref" data-artifact-path="${escapeHtml(path)}">${escapeHtml(label)}</button>`;
        })
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
        .replace(/\n+/g, " ");
    return restoreMathSegments(rendered, extracted.mathSegments);
}

type MarkdownListItem = {
    lines: string[];
};

type MarkdownBlock =
    | { type: "paragraph"; lines: string[] }
    | { type: "heading"; level: number; text: string }
    | { type: "hr" }
    | { type: "html"; html: string }
    | { type: "quote"; lines: string[] }
    | { type: "math"; content: string }
    | { type: "list"; ordered: boolean; start: number; items: MarkdownListItem[] };

function renderMarkdownBlocks(blocks: MarkdownBlock[]): string {
    return blocks.map((block) => {
        if (block.type === "paragraph") {
            return `<p>${renderInlineMarkdown(block.lines.join("\n"))}</p>`;
        }
        if (block.type === "heading") {
            return `<h${block.level}>${renderInlineMarkdown(block.text)}</h${block.level}>`;
        }
        if (block.type === "hr") {
            return "<hr>";
        }
        if (block.type === "html") {
            return block.html;
        }
        if (block.type === "quote") {
            return `<blockquote>${renderInlineMarkdown(block.lines.join("\n"))}</blockquote>`;
        }
        if (block.type === "math") {
            return `<div class="math-block">${renderDisplayMathExpression(block.content)}</div>`;
        }
        if (block.type === "list") {
            const tag = block.ordered ? "ol" : "ul";
            const startAttr = block.ordered && block.start > 1 ? ` start="${block.start}"` : "";
            return `<${tag}${startAttr}>${block.items.map((item) => `<li>${renderInlineMarkdown(item.lines.join("\n"))}</li>`).join("")}</${tag}>`;
        }
        return "";
    }).join("");
}

function parseMarkdownBlocks(raw: string): MarkdownBlock[] {
    const codeBlocks: string[] = [];
    const fenced = raw.replace(/```([^\n`]*)\n?([\s\S]*?)```/g, (_match, language: string, code: string) => {
        const token = `@@CODEBLOCK_${codeBlocks.length}@@`;
        const normalizedLanguage = normalizeCodeLanguage(language);
        const cleanCode = code.replace(/^\n+|\n+$/g, "");
        if (normalizedLanguage === "latex") {
            codeBlocks.push(renderDisplayMathExpression(cleanCode));
            return token;
        }
        const displayLanguage = normalizedLanguage === "text" ? "text" : normalizedLanguage;
        codeBlocks.push(`
            <div class="message-code-block">
                <div class="message-code-header">${escapeHtml(displayLanguage)}</div>
                <pre><code class="language-${escapeHtml(normalizedLanguage)}">${highlightCode(cleanCode, normalizedLanguage)}</code></pre>
            </div>
        `.trim());
        return token;
    });

    const blocks: MarkdownBlock[] = [];
    const paragraphLines: string[] = [];
    let activeList: { ordered: boolean; start: number; items: MarkdownListItem[] } | null = null;
    let quoteLines: string[] = [];
    let mathFence: "$$" | "\\[" | null = null;
    let mathLines: string[] = [];

    const flushParagraph = () => {
        if (!paragraphLines.length) return;
        blocks.push({ type: "paragraph", lines: [...paragraphLines] });
        paragraphLines.length = 0;
    };

    const flushList = () => {
        if (!activeList || !activeList.items.length) {
            activeList = null;
            return;
        }
        blocks.push({
            type: "list",
            ordered: activeList.ordered,
            start: activeList.start,
            items: activeList.items.map((item) => ({ lines: [...item.lines] })),
        });
        activeList = null;
    };

    const flushQuote = () => {
        if (!quoteLines.length) return;
        blocks.push({ type: "quote", lines: [...quoteLines] });
        quoteLines = [];
    };

    const flushMathBlock = () => {
        if (!mathFence) return;
        blocks.push({ type: "math", content: mathLines.join("\n").trim() });
        mathFence = null;
        mathLines = [];
    };

    const flushBlocks = () => {
        flushParagraph();
        flushList();
        flushQuote();
    };

    const appendListContinuation = (line: string) => {
        if (!activeList || !activeList.items.length) return false;
        activeList.items[activeList.items.length - 1].lines.push(line.trim());
        return true;
    };

    for (const rawLine of fenced.split("\n")) {
        const line = rawLine.replace(/\r$/, "");
        const trimmed = line.trim();

        if (mathFence) {
            const closing = mathFence === "$$" ? "$$" : "\\]";
            if (trimmed.endsWith(closing)) {
                const content = trimmed.slice(0, Math.max(0, trimmed.length - closing.length)).trim();
                if (content) mathLines.push(content);
                flushMathBlock();
            } else {
                mathLines.push(line);
            }
            continue;
        }

        if (!trimmed) {
            flushParagraph();
            flushQuote();
            continue;
        }

        if (/^@@CODEBLOCK_\d+@@$/.test(trimmed)) {
            flushBlocks();
            blocks.push({ type: "html", html: trimmed });
            continue;
        }

        const mathBlock = renderMathBlock(trimmed);
        if (mathBlock) {
            flushBlocks();
            blocks.push({ type: "html", html: mathBlock });
            continue;
        }

        if (trimmed === "$$" || trimmed === "\\[") {
            flushBlocks();
            mathFence = trimmed as "$$" | "\\[";
            mathLines = [];
            continue;
        }

        if (/^[-*_]{3,}$/.test(trimmed)) {
            flushBlocks();
            blocks.push({ type: "hr" });
            continue;
        }

        const heading = line.match(/^\s*(#{1,4})\s+(.*)$/);
        if (heading) {
            flushBlocks();
            blocks.push({ type: "heading", level: Math.min(heading[1].length, 4), text: heading[2].trim() });
            continue;
        }

        const quote = line.match(/^\s*>\s?(.*)$/);
        if (quote) {
            flushParagraph();
            flushList();
            quoteLines.push(quote[1]);
            continue;
        }
        flushQuote();

        const unordered = line.match(/^\s*[-*]\s+(.*)$/);
        if (unordered) {
            flushParagraph();
            if (activeList && activeList.ordered) flushList();
            activeList ||= { ordered: false, start: 1, items: [] };
            activeList.items.push({ lines: [unordered[1].trim()] });
            continue;
        }

        const ordered = line.match(/^\s*(\d+)\.\s+(.*)$/);
        if (ordered) {
            flushParagraph();
            if (activeList && !activeList.ordered) flushList();
            const start = Number(ordered[1]) || 1;
            if (!activeList) {
                activeList = { ordered: true, start, items: [] };
            }
            activeList.items.push({ lines: [ordered[2].trim()] });
            continue;
        }

        if (appendListContinuation(line)) {
            continue;
        }

        paragraphLines.push(line);
    }

    flushBlocks();
    flushMathBlock();

    return blocks.map((block) => {
        if (block.type !== "html") return block;
        return {
            ...block,
            html: block.html.replace(/@@CODEBLOCK_(\d+)@@/g, (_match, index: string) => codeBlocks[Number(index)] || ""),
        };
    });
}

function normalizeCodeLanguage(language?: string): string {
    const normalized = String(language || "").trim().toLowerCase();
    if (!normalized) return "text";
    if (["py", "python"].includes(normalized)) return "python";
    if (["js", "jsx", "javascript", "ts", "tsx", "typescript"].includes(normalized)) return "javascript";
    if (["sh", "shell", "bash", "zsh"].includes(normalized)) return "bash";
    if (["c", "h", "cpp", "cxx", "hpp"].includes(normalized)) return "c";
    if (["sql", "sqlite", "postgresql", "postgres", "mysql"].includes(normalized)) return "sql";
    if (["json"].includes(normalized)) return "json";
    if (["latex", "tex", "math"].includes(normalized)) return "latex";
    return normalized;
}

function keywordPatternForLanguage(language: string): RegExp | null {
    const patterns: Record<string, string> = {
        python: String.raw`\b(?:and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield)\b`,
        javascript: String.raw`\b(?:async|await|break|case|catch|class|const|continue|default|delete|else|export|extends|false|finally|for|from|function|if|import|in|instanceof|let|new|null|of|return|switch|throw|true|try|typeof|var|while|yield)\b`,
        bash: String.raw`\b(?:if|then|else|elif|fi|for|do|done|case|esac|while|in|function|return|local|export)\b`,
        c: String.raw`\b(?:auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|if|inline|int|long|register|restrict|return|short|signed|sizeof|static|struct|switch|typedef|union|unsigned|void|volatile|while)\b`,
        sql: String.raw`\b(?:add|all|alter|and|as|asc|avg|begin|between|by|case|check|column|commit|constraint|count|create|database|default|delete|desc|distinct|drop|else|end|exists|from|group|having|in|index|insert|into|is|join|key|like|limit|not|null|on|or|order|primary|references|rollback|select|set|table|then|transaction|union|unique|update|values|view|where)\b`,
        json: String.raw`\b(?:true|false|null)\b`,
    };
    const pattern = patterns[language];
    return pattern ? new RegExp(pattern, "g") : null;
}

function commentPatternForLanguage(language: string): RegExp | null {
    if (language === "python" || language === "bash") return /#.*$/gm;
    if (language === "javascript" || language === "c") return /\/\/.*$|\/\*[\s\S]*?\*\//gm;
    if (language === "sql") return /--.*$/gm;
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
    const singleFence = unwrapSingleFence(raw);
    if (singleFence && ["markdown", "md"].includes(singleFence.language)) {
        return renderRichText(singleFence.body);
    }
    return renderMarkdownBlocks(parseMarkdownBlocks(raw));
}

function renderMessages(): void {
    const previousScrollTop = chatMessages.scrollTop;
    const shouldStickToBottom = state.stickChatToBottom;
    const visibleThinking = state.thinkingStatus && (state.generating || state.thinkingStatus.persistent)
        ? state.thinkingStatus
        : null;

    if (!state.messages.length && !visibleThinking) {
        chatMessages.innerHTML = `
            <div class="empty-state">
                <div>
                    <div class="empty-state-title">Start a conversation</div>
                    <p>Ask the assistant to inspect code, make a change, or explain something here.</p>
                </div>
            </div>
        `;
        state.stickChatToBottom = true;
        syncScrollJumpButton();
        return;
    }

    const messagesHtml = state.messages.map((message, index) => {
        const role = message.error ? "assistant error" : message.role;
        const parsedArtifacts = message.role === "assistant"
            ? parseArtifactMessage(message.content || "")
            : { artifactPaths: [], displayContent: message.content || "", materializedBody: "" };
        const meta = message.role === "user"
            ? "You"
            : (message.role === "assistant" ? assistantMetaLabel(message, index) : "System");
        const artifactObjectHtml = message.role === "assistant" && parsedArtifacts.artifactPaths.length && parsedArtifacts.materializedBody
            ? renderArtifactObject(
                parsedArtifacts.artifactPaths,
                parsedArtifacts.displayContent || "",
                parsedArtifacts.displayContent ? "Updated file in viewer" : "Saved in viewer",
            )
            : "";
        const bodyHtml = parsedArtifacts.displayContent
            ? `<div class="message-content rich-text">${renderRichText(parsedArtifacts.displayContent || "")}</div>`
            : "";
        const messageBody = artifactObjectHtml || `
                ${renderArtifactReferences(parsedArtifacts.artifactPaths)}
                ${bodyHtml}
            `;

        return `
            <article class="message ${role}">
                <div class="message-role">${escapeHtml(meta)}</div>
                ${messageBody}
            </article>
        `;
    }).join("");
    const thinkingHtml = visibleThinking ? `
        <div class="message-thinking${visibleThinking.error ? " error" : ""}">
            <span class="message-thinking-label">${escapeHtml(thinkingStatusLabel(visibleThinking))}</span>
            <span class="message-thinking-text">${escapeHtml(visibleThinking.text)}</span>
        </div>
    ` : "";
    chatMessages.innerHTML = messagesHtml || thinkingHtml
        ? `${messagesHtml}${thinkingHtml}`
        : `
            <div class="empty-state">
                <div>
                    <div class="empty-state-title">Start a conversation</div>
                    <p>Ask the assistant to inspect code, make a change, or explain something here.</p>
                </div>
            </div>
        `;
    bindArtifactLinks(chatMessages);

    state.messages.forEach((message, index) => {
        maybeHandleAssistantArtifacts(message, index);
    });

    if (shouldStickToBottom) {
        scrollChatToBottom("auto");
    } else {
        chatMessages.scrollTop = previousScrollTop;
        syncScrollJumpButton();
    }
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

    conversationList.innerHTML = items.map((conversation) => {
        const editableTitle = cleanConversationText(conversation.title || "");
        const visibleTitle = displayConversationTitle(
            conversation.title || "",
            conversation.last_message || "",
            conversation.seed_message || "",
        );
        const visiblePreview = displayConversationPreview(conversation.title || "", conversation.last_message || "");
        return `
            <div class="conversation-row${conversation.id === state.currentConversationId ? " active" : ""}">
                <button
                    type="button"
                    class="conversation-item${conversation.id === state.currentConversationId ? " active" : ""}"
                    data-conversation-id="${escapeHtml(conversation.id)}"
                    data-tooltip="${escapeHtml(visiblePreview)}"
                >
                    <div class="conversation-line">
                        <span class="conversation-title">${escapeHtml(visibleTitle)}</span>
                    </div>
                </button>
                <div class="conversation-actions">
                    <button type="button" class="conversation-action-button" data-action="rename" data-conversation-id="${escapeHtml(conversation.id)}" data-title="${escapeHtml(editableTitle || visibleTitle)}" data-tooltip="Rename chat">
                        <svg class="shell-toggle-icon" viewBox="0 0 20 20" aria-hidden="true">
                            <path d="M4.25 14.75 4.75 12l8.5-8.5 2.75 2.75-8.5 8.5z"></path>
                            <path d="M12.5 4.25 15.25 7"></path>
                        </svg>
                        <span class="sr-only">Rename</span>
                    </button>
                    <button type="button" class="conversation-action-button conversation-action-danger" data-action="delete" data-conversation-id="${escapeHtml(conversation.id)}" data-tooltip="Delete chat">
                        <svg class="shell-toggle-icon" viewBox="0 0 20 20" aria-hidden="true">
                            <path d="M6.5 6.5v8"></path>
                            <path d="M10 6.5v8"></path>
                            <path d="M13.5 6.5v8"></path>
                            <path d="M4.5 5.5h11"></path>
                            <path d="M7.25 5.5 8 3.5h4l.75 2"></path>
                            <path d="M5.5 5.5v10.25h9V5.5"></path>
                        </svg>
                        <span class="sr-only">Delete</span>
                    </button>
                </div>
            </div>
        `;
    }).join("");

    conversationList.querySelectorAll<HTMLButtonElement>(".conversation-item").forEach((button) => {
        button.addEventListener("click", () => {
            const id = button.dataset.conversationId || "";
            if (!id) return;
            if (isMobileViewport()) {
                state.leftSidebarOpen = false;
                syncShellLayout();
            }
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

function renderWorkspaceSummary(): void {
    workspaceSettingsButton.disabled = false;
    syncShellLayout();
}

function setLatestWorkedFile(path: string): void {
    const normalized = resolveArtifactReferencePath(path);
    if (!normalized) return;
    state.latestWorkedFilePath = normalized;
}

function renderFileList(): void {
    const previousScrollTop = fileList.scrollTop;
    directoryPath.textContent = state.currentDirectoryPath === "." ? "/" : `/${state.currentDirectoryPath}`;
    if (state.viewerMode !== "file") {
        viewerTitle.textContent = "Files";
        viewerMeta.textContent = "";
        viewerMeta.hidden = true;
    }

    const buttons: string[] = [];
    if (state.currentDirectoryPath !== ".") {
        buttons.push(`
            <button type="button" class="file-item up" data-path="..">
                <div class="file-item-name"><span class="file-item-kind">Up</span>Parent directory</div>
            </button>
        `);
    }

    if (state.latestWorkedFilePath) {
        const latestName = state.latestWorkedFilePath.split("/").pop() || state.latestWorkedFilePath;
        buttons.push(`
            <button
                type="button"
                class="file-item latest-worked${state.latestWorkedFilePath === state.selectedFilePath ? " active" : ""}"
                data-path="${escapeHtml(state.latestWorkedFilePath)}"
                data-kind="file"
            >
                <div class="file-item-name file-item-name-stacked">
                    <span class="file-item-kind latest">Latest</span>
                    <strong>${escapeHtml(latestName)}</strong>
                    <span class="file-item-subpath">${escapeHtml(state.latestWorkedFilePath)}</span>
                </div>
                <div class="file-item-meta">Open</div>
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
                <span class="file-item-kind">${escapeHtml(displayFileKindLabel(item.path, item.content_kind || "", item.type))}</span>
                ${escapeHtml(item.name)}
            </div>
            <div class="file-item-meta">${escapeHtml(formatBytes(item.size) || "")}</div>
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
    fileList.scrollTop = previousScrollTop;

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

function renderSettingsSummary(): void {
    settingsSummary.innerHTML = `
        <div><strong>Conversation:</strong> ${escapeHtml(state.currentConversationTitle || "New chat")}</div>
        <div><strong>Runtime:</strong> ${escapeHtml(defaultComposerStatus())}</div>
    `;
}

function showSettings(): void {
    renderSettingsSummary();
    renderContextEvalReport();
    renderFixtureReviewList();
    renderFixtureReviewDetail();
    void loadContextEvalReport();
    void loadContextEvalFixtures();
    settingsOverlay.hidden = false;
}

function closeSettings(): void {
    settingsOverlay.hidden = true;
}

function renderPreviewEmpty(title: string, body: string): void {
    viewerTitle.textContent = title;
    viewerMeta.textContent = state.viewerMode === "file" ? "Preview" : "";
    viewerMeta.hidden = !viewerMeta.textContent;
    state.selectedFileContentKind = "";
    filePreview.innerHTML = `
        <div class="empty-state">
            <div>
                <div class="empty-state-title">${escapeHtml(title)}</div>
                <p>${escapeHtml(body)}</p>
            </div>
        </div>
    `;
}

function formatSpreadsheetCell(value: unknown): string {
    if (value === null || value === undefined || value === "") return "";
    if (typeof value === "string") return escapeHtml(value);
    if (typeof value === "number" || typeof value === "boolean") return escapeHtml(String(value));
    return escapeHtml(JSON.stringify(value));
}

function renderSpreadsheetPreview(payload: SpreadsheetPreviewPayload): void {
    state.selectedFileContentKind = "spreadsheet";
    const sheetNames = payload.sheet_names || [];
    const previewRows = payload.preview_rows || [];
    const columns = previewRows.length
        ? Object.keys(previewRows[0] || {})
        : (payload.columns || []).map((column) => typeof column === "string" ? column : String(column.name || ""));

    const workbookSummary = (payload.workbook_sheets || [])
        .map((sheet) => `
            <div class="spreadsheet-sheet-chip${sheet.name === payload.sheet ? " active" : ""}">
                <strong>${escapeHtml(sheet.name)}</strong>
                <span>${escapeHtml(`${sheet.row_count} rows • ${sheet.column_count} cols`)}</span>
            </div>
        `)
        .join("");

    filePreview.innerHTML = `
        <div class="spreadsheet-preview">
            <div class="spreadsheet-preview-header">
                <div class="spreadsheet-meta-grid">
                    <div><strong>Rows</strong><span>${escapeHtml(String(payload.row_count || 0))}</span></div>
                    <div><strong>Columns</strong><span>${escapeHtml(String(payload.column_count || 0))}</span></div>
                    <div><strong>Type</strong><span>${escapeHtml((payload.file_type || "sheet").toUpperCase())}</span></div>
                </div>
                ${sheetNames.length > 1 ? `
                    <label class="spreadsheet-sheet-picker">
                        <span>Sheet</span>
                        <select id="spreadsheetSheetSelect">
                            ${sheetNames.map((name) => `<option value="${escapeHtml(name)}"${name === payload.sheet ? " selected" : ""}>${escapeHtml(name)}</option>`).join("")}
                        </select>
                    </label>
                ` : ""}
            </div>
            ${workbookSummary ? `<div class="spreadsheet-sheet-list">${workbookSummary}</div>` : ""}
            ${columns.length ? `
                <div class="preview-table-shell">
                    <table class="preview-table">
                        <thead>
                            <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
                        </thead>
                        <tbody>
                            ${previewRows.length ? previewRows.map((row) => `
                                <tr>${columns.map((column) => `<td>${formatSpreadsheetCell(row[column])}</td>`).join("")}</tr>
                            `).join("") : `<tr><td colspan="${Math.max(columns.length, 1)}">No preview rows available.</td></tr>`}
                        </tbody>
                    </table>
                </div>
            ` : `<div class="empty-state"><div><div class="empty-state-title">No sheet data</div><p>This spreadsheet does not contain previewable columns.</p></div></div>`}
        </div>
    `;

    const sheetSelect = filePreview.querySelector<HTMLSelectElement>("#spreadsheetSheetSelect");
    if (sheetSelect) {
        sheetSelect.addEventListener("change", () => {
            void openSpreadsheetFile(payload.path, sheetSelect.value);
        });
    }
}

const RESPONSIVE_HTML_PREVIEW_STYLE = `
html {
    box-sizing: border-box;
    overflow-x: hidden;
}

*, *::before, *::after {
    box-sizing: inherit;
}

body {
    margin: 0;
    padding: 16px;
    max-width: 100%;
    overflow-x: hidden;
    background: white;
}

img,
svg,
video,
canvas,
iframe,
embed,
object {
    display: block;
    max-width: 100% !important;
}

img,
svg,
video {
    height: auto !important;
}

canvas {
    height: auto !important;
}

table {
    display: block;
    max-width: 100%;
    overflow-x: auto;
}

pre {
    max-width: 100%;
    overflow: auto;
}
`.trim();

function buildResponsiveHtmlPreview(rawHtml: string): string {
    const html = String(rawHtml || "");
    const viewportMeta = /<meta[^>]+name=["']viewport["']/i.test(html)
        ? ""
        : '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">';
    const previewStyle = `<style id="codex-responsive-preview">${RESPONSIVE_HTML_PREVIEW_STYLE}</style>`;
    const injectedHead = `${viewportMeta}${previewStyle}`;

    if (/<head\b/i.test(html)) {
        return html.replace(/<head([^>]*)>/i, `<head$1>${injectedHead}`);
    }

    if (/<html\b/i.test(html)) {
        return html.replace(/<html([^>]*)>/i, `<html$1><head>${injectedHead}</head>`);
    }

    return `<!DOCTYPE html><html><head>${injectedHead}</head><body>${html}</body></html>`;
}

async function openSpreadsheetFile(path: string, sheet = ""): Promise<void> {
    if (!state.currentWorkspaceId) return;
    const query = sheet ? `&sheet=${encodeURIComponent(sheet)}` : "";
    const payload = await fetchJson<SpreadsheetPreviewPayload>(
        `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/spreadsheet?path=${encodeURIComponent(path)}${query}`
    );
    renderSpreadsheetPreview(payload);
}

function renderPreview(payload: WorkspaceFilePayload): void {
    const contentKind = payload.content_kind || "text";
    state.selectedFileContentKind = contentKind;
    viewerTitle.textContent = payload.path;
    viewerMeta.textContent = `${displayFilePreviewLabel(payload.path, contentKind)} preview`;
    const structureMeta = documentStructureMeta(payload);
    if (structureMeta) {
        viewerMeta.textContent = `${viewerMeta.textContent} • ${structureMeta}`;
    }
    viewerMeta.hidden = false;

    if (contentKind === "image") {
        filePreview.innerHTML = `<img class="file-preview-media" alt="${escapeHtml(payload.path)}" src="${fileViewUrl(payload.path, state.activePreviewNonce)}">`;
        return;
    }

    if (contentKind === "pdf") {
        filePreview.innerHTML = `${renderDocumentStructurePanel(payload)}<iframe class="file-preview-frame" title="${escapeHtml(payload.path)}" src="${fileViewUrl(payload.path, state.activePreviewNonce)}"></iframe>`;
        return;
    }

    if (contentKind === "html") {
        filePreview.innerHTML = `<iframe class="file-preview-frame" sandbox="allow-scripts" title="${escapeHtml(payload.path)}" src="${workspaceFileRenderUrl(payload.path, state.activePreviewNonce)}"></iframe>`;
        return;
    }

    if (contentKind === "markdown") {
        filePreview.innerHTML = `<div class="preview-markdown rich-text">${renderRichText(payload.content || "")}</div>`;
        bindArtifactLinks(filePreview);
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

    if (contentKind === "text") {
        if (shouldRenderTextAsMarkdown(payload.path || "", payload.content || "")) {
            viewerMeta.textContent = "Markdown preview";
            if (structureMeta) {
                viewerMeta.textContent = `${viewerMeta.textContent} • ${structureMeta}`;
            }
            filePreview.innerHTML = `${renderDocumentStructurePanel(payload)}<div class="preview-markdown rich-text">${renderRichText(payload.content || "")}</div>`;
            bindArtifactLinks(filePreview);
            return;
        }
        if (shouldRenderTextAsCode(payload.path || "")) {
            const language = inferredPreviewLanguage(payload.path || "");
            filePreview.innerHTML = `${renderDocumentStructurePanel(payload)}<div class="preview-markdown rich-text">${renderRichText(`\`\`\`${language}\n${payload.content || ""}\n\`\`\``)}</div>`;
            bindArtifactLinks(filePreview);
            return;
        }
    }

    filePreview.innerHTML = `${renderDocumentStructurePanel(payload)}<pre class="preview-code">${escapeHtml(payload.content || "")}</pre>`;
}

function documentStructureMeta(payload: WorkspaceFilePayload): string {
    const parts: string[] = [];
    if (typeof payload.page_count === "number" && payload.page_count > 0) {
        parts.push(`${payload.page_count} page${payload.page_count === 1 ? "" : "s"}`);
    }
    const sectionCount = Array.isArray(payload.section_titles) ? payload.section_titles.filter(Boolean).length : 0;
    if (sectionCount > 0) {
        parts.push(`${sectionCount} section${sectionCount === 1 ? "" : "s"}`);
    }
    if (typeof payload.chunk_count === "number" && payload.chunk_count > 0) {
        parts.push(`${payload.chunk_count} chunk${payload.chunk_count === 1 ? "" : "s"}`);
    }
    return parts.join(" • ");
}

function formatPreviewChunkLabel(chunk: NonNullable<WorkspaceFilePayload["preview_chunks"]>[number]): string {
    const parts: string[] = [];
    const title = String(chunk.section_title || "").trim();
    if (title) {
        parts.push(title);
    }
    const pageStart = typeof chunk.page_start === "number" ? chunk.page_start : null;
    const pageEnd = typeof chunk.page_end === "number" ? chunk.page_end : null;
    if (pageStart && pageEnd && pageStart !== pageEnd) {
        parts.push(`pages ${pageStart}-${pageEnd}`);
    } else if (pageStart) {
        parts.push(`page ${pageStart}`);
    }
    if (!parts.length) {
        parts.push(`Chunk ${Number(chunk.chunk_index || 0) + 1}`);
    }
    return parts.join(" • ");
}

function renderDocumentStructurePanel(payload: WorkspaceFilePayload): string {
    const sections = Array.isArray(payload.section_titles)
        ? payload.section_titles.map((item) => String(item || "").trim()).filter(Boolean).slice(0, 10)
        : [];
    const previewChunks = Array.isArray(payload.preview_chunks)
        ? payload.preview_chunks.filter((chunk) => chunk && String(chunk.text || "").trim()).slice(0, 3)
        : [];
    const openingPreview = String(payload.opening_preview || "").trim();
    const summary = String(payload.summary || "").trim();
    const hasStructure = Boolean(summary || sections.length || openingPreview || previewChunks.length);
    if (!hasStructure) {
        return "";
    }

    const statItems = [
        typeof payload.page_count === "number" && payload.page_count > 0
            ? `<span class="document-structure-stat">${payload.page_count} page${payload.page_count === 1 ? "" : "s"}</span>`
            : "",
        typeof payload.chunk_count === "number" && payload.chunk_count > 0
            ? `<span class="document-structure-stat">${payload.chunk_count} chunk${payload.chunk_count === 1 ? "" : "s"}</span>`
            : "",
        typeof payload.line_count === "number" && payload.line_count > 0
            ? `<span class="document-structure-stat">${payload.line_count.toLocaleString()} lines</span>`
            : "",
        String(payload.extractor || "").trim()
            ? `<span class="document-structure-stat">${escapeHtml(String(payload.extractor || "").trim())}</span>`
            : "",
    ].filter(Boolean).join("");

    const sectionHtml = sections.length
        ? `<div class="document-structure-sections">${sections.map((section) => `<span class="document-structure-section">${escapeHtml(section)}</span>`).join("")}</div>`
        : "";
    const openingHtml = openingPreview
        ? `<div class="document-structure-opening">${escapeHtml(openingPreview)}</div>`
        : "";
    const chunkHtml = previewChunks.length
        ? `<div class="document-structure-chunks">${previewChunks.map((chunk) => `
            <article class="document-structure-chunk">
                <div class="document-structure-chunk-label">${escapeHtml(formatPreviewChunkLabel(chunk))}</div>
                <div class="document-structure-chunk-text">${escapeHtml(String(chunk.text || "").trim())}</div>
            </article>
        `).join("")}</div>`
        : "";

    return `
        <section class="document-structure-panel">
            <div class="document-structure-header">
                <div class="document-structure-kicker">Document structure</div>
                ${summary ? `<div class="document-structure-summary">${escapeHtml(summary)}</div>` : ""}
                ${statItems ? `<div class="document-structure-stats">${statItems}</div>` : ""}
            </div>
            ${sectionHtml}
            ${openingHtml}
            ${chunkHtml}
        </section>
    `;
}

function closeViewer(): void {
    setViewerMode("closed");
}

function openFileTree(): void {
    if (!state.currentWorkspaceId) return;
    if (isMobileViewport()) {
        state.leftSidebarOpen = false;
    }
    setViewerMode("tree");
    renderFileList();
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
    const requestInit: RequestInit = {
        cache: "no-store",
        ...init,
    };
    const response = await fetch(url, requestInit);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const message = typeof data.detail === "string" ? data.detail : `Request failed: ${response.status}`;
        throw new Error(message);
    }
    return data as T;
}

const {
    loadContextEvalFixtures,
    loadContextEvalReport,
    renderContextEvalReport,
    renderFixtureReviewDetail,
    renderFixtureReviewList,
} = createContextEvalController({
    escapeHtml,
    fetchJson,
    formatDecimal,
    openFile,
    setComposerHint,
    showSettings,
});

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

    const previousWorkspaceId = state.currentWorkspaceId;
    state.workspaces = payload.workspaces || [];
    const nextWorkspaceId = preferredId
        || state.currentWorkspaceId
        || payload.default_workspace_id
        || state.workspaces[0]?.id
        || "";

    state.currentWorkspaceId = nextWorkspaceId;
    if (nextWorkspaceId !== previousWorkspaceId) {
        state.latestWorkedFilePath = "";
        state.pendingAttachments = [];
        setComposerUploadState(0);
    }
    if (nextWorkspaceId) {
        localStorage.setItem("lastWorkspaceId", nextWorkspaceId);
    } else {
        state.selectedFilePath = "";
        state.latestWorkedFilePath = "";
        closeViewer();
    }

    renderWorkspaceSummary();
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

async function loadConversation(id: string): Promise<void> {
    if (state.generating && state.currentConversationId && state.currentConversationId !== id) {
        detachGenerationFromVisibleConversation(state.currentConversationId);
    }
    const payload = await fetchJson<{
        messages: ChatMessage[];
        workspace_id?: string;
    }>(`/api/conversation/${encodeURIComponent(id)}`);

    state.currentConversationId = id;
    state.messages = payload.messages || [];
    state.stickChatToBottom = true;
    state.activeAssistantIndex = -1;
    state.thinkingStatus = null;
    state.pendingAttachments = [];
    setComposerUploadState(0);
    state.selectedFilePath = "";
    state.latestWorkedFilePath = "";
    state.viewerMode = "closed";

    const matchingConversation = state.conversations.find((conversation) => conversation.id === id);
    state.currentConversationTitle = matchingConversation?.title || "New chat";

    if (payload.workspace_id && payload.workspace_id !== state.currentWorkspaceId) {
        await loadWorkspaces(payload.workspace_id);
    } else {
        renderConversations();
    }

    renderComposerAttachments();
    renderMessages();
    syncShellLayout();
    if (state.activeStreamConversationId === id) {
        setGenerating(true);
        setThinkingStatus("Continuing the reply.", { phase: "thinking" });
        renderMessages();
    }
    await loadContextEvalReport();
}

function startNewChat(): void {
    if (state.generating && state.currentConversationId) {
        detachGenerationFromVisibleConversation(state.currentConversationId);
    }
    state.currentConversationId = generateId();
    state.currentConversationTitle = "New chat";
    state.messages = [];
    state.pendingAttachments = [];
    setComposerUploadState(0);
    state.stickChatToBottom = true;
    state.activeAssistantIndex = -1;
    state.thinkingStatus = null;
    state.selectedFilePath = "";
    state.latestWorkedFilePath = "";
    state.viewerMode = "closed";
    renderComposerAttachments();
    renderMessages();
    renderConversations();
    syncShellLayout();
    syncComposerHeight();
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

function finishGeneration(options: { preserveThinking?: boolean } = {}): void {
    state.activeStreamConversationId = "";
    state.activeAssistantIndex = -1;
    setGenerating(false);
    if (!options.preserveThinking) {
        state.thinkingStatus = null;
        setComposerHint(defaultComposerStatus());
    } else if (state.thinkingStatus) {
        setComposerHint(`Thinking: ${state.thinkingStatus.text}`);
    } else {
        setComposerHint(defaultComposerStatus());
    }
    renderMessages();
    window.clearTimeout(state.conversationRefreshTimer);
    void loadConversations();
    state.conversationRefreshTimer = window.setTimeout(() => {
        void loadConversations();
    }, 1600);
    void fetchHealth();
    void loadDirectory(state.currentDirectoryPath);
    void loadContextEvalReport();
    syncComposerHeight();
}

function handleChatEvent(event: ChatEvent): void {
    if (!event || typeof event !== "object") return;
    const eventConversationId = String(event.conversation_id || state.activeStreamConversationId || "").trim();
    const isVisibleConversationEvent = !eventConversationId || eventConversationId === state.currentConversationId;

    if (event.type === "start") {
        if (eventConversationId) {
            state.activeStreamConversationId = eventConversationId;
        }
        if (!isVisibleConversationEvent) return;
        ensureAssistantMessage();
        setGenerating(true);
        setThinkingStatus(buildThinkingSummary(event));
        return;
    }

    if (event.type === "assistant_note") {
        if (!isVisibleConversationEvent) return;
        if (isSuppressedAssistantNote(event.content || "")) {
            setThinkingStatus("Switching to the fallback path and continuing.", { phase: "thinking" });
            return;
        }
        const assistantMessage = ensureAssistantMessage();
        assistantMessage.content = event.content || "";
        renderMessages();
        return;
    }

    if (event.type === "final_replace") {
        if (!isVisibleConversationEvent) return;
        const assistantMessage = ensureAssistantMessage();
        assistantMessage.content = event.content || "";
        renderMessages();
        return;
    }

    if (event.type === "token") {
        if (!isVisibleConversationEvent) return;
        const assistantMessage = ensureAssistantMessage();
        assistantMessage.content += event.content || "";
        renderMessages();
        return;
    }

    if ((event.type === "activity" || event.type === "reasoning_note" || event.type === "status") && event.content) {
        if (!isVisibleConversationEvent) return;
        setThinkingStatus(buildThinkingSummary(event), { phase: event.phase || "thinking" });
        return;
    }

    if (event.type === "tool_result" && event.payload?.open_path) {
        if (!isVisibleConversationEvent) return;
        setLatestWorkedFile(event.payload.open_path);
        void openFile(event.payload.open_path);
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
            if (isVisibleConversationEvent) {
                setThinkingStatus(buildThinkingSummary(event), { phase: "permission" });
            }
        } else {
            if (isVisibleConversationEvent) {
                setThinkingStatus(buildThinkingSummary(event), { phase: "permission", error: true, persistent: true });
            }
        }
        return;
    }

    if (event.type === "error") {
        if (!isVisibleConversationEvent) {
            if (eventConversationId && eventConversationId === state.activeStreamConversationId) {
                state.activeStreamConversationId = "";
            }
            void loadConversations();
            void fetchHealth();
            return;
        }
        const assistantMessage = state.activeAssistantIndex >= 0
            ? state.messages[state.activeAssistantIndex]
            : null;
        const hasVisibleReply = Boolean(
            assistantMessage
            && assistantMessage.role === "assistant"
            && String(assistantMessage.content || "").trim()
        );
        setThinkingStatus(buildThinkingSummary(event), {
            phase: event.phase || "thinking",
            error: true,
            persistent: !hasVisibleReply,
        });
        if (assistantMessage && assistantMessage.role === "assistant" && hasVisibleReply) {
            assistantMessage.error = true;
            renderMessages();
        }
        finishGeneration({ preserveThinking: !hasVisibleReply });
        return;
    }

    if (event.type === "canceled" || event.type === "done") {
        if (!isVisibleConversationEvent) {
            if (eventConversationId && eventConversationId === state.activeStreamConversationId) {
                state.activeStreamConversationId = "";
            }
            void loadConversations();
            void fetchHealth();
            return;
        }
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
    if (state.composerUploadingCount > 0) {
        setComposerHint(defaultComposerStatus());
        return;
    }

    const message = composerInput.value.trim();
    const attachments = pendingAttachmentPaths();
    if (!message && !attachments.length) return;
    if (!state.currentConversationId) {
        startNewChat();
    }

    state.messages.push({
        role: "user",
        content: message || attachments.map((path) => `Attached \`${path.split("/").pop() || path}\``).join(", "),
        timestamp: new Date().toISOString(),
    });
    state.stickChatToBottom = true;

    composerInput.value = "";
    state.pendingAttachments = [];
    syncComposerHeight();
    renderComposerAttachments();
    renderMessages();
    renderConversations();
    setGenerating(true);
    state.activeStreamConversationId = state.currentConversationId;
    setThinkingStatus(state.modelAvailable ? "Getting started." : "Getting started while the model finishes loading.");

    const payload = {
        message,
        conversation_id: state.currentConversationId,
        workspace_id: state.currentWorkspaceId || null,
        attachments,
        mode: "auto",
        turn_kind: "visible_chat",
        features: {
            agent_tools: true,
            workspace_write: true,
            auto_approve_tool_permissions: true,
        },
    };

    try {
        await dispatchChatPayload(payload);
    } catch (error) {
        state.pendingAttachments = mergePendingAttachments([
            ...state.pendingAttachments,
            ...attachments.map((path) => ({
                path,
                name: path.split("/").pop() || path,
            })),
        ]);
        renderComposerAttachments();
        setThinkingStatus(
            error instanceof Error ? summarizeRuntimeError(error.message) : "The reply could not be sent.",
            { error: true, persistent: true },
        );
        finishGeneration({ preserveThinking: true });
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
    state.stickChatToBottom = true;
    state.selectedFilePath = "";
    state.viewerMode = "closed";
    state.pendingAttachments = [];
    setComposerUploadState(0);
    renderMessages();
    renderPreviewEmpty("Open a file", "Select a file from the workspace to preview it here.");
    await loadWorkspaces();
    await loadConversations();
}

async function loadDirectory(path: string): Promise<void> {
    if (!state.currentWorkspaceId) {
        state.selectedFilePath = "";
        state.viewerMode = "closed";
        syncShellLayout();
        renderPreviewEmpty("No workspace", "Create or select a workspace first.");
        return;
    }

    const payload = await fetchJson<{
        path: string;
        items: WorkspaceItem[];
    }>(withCacheBust(
        `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/files?path=${encodeURIComponent(path)}`,
        String(Date.now()),
    ));

    state.currentDirectoryPath = payload.path || ".";
    state.fileItems = payload.items || [];
    renderFileList();
}

async function openFile(path: string): Promise<void> {
    if (!state.currentWorkspaceId) return;

    const samePath = state.selectedFilePath === path;
    const previousPreviewScrollTop = samePath ? filePreview.scrollTop : 0;
    state.selectedFilePath = path;
    state.activePreviewNonce = String(Date.now());
    if (isMobileViewport()) {
        state.leftSidebarOpen = false;
    }
    state.viewerMode = "file";
    syncShellLayout();
    renderFileList();

    try {
        if (isSpreadsheetPreviewPath(path)) {
            await openSpreadsheetFile(path);
            filePreview.scrollTop = previousPreviewScrollTop;
            return;
        }
        const payload = await fetchJson<WorkspaceFilePayload>(
            workspaceFileApiUrl(path, state.activePreviewNonce)
        );
        renderPreview(payload);
        filePreview.scrollTop = previousPreviewScrollTop;
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
            setThinkingStatus("The reply stopped before it could finish.", { error: true, persistent: true });
            finishGeneration({ preserveThinking: true });
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
    refreshFixtureReviewButton.addEventListener("click", () => {
        void loadContextEvalFixtures();
    });
    fixtureReviewFilter.addEventListener("change", () => {
        state.fixtureReviewFilter = fixtureReviewFilter.value || "all";
        renderFixtureReviewList();
    });
    resetAppButton.addEventListener("click", () => {
        void resetAppData();
    });
    settingsOverlay.addEventListener("click", (event) => {
        if (event.target === settingsOverlay) closeSettings();
    });
    chatMessages.addEventListener("scroll", () => {
        state.stickChatToBottom = isNearBottom(chatMessages);
        syncScrollJumpButton();
    });
    scrollToBottomButton.addEventListener("click", () => {
        scrollChatToBottom("smooth");
    });
    refreshContextEvalButton.addEventListener("click", () => {
        void loadContextEvalReport();
    });

    refreshWorkspaceButton.addEventListener("click", () => {
        void (async () => {
            await loadDirectory(state.currentDirectoryPath);
            if (state.viewerMode === "file" && state.selectedFilePath) {
                await openFile(state.selectedFilePath);
            }
        })();
    });
    openInTabButton.addEventListener("click", () => {
        openSelectedFileInTab();
    });
    downloadFileButton.addEventListener("click", () => {
        downloadSelectedFile();
    });

    sidebarToggle.addEventListener("click", () => {
        state.leftSidebarOpen = !state.leftSidebarOpen;
        if (state.leftSidebarOpen && isMobileViewport()) {
            state.viewerMode = "closed";
        }
        syncShellLayout();
    });

    viewerToggle.addEventListener("click", () => {
        if (state.viewerMode === "closed") {
            openFileTree();
            return;
        }
        closeViewer();
    });

    viewerModeButton.addEventListener("click", () => {
        if (!state.selectedFilePath) return;
        if (state.viewerMode === "file") {
            openFileTree();
            return;
        }
        void openFile(state.selectedFilePath);
    });

    viewerCloseButton.addEventListener("click", () => {
        closeViewer();
    });

    composerForm.addEventListener("submit", (event) => {
        event.preventDefault();
        void sendCurrentMessage();
    });

    composerInput.addEventListener("input", () => {
        syncComposerHeight();
    });

    composerInput.addEventListener("paste", (event: ClipboardEvent) => {
        const pastedText = event.clipboardData?.getData("text/plain") || "";
        if (!shouldMaterializePastedText(pastedText)) return;
        event.preventDefault();
        void materializeComposerPaste(pastedText)
            .then((path) => {
                const filename = path.split("/").pop() || path;
                mergeComposerText(`Attached \`${filename}\`. Please use that file for this request.`);
                setComposerHint(`Attached pasted file: ${filename}`);
            })
            .catch((error) => {
                mergeComposerText(pastedText);
                setComposerHint(error instanceof Error ? error.message : "Could not attach pasted file.");
            });
    });

    composerForm.addEventListener("dragenter", (event: DragEvent) => {
        if (!event.dataTransfer?.types.includes("Files")) return;
        event.preventDefault();
        setComposerDropActive(true);
    });

    composerForm.addEventListener("dragover", (event: DragEvent) => {
        if (!event.dataTransfer?.types.includes("Files")) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        if (!state.composerDropActive) setComposerDropActive(true);
    });

    composerForm.addEventListener("dragleave", (event: DragEvent) => {
        const relatedTarget = event.relatedTarget as Node | null;
        if (relatedTarget && composerForm.contains(relatedTarget)) return;
        setComposerDropActive(false);
    });

    composerForm.addEventListener("drop", (event: DragEvent) => {
        const droppedFiles = [...(event.dataTransfer?.files || [])];
        if (!droppedFiles.length) return;
        event.preventDefault();
        setComposerDropActive(false);
        void uploadComposerFiles(droppedFiles)
            .then((attachments) => {
                if (!attachments.length) return;
                const label = attachments.length === 1
                    ? `Attached ${attachments[0].name}.`
                    : `Attached ${attachments.length} files.`;
                setComposerHint(label);
            })
            .catch((error) => {
                setComposerHint(error instanceof Error ? error.message : "Could not attach dropped files.");
            });
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
    renderComposerAttachments();
    renderContextEvalReport();
    renderPreviewEmpty("Open a file", "Select a file from the workspace to preview it here.");
    syncComposerHeight();
    connectWebSocket();
    window.clearInterval(state.healthPollTimer);
    state.healthPollTimer = window.setInterval(() => {
        void fetchHealth();
    }, 2500);
    void fetchHealth();

    const conversationsPromise = loadConversations();
    const workspacesPromise = loadWorkspaces();

    await conversationsPromise;
    await workspacesPromise;

    if (state.conversations.length) {
        await loadConversation(state.conversations[0].id);
    } else {
        startNewChat();
    }
}

void bootstrap();
