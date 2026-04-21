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
    label?: string;
    phase?: string;
    step_label?: string;
    permission_key?: string;
    approval_target?: string;
    payload?: {
        path?: string;
        open_path?: string;
    };
}

interface ThinkingStatus {
    text: string;
    phase: string;
    error: boolean;
    persistent: boolean;
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
    fixture_coverage: {
        bucket_key: string;
        total_fixtures: number;
        suggested_count: number;
        candidate_count: number;
        accepted_count: number;
        superseded_count: number;
        fixtures: Array<{
            name: string;
            path: string;
            review_status: string;
        }>;
    };
    promotion_suggestion: {
        should_suggest: boolean;
        reason: string;
        suggested_review_status: string;
    };
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
    created?: boolean;
    fixture_name: string;
    fixture_path: string;
    source_path: string;
    review_status: string;
}

interface ContextEvalFixtureRecord {
    name: string;
    path: string;
    review_status: string;
    promoted_at: string;
    updated_at: string;
    source_path: string;
    source_trigger: string;
    source_conversation_id: string;
}

interface ContextEvalFixtureListResponse {
    fixtures: ContextEvalFixtureRecord[];
}

interface ContextEvalFixtureDetailRecord extends ContextEvalFixtureRecord {
    payload: Record<string, unknown>;
}

interface ContextEvalFixtureDetailResponse extends ContextEvalFixtureDetailRecord {}

interface ContextEvalFixtureReviewResponse {
    status: string;
    fixture_name: string;
    fixture_path: string;
    review_status: string;
}

type ConnectionState = "offline" | "online" | "streaming";
type ViewerMode = "closed" | "tree" | "file";

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
    viewerMode: "closed" as ViewerMode,
    contextEvalReport: null as ContextEvalReport | null,
    contextEvalLoading: false,
    contextEvalError: "",
    selectedContextEvalBucketKey: "",
    selectedContextEvalExampleIndex: 0,
    contextEvalFixtures: [] as ContextEvalFixtureRecord[],
    fixtureReviewLoading: false,
    fixtureReviewFilter: "all",
    selectedFixturePath: "",
    compareFixturePath: "",
    selectedFixtureDetail: null as ContextEvalFixtureDetailRecord | null,
    compareFixtureDetail: null as ContextEvalFixtureDetailRecord | null,
    fixtureDetailLoading: false,
    thinkingStatus: null as ThinkingStatus | null,
    stickChatToBottom: true,
    handledArtifactKeys: new Set<string>(),
    conversationRefreshTimer: 0 as number,
    healthPollTimer: 0 as number,
    pendingAttachmentPaths: [] as string[],
};

function query<T extends Element>(selector: string): T {
    const element = document.querySelector<T>(selector);
    if (!element) {
        throw new Error(`Missing required element: ${selector}`);
    }
    return element;
}

const sidebarToggle = query<HTMLButtonElement>("#sidebarToggle");
const viewerToggle = query<HTMLButtonElement>("#viewerToggle");
const refreshWorkspaceButton = query<HTMLButtonElement>("#refreshWorkspaceButton");
const workspaceSettingsButton = query<HTMLButtonElement>("#workspaceSettingsButton");
const refreshContextEvalButton = query<HTMLButtonElement>("#refreshContextEvalButton");
const newChatButton = query<HTMLButtonElement>("#newChatButton");
const conversationList = query<HTMLDivElement>("#conversationList");
const workspaceName = query<HTMLDivElement>("#workspaceName");
const connectionBadge = query<HTMLSpanElement>("#connectionBadge");
const chatMessages = query<HTMLDivElement>("#chatMessages");
const scrollToBottomButton = query<HTMLButtonElement>("#scrollToBottomButton");
const composerForm = query<HTMLFormElement>("#composerForm");
const composerInput = query<HTMLTextAreaElement>("#composerInput");
const composerHint = query<HTMLSpanElement>("#composerHint");
const sendButton = query<HTMLButtonElement>("#sendButton");
const viewerTitle = query<HTMLHeadingElement>("#viewerTitle");
const viewerMeta = query<HTMLParagraphElement>("#viewerMeta");
const viewerModeButton = query<HTMLButtonElement>("#viewerModeButton");
const upDirectoryButton = query<HTMLButtonElement>("#upDirectoryButton");
const viewerCloseButton = query<HTMLButtonElement>("#viewerCloseButton");
const directoryPath = query<HTMLDivElement>("#directoryPath");
const fileList = query<HTMLDivElement>("#fileList");
const filePreview = query<HTMLDivElement>("#filePreview");
const contextEvalReport = query<HTMLDivElement>("#contextEvalReport");
const settingsOverlay = query<HTMLDivElement>("#settingsOverlay");
const closeSettingsButton = query<HTMLButtonElement>("#closeSettingsButton");
const settingsSummary = query<HTMLDivElement>("#settingsSummary");
const fixtureReviewFilter = query<HTMLSelectElement>("#fixtureReviewFilter");
const refreshFixtureReviewButton = query<HTMLButtonElement>("#refreshFixtureReviewButton");
const fixtureReviewList = query<HTMLDivElement>("#fixtureReviewList");
const fixtureReviewDetail = query<HTMLDivElement>("#fixtureReviewDetail");
const resetAppButton = query<HTMLButtonElement>("#resetAppButton");

const PASTE_ATTACH_CHAR_THRESHOLD = 1200;
const PASTE_ATTACH_LINE_THRESHOLD = 28;
const FIRST_TURN_COMPOSER_MAX_HEIGHT = 260;
const LATER_TURN_COMPOSER_MAX_HEIGHT = 152;

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

function describeFixtureCoverage(coverage?: ContextEvalTriageBucket["fixture_coverage"] | null): string {
    if (coverage?.accepted_count) return "covered by accepted fixture";
    if (coverage?.candidate_count) return "candidate fixture exists";
    if (coverage?.suggested_count) return "suggested fixture queued";
    return "needs promotion";
}

function renderFixtureReviewList(): void {
    if (state.fixtureReviewLoading) {
        fixtureReviewList.innerHTML = `<div class="context-eval-empty">Loading fixtures…</div>`;
        return;
    }

    const selectedFilter = state.fixtureReviewFilter || "all";
    const fixtures = (state.contextEvalFixtures || []).filter((fixture) => (
        selectedFilter === "all" ? true : fixture.review_status === selectedFilter
    ));
    if (!fixtures.length) {
        fixtureReviewList.innerHTML = `<div class="context-eval-empty">No fixtures for this filter yet.</div>`;
        return;
    }

    fixtureReviewList.innerHTML = fixtures.map((fixture) => {
        const fixturePath = workspaceRelativeCapturePath(fixture.path) || fixture.path;
        const sourcePath = workspaceRelativeCapturePath(fixture.source_path) || fixture.source_path;
        return `
            <article class="fixture-review-item${state.selectedFixturePath === fixture.path ? " active" : ""}">
                <div class="context-eval-focus">
                    <strong>${escapeHtml(fixture.name || "Unnamed fixture")}</strong>
                    <span class="context-eval-severity ${escapeHtml(fixture.review_status || "candidate")}">${escapeHtml(fixture.review_status || "candidate")}</span>
                </div>
                <div class="context-eval-meta">
                    <span>${escapeHtml(fixture.source_trigger || "unknown trigger")}</span>
                    <span>${escapeHtml(fixture.updated_at || fixture.promoted_at || "")}</span>
                </div>
                <div class="fixture-review-paths">
                    <code>${escapeHtml(fixturePath)}</code>
                    ${sourcePath ? `<code>${escapeHtml(sourcePath)}</code>` : ""}
                </div>
                <div class="context-eval-actions">
                    <button type="button" class="context-eval-button" data-fixture-select="${escapeHtml(fixture.path)}">
                        ${state.selectedFixturePath === fixture.path ? "Selected" : "Inspect"}
                    </button>
                    <button type="button" class="context-eval-button" data-fixture-open="${escapeHtml(fixturePath)}">
                        Open
                    </button>
                    <button type="button" class="context-eval-button" data-fixture-status="suggested" data-fixture-path="${escapeHtml(fixture.path)}">
                        Suggested
                    </button>
                    <button type="button" class="context-eval-button" data-fixture-status="candidate" data-fixture-path="${escapeHtml(fixture.path)}">
                        Candidate
                    </button>
                    <button type="button" class="context-eval-button" data-fixture-status="accepted" data-fixture-path="${escapeHtml(fixture.path)}">
                        Accepted
                    </button>
                    <button type="button" class="context-eval-button" data-fixture-status="superseded" data-fixture-path="${escapeHtml(fixture.path)}">
                        Supersede
                    </button>
                </div>
            </article>
        `;
    }).join("");

    fixtureReviewList.querySelectorAll<HTMLButtonElement>("[data-fixture-select]").forEach((button) => {
        button.addEventListener("click", async () => {
            const fixturePath = button.dataset.fixtureSelect || "";
            if (!fixturePath) return;
            state.selectedFixturePath = fixturePath;
            if (state.compareFixturePath === fixturePath) {
                state.compareFixturePath = "";
                state.compareFixtureDetail = null;
            }
            renderFixtureReviewList();
            await loadSelectedFixtureDetails();
        });
    });
    fixtureReviewList.querySelectorAll<HTMLButtonElement>("[data-fixture-open]").forEach((button) => {
        button.addEventListener("click", () => {
            const path = button.dataset.fixtureOpen || "";
            if (!path || path.startsWith("/")) return;
            void openFile(path);
        });
    });
    fixtureReviewList.querySelectorAll<HTMLButtonElement>("[data-fixture-status]").forEach((button) => {
        button.addEventListener("click", async () => {
            const fixturePath = button.dataset.fixturePath || "";
            const reviewStatus = button.dataset.fixtureStatus || "";
            if (!fixturePath || !reviewStatus) return;
            try {
                const response = await fetchJson<ContextEvalFixtureReviewResponse>("/api/context-evals/fixtures/review", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        fixture_path: fixturePath,
                        review_status: reviewStatus,
                    }),
                });
                setComposerHint(`Updated ${response.fixture_name} to ${response.review_status}.`);
                await loadContextEvalFixtures();
            } catch (error) {
                setComposerHint(error instanceof Error ? error.message : "Could not update fixture review state.");
            }
        });
    });
}

function renderFixtureDetailValue(value: unknown): string {
    if (Array.isArray(value)) {
        if (!value.length) return `<div class="context-eval-empty">None</div>`;
        return `<ul class="context-eval-detail-list">${value.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul>`;
    }
    if (value && typeof value === "object") {
        const entries = Object.entries(value as Record<string, unknown>);
        if (!entries.length) return `<div class="context-eval-empty">None</div>`;
        return `
            <div class="fixture-detail-kv-list">
                ${entries.map(([key, itemValue]) => `
                    <div class="fixture-detail-kv-row">
                        <span class="fixture-detail-kv-key">${escapeHtml(key)}</span>
                        <code>${escapeHtml(JSON.stringify(itemValue))}</code>
                    </div>
                `).join("")}
            </div>
        `;
    }
    if (value === undefined || value === null || value === "") {
        return `<div class="context-eval-empty">None</div>`;
    }
    return `<code>${escapeHtml(String(value))}</code>`;
}

function renderFixtureComparison(
    selectedFixture: ContextEvalFixtureDetailRecord,
    compareFixture: ContextEvalFixtureDetailRecord | null,
): string {
    if (!compareFixture) {
        return `<div class="context-eval-empty">Choose a comparison fixture to inspect differences.</div>`;
    }
    const selectedExpectation = JSON.stringify(selectedFixture.payload.expectation || {});
    const compareExpectation = JSON.stringify(compareFixture.payload.expectation || {});
    const selectedCandidates = JSON.stringify(selectedFixture.payload.selection_candidates || []);
    const compareCandidates = JSON.stringify(compareFixture.payload.selection_candidates || []);
    const differences = [
        selectedFixture.review_status !== compareFixture.review_status
            ? `Review status: ${selectedFixture.review_status} vs ${compareFixture.review_status}`
            : "",
        selectedExpectation !== compareExpectation
            ? "Expectation payload differs"
            : "",
        selectedCandidates !== compareCandidates
            ? "Selection candidates differ"
            : "",
        JSON.stringify(selectedFixture.payload.policy_inputs || {}) !== JSON.stringify(compareFixture.payload.policy_inputs || {})
            ? "Policy inputs differ"
            : "",
    ].filter(Boolean);
    return differences.length
        ? `<ul class="context-eval-detail-list">${differences.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
        : `<div class="context-eval-empty">These fixtures currently have matching core payload sections.</div>`;
}

function renderFixtureReviewDetail(): void {
    if (state.fixtureDetailLoading) {
        fixtureReviewDetail.innerHTML = `<div class="context-eval-empty">Loading fixture detail…</div>`;
        return;
    }

    const selectedFixture = state.selectedFixtureDetail;
    if (!selectedFixture) {
        fixtureReviewDetail.innerHTML = `<div class="context-eval-empty">Select a fixture to inspect its full payload and compare it with another reviewed case.</div>`;
        return;
    }

    const compareOptions = (state.contextEvalFixtures || [])
        .filter((fixture) => fixture.path !== selectedFixture.path)
        .map((fixture) => `
            <option value="${escapeHtml(fixture.path)}"${fixture.path === state.compareFixturePath ? " selected" : ""}>
                ${escapeHtml(`${fixture.name} (${fixture.review_status})`)}
            </option>
        `)
        .join("");

    fixtureReviewDetail.innerHTML = `
        <div class="context-eval-card">
            <div class="context-eval-label">Fixture Detail</div>
            <div class="context-eval-focus">
                <strong>${escapeHtml(selectedFixture.name)}</strong>
                <span class="context-eval-severity ${escapeHtml(selectedFixture.review_status)}">${escapeHtml(selectedFixture.review_status)}</span>
            </div>
            <div class="context-eval-meta">
                <span>${escapeHtml(selectedFixture.source_trigger || "unknown trigger")}</span>
                <span>${escapeHtml(selectedFixture.updated_at || selectedFixture.promoted_at || "")}</span>
            </div>
            <div class="fixture-review-paths">
                <code>${escapeHtml(workspaceRelativeCapturePath(selectedFixture.path) || selectedFixture.path)}</code>
                ${selectedFixture.source_path ? `<code>${escapeHtml(workspaceRelativeCapturePath(selectedFixture.source_path) || selectedFixture.source_path)}</code>` : ""}
            </div>
        </div>
        <div class="context-eval-card">
            <div class="settings-section-header">
                <strong>Compare</strong>
                <div class="settings-filter-row">
                    <select id="fixtureCompareSelect" aria-label="Compare fixture">
                        <option value="">No comparison</option>
                        ${compareOptions}
                    </select>
                </div>
            </div>
            ${renderFixtureComparison(selectedFixture, state.compareFixtureDetail)}
        </div>
        <div class="context-eval-card">
            <div class="context-eval-label">Policy Inputs</div>
            ${renderFixtureDetailValue(selectedFixture.payload.policy_inputs)}
        </div>
        <div class="context-eval-card">
            <div class="context-eval-label">Expectation</div>
            ${renderFixtureDetailValue(selectedFixture.payload.expectation)}
        </div>
        <div class="context-eval-card">
            <div class="context-eval-label">Selection Candidates</div>
            ${renderFixtureDetailValue(selectedFixture.payload.selection_candidates)}
        </div>
    `;

    const compareSelect = fixtureReviewDetail.querySelector<HTMLSelectElement>("#fixtureCompareSelect");
    if (compareSelect) {
        compareSelect.addEventListener("change", async () => {
            state.compareFixturePath = compareSelect.value || "";
            await loadSelectedFixtureDetails();
        });
    }
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

function currentWorkspaceName(): string {
    const workspace = state.workspaces.find((entry) => entry.id === state.currentWorkspaceId);
    if (!workspace) return "";
    const displayName = String(workspace.display_name || "").trim();
    const genericNames = new Set(["workspace", "project", "root"]);
    if (displayName && !genericNames.has(displayName.toLowerCase())) {
        return displayName;
    }
    const rootPath = String(workspace.root_path || "").trim().replace(/\/+$/, "");
    if (!rootPath) return "";
    const segments = rootPath.split("/").filter(Boolean);
    const leafName = segments.at(-1) || rootPath;
    return genericNames.has(leafName.toLowerCase()) ? "" : leafName;
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
    const showModeButton = Boolean(state.selectedFilePath) && state.viewerMode === "file";
    viewerModeButton.hidden = !showModeButton;
    if (showModeButton) {
        const viewerModeLabel = "Back to file list";
        viewerModeButton.setAttribute("aria-label", viewerModeLabel);
        viewerModeButton.dataset.tooltip = viewerModeLabel;
    }

    upDirectoryButton.hidden = state.viewerMode !== "tree";
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
    setConnectionState(nextValue ? "streaming" : (state.ws?.readyState === WebSocket.OPEN ? "online" : "offline"));
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

function displayConversationTitle(title: string, preview: string): string {
    const cleanedTitle = cleanConversationText(title);
    const cleanedPreview = cleanConversationText(preview);
    const base = cleanedTitle && cleanedTitle.toLowerCase() !== "untitled chat"
        ? cleanedTitle
        : (cleanedPreview || "New chat");
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
    await fetchJson(`/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            path,
            content: text,
        }),
    });
    state.pendingAttachmentPaths = [...state.pendingAttachmentPaths, path];
    await loadDirectory(parentDirectory(path));
    await openFile(path);
    return path;
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

function unwrapSingleFence(raw: string): { language: string; body: string } | null {
    const trimmed = String(raw || "").trim();
    const match = trimmed.match(/^```([^\n`]*)\n([\s\S]*)\n```$/);
    if (!match) return null;
    return {
        language: normalizeCodeLanguage(match[1] || ""),
        body: String(match[2] || "").replace(/\r\n?/g, "\n"),
    };
}

function stripArtifactHelperLines(text: string): string {
    return String(text || "")
        .split(/\r?\n/)
        .filter((line) => !ARTIFACT_HELPER_TEXT_PATTERNS.some((pattern) => pattern.test(line.trim())))
        .join("\n");
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
    const displayParts = [before];
    if (!after) {
        displayParts.push(cleaned.replace(/\[\[artifact:[^\]]+\]\]/gi, "").trim());
    }
    return {
        artifactPaths,
        displayContent: displayParts.filter(Boolean).join("\n\n").trim(),
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
    const fence = unwrapSingleFence(trimmed);
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

    const primaryPath = parsed.artifactPaths[0];
    const artifactKey = [
        state.currentConversationId,
        primaryPath,
        parsed.materializedBody.length,
        parsed.materializedBody.slice(0, 120),
    ].join(":");
    if (state.handledArtifactKeys.has(artifactKey)) return;
    state.handledArtifactKeys.add(artifactKey);

    void (async () => {
        try {
            if (shouldMaterializeArtifact(primaryPath, parsed.materializedBody)) {
                await materializeArtifactFromAssistant(primaryPath, parsed.materializedBody);
                return;
            }
            await openFile(primaryPath);
        } catch (_error) {
            state.handledArtifactKeys.delete(artifactKey);
        }
    })();
}

function renderInlineMarkdown(text: string): string {
    return text
        .replace(/\[\[artifact:([^\]]+)\]\]/gi, (_match, rawPath: string) => {
            const path = resolveArtifactReferencePath(rawPath);
            const label = path.split("/").pop() || path;
            return `<button type="button" class="artifact-inline-ref" data-artifact-path="${escapeHtml(path)}">${escapeHtml(label)}</button>`;
        })
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
    if (["sql", "sqlite", "postgresql", "postgres", "mysql"].includes(normalized)) return "sql";
    if (["json"].includes(normalized)) return "json";
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

            if (lines.every((line) => /^[-*_]{3,}$/.test(line.trim()))) {
                return "<hr>";
            }

            if (lines.every((line) => /^&gt;\s?/.test(line))) {
                const quote = lines
                    .map((line) => line.replace(/^&gt;\s?/, ""))
                    .join("<br>");
                return `<blockquote>${renderInlineMarkdown(quote)}</blockquote>`;
            }

            if (lines.every((line) => /^- /.test(line))) {
                return `<ul>${lines.map((line) => `<li>${renderInlineMarkdown(line.replace(/^- /, ""))}</li>`).join("")}</ul>`;
            }

            if (lines.every((line) => /^\d+\. /.test(line))) {
                return `<ol>${lines.map((line) => `<li>${renderInlineMarkdown(line.replace(/^\d+\. /, ""))}</li>`).join("")}</ol>`;
            }

            const heading = lines[0].match(/^(#{1,4})\s+(.*)$/);
            if (heading) {
                const level = Math.min(heading[1].length, 4);
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

        return `
            <article class="message ${role}">
                <div class="message-role">${escapeHtml(meta)}</div>
                ${renderArtifactReferences(parsedArtifacts.artifactPaths)}
                <div class="message-content rich-text">${renderRichText(parsedArtifacts.displayContent || "")}</div>
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
        const visibleTitle = displayConversationTitle(conversation.title || "", conversation.last_message || "");
        const visiblePreview = displayConversationPreview(conversation.title || "", conversation.last_message || "");
        const visibleTime = formatRelativeTime(conversation.updated_at) || "Just now";
        return `
            <div class="conversation-row${conversation.id === state.currentConversationId ? " active" : ""}">
                <button
                    type="button"
                    class="conversation-item${conversation.id === state.currentConversationId ? " active" : ""}"
                    data-conversation-id="${escapeHtml(conversation.id)}"
                >
                    <div class="conversation-line">
                        <span class="conversation-title">${escapeHtml(visibleTitle)}</span>
                        <span class="conversation-preview">${escapeHtml(visiblePreview)}</span>
                        <span class="conversation-time">${escapeHtml(visibleTime)}</span>
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
    const label = currentWorkspaceName();
    workspaceName.textContent = label;
    workspaceName.hidden = !label;
    syncShellLayout();
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
    const needsPromotion = Boolean(
        selectedBucket
        && !selectedBucket.fixture_coverage?.accepted_count
        && !selectedBucket.fixture_coverage?.suggested_count
        && !selectedBucket.fixture_coverage?.candidate_count
    );
    const suggestedFixtureName = selectedBucket && selectedExample
        ? suggestContextEvalFixtureName(selectedBucket, selectedExample)
        : "";
    const promotionSuggestion = selectedBucket?.promotion_suggestion || null;
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
                ${recommended.promotion_suggestion?.should_suggest ? `
                    <div class="context-eval-example context-eval-suggestion">
                        <div class="context-eval-label">Suggested Promotion</div>
                        <p>${escapeHtml(recommended.promotion_suggestion.reason)}</p>
                    </div>
                ` : ""}
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
                                <span>${describeFixtureCoverage(bucket.fixture_coverage)}</span>
                                ${bucket.promotion_suggestion?.should_suggest ? "<span>suggested candidate fixture</span>" : ""}
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
                <div class="context-eval-meta">
                    <span>${selectedBucket.fixture_coverage?.accepted_count ? "Accepted fixture exists" : "No accepted fixture yet"}</span>
                    ${selectedBucket.fixture_coverage?.suggested_count ? `<span>${escapeHtml(String(selectedBucket.fixture_coverage.suggested_count))} suggested fixture(s)</span>` : ""}
                    ${selectedBucket.fixture_coverage?.candidate_count ? `<span>${escapeHtml(String(selectedBucket.fixture_coverage.candidate_count))} candidate fixture(s)</span>` : ""}
                    ${selectedBucket.fixture_coverage?.superseded_count ? `<span>${escapeHtml(String(selectedBucket.fixture_coverage.superseded_count))} superseded</span>` : ""}
                </div>
                ${selectedBucket.fixture_coverage?.fixtures?.length ? `
                    <div class="context-eval-example">
                        <div class="context-eval-label">Fixture Coverage</div>
                        <div class="context-eval-list">
                            ${selectedBucket.fixture_coverage.fixtures.map((fixture) => `
                                <article class="context-eval-item">
                                    <strong>${escapeHtml(fixture.name)}</strong>
                                    <div class="context-eval-meta">
                                        <span>${escapeHtml(fixture.review_status)}</span>
                                    </div>
                                </article>
                            `).join("")}
                        </div>
                    </div>
                ` : ""}
                ${promotionSuggestion?.should_suggest ? `
                    <div class="context-eval-example context-eval-suggestion">
                        <div class="context-eval-label">Suggested Promotion</div>
                        <p>${escapeHtml(promotionSuggestion.reason)}</p>
                        <div class="context-eval-meta">
                            <span>Draft name: ${escapeHtml(suggestedFixtureName || selectedExample.name)}</span>
                            <span>${escapeHtml(promotionSuggestion.suggested_review_status)}</span>
                        </div>
                    </div>
                ` : ""}
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
                    ${needsPromotion ? `
                        <button
                            type="button"
                            class="context-eval-button active"
                            data-context-eval-promote-candidate="true"
                        >
                            Promote as Candidate
                        </button>
                    ` : ""}
                </div>
                ${selectedBucket.fixture_coverage?.fixtures?.length ? `
                    <div class="context-eval-actions">
                        ${selectedBucket.fixture_coverage.fixtures.map((fixture) => `
                            <button
                                type="button"
                                class="context-eval-button"
                                data-context-eval-open-fixture="${escapeHtml(fixture.path)}"
                            >
                                Inspect ${escapeHtml(fixture.review_status)} fixture
                            </button>
                        `).join("")}
                    </div>
                ` : ""}
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
            void openFile(selectedCapturePath);
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
            const reviewStatus = window.prompt("Review status (suggested, candidate, accepted, superseded)", "candidate");
            if (!reviewStatus || !reviewStatus.trim()) return;
            try {
                const response = await promoteContextEvalCapture(
                    selectedExample.source_path,
                    fixtureName.trim(),
                    reviewStatus.trim().toLowerCase(),
                );
                setComposerHint(`Promoted replay case as ${response.fixture_name} (${response.review_status}).`);
            } catch (error) {
                setComposerHint(error instanceof Error ? error.message : "Could not promote replay case.");
            }
        });
    });
    contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-promote-candidate]").forEach((button) => {
        button.addEventListener("click", async () => {
            if (!selectedExample?.source_path || !selectedBucket) return;
            const fixtureName = suggestContextEvalFixtureName(selectedBucket, selectedExample);
            try {
                const response = await promoteContextEvalCapture(
                    selectedExample.source_path,
                    fixtureName,
                    "candidate",
                );
                setComposerHint(`Promoted ${response.fixture_name} as candidate.`);
            } catch (error) {
                setComposerHint(error instanceof Error ? error.message : "Could not promote candidate fixture.");
            }
        });
    });
    contextEvalReport.querySelectorAll<HTMLButtonElement>("[data-context-eval-open-fixture]").forEach((button) => {
        button.addEventListener("click", async () => {
            const fixturePath = button.dataset.contextEvalOpenFixture || "";
            if (!fixturePath) return;
            await inspectFixtureFromTriage(fixturePath);
        });
    });
}

async function promoteContextEvalCapture(
    sourcePath: string,
    fixtureName: string,
    reviewStatus: string,
): Promise<ContextEvalPromotionResponse> {
    const response = await fetchJson<ContextEvalPromotionResponse>("/api/context-evals/promote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            source_path: sourcePath,
            fixture_name: fixtureName,
            review_status: reviewStatus,
        }),
    });
    await loadContextEvalFixtures();
    state.selectedFixturePath = response.fixture_path;
    await loadSelectedFixtureDetails();
    await loadContextEvalReport();
    return response;
}

async function autoDraftContextEvalFixture(
    sourcePath: string,
    bucketKey: string,
    fixtureName: string,
): Promise<ContextEvalPromotionResponse> {
    return fetchJson<ContextEvalPromotionResponse>("/api/context-evals/auto-draft", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            source_path: sourcePath,
            bucket_key: bucketKey,
            fixture_name: fixtureName,
        }),
    });
}

async function inspectFixtureFromTriage(fixturePath: string): Promise<void> {
    state.selectedFixturePath = fixturePath;
    state.compareFixturePath = "";
    await loadContextEvalFixtures();
    await loadSelectedFixtureDetails();
    showSettings();
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
    viewerMeta.hidden = false;

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

    filePreview.innerHTML = `<pre class="preview-code">${escapeHtml(payload.content || "")}</pre>`;
}

function closeViewer(): void {
    setViewerMode("closed");
}

function openFileTree(): void {
    if (!state.currentWorkspaceId) return;
    setViewerMode("tree");
    renderFileList();
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
    } else {
        state.selectedFilePath = "";
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
    const reportUrl = `/api/context-evals/report?${params.toString()}`;

    try {
        let report = await fetchJson<ContextEvalReport>(reportUrl);
        const suggestionTargets = (report.top_triage_buckets || [])
            .filter((bucket) => (
                bucket.promotion_suggestion?.should_suggest
                && !bucket.fixture_coverage?.accepted_count
                && !bucket.fixture_coverage?.suggested_count
                && !bucket.fixture_coverage?.candidate_count
            ))
            .slice(0, 3);
        let createdAnySuggestions = false;
        for (const bucket of suggestionTargets) {
            const exampleCase = bucket.example_cases?.[0];
            if (!exampleCase?.source_path) continue;
            const response = await autoDraftContextEvalFixture(
                exampleCase.source_path,
                bucket.key,
                "",
            );
            createdAnySuggestions = Boolean(response.created) || createdAnySuggestions;
        }
        if (createdAnySuggestions) {
            report = await fetchJson<ContextEvalReport>(reportUrl);
            if (!settingsOverlay.hidden) {
                await loadContextEvalFixtures();
            }
        }
        state.contextEvalReport = report;
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

async function loadContextEvalFixtures(): Promise<void> {
    state.fixtureReviewLoading = true;
    renderFixtureReviewList();
    try {
        const payload = await fetchJson<ContextEvalFixtureListResponse>("/api/context-evals/fixtures");
        state.contextEvalFixtures = payload.fixtures || [];
        const selectedStillExists = state.contextEvalFixtures.some((fixture) => fixture.path === state.selectedFixturePath);
        if (!selectedStillExists) {
            state.selectedFixturePath = state.contextEvalFixtures[0]?.path || "";
            state.compareFixturePath = "";
            state.selectedFixtureDetail = null;
            state.compareFixtureDetail = null;
        }
    } catch (error) {
        state.contextEvalFixtures = [];
        setComposerHint(error instanceof Error ? error.message : "Could not load promoted fixtures.");
    } finally {
        state.fixtureReviewLoading = false;
        renderFixtureReviewList();
        void loadSelectedFixtureDetails();
    }
}

async function loadSelectedFixtureDetails(): Promise<void> {
    if (!state.selectedFixturePath) {
        state.selectedFixtureDetail = null;
        state.compareFixtureDetail = null;
        state.fixtureDetailLoading = false;
        renderFixtureReviewDetail();
        return;
    }

    state.fixtureDetailLoading = true;
    renderFixtureReviewDetail();
    try {
        state.selectedFixtureDetail = await fetchJson<ContextEvalFixtureDetailResponse>(
            `/api/context-evals/fixtures/detail?fixture_path=${encodeURIComponent(state.selectedFixturePath)}`
        );
        if (state.compareFixturePath) {
            state.compareFixtureDetail = await fetchJson<ContextEvalFixtureDetailResponse>(
                `/api/context-evals/fixtures/detail?fixture_path=${encodeURIComponent(state.compareFixturePath)}`
            );
        } else {
            state.compareFixtureDetail = null;
        }
    } catch (error) {
        state.selectedFixtureDetail = null;
        state.compareFixtureDetail = null;
        setComposerHint(error instanceof Error ? error.message : "Could not load fixture detail.");
    } finally {
        state.fixtureDetailLoading = false;
        renderFixtureReviewDetail();
    }
}

async function loadConversation(id: string): Promise<void> {
    const payload = await fetchJson<{
        messages: ChatMessage[];
        workspace_id?: string;
    }>(`/api/conversation/${encodeURIComponent(id)}`);

    state.currentConversationId = id;
    state.messages = payload.messages || [];
    state.stickChatToBottom = true;
    state.activeAssistantIndex = -1;
    state.thinkingStatus = null;
    state.selectedFilePath = "";
    state.viewerMode = "closed";

    const matchingConversation = state.conversations.find((conversation) => conversation.id === id);
    state.currentConversationTitle = matchingConversation?.title || "New chat";

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
    state.pendingAttachmentPaths = [];
    state.stickChatToBottom = true;
    state.activeAssistantIndex = -1;
    state.thinkingStatus = null;
    state.selectedFilePath = "";
    state.viewerMode = "closed";
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

    if (event.type === "start") {
        ensureAssistantMessage();
        setGenerating(true);
        setThinkingStatus(buildThinkingSummary(event));
        return;
    }

    if (event.type === "assistant_note") {
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

    if ((event.type === "activity" || event.type === "reasoning_note" || event.type === "status") && event.content) {
        setThinkingStatus(buildThinkingSummary(event), { phase: event.phase || "thinking" });
        return;
    }

    if (event.type === "tool_result" && event.payload?.open_path) {
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
            setThinkingStatus(buildThinkingSummary(event), { phase: "permission" });
        } else {
            setThinkingStatus(buildThinkingSummary(event), { phase: "permission", error: true, persistent: true });
        }
        return;
    }

    if (event.type === "error") {
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
    const attachments = [...state.pendingAttachmentPaths];
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
    state.pendingAttachmentPaths = [];
    syncComposerHeight();
    renderMessages();
    renderConversations();
    setGenerating(true);
    setThinkingStatus(state.modelAvailable ? "Getting started." : "Getting started while the model finishes loading.");

    const payload = {
        message,
        conversation_id: state.currentConversationId,
        workspace_id: state.currentWorkspaceId || null,
        attachments,
        mode: "deep",
        turn_kind: "visible_chat",
        auto_approve_tool_permissions: true,
    };

    try {
        await dispatchChatPayload(payload);
    } catch (error) {
        state.pendingAttachmentPaths = attachments;
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
    }>(`/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/files?path=${encodeURIComponent(path)}`);

    state.currentDirectoryPath = payload.path || ".";
    state.fileItems = payload.items || [];
    renderFileList();
}

async function openFile(path: string): Promise<void> {
    if (!state.currentWorkspaceId) return;

    const samePath = state.selectedFilePath === path;
    const previousPreviewScrollTop = samePath ? filePreview.scrollTop : 0;
    state.selectedFilePath = path;
    state.viewerMode = "file";
    syncShellLayout();
    renderFileList();

    try {
        const payload = await fetchJson<WorkspaceFilePayload>(
            `/api/workspaces/${encodeURIComponent(state.currentWorkspaceId)}/file?path=${encodeURIComponent(path)}`
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
        void loadDirectory(state.currentDirectoryPath);
        if (state.viewerMode === "file" && state.selectedFilePath) void openFile(state.selectedFilePath);
    });

    sidebarToggle.addEventListener("click", () => {
        state.leftSidebarOpen = !state.leftSidebarOpen;
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

    upDirectoryButton.addEventListener("click", () => {
        void loadDirectory(parentDirectory(state.currentDirectoryPath));
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
    syncComposerHeight();
    connectWebSocket();
    window.clearInterval(state.healthPollTimer);
    state.healthPollTimer = window.setInterval(() => {
        void fetchHealth();
    }, 2500);
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
