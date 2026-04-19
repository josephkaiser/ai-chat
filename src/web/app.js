// AI Chat Application

// ==================== State ====================

let ws = null;
let currentConvId = generateId();
let currentWorkspaceId = localStorage.getItem('lastWorkspaceId') || '';
let workspaceCatalog = [];
let currentWorkspaceMeta = null;
let isGenerating = false;
let activeStreamConversationId = null;
let streamingAssistantMessage = null;
let renameConvId = null;
let modelAvailable = false;
let runtimeAvailabilityStatus = 'loading';
let markedReady = false;
let healthPollInterval = null;
let pastedBlocks = []; // tracks collapsed long pastes: [{placeholder, actual}]
const BASE_REASONING_MODE = 'deep';
let deepMode = true;
let chatCommandAllowlists = loadStoredObject('chatCommandAllowlists', {});
let chatToolPermissionAllowlists = loadStoredObject('chatToolPermissionAllowlists', {});
let chatToolAutoApproveSettings = loadStoredObject('chatToolAutoApproveSettings', {});
let conversationTurnFeatureMemory = loadStoredObject('conversationTurnFeatures', {});
let websocketConnected = false;
let currentTurnTransport = null;
let httpTurnAbortController = null;
let workspaceEntries = [];
let workspaceTree = null;
let workspaceStats = { files: 0, directories: 0 };
let workspaceActivity = [];
let buildSteps = [];
let currentBuildStepIndex = null;
let currentPlanFocus = null;
let selectedWorkspaceFile = '';
let selectedSpreadsheetSheet = '';
let workspaceRefreshTimer = null;
let workspacePanelOpen = localStorage.getItem('workspacePanelOpen') !== 'false';
let workspaceConsoleOpen = localStorage.getItem('workspaceConsoleOpen') === 'true';
let workspaceViewMode = localStorage.getItem('workspaceViewMode') === 'reader' ? 'reader' : 'tree';
let collapsedWorkspaceDirs = new Set();
let featureSettings = null;
let voiceSettings = null;
let pendingAttachments = [];
let draftFileSessions = [];
let draftSessionDefaultPath = '';
let inlineViewerPath = '';
let inlineViewerKind = 'text';
let inlineViewerEditable = false;
let inlineViewerView = 'preview';
let inlineEditor = null;
let inlineEditorReady = false;
let inlineViewerAutosaveTimer = null;
let inlineViewerLastSavedContent = '';
let inlineViewerSaveState = 'idle';
let inlineViewerUndoStacks = {};
let inlineViewerApplyingRemoteContent = false;
let inlineViewerPerformingUndo = false;
let inlineViewerRequestToken = 0;
let activeDraftPath = '';
let activeDraftProfileKey = 'html';
let draftOutputPreviewPath = '';
let draftEditor = null;
let draftEditorReady = false;
let draftAutosaveTimer = null;
let draftLastSavedContent = '';
let draftSaveState = 'idle';
let draftView = localStorage.getItem('draftView') === 'preview' ? 'preview' : 'edit';
let draftApplyingRemoteContent = false;
let draftOpenRequestToken = 0;
let draftAgentSyncTimer = null;
let draftPendingAgentSync = false;
let draftLastAgentSpecContent = '';
let activeDraftGeneratedContent = '';
let activeDraftForegroundJobId = '';
let currentFileSymbols = [];
let symbolGroupState = {};
let currentAssistantTurnStartedAt = null;
let currentAssistantTurnArtifactPaths = new Set();
let latestAssistantTurnArtifactPaths = new Set();
let slashMenuItems = [];
let slashMenuSelectedIndex = 0;
let welcomeHintTimer = null;
let transientHintTimer = null;
let currentRunId = null;
let pendingExecutionPlan = null;
let pendingPermissionRequest = null;
let activeClientTurnId = '';
let dictationActive = false;
let currentAudio = null;
let speechQueue = [];
let speechQueueBusy = false;
let activeSpeechTask = null;
let speechQueueVersion = 0;
let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let audioAttachmentUploadInFlight = false;
let dictationStartedAt = 0;
let mobileComposerResizeObserver = null;
let mobileKeyboardInset = 0;
const MOBILE_WORKSPACE_MEDIA_QUERY = '(max-width: 768px)';
const COMPACT_WORKSPACE_MEDIA_QUERY = '(max-width: 1180px)';
const DEFAULT_THEME = 'dark';
const DEFAULT_WORKSPACE_NAME = 'Workspace';
const SEND_BUTTON_ICON = buildComposerIconMarkup('<path d="M4.5 19.5 19.5 12 4.5 4.5l2.75 6.25L14 12l-6.75 1.25L4.5 19.5Z"></path>');
const STOP_BUTTON_ICON = buildComposerIconMarkup('<rect x="7.25" y="7.25" width="9.5" height="9.5" rx="1.8"></rect>');
const MIC_BUTTON_ICON = buildComposerIconMarkup('<rect x="9" y="4.5" width="6" height="10" rx="3"></rect><path d="M6.5 11.5a5.5 5.5 0 0 0 11 0"></path><path d="M12 17v2.5"></path><path d="M9.5 19.5h5"></path>');
const MIC_ACTIVE_BUTTON_ICON = buildComposerIconMarkup('<circle cx="12" cy="12" r="7.5"></circle><rect x="9.2" y="9.2" width="5.6" height="5.6" rx="1.2"></rect>');
const MAX_PENDING_ATTACHMENTS = 8;
const MIN_CHAT_SURFACE_WIDTH = 520;
const CHAT_SURFACE_GAP_PX = 14;
const ARTIFACT_REFERENCE_PATTERN = /\[\[artifact:([^[\]\r\n]+?)\]\]/g;
const ARTIFACT_HELPER_TEXT_PATTERNS = [
    /^\s*(?:you can )?(?:open|view|inspect|see|read|preview|find)\b[^.!?]*?(?:here|below)?[:.]?\s*$/i,
    /^\s*(?:saved plan(?: and notes)?|saved progress|task board|feedback digest|full feedback digest|artifact|artifacts?)[:.]?\s*$/i,
];
let voiceRuntime = {
    tts_available: false,
    stt_available: false,
    tts_backend: 'unavailable',
    stt_backend: 'unavailable',
    tts_voice: '',
};
let runtimeHealth = null;
const SPEECH_SPEED_OPTIONS = Object.freeze([
    { value: '0', label: 'Low', rate: 0.85 },
    { value: '1', label: 'Low-Med', rate: 0.95 },
    { value: '2', label: 'Medium', rate: 1.0 },
    { value: '3', label: 'Med-Fast', rate: 1.12 },
    { value: '4', label: 'Fast', rate: 1.5 },
    { value: '5', label: 'Mid-High', rate: 2.0 },
    { value: '6', label: 'High', rate: 3.0 },
]);
const WELCOME_HINT_ROTATE_MS = 120000;
const LOADING_HINT_ROTATE_MS = 120000;
const PLAN_STEP_LIMIT = 4;
const INLINE_ACTIVITY_LIMIT = 4;
const DRAFT_AUTOSAVE_DELAY_MS = 900;
const DRAFT_AGENT_SYNC_DELAY_MS = 1400;
const DEFAULT_DRAFT_FILENAME = 'index.html';
const DOCUMENT_CENTERED_MODE = true;
const DRAFT_PROFILE_DEFINITIONS = Object.freeze({
    html: Object.freeze({
        label: 'HTML',
        extension: 'html',
        kind: 'html',
        lineWrapping: true,
        defaultView: 'edit',
        starter: filename => `Build the file \`${basename(filename || 'index.html') || 'index.html'}\`.

Goal:
- Describe what this page should do.

Content:
- List the main sections and what each one should say.

Style:
- Describe the mood, layout, typography, and colors you want.

Notes:
- Add any semantic HTML, responsiveness, or interaction wishes here.
`,
    }),
    markdown: Object.freeze({
        label: 'Markdown',
        extension: 'md',
        kind: 'markdown',
        lineWrapping: true,
        defaultView: 'edit',
        starter: filename => `Write the file \`${basename(filename || 'notes.md') || 'notes.md'}\`.\n\nGoal:\n- Describe what this document should communicate.\n\nStructure:\n- List the headings or sections you want.\n\nTone:\n- Describe the writing style and level of detail.\n`,
    }),
    python: Object.freeze({
        label: 'Python',
        extension: 'py',
        kind: 'text',
        lineWrapping: false,
        defaultView: 'edit',
        starter: filename => `Build the Python file \`${basename(filename || 'app.py') || 'app.py'}\`.\n\nPurpose:\n- Describe what the script or module should do.\n\nInputs and outputs:\n- Describe the important functions, classes, or CLI behavior.\n\nConstraints:\n- Mention libraries, style preferences, or edge cases to handle.\n`,
    }),
    json: Object.freeze({
        label: 'JSON',
        extension: 'json',
        kind: 'text',
        lineWrapping: false,
        defaultView: 'edit',
        starter: filename => `Build the JSON file \`${basename(filename || 'data.json') || 'data.json'}\`.\n\nSchema ideas:\n- Describe the keys and nested structure you want.\n\nRules:\n- Mention any required values, formats, or constraints.\n`,
    }),
    text: Object.freeze({
        label: 'Text',
        extension: 'txt',
        kind: 'text',
        lineWrapping: true,
        defaultView: 'edit',
        starter: filename => `Create the file \`${basename(filename || 'notes.txt') || 'notes.txt'}\`.\n\nUse this draft as a wish list, outline, or rough brief for what the generated file should become.\n`,
    }),
});
const DISCOVERY_HINTS = Object.freeze([
    'Try Python! Ask me to build a small script, explain it, and verify it.',
    'Try Python! Say: "Make a quick script for this and keep it in the workspace."',
    'Need a package? Try `/pip pandas matplotlib requests`.',
    'Need fresh docs or package ideas? Ask me to search the web first.',
    'Try: "Build this in Python and explain the tradeoffs as you go."',
    'I can analyze pasted text, notes, logs, screenshots, and attached files.',
    'Prompt idea: "Summarize this and pull out the key decisions or risks."',
    'Try: "Turn this rough idea into a clear plan, checklist, or draft."',
    'Prompt idea: "Question my assumptions and point out what I may be missing."',
    'When useful, I can inspect files, run checks, install Python packages, and verify results in the workspace.',
]);

// Must match src/python/ai_chat/thinking_stream.py THINK_TAG_PAIRS (redacted_thinking + think)
const THINK_TAG_PAIRS = Object.freeze([
    ['<' + 'redacted_thinking' + '>', '</' + 'redacted_thinking' + '>'],
    ['<' + 'think' + '>', '</' + 'think' + '>'],
]);

const SLASH_COMMAND_KIND_LABELS = Object.freeze({
    action: 'Action',
    prompt: 'Prompt',
    toggle: 'Toggle',
    tool: 'Tool',
});

const DIRECT_SLASH_COMMANDS = Object.freeze({
    search: Object.freeze({
        canonical: 'search',
        aliases: ['search', 'web'],
        label: '/search',
        template: '/search ',
    }),
    grep: Object.freeze({
        canonical: 'grep',
        aliases: ['grep'],
        label: '/grep',
        template: '/grep ',
    }),
    plan: Object.freeze({
        canonical: 'plan',
        aliases: ['plan'],
        label: '/plan',
        template: '/plan ',
    }),
    code: Object.freeze({
        canonical: 'code',
        aliases: ['code', 'edit'],
        label: '/code',
        template: '/code ',
    }),
    pip: Object.freeze({
        canonical: 'pip',
        aliases: ['pip'],
        label: '/pip',
        template: '/pip ',
    }),
});

const WORKSPACE_ACTIVITY_LIMIT = 80;
const WORKSPACE_ARTIFACT_LIMIT = 6;
const INLINE_VIEWER_AUTOSAVE_DELAY_MS = 700;
const INLINE_VIEWER_UNDO_LIMIT = 40;
const LIVE_AREA_READ_ONLY = true;
const DEFAULT_INLINE_VIEWER_EMPTY_TEXT = 'Select an artifact or file to preview it here.';
const MESSAGE_SCROLL_BOTTOM_THRESHOLD = 96;
const PLAN_EXECUTION_MARKERS = Object.freeze([
    'execute approved plan',
    'execute this approved plan',
    'execute the approved plan',
    'execute plan now',
    'execute this plan',
    'run this plan',
    'run the plan',
]);
const PLAN_APPROVAL_REPLY_MARKERS = Object.freeze([
    'yes',
    'yes please',
    'yes do it',
    'yes do that',
    'yes go ahead',
    'yes start',
    'yep',
    'sure',
    'ok',
    'okay',
    'approve',
    'approve this plan',
    'approve the plan',
    'approve and run',
    'run',
    'run it',
    'run this plan',
    'execute',
    'execute it',
    'execute this plan',
    'go ahead',
    'go ahead please',
    'go ahead and run it',
    'do it',
    'please do',
    'sounds good',
    'looks good',
    'that works',
    'lets do it',
    "let's do it",
    'start',
    'start it',
    'start with step 1',
    'start with the first step',
]);
const RESUME_TURN_MARKERS = Object.freeze([
    'continue',
    'resume',
    'keep going',
    'keep working',
    'carry on',
    'go on',
    'finish that',
    'finish it',
    'next',
    'next step',
    'do the next step',
    'move to the next step',
]);
const APPROVED_PLAN_EXECUTION_MESSAGE = 'yes';
const MESSAGE_FEEDBACK_OPTIONS = Object.freeze([
    Object.freeze({
        value: 'positive',
        label: 'Good',
        icon: '&#128077;',
        title: 'Mark this response as helpful.',
    }),
    Object.freeze({
        value: 'negative',
        label: 'Bad',
        icon: '&#128078;',
        title: 'Mark this response as unhelpful.',
    }),
]);
const MESSAGE_FEEDBACK_OPTION_MAP = Object.freeze(
    Object.fromEntries(MESSAGE_FEEDBACK_OPTIONS.map(option => [option.value, option]))
);

featureSettings = loadFeatureSettings();
voiceSettings = loadVoiceSettings();

// ==================== Theme ====================

function loadStoredObject(key, fallback = {}) {
    try {
        const raw = localStorage.getItem(key);
        if (!raw) return { ...fallback };
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' ? parsed : { ...fallback };
    } catch (error) {
        console.warn(`Failed to load ${key}:`, error);
        return { ...fallback };
    }
}

function persistStoredObject(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value || {}));
    } catch (error) {
        console.warn(`Failed to persist ${key}:`, error);
    }
}

function buildComposerIconMarkup(paths, viewBox = '0 0 24 24') {
    return `<span class="composer-icon" aria-hidden="true"><svg viewBox="${viewBox}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${paths}</svg></span>`;
}

function isMobileViewport() {
    return window.matchMedia(MOBILE_WORKSPACE_MEDIA_QUERY).matches;
}

function updateThemeChrome(mode, colors = {}) {
    const meta = document.getElementById('themeColorMeta');
    if (!meta) return;
    const fallback = mode === 'dark' ? '#111827' : '#f5f0e8';
    meta.setAttribute('content', colors.bg_secondary || colors.bg_primary || fallback);
}

function applyTheme(mode) {
    const themes = window.THEMES || {};
    const colors = themes[mode] || themes.light || {};
    const root = document.documentElement;
    root.dataset.theme = mode;
    for (const [key, value] of Object.entries(colors)) {
        root.style.setProperty('--' + key, value);
    }
    localStorage.setItem('theme', mode);

    // Update highlight.js theme
    const hljsLink = document.getElementById('hljs-theme');
    if (hljsLink) {
        hljsLink.href = mode === 'dark'
            ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css'
            : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
    }

    // Update theme icon
    const icon = document.getElementById('themeIcon');
    if (icon) icon.textContent = mode === 'dark' ? 'Dark' : 'Light';

    updateThemeChrome(mode, colors);
}

function toggleTheme() {
    const current = localStorage.getItem('theme') || DEFAULT_THEME;
    applyTheme(current === 'light' ? 'dark' : 'light');
    // Re-render markdown for new theme
    document.querySelectorAll('.message.assistant[data-rendered="true"]').forEach(msg => {
        msg.dataset.needsMarkdown = 'true';
        msg.dataset.rendered = 'false';
    });
    renderMarkdown();
}

// Apply theme immediately
applyTheme(localStorage.getItem('theme') || DEFAULT_THEME);

// ==================== Marked Init ====================

function initMarked() {
    if (typeof marked !== 'undefined') {
        markedReady = true;
        marked.setOptions({
            breaks: true, gfm: true, headerIds: false, mangle: false,
            pedantic: false, sanitize: false, smartLists: true, smartypants: true
        });
        renderMarkdown();
    } else {
        setTimeout(initMarked, 50);
    }
}

function renderMathContent(container) {
    if (!container || typeof renderMathInElement !== 'function') return;
    try {
        renderMathInElement(container, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false },
            ],
            throwOnError: false,
            strict: 'ignore',
            trust: false,
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
        });
    } catch (e) {
        console.warn('Failed to render math content', e);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMarked);
} else {
    initMarked();
}

// ==================== Utilities ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeHtmlTextForTemplate(text) {
    return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function generateId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

function syncReasoningSelector() {
    const select = document.getElementById('reasoningSelect');
    if (select) {
        select.value = 'base';
        select.title = 'Base reasoning mode uses the high-thinking path by default.';
    }
    syncReasoningToggleButton();
}

function handleReasoningSelectChange(value) {
    deepMode = true;
    syncReasoningSelector();
    syncMobileReasoningBadge();
}

function toggleDeepModeMobile() {
    deepMode = true;
    syncReasoningSelector();
    syncMobileReasoningBadge();
}

function syncMobileReasoningBadge() {
    const badge = document.getElementById('mobileReasoningBadge');
    if (!badge) return;
    badge.hidden = true;
    badge.textContent = 'Base';
    badge.classList.add('is-high');
}

function syncReasoningToggleButton() {
    const button = document.getElementById('reasoningToggleButton');
    if (!button) return;
    button.hidden = true;
    button.classList.add('is-active');
    button.dataset.mode = 'base';
    button.title = 'Base reasoning mode uses the high-thinking path by default.';
    button.setAttribute('aria-label', 'Reasoning mode: base');
}

function dismissMobileKeyboard(force = false) {
    if (!isMobileViewport()) return;
    const input = document.getElementById('input');
    if (!input) return;
    if (!force && document.activeElement !== input) return;
    input.blur();
}

function syncMobileViewportState() {
    const root = document.documentElement;
    const inputArea = document.getElementById('inputArea');
    const mobile = isMobileViewport();
    if (!mobile) {
        mobileKeyboardInset = 0;
        root.classList.remove('mobile-keyboard-open');
        root.style.setProperty('--mobile-keyboard-gap', '0px');
        root.style.setProperty('--mobile-viewport-height', `${window.innerHeight || document.documentElement.clientHeight || 0}px`);
        root.style.setProperty('--mobile-composer-height', inputArea ? `${Math.ceil(inputArea.getBoundingClientRect().height)}px` : '132px');
        return;
    }

    const viewport = window.visualViewport;
    const viewportHeight = Math.round(viewport?.height || window.innerHeight || document.documentElement.clientHeight || 0);
    const viewportOffsetTop = Math.round(viewport?.offsetTop || 0);
    const layoutHeight = Math.max(
        window.innerHeight || 0,
        document.documentElement.clientHeight || 0,
        viewportHeight + viewportOffsetTop,
    );
    const keyboardInset = Math.max(0, layoutHeight - viewportHeight - viewportOffsetTop);
    mobileKeyboardInset = keyboardInset;
    root.style.setProperty('--mobile-keyboard-gap', `${keyboardInset}px`);
    root.style.setProperty('--mobile-viewport-height', `${viewportHeight}px`);
    root.style.setProperty('--mobile-composer-height', inputArea ? `${Math.ceil(inputArea.getBoundingClientRect().height)}px` : '132px');
    root.classList.toggle('mobile-keyboard-open', keyboardInset > 120);
}

function observeMobileComposerSize() {
    const inputArea = document.getElementById('inputArea');
    if (!inputArea) return;
    if (mobileComposerResizeObserver) mobileComposerResizeObserver.disconnect();
    if (typeof ResizeObserver === 'undefined') {
        syncMobileViewportState();
        return;
    }
    mobileComposerResizeObserver = new ResizeObserver(() => {
        syncMobileViewportState();
        positionSlashMenu();
    });
    mobileComposerResizeObserver.observe(inputArea);
    syncMobileViewportState();
}

function loadFeatureSettings() {
    localStorage.removeItem('feature.workspace_panel');
    return {
        agent_tools: true,
        local_rag: true,
        web_search: true,
    };
}

function loadVoiceSettings() {
    return {
        autoSpeakReplies: localStorage.getItem('voice.autoSpeakReplies') === 'true',
        speechSpeed: normalizeSpeechSpeed(localStorage.getItem('voice.speechSpeed')),
    };
}

function normalizeSpeechSpeed(value) {
    const normalized = String(value ?? '2');
    return SPEECH_SPEED_OPTIONS.some(option => option.value === normalized) ? normalized : '2';
}

function getSpeechSpeedOption(value = voiceSettings.speechSpeed) {
    const normalized = normalizeSpeechSpeed(value);
    return SPEECH_SPEED_OPTIONS.find(option => option.value === normalized) || SPEECH_SPEED_OPTIONS[2];
}

function persistFeatureSettings() {
    Object.entries(featureSettings).forEach(([key, value]) => {
        localStorage.setItem(`feature.${key}`, value ? 'true' : 'false');
    });
    localStorage.removeItem('feature.workspace_panel');
}

function persistVoiceSettings() {
    Object.entries(voiceSettings).forEach(([key, value]) => {
        if (typeof value === 'boolean') {
            localStorage.setItem(`voice.${key}`, value ? 'true' : 'false');
            return;
        }
        localStorage.setItem(`voice.${key}`, String(value));
    });
}

function formatTimestamp(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const msgDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    const timeStr = date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    if (msgDate.getTime() === today.getTime()) return timeStr;
    const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1);
    if (msgDate.getTime() === yesterday.getTime()) return `Yesterday ${timeStr}`;
    return `${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${timeStr}`;
}

function formatFullTimestamp(isoString) {
    const date = new Date(isoString);
    return `${date.toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' })} ${date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })}`;
}

// ==================== Status ====================

function renderStatusDot(status) {
    const dot = document.getElementById('statusDot');
    if (!dot) return;

    if (status === 'connected') {
        dot.className = 'status-dot connected';
    } else if (status === 'loading') {
        dot.className = 'status-dot loading';
    } else {
        dot.className = 'status-dot';
    }
}

function syncStatusIndicator() {
    renderStatusDot(modelAvailable ? 'connected' : runtimeAvailabilityStatus);
}

function updateStatus(status) {
    runtimeAvailabilityStatus = status === 'connected' ? 'connected' : (status === 'loading' ? 'loading' : 'disconnected');
    modelAvailable = status === 'connected';
    renderStatusDot(status);
    renderSettingsRuntimeStatus();
    syncSendButton();
    if (modelAvailable && draftPendingAgentSync) {
        draftPendingAgentSync = false;
        scheduleDraftAgentSync({ immediate: true });
    }
}

function setWebsocketConnected(connected) {
    websocketConnected = Boolean(connected);
    syncStatusIndicator();
    renderSettingsRuntimeStatus();
    syncSendButton();
}

function setLoadingText(text) {
    const loadingText = document.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = text;
}

function canUseWebsocketTransport() {
    return Boolean(ws && ws.readyState === WebSocket.OPEN);
}

function buildHttpFallbackPayload(payload = {}) {
    const nextPayload = { ...payload };
    if (!nextPayload.file_path && activeDraftPath) {
        nextPayload.file_path = normalizeDraftFilename(activeDraftPath);
    }
    if (nextPayload.features && typeof nextPayload.features === 'object') {
        nextPayload.features = {
            ...nextPayload.features,
            workspace_run_commands: false,
            allowed_commands: [],
        };
    }
    return nextPayload;
}

function syncChatShellLayout() {
    const shell = document.querySelector('.chat-shell');
    if (!shell) return;

    const mobileViewport = window.matchMedia(MOBILE_WORKSPACE_MEDIA_QUERY).matches;
    if (mobileViewport) {
        shell.style.setProperty('--chat-main-inset-left', '0px');
        shell.style.setProperty('--chat-main-inset-right', '0px');
        shell.classList.remove('menu-open', 'chrome-shifted');
        return;
    }

    const shellRect = shell.getBoundingClientRect();
    if (!shellRect.width) return;

    let desiredLeft = 0;
    let desiredRight = 0;

    const menuOverlay = document.getElementById('menuOverlay');
    const menuPanel = menuOverlay?.querySelector('.menu-panel');
    const menuOpen = Boolean(menuOverlay?.classList.contains('show') && menuPanel);
    if (menuOpen && menuPanel) {
        const menuRect = menuPanel.getBoundingClientRect();
        desiredLeft = Math.max(0, menuRect.right - shellRect.left + CHAT_SURFACE_GAP_PX);
    }

    const fileExplorerOverlay = document.getElementById('fileExplorerOverlay');
    const fileExplorerPanel = fileExplorerOverlay?.querySelector('.file-explorer-panel');
    const fileExplorerOpen = Boolean(fileExplorerOverlay?.classList.contains('show') && fileExplorerPanel);
    if (fileExplorerOpen && fileExplorerPanel) {
        const explorerRect = fileExplorerPanel.getBoundingClientRect();
        desiredRight = Math.max(desiredRight, shellRect.right - explorerRect.left + CHAT_SURFACE_GAP_PX);
    }

    const ideWorkspace = document.getElementById('ideWorkspace');
    const workspaceOpen = Boolean(shell.classList.contains('ide-mode') && ideWorkspace);
    if (workspaceOpen && ideWorkspace) {
        const visibleWorkspaceRects = [];
        const workspacePanel = document.getElementById('workspacePanel');
        const inlineViewer = ideWorkspace.querySelector('.ide-viewer:not(.is-hidden)');

        if (workspacePanel && workspacePanel.offsetParent !== null) {
            visibleWorkspaceRects.push(workspacePanel.getBoundingClientRect());
        }
        if (inlineViewer && inlineViewer instanceof HTMLElement && inlineViewer.offsetParent !== null) {
            visibleWorkspaceRects.push(inlineViewer.getBoundingClientRect());
        }

        if (visibleWorkspaceRects.length) {
            const chromeLeftEdge = Math.min(...visibleWorkspaceRects.map(rect => rect.left));
            desiredRight = Math.max(0, shellRect.right - chromeLeftEdge + CHAT_SURFACE_GAP_PX);
        }
    }

    const totalDesiredInset = desiredLeft + desiredRight;
    const maxInset = Math.max(0, shellRect.width - MIN_CHAT_SURFACE_WIDTH);
    if (totalDesiredInset > maxInset && totalDesiredInset > 0) {
        const scale = maxInset / totalDesiredInset;
        desiredLeft *= scale;
        desiredRight *= scale;
    }

    shell.style.setProperty('--chat-main-inset-left', `${Math.round(desiredLeft)}px`);
    shell.style.setProperty('--chat-main-inset-right', `${Math.round(desiredRight)}px`);
    shell.classList.toggle('menu-open', menuOpen);
    shell.classList.toggle('chrome-shifted', desiredLeft > 0 || desiredRight > 0);
}

function applyWorkspacePanelState() {
    const panel = document.getElementById('workspacePanel');
    const toggle = document.getElementById('workspaceToggle');
    const shell = document.querySelector('.chat-shell');
    const root = document.getElementById('chatRoot');
    if (!panel || !toggle) return;
    const agentToolsEnabled = featureSettings.agent_tools;
    const mobileViewport = workspaceUsesMobileLayout();
    const workspaceAllowed = agentToolsEnabled;
    if (!workspaceAllowed) {
        workspacePanelOpen = false;
    }
    toggle.style.display = workspaceAllowed ? '' : 'none';
    const open = workspaceAllowed && workspacePanelOpen;
    panel.classList.toggle('is-open', open);
    toggle.classList.toggle('active', open);
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (shell) shell.classList.toggle('ide-mode', open);
    if (shell) shell.classList.toggle('workspace-open', open);
    if (root) {
        root.classList.toggle('mobile-chat-mode', mobileViewport);
        root.classList.toggle('workspace-mobile-open', mobileViewport && open);
    }
    localStorage.setItem('workspacePanelOpen', open ? 'true' : 'false');
    applyWorkspaceConsoleState();
    syncWorkspaceViewMode();
    syncChatShellLayout();
}

function workspaceUsesMobileLayout() {
    return window.matchMedia(MOBILE_WORKSPACE_MEDIA_QUERY).matches
        || window.matchMedia(COMPACT_WORKSPACE_MEDIA_QUERY).matches;
}

function effectiveWorkspaceViewMode() {
    if (workspaceViewMode === 'reader' && !inlineViewerPath) return 'tree';
    return workspaceViewMode === 'reader' ? 'reader' : 'tree';
}

function syncWorkspaceViewButtons(mode = effectiveWorkspaceViewMode(), hasReader = Boolean(inlineViewerPath)) {
    const treeToggle = document.getElementById('workspaceTreeContentToggle');
    const viewerToggle = document.getElementById('inlineViewerTreeToggle');

    if (treeToggle) {
        treeToggle.textContent = 'Content';
        treeToggle.disabled = !hasReader;
        treeToggle.title = hasReader ? 'Show the selected file' : 'Open a file to enable the content viewer';
        treeToggle.setAttribute('aria-label', hasReader ? 'Show content viewer' : 'Open a file to enable the content viewer');
    }

    if (viewerToggle) {
        viewerToggle.textContent = 'Files';
        viewerToggle.disabled = false;
        viewerToggle.title = 'Show the workspace tree';
        viewerToggle.setAttribute('aria-label', 'Show workspace tree');
    }
}

function syncWorkspaceViewMode() {
    const root = document.getElementById('chatRoot');
    const panel = document.getElementById('workspacePanel');
    const viewer = document.querySelector('.ide-viewer');
    const workspace = document.getElementById('ideWorkspace');
    const mobileLayout = workspaceUsesMobileLayout();
    const mode = effectiveWorkspaceViewMode();
    const hasReader = Boolean(inlineViewerPath);
    const showTree = mode === 'tree';
    const showReader = hasReader && mode === 'reader';

    workspaceViewMode = mode;
    if (root) root.classList.toggle('workspace-reader-open', !mobileLayout && showReader);
    if (panel) panel.classList.toggle('is-hidden', !showTree);
    if (viewer) viewer.classList.toggle('is-hidden', !showReader);
    if (workspace) {
        workspace.classList.toggle('is-tree-mode', showTree);
        workspace.classList.toggle('is-reader-mode', showReader);
    }
    syncWorkspaceViewButtons(mode, hasReader);

    localStorage.setItem('workspaceViewMode', mode);
}

function setWorkspaceViewMode(mode) {
    const nextMode = mode === 'reader' ? 'reader' : 'tree';
    if (nextMode === 'reader' && !inlineViewerPath) return;
    workspaceViewMode = nextMode;
    syncWorkspaceViewMode();
    syncChatShellLayout();
}

function applyWorkspaceConsoleState() {
    const drawer = document.getElementById('workspaceActivityDrawer');
    const toggle = document.getElementById('workspaceConsoleToggle');
    if (!drawer || !toggle) return;
    const allowed = Boolean(featureSettings?.agent_tools);
    const open = allowed && workspaceConsoleOpen;
    drawer.hidden = !open;
    toggle.hidden = !allowed;
    toggle.classList.toggle('active', open);
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    toggle.title = open ? 'Hide activity log' : 'Show activity log';
    toggle.setAttribute('aria-label', open ? 'Hide activity log' : 'Show activity log');
    localStorage.setItem('workspaceConsoleOpen', workspaceConsoleOpen ? 'true' : 'false');
    if (open) renderWorkspaceActivity();
}

function toggleWorkspaceConsole() {
    workspaceConsoleOpen = !workspaceConsoleOpen;
    applyWorkspaceConsoleState();
}

function closeWorkspacePanel() {
    if (!workspacePanelOpen) return;
    if (inlineViewerPath) {
        closeInlineViewer();
    }
    workspacePanelOpen = false;
    applyWorkspacePanelState();
}

function toggleWorkspacePanel() {
    if (!featureSettings.agent_tools) return;
    dismissMobileKeyboard(true);
    if (!workspacePanelOpen) closeMenu();
    workspacePanelOpen = !workspacePanelOpen;
    applyWorkspacePanelState();
    if (workspacePanelOpen) refreshWorkspace(true);
}

function syncFeatureControls() {
    const agentTools = document.getElementById('settingAgentTools');
    const localRag = document.getElementById('settingLocalRag');
    const webSearch = document.getElementById('settingWebSearch');
    if (agentTools) agentTools.checked = featureSettings.agent_tools;
    if (localRag) localRag.checked = featureSettings.local_rag;
    if (webSearch) webSearch.checked = featureSettings.web_search;
    const autoSpeak = document.getElementById('settingAutoSpeak');
    if (autoSpeak) {
        autoSpeak.checked = voiceSettings.autoSpeakReplies;
        autoSpeak.disabled = !supportsSpeechSynthesis();
    }
    const speechSpeed = document.getElementById('settingSpeechSpeed');
    const speechSpeedReadout = document.getElementById('speechSpeedReadout');
    const speedOption = getSpeechSpeedOption();
    if (speechSpeed) {
        speechSpeed.value = speedOption.value;
        speechSpeed.disabled = !supportsSpeechSynthesis();
    }
    if (speechSpeedReadout) {
        speechSpeedReadout.textContent = speedOption.label;
    }
}

function applyFeatureSettingsToUI() {
    if (!featureSettings.agent_tools) {
        workspacePanelOpen = false;
    }
    if (!featureSettings.agent_tools) {
        if (workspaceRefreshTimer) {
            clearTimeout(workspaceRefreshTimer);
            workspaceRefreshTimer = null;
        }
    }
    applyWorkspacePanelState();
    syncFeatureControls();
}

function updateFeatureSetting(name, enabled) {
    if (!(name in featureSettings)) return;
    featureSettings[name] = enabled;
    persistFeatureSettings();
    applyFeatureSettingsToUI();
    if (name === 'agent_tools') refreshWorkspace(true);
}

async function downloadWorkspaceZip() {
    if (!currentWorkspaceId) return;
    try {
        const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/download`);
        if (resp.status === 204) {
            showTransientComposerHint('The workspace is empty, so there is nothing to download.', 5000);
            recordWorkspaceActivity('Download', 'Skipped workspace download because the workspace is empty.');
            return;
        }
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
        }
        const blob = await resp.blob();
        if (!blob.size) {
            showTransientComposerHint('The workspace is empty, so there is nothing to download.', 5000);
            recordWorkspaceActivity('Download', 'Skipped workspace download because the archive was empty.');
            return;
        }
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'workspace.zip';
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
        recordWorkspaceActivity('Download', 'Downloaded workspace.zip.');
    } catch (error) {
        showTransientComposerHint(`Workspace download failed: ${error.message}`, 6000);
        recordWorkspaceActivity('Download Error', `Workspace download failed: ${error.message}`, { error: true });
    }
}

function downloadWorkspaceFile(path) {
    if (!currentWorkspaceId || !path) return;
    const params = new URLSearchParams({ path });
    window.open(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file/download?${params.toString()}`, '_blank', 'noopener');
}

function downloadActiveGeneratedFile() {
    if (!activeDraftPath) return;
    downloadWorkspaceFile(activeDraftPath);
}

async function copyActiveGeneratedFile() {
    if (!activeDraftPath) return;
    try {
        const content = await readWorkspaceTextFile(activeDraftPath);
        if (!content) {
            showTransientComposerHint('There is no generated file content to copy yet.', 5000);
            return;
        }
        copyToClipboard(content);
        showTransientComposerHint(`Copied ${basename(activeDraftPath)}.`, 4000);
    } catch (error) {
        showTransientComposerHint(`Copy failed: ${error.message}`, 5000);
    }
}

function closeDraftVersionsModal() {
    document.getElementById('draftVersionsModal')?.classList.remove('show');
}

async function openDraftVersionsModal() {
    const modal = document.getElementById('draftVersionsModal');
    const currentEl = document.getElementById('draftVersionsCurrentFile');
    const listEl = document.getElementById('draftVersionsList');
    if (!modal || !currentEl || !listEl) return;

    const targetPath = normalizeDraftFilename(activeDraftPath || DEFAULT_DRAFT_FILENAME);
    currentEl.textContent = `Target file: ${targetPath}`;
    listEl.innerHTML = '<div class="workspace-empty">Loading versions...</div>';
    modal.classList.add('show');

    if (!currentWorkspaceId || !targetPath) {
        listEl.innerHTML = '<div class="workspace-empty">Choose a workspace and draft first.</div>';
        return;
    }

    try {
        const currentContent = await readWorkspaceTextFile(targetPath);
        const listing = await fetchWorkspaceDirectory(draftVersionDirectoryForTargetPath(targetPath)).catch(() => ({ items: [] }));
        const items = Array.isArray(listing.items) ? listing.items.filter(item => item.type === 'file') : [];
        const sortedItems = items.slice().sort((a, b) => String(b.modified_at || '').localeCompare(String(a.modified_at || '')));
        const rows = [];
        rows.push({
            title: 'Current generated file',
            path: targetPath,
            downloadPath: targetPath,
            modifiedAt: '',
            empty: !currentContent,
        });
        sortedItems.forEach(item => {
            rows.push({
                title: new Date(item.modified_at || Date.now()).toLocaleString(),
                path: item.path,
                downloadPath: item.path,
                modifiedAt: item.modified_at || '',
                empty: false,
            });
        });

        listEl.innerHTML = '';
        if (!rows.length) {
            listEl.innerHTML = '<div class="workspace-empty">No saved versions yet.</div>';
            return;
        }

        rows.forEach(row => {
            const item = document.createElement('div');
            item.className = 'draft-version-item';
            item.innerHTML = `
                <div class="draft-version-copy">
                    <div class="draft-version-title">${escapeHtml(row.title)}</div>
                    <div class="draft-version-path">${escapeHtml(row.path)}</div>
                </div>
                <div class="draft-version-actions">
                    <button type="button"${row.empty ? ' disabled' : ''}>Download</button>
                </div>
            `;
            const button = item.querySelector('button');
            if (button && !row.empty) {
                button.onclick = () => downloadWorkspaceFile(row.downloadPath);
            }
            listEl.appendChild(item);
        });
    } catch (error) {
        listEl.innerHTML = `<div class="workspace-empty">Could not load version history: ${escapeHtml(error.message)}</div>`;
    }
}

function workspaceFileInlineViewUrl(path) {
    if (!currentWorkspaceId || !path) return '';
    const params = new URLSearchParams({ path });
    return `/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file/view?${params.toString()}`;
}

function inferTurnPermissions(message, attachmentPaths = []) {
    const text = String(message || '').toLowerCase();
    const words = new Set((text.match(/[a-z0-9_+-]+/g) || []));
    const writeTerms = new Set([
        'create', 'build', 'generate', 'scaffold', 'template', 'starter', 'example',
        'sample', 'repo', 'repository', 'app', 'project', 'saas', 'mvp', 'boilerplate',
        'skeleton', 'write', 'edit', 'update', 'patch', 'implement', 'fix',
    ]);
    const runTerms = new Set([
        'run', 'test', 'verify', 'check', 'execute', 'start', 'launch', 'install',
        'lint', 'compile', 'build',
    ]);
    const hasAttachment = Array.isArray(attachmentPaths) && attachmentPaths.length > 0;
    const wantsWrite = [...writeTerms].some(term => words.has(term))
        || /make .*app|create .*app|build .*app|starter .*project|example .*project/.test(text)
        || (hasAttachment && [...words].some(term => ['fix', 'edit', 'update', 'patch'].includes(term)));
    const wantsRun = [...runTerms].some(term => words.has(term))
        || /try it|make sure it works|verify it works|test it/.test(text);
    return { wantsWrite, wantsRun };
}

function normalizeTurnMessage(message) {
    return String(message || '').trim().toLowerCase().replace(/\s+/g, ' ');
}

function normalizeApprovalReply(message) {
    return normalizeTurnMessage(message).replace(/[.!?,;:]+/g, ' ').replace(/\s+/g, ' ').trim();
}

function messageRequestsPlanExecution(message, slashCommand = null) {
    const text = normalizeTurnMessage(message);
    const approvalText = normalizeApprovalReply(message);
    if (!text) return false;
    if (PLAN_EXECUTION_MARKERS.some(marker => text.includes(marker))) return true;

    const slashName = slashCommand?.name || '';
    const slashArgs = normalizeApprovalReply(slashCommand?.args || '');
    if (slashName === 'plan' && PLAN_APPROVAL_REPLY_MARKERS.includes(slashArgs)) {
        return true;
    }

    return Boolean(pendingExecutionPlan?.executePrompt || buildSteps.length) && PLAN_APPROVAL_REPLY_MARKERS.includes(approvalText);
}

function messageRequestsContinuation(message, slashCommand = null) {
    if (messageRequestsPlanExecution(message, slashCommand)) return true;
    const approvalText = normalizeApprovalReply(message);
    if (!approvalText) return false;
    return RESUME_TURN_MARKERS.includes(approvalText);
}

function messageRequestsWebSearch(message) {
    const text = normalizeTurnMessage(message);
    if (!text) return false;
    return [
        'search the web',
        'search web',
        'search online',
        'use search',
        'look up',
        'browse',
        'google',
        'find sources',
        'add citations',
        'cite sources',
        'with citations',
        'with sources',
    ].some(phrase => text.includes(phrase));
}

function messageRequestsHistoryLookup(message) {
    const text = normalizeTurnMessage(message);
    if (!text) return false;
    return [
        'search history',
        'search the history',
        'search our chat',
        'search this chat',
        'conversation history',
        'earlier in this chat',
        'what did we say',
        'what did i say',
    ].some(phrase => text.includes(phrase));
}

function getAllowedCommandsForConversation(conversationId) {
    const key = String(conversationId || '').trim();
    const values = Array.isArray(chatCommandAllowlists[key]) ? chatCommandAllowlists[key] : [];
    return [...new Set(values.map(value => String(value || '').trim().toLowerCase()).filter(Boolean))];
}

function rememberAllowedCommand(conversationId, commandKey) {
    const key = String(conversationId || '').trim();
    const normalized = String(commandKey || '').trim().toLowerCase();
    if (!key || !normalized) return;
    chatCommandAllowlists[key] = getAllowedCommandsForConversation(key).concat(normalized)
        .filter((value, index, items) => value && items.indexOf(value) === index);
    persistStoredObject('chatCommandAllowlists', chatCommandAllowlists);
}

function clearAllowedCommandsForConversation(conversationId) {
    const key = String(conversationId || '').trim();
    if (!key) return;
    delete chatCommandAllowlists[key];
    persistStoredObject('chatCommandAllowlists', chatCommandAllowlists);
}

function getAllowedToolPermissionsForConversation(conversationId) {
    const key = String(conversationId || '').trim();
    const values = Array.isArray(chatToolPermissionAllowlists[key]) ? chatToolPermissionAllowlists[key] : [];
    return [...new Set(values.map(value => String(value || '').trim().toLowerCase()).filter(Boolean))];
}

function rememberAllowedToolPermission(conversationId, permissionKey) {
    const key = String(conversationId || '').trim();
    const normalized = String(permissionKey || '').trim().toLowerCase();
    if (!key || !normalized) return;
    chatToolPermissionAllowlists[key] = getAllowedToolPermissionsForConversation(key).concat(normalized)
        .filter((value, index, items) => value && items.indexOf(value) === index);
    persistStoredObject('chatToolPermissionAllowlists', chatToolPermissionAllowlists);
}

function clearAllowedToolPermissionsForConversation(conversationId) {
    const key = String(conversationId || '').trim();
    if (!key) return;
    delete chatToolPermissionAllowlists[key];
    persistStoredObject('chatToolPermissionAllowlists', chatToolPermissionAllowlists);
}

function isToolAutoApproveEnabledForConversation(conversationId) {
    const key = String(conversationId || '').trim();
    return Boolean(key && chatToolAutoApproveSettings[key]);
}

function setToolAutoApproveForConversation(conversationId, enabled) {
    const key = String(conversationId || '').trim();
    if (!key) return;
    if (enabled) {
        chatToolAutoApproveSettings[key] = true;
    } else {
        delete chatToolAutoApproveSettings[key];
    }
    persistStoredObject('chatToolAutoApproveSettings', chatToolAutoApproveSettings);
}

function clearToolAutoApproveForConversation(conversationId) {
    const key = String(conversationId || '').trim();
    if (!key) return;
    delete chatToolAutoApproveSettings[key];
    persistStoredObject('chatToolAutoApproveSettings', chatToolAutoApproveSettings);
}

function getRememberedTurnFeatures(conversationId) {
    const key = String(conversationId || '').trim();
    const stored = key ? conversationTurnFeatureMemory[key] : null;
    if (!stored || typeof stored !== 'object') {
        return {
            workspace_write: false,
            workspace_run_commands: false,
        };
    }
    return {
        workspace_write: Boolean(stored.workspace_write),
        workspace_run_commands: Boolean(stored.workspace_run_commands),
    };
}

function rememberTurnFeatures(conversationId, features) {
    const key = String(conversationId || '').trim();
    if (!key || !features || typeof features !== 'object') return;
    const previous = getRememberedTurnFeatures(key);
    conversationTurnFeatureMemory[key] = {
        workspace_write: Boolean(previous.workspace_write || features.workspace_write),
        workspace_run_commands: Boolean(previous.workspace_run_commands || features.workspace_run_commands),
    };
    persistStoredObject('conversationTurnFeatures', conversationTurnFeatureMemory);
}

function clearRememberedTurnFeatures(conversationId) {
    const key = String(conversationId || '').trim();
    if (!key) return;
    delete conversationTurnFeatureMemory[key];
    persistStoredObject('conversationTurnFeatures', conversationTurnFeatureMemory);
}

function resolveTurnFeatures(message, attachmentPaths = [], slashCommand = null) {
    const permissions = inferTurnPermissions(message, attachmentPaths);
    const executionRequest = messageRequestsPlanExecution(message, slashCommand);
    const continuationRequest = messageRequestsContinuation(message, slashCommand);
    const rememberedFeatures = getRememberedTurnFeatures(currentConvId);
    const allowedCommands = getAllowedCommandsForConversation(currentConvId);
    const allowedToolPermissions = getAllowedToolPermissionsForConversation(currentConvId);
    const autoApproveToolPermissions = isToolAutoApproveEnabledForConversation(currentConvId);
    const slashName = slashCommand?.name || '';
    const slashWantsWrite = slashName === 'code';
    const slashWantsRun = slashName === 'code' || slashName === 'pip';
    const carryForwardWrite = continuationRequest && rememberedFeatures.workspace_write && !permissions.wantsWrite && !slashWantsWrite;
    const workspaceWrite = carryForwardWrite || permissions.wantsWrite || slashWantsWrite || executionRequest;
    const workspaceRunCommands = (
        permissions.wantsRun
        || permissions.wantsWrite
        || slashWantsRun
        || executionRequest
        || allowedCommands.length > 0
        || (continuationRequest && rememberedFeatures.workspace_run_commands)
    );

    return {
        ...featureSettings,
        agent_tools: true,
        local_rag: true,
        web_search: true,
        workspace_write: workspaceWrite,
        workspace_run_commands: workspaceRunCommands,
        allowed_commands: allowedCommands,
        allowed_tool_permissions: allowedToolPermissions,
        auto_approve_tool_permissions: DOCUMENT_CENTERED_MODE || autoApproveToolPermissions,
    };
}

function beginClientTurn() {
    activeClientTurnId = generateId();
    return activeClientTurnId;
}

function clearActiveClientTurn(turnId = '') {
    const target = String(turnId || '').trim();
    if (!target || activeClientTurnId === target) {
        activeClientTurnId = '';
    }
}

function renderPermissionPanel() {
    const panel = document.getElementById('permissionPanel');
    const title = document.getElementById('permissionPanelTitle');
    const text = document.getElementById('permissionPanelText');
    const preview = document.getElementById('permissionPanelPreview');
    const note = document.getElementById('permissionPanelNote');
    const allowButton = document.getElementById('permissionAllowButton');
    const denyButton = document.getElementById('permissionDenyButton');
    if (!panel) return;
    if (DOCUMENT_CENTERED_MODE) {
        panel.hidden = true;
        return;
    }

    if (!pendingPermissionRequest) {
        panel.hidden = true;
        syncMobileViewportState();
        return;
    }

    const requestConversationId = String(pendingPermissionRequest.conversationId || '').trim();
    if (requestConversationId && requestConversationId !== currentConvId) {
        panel.hidden = true;
        syncMobileViewportState();
        return;
    }

    panel.hidden = false;
    if (title) title.textContent = pendingPermissionRequest.title || 'Permission needed';
    if (text) text.textContent = pendingPermissionRequest.content || 'The assistant needs your approval to continue.';
    if (preview) {
        const previewText = String(pendingPermissionRequest.preview || '').trim();
        preview.hidden = !previewText;
        preview.textContent = previewText;
    }
    if (note) {
        note.textContent = pendingPermissionRequest.note
            || 'Approving allows this for this chat and resumes immediately. If you do not approve it, the task pauses here.';
    }
    if (allowButton) allowButton.textContent = pendingPermissionRequest.allowLabel || 'Approve and continue';
    if (denyButton) denyButton.textContent = pendingPermissionRequest.denyLabel || 'Pause task';
    const input = document.getElementById('input');
    const inputHasDraft = Boolean((input && input.value.trim()) || pendingAttachments.length);
    if (allowButton && !inputHasDraft) {
        window.requestAnimationFrame(() => {
            if (!pendingPermissionRequest || panel.hidden) return;
            if (document.activeElement === input) return;
            allowButton.focus({ preventScroll: true });
        });
    }
    syncMobileViewportState();
}

function clearPendingPermissionRequest() {
    pendingPermissionRequest = null;
    renderPermissionPanel();
}

function rememberApprovalForPendingRequest() {
    if (!pendingPermissionRequest) return;
    const permissionKey = String(pendingPermissionRequest.permissionKey || '').trim().toLowerCase();
    if (!permissionKey) return;
    if (pendingPermissionRequest.approvalTarget === 'command') {
        rememberAllowedCommand(currentConvId, permissionKey);
        return;
    }
    rememberAllowedToolPermission(currentConvId, permissionKey);
}

function respondToPendingPermission(approved, options = {}) {
    if (!pendingPermissionRequest) return;
    const rememberApproval = options.rememberApproval !== false;
    const autoApproved = options.autoApproved === true;
    const permissionKey = String(pendingPermissionRequest.permissionKey || '').trim().toLowerCase();
    const approvalTarget = String(pendingPermissionRequest.approvalTarget || 'tool').trim().toLowerCase();
    const title = pendingPermissionRequest.title || 'Permission';
    const conversationId = String(pendingPermissionRequest.conversationId || currentConvId || '').trim();
    const clientTurnId = String(pendingPermissionRequest.clientTurnId || activeClientTurnId || '').trim();
    if (approved) {
        if (rememberApproval) {
            rememberApprovalForPendingRequest();
        }
        recordWorkspaceActivity(autoApproved ? 'Auto-Approve' : 'Approve', `${title} approved for this chat. Resuming the task.`);
        setLoadingText(`${title} approved. Resuming...`);
        showTransientComposerHint(
            autoApproved
                ? 'Tool approvals are automatic in this chat. The assistant is resuming.'
                : 'Approved. The assistant is resuming. Type to interrupt, or press Stop.',
            7000,
        );
    } else {
        recordWorkspaceActivity('Blocked', `${title} was not approved for this chat. The task is pausing here.`);
        setLoadingText(`${title} not approved. Pausing this task...`);
        showTransientComposerHint('Not approved. This task will pause here. Approve it later and say continue to resume.', 7000);
    }
    if (canUseWebsocketTransport()) {
        ws.send(JSON.stringify({
            type: 'permission_response',
            conversation_id: conversationId,
            client_turn_id: clientTurnId,
            permission_key: permissionKey,
            approval_target: approvalTarget,
            approved: Boolean(approved),
        }));
    }
    clearPendingPermissionRequest();
    syncSendButton();
    focusInputAtStart();
}

function syncToolApprovalToggle() {
    const toggle = document.getElementById('toolApprovalToggle');
    const value = document.getElementById('toolApprovalToggleValue');
    if (!toggle) return;
    const enabled = DOCUMENT_CENTERED_MODE ? true : isToolAutoApproveEnabledForConversation(currentConvId);
    toggle.classList.toggle('is-enabled', enabled);
    toggle.setAttribute('aria-pressed', enabled ? 'true' : 'false');
    toggle.setAttribute(
        'aria-label',
        enabled
            ? 'Automatic tool approvals are on for this chat'
            : 'Automatic tool approvals are off for this chat',
    );
    toggle.title = enabled
        ? 'Tool and command approvals auto-run in this chat. Plans still ask first.'
        : 'Ask before tool and command use in this chat. Plans still ask first.';
    if (value) value.textContent = enabled ? 'Auto' : 'Ask';
}

function toggleToolApprovalMode() {
    if (DOCUMENT_CENTERED_MODE) return;
    const nextEnabled = !isToolAutoApproveEnabledForConversation(currentConvId);
    setToolAutoApproveForConversation(currentConvId, nextEnabled);
    syncToolApprovalToggle();
    if (nextEnabled && pendingPermissionRequest) {
        const requestConversationId = String(pendingPermissionRequest.conversationId || currentConvId || '').trim();
        if (!requestConversationId || requestConversationId === currentConvId) {
            respondToPendingPermission(true, { rememberApproval: false, autoApproved: true });
            return;
        }
    }
    recordWorkspaceActivity(
        'Tools',
        nextEnabled
            ? 'Tool approvals now auto-approve in this chat. Plans still require approval.'
            : 'Tool approvals now ask first in this chat. Plans still require approval.',
    );
    showTransientComposerHint(
        nextEnabled
            ? 'Tool approvals are now automatic in this chat. Plans will still ask first.'
            : 'Tool approvals will ask first again in this chat. Plans still ask first.',
        5000,
    );
}

function hasActiveTurnForConversation(conversationId = '') {
    const target = String(conversationId || '').trim();
    if (!target) {
        return Boolean(isGenerating || currentTurnTransport || activeStreamConversationId);
    }
    return Boolean(
        isGenerating
        && (
            activeStreamConversationId === target
            || currentConvId === target
        )
    );
}

function clearResolvedPermissionPrompt() {
    if (!pendingPermissionRequest) return;
    clearPendingPermissionRequest();
}

function updateVoiceSetting(name, enabled) {
    if (!(name in voiceSettings)) return;
    voiceSettings[name] = enabled;
    persistVoiceSettings();
    syncVoiceSupportUI();
    if (!enabled) stopSpeaking();
}

function updateSpeechSpeed(value) {
    voiceSettings.speechSpeed = normalizeSpeechSpeed(value);
    persistVoiceSettings();
    syncFeatureControls();
    if (currentAudio) {
        currentAudio.playbackRate = getSpeechSpeedOption().rate;
    }
}

function supportsDictation() {
    return Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
}

function supportsSpeechSynthesis() {
    return Boolean(window.Audio && voiceRuntime.tts_available);
}

async function refreshVoiceRuntime() {
    try {
        const resp = await fetch('/api/voice/status');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        voiceRuntime = await resp.json();
    } catch (error) {
        voiceRuntime = {
            tts_available: false,
            stt_available: false,
            tts_backend: 'unavailable',
            stt_backend: 'unavailable',
            tts_voice: '',
        };
    }
    syncVoiceSupportUI();
    renderSettingsRuntimeStatus();
}

async function refreshRuntimeHealth() {
    try {
        const resp = await fetch('/health');
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
        runtimeHealth = data;
    } catch (error) {
        runtimeHealth = null;
    }
    renderSettingsRuntimeStatus();
}

function renderSettingsRuntimeStatus() {
    const runtimeBadge = document.getElementById('settingsRuntimeBadge');
    const runtimeNote = document.getElementById('settingsRuntimeNote');
    const modelName = document.getElementById('settingsModelName');
    const connectionBadge = document.getElementById('settingsConnectionBadge');
    if (!runtimeBadge || !runtimeNote || !modelName || !connectionBadge) return;

    const health = runtimeHealth || {};
    const loading = health.loading || {};
    let runtimeLabel = 'Offline';
    let note = 'The model server is disconnected or unavailable.';
    if (health.model_available) {
        runtimeLabel = 'Ready';
        note = health.message || 'Model server is ready.';
    } else if (loading.status === 'loading' || runtimeAvailabilityStatus === 'loading') {
        runtimeLabel = 'Loading';
        note = health.message || 'Model server is loading.';
    }
    runtimeBadge.textContent = runtimeLabel;
    runtimeNote.textContent = note;
    modelName.textContent = health.model || health.message || 'Unknown';
    connectionBadge.textContent = websocketConnected ? 'WS Live' : 'WS Down';
}

function updateVoiceNote(text = '') {
    const note = document.getElementById('voiceNote');
    if (!note) return;
    note.hidden = !text;
    note.textContent = text;
}

function syncVoiceSupportUI() {
    const micButton = document.getElementById('micButton');
    if (micButton) {
        micButton.disabled = !supportsDictation() || audioAttachmentUploadInFlight;
        micButton.classList.toggle('is-active', dictationActive);
        micButton.innerHTML = dictationActive ? MIC_ACTIVE_BUTTON_ICON : MIC_BUTTON_ICON;
        micButton.setAttribute('aria-label', dictationActive ? 'Stop recording' : 'Record audio');
        micButton.title = dictationActive
            ? 'Stop recording'
            : (supportsDictation() ? 'Record an audio attachment' : 'Audio recording is unavailable');
    }

    const autoSpeak = document.getElementById('settingAutoSpeak');
    if (autoSpeak) autoSpeak.disabled = !supportsSpeechSynthesis();
    const speechSpeed = document.getElementById('settingSpeechSpeed');
    if (speechSpeed) speechSpeed.disabled = !supportsSpeechSynthesis();

    if (dictationActive || audioAttachmentUploadInFlight) return;

    if (!supportsDictation() && !supportsSpeechSynthesis()) {
        updateVoiceNote('Audio recording and reply playback are unavailable right now.');
    } else if (!supportsDictation()) {
        updateVoiceNote('Audio recording is unavailable right now. Reply playback is still available.');
    } else if (!supportsSpeechSynthesis()) {
        updateVoiceNote('Reply playback is unavailable right now. Audio recording is still available.');
    } else {
        updateVoiceNote('');
    }
}

function mediaRecorderMimeType() {
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus', 'audio/ogg'];
    return candidates.find(type => window.MediaRecorder?.isTypeSupported?.(type)) || '';
}

async function stopDictationAndUpload() {
    if (!mediaRecorder) return;
    const recorder = mediaRecorder;
    if (recorder.state !== 'inactive') recorder.stop();
}

async function toggleDictation() {
    if (dictationActive) {
        await stopDictationAndUpload();
        return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        updateVoiceNote('Audio capture is not available in this browser.');
        return;
    }
    updateVoiceNote('');
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recordedChunks = [];
        const mimeType = mediaRecorderMimeType();
        mediaRecorder = mimeType ? new MediaRecorder(mediaStream, { mimeType }) : new MediaRecorder(mediaStream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size) recordedChunks.push(event.data);
        };
        mediaRecorder.onerror = () => {
            dictationActive = false;
            dictationStartedAt = 0;
            mediaRecorder = null;
            recordedChunks = [];
            audioAttachmentUploadInFlight = false;
            mediaStream?.getTracks?.().forEach(track => track.stop());
            mediaStream = null;
            syncVoiceSupportUI();
            syncSendButton();
            updateVoiceNote('Recording failed before the audio could be attached.');
        };
        mediaRecorder.onstop = async () => {
            dictationActive = false;
            audioAttachmentUploadInFlight = true;
            syncVoiceSupportUI();
            syncSendButton();
            const chunks = recordedChunks.slice();
            recordedChunks = [];
            const recorderMimeType = mediaRecorder?.mimeType || mimeType || 'audio/webm';
            const durationSeconds = Math.max(1, Math.round((Date.now() - dictationStartedAt) / 1000));
            dictationStartedAt = 0;
            mediaRecorder = null;
            mediaStream?.getTracks?.().forEach(track => track.stop());
            mediaStream = null;
            if (!chunks.length) {
                audioAttachmentUploadInFlight = false;
                syncVoiceSupportUI();
                syncSendButton();
                updateVoiceNote('');
                return;
            }
            updateVoiceNote('Saving audio attachment...');
            try {
                const blob = new Blob(chunks, { type: recorderMimeType });
                const extension = blob.type.includes('ogg') ? 'ogg' : (blob.type.includes('mp4') ? 'mp4' : 'webm');
                const filename = `recording-${new Date().toISOString().replace(/[:.]/g, '-')}.${extension}`;
                await uploadPendingFiles([{
                    blob,
                    name: filename,
                    size: blob.size,
                    contentType: blob.type || recorderMimeType,
                    duration: durationSeconds,
                    kind: 'audio',
                }], {
                    failureMessage: 'Audio attachment failed',
                });
                updateVoiceNote('');
            } catch (error) {
                updateVoiceNote(`Audio attachment failed: ${error.message}`);
            } finally {
                audioAttachmentUploadInFlight = false;
                syncVoiceSupportUI();
                syncSendButton();
            }
        };
        mediaRecorder.start();
        dictationActive = true;
        dictationStartedAt = Date.now();
        syncVoiceSupportUI();
        syncSendButton();
        updateVoiceNote('Recording audio...');
    } catch (error) {
        dictationActive = false;
        audioAttachmentUploadInFlight = false;
        dictationStartedAt = 0;
        syncVoiceSupportUI();
        syncSendButton();
        updateVoiceNote('Recording could not start. Allow microphone access and use HTTPS if the page is remote.');
    }
}

function extractMessageSpeechText(msg) {
    if (!msg) return '';
    // Read only the final assistant answer; reasoning is stored separately in .think-stack.
    let text = String(msg.dataset.originalContent || msg.querySelector('.message-content')?.innerText || '');
    // Strip fenced code blocks (```...```)
    text = text.replace(/```[\s\S]*?```/g, '');
    // Strip inline code (`...`)
    text = text.replace(/`[^`]*`/g, '');
    // Strip markdown image syntax ![alt](url)
    text = text.replace(/!\[[^\]]*\]\([^)]*\)/g, '');
    // Strip markdown link URLs but keep link text: [text](url) -> text
    text = text.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1');
    // Strip markdown heading markers
    text = text.replace(/^#{1,6}\s+/gm, '');
    // Strip markdown bold/italic markers
    text = text.replace(/(\*{1,3}|_{1,3})(.*?)\1/g, '$2');
    // Strip markdown horizontal rules
    text = text.replace(/^[-*_]{3,}\s*$/gm, '');
    // Strip emojis and other symbol characters
    text = text.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{200D}\u{20E3}\u{E0020}-\u{E007F}]/gu, '');
    // Strip remaining special characters (keep letters, numbers, basic punctuation, whitespace)
    text = text.replace(/[^\w\s.,;:!?''""\-()\/&%$#@+=%\n]/g, '');
    // Collapse whitespace
    text = text.replace(/\s+/g, ' ').trim();
    return text;
}

function stopSpeaking() {
    speechQueueVersion += 1;
    speechQueue = [];
    const task = activeSpeechTask;
    activeSpeechTask = null;
    if (task?.audio) {
        task.audio.onended = null;
        task.audio.onerror = null;
        task.audio.pause();
        task.audio.currentTime = 0;
    }
    currentAudio = null;
    if (typeof task?.settle === 'function') {
        task.settle({ stopped: true });
    }
    updateVoiceNote('');
}

async function requestServerSpeech(text) {
    const payload = { text };
    if (currentConvId) payload.conversation_id = currentConvId;
    const resp = await fetch('/api/voice/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
    return data;
}

function enqueueAssistantSpeech(msg) {
    if (!supportsSpeechSynthesis() || !voiceSettings.autoSpeakReplies || !msg) return;
    const text = extractMessageSpeechText(msg);
    if (!text) return;
    if (
        speechQueue.some(entry => entry.msg === msg) ||
        activeSpeechTask?.message === msg ||
        (currentAudio && currentAudio._messageElement === msg)
    ) {
        msg.dataset.autoSpoken = 'true';
        return;
    }
    msg.dataset.autoSpoken = 'true';
    speechQueue.push({ msg, text });
    processSpeechQueue();
}

async function playSpeechAudio(task, audio) {
    return new Promise((resolve, reject) => {
        let settled = false;
        const finish = (error = null, result = {}) => {
            if (settled) return;
            settled = true;
            task.settle = null;
            audio.onended = null;
            audio.onerror = null;
            if (error) reject(error);
            else resolve(result);
        };
        task.settle = (result = { stopped: true }) => finish(null, result);
        audio.onended = () => finish(null, { stopped: false });
        audio.onerror = () => finish(new Error('Audio playback failed.'));
        const playPromise = audio.play();
        if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(error => finish(error));
        }
    });
}

async function processSpeechQueue() {
    if (speechQueueBusy) return;
    speechQueueBusy = true;
    try {
        while (speechQueue.length) {
            if (!voiceSettings.autoSpeakReplies || !supportsSpeechSynthesis()) {
                speechQueue = [];
                break;
            }
            const entry = speechQueue.shift();
            if (!entry?.msg?.isConnected || !entry.text) continue;
            const task = {
                version: speechQueueVersion,
                message: entry.msg,
                audio: null,
                settle: null,
            };
            activeSpeechTask = task;
            currentAudio = null;
            updateVoiceNote('Generating server speech...');
            try {
                const result = await requestServerSpeech(entry.text);
                if (speechQueueVersion !== task.version || activeSpeechTask !== task) continue;
                const audio = new Audio(result.audio_url);
                audio._messageElement = entry.msg;
                audio.playbackRate = getSpeechSpeedOption().rate;
                task.audio = audio;
                currentAudio = audio;
                updateVoiceNote('');
                await playSpeechAudio(task, audio);
            } catch (error) {
                if (speechQueueVersion === task.version && activeSpeechTask === task) {
                    const prefix = task.audio ? 'Speech playback failed' : 'Speech generation failed';
                    updateVoiceNote(`${prefix}: ${error.message}`);
                }
            } finally {
                if (activeSpeechTask === task) activeSpeechTask = null;
                if (currentAudio === task.audio) currentAudio = null;
            }
        }
    } finally {
        speechQueueBusy = false;
        const note = document.getElementById('voiceNote');
        if (
            !currentAudio &&
            !activeSpeechTask &&
            !speechQueue.length &&
            note?.textContent === 'Generating server speech...'
        ) {
            updateVoiceNote('');
        }
    }
}

function startHealthPolling() {
    if (healthPollInterval) return;
    healthPollInterval = setInterval(async () => {
        try {
            const resp = await fetch('/health');
            const health = await resp.json();
            if (health.model_available) {
                updateStatus('connected');
                clearInterval(healthPollInterval);
                healthPollInterval = null;
            } else if (health.loading?.status === 'failed') {
                updateStatus('disconnected');
                clearInterval(healthPollInterval);
                healthPollInterval = null;
            } else {
                if (!modelAvailable) updateStatus('loading');
            }
        } catch (e) {
            if (!modelAvailable) updateStatus('loading');
        }
    }, 3000);
}

// ==================== WebSocket ====================

function handleChatEvent(data) {
    if (!data || typeof data !== 'object') return;
    if (data.type === 'file_session_bound') {
        const boundPath = normalizeDraftFilename(data.file_path || DEFAULT_DRAFT_FILENAME);
        const boundConversationId = String(data.conversation_id || '').trim();
        if (boundConversationId && compareWorkspacePaths(boundPath, activeDraftPath || '')) {
            currentConvId = boundConversationId;
            syncToolApprovalToggle();
            renderDraftFileExplorer();
        }
    } else if (data.type === 'start') {
        if (DOCUMENT_CENTERED_MODE && activeDraftForegroundJobId) {
            updateDraftFileJobStatus(activeDraftForegroundJobId, 'running').catch(error => {
                console.error('Failed to mark the foreground file job as running:', error);
            });
        }
        if (data.client_turn_id) clearActiveClientTurn(String(data.client_turn_id));
        clearPendingPermissionRequest();
        removeLoading();
        ensureStreamingAssistantMessage();
        if (DOCUMENT_CENTERED_MODE) {
            setDraftSaveState('saving', 'Agent');
        }
        currentAssistantTurnStartedAt = Date.now();
        currentAssistantTurnArtifactPaths = new Set();
        currentPlanFocus = null;
        recordWorkspaceActivity('Turn', 'Turn started.');
    } else if (data.type === 'reasoning_note') {
        const noteText = data.content || 'Reasoning noted.';
        recordWorkspaceActivity('Think', noteText, {
            phase: data.phase || 'think',
            stepLabel: data.step_label || '',
        });
        updatePlanFocus({
            text: noteText,
            phase: data.phase || 'think',
            stepLabel: data.step_label || currentBuildStepLabel(),
        });
    } else if (data.type === 'assistant_note') {
        ensureStreamingAssistantMessage();
        replaceAssistantAnswer(data.content || '');
        recordWorkspaceActivity('Note', data.content || 'Updated the working draft.');
    } else if (data.type === 'plan_ready') {
        storeExecutionPlan(data.plan || '', data.execute_prompt || '', data.builder_steps || []);
        recordWorkspaceActivity('Plan', DOCUMENT_CENTERED_MODE ? 'Execution plan auto-approved.' : 'Execution plan ready for approval.');
    } else if (data.type === 'build_steps') {
        updateBuildSteps(
            data.steps || [],
            data.active_index ?? null,
            data.completed_count || 0,
            data.step_details || [],
        );
    } else if (data.type === 'think_start') {
        onAssistantThinkStart();
    } else if (data.type === 'think_token') {
        onAssistantThinkToken(data.content);
    } else if (data.type === 'think_end') {
        onAssistantThinkEnd();
    } else if (data.type === 'token') {
        clearResolvedPermissionPrompt();
        appendAssistantAnswerToken(data.content);
    } else if (data.type === 'status') {
        setLoadingText(data.content);
    } else if (data.type === 'activity') {
        if ((data.phase || '').toLowerCase() !== 'blocked') {
            clearResolvedPermissionPrompt();
        }
        if (data.content) setLoadingText(data.content);
        recordWorkspaceActivity(
            data.label || 'Activity',
            data.content || 'Working...',
            {
                phase: data.phase || 'status',
                stepLabel: data.step_label || '',
            },
        );
        updatePlanFocus({
            text: data.content || 'Working...',
            phase: data.phase || 'status',
            stepLabel: data.step_label || currentBuildStepLabel(),
        });
    } else if (data.type === 'tool_start') {
        clearResolvedPermissionPrompt();
        const toolText = data.content || `Using ${data.name || 'tool'}...`;
        setLoadingText(toolText);
        recordWorkspaceActivity('Tool', toolText, {
            phase: 'tool',
            stepLabel: data.step_label || '',
        });
        updatePlanFocus({
            text: toolText,
            phase: 'tool',
            stepLabel: data.step_label || currentBuildStepLabel(),
        });
    } else if (data.type === 'tool_result') {
        clearResolvedPermissionPrompt();
        const toolResultText = data.content || (data.ok === false ? 'Tool failed' : 'Tool finished');
        setLoadingText(toolResultText);
        recordWorkspaceActivity(data.ok === false ? 'Tool Error' : 'Tool Result', toolResultText, {
            phase: data.ok === false ? 'error' : 'tool',
            stepLabel: data.step_label || '',
            error: data.ok === false,
        });
        updatePlanFocus({
            text: toolResultText,
            phase: data.ok === false ? 'error' : 'tool',
            stepLabel: data.step_label || currentBuildStepLabel(),
        });
        if (data.ok !== false) noteAssistantArtifactsFromToolResult(data);
        if (data.name === 'workspace.render' && data.ok !== false && data.payload?.path) {
            openWorkspaceFile(data.payload.path);
        } else if (data.name === 'workspace.run_command' && data.ok !== false && shouldAutoOpenArtifactPreview(data.payload?.open_path)) {
            openWorkspaceFile(data.payload.open_path);
        }
        scheduleWorkspaceRefresh();
    } else if (data.type === 'tool_error') {
        clearResolvedPermissionPrompt();
        const toolErrorText = data.content || 'Tool error';
        setLoadingText(toolErrorText);
        recordWorkspaceActivity('Tool Error', toolErrorText, {
            phase: 'error',
            stepLabel: data.step_label || '',
            error: true,
        });
        updatePlanFocus({
            text: toolErrorText,
            phase: 'error',
            stepLabel: data.step_label || currentBuildStepLabel(),
        });
    } else if (data.type === 'permission_required') {
        const permissionConversationId = String(data.conversation_id || '').trim();
        const clientTurnId = String(data.client_turn_id || '').trim();
        const permissionKey = String(data.permission_key || '').trim().toLowerCase();
        const approvalTarget = String(data.approval_target || 'tool').trim().toLowerCase();
        if (permissionConversationId && permissionConversationId !== currentConvId) {
            if (canUseWebsocketTransport()) {
                ws.send(JSON.stringify({
                    type: 'permission_response',
                    conversation_id: permissionConversationId,
                    client_turn_id: clientTurnId,
                    permission_key: permissionKey,
                    approval_target: approvalTarget,
                    approved: false,
                }));
            }
            return;
        }
        if (DOCUMENT_CENTERED_MODE) {
            if (canUseWebsocketTransport()) {
                ws.send(JSON.stringify({
                    type: 'permission_response',
                    conversation_id: permissionConversationId || currentConvId,
                    client_turn_id: clientTurnId,
                    permission_key: permissionKey,
                    approval_target: approvalTarget,
                    approved: true,
                }));
            }
            recordWorkspaceActivity('Auto-Approve', `${data.title || 'Permission'} approved automatically.`, {
                phase: 'active',
                stepLabel: data.step_label || '',
            });
            return;
        }
        const requestConversationId = permissionConversationId || currentConvId;
        if (isToolAutoApproveEnabledForConversation(requestConversationId)) {
            if (canUseWebsocketTransport()) {
                ws.send(JSON.stringify({
                    type: 'permission_response',
                    conversation_id: requestConversationId,
                    client_turn_id: clientTurnId,
                    permission_key: permissionKey,
                    approval_target: approvalTarget,
                    approved: true,
                }));
            }
            recordWorkspaceActivity(
                'Auto-Approve',
                `${data.title || 'Permission needed'} auto-approved for this chat.`,
                {
                    phase: 'active',
                    stepLabel: data.step_label || '',
                },
            );
            setLoadingText(`${data.title || 'Permission'} approved. Resuming...`);
            showTransientComposerHint('Tool approvals are automatic in this chat. The assistant is resuming.', 5000);
            return;
        }
        if (!hasActiveTurnForConversation(requestConversationId) && !isGenerating) {
            if (canUseWebsocketTransport()) {
                ws.send(JSON.stringify({
                    type: 'permission_response',
                    conversation_id: requestConversationId,
                    client_turn_id: clientTurnId,
                    permission_key: permissionKey,
                    approval_target: approvalTarget,
                    approved: false,
                }));
            }
            return;
        }
        pendingPermissionRequest = {
            conversationId: requestConversationId,
            clientTurnId,
            permissionKey,
            approvalTarget,
            title: data.title || 'Permission needed',
            content: data.content || 'The assistant needs your approval to continue.',
            preview: data.preview || '',
            note: data.note || '',
            allowLabel: data.allow_label || 'Approve and continue',
            denyLabel: data.deny_label || 'Pause task',
        };
        renderPermissionPanel();
        recordWorkspaceActivity('Permission', pendingPermissionRequest.content, {
            phase: 'blocked',
            stepLabel: data.step_label || '',
        });
        setLoadingText(pendingPermissionRequest.content);
    } else if (data.type === 'final_replace') {
        clearResolvedPermissionPrompt();
        replaceAssistantAnswer(data.content);
        if (DOCUMENT_CENTERED_MODE) setDraftSaveState('saved', 'Ready');
        recordWorkspaceActivity('Draft', 'Draft updated.');
    } else if (data.type === 'message_id') {
        clearResolvedPermissionPrompt();
        const msg = currentStreamingAssistantMessage();
        if (msg && data.message_id !== undefined && data.message_id !== null) {
            msg.dataset.messageId = String(data.message_id);
            msg.dataset.feedback = normalizeMessageFeedback(msg.dataset.feedback);
            syncAssistantMessageActions(msg);
        }
    } else if (data.type === 'done') {
        clearActiveClientTurn();
        clearPendingPermissionRequest();
        isGenerating = false;
        currentTurnTransport = null;
        httpTurnAbortController = null;
        if (DOCUMENT_CENTERED_MODE) setDraftSaveState('saved', 'Ready');
        syncSendButton();
        focusInput();
        renderMarkdown();
        loadConversations();
        finalizeAssistantTurnArtifacts();
        refreshWorkspace(true);
        if (DOCUMENT_CENTERED_MODE) {
            if (activeDraftForegroundJobId) {
                updateDraftFileJobStatus(activeDraftForegroundJobId, 'completed').catch(error => {
                    console.error('Failed to mark the foreground file job as completed:', error);
                });
                activeDraftForegroundJobId = '';
            }
            persistDraftSession(currentWorkspaceId, activeDraftPath, currentConvId);
            syncCurrentConversationTitleToDraft(activeDraftPath).catch(() => {});
            snapshotGeneratedOutputIfChanged(activeDraftPath).catch(error => {
                console.error('Failed to snapshot the generated file version:', error);
            });
            if (draftPendingAgentSync) {
                draftPendingAgentSync = false;
                scheduleDraftAgentSync({ immediate: true });
            } else if (activeDraftPath) {
                loadDraftDocument(activeDraftPath, { preserveAgentState: true }).catch(error => {
                    console.error('Failed to refresh the active draft spec after completion:', error);
                });
            }
            renderDraftFileExplorer();
            refreshDraftVersionsSidebar().catch(error => {
                console.error('Failed to refresh draft versions:', error);
            });
            refreshDraftOutputPane().catch(error => {
                console.error('Failed to refresh generated output preview:', error);
            });
        }
        recordWorkspaceActivity('Done', 'Turn complete.');
        activeStreamConversationId = null;
        streamingAssistantMessage = null;
    } else if (data.type === 'canceled') {
        clearActiveClientTurn();
        clearPendingPermissionRequest();
        removeLoading();
        isGenerating = false;
        currentTurnTransport = null;
        httpTurnAbortController = null;
        syncSendButton();
        if (DOCUMENT_CENTERED_MODE && activeDraftForegroundJobId) {
            updateDraftFileJobStatus(activeDraftForegroundJobId, 'canceled', data.content || 'Stopped').catch(error => {
                console.error('Failed to mark the foreground file job as canceled:', error);
            });
            activeDraftForegroundJobId = '';
        }
        recordWorkspaceActivity('Canceled', data.content || 'Stopped');
    } else if (data.type === 'error') {
        clearActiveClientTurn();
        clearPendingPermissionRequest();
        removeLoading();
        isGenerating = false;
        currentTurnTransport = null;
        httpTurnAbortController = null;
        syncSendButton();
        if (DOCUMENT_CENTERED_MODE && activeDraftForegroundJobId) {
            updateDraftFileJobStatus(activeDraftForegroundJobId, 'failed', data.content || 'The assistant hit an error.').catch(error => {
                console.error('Failed to mark the foreground file job as failed:', error);
            });
            activeDraftForegroundJobId = '';
        }
        recordWorkspaceActivity('Error', data.content || 'The assistant hit an error.', { error: true });
        const errorMsg = streamingAssistantMessage || createMessageElement('', 'assistant');
        const contentDiv = errorMsg.querySelector('.message-content');
        if (contentDiv) {
            contentDiv.style.color = '#ef4444';
            contentDiv.textContent = data.content;
        }
        streamingAssistantMessage = errorMsg;
        appendStreamingAssistantMessageIfVisible();
        if (DOCUMENT_CENTERED_MODE) setDraftSaveState('error', 'Agent');
        if (currentConvId === activeStreamConversationId) {
            activeStreamConversationId = null;
            streamingAssistantMessage = null;
        }
    }
}

async function dispatchChatPayload(payload, options = {}) {
    const nextPayload = {
        ...payload,
        file_path: payload?.file_path || (activeDraftPath ? normalizeDraftFilename(activeDraftPath) : null),
    };
    if (canUseWebsocketTransport()) {
        currentTurnTransport = 'ws';
        ws.send(JSON.stringify(nextPayload));
        return;
    }

    const controller = new AbortController();
    httpTurnAbortController = controller;
    currentTurnTransport = 'http';
    setLoadingText(options.fallbackStatusText || 'WebSocket unavailable. Sending over HTTP fallback...');
    recordWorkspaceActivity('Transport', options.fallbackActivityText || 'WebSocket unavailable. Using HTTP fallback.');

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(buildHttpFallbackPayload(nextPayload)),
            signal: controller.signal,
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
        const events = Array.isArray(data.events) ? data.events : [];
        for (const event of events) {
            handleChatEvent(event);
        }
        if (!events.some(event => ['done', 'error', 'canceled'].includes(event?.type))) {
            handleChatEvent({ type: 'done' });
        }
    } catch (error) {
        if (controller.signal.aborted) {
            handleChatEvent({ type: 'canceled', content: 'Stopped' });
            return;
        }
        handleChatEvent({ type: 'error', content: `HTTP fallback failed: ${error.message}` });
    } finally {
        if (httpTurnAbortController === controller) httpTurnAbortController = null;
    }
}

function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
    setWebsocketConnected(false);

    ws.onopen = () => {
        setWebsocketConnected(true);
        fetch('/health').then(r => r.json()).then(health => {
            if (health.model_available) {
                updateStatus('connected');
            } else {
                if (!modelAvailable) updateStatus('loading');
                startHealthPolling();
            }
        }).catch(() => {
            if (!modelAvailable) updateStatus('loading');
            startHealthPolling();
        });
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'ping') {
            ws.send(JSON.stringify({ type: 'pong' }));
            return;
        }
        handleChatEvent(data);
    };

    ws.onerror = () => {
        if (currentTurnTransport === 'ws') {
            clearActiveClientTurn();
            clearPendingPermissionRequest();
            removeLoading();
            isGenerating = false;
            currentTurnTransport = null;
            activeStreamConversationId = null;
            streamingAssistantMessage = null;
            syncSendButton();
        }
        setWebsocketConnected(false);
    };
    ws.onclose = () => {
        const activeWsTurn = currentTurnTransport === 'ws';
        if (activeWsTurn) {
            clearActiveClientTurn();
            clearPendingPermissionRequest();
            removeLoading();
            isGenerating = false;
            currentTurnTransport = null;
            activeStreamConversationId = null;
            streamingAssistantMessage = null;
            syncSendButton();
            recordWorkspaceActivity('Error', 'Live connection lost. You can resend; HTTP fallback will be used until streaming reconnects.', { error: true });
        }
        setWebsocketConnected(false);
        setTimeout(connectWS, 2000);
    };
}

// ==================== Menu ====================

function toggleMenu() {
    const overlay = document.getElementById('menuOverlay');
    const button = document.getElementById('menuBtn');
    const shell = document.querySelector('.chat-shell');
    if (!overlay) return;
    dismissMobileKeyboard(true);
    const open = !overlay.classList.contains('show');
    if (open) {
        closeWorkspacePanel();
        closeFileExplorer();
        refreshDraftVersionsSidebar().catch(() => {});
    }
    overlay.classList.toggle('show', open);
    if (shell) shell.classList.toggle('menu-open', open);
    if (button) {
        button.classList.toggle('active', open);
        button.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    syncChatShellLayout();
}
function closeMenu() {
    const overlay = document.getElementById('menuOverlay');
    const button = document.getElementById('menuBtn');
    const shell = document.querySelector('.chat-shell');
    if (overlay) overlay.classList.remove('show');
    if (shell) shell.classList.remove('menu-open');
    if (button) {
        button.classList.remove('active');
        button.setAttribute('aria-expanded', 'false');
    }
    syncChatShellLayout();
}

function toggleFileExplorer() {
    const overlay = document.getElementById('fileExplorerOverlay');
    const button = document.getElementById('fileExplorerToggle');
    if (!overlay) return;
    dismissMobileKeyboard(true);
    const open = !overlay.classList.contains('show');
    if (open) {
        closeWorkspacePanel();
        closeMenu();
        renderDraftFileExplorer();
    }
    overlay.classList.toggle('show', open);
    if (button) {
        button.classList.toggle('active', open);
        button.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    syncChatShellLayout();
}

function closeFileExplorer() {
    const overlay = document.getElementById('fileExplorerOverlay');
    const button = document.getElementById('fileExplorerToggle');
    if (overlay) overlay.classList.remove('show');
    if (button) {
        button.classList.remove('active');
        button.setAttribute('aria-expanded', 'false');
    }
    syncChatShellLayout();
}

// ==================== Settings ====================

function showSettings() {
    dismissMobileKeyboard(true);
    closeAbout();
    syncFeatureControls();
    refreshRuntimeHealth().catch(() => {});
    document.getElementById('settingsOverlay').classList.add('show');
}
function closeSettings() {
    document.getElementById('settingsOverlay').classList.remove('show');
}

function showAbout() {
    dismissMobileKeyboard(true);
    closeSettings();
    document.getElementById('aboutOverlay').classList.add('show');
}

function closeAbout() {
    document.getElementById('aboutOverlay').classList.remove('show');
}

function setResetButtonState(isBusy) {
    const button = document.getElementById('resetAppButton');
    if (!button) return;
    button.disabled = isBusy;
    button.textContent = isBusy ? 'Resetting...' : 'Delete All Data';
}

function deleteIndexedDbDatabase(name) {
    return new Promise(resolve => {
        if (!name || !window.indexedDB) {
            resolve();
            return;
        }
        try {
            const request = window.indexedDB.deleteDatabase(name);
            request.onsuccess = () => resolve();
            request.onerror = () => resolve();
            request.onblocked = () => resolve();
        } catch (error) {
            resolve();
        }
    });
}

async function clearBrowserPersistence() {
    try {
        window.localStorage.clear();
    } catch (error) {
        console.warn('Failed to clear localStorage:', error);
    }
    try {
        window.sessionStorage.clear();
    } catch (error) {
        console.warn('Failed to clear sessionStorage:', error);
    }

    if (!window.indexedDB || typeof window.indexedDB.databases !== 'function') {
        return;
    }

    try {
        const databases = await window.indexedDB.databases();
        await Promise.all(
            (databases || [])
                .map(entry => entry?.name)
                .filter(Boolean)
                .map(name => deleteIndexedDbDatabase(name))
        );
    } catch (error) {
        console.warn('Failed to clear IndexedDB state:', error);
    }
}

async function resetAllAppData() {
    const firstConfirmation = confirm(
        'Delete all chats, workspaces, artifacts, and saved browser preferences?\n\nThis also removes packages and files installed inside conversation workspaces.'
    );
    if (!firstConfirmation) return;

    const secondConfirmation = confirm(
        'This reset is permanent and cannot be undone.\n\nContinue with a full app reset?'
    );
    if (!secondConfirmation) return;

    setResetButtonState(true);
    try {
        const resp = await fetch('/api/reset-all', { method: 'POST' });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || data.message || `HTTP ${resp.status}`);

        stopSpeaking();
        await clearBrowserPersistence();
        closeSettings();
        window.location.reload();
    } catch (error) {
        alert(`Reset failed: ${error.message}`);
        setResetButtonState(false);
    }
}

function buildWelcomeMarkup() {
    const path = normalizeDraftFilename(activeDraftPath || defaultDraftPathForWorkspace());
    const profile = getDraftProfileDefinition(activeDraftProfileKey || draftProfileKeyForPath(path));
    return `<div class="welcome">
            <div class="agent-rail-empty">
                <div class="agent-rail-empty-title">Agent standing by</div>
                <div class="agent-rail-empty-copy">Keep drafting in <code>${escapeHtml(path)}</code>. The app is treating it as ${escapeHtml(profile.label)} automatically, so chat can stay focused on guided edits, questions, and bigger changes when you need them.</div>
            </div>
        </div>`;
}

function stopWelcomeHintRotation() {
    if (!welcomeHintTimer) return;
    window.clearInterval(welcomeHintTimer);
    welcomeHintTimer = null;
}

function pickRotatingHint(previousHint = '') {
    if (!DISCOVERY_HINTS.length) return '';
    const pool = DISCOVERY_HINTS.filter(hint => hint !== previousHint);
    const candidates = pool.length ? pool : DISCOVERY_HINTS;
    return candidates[Math.floor(Math.random() * candidates.length)] || DISCOVERY_HINTS[0];
}

function startWelcomeHintRotation() {
    stopWelcomeHintRotation();
    const hint = document.getElementById('appHint');
    if (!hint) return;

    let currentHint = '';
    const renderHint = () => {
        currentHint = pickRotatingHint(currentHint);
        hint.textContent = currentHint;
    };

    renderHint();
    welcomeHintTimer = window.setInterval(renderHint, WELCOME_HINT_ROTATE_MS);
}

function clearTransientComposerHint() {
    if (transientHintTimer) {
        window.clearTimeout(transientHintTimer);
        transientHintTimer = null;
    }
}

function showTransientComposerHint(text, durationMs = 6000) {
    const hint = document.getElementById('appHint');
    if (!hint) return;
    clearTransientComposerHint();
    hint.textContent = String(text || '').trim();
    if (!durationMs) return;
    transientHintTimer = window.setTimeout(() => {
        transientHintTimer = null;
        const area = document.getElementById('inputArea');
        if (area?.classList.contains('welcome-mode')) {
            startWelcomeHintRotation();
            return;
        }
        if (hint.textContent === text) {
            hint.textContent = '';
        }
    }, durationMs);
}

// ==================== Chat ====================

function isComposerCompact(textarea = document.getElementById('input')) {
    const area = document.getElementById('inputArea');
    if (!textarea || !area || area.classList.contains('welcome-mode')) return false;
    const value = String(textarea.value || '');
    const trimmed = value.trim();
    if (!trimmed) return true;
    if (value.includes('\n')) return false;
    const compactThreshold = isMobileViewport() ? 48 : 64;
    return trimmed.length <= compactThreshold;
}

function syncComposerDensity(textarea = document.getElementById('input')) {
    const area = document.getElementById('inputArea');
    if (!textarea || !area) return false;
    const compact = isComposerCompact(textarea);
    area.classList.toggle('composer-compact', compact);
    textarea.rows = compact ? 1 : 3;
    return compact;
}

function autoResizeTextarea(textarea) {
    if (!textarea) return;
    const compact = syncComposerDensity(textarea);
    const mobile = isMobileViewport();
    const minHeight = compact
        ? (mobile ? (mobileKeyboardInset > 120 ? 32 : 36) : 28)
        : (mobile ? (mobileKeyboardInset > 120 ? 38 : 46) : 72);
    const viewportHeight = window.visualViewport?.height || window.innerHeight || document.documentElement.clientHeight || 0;
    const maxHeight = mobile ? Math.max(96, Math.min(160, Math.round(viewportHeight * 0.26))) : 220;
    textarea.style.minHeight = `${minHeight}px`;
    textarea.style.maxHeight = `${maxHeight}px`;
    textarea.style.height = 'auto';
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
}

function exitWelcomeMode() {
    stopWelcomeHintRotation();
    const area = document.getElementById('inputArea');
    if (area) area.classList.remove('welcome-mode');
    const chat = document.querySelector('.chat');
    if (chat) chat.classList.remove('welcome-layout');
    const input = document.getElementById('input');
    if (input) {
        input.style.height = '';
        autoResizeTextarea(input);
    }
}

function enterWelcomeMode() {
    const area = document.getElementById('inputArea');
    if (area) area.classList.add('welcome-mode');
    const chat = document.querySelector('.chat');
    if (chat) chat.classList.add('welcome-layout');
    const input = document.getElementById('input');
    if (input) {
        input.style.height = '';
        autoResizeTextarea(input);
    }
    startWelcomeHintRotation();
}

function toggleDeepMode() {
    deepMode = true;
    syncReasoningSelector();
}

function focusInput() {
    if (DOCUMENT_CENTERED_MODE) {
        const editor = ensureDraftEditor();
        if (editor) {
            editor.focus();
            return;
        }
        const draftTextarea = document.getElementById('draftEditor');
        if (draftTextarea) {
            draftTextarea.focus();
            return;
        }
    }
    const input = document.getElementById('input');
    if (input) {
        input.focus();
        syncMobileViewportState();
    }
}

function focusInputAtStart() {
    if (DOCUMENT_CENTERED_MODE) {
        focusInput();
        return;
    }
    const input = document.getElementById('input');
    if (!input) return;
    input.focus();
    input.setSelectionRange(0, 0);
}

function normalizeExecutionPlanSteps(steps) {
    if (!Array.isArray(steps)) return [];
    return steps
        .map(step => String(step || '').trim())
        .filter(Boolean)
        .slice(0, PLAN_STEP_LIMIT);
}

function extractExecutionPlanStepsFromPrompt(executePrompt) {
    const prompt = String(executePrompt || '');
    if (!prompt) return [];
    const match = prompt.match(/\nSteps:\n([\s\S]*?)(?:\n\nVerification:\n|$)/);
    if (!match) return [];
    return match[1]
        .split(/\r?\n/)
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => line.replace(/^\d+[.)]?\s+/, '').trim())
        .filter(Boolean)
        .slice(0, PLAN_STEP_LIMIT);
}

function formatExecutionPlanDraft(steps) {
    return normalizeExecutionPlanSteps(steps)
        .map((step, index) => `${index + 1}. ${step}`)
        .join('\n');
}

function parseExecutionPlanDraft(value) {
    return String(value || '')
        .split(/\r?\n/)
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => line.replace(/^[-*]\s+/, '').replace(/^\d+[.)]?\s+/, '').trim())
        .filter(Boolean)
        .slice(0, PLAN_STEP_LIMIT);
}

function getExecutionPlanSourceSteps() {
    return normalizeExecutionPlanSteps(
        pendingExecutionPlan?.builderSteps?.length
            ? pendingExecutionPlan.builderSteps
            : extractExecutionPlanStepsFromPrompt(pendingExecutionPlan?.executePrompt || ''),
    );
}

function getExecutionPlanDraftSteps() {
    const draft = document.getElementById('planApprovalDraft');
    if (draft) {
        return parseExecutionPlanDraft(draft.value);
    }
    if (pendingExecutionPlan && typeof pendingExecutionPlan.draftText === 'string') {
        return parseExecutionPlanDraft(pendingExecutionPlan.draftText);
    }
    return getExecutionPlanSourceSteps();
}

function focusExecutionPlanDraft(position = 'end') {
    const draft = document.getElementById('planApprovalDraft');
    if (!draft || draft.offsetParent === null) return false;
    draft.focus();
    const caretPosition = position === 'start' ? 0 : draft.value.length;
    draft.setSelectionRange(caretPosition, caretPosition);
    return true;
}

function focusExecutionPlanApproveButton() {
    const button = document.getElementById('planApprovalApproveButton');
    if (!button || button.disabled || button.offsetParent === null) return false;
    button.focus({ preventScroll: true });
    return true;
}

function syncInterveneButton() {
    const textarea = document.getElementById('interveneInput');
    const button = document.getElementById('interveneButton');
    const panel = document.getElementById('buildStepsIntervene');
    if (!textarea || !button || !panel) return;
    panel.hidden = !(isGenerating && buildSteps.length);
    button.disabled = !textarea.value.trim() || !isGenerating || !ws || ws.readyState !== WebSocket.OPEN;
}

function syncSendButton() {
    const send = document.getElementById('send');
    const input = document.getElementById('input');
    if (!send) return;

    if (isGenerating) {
        const canInterruptWithMessage = currentTurnTransport === 'ws' && canUseWebsocketTransport();
        send.disabled = false;
        const hasDraft = Boolean((input && input.value.trim()) || pendingAttachments.length);
        const showInterruptAction = canInterruptWithMessage && hasDraft;
        send.innerHTML = showInterruptAction ? SEND_BUTTON_ICON : STOP_BUTTON_ICON;
        send.classList.add('is-stop');
        send.setAttribute('aria-label', showInterruptAction ? 'Interrupt with message' : 'Stop response');
        send.title = showInterruptAction ? 'Interrupt with message' : 'Stop response';
        syncInterveneButton();
        renderExecutionPlanApproval();
        return;
    }

    send.innerHTML = SEND_BUTTON_ICON;
    send.classList.remove('is-stop');
    const hasDraft = Boolean(input && input.value.trim());
    const hasPendingContent = hasDraft || pendingAttachments.length > 0;
    send.setAttribute('aria-label', 'Send message');
    send.title = 'Send message';
    send.disabled = !modelAvailable || !input || audioAttachmentUploadInFlight || !hasPendingContent;
    syncInterveneButton();
    renderExecutionPlanApproval();
}

function handleInputChange(textarea) {
    const area = document.getElementById('inputArea');
    if (!area?.classList.contains('welcome-mode')) {
        clearTransientComposerHint();
        const hint = document.getElementById('appHint');
        if (hint && hint.textContent && !pendingPermissionRequest) {
            hint.textContent = '';
        }
    }
    autoResizeTextarea(textarea);
    syncSendButton();
    updateSlashMenu(textarea);
}

function setInputValue(value, cursorPosition = value.length) {
    const input = document.getElementById('input');
    if (!input) return;
    input.value = value;
    input.focus();
    input.setSelectionRange(cursorPosition, cursorPosition);
    handleInputChange(input);
}

function normalizeDirectSlashCommand(name) {
    const candidate = String(name || '').trim().toLowerCase();
    if (!candidate) return null;
    return Object.values(DIRECT_SLASH_COMMANDS).find(command => command.aliases.includes(candidate)) || null;
}

function parseDirectSlashCommandInput(text) {
    const match = String(text || '').match(/^\s*\/([a-z0-9_-]+)(?:\s+([\s\S]*\S))?\s*$/i);
    if (!match) return null;
    const command = normalizeDirectSlashCommand(match[1]);
    if (!command) return null;
    return {
        rawName: String(match[1] || '').trim().toLowerCase(),
        name: command.canonical,
        label: command.label,
        template: command.template,
        args: String(match[2] || '').trim(),
    };
}

function getSlashCommandQuery(textarea) {
    if (!textarea) return null;
    const caret = textarea.selectionStart ?? textarea.value.length;
    const beforeCaret = textarea.value.slice(0, caret);
    const match = beforeCaret.match(/^\s*\/([^\s\n]*)$/);
    return match ? match[1].toLowerCase() : null;
}

function setDraftFromSlash(template) {
    const value = `${template}\n`;
    setInputValue(value, value.length);
}

function setSlashCommandDraft(commandName) {
    const command = normalizeDirectSlashCommand(commandName);
    if (!command) return;
    setInputValue(command.template, command.template.length);
}

function buildSlashCommands() {
    return [
        {
            name: 'search',
            label: '/search',
            description: featureSettings.web_search
                ? 'Run a direct web search tool flow.'
                : 'Enable Web Search and start a direct web search tool flow.',
            kind: 'tool',
            keywords: ['search internet', 'browse', 'current', 'web'],
            onSelect: () => {
                if (!featureSettings.web_search) updateFeatureSetting('web_search', true);
                setSlashCommandDraft('search');
            },
        },
        {
            name: 'code',
            label: '/code',
            description: 'Run the direct code workflow: inspect, plan, edit, and verify.',
            kind: 'tool',
            keywords: ['change', 'implement', 'modify code', 'edit'],
            onSelect: () => {
                if (!featureSettings.agent_tools) updateFeatureSetting('agent_tools', true);
                setSlashCommandDraft('code');
            },
        },
        {
            name: 'pip',
            label: '/pip',
            description: 'Install Python packages into the managed chat environment and use them in follow-up commands.',
            kind: 'tool',
            keywords: ['python package', 'dependency', 'install library', 'pip'],
            onSelect: () => {
                if (!featureSettings.agent_tools) updateFeatureSetting('agent_tools', true);
                setSlashCommandDraft('pip');
            },
        },
        {
            name: 'grep',
            label: '/grep',
            description: 'Run workspace grep first, then summarize the relevant matches.',
            kind: 'tool',
            keywords: ['search code', 'find text', 'ripgrep'],
            onSelect: () => {
                if (!featureSettings.agent_tools) updateFeatureSetting('agent_tools', true);
                setSlashCommandDraft('grep');
            },
        },
        {
            name: 'plan',
            label: '/plan',
            description: 'Run the direct planning flow and prepare an executable plan draft.',
            kind: 'tool',
            keywords: ['approach', 'design', 'scoping'],
            onSelect: () => {
                if (!featureSettings.agent_tools) updateFeatureSetting('agent_tools', true);
                setSlashCommandDraft('plan');
            },
        },
    ];
}

function filterSlashCommands(query) {
    const normalized = (query || '').trim().toLowerCase();
    const commands = buildSlashCommands();
    if (!normalized) return commands;
    return commands.filter(command => {
        const haystack = [
            command.name,
            command.label,
            command.description,
            ...(command.keywords || []),
        ].join(' ').toLowerCase();
        return haystack.includes(normalized);
    });
}

function renderSlashMenu() {
    const menu = document.getElementById('slashMenu');
    const list = document.getElementById('slashMenuList');
    if (!menu || !list) return;

    if (!slashMenuItems.length) {
        menu.hidden = false;
        positionSlashMenu();
        list.innerHTML = '<div class="slash-menu-empty">No matching commands</div>';
        return;
    }

    menu.hidden = false;
    list.innerHTML = '';
    slashMenuItems.forEach((command, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'slash-menu-item' + (index === slashMenuSelectedIndex ? ' active' : '');
        button.innerHTML = `
            <div class="slash-menu-copy">
                <div class="slash-menu-command">${escapeHtml(command.label)}</div>
                <div class="slash-menu-description">${escapeHtml(command.description)}</div>
            </div>
            <div class="slash-menu-meta">${escapeHtml(SLASH_COMMAND_KIND_LABELS[command.kind] || 'Command')}</div>
        `;
        button.onmouseenter = () => {
            slashMenuSelectedIndex = index;
            renderSlashMenu();
        };
        button.onmousedown = (event) => {
            event.preventDefault();
            executeSlashCommand(index);
        };
        list.appendChild(button);
    });
    positionSlashMenu();
    scrollActiveSlashItemIntoView();
}

function hideSlashMenu() {
    slashMenuItems = [];
    slashMenuSelectedIndex = 0;
    const menu = document.getElementById('slashMenu');
    const list = document.getElementById('slashMenuList');
    if (menu) menu.hidden = true;
    if (list) list.innerHTML = '';
}

function positionSlashMenu() {
    const menu = document.getElementById('slashMenu');
    const inputArea = document.getElementById('inputArea');
    if (!menu || !inputArea || menu.hidden) return;

    const rect = inputArea.getBoundingClientRect();
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
    const safeMargin = 12;
    const spaceAbove = Math.max(140, Math.floor(rect.top - safeMargin));
    const spaceBelow = Math.max(140, Math.floor(viewportHeight - rect.bottom - safeMargin));
    const placeBelow = spaceBelow > spaceAbove && spaceBelow >= 220;
    const available = placeBelow ? spaceBelow : spaceAbove;

    menu.classList.toggle('slash-menu-below', placeBelow);
    menu.style.setProperty('--slash-menu-max-height', `${available}px`);
}

function scrollActiveSlashItemIntoView() {
    const list = document.getElementById('slashMenuList');
    const activeItem = list?.querySelector('.slash-menu-item.active');
    if (!list || !activeItem) return;

    const itemTop = activeItem.offsetTop;
    const itemBottom = itemTop + activeItem.offsetHeight;
    const viewTop = list.scrollTop;
    const viewBottom = viewTop + list.clientHeight;

    if (itemTop < viewTop) {
        list.scrollTop = itemTop;
        return;
    }
    if (itemBottom > viewBottom) {
        list.scrollTop = itemBottom - list.clientHeight;
    }
}

function updateSlashMenu(textarea = document.getElementById('input')) {
    const query = getSlashCommandQuery(textarea);
    if (query == null) {
        hideSlashMenu();
        return;
    }

    slashMenuItems = filterSlashCommands(query);
    if (slashMenuSelectedIndex >= slashMenuItems.length) {
        slashMenuSelectedIndex = 0;
    }
    renderSlashMenu();
}

function moveSlashSelection(direction) {
    if (!slashMenuItems.length) return;
    const total = slashMenuItems.length;
    slashMenuSelectedIndex = (slashMenuSelectedIndex + direction + total) % total;
    renderSlashMenu();
}

function executeSlashCommand(index = slashMenuSelectedIndex) {
    const command = slashMenuItems[index];
    if (!command) return;
    command.onSelect();
    updateSlashMenu();
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 1024) return `${bytes || 0} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatRelativeTime(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) return '';
    const diffMs = Date.now() - date.getTime();
    const diffMin = Math.round(diffMs / 60000);
    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.round(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.round(diffHr / 24);
    if (diffDay < 7) return `${diffDay}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function noteAssistantTurnArtifactPath(path) {
    if (!path) return;
    currentAssistantTurnArtifactPaths.add(path);
}

function isWorkspaceArtifactRailCandidate(entry) {
    if (!entry || entry.type !== 'file') return false;
    const path = String(entry.path || '').trim();
    if (!path) return false;
    const contentKind = String(entry.content_kind || '').toLowerCase();
    if (['image', 'html', 'pdf', 'csv', 'spreadsheet'].includes(contentKind)) return true;
    if (isMarkdownPath(path) && !/\.txt$/i.test(path)) return true;
    return false;
}

function noteAssistantArtifactsFromToolResult(data) {
    const payload = data?.payload || {};
    if (!payload || typeof payload !== 'object') return;
    if (typeof payload.path === 'string' && payload.path) {
        const entry = findWorkspaceFileEntry(payload.path) || { path: payload.path, type: 'file', content_kind: '' };
        if (isWorkspaceArtifactRailCandidate(entry)) noteAssistantTurnArtifactPath(payload.path);
    }
    if (typeof payload.open_path === 'string' && payload.open_path) {
        noteAssistantTurnArtifactPath(payload.open_path);
    }
    if (Array.isArray(payload.items)) {
        payload.items.forEach(item => {
            if (item && typeof item.path === 'string' && isWorkspaceArtifactRailCandidate(item)) {
                noteAssistantTurnArtifactPath(item.path);
            }
        });
    }
}

function finalizeAssistantTurnArtifacts() {
    latestAssistantTurnArtifactPaths = new Set(currentAssistantTurnArtifactPaths);
    currentAssistantTurnArtifactPaths = new Set();
}

function artifactBadgeForPath(path) {
    const lower = String(path || '').toLowerCase();
    if (/\.(html?)$/.test(lower)) return 'Web';
    if (/\.(png|jpg|jpeg|gif|svg|webp)$/.test(lower)) return 'Media';
    if (/\.(md|txt|pdf)$/.test(lower)) return 'Doc';
    if (/\.(json|ya?ml|toml|csv|tsv|xlsx|xls|xlsm)$/.test(lower)) return 'Data';
    if (/\.(js|jsx|ts|tsx|py|rb|go|rs|java|c|cc|cpp|hpp|css|sh|sql)$/.test(lower)) return 'Code';
    return 'File';
}

function sortWorkspaceFilesByRecent(entries) {
    return [...entries].sort((a, b) => {
        const aTime = new Date(a.modified_at || 0).getTime() || 0;
        const bTime = new Date(b.modified_at || 0).getTime() || 0;
        return bTime - aTime;
    });
}

function workspaceArtifactIconLabel(path) {
    const ext = ((path || '').split('.').pop() || '').trim().toUpperCase();
    if (ext && ext.length <= 4) return ext;
    if (isHtmlPath(path)) return 'WEB';
    if (isMarkdownPath(path)) return 'DOC';
    if (/\.(csv|tsv|xlsx|xls|xlsm|json|ya?ml|toml)$/i.test(path || '')) return 'DATA';
    if (/\.(png|jpg|jpeg|gif|svg|webp)$/i.test(path || '')) return 'IMG';
    if (/\.(js|jsx|ts|tsx|py|rb|go|rs|java|c|cc|cpp|hpp|css|sh|sql)$/i.test(path || '')) return 'CODE';
    return 'FILE';
}

function normalizeWorkspacePath(path) {
    const value = String(path || '.').trim();
    return value === '' || value === '.' ? '.' : value.replace(/^\.\/+/, '').replace(/\/+/g, '/');
}

function compareWorkspacePaths(a, b) {
    return normalizeWorkspacePath(a) === normalizeWorkspacePath(b);
}

function findWorkspaceFileEntry(path) {
    const normalized = normalizeWorkspacePath(path);
    return Array.isArray(workspaceEntries)
        ? workspaceEntries.find(entry => entry && entry.type === 'file' && compareWorkspacePaths(entry.path, normalized))
        : null;
}

function normalizeArtifactReferencePath(path) {
    return normalizeWorkspacePath(String(path || '').trim());
}

function resolveArtifactReferencePath(path) {
    const normalized = normalizeArtifactReferencePath(path);
    return findWorkspaceFileEntry(normalized)?.path || normalized;
}

function extractArtifactReferences(text) {
    const references = [];
    const seen = new Set();
    const value = String(text || '');
    for (const match of value.matchAll(ARTIFACT_REFERENCE_PATTERN)) {
        const normalized = normalizeArtifactReferencePath(match[1]);
        if (!normalized || seen.has(normalized)) continue;
        seen.add(normalized);
        references.push(normalized);
    }
    return references;
}

function assistantTurnIncludesArtifactPath(path) {
    const normalized = normalizeWorkspacePath(path);
    const sets = [currentAssistantTurnArtifactPaths, latestAssistantTurnArtifactPaths];
    return sets.some(paths => Array.from(paths || []).some(candidate => compareWorkspacePaths(candidate, normalized)));
}

function buildArtifactReferenceButton(path) {
    const resolvedPath = resolveArtifactReferencePath(path);
    const entry = findWorkspaceFileEntry(resolvedPath);
    const previewable = shouldAutoOpenArtifactPreview(resolvedPath);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `message-artifact-link${previewable ? ' is-previewable' : ''}`;
    button.dataset.path = resolvedPath;
    button.title = previewable
        ? `Open ${basename(resolvedPath) || resolvedPath} in the viewer`
        : `Open ${basename(resolvedPath) || resolvedPath} in the workspace`;
    button.setAttribute(
        'aria-label',
        previewable
            ? `Open ${basename(resolvedPath) || resolvedPath} in the viewer`
            : `Open ${basename(resolvedPath) || resolvedPath} in the workspace`,
    );
    button.onclick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        openWorkspaceFile(resolvedPath);
        focusInput();
    };

    const icon = document.createElement('span');
    icon.className = 'message-artifact-icon';
    icon.textContent = workspaceArtifactIconLabel(resolvedPath);
    button.appendChild(icon);

    const copy = document.createElement('span');
    copy.className = 'message-artifact-copy';

    const head = document.createElement('span');
    head.className = 'message-artifact-head';
    const badge = document.createElement('span');
    badge.className = 'message-artifact-badge';
    badge.textContent = previewable ? 'Viewer' : artifactBadgeForPath(resolvedPath);
    head.appendChild(badge);
    copy.appendChild(head);

    const title = document.createElement('span');
    title.className = 'message-artifact-title';
    title.textContent = entry?.name || basename(resolvedPath) || resolvedPath;
    copy.appendChild(title);

    const meta = document.createElement('span');
    meta.className = 'message-artifact-meta';
    meta.textContent = resolvedPath;
    copy.appendChild(meta);

    button.appendChild(copy);
    return button;
}

function replaceArtifactReferencesInTextNode(node) {
    const text = node?.textContent || '';
    if (!text || !text.includes('[[artifact:')) return false;
    const fragment = document.createDocumentFragment();
    let replaced = false;
    let lastIndex = 0;

    for (const match of text.matchAll(ARTIFACT_REFERENCE_PATTERN)) {
        const fullMatch = match[0] || '';
        const index = match.index || 0;
        const before = text.slice(lastIndex, index);
        if (before) fragment.appendChild(document.createTextNode(before));
        fragment.appendChild(buildArtifactReferenceButton(match[1]));
        lastIndex = index + fullMatch.length;
        replaced = true;
    }

    if (!replaced) return false;

    const after = text.slice(lastIndex);
    if (after) fragment.appendChild(document.createTextNode(after));
    node.parentNode?.replaceChild(fragment, node);
    return true;
}

function paragraphArtifactButtons(paragraph) {
    return Array.from(paragraph.children || []).filter(node => {
        return node instanceof HTMLElement && node.classList.contains('message-artifact-link');
    });
}

function artifactHelperParagraphText(paragraph) {
    const parts = [];
    paragraph.childNodes.forEach(node => {
        if (node.nodeType === Node.TEXT_NODE) {
            parts.push(node.textContent || '');
            return;
        }
        if (!(node instanceof HTMLElement) || node.classList.contains('message-artifact-link')) return;
        parts.push(node.textContent || '');
    });
    return parts.join(' ').replace(/\s+/g, ' ').trim();
}

function shouldCollapseArtifactHelperParagraph(paragraph) {
    const buttons = paragraphArtifactButtons(paragraph);
    if (buttons.length !== 1) return false;
    const helperText = artifactHelperParagraphText(paragraph);
    if (!helperText) return true;
    return ARTIFACT_HELPER_TEXT_PATTERNS.some(pattern => pattern.test(helperText));
}

function markStandaloneArtifactReferences(container) {
    if (!container) return;
    container.querySelectorAll('p').forEach(paragraph => {
        if (!shouldCollapseArtifactHelperParagraph(paragraph)) return;
        const [onlyNode] = paragraphArtifactButtons(paragraph);
        if (!(onlyNode instanceof HTMLElement)) return;
        paragraph.innerHTML = '';
        paragraph.appendChild(onlyNode);
        paragraph.classList.add('message-artifact-paragraph');
        onlyNode.classList.add('is-standalone');
    });
}

function isMostRecentAssistantMessage(msg) {
    if (!msg || !msg.classList.contains('assistant')) return false;
    const assistantMessages = Array.from(document.querySelectorAll('#messages .message.assistant'));
    return assistantMessages.length > 0 && assistantMessages[assistantMessages.length - 1] === msg;
}

function maybeAutoOpenReferencedArtifact(msg, rawContent = '') {
    if (!msg || msg.dataset.artifactPreviewAutoOpened === 'true') return;
    const previewPath = extractArtifactReferences(rawContent)
        .map(path => resolveArtifactReferencePath(path))
        .find(path => {
            if (!shouldAutoOpenArtifactPreview(path) || !findWorkspaceFileEntry(path)) return false;
            if (assistantTurnIncludesArtifactPath(path)) return true;
            return extractArtifactReferences(rawContent).length === 1 && isMostRecentAssistantMessage(msg);
        });
    if (!previewPath) return;
    msg.dataset.artifactPreviewAutoOpened = 'true';
    openWorkspaceFile(previewPath);
    requestAnimationFrame(() => focusInput());
}

function enhanceMessageArtifactReferences(msg, container, rawContent = '') {
    if (!container) return;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            const text = node?.textContent || '';
            if (!text.includes('[[artifact:')) return NodeFilter.FILTER_REJECT;
            const parent = node.parentElement;
            if (!parent || parent.closest('pre, code, a, button, textarea')) {
                return NodeFilter.FILTER_REJECT;
            }
            return NodeFilter.FILTER_ACCEPT;
        },
    });
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    textNodes.forEach(node => replaceArtifactReferencesInTextNode(node));
    markStandaloneArtifactReferences(container);
    maybeAutoOpenReferencedArtifact(msg, rawContent);
}

function getWorkspaceArtifactEntries(limit = WORKSPACE_ARTIFACT_LIMIT) {
    const files = Array.isArray(workspaceEntries)
        ? workspaceEntries.filter(entry => entry && entry.type === 'file')
        : [];
    if (!files.length) return [];

    const fileMap = new Map(files.map(entry => [normalizeWorkspacePath(entry.path), entry]));
    const seen = new Set();
    const ordered = [];
    const addEntryPath = (path) => {
        const normalized = normalizeWorkspacePath(path);
        const entry = fileMap.get(normalized);
        if (!entry || seen.has(normalized)) return;
        seen.add(normalized);
        ordered.push(entry);
    };

    if (selectedWorkspaceFile) addEntryPath(selectedWorkspaceFile);
    latestAssistantTurnArtifactPaths.forEach(path => addEntryPath(path));
    files
        .filter(entry => isWorkspaceArtifactRailCandidate(entry))
        .filter(entry => isWorkspaceEntryRecentlyChanged(entry))
        .sort((a, b) => {
            const aModified = new Date(a.modified_at || 0).getTime() || 0;
            const bModified = new Date(b.modified_at || 0).getTime() || 0;
            return bModified - aModified;
        })
        .forEach(entry => addEntryPath(entry.path));

    return ordered.slice(0, limit);
}

function renderWorkspaceArtifactList() {
    const listEl = document.getElementById('workspaceArtifactList');
    const countEl = document.getElementById('workspaceArtifactsCount');
    if (!listEl || !countEl) return;

    const files = Array.isArray(workspaceEntries)
        ? workspaceEntries.filter(entry => entry && entry.type === 'file')
        : [];
    const artifacts = getWorkspaceArtifactEntries();
    const fileCount = files.length;
    countEl.textContent = fileCount
        ? `${artifacts.length} recent${fileCount ? ` • ${fileCount} file${fileCount === 1 ? '' : 's'}` : ''}`
        : 'No files';

    if (!fileCount) {
        listEl.innerHTML = '<div class="workspace-empty">Waiting for the first artifact. Ask the assistant to create or edit a file and it will show up here.</div>';
        return;
    }

    if (!artifacts.length) {
        listEl.innerHTML = '<div class="workspace-empty">No recent turn artifacts yet. Browse All Files below or ask the assistant to create, edit, or run something.</div>';
        return;
    }

    listEl.innerHTML = '';
    artifacts.forEach(entry => {
        const card = document.createElement('article');
        const isActive = compareWorkspacePaths(entry.path, selectedWorkspaceFile);
        const isChanged = isWorkspaceEntryRecentlyChanged(entry);
        card.className = `workspace-artifact-card${isActive ? ' active' : ''}${isChanged ? ' is-live' : ''}`;

        const openButton = document.createElement('button');
        openButton.type = 'button';
        openButton.className = 'workspace-artifact-main';
        openButton.onclick = () => openWorkspaceFile(entry.path);

        const badge = artifactBadgeForPath(entry.path);
        const modified = formatRelativeTime(entry.modified_at);
        const meta = [formatBytes(entry.size || 0)];
        if (modified) meta.push(modified);
        openButton.innerHTML = `
            <span class="workspace-artifact-icon">${escapeHtml(workspaceArtifactIconLabel(entry.path))}</span>
            <span class="workspace-artifact-copy">
                <span class="workspace-artifact-head">
                    <span class="workspace-artifact-badge">${escapeHtml(badge)}</span>
                    ${isChanged ? '<span class="workspace-artifact-state">Live</span>' : ''}
                </span>
                <span class="workspace-artifact-title">${escapeHtml(entry.name || basename(entry.path) || entry.path)}</span>
                <span class="workspace-artifact-path">${escapeHtml(entry.path)}</span>
                <span class="workspace-artifact-meta">${escapeHtml(meta.join(' • '))}</span>
            </span>
        `;
        card.appendChild(openButton);

        const actions = document.createElement('div');
        actions.className = 'workspace-artifact-actions';

        const openAction = document.createElement('button');
        openAction.type = 'button';
        openAction.className = 'workspace-artifact-action';
        openAction.textContent = isActive ? 'Open' : 'Preview';
        openAction.onclick = (event) => {
            event.stopPropagation();
            openWorkspaceFile(entry.path);
        };
        actions.appendChild(openAction);

        const insertAction = document.createElement('button');
        insertAction.type = 'button';
        insertAction.className = 'workspace-artifact-action';
        insertAction.textContent = 'Insert';
        insertAction.onclick = (event) => {
            event.stopPropagation();
            insertArtifactReference(entry.path);
        };
        actions.appendChild(insertAction);

        card.appendChild(actions);
        listEl.appendChild(card);
    });
}

function isWorkspaceEntryRecentlyChanged(entry) {
    if (!entry || entry.type !== 'file') return false;
    return latestAssistantTurnArtifactPaths.has(entry.path)
        || currentAssistantTurnArtifactPaths.has(entry.path);
}

function flattenWorkspaceFiles(node, files = []) {
    if (!node) return files;
    if (node.type === 'file') {
        files.push(node);
        return files;
    }
    (node.children || []).forEach(child => flattenWorkspaceFiles(child, files));
    return files;
}

function collectWorkspaceStats(node) {
    const totals = { files: 0, directories: 0 };
    if (!node) return totals;
    const walk = current => {
        if (current.type === 'directory') {
            if (!compareWorkspacePaths(current.path, '.')) totals.directories += 1;
            (current.children || []).forEach(walk);
            return;
        }
        totals.files += 1;
    };
    walk(node);
    return totals;
}

function toggleWorkspaceDirectory(path) {
    const normalized = normalizeWorkspacePath(path);
    if (compareWorkspacePaths(normalized, '.')) return;
    if (collapsedWorkspaceDirs.has(normalized)) collapsedWorkspaceDirs.delete(normalized);
    else collapsedWorkspaceDirs.add(normalized);
    renderWorkspaceTrees();
}

function workspaceTreeNodeMeta(entry) {
    if (entry.type === 'directory') {
        const childCount = Array.isArray(entry.children) ? entry.children.length : 0;
        if (!childCount) return '';
        return `${childCount} item${childCount === 1 ? '' : 's'}`;
    }
    const parts = [formatBytes(entry.size || 0)];
    if (entry.modified_at) parts.push(formatRelativeTime(entry.modified_at));
    return parts.join(' • ');
}

function renderWorkspaceTreeNode(entry, options = {}) {
    const node = document.createElement('div');
    const isDirectory = entry.type === 'directory';
    const normalizedPath = normalizeWorkspacePath(entry.path);
    const collapsed = isDirectory && !compareWorkspacePaths(normalizedPath, '.') && collapsedWorkspaceDirs.has(normalizedPath);
    const isActive = !isDirectory && compareWorkspacePaths(entry.path, selectedWorkspaceFile);
    const isChanged = isWorkspaceEntryRecentlyChanged(entry);
    node.className = `workspace-tree-node ${isDirectory ? 'is-directory' : 'is-file'}${isActive ? ' active' : ''}${isChanged ? ' is-changed' : ''}`;

    const row = document.createElement('div');
    row.className = 'workspace-tree-row';
    row.style.setProperty('--workspace-depth', String(options.depth || 0));
    const mainButton = document.createElement('button');
    mainButton.type = 'button';
    mainButton.className = 'workspace-tree-main';
    mainButton.onclick = (event) => {
        event.stopPropagation();
        if (isDirectory) {
            toggleWorkspaceDirectory(entry.path);
            return;
        }
        openWorkspaceFile(entry.path);
    };

    const caret = document.createElement('span');
    caret.className = `workspace-tree-caret${isDirectory ? '' : ' is-placeholder'}${collapsed ? ' is-collapsed' : ''}`;
    caret.innerHTML = isDirectory ? '&#9656;' : '&#8226;';
    mainButton.appendChild(caret);

    const icon = document.createElement('span');
    icon.className = 'workspace-tree-icon';
    icon.innerHTML = isDirectory ? (collapsed ? '&#128193;' : '&#128194;') : '&#128196;';
    mainButton.appendChild(icon);

    const copy = document.createElement('span');
    copy.className = 'workspace-tree-copy';
    const metaText = workspaceTreeNodeMeta(entry);
    copy.innerHTML = `
        <span class="workspace-tree-name">${escapeHtml(entry.name || basename(entry.path) || 'workspace')}</span>
        ${metaText ? `<span class="workspace-tree-meta">${escapeHtml(metaText)}</span>` : ''}
    `;
    mainButton.appendChild(copy);

    if (isChanged) {
        const badge = document.createElement('span');
        badge.className = 'workspace-tree-status';
        badge.textContent = 'Live';
        mainButton.appendChild(badge);
    }

    row.appendChild(mainButton);
    if (!isDirectory) {
        const actions = document.createElement('div');
        actions.className = 'workspace-tree-actions';
        const downloadButton = document.createElement('button');
        downloadButton.type = 'button';
        downloadButton.className = 'workspace-tree-action';
        downloadButton.title = `Download ${entry.name || basename(entry.path) || 'file'}`;
        downloadButton.setAttribute('aria-label', `Download ${entry.name || basename(entry.path) || 'file'}`);
        downloadButton.innerHTML = '&#8681;';
        downloadButton.onclick = (event) => {
            event.stopPropagation();
            downloadWorkspaceFile(entry.path);
        };
        actions.appendChild(downloadButton);
        row.appendChild(actions);
    }

    node.appendChild(row);

    if (isDirectory && !collapsed && Array.isArray(entry.children) && entry.children.length) {
        const children = document.createElement('div');
        children.className = 'workspace-tree-children';
        entry.children.forEach(child => {
            children.appendChild(renderWorkspaceTreeNode(child, { ...options, depth: (options.depth || 0) + 1 }));
        });
        node.appendChild(children);
    }

    return node;
}

function renderWorkspaceTree(containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!workspaceTree) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = '';
    const rootChildren = Array.isArray(workspaceTree.children) ? workspaceTree.children : [];
    rootChildren.forEach(child => {
        container.appendChild(renderWorkspaceTreeNode(child, { depth: 0, ...options }));
    });
}

function renderWorkspaceTrees() {
    renderWorkspaceTree('workspaceFileList');
}

function isSpreadsheetPath(path) {
    return /\.(csv|tsv|xlsx|xls|xlsm)$/i.test(path || '');
}

function isHtmlPath(path) {
    return /\.(html?)$/i.test(path || '');
}

function isMarkdownPath(path) {
    return /\.(md|markdown)$/i.test(path || '');
}

function basename(path) {
    const parts = String(path || '').split('/');
    return parts[parts.length - 1] || path || '';
}

function insertAtCursor(text) {
    const input = document.getElementById('input');
    if (!input) return;
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? input.value.length;
    const before = input.value.slice(0, start);
    const after = input.value.slice(end);
    input.value = before + text + after;
    const caret = start + text.length;
    input.setSelectionRange(caret, caret);
    input.focus();
    autoResizeTextarea(input);
    syncSendButton();
}

function insertArtifactReference(path) {
    const token = `[[artifact:${path}]]`;
    const input = document.getElementById('input');
    const needsLeadingSpace = input && input.value && !/\s$/.test(input.value);
    insertAtCursor(`${needsLeadingSpace ? ' ' : ''}${token}`);
}

function isEditableTextPath(path) {
    return !/\.(pdf|xlsx|xls|xlsm|png|jpg|jpeg|gif|svg|webp)$/i.test(path || '');
}

function isImagePath(path) {
    return /\.(png|jpg|jpeg|gif|svg|webp)$/i.test(path || '');
}

function normalizeViewerMetadata(path, data = {}) {
    const backendKind = typeof data.content_kind === 'string' ? data.content_kind.toLowerCase() : '';
    const kind = (
        ['text', 'markdown', 'html', 'csv', 'pdf', 'spreadsheet', 'image'].includes(backendKind)
            ? backendKind
            : (isImagePath(path) ? 'image' : (isHtmlPath(path) ? 'html' : (isMarkdownPath(path) ? 'markdown' : (/\.(csv|tsv)$/i.test(path || '') ? 'csv' : 'text'))))
    );
    const backendEditable = typeof data.editable === 'boolean' ? data.editable : isEditableTextPath(path);
    const editable = LIVE_AREA_READ_ONLY ? false : backendEditable;
    const defaultView = data.default_view === 'edit' && editable ? 'edit' : 'preview';
    return { kind, editable, defaultView };
}

function getLanguageForPath(path) {
    const ext = (path.split('.').pop() || '').toLowerCase();
    const map = {
        js: 'javascript',
        mjs: 'javascript',
        cjs: 'javascript',
        ts: 'typescript',
        jsx: 'javascript',
        tsx: 'typescript',
        py: 'python',
        rb: 'ruby',
        go: 'go',
        rs: 'rust',
        java: 'java',
        c: 'c',
        h: 'c',
        cpp: 'cpp',
        cc: 'cpp',
        hpp: 'cpp',
        cs: 'csharp',
        php: 'php',
        swift: 'swift',
        kt: 'kotlin',
        html: 'xml',
        htm: 'xml',
        xml: 'xml',
        css: 'css',
        json: 'json',
        yml: 'yaml',
        yaml: 'yaml',
        md: 'markdown',
        sh: 'bash',
        sql: 'sql',
    };
    return map[ext] || '';
}

function renderBuildSteps() {
    renderExecutionPlanApproval();
    syncStreamingAssistantStatusPanels();
}

function clearBuildSteps() {
    buildSteps = [];
    currentBuildStepIndex = null;
    currentPlanFocus = null;
    renderBuildSteps();
    syncInterveneButton();
}

function normalizeBuildStepStatePayload(steps, activeIndex = null, completedCount = 0, stepDetails = []) {
    return (steps || []).map((text, index) => {
        const provided = stepDetails[index] && typeof stepDetails[index] === 'object' ? stepDetails[index] : {};
        let state = 'pending';
        if (index < completedCount) state = 'complete';
        else if (activeIndex !== null && index === activeIndex) state = 'active';
        const nestedSteps = Array.isArray(provided.substeps) ? provided.substeps : [];
        return {
            text,
            state,
            goal: String(provided.goal || '').trim(),
            successSignal: String(provided.success_signal || '').trim(),
            progressNote: String(provided.progress_note || '').trim(),
            completedNotes: Array.isArray(provided.completed_notes)
                ? provided.completed_notes.map(item => String(item || '').trim()).filter(Boolean)
                : [],
            completedReports: Array.isArray(provided.completed_reports)
                ? provided.completed_reports.filter(item => item && typeof item === 'object')
                : [],
            substeps: nestedSteps.map(item => ({
                text: String(item?.text || '').trim(),
                state: String(item?.state || 'pending').trim() || 'pending',
                report: item?.report && typeof item.report === 'object' ? item.report : {},
            })).filter(item => item.text),
        };
    });
}

function updateBuildSteps(steps, activeIndex = null, completedCount = 0, stepDetails = []) {
    buildSteps = normalizeBuildStepStatePayload(steps, activeIndex, completedCount, stepDetails);
    currentBuildStepIndex = activeIndex !== null ? activeIndex : null;
    renderBuildSteps();
    requestAnimationFrame(scrollActivePlanStepIntoView);
    syncInterveneButton();
}

function stepStateMarker(state) {
    if (state === 'complete') return '✓';
    if (state === 'active') return '•';
    return '○';
}

function formatSubstepReportMeta(report = {}) {
    if (!report || typeof report !== 'object') return '';
    const parts = [];
    if (Number.isFinite(report.tool_calls)) parts.push(`${report.tool_calls} tools`);
    if (Number.isFinite(report.successful_tools)) parts.push(`${report.successful_tools} ok tools`);
    if (Number.isFinite(report.successful_commands) && report.successful_commands > 0) {
        parts.push(`${report.successful_commands} commands`);
    }
    if (Array.isArray(report.paths) && report.paths.length) {
        parts.push(report.paths.slice(0, 2).join(', '));
    }
    return parts.join(' • ');
}

function ensureStreamingAssistantStatusStack(msg) {
    const stack = msg?.querySelector('.think-stack');
    if (!stack) return null;
    let statusStack = stack.querySelector('.turn-status-stack');
    if (!statusStack) {
        statusStack = document.createElement('div');
        statusStack.className = 'turn-status-stack';
        stack.prepend(statusStack);
    }
    return statusStack;
}

function cleanupStreamingAssistantStatusStack(msg) {
    const statusStack = msg?.querySelector('.turn-status-stack');
    if (statusStack && !statusStack.children.length) {
        statusStack.remove();
    }
}

function buildInlineStatusHeading(titleText, metaText = '') {
    const heading = document.createElement('div');
    heading.className = 'turn-status-heading';

    const title = document.createElement('span');
    title.className = 'turn-status-title';
    title.textContent = titleText;
    heading.appendChild(title);

    if (metaText) {
        const meta = document.createElement('span');
        meta.className = 'turn-status-meta';
        meta.textContent = metaText;
        heading.appendChild(meta);
    }

    return heading;
}

function ensureInlineStatusBox(statusStack, className, { prepend = false } = {}) {
    if (!statusStack) return null;
    let box = statusStack.querySelector(`.${className}`);
    if (!box) {
        box = document.createElement('section');
        box.className = `turn-status-box ${className}`;
        if (prepend && statusStack.firstChild) statusStack.insertBefore(box, statusStack.firstChild);
        else statusStack.appendChild(box);
    }
    return box;
}

function renderInlinePlanSummary(msg = currentStreamingAssistantMessage()) {
    if (!msg) return;
    const statusStack = ensureStreamingAssistantStatusStack(msg);
    if (!statusStack) return;

    let box = statusStack.querySelector('.turn-plan-box');
    if (!buildSteps.length) {
        box?.remove();
        cleanupStreamingAssistantStatusStack(msg);
        return;
    }

    box = ensureInlineStatusBox(statusStack, 'turn-plan-box', { prepend: true });
    box.innerHTML = '';

    const completedCount = buildSteps.filter(step => step.state === 'complete').length;
    const totalCount = buildSteps.length;
    box.appendChild(buildInlineStatusHeading(
        completedCount >= totalCount ? 'Plan Complete' : 'Plan',
        `${completedCount}/${totalCount} complete`,
    ));

    const list = document.createElement('ol');
    list.className = 'turn-plan-list';
    buildSteps.forEach(step => {
        const item = document.createElement('li');
        item.className = `turn-plan-item is-${step.state || 'pending'}`;

        const marker = document.createElement('span');
        marker.className = 'turn-plan-marker';
        marker.textContent = stepStateMarker(step.state || 'pending');
        item.appendChild(marker);

        const text = document.createElement('span');
        text.className = 'turn-plan-text';
        text.textContent = step.text || '';
        item.appendChild(text);

        list.appendChild(item);
    });
    box.appendChild(list);
}

function shouldShowInlineWorkspaceActivityEntry(entry) {
    if (!entry) return false;
    if (['Request', 'Turn', 'Interrupt'].includes(entry.kind)) return false;
    return true;
}

function renderInlineActivitySummary(msg = currentStreamingAssistantMessage()) {
    if (!msg) return;
    const statusStack = ensureStreamingAssistantStatusStack(msg);
    if (!statusStack) return;

    const entries = workspaceActivity
        .filter(shouldShowInlineWorkspaceActivityEntry)
        .slice(-INLINE_ACTIVITY_LIMIT);

    let box = statusStack.querySelector('.turn-activity-box');
    if (!isGenerating || !entries.length) {
        box?.remove();
        cleanupStreamingAssistantStatusStack(msg);
        return;
    }

    box = ensureInlineStatusBox(statusStack, 'turn-activity-box');
    box.innerHTML = '';
    box.appendChild(buildInlineStatusHeading('Activity', isGenerating ? 'Live' : 'Recent'));

    const list = document.createElement('div');
    list.className = 'turn-activity-list';
    entries.forEach(entry => {
        const row = document.createElement('div');
        row.className = `turn-activity-item is-${entry.phase || 'status'}${entry.error ? ' is-error' : ''}`;

        const meta = document.createElement('div');
        meta.className = 'turn-activity-meta';
        meta.textContent = entry.stepLabel
            ? `${entry.kind} · ${entry.stepLabel}`
            : `${entry.kind}`;
        row.appendChild(meta);

        const text = document.createElement('div');
        text.className = 'turn-activity-text';
        text.textContent = truncateWorkspaceActivityText(String(entry.text || '').trim(), 140);
        row.appendChild(text);

        list.appendChild(row);
    });
    box.appendChild(list);
}

function syncStreamingAssistantStatusPanels(msg = currentStreamingAssistantMessage()) {
    if (!msg) return;
    renderInlinePlanSummary(msg);
    renderInlineActivitySummary(msg);
}

function renderExecutionPlanApproval() {
    const panel = document.getElementById('planApprovalPanel');
    const title = document.getElementById('planApprovalTitle');
    const subtitle = document.getElementById('planApprovalSubtitle');
    const callout = document.getElementById('planApprovalCallout');
    const dismissButton = document.getElementById('planApprovalDismiss');
    const actions = document.getElementById('planApprovalActions');
    const approveButton = document.getElementById('planApprovalApproveButton');
    const editButton = document.getElementById('planApprovalEditButton');
    const editor = document.getElementById('planApprovalEditor');
    const draft = document.getElementById('planApprovalDraft');
    const editActions = document.getElementById('planApprovalEditActions');
    const cancelEditButton = document.getElementById('planApprovalCancelEditButton');
    const runEditedButton = document.getElementById('planApprovalRunEditedButton');
    const reasoning = document.getElementById('planApprovalReasoning');
    const summary = document.getElementById('planApprovalSummary');
    const currentBox = document.getElementById('planApprovalCurrent');
    const currentStep = document.getElementById('planApprovalCurrentStep');
    const currentText = document.getElementById('planApprovalCurrentText');
    const checklist = document.getElementById('planApprovalChecklist');
    const checklistList = document.getElementById('planApprovalSteps');
    if (!panel || !summary || !checklist || !checklistList) return;
    if (DOCUMENT_CENTERED_MODE) {
        panel.hidden = true;
        return;
    }

    const hasPlan = Boolean(pendingExecutionPlan?.executePrompt);
    panel.hidden = !hasPlan;

    if (!hasPlan) {
        if (actions) actions.hidden = true;
        if (editor) editor.hidden = true;
        if (editActions) editActions.hidden = true;
        if (callout) callout.hidden = true;
        if (reasoning) reasoning.hidden = true;
        if (currentBox) currentBox.hidden = true;
        checklist.hidden = true;
        checklistList.innerHTML = '';
        if (dismissButton) dismissButton.hidden = true;
        return;
    }

    const isEditing = Boolean(pendingExecutionPlan?.isEditing);
    const sourceSteps = isEditing ? getExecutionPlanDraftSteps() : getExecutionPlanSourceSteps();
    const canRun = sourceSteps.length > 0 && !isGenerating && modelAvailable;
    const stepCount = sourceSteps.length;

    if (title) title.textContent = 'Approval Required';
    if (subtitle) {
        subtitle.textContent = isEditing
            ? 'Edit the execution steps here. Approving the edited plan will run them in the workspace for this request.'
            : 'Review the execution steps here. Approving will run them in the workspace for this request.';
    }
    if (callout) {
        callout.hidden = false;
        if (!stepCount) {
            callout.textContent = isEditing
                ? 'Edit the execution steps, then approve the revised plan to run it in the workspace for this request.'
                : 'Approving will run this plan in the workspace for this request.';
        } else if (isEditing) {
            callout.textContent = stepCount === 1
                ? 'You are editing 1 execution step. Approve Edited Plan will run it in the workspace for this request.'
                : `You are editing ${stepCount} execution steps. Approve Edited Plan will run them in the workspace for this request.`;
        } else {
            callout.textContent = stepCount === 1
                ? 'Approving will run this 1-step plan in the workspace for this request.'
                : `Approving will run these ${stepCount} steps in the workspace for this request.`;
        }
    }

    if (actions) actions.hidden = isEditing;
    if (approveButton) {
        approveButton.disabled = !canRun;
        approveButton.textContent = 'Approve And Run';
    }
    if (editButton) {
        editButton.disabled = isGenerating;
        editButton.textContent = 'Edit Steps';
    }

    if (editor) editor.hidden = !isEditing;
    if (draft) {
        const nextDraftText = hasPlan ? String(pendingExecutionPlan?.draftText || '') : '';
        if (draft.value !== nextDraftText) {
            draft.value = nextDraftText;
        }
        autoResizeTextarea(draft);
    }
    if (editActions) editActions.hidden = !isEditing;
    if (cancelEditButton) {
        cancelEditButton.disabled = isGenerating;
        cancelEditButton.textContent = 'Cancel Edit';
    }
    if (runEditedButton) {
        runEditedButton.disabled = !canRun;
        runEditedButton.textContent = 'Approve Edited Plan';
    }

    const hasSummary = Boolean(pendingExecutionPlan?.summary);
    if (reasoning) reasoning.hidden = !hasSummary;
    summary.textContent = hasSummary ? pendingExecutionPlan.summary : '';

    if (currentBox) currentBox.hidden = true;
    if (currentStep) currentStep.textContent = '';
    if (currentText) currentText.textContent = '';
    checklist.hidden = isEditing || !sourceSteps.length;
    checklistList.innerHTML = '';
    if (!isEditing && sourceSteps.length) {
        sourceSteps.forEach((stepText, index) => {
            const item = document.createElement('li');
            item.className = 'is-pending';
            item.innerHTML = `
                <span class="plan-approval-step-marker">${index + 1}.</span>
                <span class="plan-approval-step-label">${escapeHtml(stepText)}</span>
            `;
            checklistList.appendChild(item);
        });
    }

    if (dismissButton) dismissButton.hidden = false;
}

function handleExecutionPlanDraftChange(textarea) {
    if (pendingExecutionPlan) {
        pendingExecutionPlan.draftText = textarea.value;
    }
    autoResizeTextarea(textarea);
    syncSendButton();
}

function handleExecutionPlanDraftKeyDown(event) {
    if (event.key === 'Escape') {
        event.preventDefault();
        cancelExecutionPlanEdit();
        return;
    }
    if (event.key === 'ArrowDown') {
        const target = event.target;
        const atEnd = (target.selectionStart ?? 0) === target.value.length && (target.selectionEnd ?? 0) === target.value.length;
        if (atEnd) {
            event.preventDefault();
            focusInputAtStart();
            return;
        }
    }
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        approveExecutionPlan();
    }
}

function startExecutionPlanEdit() {
    if (!pendingExecutionPlan || isGenerating) return;
    pendingExecutionPlan.isEditing = true;
    renderExecutionPlanApproval();
    requestAnimationFrame(() => {
        focusExecutionPlanDraft('end');
    });
}

function cancelExecutionPlanEdit() {
    if (!pendingExecutionPlan) {
        focusInputAtStart();
        return;
    }
    pendingExecutionPlan.isEditing = false;
    pendingExecutionPlan.draftText = formatExecutionPlanDraft(getExecutionPlanSourceSteps());
    renderExecutionPlanApproval();
    if (!focusExecutionPlanApproveButton()) {
        focusInputAtStart();
    }
}

function storeExecutionPlan(summary, executePrompt, builderSteps = [], options = {}) {
    const cleanedPrompt = String(executePrompt || '').trim();
    if (!cleanedPrompt) return;
    const normalizedBuilderSteps = normalizeExecutionPlanSteps(builderSteps);
    const normalizedSourceSteps = normalizedBuilderSteps.length
        ? normalizedBuilderSteps
        : extractExecutionPlanStepsFromPrompt(cleanedPrompt);
    pendingExecutionPlan = {
        summary: String(summary || '').trim(),
        executePrompt: cleanedPrompt,
        builderSteps: normalizedSourceSteps,
        draftText: formatExecutionPlanDraft(normalizedSourceSteps),
        isEditing: false,
    };
    renderExecutionPlanApproval();
    syncSendButton();
    if (DOCUMENT_CENTERED_MODE) {
        window.setTimeout(() => approveExecutionPlan(), 0);
        return;
    }

    if (options.focusComposer !== false) {
        const input = document.getElementById('input');
        const hasDraft = Boolean(input && input.value.trim());
        if (!hasDraft && !pendingAttachments.length) {
            requestAnimationFrame(() => {
                if (!focusExecutionPlanApproveButton()) {
                    focusInputAtStart();
                }
            });
        }
    }
}

function dismissExecutionPlan() {
    pendingExecutionPlan = null;
    renderExecutionPlanApproval();
    syncSendButton();
}

function approveExecutionPlan() {
    const builderSteps = getExecutionPlanDraftSteps();
    if (!pendingExecutionPlan?.executePrompt || !builderSteps.length || isGenerating || !modelAvailable) {
        return;
    }

    const turnFeatures = resolveTurnFeatures(APPROVED_PLAN_EXECUTION_MESSAGE, [], null);
    if (!turnFeatures.workspace_write) return;
    rememberTurnFeatures(currentConvId, turnFeatures);

    hideSlashMenu();
    document.querySelector('.welcome')?.remove();
    exitWelcomeMode();
    showLoading();
    setLoadingText('Execution plan approved. Starting workspace execution...');
    clearBuildSteps();
    clearWorkspaceActivity();
    recordWorkspaceActivity('Approval', 'Execution plan approved. Starting workspace execution.');

    isGenerating = true;
    activeStreamConversationId = currentConvId;
    streamingAssistantMessage = null;
    const clientTurnId = beginClientTurn();
    dismissExecutionPlan();
    syncSendButton();
    dispatchChatPayload({
        message: APPROVED_PLAN_EXECUTION_MESSAGE,
        attachments: [],
        conversation_id: currentConvId,
        workspace_id: currentWorkspaceId || null,
        client_turn_id: clientTurnId,
        system_prompt: localStorage.getItem('customSystemPrompt') || null,
        mode: BASE_REASONING_MODE,
        features: turnFeatures,
        plan_override_steps: builderSteps,
        slash_command: null,
    }, {
        fallbackStatusText: 'WebSocket unavailable. Running the approved plan over HTTP fallback...',
        fallbackActivityText: 'WebSocket unavailable. Running the approved plan over HTTP fallback.',
    });
}

function renderWorkspaceActivity() {
    const activityEl = document.getElementById('workspaceActivityList');
    if (!activityEl) return;

    if (!featureSettings.agent_tools) {
        activityEl.innerHTML = '<div class="workspace-empty">Agent dev tools are turned off.</div>';
        return;
    }

    if (!workspaceActivity.length) {
        activityEl.innerHTML = '<div class="workspace-empty">The assistant\'s workspace activity will appear here during a turn.</div>';
        return;
    }

    activityEl.innerHTML = '';
    workspaceActivity.forEach(entry => {
        const row = document.createElement('div');
        row.className = `workspace-activity-item${entry.error ? ' is-error' : ''}${entry.phase ? ` is-${entry.phase}` : ''}`;
        const repeatBadge = entry.repeats > 1
            ? `<span class="workspace-activity-repeat">${entry.collapsedSummary ? `${entry.repeats} events` : `x${entry.repeats}`}</span>`
            : '';
        const stepMeta = entry.stepLabel
            ? `<span class="workspace-activity-sep" aria-hidden="true">·</span><span class="workspace-activity-step">${escapeHtml(entry.stepLabel)}</span>`
            : '';
        row.innerHTML = `
            <span class="workspace-activity-time">${escapeHtml(entry.time || '')}</span>
            <span class="workspace-activity-sep" aria-hidden="true">·</span>
            <span class="workspace-activity-kind">${escapeHtml(entry.kind || 'Activity')}</span>
            ${repeatBadge ? `<span class="workspace-activity-sep" aria-hidden="true">·</span>${repeatBadge}` : ''}
            ${stepMeta}
            <span class="workspace-activity-sep" aria-hidden="true">·</span>
            <span class="workspace-activity-text">${escapeHtml(entry.text || '')}</span>
        `;
        activityEl.appendChild(row);
    });
    activityEl.scrollTop = activityEl.scrollHeight;
}

function clearWorkspaceActivity() {
    workspaceActivity = [];
    renderWorkspaceActivity();
    syncStreamingAssistantStatusPanels();
}

function classifyWorkspaceActivity(kind, text) {
    const normalizedKind = String(kind || '').trim();
    const normalizedText = String(text || '').trim();
    const lower = normalizedText.toLowerCase();

    if (normalizedKind === 'Status') {
        if (lower.includes('analyzing query')) return { kind: 'Analyze', phase: 'analyze' };
        if (lower.includes('inspecting workspace')) return { kind: 'Inspect', phase: 'inspect' };
        if (lower.includes('planning deep mode')) return { kind: 'Plan', phase: 'plan' };
        if (lower.includes('build step') || lower.includes('building from task board')) return { kind: 'Execute', phase: 'execute' };
        if (lower.includes('verifying workspace') || lower.includes('quality check')) return { kind: 'Verify', phase: 'verify' };
        if (lower.includes('synthesizing from artifacts')) return { kind: 'Synthesize', phase: 'synthesize' };
        if (lower.includes('audit')) return { kind: 'Audit', phase: 'audit' };
        if (lower.includes('thinking carefully') || lower === 'thinking...') return { kind: 'Think', phase: 'think' };
    }

    if (normalizedKind === 'Tool') {
        if (lower.startsWith('inspect:')) return { kind: 'Inspect Tool', phase: 'inspect' };
        if (lower.startsWith('build ')) return { kind: 'Build Tool', phase: 'execute' };
        if (lower.startsWith('verify:')) return { kind: 'Verify Tool', phase: 'verify' };
        if (lower.startsWith('synthesize:')) return { kind: 'Synthesize Tool', phase: 'synthesize' };
        return { kind: 'Tool', phase: 'tool' };
    }

    if (normalizedKind === 'Command') return { kind: 'Command', phase: 'execute' };
    if (normalizedKind === 'Explore') return { kind: 'Explore', phase: 'inspect' };
    if (normalizedKind === 'Edit') return { kind: 'Edit', phase: 'execute' };
    if (normalizedKind === 'Render') return { kind: 'Render', phase: 'execute' };
    if (normalizedKind === 'History') return { kind: 'History', phase: 'inspect' };
    if (normalizedKind === 'Web') return { kind: 'Web', phase: 'respond' };

    if (normalizedKind === 'Tool Result') {
        if (lower.startsWith('inspect:')) return { kind: 'Inspect Result', phase: 'inspect' };
        if (lower.startsWith('build ')) return { kind: 'Build Result', phase: 'execute' };
        if (lower.startsWith('verify:')) return { kind: 'Verify Result', phase: 'verify' };
        if (lower.startsWith('synthesize:')) return { kind: 'Synthesize Result', phase: 'synthesize' };
        return { kind: 'Tool Result', phase: 'tool' };
    }

    if (normalizedKind === 'Tool Error') return { kind: 'Tool Error', phase: 'error' };
    if (normalizedKind === 'Plan') return { kind: 'Plan', phase: 'plan' };
    if (normalizedKind === 'Note') return { kind: 'Note', phase: 'note' };
    if (normalizedKind === 'Draft') return { kind: 'Draft', phase: 'synthesize' };
    if (normalizedKind === 'Done') return { kind: 'Done', phase: 'done' };
    if (normalizedKind === 'Error') return { kind: 'Error', phase: 'error' };
    if (normalizedKind === 'Turn') return { kind: 'Turn', phase: 'status' };
    if (normalizedKind === 'Request') return { kind: 'Request', phase: 'status' };
    if (normalizedKind === 'Interrupt') return { kind: 'Interrupt', phase: 'status' };
    if (normalizedKind === 'Canceled') return { kind: 'Canceled', phase: 'status' };
    return { kind: normalizedKind || 'Status', phase: 'status' };
}

function currentBuildStepLabel() {
    if (currentBuildStepIndex === null || currentBuildStepIndex === undefined) return '';
    const step = buildSteps[currentBuildStepIndex];
    if (!step || !step.text) return `Step ${currentBuildStepIndex + 1}`;
    const activeSubstepIndex = Array.isArray(step.substeps)
        ? step.substeps.findIndex(item => item?.state === 'active')
        : -1;
    if (activeSubstepIndex >= 0) {
        const activeSubstep = step.substeps[activeSubstepIndex];
        if (activeSubstep?.text) {
            return `Step ${currentBuildStepIndex + 1}.${activeSubstepIndex + 1}: ${activeSubstep.text}`;
        }
    }
    return `Step ${currentBuildStepIndex + 1}: ${step.text}`;
}

function updatePlanFocus({ text = '', stepLabel = '', phase = 'status' } = {}) {
    const normalizedText = compactWorkspaceActivityText(text || '');
    const normalizedStepLabel = compactWorkspaceActivityText(stepLabel || currentBuildStepLabel());
    if (!normalizedText && !normalizedStepLabel) return;
    currentPlanFocus = {
        text: normalizedText,
        stepLabel: normalizedStepLabel,
        phase: String(phase || 'status').trim() || 'status',
    };
    renderExecutionPlanApproval();
}

function summarizeCompletedStep(step = {}) {
    const reports = Array.isArray(step?.completedReports) ? step.completedReports : [];
    const notes = Array.isArray(step?.completedNotes) ? step.completedNotes : [];
    const reportSummary = String(reports[reports.length - 1]?.summary || '').trim();
    if (reportSummary) return reportSummary;
    return String(notes[notes.length - 1] || '').trim();
}

function collectRecentStepActivity(stepIndex, limit = 3) {
    const prefix = `step ${stepIndex + 1}`;
    return workspaceActivity
        .filter(entry => {
            const label = String(entry?.stepLabel || '').trim().toLowerCase();
            return label === prefix || label.startsWith(`${prefix}:`) || label.startsWith(`${prefix}.`);
        })
        .slice(-limit);
}

function formatPlanActivityMeta(entry = {}) {
    const parts = [];
    if (entry.kind) parts.push(String(entry.kind));
    if (entry.time) parts.push(String(entry.time));
    return parts.join(' • ');
}

function scrollActivePlanStepIntoView() {
    const checklist = document.getElementById('planApprovalChecklist');
    if (!checklist || checklist.hidden) return;
    const activeItem = checklist.querySelector('.plan-approval-steps > li.is-active, .plan-approval-substeps li.is-active');
    activeItem?.scrollIntoView({ block: 'nearest' });
}

function shouldCollapseWorkspaceActivity(entry) {
    return ['tool', 'inspect', 'execute', 'verify', 'synthesize'].includes(entry.phase)
        && /tool|result/i.test(entry.kind);
}

function collapseWorkspaceActivityEntries(entries, nextEntry) {
    if (!entries.length) return entries.concat(nextEntry);
    const previous = entries[entries.length - 1];
    const sameSignature = previous.kind === nextEntry.kind
        && previous.phase === nextEntry.phase
        && previous.text === nextEntry.text
        && previous.stepLabel === nextEntry.stepLabel
        && previous.error === nextEntry.error;
    if (sameSignature) {
        const merged = {
            ...previous,
            repeats: (previous.repeats || 1) + 1,
            time: nextEntry.time,
        };
        return entries.slice(0, -1).concat(merged);
    }

    if (shouldCollapseWorkspaceActivity(previous) && shouldCollapseWorkspaceActivity(nextEntry)) {
        if (previous.stepLabel !== nextEntry.stepLabel) {
            return entries.concat(nextEntry);
        }
        const previousCount = previous.repeats || 1;
        const merged = {
            ...nextEntry,
            kind: `${nextEntry.kind}`,
            repeats: previousCount + 1,
            collapsedSummary: true,
        };
        return entries.slice(0, -1).concat(merged);
    }

    return entries.concat(nextEntry);
}

function compactWorkspaceActivityText(text) {
    return String(text || '').replace(/\s+/g, ' ').trim();
}

function truncateWorkspaceActivityText(text, maxLength = 180) {
    if (text.length <= maxLength) return text;
    return `${text.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

function formatWorkspaceActivityText(kind, text) {
    const normalized = compactWorkspaceActivityText(text);
    if (!normalized) return '';
    if (kind === 'Request' || kind === 'Interrupt') {
        return truncateWorkspaceActivityText(normalized, 160);
    }
    return normalized;
}

function recordWorkspaceActivity(kind, text, options = {}) {
    const normalizedKind = String(kind || 'Activity').trim() || 'Activity';
    const normalizedText = formatWorkspaceActivityText(normalizedKind, text);
    if (!normalizedText) return;
    const timestamp = new Date();
    const classified = options.phase
        ? { kind: normalizedKind, phase: options.phase }
        : classifyWorkspaceActivity(normalizedKind, normalizedText);
    const entry = {
        kind: classified.kind,
        phase: classified.phase,
        text: normalizedText,
        error: Boolean(options.error),
        stepLabel: compactWorkspaceActivityText(options.stepLabel || (
            ['execute', 'tool', 'verify', 'synthesize'].includes(classified.phase)
                ? currentBuildStepLabel()
                : ''
        )),
        repeats: 1,
        time: timestamp.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' }),
    };
    workspaceActivity = collapseWorkspaceActivityEntries(workspaceActivity, entry).slice(-WORKSPACE_ACTIVITY_LIMIT);
    renderWorkspaceActivity();
    syncStreamingAssistantStatusPanels();
}

function renderFileList() {
    const listEl = document.getElementById('workspaceFileList');
    const pathEl = document.getElementById('workspacePath');
    if (!listEl || !pathEl) return;

    if (!workspaceTree) {
        pathEl.textContent = 'Ready for artifacts';
        listEl.innerHTML = '';
        renderWorkspaceArtifactList();
        return;
    }

    const summary = [];
    if (workspaceStats.files > 0) summary.push(`${workspaceStats.files} file${workspaceStats.files === 1 ? '' : 's'}`);
    if (workspaceStats.directories > 0) summary.push(`${workspaceStats.directories} folder${workspaceStats.directories === 1 ? '' : 's'}`);
    pathEl.textContent = summary.join(' • ') || 'Ready for artifacts';
    renderWorkspaceArtifactList();
    renderWorkspaceTrees();
}

function getCodeMirrorModeForPath(path) {
    if (!(window.CodeMirror && Array.isArray(window.CodeMirror.modeInfo))) return null;
    return window.CodeMirror.findModeByFileName(path || '') || null;
}

function draftProfileKeyForPath(path) {
    const normalized = String(path || '').trim().toLowerCase();
    if (/\.(html?|xhtml)$/.test(normalized)) return 'html';
    if (/\.(md|markdown|mdx|rst)$/.test(normalized)) return 'markdown';
    if (/\.py$/.test(normalized)) return 'python';
    if (/\.json$/.test(normalized)) return 'json';
    return 'text';
}

function getDraftProfileDefinition(profileKey = activeDraftProfileKey) {
    return DRAFT_PROFILE_DEFINITIONS[profileKey] || DRAFT_PROFILE_DEFINITIONS.text;
}

function normalizeDraftFilename(path) {
    const normalized = normalizeWorkspacePath(path || '');
    return normalized === '.' ? DEFAULT_DRAFT_FILENAME : normalized;
}

function draftStorageSlugForTargetPath(path) {
    return normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME)
        .replace(/[\\/]+/g, '__')
        .replace(/[^a-zA-Z0-9._-]+/g, '_')
        .replace(/^_+|_+$/g, '') || 'draft';
}

function draftSpecPathForTargetPath(path) {
    return `.ai-chat/drafts/${draftStorageSlugForTargetPath(path)}.draft.md`;
}

function draftVersionDirectoryForTargetPath(path) {
    return `.ai-chat/versions/${draftStorageSlugForTargetPath(path)}`;
}

async function loadDraftFileSessions(workspaceId = currentWorkspaceId) {
    const safeWorkspaceId = String(workspaceId || '').trim();
    if (!safeWorkspaceId) {
        draftFileSessions = [];
        draftSessionDefaultPath = '';
        renderDraftFileExplorer();
        return [];
    }
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(safeWorkspaceId)}/file-sessions`);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    draftFileSessions = Array.isArray(data.sessions) ? data.sessions : [];
    draftSessionDefaultPath = normalizeDraftFilename(data.default_path || DEFAULT_DRAFT_FILENAME);
    renderDraftFileExplorer();
    return draftFileSessions;
}

async function ensureDraftFileSession(path, preferredConversationId = '') {
    if (!currentWorkspaceId) throw new Error('Choose a workspace first.');
    const normalizedPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file-sessions/ensure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            path: normalizedPath,
            preferred_conversation_id: String(preferredConversationId || '').trim() || null,
        }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    await loadDraftFileSessions(currentWorkspaceId).catch(() => {});
    return data;
}

function getDraftHistoryForWorkspace(workspaceId = currentWorkspaceId) {
    const safeWorkspaceId = String(workspaceId || '').trim();
    if (!safeWorkspaceId) return [];
    return draftFileSessions
        .filter(item => String(item?.workspace_id || '').trim() === safeWorkspaceId)
        .map(item => normalizeDraftFilename(item?.path || ''))
        .filter(Boolean);
}

function persistDraftHistoryEntry(workspaceId, path) {
    void workspaceId;
    void path;
}

function getDraftSessionRecord(workspaceId = currentWorkspaceId, path = activeDraftPath) {
    const safeWorkspaceId = String(workspaceId || '').trim();
    const normalizedPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    if (!safeWorkspaceId || !normalizedPath) return null;
    return draftFileSessions.find(item =>
        String(item?.workspace_id || '').trim() === safeWorkspaceId
        && compareWorkspacePaths(item?.path || '', normalizedPath)
    ) || null;
}

function getDraftSessionConversationId(workspaceId = currentWorkspaceId, path = activeDraftPath) {
    return String(getDraftSessionRecord(workspaceId, path)?.conversation_id || '').trim();
}

function getDraftSessionPathForConversation(workspaceId = currentWorkspaceId, conversationId = currentConvId) {
    const safeWorkspaceId = String(workspaceId || '').trim();
    const safeConversationId = String(conversationId || '').trim();
    if (!safeWorkspaceId || !safeConversationId) return '';
    const match = draftFileSessions.find(item =>
        String(item?.workspace_id || '').trim() === safeWorkspaceId
        && String(item?.conversation_id || '').trim() === safeConversationId
    );
    return normalizeDraftFilename(match?.path || '');
}

function persistDraftSession(workspaceId = currentWorkspaceId, path = activeDraftPath, conversationId = currentConvId) {
    const safeWorkspaceId = String(workspaceId || '').trim();
    const normalizedPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    const safeConversationId = String(conversationId || '').trim();
    if (!safeWorkspaceId || !normalizedPath || !safeConversationId) return;
    ensureDraftFileSession(normalizedPath, safeConversationId).catch(error => {
        console.error('Failed to persist draft file session:', error);
    });
}

async function createDraftFileJob({
    path = activeDraftPath,
    lane = 'foreground',
    jobKind = 'realize_draft',
    title = '',
    payload = {},
    status = 'queued',
    sourceConversationId = currentConvId,
    supersedeMatchingKind = false,
} = {}) {
    if (!currentWorkspaceId) throw new Error('Choose a workspace first.');
    const normalizedPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file-session-jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            path: normalizedPath,
            lane,
            job_kind: jobKind,
            title,
            payload,
            status,
            source_conversation_id: String(sourceConversationId || '').trim() || null,
            supersede_matching_kind: Boolean(supersedeMatchingKind),
        }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    if (data.session?.conversation_id) {
        currentConvId = String(data.session.conversation_id).trim() || currentConvId;
        syncToolApprovalToggle();
    }
    await loadDraftFileSessions(currentWorkspaceId).catch(() => {});
    return data;
}

async function updateDraftFileJobStatus(jobId, status, errorText = '') {
    const safeJobId = String(jobId || '').trim();
    if (!currentWorkspaceId || !safeJobId) return null;
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file-session-jobs/${encodeURIComponent(safeJobId)}/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            status,
            error_text: String(errorText || '').trim() || null,
        }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    await loadDraftFileSessions(currentWorkspaceId).catch(() => {});
    return data.job || null;
}

function adoptDraftSessionConversation(path = activeDraftPath) {
    const session = getDraftSessionRecord(currentWorkspaceId, path);
    const nextConversationId = String(session?.conversation_id || '').trim();
    if (!nextConversationId || nextConversationId === currentConvId) return false;
    currentConvId = nextConversationId;
    syncToolApprovalToggle();
    return true;
}

async function syncCurrentConversationTitleToDraft(path = activeDraftPath) {
    const normalizedPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    const conversationId = String(currentConvId || '').trim();
    if (!conversationId || !normalizedPath) return;
    try {
        await fetch(`/api/conversation/${encodeURIComponent(conversationId)}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: normalizedPath }),
        });
    } catch (error) {
        console.debug('Draft session title sync skipped:', error);
    }
}

function renderDraftFileExplorer() {
    const listEl = document.getElementById('draftFileExplorerList');
    if (!listEl) return;
    const history = getDraftHistoryForWorkspace();
    if (!currentWorkspaceId) {
        listEl.innerHTML = '<div class="workspace-catalog-empty">Choose a workspace first.</div>';
        return;
    }
    if (!history.length) {
        listEl.innerHTML = '<div class="workspace-catalog-empty">No files yet.</div>';
        return;
    }
    listEl.innerHTML = '';
    history.forEach(path => {
        const item = document.createElement('div');
        const isActive = compareWorkspacePaths(path, activeDraftPath || '');
        const hasSession = Boolean(getDraftSessionConversationId(currentWorkspaceId, path));
        item.className = `conv-item is-file${isActive ? ' active' : ''}`;
        item.innerHTML = `
            <div class="conv-title">${escapeHtml(basename(path) || path)}${hasSession ? '<span class="conv-file-badge">Session</span>' : ''}</div>
            <div class="conv-preview">${escapeHtml(path)}</div>
        `;
        item.onclick = () => {
            switchToDraftFile(path).catch(error => {
                alert(`Couldn't open that file: ${error.message}`);
            });
            closeFileExplorer();
        };
        listEl.appendChild(item);
    });
}

async function listDraftVersionRows(targetPath = activeDraftPath) {
    const normalizedPath = normalizeDraftFilename(targetPath || DEFAULT_DRAFT_FILENAME);
    if (!currentWorkspaceId || !normalizedPath) return [];
    const currentContent = await readWorkspaceTextFile(normalizedPath);
    const listing = await fetchWorkspaceDirectory(draftVersionDirectoryForTargetPath(normalizedPath)).catch(() => ({ items: [] }));
    const items = Array.isArray(listing.items) ? listing.items.filter(item => item.type === 'file') : [];
    const sortedItems = items.slice().sort((a, b) => String(b.modified_at || '').localeCompare(String(a.modified_at || '')));
    const rows = [{
        title: 'Current output',
        path: normalizedPath,
        previewPath: normalizedPath,
        meta: currentContent ? 'Live file' : 'Waiting for the first generated result',
        empty: !currentContent,
        current: true,
    }];
    sortedItems.forEach(item => {
        rows.push({
            title: new Date(item.modified_at || Date.now()).toLocaleString(),
            path: item.path,
            previewPath: item.path,
            meta: item.path,
            empty: false,
            current: false,
        });
    });
    return rows;
}

async function refreshDraftVersionsSidebar() {
    const listEl = document.getElementById('draftVersionsSidebarList');
    const fileLabelEl = document.getElementById('draftVersionMenuFileLabel');
    if (!listEl || !fileLabelEl) return;
    const targetPath = normalizeDraftFilename(activeDraftPath || DEFAULT_DRAFT_FILENAME);
    fileLabelEl.textContent = basename(targetPath) || targetPath;
    if (!currentWorkspaceId || !targetPath) {
        listEl.innerHTML = '<div class="workspace-catalog-empty">Choose a file first.</div>';
        return;
    }
    listEl.innerHTML = '<div class="workspace-catalog-empty">Loading versions...</div>';
    const rows = await listDraftVersionRows(targetPath);
    if (!rows.length) {
        listEl.innerHTML = '<div class="workspace-catalog-empty">No saved versions yet.</div>';
        return;
    }
    listEl.innerHTML = '';
    rows.forEach(row => {
        const item = document.createElement('div');
        const active = compareWorkspacePaths(draftOutputPreviewPath || targetPath, row.previewPath);
        item.className = `conv-item is-version${active ? ' active' : ''}`;
        item.innerHTML = `
            <div class="conv-title">${escapeHtml(row.title)}</div>
            <div class="conv-preview">${escapeHtml(row.meta)}</div>
            <div class="conv-timestamp">${row.current ? 'Current' : 'Snapshot'}</div>
        `;
        item.onclick = () => {
            draftOutputPreviewPath = row.previewPath;
            refreshDraftOutputPane().catch(error => {
                console.error('Failed to preview that version:', error);
            });
            closeMenu();
        };
        listEl.appendChild(item);
    });
}

async function readWorkspaceTextFile(path) {
    if (!currentWorkspaceId || !path) return '';
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file?path=${encodeURIComponent(path)}`);
    const data = await resp.json().catch(() => ({}));
    if (resp.status === 404) return '';
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    return typeof data.content === 'string' ? data.content : '';
}

async function writeWorkspaceTextFile(path, content) {
    if (!currentWorkspaceId) throw new Error('Choose a workspace before saving.');
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, content }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    return data;
}

function draftVersionFilenameForTargetPath(path, timestamp = new Date()) {
    const normalizedPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    const ext = normalizedPath.includes('.') ? normalizedPath.slice(normalizedPath.lastIndexOf('.')) : '.txt';
    const iso = timestamp.toISOString().replace(/[:.]/g, '-');
    return `${draftVersionDirectoryForTargetPath(normalizedPath)}/${iso}${ext}`;
}

async function syncGeneratedOutputBaseline(path = activeDraftPath) {
    if (!currentWorkspaceId || !path) {
        activeDraftGeneratedContent = '';
        return '';
    }
    activeDraftGeneratedContent = await readWorkspaceTextFile(path);
    return activeDraftGeneratedContent;
}

async function snapshotGeneratedOutputIfChanged(path = activeDraftPath) {
    if (!currentWorkspaceId || !path) return '';
    const nextContent = await readWorkspaceTextFile(path);
    if (nextContent === activeDraftGeneratedContent) return nextContent;
    if (activeDraftGeneratedContent) {
        await writeWorkspaceTextFile(
            draftVersionFilenameForTargetPath(path),
            activeDraftGeneratedContent,
        );
    }
    activeDraftGeneratedContent = nextContent;
    return nextContent;
}

async function refreshDraftOutputPane() {
    const previewEl = document.getElementById('draftOutputPreview');
    const pathEl = document.getElementById('draftOutputPath');
    const stateEl = document.getElementById('draftOutputState');
    if (!previewEl || !pathEl || !stateEl) return;

    const targetPath = normalizeDraftFilename(activeDraftPath || DEFAULT_DRAFT_FILENAME);
    const previewPath = String(draftOutputPreviewPath || targetPath || '').trim();
    if (!currentWorkspaceId || !previewPath) {
        pathEl.textContent = basename(targetPath) || targetPath || 'Generated file';
        stateEl.textContent = 'Waiting';
        previewEl.innerHTML = '<div class="workspace-empty">The generated file will appear here as the agent works.</div>';
        return;
    }

    const viewingCurrent = compareWorkspacePaths(previewPath, targetPath);
    pathEl.textContent = basename(previewPath) || previewPath;
    stateEl.textContent = viewingCurrent ? 'Live' : 'Snapshot';

    const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file?path=${encodeURIComponent(previewPath)}`);
    const data = await resp.json().catch(() => ({}));
    if (resp.status === 404) {
        previewEl.innerHTML = '<div class="workspace-empty">No generated file yet. Keep drafting and the agent will build it here.</div>';
        return;
    }
    if (!resp.ok) {
        throw new Error(data.detail || `HTTP ${resp.status}`);
    }

    const metadata = normalizeViewerMetadata(previewPath, data);
    const content = typeof data.content === 'string' ? data.content : '';
    previewEl.innerHTML = '';
    previewEl.classList.remove('file-modal-markdown');
    if (metadata.kind === 'pdf') renderPdfPreview('draftOutputPreview', previewPath);
    else if (metadata.kind === 'image') renderImagePreview('draftOutputPreview', previewPath);
    else if (metadata.kind === 'html') renderHtmlPreview('draftOutputPreview', content, previewPath);
    else if (metadata.kind === 'markdown') renderMarkdownPreview('draftOutputPreview', content);
    else if (metadata.kind === 'csv') renderDelimitedPreview('draftOutputPreview', content, previewPath);
    else renderCodePreview('draftOutputPreview', content, previewPath);
}

function draftSubtitleForProfile(profileKey, path) {
    const normalizedPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    const workspaceName = currentWorkspaceMeta?.display_name || currentWorkspaceMeta?.root_path || 'this workspace';
    if (profileKey === 'html') {
        return `Drafting ${normalizedPath} in ${workspaceName}. This editor is the natural-language outline for the HTML file the agent generates on your behalf.`;
    }
    if (profileKey === 'markdown') {
        return `Drafting ${normalizedPath} in ${workspaceName}. This editor is the outline for the Markdown file the agent generates on your behalf.`;
    }
    if (profileKey === 'python') {
        return `Drafting ${normalizedPath} in ${workspaceName}. This editor is the outline for the Python file the agent generates on your behalf.`;
    }
    if (profileKey === 'json') {
        return `Drafting ${normalizedPath} in ${workspaceName}. This editor is the outline for the JSON file the agent generates on your behalf.`;
    }
    return `Drafting ${normalizedPath} in ${workspaceName}. This editor is the outline for the generated file, not the final output itself.`;
}

function previewDraftFilenameState(pathValue) {
    const profileKey = draftProfileKeyForPath(pathValue || DEFAULT_DRAFT_FILENAME);
    const badge = document.getElementById('draftProfileBadge');
    const workspaceBadge = document.getElementById('draftWorkspaceBadge');
    const subtitle = document.getElementById('draftShellSubtitle');
    if (badge) badge.textContent = `Auto: ${getDraftProfileDefinition(profileKey).label}`;
    if (workspaceBadge) workspaceBadge.textContent = currentWorkspaceMeta?.display_name || 'No workspace';
    if (subtitle) subtitle.textContent = draftSubtitleForProfile(profileKey, pathValue || activeDraftPath || DEFAULT_DRAFT_FILENAME);
    refreshWelcomeMarkupIfNeeded();
}

function defaultDraftPathForWorkspace(workspaceId = currentWorkspaceId) {
    const safeWorkspaceId = String(workspaceId || '').trim();
    if (safeWorkspaceId && safeWorkspaceId === currentWorkspaceId) {
        const sessionPath = getDraftHistoryForWorkspace(safeWorkspaceId)[0] || '';
        if (sessionPath) return normalizeDraftFilename(sessionPath);
        if (draftSessionDefaultPath) return normalizeDraftFilename(draftSessionDefaultPath);
        if (activeDraftPath) return normalizeDraftFilename(activeDraftPath);
    }
    return normalizeDraftFilename(DEFAULT_DRAFT_FILENAME);
}

function persistDraftSelection(workspaceId, path) {
    void workspaceId;
    void path;
}

async function switchToDraftFile(path, options = {}) {
    const nextPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    const changingFiles = !compareWorkspacePaths(nextPath, activeDraftPath || '');
    const preferredConversationId = changingFiles && hasConversationMessages() ? '' : currentConvId;
    const session = await ensureDraftFileSession(nextPath, preferredConversationId);
    const nextSessionId = String(session?.conversation_id || '').trim();

    if (nextSessionId && nextSessionId !== currentConvId) {
        await loadConversation(nextSessionId, { filePath: nextPath, preserveActivity: false });
        return;
    }

    await loadDraftDocument(nextPath);
}

function syncDraftHeader(path = activeDraftPath) {
    const filenameInput = document.getElementById('draftFilename');
    const normalizedPath = normalizeDraftFilename(path || DEFAULT_DRAFT_FILENAME);
    if (filenameInput && document.activeElement !== filenameInput) {
        filenameInput.value = normalizedPath;
    }
    previewDraftFilenameState(normalizedPath);
}

function setDraftSaveState(state, text = '') {
    draftSaveState = state;
    const el = document.getElementById('draftSaveState');
    if (!el) return;
    el.classList.remove('is-dirty', 'is-saving', 'is-saved', 'is-error');
    if (state === 'dirty') {
        el.textContent = text || 'Draft';
        el.classList.add('is-dirty');
        return;
    }
    if (state === 'saving') {
        el.textContent = text || 'Saving';
        el.classList.add('is-saving');
        return;
    }
    if (state === 'saved') {
        el.textContent = text || 'Saved';
        el.classList.add('is-saved');
        return;
    }
    if (state === 'error') {
        el.textContent = text || 'Error';
        el.classList.add('is-error');
        return;
    }
    el.textContent = text || 'Idle';
}

function refreshWelcomeMarkupIfNeeded() {
    const messages = document.getElementById('messages');
    if (!messages || hasConversationMessages()) return;
    messages.innerHTML = buildWelcomeMarkup();
}

function buildDraftStarterContent(path, profileKey = draftProfileKeyForPath(path)) {
    const profile = getDraftProfileDefinition(profileKey);
    const starter = profile.starter;
    return typeof starter === 'function' ? String(starter(path) || '') : '';
}

function syncDraftView() {
    const editTab = document.getElementById('draftEditTab');
    const previewTab = document.getElementById('draftPreviewTab');
    const editPane = document.getElementById('draftEditPane');
    const previewPane = document.getElementById('draftPreviewPane');
    const editing = draftView !== 'preview';
    if (editTab) editTab.classList.toggle('active', editing);
    if (previewTab) previewTab.classList.toggle('active', !editing);
    if (editPane) editPane.classList.toggle('file-modal-pane-hidden', !editing);
    if (previewPane) previewPane.classList.toggle('file-modal-pane-hidden', editing);
    if (editing) {
        const editor = ensureDraftEditor();
        if (editor) setTimeout(() => editor.refresh(), 0);
    } else {
        renderDraftPreview();
    }
}

function setDraftView(view) {
    draftView = DOCUMENT_CENTERED_MODE ? 'edit' : (view === 'preview' ? 'preview' : 'edit');
    localStorage.setItem('draftView', draftView);
    syncDraftView();
}

function configureDraftEditorLanguage(path) {
    const editor = ensureDraftEditor();
    if (!editor) return;
    const modeInfo = getCodeMirrorModeForPath(path);
    editor.setOption('lineWrapping', Boolean(getDraftProfileDefinition(draftProfileKeyForPath(path)).lineWrapping));
    if (!modeInfo) {
        editor.setOption('mode', null);
        return;
    }
    editor.setOption('mode', modeInfo.mime || modeInfo.mode || null);
    if (window.CodeMirror.autoLoadMode && modeInfo.mode) {
        window.CodeMirror.autoLoadMode(editor, modeInfo.mode);
    }
}

function configureInlineEditorLanguage(path) {
    const editor = ensureInlineEditor();
    if (!editor) return;
    const modeInfo = getCodeMirrorModeForPath(path);
    if (!modeInfo) {
        editor.setOption('mode', null);
        return;
    }
    editor.setOption('mode', modeInfo.mime || modeInfo.mode || null);
    if (window.CodeMirror.autoLoadMode && modeInfo.mode) {
        window.CodeMirror.autoLoadMode(editor, modeInfo.mode);
    }
}

function ensureDraftEditor() {
    if (draftEditorReady) return draftEditor;
    const textarea = document.getElementById('draftEditor');
    if (!textarea) return null;
    if (!(window.CodeMirror && typeof window.CodeMirror.fromTextArea === 'function')) return null;
    if (window.CodeMirror.modeURL === undefined) {
        window.CodeMirror.modeURL = 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/%N/%N.min.js';
    }
    draftEditor = window.CodeMirror.fromTextArea(textarea, {
        lineNumbers: true,
        keyMap: 'default',
        lineWrapping: true,
        indentUnit: 2,
        tabSize: 2,
        viewportMargin: Infinity,
        gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
        foldGutter: true,
        autoCloseBrackets: true,
        matchBrackets: true,
        styleActiveLine: true,
    });
    textarea.dataset.editorReady = 'true';
    draftEditor.on('change', () => {
        if (draftApplyingRemoteContent) return;
        handleDraftEditorInput();
    });
    draftEditor.on('focus', () => {
        if (draftSaveState === 'saved') setDraftSaveState('idle');
    });
    draftEditorReady = true;
    configureDraftEditorLanguage(activeDraftPath || DEFAULT_DRAFT_FILENAME);
    return draftEditor;
}

function getDraftContent() {
    const editor = ensureDraftEditor();
    if (editor) return editor.getValue();
    const textarea = document.getElementById('draftEditor');
    return textarea ? (textarea.value || '') : '';
}

function setDraftContent(content) {
    const editor = ensureDraftEditor();
    if (editor) {
        draftApplyingRemoteContent = true;
        if (editor.getValue() !== (content || '')) editor.setValue(content || '');
        draftApplyingRemoteContent = false;
        configureDraftEditorLanguage(activeDraftPath || DEFAULT_DRAFT_FILENAME);
        editor.refresh();
        return;
    }
    const textarea = document.getElementById('draftEditor');
    if (textarea) textarea.value = content || '';
}

function renderDraftPreview() {
    const content = getDraftContent();
    const profile = getDraftProfileDefinition(activeDraftProfileKey);
    if (profile.kind === 'html') {
        renderHtmlPreview('draftPreview', content, activeDraftPath || DEFAULT_DRAFT_FILENAME);
        return;
    }
    if (profile.kind === 'markdown') {
        renderMarkdownPreview('draftPreview', content);
        return;
    }
    renderCodePreview('draftPreview', content, activeDraftPath || DEFAULT_DRAFT_FILENAME);
}

function handleDraftEditorInput() {
    renderDraftPreview();
    scheduleDraftAutosave();
}

function scheduleDraftAutosave() {
    const content = getDraftContent();
    if (content === draftLastSavedContent) {
        if (draftAutosaveTimer) {
            clearTimeout(draftAutosaveTimer);
            draftAutosaveTimer = null;
        }
        setDraftSaveState('saved');
        return;
    }
    setDraftSaveState('dirty');
    if (draftAutosaveTimer) clearTimeout(draftAutosaveTimer);
    draftAutosaveTimer = setTimeout(() => {
        draftAutosaveTimer = null;
        saveDraftDocument({ autosave: true }).catch(() => {});
    }, DRAFT_AUTOSAVE_DELAY_MS);
}

async function saveDraftDocument(options = {}) {
    if (!currentWorkspaceId) throw new Error('Choose a workspace before saving a draft.');
    const saveButton = document.getElementById('draftSaveButton');
    const draftPath = normalizeDraftFilename(document.getElementById('draftFilename')?.value || activeDraftPath || DEFAULT_DRAFT_FILENAME);
    const specPath = draftSpecPathForTargetPath(draftPath);
    const content = getDraftContent();
    activeDraftPath = draftPath;
    activeDraftProfileKey = draftProfileKeyForPath(draftPath);
    syncDraftHeader(draftPath);
    if (saveButton) saveButton.disabled = true;
    setDraftSaveState('saving', options.autosave ? 'Autosaving' : 'Saving');
    try {
        const data = await writeWorkspaceTextFile(specPath, content);
        const session = await ensureDraftFileSession(draftPath, currentConvId).catch(error => {
            console.error('Failed to sync file session after saving:', error);
            return null;
        });
        if (session?.conversation_id) {
            currentConvId = String(session.conversation_id).trim() || currentConvId;
            syncToolApprovalToggle();
        }
        draftLastSavedContent = content;
        persistDraftSelection(currentWorkspaceId, draftPath);
        persistDraftHistoryEntry(currentWorkspaceId, draftPath);
        renderDraftFileExplorer();
        setDraftSaveState('saved');
        if (options.autosave) scheduleWorkspaceRefresh();
        else await refreshWorkspace(false);
        if (!options.skipAgentSync) scheduleDraftAgentSync();
        return data;
    } catch (error) {
        setDraftSaveState('error');
        if (!options.autosave) throw error;
        console.error('Draft autosave failed:', error);
        return null;
    } finally {
        if (saveButton) saveButton.disabled = false;
    }
}

async function loadDraftDocument(path, options = {}) {
    const draftPath = normalizeDraftFilename(path || defaultDraftPathForWorkspace());
    const profileKey = draftProfileKeyForPath(draftPath);
    const previousPath = activeDraftPath;
    const specPath = draftSpecPathForTargetPath(draftPath);
    const requestToken = ++draftOpenRequestToken;
    activeDraftPath = draftPath;
    activeDraftProfileKey = profileKey;
    draftOutputPreviewPath = draftPath;
    adoptDraftSessionConversation(draftPath);
    if (!compareWorkspacePaths(previousPath || '', draftPath)) {
        draftView = DOCUMENT_CENTERED_MODE ? 'edit' : (getDraftProfileDefinition(profileKey).defaultView || 'edit');
        localStorage.setItem('draftView', draftView);
    }
    syncDraftHeader(draftPath);
    persistDraftHistoryEntry(currentWorkspaceId, draftPath);
    renderDraftFileExplorer();
    configureDraftEditorLanguage(draftPath);
    if (!currentWorkspaceId) {
        const starter = buildDraftStarterContent(draftPath, profileKey);
        setDraftContent(starter);
        draftLastSavedContent = '';
        draftLastAgentSpecContent = starter;
        renderDraftPreview();
        setDraftSaveState('idle');
        syncDraftView();
        refreshDraftVersionsSidebar().catch(() => {});
        refreshDraftOutputPane().catch(() => {});
        return;
    }
    try {
        const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file?path=${encodeURIComponent(specPath)}`);
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
            if (resp.status !== 404) throw new Error(data.detail || `HTTP ${resp.status}`);
            const starter = buildDraftStarterContent(draftPath, profileKey);
            if (requestToken !== draftOpenRequestToken) return;
            setDraftContent(starter);
            draftLastSavedContent = '';
            draftLastAgentSpecContent = starter;
            renderDraftPreview();
            setDraftSaveState(starter ? 'dirty' : 'idle');
            syncDraftView();
            await syncGeneratedOutputBaseline(draftPath);
            refreshDraftVersionsSidebar().catch(() => {});
            refreshDraftOutputPane().catch(() => {});
            return;
        }
        if (requestToken !== draftOpenRequestToken) return;
        const content = typeof data.content === 'string' ? data.content : '';
        setDraftContent(content);
        draftLastSavedContent = content;
        draftLastAgentSpecContent = options.preserveAgentState ? draftLastAgentSpecContent : content;
        renderDraftPreview();
        setDraftSaveState('saved');
        syncDraftView();
        await syncGeneratedOutputBaseline(draftPath);
        refreshDraftVersionsSidebar().catch(() => {});
        refreshDraftOutputPane().catch(() => {});
    } catch (error) {
        console.error('Failed to load draft document:', error);
        if (requestToken !== draftOpenRequestToken) return;
        setDraftContent('');
        draftLastSavedContent = '';
        draftLastAgentSpecContent = '';
        renderDraftPreview();
        setDraftSaveState('error');
        refreshDraftVersionsSidebar().catch(() => {});
        refreshDraftOutputPane().catch(() => {});
    }
}

function handleDraftFilenameInput(value) {
    previewDraftFilenameState(value || DEFAULT_DRAFT_FILENAME);
}

async function commitDraftFilename() {
    const input = document.getElementById('draftFilename');
    const nextPath = normalizeDraftFilename(input?.value || DEFAULT_DRAFT_FILENAME);
    if (compareWorkspacePaths(nextPath, activeDraftPath || '')) {
        syncDraftHeader(nextPath);
        return;
    }
    if (activeDraftPath && getDraftContent() !== draftLastSavedContent) {
        try {
            await saveDraftDocument({ autosave: true });
        } catch (error) {
            console.error('Failed to save the previous draft before switching files:', error);
        }
    }
    await switchToDraftFile(nextPath);
}

function handleDraftFilenameKeyDown(event) {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    commitDraftFilename().catch(error => {
        alert(`Couldn't open that draft: ${error.message}`);
    });
}

function seedChatFromDraft() {
    const input = document.getElementById('input');
    if (!input) return;
    const prompt = [
        `Please work on the active draft \`${activeDraftPath || DEFAULT_DRAFT_FILENAME}\`.`,
        '',
        'Use the draft as the main working document, keep its current file type and structure, and make the smallest useful change that matches my request.'
    ].join('\n');
    setInputValue(prompt, prompt.length);
    focusInput();
}

function buildDraftRealizationPrompt() {
    const targetPath = normalizeDraftFilename(activeDraftPath || DEFAULT_DRAFT_FILENAME);
    const specPath = draftSpecPathForTargetPath(targetPath);
    return [
        `Read the active draft spec at \`${specPath}\` and realize it into the target file \`${targetPath}\`.`,
        '',
        'Treat the visible draft as a wish list, outline, and intent document rather than finished code.',
        'Translate the natural-language draft into a proper file of the same type as the target filename.',
        'Create or update the target file directly in the workspace, but do not overwrite the spec file.',
        'Preserve any good existing structure in the target file and make the smallest useful revision that keeps the file coherent.',
        'Use web search when it would materially improve the result, especially for factual content, examples, references, or content expansion.',
    ].join('\n');
}

function draftAgentHintForProfile(profileKey) {
    if (profileKey === 'html') {
        return 'The visible draft is a natural-language outline for an HTML file. Read the spec, then generate semantic HTML in the target file.';
    }
    if (profileKey === 'markdown') {
        return 'The visible draft is a natural-language outline for a Markdown file. Turn it into readable, well-structured Markdown in the target file.';
    }
    if (profileKey === 'python') {
        return 'The visible draft is a natural-language outline for a Python file. Translate it into proper Python code in the target file.';
    }
    if (profileKey === 'json') {
        return 'The visible draft is a natural-language outline for a JSON file. Generate valid JSON in the target file.';
    }
    return 'The visible draft is a natural-language outline for the target file. Read the spec and realize it in the target path.';
}

async function ensureDraftReadyForTurn() {
    if (!activeDraftPath) return;
    if (getDraftContent() !== draftLastSavedContent) {
        await saveDraftDocument({ autosave: true });
    }
}

function buildActiveDraftContextForTurn() {
    if (!activeDraftPath) return '';
    const profile = getDraftProfileDefinition(activeDraftProfileKey);
    const specPath = draftSpecPathForTargetPath(activeDraftPath);
    return [
        '',
        '<active_draft>',
        `target_path: ${activeDraftPath}`,
        `spec_path: ${specPath}`,
        `profile: ${profile.label}`,
        `extension: .${profile.extension}`,
        'The visible editor is a natural-language spec, outline, and wish list for the target file.',
        'Read the spec file first, then create or update the target file.',
        'Do not overwrite the spec file when generating the target output.',
        draftAgentHintForProfile(activeDraftProfileKey),
        '</active_draft>',
    ].join('\n');
}

function scheduleDraftAgentSync(options = {}) {
    if (!DOCUMENT_CENTERED_MODE || !currentWorkspaceId || !activeDraftPath) return;
    if (draftAgentSyncTimer) {
        clearTimeout(draftAgentSyncTimer);
        draftAgentSyncTimer = null;
    }
    const run = () => {
        draftAgentSyncTimer = null;
        requestDraftRealization(options).catch(error => {
            console.error('Failed to sync the draft spec into the generated file:', error);
            setDraftSaveState('error', 'Agent');
        });
    };
    if (options.immediate) {
        run();
        return;
    }
    draftAgentSyncTimer = window.setTimeout(run, DRAFT_AGENT_SYNC_DELAY_MS);
}

async function requestDraftRealization(options = {}) {
    if (!DOCUMENT_CENTERED_MODE || !currentWorkspaceId || !activeDraftPath) return;
    if (!modelAvailable) {
        draftPendingAgentSync = true;
        return;
    }
    const specContent = getDraftContent();
    if (!specContent.trim()) return;
    if (!options.force && specContent === draftLastAgentSpecContent) return;
    if (isGenerating) {
        draftPendingAgentSync = true;
        return;
    }
    await ensureDraftReadyForTurn();
    const prompt = `${buildDraftRealizationPrompt()}${buildActiveDraftContextForTurn()}`;
    const specPath = draftSpecPathForTargetPath(activeDraftPath);
    const queuedJob = await createDraftFileJob({
        path: activeDraftPath,
        lane: 'foreground',
        jobKind: 'realize_draft',
        title: `Realize ${basename(activeDraftPath) || activeDraftPath}`,
        payload: {
            trigger: options.force ? 'manual' : 'live',
            profile: activeDraftProfileKey,
            target_path: activeDraftPath,
            spec_path: specPath,
        },
        status: 'queued',
        sourceConversationId: currentConvId,
    }).catch(error => {
        console.error('Failed to queue the foreground draft job:', error);
        return null;
    });
    if (queuedJob?.job?.id) {
        activeDraftForegroundJobId = String(queuedJob.job.id).trim();
    }
    const turnFeatures = resolveTurnFeatures(prompt, [], null);
    turnFeatures.workspace_write = true;
    turnFeatures.workspace_run_commands = turnFeatures.workspace_run_commands || activeDraftProfileKey === 'python';
    turnFeatures.auto_approve_tool_permissions = true;
    rememberTurnFeatures(currentConvId, turnFeatures);
    persistDraftSession(currentWorkspaceId, activeDraftPath, currentConvId);
    syncCurrentConversationTitleToDraft(activeDraftPath).catch(() => {});
    draftLastAgentSpecContent = specContent;
    clearPendingPermissionRequest();
    dismissExecutionPlan();
    isGenerating = true;
    activeStreamConversationId = currentConvId;
    streamingAssistantMessage = null;
    setDraftSaveState('saving', 'Agent');
    syncSendButton();
    const clientTurnId = beginClientTurn();
    dispatchChatPayload({
        message: prompt,
        attachments: [],
        conversation_id: currentConvId,
        workspace_id: currentWorkspaceId || null,
        client_turn_id: clientTurnId,
        system_prompt: localStorage.getItem('customSystemPrompt') || null,
        mode: BASE_REASONING_MODE,
        features: turnFeatures,
        slash_command: null,
    }, {
        fallbackStatusText: 'Generating the file from the active draft...',
        fallbackActivityText: 'Running a hidden draft-to-file generation turn.',
    });
}

function normalizeIndentLevel(indent) {
    const size = Number(indent) || 0;
    return Math.max(0, Math.floor(size / 4));
}

function symbolGroupForKind(kind) {
    if (['class', 'interface', 'struct', 'enum', 'type', 'module', 'namespace', 'heading'].includes(kind)) return 'Structures';
    if (['function', 'method', 'hook'].includes(kind)) return 'Functions';
    if (['property', 'constant', 'field', 'selector'].includes(kind)) return 'Members';
    if (['key', 'route', 'section'].includes(kind)) return 'Data';
    return 'Other';
}

function extractSymbolsForPath(path, content) {
    const text = String(content || '');
    const ext = ((path || '').split('.').pop() || '').toLowerCase();
    const symbols = [];
    const lines = text.split(/\r?\n/);
    const addSymbol = (name, kind, line, indent = 0, extra = {}) => {
        if (!name) return;
        symbols.push({
            name: name.trim(),
            kind,
            line,
            indent,
            level: normalizeIndentLevel(indent),
            group: symbolGroupForKind(kind),
            ...extra,
        });
    };

    lines.forEach((lineText, index) => {
        let match = null;
        const indent = (lineText.match(/^\s*/) || [''])[0].replace(/\t/g, '    ').length;
        if (['js', 'jsx', 'ts', 'tsx'].includes(ext)) {
            match = lineText.match(/^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)/);
            if (match) return addSymbol(match[1], 'function', index + 1, indent);
            match = lineText.match(/^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\(/);
            if (match) return addSymbol(match[1], 'function', index + 1, indent);
            match = lineText.match(/^\s*(?:export\s+)?class\s+([A-Za-z0-9_$]+)/);
            if (match) return addSymbol(match[1], 'class', index + 1, indent);
            match = lineText.match(/^\s*(?:export\s+)?interface\s+([A-Za-z0-9_$]+)/);
            if (match) return addSymbol(match[1], 'interface', index + 1, indent);
            match = lineText.match(/^\s*(?:export\s+)?type\s+([A-Za-z0-9_$]+)/);
            if (match) return addSymbol(match[1], 'type', index + 1, indent);
            match = lineText.match(/^\s*(?:export\s+)?enum\s+([A-Za-z0-9_$]+)/);
            if (match) return addSymbol(match[1], 'enum', index + 1, indent);
            match = lineText.match(/^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=/);
            if (match && /=>/.test(lineText)) return addSymbol(match[1], 'function', index + 1, indent);
            match = lineText.match(/^\s*([A-Za-z0-9_$]+)\s*\([^)]*\)\s*\{/);
            if (match && indent > 0) return addSymbol(match[1], 'method', index + 1, indent);
            match = lineText.match(/^\s*(?:get|set)\s+([A-Za-z0-9_$]+)\s*\(/);
            if (match) return addSymbol(match[1], 'property', index + 1, indent);
        } else if (ext === 'py') {
            match = lineText.match(/^\s*def\s+([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], indent > 0 ? 'method' : 'function', index + 1, indent);
            match = lineText.match(/^\s*async\s+def\s+([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], indent > 0 ? 'method' : 'function', index + 1, indent);
            match = lineText.match(/^\s*class\s+([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], 'class', index + 1, indent);
        } else if (['rb'].includes(ext)) {
            match = lineText.match(/^\s*def\s+([A-Za-z0-9_?!]+)/);
            if (match) return addSymbol(match[1], 'method', index + 1, indent);
            match = lineText.match(/^\s*module\s+([A-Za-z0-9_:]+)/);
            if (match) return addSymbol(match[1], 'module', index + 1, indent);
            match = lineText.match(/^\s*class\s+([A-Za-z0-9_:]+)/);
            if (match) return addSymbol(match[1], 'class', index + 1, indent);
        } else if (['go'].includes(ext)) {
            match = lineText.match(/^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], lineText.includes('(') && /^\s*func\s+\([^)]+\)\s+/.test(lineText) ? 'method' : 'function', index + 1, indent);
            match = lineText.match(/^\s*type\s+([A-Za-z0-9_]+)\s+struct/);
            if (match) return addSymbol(match[1], 'struct', index + 1, indent);
            match = lineText.match(/^\s*type\s+([A-Za-z0-9_]+)\s+interface/);
            if (match) return addSymbol(match[1], 'interface', index + 1, indent);
        } else if (['java', 'kt', 'swift', 'rs', 'c', 'cc', 'cpp', 'hpp', 'h', 'cs', 'php'].includes(ext)) {
            match = lineText.match(/^\s*(?:public|private|protected|internal|static|final|open|abstract|\s)*\s*class\s+([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], 'class', index + 1, indent);
            match = lineText.match(/^\s*(?:public|private|protected|internal|static|final|open|abstract|\s)*\s*(?:interface|protocol|trait)\s+([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], 'interface', index + 1, indent);
            match = lineText.match(/^\s*(?:public|private|protected|internal|static|final|open|abstract|\s)*\s*enum\s+([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], 'enum', index + 1, indent);
            match = lineText.match(/^\s*(?:public|private|protected|internal|static|final|open|abstract|\s)*\s*(?:struct|record)\s+([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], 'struct', index + 1, indent);
            match = lineText.match(/^\s*(?:public|private|protected|internal|static|final|open|abstract|\s)*\s*(?:fn|func|void|int|float|double|bool|boolean|char|String|auto|const\s+auto|[A-Za-z0-9_:<>~*&]+\s+)+([A-Za-z0-9_~]+)\s*\(/);
            if (match) return addSymbol(match[1], indent > 0 ? 'method' : 'function', index + 1, indent);
        } else if (['json', 'yml', 'yaml', 'toml'].includes(ext)) {
            match = lineText.match(/^\s*([A-Za-z0-9_.-]+)\s*[:=]/);
            if (match) return addSymbol(match[1], 'key', index + 1, indent);
        } else if (['css', 'scss', 'sass', 'less'].includes(ext)) {
            match = lineText.match(/^\s*([^{][^{]+)\s*\{/);
            if (match) return addSymbol(match[1], 'selector', index + 1, indent);
        } else if (['md'].includes(ext)) {
            match = lineText.match(/^\s*(#{1,6})\s+(.+)/);
            if (match) return addSymbol(match[2], 'heading', index + 1, 0, { level: Math.max(0, match[1].length - 1), headingLevel: match[1].length });
        } else if (['sh', 'bash', 'zsh'].includes(ext)) {
            match = lineText.match(/^\s*([A-Za-z0-9_]+)\s*\(\)\s*\{/);
            if (match) return addSymbol(match[1], 'function', index + 1, indent);
            match = lineText.match(/^\s*function\s+([A-Za-z0-9_]+)/);
            if (match) return addSymbol(match[1], 'function', index + 1, indent);
        } else if (['sql'].includes(ext)) {
            match = lineText.match(/^\s*create\s+(?:or\s+replace\s+)?(?:view|table|function|procedure)\s+([A-Za-z0-9_."]+)/i);
            if (match) return addSymbol(match[1].replace(/"/g, ''), 'section', index + 1, indent);
        }
        return null;
    });

    return symbols.slice(0, 200);
}

function renderCurrentFileSymbols() {
    const listEl = document.getElementById('workspaceSymbolList');
    if (!listEl) return;
    if (!inlineViewerPath || !currentFileSymbols.length) {
        listEl.innerHTML = '<div class="workspace-empty">Open a code file to jump between symbols.</div>';
        return;
    }

    listEl.innerHTML = '';
    const groups = new Map();
    currentFileSymbols.forEach(symbol => {
        const key = symbol.group || 'Other';
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(symbol);
    });

    Array.from(groups.entries()).forEach(([groupName, items]) => {
        const groupEl = document.createElement('div');
        groupEl.className = 'workspace-symbol-group';
        const collapsed = Boolean(symbolGroupState[groupName]);
        const header = document.createElement('button');
        header.type = 'button';
        header.className = 'workspace-symbol-group-toggle';
        header.innerHTML = `
            <span class="workspace-symbol-group-caret">${collapsed ? '▸' : '▾'}</span>
            <span>${escapeHtml(groupName)}</span>
            <span class="workspace-symbol-group-count">${items.length}</span>
        `;
        header.onclick = () => {
            symbolGroupState[groupName] = !symbolGroupState[groupName];
            renderCurrentFileSymbols();
        };
        groupEl.appendChild(header);

        const body = document.createElement('div');
        body.className = `workspace-symbol-group-body${collapsed ? ' is-collapsed' : ''}`;
        items.forEach(symbol => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'workspace-symbol-item';
            button.style.setProperty('--symbol-level', String(Math.min(symbol.level || 0, 6)));
            button.innerHTML = `
                <div class="workspace-symbol-name">${escapeHtml(symbol.name)}</div>
                <div class="workspace-symbol-meta">${escapeHtml(`${symbol.kind} • line ${symbol.line}`)}</div>
            `;
            button.onclick = () => jumpToInlineSymbol(symbol.line);
            body.appendChild(button);
        });
        groupEl.appendChild(body);
        listEl.appendChild(groupEl);
    });
}

function updateCurrentFileSymbols(path, content) {
    currentFileSymbols = extractSymbolsForPath(path, content);
    renderCurrentFileSymbols();
}

function jumpToInlineSymbol(line) {
    const editor = ensureInlineEditor();
    if (!editor || !Number.isFinite(line)) return;
    setInlineViewerView('edit');
    const targetLine = Math.max(0, Number(line) - 1);
    editor.focus();
    editor.setCursor({ line: targetLine, ch: 0 });
    editor.scrollIntoView({ line: targetLine, ch: 0 }, 120);
}

function ensureInlineEditor() {
    if (inlineEditorReady) return inlineEditor;
    const textarea = document.getElementById('inlineViewerEditor');
    if (!textarea) return null;
    if (!(window.CodeMirror && typeof window.CodeMirror.fromTextArea === 'function')) return null;
    if (window.CodeMirror.modeURL === undefined) {
        window.CodeMirror.modeURL = 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/%N/%N.min.js';
    }

    inlineEditor = window.CodeMirror.fromTextArea(textarea, {
        lineNumbers: true,
        keyMap: 'vim',
        lineWrapping: false,
        indentUnit: 4,
        tabSize: 4,
        viewportMargin: Infinity,
        gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
        foldGutter: true,
        autoCloseBrackets: true,
        matchBrackets: true,
        styleActiveLine: true,
    });
    textarea.dataset.editorReady = 'true';
    inlineEditor.on('change', () => {
        if (inlineViewerApplyingRemoteContent) return;
        updateInlineViewerPreview();
        scheduleInlineViewerAutosave();
        updateCurrentFileSymbols(inlineViewerPath, inlineEditor.getValue());
        syncInlineSelectionAction();
    });
    inlineEditor.on('focus', () => {
        if (inlineViewerSaveState === 'saved') setInlineViewerSaveState('idle');
    });
    inlineEditor.on('cursorActivity', () => {
        syncInlineSelectionAction();
    });
    inlineEditorReady = true;
    return inlineEditor;
}

function getViewerContent(prefix) {
    if (prefix === 'inlineViewer') {
        const editor = ensureInlineEditor();
        if (editor) return editor.getValue();
    }
    const textarea = document.getElementById(`${prefix}Editor`);
    return textarea ? (textarea.value || '') : '';
}

function setViewerContent(prefix, content) {
    if (prefix === 'inlineViewer') {
        const editor = ensureInlineEditor();
        if (editor) {
            inlineViewerApplyingRemoteContent = true;
            const current = editor.getValue();
            if (current !== (content || '')) editor.setValue(content || '');
            inlineViewerApplyingRemoteContent = false;
            configureInlineEditorLanguage(inlineViewerPath);
            updateCurrentFileSymbols(inlineViewerPath, content || '');
            editor.refresh();
            return;
        }
    }
    const textarea = document.getElementById(`${prefix}Editor`);
    if (textarea) textarea.value = content || '';
}

function setInlineViewerSaveState(state, text = '') {
    inlineViewerSaveState = state;
    const el = document.getElementById('inlineViewerStatus');
    if (!el) return;
    el.classList.remove('is-dirty', 'is-saving', 'is-saved', 'is-error');
    if (state === 'dirty') {
        el.textContent = text || 'Unsaved';
        el.classList.add('is-dirty');
        return;
    }
    if (state === 'saving') {
        el.textContent = text || 'Autosaving...';
        el.classList.add('is-saving');
        return;
    }
    if (state === 'saved') {
        el.textContent = text || 'Saved';
        el.classList.add('is-saved');
        return;
    }
    if (state === 'error') {
        el.textContent = text || 'Save failed';
        el.classList.add('is-error');
        return;
    }
    el.textContent = text || 'Idle';
}

function syncInlineUndoButton() {
    const button = document.getElementById('inlineViewerUndoButton');
    if (!button) return;
    const stack = inlineViewerUndoStacks[inlineViewerPath] || [];
    button.disabled = !inlineViewerPath || !stack.length || !inlineViewerEditable;
}

function viewerSupportsCopy(state) {
    if (!state?.path) return false;
    return ['text', 'markdown', 'csv', 'html'].includes(state.kind);
}

function syncInlineViewerCopyButton() {
    const button = document.getElementById('inlineViewerCopyButton');
    if (!button) return;
    const show = viewerSupportsCopy(getViewerState('inlineViewer'));
    button.hidden = !show;
    button.disabled = !show;
    button.title = show ? 'Copy current file contents' : '';
}

function copyInlineViewerContent() {
    const button = document.getElementById('inlineViewerCopyButton');
    const state = getViewerState('inlineViewer');
    if (!button || !viewerSupportsCopy(state)) return;
    copyToClipboard(getViewerContent('inlineViewer'), button);
}

function getInlineSelection() {
    if (!inlineViewerEditable || !inlineViewerPath) return '';
    const editor = ensureInlineEditor();
    if (editor) return editor.getSelection() || '';
    const textarea = document.getElementById('inlineViewerEditor');
    if (!textarea) return '';
    const start = textarea.selectionStart ?? 0;
    const end = textarea.selectionEnd ?? start;
    return (textarea.value || '').slice(start, end);
}

function syncInlineSelectionAction() {
    const button = document.getElementById('inlineViewerAskAgentButton');
    if (!button) return;
    button.disabled = !inlineViewerEditable || !inlineViewerPath || !getInlineSelection().trim();
}

function buildSelectionPrompt(path, selection) {
    const language = getLanguageForPath(path) || '';
    const fence = language || '';
    return [
        `Help me work on this selection from \`${path}\`:`,
        '',
        `\`\`\`${fence}`,
        selection,
        '```',
        '',
        'Please explain what matters, suggest the right change, and edit the relevant workspace files if needed.'
    ].join('\n');
}

function sendInlineSelectionToChat() {
    const selection = getInlineSelection().trim();
    if (!selection || !inlineViewerPath) return;
    const prompt = buildSelectionPrompt(inlineViewerPath, selection);
    const input = document.getElementById('input');
    if (!input) return;
    const existing = input.value.trim();
    const nextValue = existing ? `${existing}\n\n${prompt}` : prompt;
    setInputValue(nextValue, nextValue.length);
    focusInput();
}

function pushInlineUndoSnapshot(path, content) {
    if (!path) return;
    const stack = inlineViewerUndoStacks[path] || [];
    if (stack[stack.length - 1] === content) return;
    stack.push(content);
    while (stack.length > INLINE_VIEWER_UNDO_LIMIT) stack.shift();
    inlineViewerUndoStacks[path] = stack;
    syncInlineUndoButton();
}

function scheduleInlineViewerAutosave() {
    if (!inlineViewerEditable || !inlineViewerPath || inlineViewerView !== 'edit') return;
    const current = getViewerContent('inlineViewer');
    if (current === inlineViewerLastSavedContent) {
        if (inlineViewerAutosaveTimer) {
            clearTimeout(inlineViewerAutosaveTimer);
            inlineViewerAutosaveTimer = null;
        }
        setInlineViewerSaveState('saved');
        return;
    }
    setInlineViewerSaveState('dirty');
    if (inlineViewerAutosaveTimer) clearTimeout(inlineViewerAutosaveTimer);
    inlineViewerAutosaveTimer = setTimeout(() => {
        inlineViewerAutosaveTimer = null;
        saveInlineViewer({ autosave: true }).catch(() => {});
    }, INLINE_VIEWER_AUTOSAVE_DELAY_MS);
}

async function undoInlineViewerChange() {
    if (!inlineViewerPath) return;
    const stack = inlineViewerUndoStacks[inlineViewerPath] || [];
    if (!stack.length) return;
    const previousContent = stack.pop();
    inlineViewerUndoStacks[inlineViewerPath] = stack;
    inlineViewerPerformingUndo = true;
    setViewerContent('inlineViewer', previousContent || '');
    updateInlineViewerPreview();
    syncInlineUndoButton();
    await saveInlineViewer({ autosave: true, skipUndoSnapshot: true });
    inlineViewerPerformingUndo = false;
}

function getViewerState(prefix) {
    return {
        editable: inlineViewerEditable,
        kind: inlineViewerKind,
        path: inlineViewerPath,
        view: inlineViewerView,
    };
}

function setViewerState(prefix, updates = {}) {
    if (updates.editable !== undefined) inlineViewerEditable = updates.editable;
    if (updates.kind !== undefined) inlineViewerKind = updates.kind;
    if (updates.path !== undefined) inlineViewerPath = updates.path;
    if (updates.view !== undefined) inlineViewerView = updates.view;
}

function setViewerView(prefix, view) {
    const editTab = document.getElementById(`${prefix}EditTab`);
    const previewTab = document.getElementById(`${prefix}PreviewTab`);
    const editPane = document.getElementById(`${prefix}EditPane`);
    const previewPane = document.getElementById(`${prefix}PreviewPane`);
    const state = getViewerState(prefix);
    if (!editTab || !previewTab || !editPane || !previewPane) return;

    const editing = view === 'edit' && state.editable;
    editTab.classList.toggle('active', editing);
    previewTab.classList.toggle('active', !editing);
    editPane.classList.toggle('file-modal-pane-hidden', !editing);
    previewPane.classList.toggle('file-modal-pane-hidden', editing);
    if (prefix === 'inlineViewer') inlineViewerView = editing ? 'edit' : 'preview';
    if (prefix === 'inlineViewer' && editing) {
        const editor = ensureInlineEditor();
        if (editor) {
            setTimeout(() => editor.refresh(), 0);
        }
    }
    if (prefix === 'inlineViewer') {
        syncInlineViewerMode();
        syncInlineViewerCopyButton();
    }
}

function setInlineViewerView(view) {
    setViewerView('inlineViewer', view);
}

function inlineViewerModeLabel() {
    if (!inlineViewerPath) return 'No file';
    if (inlineViewerKind === 'spreadsheet') return 'Preview only';
    if (!inlineViewerEditable) return 'Read only';
    if (inlineViewerView === 'preview') return 'Preview';
    return 'Editing';
}

function syncInlineViewerMode() {
    const modeEl = document.getElementById('inlineViewerMode');
    if (modeEl) modeEl.textContent = inlineViewerModeLabel();
}

function syncInlineViewerVisibility() {
    syncWorkspaceViewMode();
    syncChatShellLayout();
}

function setViewerPlaceholder(prefix, message = DEFAULT_INLINE_VIEWER_EMPTY_TEXT, kind = 'empty', visible = true) {
    const emptyEl = document.getElementById(`${prefix}Empty`);
    const editPane = document.getElementById(`${prefix}EditPane`);
    const previewPane = document.getElementById(`${prefix}PreviewPane`);
    if (!emptyEl) return;
    emptyEl.textContent = message || DEFAULT_INLINE_VIEWER_EMPTY_TEXT;
    emptyEl.dataset.kind = kind || 'empty';
    emptyEl.hidden = !visible;
    if (visible) {
        if (editPane) editPane.classList.add('file-modal-pane-hidden');
        if (previewPane) previewPane.classList.add('file-modal-pane-hidden');
    }
}

function toggleViewerEmptyState(prefix, hasFile) {
    setViewerPlaceholder(prefix, DEFAULT_INLINE_VIEWER_EMPTY_TEXT, 'empty', !hasFile);
}

function showViewerLoading(prefix, path) {
    setViewerPlaceholder(prefix, `Opening ${basename(path) || 'file'}...`, 'loading', true);
}

function showViewerError(prefix, path, message) {
    const details = String(message || '').trim() || 'The file could not be opened.';
    setViewerPlaceholder(prefix, `Couldn't open ${basename(path) || 'this file'}: ${details}`, 'error', true);
}

function clearViewerPlaceholder(prefix) {
    setViewerPlaceholder(prefix, DEFAULT_INLINE_VIEWER_EMPTY_TEXT, 'empty', false);
}

function renderCodePreview(targetId, content, path) {
    const previewEl = document.getElementById(targetId);
    if (!previewEl) return;
    previewEl.classList.remove('file-modal-markdown');
    previewEl.innerHTML = '';
    const pre = document.createElement('pre');
    pre.className = 'file-modal-code';
    const code = document.createElement('code');
    code.textContent = content || '';
    const language = getLanguageForPath(path);
    if (language) code.className = `language-${language}`;
    pre.appendChild(code);
    previewEl.appendChild(pre);
    if (typeof hljs !== 'undefined') {
        try { hljs.highlightElement(code); } catch (e) {}
    }
}

function workspacePathDirname(path) {
    const normalized = normalizeWorkspacePath(path);
    if (!normalized || normalized === '.') return '.';
    const parts = normalized.split('/');
    parts.pop();
    return parts.length ? parts.join('/') : '.';
}

function isWorkspaceExternalUrl(value) {
    const text = String(value || '').trim();
    if (!text) return true;
    return /^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i.test(text);
}

function resolveWorkspaceAssetPath(basePath, assetRef) {
    const raw = String(assetRef || '').trim();
    if (!raw || isWorkspaceExternalUrl(raw)) return '';
    const match = raw.match(/^([^?#]*)([?#].*)?$/);
    const pathPart = String(match?.[1] || '').trim();
    if (!pathPart) return '';

    const baseDir = workspacePathDirname(basePath);
    const segments = pathPart.startsWith('/')
        ? []
        : (baseDir === '.' ? [] : baseDir.split('/'));

    pathPart.split('/').forEach(part => {
        if (!part || part === '.') return;
        if (part === '..') {
            if (segments.length) segments.pop();
            return;
        }
        segments.push(part);
    });

    return segments.join('/') || '.';
}

function prepareHtmlPreviewContent(content, path) {
    const html = String(content || '');
    if (!html.trim() || typeof DOMParser === 'undefined') return html;

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const selectors = [
        ['img[src]', 'src'],
        ['source[src]', 'src'],
        ['video[src]', 'src'],
        ['audio[src]', 'src'],
        ['track[src]', 'src'],
        ['embed[src]', 'src'],
        ['iframe[src]', 'src'],
        ['script[src]', 'src'],
        ['object[data]', 'data'],
        ['link[href]', 'href'],
        ['a[href]', 'href'],
    ];

    selectors.forEach(([selector, attribute]) => {
        doc.querySelectorAll(selector).forEach(node => {
            const original = node.getAttribute(attribute);
            const resolvedPath = resolveWorkspaceAssetPath(path, original);
            if (!resolvedPath) return;
            node.setAttribute(attribute, workspaceFileInlineViewUrl(resolvedPath));
        });
    });

    return `<!DOCTYPE html>\n${doc.documentElement.outerHTML}`;
}

function renderHtmlPreview(targetId, content, path) {
    const previewEl = document.getElementById(targetId);
    if (!previewEl) return;
    previewEl.classList.remove('file-modal-markdown');
    previewEl.innerHTML = '';
    const iframe = document.createElement('iframe');
    iframe.className = 'file-modal-html';
    iframe.setAttribute('sandbox', 'allow-same-origin');
    iframe.srcdoc = prepareHtmlPreviewContent(content, path);
    previewEl.appendChild(iframe);
}

function renderPdfPreview(targetId, path) {
    const previewEl = document.getElementById(targetId);
    if (!previewEl) return;
    previewEl.classList.remove('file-modal-markdown');
    previewEl.innerHTML = '';
    const iframe = document.createElement('iframe');
    iframe.className = 'file-modal-pdf';
    iframe.title = basename(path) || 'PDF preview';
    iframe.src = workspaceFileInlineViewUrl(path);
    previewEl.appendChild(iframe);
}

function renderImagePreview(targetId, path) {
    const previewEl = document.getElementById(targetId);
    if (!previewEl) return;
    previewEl.classList.remove('file-modal-markdown');
    previewEl.innerHTML = '';
    const img = document.createElement('img');
    img.className = 'workspace-image-preview';
    img.alt = basename(path) || 'Image preview';
    img.src = workspaceFileInlineViewUrl(path);
    previewEl.appendChild(img);
}

function renderMarkdownPreview(targetId, content) {
    const previewEl = document.getElementById(targetId);
    if (!previewEl) return;
    previewEl.classList.add('file-modal-markdown');
    if (typeof marked === 'undefined' || !markedReady) {
        previewEl.textContent = content || '';
        return;
    }
    previewEl.innerHTML = (content && content.trim()) ? marked.parse(content) : '<p class="workspace-empty">Nothing to preview.</p>';
    renderMathContent(previewEl);
    enhanceMessageArtifactReferences(null, previewEl, content);
    if (typeof hljs !== 'undefined') {
        previewEl.querySelectorAll('pre code').forEach(block => {
            try { if (block.textContent?.trim()) hljs.highlightElement(block); } catch (e) {}
        });
    }
}

function renderDelimitedPreview(targetId, content, path) {
    const previewEl = document.getElementById(targetId);
    if (!previewEl) return;
    previewEl.classList.remove('file-modal-markdown');
    previewEl.innerHTML = '';
    const delimiter = /\.tsv$/i.test(path || '') ? '\t' : ',';
    const lines = String(content || '').split(/\r?\n/).filter(line => line.length);
    if (!lines.length) {
        previewEl.textContent = 'No rows to preview.';
        return;
    }
    const rows = lines.slice(0, 30).map(line => line.split(delimiter));
    const wrap = document.createElement('div');
    wrap.className = 'file-modal-table-wrap';
    const table = document.createElement('table');
    table.className = 'file-modal-table';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    (rows[0] || []).forEach(cell => {
        const th = document.createElement('th');
        th.textContent = cell;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    rows.slice(1).forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    previewEl.appendChild(wrap);
}

function renderSpreadsheetPreview(targetId, data) {
    const previewEl = document.getElementById(targetId);
    if (!previewEl) return;
    previewEl.classList.remove('file-modal-markdown');
    previewEl.innerHTML = '';
    const meta = document.createElement('div');
    meta.className = 'file-modal-subtitle';
    meta.style.marginBottom = '10px';
    meta.textContent = `${data.file_type?.toUpperCase() || 'SHEET'} • ${data.row_count || 0} rows • ${data.column_count || 0} columns`;
    previewEl.appendChild(meta);

    const wrap = document.createElement('div');
    wrap.className = 'file-modal-table-wrap';
    const table = document.createElement('table');
    table.className = 'file-modal-table';
    const thead = document.createElement('thead');
    const tr = document.createElement('tr');
    (data.columns || []).forEach(column => {
        const th = document.createElement('th');
        th.textContent = column.name || '';
        tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    (data.preview_rows || []).forEach(row => {
        const rowEl = document.createElement('tr');
        (data.columns || []).forEach(column => {
            const td = document.createElement('td');
            const value = row?.[column.name];
            td.textContent = value == null ? '' : String(value);
            rowEl.appendChild(td);
        });
        tbody.appendChild(rowEl);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    previewEl.appendChild(wrap);
}

function renderViewerSheetTabs(prefix, sheetNames, activeSheet) {
    const tabs = document.getElementById(`${prefix}SheetTabs`);
    if (!tabs) return;
    if (!Array.isArray(sheetNames) || sheetNames.length <= 1) {
        tabs.hidden = true;
        tabs.innerHTML = '';
        return;
    }
    tabs.hidden = false;
    tabs.innerHTML = '';
    sheetNames.forEach(name => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `file-modal-tab${name === activeSheet ? ' active' : ''}`;
        button.textContent = name;
        button.onclick = () => {
            selectedSpreadsheetSheet = name;
            openWorkspaceFile(getViewerState(prefix).path, { preserveSheet: true });
        };
        tabs.appendChild(button);
    });
}

function updateViewerPreview(prefix) {
    const state = getViewerState(prefix);
    if (!state.editable) return;
    const content = getViewerContent(prefix);
    const previewId = `${prefix}Preview`;
    if (state.kind === 'pdf') renderPdfPreview(previewId, state.path);
    else if (state.kind === 'image') renderImagePreview(previewId, state.path);
    else if (state.kind === 'html') renderHtmlPreview(previewId, content);
    else if (state.kind === 'markdown') renderMarkdownPreview(previewId, content);
    else if (state.kind === 'csv') renderDelimitedPreview(previewId, content, state.path);
    else renderCodePreview(previewId, content, state.path);
}

function updateInlineViewerPreview() {
    updateViewerPreview('inlineViewer');
}

async function saveViewer(prefix, options = {}) {
    const state = getViewerState(prefix);
    if (!state.editable || !state.path) return;
    const editor = ensureInlineEditor();
    const saveButton = document.getElementById(`${prefix}SaveButton`);
    if (!editor || !saveButton) return;
    const content = getViewerContent(prefix);

    saveButton.disabled = true;
    setInlineViewerSaveState('saving', options.autosave ? 'Autosaving...' : 'Saving...');
    try {
        const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: state.path, content }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
        if (!options.skipUndoSnapshot && !inlineViewerPerformingUndo && inlineViewerLastSavedContent !== content) {
            pushInlineUndoSnapshot(state.path, inlineViewerLastSavedContent);
        }
        inlineViewerLastSavedContent = content;
        setInlineViewerSaveState('saved');
        syncInlineUndoButton();
        await refreshWorkspace(false);
    } catch (e) {
        setInlineViewerSaveState('error', 'Save failed');
        throw e;
    } finally {
        saveButton.disabled = false;
        saveButton.textContent = 'Save';
    }
}

async function saveInlineViewer(options = {}) {
    await saveViewer('inlineViewer', options);
}

function applyViewerMetadata(prefix, path) {
    const titleEl = document.getElementById(`${prefix}Title`);
    const subtitleEl = document.getElementById(`${prefix}Subtitle`);
    const preview = document.getElementById(`${prefix}Preview`);
    if (titleEl) titleEl.textContent = path ? basename(path) : 'Artifact Preview';
    if (subtitleEl) subtitleEl.textContent = path || 'Select a generated file or workspace file to preview it here.';
    if (!path) {
        setViewerContent(prefix, '');
        if (preview) preview.innerHTML = '';
        renderViewerSheetTabs(prefix, [], '');
        currentFileSymbols = [];
        renderCurrentFileSymbols();
        inlineViewerLastSavedContent = '';
        setInlineViewerSaveState('idle');
        syncInlineUndoButton();
        syncInlineSelectionAction();
    }
    if (path) clearViewerPlaceholder(prefix);
    else toggleViewerEmptyState(prefix, false);
    syncInlineViewerMode();
    syncInlineViewerVisibility();
    syncInlineViewerCopyButton();
}

function syncViewerControls(prefix, editable, defaultView = 'preview') {
    const editTab = document.getElementById(`${prefix}EditTab`);
    const previewTab = document.getElementById(`${prefix}PreviewTab`);
    const undoButton = document.getElementById(`${prefix}UndoButton`);
    const saveButton = document.getElementById(`${prefix}SaveButton`);
    if (editTab) editTab.hidden = !editable;
    if (previewTab) previewTab.hidden = !editable;
    if (undoButton) undoButton.hidden = !editable;
    if (saveButton) saveButton.hidden = !editable;
    setViewerView(prefix, editable ? defaultView : 'preview');
    if (!getViewerState(prefix).path) toggleViewerEmptyState(prefix, false);
    syncInlineUndoButton();
    syncInlineSelectionAction();
    syncInlineViewerMode();
    syncInlineViewerVisibility();
    syncInlineViewerCopyButton();
}

function closeInlineViewer() {
    if (inlineViewerAutosaveTimer) {
        clearTimeout(inlineViewerAutosaveTimer);
        inlineViewerAutosaveTimer = null;
    }
    inlineViewerRequestToken += 1;
    selectedWorkspaceFile = '';
    selectedSpreadsheetSheet = '';
    inlineViewerPath = '';
    inlineViewerKind = 'text';
    inlineViewerEditable = false;
    inlineViewerView = 'preview';
    currentFileSymbols = [];
    applyViewerMetadata('inlineViewer', '');
    renderCurrentFileSymbols();
    syncViewerControls('inlineViewer', false, 'preview');
    setWorkspaceViewMode('tree');
    renderFileList();
}

function shouldAutoOpenArtifactPreview(path) {
    if (!path) return false;
    if (!/\.(png|jpg|jpeg|gif|svg|webp|pdf|html?|md|markdown|rst)$/i.test(path)) {
        return false;
    }
    const state = getViewerState('inlineViewer');
    if (state.editable && state.path && !compareWorkspacePaths(state.path, path)) {
        return false;
    }
    return true;
}

async function openWorkspaceFile(path, options = {}) {
    if (!featureSettings.agent_tools || !path) return;
    const revealViewer = options.reveal !== false;
    const preserveSheet = Boolean(options.preserveSheet);
    if (revealViewer && !workspacePanelOpen) {
        dismissMobileKeyboard(true);
        closeMenu();
        workspacePanelOpen = true;
        applyWorkspacePanelState();
    }
    const requestToken = ++inlineViewerRequestToken;
    const provisionalKind = /\.(xlsx|xls|xlsm)$/i.test(path)
        ? 'spreadsheet'
        : (/\.pdf$/i.test(path) ? 'pdf' : (isImagePath(path) ? 'image' : (isHtmlPath(path) ? 'html' : (isMarkdownPath(path) ? 'markdown' : (/\.(csv|tsv)$/i.test(path) ? 'csv' : 'text')))));
    if (!preserveSheet) selectedSpreadsheetSheet = '';
    selectedWorkspaceFile = path;
    inlineViewerPath = path;
    setViewerState('inlineViewer', { editable: false, kind: provisionalKind, path, view: 'preview' });
    renderFileList();
    applyViewerMetadata('inlineViewer', path);
    setViewerContent('inlineViewer', '');
    renderViewerSheetTabs('inlineViewer', [], '');
    currentFileSymbols = [];
    renderCurrentFileSymbols();
    syncViewerControls('inlineViewer', false, 'preview');
    showViewerLoading('inlineViewer', path);
    setWorkspaceViewMode('reader');

    try {
        if (/\.(xlsx|xls|xlsm)$/i.test(path)) {
            setViewerState('inlineViewer', { kind: 'spreadsheet', editable: false, path, view: 'preview' });
            const params = new URLSearchParams({ path });
            if (selectedSpreadsheetSheet) params.set('sheet', selectedSpreadsheetSheet);
            const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/spreadsheet?${params.toString()}`);
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
            if (requestToken !== inlineViewerRequestToken || !compareWorkspacePaths(path, inlineViewerPath)) return;
            selectedSpreadsheetSheet = data.sheet || '';
            renderViewerSheetTabs('inlineViewer', data.sheet_names || [], selectedSpreadsheetSheet);
            renderSpreadsheetPreview('inlineViewerPreview', data);
            clearViewerPlaceholder('inlineViewer');
            currentFileSymbols = [];
            renderCurrentFileSymbols();
            setInlineViewerSaveState('idle');
            syncViewerControls('inlineViewer', false, 'preview');
            return;
        }

        const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/file?path=${encodeURIComponent(path)}`);
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
        if (requestToken !== inlineViewerRequestToken || !compareWorkspacePaths(path, inlineViewerPath)) return;

        const metadata = normalizeViewerMetadata(path, data);
        const editable = metadata.editable;
        const kind = metadata.kind;
        const defaultView = metadata.defaultView;
        const content = typeof data.content === 'string' ? data.content : '';
        setViewerState('inlineViewer', { editable, kind, path, view: defaultView });
        setViewerContent('inlineViewer', content);
        inlineViewerLastSavedContent = content;
        setInlineViewerSaveState('idle');
        syncInlineUndoButton();

        if (kind === 'pdf') {
            renderPdfPreview('inlineViewerPreview', path);
        }
        else if (kind === 'image') {
            renderImagePreview('inlineViewerPreview', path);
        }
        else if (kind === 'html') {
            renderHtmlPreview('inlineViewerPreview', content, path);
        }
        else if (kind === 'markdown') {
            renderMarkdownPreview('inlineViewerPreview', content);
        }
        else if (kind === 'csv') {
            const summaryResp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/spreadsheet?path=${encodeURIComponent(path)}`);
            const summaryData = await summaryResp.json();
            if (requestToken !== inlineViewerRequestToken || !compareWorkspacePaths(path, inlineViewerPath)) return;
            if (summaryResp.ok) {
                renderSpreadsheetPreview('inlineViewerPreview', summaryData);
            } else {
                renderDelimitedPreview('inlineViewerPreview', content, path);
            }
        } else {
            renderCodePreview('inlineViewerPreview', content, path);
        }

        clearViewerPlaceholder('inlineViewer');
        syncViewerControls('inlineViewer', editable, defaultView);
        configureInlineEditorLanguage(path);
        updateCurrentFileSymbols(path, content);
        updateInlineViewerPreview();
    } catch (e) {
        if (requestToken !== inlineViewerRequestToken || !compareWorkspacePaths(path, inlineViewerPath)) return;
        setViewerState('inlineViewer', { editable: false, kind: provisionalKind, path, view: 'preview' });
        setInlineViewerSaveState('error', 'Open failed');
        currentFileSymbols = [];
        renderCurrentFileSymbols();
        showViewerError('inlineViewer', path, e.message);
        syncViewerControls('inlineViewer', false, 'preview');
    }
}

function isAudioAttachment(file) {
    const contentType = String(file?.content_type || file?.contentType || '').toLowerCase();
    if (contentType.startsWith('audio/')) return true;
    const path = String(file?.name || file?.path || '').toLowerCase();
    return /\.(aac|aif|aiff|caf|flac|m4a|mp3|ogg|opus|wav|wave|webm)$/i.test(path);
}

function formatAudioDuration(totalSeconds) {
    const seconds = Math.max(0, Math.round(Number(totalSeconds) || 0));
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${String(seconds % 60).padStart(2, '0')}`;
}

function buildAudioWaveMarkup(seed = '') {
    const pattern = [8, 14, 10, 18, 12, 20, 9, 16, 11, 17, 13, 15];
    const offset = Array.from(String(seed)).reduce((sum, ch) => sum + ch.charCodeAt(0), 0) % pattern.length;
    return `
        <span class="attachment-chip-wave" aria-hidden="true">
            ${pattern.map((_, index) => {
                const height = pattern[(index + offset) % pattern.length];
                return `<span class="attachment-chip-wave-bar" style="height:${height}px; animation-delay:${(index % 6) * 0.08}s"></span>`;
            }).join('')}
        </span>
    `;
}

function buildAttachmentChipMarkup(file) {
    const audio = isAudioAttachment(file);
    const label = file.name || file.path || 'attachment';
    const meta = [];
    meta.push(audio ? 'audio' : String(file.kind || 'file'));
    if (audio && Number.isFinite(Number(file.duration))) meta.push(formatAudioDuration(file.duration));
    meta.push(formatBytes(file.size || 0));
    return `
        ${audio ? buildAudioWaveMarkup(label) : ''}
        <div class="attachment-chip-copy${audio ? ' is-audio' : ''}">
            <span class="attachment-chip-name">${escapeHtml(label)}</span>
            <span class="attachment-chip-meta">${escapeHtml(meta.join(' • '))}</span>
        </div>
        <button type="button" class="attachment-chip-remove" title="Remove attachment" aria-label="Remove attachment">&times;</button>
    `;
}

function buildAttachmentNoteText(attachments = pendingAttachments) {
    const lines = attachments
        .map(file => String(file.path || file.name || '').trim())
        .filter(Boolean);
    if (!lines.length) return '';
    return `Attached files:\n${lines.map(line => `- ${line}`).join('\n')}`;
}

function buildDisplayedMessageText(text, attachments = pendingAttachments) {
    const trimmed = String(text || '').trim();
    const attachmentNote = buildAttachmentNoteText(attachments);
    if (trimmed && attachmentNote) return `${trimmed}\n\n${attachmentNote}`;
    return trimmed || attachmentNote;
}

function buildAttachmentOnlyMessage(attachments = pendingAttachments) {
    const usableAttachments = attachments.filter(Boolean);
    if (!usableAttachments.length) return '';
    if (usableAttachments.every(file => isAudioAttachment(file))) {
        return usableAttachments.length === 1
            ? 'Please review the attached audio recording.'
            : 'Please review the attached audio recordings.';
    }
    return usableAttachments.length === 1
        ? 'Please review the attached file.'
        : 'Please review the attached files.';
}

async function uploadPendingFiles(fileEntries, options = {}) {
    const {
        failureMessage = 'Attachment upload failed',
    } = options;
    const availableSlots = Math.max(0, MAX_PENDING_ATTACHMENTS - pendingAttachments.length);
    if (!availableSlots) {
        throw new Error(`You can attach up to ${MAX_PENDING_ATTACHMENTS} files per message.`);
    }

    const descriptors = fileEntries
        .slice(0, availableSlots)
        .map((entry, index) => {
            const blob = entry?.blob || entry;
            const name = entry?.name || blob?.name || `attachment-${index + 1}`;
            if (!blob) return null;
            return {
                blob,
                name,
                size: entry?.size ?? blob.size ?? 0,
                duration: entry?.duration,
                kind: entry?.kind || '',
                contentType: entry?.contentType || blob.type || 'application/octet-stream',
            };
        })
        .filter(Boolean);

    if (!descriptors.length) return [];

    if (!featureSettings.agent_tools) {
        featureSettings.agent_tools = true;
        persistFeatureSettings();
        applyFeatureSettingsToUI();
    }

    const formData = new FormData();
    descriptors.forEach(file => formData.append('files', file.blob, file.name));
    formData.append('target_path', '.');

    const attachButton = document.getElementById('attachButton');
    if (attachButton) attachButton.disabled = true;

    try {
        const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/upload`, {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);

        const savedFiles = (data.files || []).map((saved, index) => ({
            ...saved,
            kind: descriptors[index]?.kind || saved.kind,
            content_type: descriptors[index]?.contentType || saved.content_type,
            duration: descriptors[index]?.duration,
        }));
        pendingAttachments = pendingAttachments.concat(savedFiles).slice(0, MAX_PENDING_ATTACHMENTS);
        renderPendingAttachments();
        if (savedFiles.length) {
            selectedWorkspaceFile = savedFiles[0].path || selectedWorkspaceFile;
            refreshWorkspace(true);
        }
        return savedFiles;
    } catch (error) {
        throw new Error(error.message || failureMessage);
    } finally {
        if (attachButton) attachButton.disabled = false;
    }
}

function renderPendingAttachments() {
    const bar = document.getElementById('attachmentBar');
    if (!bar) return;

    if (!pendingAttachments.length) {
        bar.hidden = true;
        bar.innerHTML = '';
        syncSendButton();
        return;
    }

    bar.hidden = false;
    bar.innerHTML = '';
    pendingAttachments.forEach((file, index) => {
        const chip = document.createElement('div');
        chip.className = `attachment-chip${isAudioAttachment(file) ? ' is-audio' : ''}`;
        chip.innerHTML = buildAttachmentChipMarkup(file);
        chip.querySelector('.attachment-chip-remove').onclick = () => {
            pendingAttachments.splice(index, 1);
            renderPendingAttachments();
        };
        bar.appendChild(chip);
    });
    syncSendButton();
}

function clearPendingAttachments() {
    pendingAttachments = [];
    const input = document.getElementById('attachmentInput');
    if (input) input.value = '';
    renderPendingAttachments();
}

function openAttachmentPicker() {
    dismissMobileKeyboard(true);
    document.getElementById('attachmentInput')?.click();
}

async function handleAttachmentSelection(event) {
    const input = event.target;
    const files = Array.from(input.files || []);
    if (!files.length) return;

    try {
        await uploadPendingFiles(files, { failureMessage: 'Attachment upload failed' });
    } catch (e) {
        alert(`Attachment upload failed: ${e.message}`);
    } finally {
        input.value = '';
    }
}

async function sendMessage() {
    const input = document.getElementById('input');
    const displayText = input.value.trim();
    const attachmentPaths = pendingAttachments.map(file => file.path).filter(Boolean);
    if (!modelAvailable || (!displayText && !attachmentPaths.length) || isGenerating) return;
    hideSlashMenu();
    const slashCommand = parseDirectSlashCommandInput(displayText);

    // Resolve any collapsed paste placeholders to actual content
    let message = displayText;
    for (const block of pastedBlocks) {
        message = message.replace(block.placeholder, block.actual);
    }
    if (!message) message = buildAttachmentOnlyMessage(pendingAttachments);
    try {
        await ensureDraftReadyForTurn();
    } catch (error) {
        alert(`Draft save failed: ${error.message}`);
        return;
    }
    const outboundMessage = `${message}${buildActiveDraftContextForTurn()}`;

    document.querySelector('.welcome')?.remove();
    exitWelcomeMode();
    addMessage(buildDisplayedMessageText(displayText, pendingAttachments) || message, 'user', null, { forceScroll: true });
    input.value = '';
    pastedBlocks = [];
    const turnFeatures = resolveTurnFeatures(outboundMessage, attachmentPaths, slashCommand);
    rememberTurnFeatures(currentConvId, turnFeatures);
    dismissExecutionPlan();
    clearPendingPermissionRequest();
    clearPendingAttachments();
    autoResizeTextarea(input);
    showLoading();
    clearBuildSteps();
    clearWorkspaceActivity();
    recordWorkspaceActivity('Request', displayText || message);

    isGenerating = true;
    activeStreamConversationId = currentConvId;
    streamingAssistantMessage = null;
    syncSendButton();
    const clientTurnId = beginClientTurn();
    dispatchChatPayload({
        message: outboundMessage,
        attachments: attachmentPaths,
        conversation_id: currentConvId,
        workspace_id: currentWorkspaceId || null,
        client_turn_id: clientTurnId,
        system_prompt: localStorage.getItem('customSystemPrompt') || null,
        mode: BASE_REASONING_MODE,
        features: turnFeatures,
        slash_command: slashCommand ? {
            name: slashCommand.name,
            raw_name: slashCommand.rawName,
            args: slashCommand.args,
        } : null,
    });
}

async function sendInterruptMessage(messageText) {
    const text = String(messageText || '').trim();
    const attachmentPaths = pendingAttachments.map(file => file.path).filter(Boolean);
    const outboundText = text || buildAttachmentOnlyMessage(pendingAttachments);
    if (!outboundText || !canUseWebsocketTransport()) return;
    const slashCommand = parseDirectSlashCommandInput(outboundText);
    try {
        await ensureDraftReadyForTurn();
    } catch (error) {
        alert(`Draft save failed: ${error.message}`);
        return;
    }
    const fullOutboundText = `${outboundText}${buildActiveDraftContextForTurn()}`;
    const turnFeatures = resolveTurnFeatures(fullOutboundText, attachmentPaths, slashCommand);
    rememberTurnFeatures(currentConvId, turnFeatures);
    dismissExecutionPlan();
    clearPendingPermissionRequest();
    addMessage(buildDisplayedMessageText(text, pendingAttachments) || outboundText, 'user', null, { forceScroll: true });
    clearPendingAttachments();
    document.querySelector('.welcome')?.remove();
    exitWelcomeMode();
    recordWorkspaceActivity('Interrupt', outboundText);
    const clientTurnId = beginClientTurn();
    ws.send(JSON.stringify({
        type: 'interrupt',
        message: fullOutboundText,
        attachments: attachmentPaths,
        conversation_id: currentConvId,
        workspace_id: currentWorkspaceId || null,
        client_turn_id: clientTurnId,
        system_prompt: localStorage.getItem('customSystemPrompt') || null,
        mode: BASE_REASONING_MODE,
        features: turnFeatures,
        slash_command: slashCommand ? {
            name: slashCommand.name,
            raw_name: slashCommand.rawName,
            args: slashCommand.args,
        } : null,
    }));
}

function sendIntervention() {
    const textarea = document.getElementById('interveneInput');
    if (!textarea) return;
    const intervention = textarea.value.trim();
    if (!intervention) return;
    textarea.value = '';
    syncInterveneButton();
    sendInterruptMessage(`Continue from the current task board with this intervention:\n${intervention}`);
}

function handlePrimaryAction() {
    const input = document.getElementById('input');
    const draft = input?.value.trim() || '';
    const hasPendingAttachments = pendingAttachments.length > 0;
    if (isGenerating) {
        const canInterruptWithMessage = currentTurnTransport === 'ws' && canUseWebsocketTransport();
        if (canInterruptWithMessage && (draft || hasPendingAttachments)) {
            input.value = '';
            autoResizeTextarea(input);
            syncSendButton();
            sendInterruptMessage(draft || buildAttachmentOnlyMessage(pendingAttachments));
        } else {
            stopMessage();
        }
        return;
    }
    if (!draft && !hasPendingAttachments) return;
    sendMessage();
}

function stopMessage() {
    if (!isGenerating) return;
    clearPendingPermissionRequest();
    if (currentTurnTransport === 'http' && httpTurnAbortController) {
        httpTurnAbortController.abort();
        return;
    }
    if (!canUseWebsocketTransport()) return;
    ws.send(JSON.stringify({ type: 'stop' }));
}

function parseAssistantRaw(raw) {
    const segments = [];
    let i = 0;
    while (i < raw.length) {
        let nextIdx = -1;
        let openTag = '';
        let closeTag = '';
        for (const [o, c] of THINK_TAG_PAIRS) {
            const idx = raw.indexOf(o, i);
            if (idx !== -1 && (nextIdx === -1 || idx < nextIdx)) {
                nextIdx = idx;
                openTag = o;
                closeTag = c;
            }
        }
        if (nextIdx === -1) {
            if (i < raw.length) segments.push({ t: 'a', s: raw.slice(i) });
            break;
        }
        if (nextIdx > i) segments.push({ t: 'a', s: raw.slice(i, nextIdx) });
        const startInner = nextIdx + openTag.length;
        const closeIdx = raw.indexOf(closeTag, startInner);
        if (closeIdx === -1) {
            segments.push({ t: 'k', s: raw.slice(startInner) });
            break;
        }
        segments.push({ t: 'k', s: raw.slice(startInner, closeIdx) });
        i = closeIdx + closeTag.length;
    }
    return segments;
}

function attachThinkToolbarHandlers(box, toolbar) {
    const chevron = toolbar.querySelector('.think-chevron');
    toolbar.addEventListener('click', () => {
        box.classList.toggle('think-box--expanded');
        if (chevron) chevron.textContent = box.classList.contains('think-box--expanded') ? '\u25b4' : '\u25be';
    });
}

function formatReasoningDuration(durationMs) {
    const totalSeconds = Math.max(1, Math.round((Number(durationMs) || 0) / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const parts = [];
    if (minutes) parts.push(`${minutes} minute${minutes === 1 ? '' : 's'}`);
    if (seconds || !parts.length) parts.push(`${seconds} second${seconds === 1 ? '' : 's'}`);
    return parts.join(' ');
}

function updateThinkLabel(box, labelText) {
    const label = box?.querySelector('.think-label');
    if (label) label.textContent = labelText;
}

function buildThinkBoxCollapsed(text) {
    const box = document.createElement('div');
    box.className = 'think-box';
    const toolbar = document.createElement('div');
    toolbar.className = 'think-toolbar';
    toolbar.innerHTML = '<span class="think-label">Reasoning</span><span class="think-chevron" aria-hidden="true">\u25be</span>';
    const body = document.createElement('div');
    body.className = 'think-body think-body--collapsed';
    body.textContent = text;
    box.appendChild(toolbar);
    box.appendChild(body);
    attachThinkToolbarHandlers(box, toolbar);
    return box;
}

function hydrateAssistantFromRaw(msg, raw) {
    const stack = msg.querySelector('.think-stack');
    if (!stack) return;
    const segments = parseAssistantRaw(raw);
    let answer = '';
    for (const seg of segments) {
        if (seg.t === 'k' && seg.s) stack.appendChild(buildThinkBoxCollapsed(seg.s));
        else if (seg.t === 'a') answer += seg.s;
    }
    const trimmed = answer.trim();
    msg.dataset.originalContent = trimmed;
    const contentDiv = msg.querySelector('.message-content');
    if (contentDiv) contentDiv.textContent = trimmed;
}

function normalizeMessageFeedback(value) {
    const feedback = String(value || '').trim().toLowerCase();
    if (feedback === 'positive' || feedback === 'negative' || feedback === 'neutral') return feedback;
    return 'neutral';
}

function isRecordedMessageFeedback(value) {
    const feedback = normalizeMessageFeedback(value);
    return feedback === 'positive' || feedback === 'negative';
}

function getMessageFeedbackOption(value) {
    return MESSAGE_FEEDBACK_OPTION_MAP[normalizeMessageFeedback(value)] || null;
}

function getAssistantCopyText(msg) {
    return String(msg?.dataset?.originalContent || msg?.querySelector('.message-content')?.textContent || '').trim();
}

function syncAssistantMessageActions(msg) {
    if (!msg || !msg.classList.contains('assistant')) return;
    const feedback = normalizeMessageFeedback(msg.dataset.feedback);
    const hasMessageId = Boolean(String(msg.dataset.messageId || '').trim());
    const pending = msg.dataset.feedbackPending === 'true';
    const isEditing = msg.dataset.feedbackEditing === 'true';
    const hasRecordedFeedback = isRecordedMessageFeedback(feedback);
    const showSavedState = hasRecordedFeedback && !isEditing;
    const feedbackGroup = msg.querySelector('.message-feedback-group');
    const feedbackStatus = msg.querySelector('.message-feedback-status');

    if (feedbackGroup) {
        feedbackGroup.hidden = showSavedState;
    }
    if (feedbackStatus) {
        const option = getMessageFeedbackOption(feedback);
        const showStatus = hasRecordedFeedback;
        feedbackStatus.hidden = !showStatus;
        feedbackStatus.disabled = pending;
        feedbackStatus.classList.toggle('is-positive', feedback === 'positive' && !pending && !isEditing);
        feedbackStatus.classList.toggle('is-negative', feedback === 'negative' && !pending && !isEditing);
        feedbackStatus.classList.toggle('is-pending', pending || isEditing);
        if (showStatus && option) {
            const label = pending
                ? `Saving as ${option.label}...`
                : isEditing
                    ? 'Cancel'
                    : `Saved as ${option.label}`;
            feedbackStatus.textContent = label;
            if (pending) {
                feedbackStatus.title = 'Saving feedback...';
                feedbackStatus.setAttribute('aria-label', `Saving feedback as ${option.label}.`);
            } else if (isEditing) {
                feedbackStatus.title = 'Keep the saved feedback and close these controls.';
                feedbackStatus.setAttribute('aria-label', 'Keep the saved feedback and close these controls.');
            } else {
                feedbackStatus.title = `Feedback saved as ${option.label}. Click to change it.`;
                feedbackStatus.setAttribute('aria-label', `Feedback saved as ${option.label}. Click to change it.`);
            }
        } else {
            feedbackStatus.textContent = '';
            feedbackStatus.title = '';
            feedbackStatus.removeAttribute('aria-label');
        }
    }

    msg.querySelectorAll('.feedback-btn').forEach(btn => {
        const value = String(btn.dataset.feedbackValue || '');
        const active = feedback === value;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        btn.disabled = pending || !hasMessageId;
        if (!hasMessageId) {
            btn.title = 'Feedback is available after the reply is saved.';
        } else if (active) {
            btn.title = 'Click again to clear this rating.';
        } else if (value === 'positive') {
            btn.title = 'Mark this response as helpful.';
        } else {
            btn.title = 'Mark this response as unhelpful.';
        }
    });
}

async function submitMessageFeedback(msg, feedback) {
    if (!msg || !msg.classList.contains('assistant')) return;
    const messageId = Number.parseInt(msg.dataset.messageId || '', 10);
    if (!Number.isFinite(messageId)) return;

    const previousFeedback = normalizeMessageFeedback(msg.dataset.feedback);
    const previousEditing = msg.dataset.feedbackEditing === 'true';
    const nextFeedback = normalizeMessageFeedback(feedback);
    msg.dataset.feedback = nextFeedback;
    msg.dataset.feedbackPending = 'true';
    delete msg.dataset.feedbackEditing;
    syncAssistantMessageActions(msg);

    try {
        const resp = await fetch(`/api/message/${encodeURIComponent(messageId)}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feedback: nextFeedback }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
        msg.dataset.feedback = normalizeMessageFeedback(data.feedback || nextFeedback);
    } catch (error) {
        msg.dataset.feedback = previousFeedback;
        if (previousEditing) msg.dataset.feedbackEditing = 'true';
        else delete msg.dataset.feedbackEditing;
        console.error('Failed to save message feedback:', error);
    } finally {
        delete msg.dataset.feedbackPending;
        syncAssistantMessageActions(msg);
    }
}

function ensureAssistantMessageActions(msg) {
    if (!msg || !msg.classList.contains('assistant')) return;

    let actions = msg.querySelector('.message-actions');
    const tsDiv = msg.querySelector('.message-timestamp');
    if (!actions) {
        actions = document.createElement('div');
        actions.className = 'message-actions';
        if (tsDiv) msg.insertBefore(actions, tsDiv);
        else msg.appendChild(actions);
    }

    let copyBtn = actions.querySelector('[data-action="copy"]');
    if (!copyBtn) {
        copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.className = 'action-btn';
        copyBtn.dataset.action = 'copy';
        copyBtn.innerHTML = '&#128203; Copy';
        copyBtn.onclick = (e) => {
            e.stopPropagation();
            copyToClipboard(getAssistantCopyText(msg), copyBtn);
        };
        actions.appendChild(copyBtn);
    }

    let feedbackGroup = actions.querySelector('.message-feedback-group');
    if (!feedbackGroup && !DOCUMENT_CENTERED_MODE) {
        feedbackGroup = document.createElement('div');
        feedbackGroup.className = 'message-feedback-group';
        feedbackGroup.setAttribute('role', 'group');
        feedbackGroup.setAttribute('aria-label', 'Assistant response feedback');

        MESSAGE_FEEDBACK_OPTIONS.forEach(option => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'action-btn feedback-btn';
            btn.dataset.feedbackValue = option.value;
            btn.innerHTML = `${option.icon} ${option.label}`;
            btn.title = option.title;
            btn.onclick = (e) => {
                e.stopPropagation();
                const currentFeedback = normalizeMessageFeedback(msg.dataset.feedback);
                const nextFeedback = currentFeedback === option.value ? 'neutral' : option.value;
                submitMessageFeedback(msg, nextFeedback);
            };
            feedbackGroup.appendChild(btn);
        });
        actions.appendChild(feedbackGroup);
    }

    let feedbackStatus = actions.querySelector('.message-feedback-status');
    if (!feedbackStatus && !DOCUMENT_CENTERED_MODE) {
        feedbackStatus = document.createElement('button');
        feedbackStatus.type = 'button';
        feedbackStatus.className = 'message-feedback-status';
        feedbackStatus.hidden = true;
        feedbackStatus.setAttribute('aria-live', 'polite');
        feedbackStatus.onclick = (e) => {
            e.stopPropagation();
            if (msg.dataset.feedbackPending === 'true') return;
            if (msg.dataset.feedbackEditing === 'true') delete msg.dataset.feedbackEditing;
            else msg.dataset.feedbackEditing = 'true';
            syncAssistantMessageActions(msg);
        };
        actions.appendChild(feedbackStatus);
    }

    syncAssistantMessageActions(msg);
}

function createMessageElement(content, role, timestamp = null, options = {}) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    if (options.messageId !== undefined && options.messageId !== null) {
        msg.dataset.messageId = String(options.messageId);
    }

    if (role === 'assistant') {
        const stack = document.createElement('div');
        stack.className = 'think-stack';
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        msg.appendChild(stack);
        msg.appendChild(contentDiv);
        msg.dataset.needsMarkdown = 'true';
        msg.dataset.originalContent = '';
        msg.dataset.autoSpoken = 'false';
        msg.dataset.feedback = normalizeMessageFeedback(options.feedback);
        if (content && content.trim()) hydrateAssistantFromRaw(msg, content);
    } else {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        msg.appendChild(contentDiv);
    }

    const tsDiv = document.createElement('div');
    tsDiv.className = 'message-timestamp';
    tsDiv.textContent = formatTimestamp(timestamp || new Date().toISOString());
    msg.appendChild(tsDiv);

    return msg;
}

function appendStreamingAssistantMessageIfVisible() {
    if (!streamingAssistantMessage || activeStreamConversationId !== currentConvId) return;
    const messages = document.getElementById('messages');
    if (!messages || streamingAssistantMessage.parentElement === messages) return;
    preserveMessagesViewport(() => {
        messages.appendChild(streamingAssistantMessage);
    });
}

function ensureStreamingAssistantMessage() {
    if (!streamingAssistantMessage) {
        streamingAssistantMessage = createMessageElement('', 'assistant');
    }
    appendStreamingAssistantMessageIfVisible();
    syncStreamingAssistantStatusPanels(streamingAssistantMessage);
    return streamingAssistantMessage;
}

function currentStreamingAssistantMessage() {
    return streamingAssistantMessage;
}

function onAssistantThinkStart() {
    const msg = currentStreamingAssistantMessage();
    if (!msg) return;
    const stack = msg.querySelector('.think-stack');
    if (!stack) return;
    const box = document.createElement('div');
    box.className = 'think-box think-box--streaming';
    box.dataset.startedAt = String(Date.now());
    const toolbar = document.createElement('div');
    toolbar.className = 'think-toolbar';
    toolbar.innerHTML = '<span class="think-label">Reasoning\u2026</span><span class="think-chevron" aria-hidden="true">\u25be</span>';
    const body = document.createElement('div');
    body.className = 'think-body think-body--collapsed';
    box.appendChild(toolbar);
    box.appendChild(body);
    attachThinkToolbarHandlers(box, toolbar);
    preserveMessagesViewport(() => {
        stack.appendChild(box);
    });
}

function onAssistantThinkToken(text) {
    const msg = currentStreamingAssistantMessage();
    const box = msg?.querySelector('.think-box--streaming');
    const body = box?.querySelector('.think-body');
    if (!body) return;
    preserveMessagesViewport(() => {
        body.appendChild(document.createTextNode(text));
    });
}

function onAssistantThinkEnd() {
    const msg = currentStreamingAssistantMessage();
    const box = msg?.querySelector('.think-box--streaming');
    if (!box) return;
    box.classList.remove('think-box--streaming');
    const startedAt = Number(box.dataset.startedAt || 0);
    const durationMs = startedAt > 0 ? Date.now() - startedAt : 0;
    box.dataset.durationMs = String(durationMs);
    updateThinkLabel(box, `Reasoned for ${formatReasoningDuration(durationMs)}`);
    const body = box.querySelector('.think-body');
    if (body) {
        body.classList.add('think-body--collapsed');
    }
    preserveMessagesViewport(() => {});
}

function appendAssistantAnswerToken(text) {
    const msg = currentStreamingAssistantMessage();
    if (!msg) return;
    const contentDiv = msg.querySelector('.message-content');
    if (!contentDiv) return;
    preserveMessagesViewport(() => {
        msg.dataset.originalContent = (msg.dataset.originalContent || '') + text;
        contentDiv.textContent = msg.dataset.originalContent;
    });
}

function replaceAssistantAnswer(content) {
    const msg = currentStreamingAssistantMessage();
    if (!msg) return;
    const stack = msg.querySelector('.think-stack');
    const contentDiv = msg.querySelector('.message-content');
    if (!stack || !contentDiv) return;
    preserveMessagesViewport(() => {
        stack.innerHTML = '';
        msg.dataset.originalContent = '';
        msg.dataset.autoSpoken = 'false';
        delete msg.dataset.artifactPreviewAutoOpened;
        hydrateAssistantFromRaw(msg, content || '');
    });
}

function addMessage(content, role, timestamp = null, options = {}) {
    const msg = createMessageElement(content, role, timestamp, options);
    preserveMessagesViewport(() => {
        document.getElementById('messages').appendChild(msg);
    }, { forceBottom: Boolean(options.forceScroll) });
    return msg;
}
function showLoading() {
    removeLoading();
    const loading = document.createElement('div');
    loading.className = 'loading';
    loading.id = 'loadingIndicator';
    loading.innerHTML = `
        <div class="loading-spinner">
            <div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>
        </div>
        <div class="loading-text">Thinking...</div>
    `;
    preserveMessagesViewport(() => {
        document.getElementById('messages').appendChild(loading);
    }, { forceBottom: true });
}

function removeLoading() {
    document.getElementById('loadingIndicator')?.remove();
}

function scrollToBottom() {
    const messages = document.getElementById('messages');
    if (!messages) return;
    messages.scrollTop = messages.scrollHeight;
}

function messagesNearBottom(threshold = MESSAGE_SCROLL_BOTTOM_THRESHOLD) {
    const messages = document.getElementById('messages');
    if (!messages) return true;
    return (messages.scrollHeight - messages.clientHeight - messages.scrollTop) <= threshold;
}

function captureMessagesViewportAnchor(messages) {
    if (!messages) return null;
    const containerTop = messages.getBoundingClientRect().top;
    const anchor = Array.from(messages.children).find(child => {
        const rect = child.getBoundingClientRect();
        return rect.bottom > containerTop + 1;
    });
    if (!anchor) {
        return { element: null, offsetTop: 0, scrollTop: messages.scrollTop };
    }
    return {
        element: anchor,
        offsetTop: anchor.getBoundingClientRect().top - containerTop,
        scrollTop: messages.scrollTop,
    };
}

function restoreMessagesViewportAnchor(messages, anchor) {
    if (!messages || !anchor) return;
    if (!anchor.element || !anchor.element.isConnected) {
        messages.scrollTop = anchor.scrollTop;
        return;
    }
    const containerTop = messages.getBoundingClientRect().top;
    const currentOffsetTop = anchor.element.getBoundingClientRect().top - containerTop;
    messages.scrollTop += currentOffsetTop - anchor.offsetTop;
}

function preserveMessagesViewport(mutate, options = {}) {
    if (typeof mutate !== 'function') return;
    const messages = document.getElementById('messages');
    if (!messages) {
        mutate();
        return;
    }
    const shouldStickToBottom = Boolean(options.forceBottom) || messagesNearBottom();
    const viewportAnchor = shouldStickToBottom ? null : captureMessagesViewportAnchor(messages);
    mutate();
    requestAnimationFrame(() => {
        if (!messages.isConnected) return;
        if (shouldStickToBottom) {
            scrollToBottom();
            return;
        }
        restoreMessagesViewportAnchor(messages, viewportAnchor);
    });
}

function handleKeyDown(event) {
    const target = event.target;
    if (
        pendingPermissionRequest &&
        event.key === 'Enter' &&
        !event.shiftKey &&
        target?.id === 'input' &&
        !String(target.value || '').trim() &&
        !pendingAttachments.length
    ) {
        event.preventDefault();
        respondToPendingPermission(true);
        return;
    }

    const slashMenuOpen = !document.getElementById('slashMenu')?.hidden;
    if (slashMenuOpen) {
        if (event.key === 'ArrowDown') {
            event.preventDefault();
            event.stopPropagation();
            moveSlashSelection(1);
            return;
        }
        if (event.key === 'ArrowUp') {
            event.preventDefault();
            event.stopPropagation();
            moveSlashSelection(-1);
            return;
        }
        if (event.key === 'Tab') {
            event.preventDefault();
            event.stopPropagation();
            moveSlashSelection(event.shiftKey ? -1 : 1);
            return;
        }
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            event.stopPropagation();
            executeSlashCommand();
            return;
        }
        if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            hideSlashMenu();
            return;
        }
    }

    if (event.key === 'Escape' && isGenerating) {
        event.preventDefault();
        stopMessage();
        return;
    }

    const isPlainBackspace = event.key === 'Backspace' && !event.metaKey && !event.ctrlKey && !event.altKey;
    if (
        isPlainBackspace &&
        pendingAttachments.length &&
        target &&
        target.value === '' &&
        (target.selectionStart ?? 0) === 0 &&
        (target.selectionEnd ?? 0) === 0
    ) {
        event.preventDefault();
        pendingAttachments.pop();
        renderPendingAttachments();
        return;
    }

    if (
        event.key === 'ArrowUp' &&
        target?.id === 'input' &&
        pendingExecutionPlan?.executePrompt &&
        !buildSteps.length &&
        target.value === '' &&
        (target.selectionStart ?? 0) === 0 &&
        (target.selectionEnd ?? 0) === 0
    ) {
        if (focusExecutionPlanDraft('end')) {
            event.preventDefault();
            return;
        }
    }

    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handlePrimaryAction();
    }
}

function handleInterveneKeyDown(event) {
    if (event.key === 'Escape' && isGenerating) {
        event.preventDefault();
        stopMessage();
        return;
    }
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        sendIntervention();
    }
}

// ==================== Markdown ====================

function renderMarkdown() {
    if (!markedReady || typeof marked === 'undefined') {
        setTimeout(renderMarkdown, 100);
        return;
    }
    preserveMessagesViewport(() => {
        document.querySelectorAll('.message.assistant[data-needs-markdown="true"]').forEach(msg => {
            const contentDiv = msg.querySelector('.message-content');
            if (!contentDiv) return;
            // Read raw content from data attribute (answer only); thinking lives in .think-stack
            const content = msg.dataset.originalContent || contentDiv.textContent || contentDiv.innerText;
            const hasThink = msg.querySelector('.think-stack .think-box');
            if ((!content || !content.trim()) && !hasThink) return;

            const tsDiv = msg.querySelector('.message-timestamp');
            try {
                const html = (content && content.trim()) ? marked.parse(content) : '';
                contentDiv.innerHTML = html;
                renderMathContent(contentDiv);
                enhanceMessageArtifactReferences(msg, contentDiv, content);
                msg.dataset.needsMarkdown = 'false';
                msg.dataset.rendered = 'true';

                if (tsDiv && !msg.querySelector('.message-timestamp')) msg.appendChild(tsDiv);
                ensureAssistantMessageActions(msg);

                // Syntax highlighting + code copy buttons
                if (typeof hljs !== 'undefined') {
                    msg.querySelectorAll('pre code').forEach(block => {
                        try { if (block.textContent?.trim()) hljs.highlightElement(block); } catch (e) {}
                    });
                }
                msg.querySelectorAll('pre').forEach(pre => {
                    if (pre.querySelector('.copy-btn')) return;
                    const codeText = pre.querySelector('code')?.textContent || pre.textContent;
                    if (!codeText) return;
                    const btn = document.createElement('button');
                    btn.className = 'copy-btn';
                    btn.innerHTML = '&#128203; Copy';
                    btn.onclick = (e) => { e.stopPropagation(); copyToClipboard(codeText, btn); };
                    pre.appendChild(btn);
                });
                if (voiceSettings.autoSpeakReplies && supportsSpeechSynthesis() && msg.dataset.autoSpoken !== 'true') {
                    enqueueAssistantSpeech(msg);
                }
            } catch (e) {
                contentDiv.textContent = content;
                ensureAssistantMessageActions(msg);
            }
        });
    });
}

function copyToClipboard(text, button) {
    const onSuccess = () => {
        if (!button) return;
        const orig = button.innerHTML;
        button.innerHTML = '&#10003; Copied';
        button.classList.add('copied');
        setTimeout(() => { button.innerHTML = orig; button.classList.remove('copied'); }, 2000);
    };
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(onSuccess).catch(() => fallbackCopy(text, onSuccess));
    } else {
        fallbackCopy(text, onSuccess);
    }
}

function fallbackCopy(text, onSuccess) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    ta.style.top = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try { if (document.execCommand('copy')) onSuccess(); } catch (e) {}
    document.body.removeChild(ta);
}

function scheduleWorkspaceRefresh() {
    if (!featureSettings.agent_tools) return;
    if (workspaceRefreshTimer) clearTimeout(workspaceRefreshTimer);
    workspaceRefreshTimer = setTimeout(() => {
        refreshWorkspace(true);
        workspaceRefreshTimer = null;
    }, 150);
}

async function fetchWorkspaceDirectory(path = '.') {
    if (!currentWorkspaceId) throw new Error('No workspace selected');
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(currentWorkspaceId)}/files?path=${encodeURIComponent(path)}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    return data;
}

async function buildWorkspaceTree(path = '.', workspacePathHint = '') {
    const listing = await fetchWorkspaceDirectory(path);
    const children = await Promise.all((listing.items || []).map(async item => {
        if (item.type === 'directory') {
            return buildWorkspaceTree(item.path, workspacePathHint || listing.workspace_path || '');
        }
        return item;
    }));
    return {
        name: compareWorkspacePaths(path, '.') ? basename(workspacePathHint || listing.workspace_path || 'workspace') : basename(listing.path),
        path: normalizeWorkspacePath(listing.path || path || '.'),
        type: 'directory',
        children,
        run_id: listing.run_id || null,
        workspace_path: listing.workspace_path || workspacePathHint || '',
    };
}

async function refreshWorkspace(force = false) {
    const pathEl = document.getElementById('workspacePath');
    const listEl = document.getElementById('workspaceFileList');
    if (!pathEl || !listEl) return;

    if (!featureSettings.agent_tools) {
        workspaceEntries = [];
        workspaceTree = null;
        workspaceStats = { files: 0, directories: 0 };
        workspaceActivity = [];
        buildSteps = [];
        selectedWorkspaceFile = '';
        selectedSpreadsheetSheet = '';
        closeInlineViewer();
        pathEl.textContent = 'Agent tools disabled';
        listEl.innerHTML = '<div class="workspace-empty">Agent dev tools are turned off.</div>';
        renderWorkspaceArtifactList();
        renderBuildSteps();
        renderWorkspaceActivity();
        return;
    }

    if (!currentWorkspaceId) {
        workspaceEntries = [];
        workspaceTree = null;
        workspaceStats = { files: 0, directories: 0 };
        currentRunId = null;
        selectedWorkspaceFile = '';
        selectedSpreadsheetSheet = '';
        closeInlineViewer();
        pathEl.textContent = 'No workspace selected';
        listEl.innerHTML = '<div class="workspace-empty">Choose a workspace to browse its files here.</div>';
        renderWorkspaceArtifactList();
        return;
    }

    if (force) {
        const label = currentWorkspaceMeta?.display_name || currentWorkspaceMeta?.root_path || 'Refreshing workspace...';
        pathEl.textContent = label;
    }

    try {
        const tree = await buildWorkspaceTree('.');
        workspaceTree = tree;
        workspaceEntries = flattenWorkspaceFiles(tree, []);
        workspaceStats = collectWorkspaceStats(tree);
        currentRunId = tree.run_id || null;
        pathEl.textContent = currentWorkspaceMeta?.root_path || tree.workspace_path || currentWorkspaceMeta?.display_name || 'Workspace';
        const hasSelectedFile = workspaceEntries.some(item => item.path === selectedWorkspaceFile);
        if (!hasSelectedFile) selectedWorkspaceFile = '';
        if (inlineViewerPath && !workspaceEntries.some(item => item.path === inlineViewerPath)) closeInlineViewer();
        if (workspaceEntries.length) {
            applyWorkspacePanelState();
        } else {
            closeInlineViewer();
        }
        renderFileList();
    } catch (e) {
        currentRunId = null;
        workspaceTree = null;
        workspaceEntries = [];
        workspaceStats = { files: 0, directories: 0 };
        pathEl.textContent = currentWorkspaceMeta?.root_path || 'Workspace unavailable';
        listEl.innerHTML = `<div class="workspace-empty">Workspace is unavailable: ${escapeHtml(e.message)}</div>`;
        renderWorkspaceArtifactList();
    }
}

function setCurrentWorkspaceSelection(workspaceId) {
    const nextId = String(workspaceId || '').trim();
    const workspaceChanged = nextId !== currentWorkspaceId;
    currentWorkspaceId = nextId;
    currentWorkspaceMeta = workspaceCatalog.find(item => item.id === nextId) || null;
    if (nextId) localStorage.setItem('lastWorkspaceId', nextId);
    else localStorage.removeItem('lastWorkspaceId');
    if (workspaceChanged) {
        draftFileSessions = [];
        draftSessionDefaultPath = '';
    }
    renderDraftFileExplorer();
    if (workspaceChanged || !activeDraftPath) {
        syncDraftHeader(defaultDraftPathForWorkspace(nextId));
        return;
    }
    syncDraftHeader(activeDraftPath);
}

function hasConversationMessages() {
    if (DOCUMENT_CENTERED_MODE) {
        const session = getDraftSessionRecord(currentWorkspaceId, activeDraftPath);
        if ((Number(session?.message_count) || 0) > 0) return true;
    }
    return Boolean(document.querySelector('#messages .message'));
}

function renderWorkspaceCatalog() {
    const container = document.getElementById('workspaceCatalog');
    if (!container) return;
    if (!workspaceCatalog.length) {
        container.innerHTML = '<div class="workspace-catalog-empty">No workspaces yet.</div>';
        return;
    }

    container.innerHTML = '';
    workspaceCatalog.forEach(workspace => {
        const item = document.createElement('div');
        item.className = `workspace-catalog-item${workspace.id === currentWorkspaceId ? ' active' : ''}`;
        item.innerHTML = `
            <div class="workspace-catalog-copy">
                <div class="workspace-catalog-name">${escapeHtml(workspace.display_name || DEFAULT_WORKSPACE_NAME)}</div>
                <div class="workspace-catalog-path">${escapeHtml(workspace.root_path || '')}</div>
            </div>
            <div class="workspace-catalog-actions">
                <button class="conv-btn" onclick="event.stopPropagation(); renameWorkspace('${workspace.id}', '${(workspace.display_name || '').replace(/'/g, "\\'")}')" title="Rename workspace">&#9998;</button>
                <button class="conv-btn delete" onclick="event.stopPropagation(); removeWorkspace('${workspace.id}')" title="Remove workspace">&#128465;</button>
            </div>
        `;
        item.onclick = () => { selectWorkspace(workspace.id); };
        container.appendChild(item);
    });
}

async function loadWorkspaceCatalog(options = {}) {
    try {
        const resp = await fetch('/api/workspaces');
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
        workspaceCatalog = Array.isArray(data.workspaces) ? data.workspaces : [];

        let nextWorkspaceId = String(options.workspaceId || '').trim() || currentWorkspaceId;
        if (!workspaceCatalog.some(item => item.id === nextWorkspaceId)) {
            nextWorkspaceId = String(data.default_workspace_id || workspaceCatalog[0]?.id || '').trim();
        }
        setCurrentWorkspaceSelection(nextWorkspaceId);
        if (nextWorkspaceId) {
            await loadDraftFileSessions(nextWorkspaceId);
        } else {
            draftFileSessions = [];
            draftSessionDefaultPath = '';
            renderDraftFileExplorer();
        }
        renderWorkspaceCatalog();
        return workspaceCatalog;
    } catch (error) {
        workspaceCatalog = [];
        currentWorkspaceMeta = null;
        draftFileSessions = [];
        draftSessionDefaultPath = '';
        renderDraftFileExplorer();
        renderWorkspaceCatalog();
        throw error;
    }
}

async function selectWorkspace(workspaceId, options = {}) {
    const nextWorkspaceId = String(workspaceId || '').trim();
    if (!nextWorkspaceId || nextWorkspaceId === currentWorkspaceId) return;

    const switchingLoadedConversation = hasConversationMessages() && currentWorkspaceId && currentWorkspaceId !== nextWorkspaceId;
    if (switchingLoadedConversation && options.skipConfirm !== true) {
        const confirmed = confirm('Switching workspaces starts a new chat so the current transcript stays attached to its existing workspace. Continue?');
        if (!confirmed) return;
    }

    setCurrentWorkspaceSelection(nextWorkspaceId);
    renderWorkspaceCatalog();
    if (switchingLoadedConversation) {
        newChat({ preserveWorkspaceSelection: true });
    } else {
        currentRunId = null;
        await loadDraftFileSessions(nextWorkspaceId);
        refreshWorkspace(true);
        await loadDraftDocument(defaultDraftPathForWorkspace(nextWorkspaceId));
    }
}

async function promptCreateWorkspace() {
    const displayName = String(prompt('Workspace name', DEFAULT_WORKSPACE_NAME) || '').trim();
    if (!displayName) return;
    const rootInput = prompt('Absolute folder path to use. Leave blank to create a folder under the server workspace root.', '') || '';
    const requestedPath = String(rootInput).trim();
    const createIfMissing = requestedPath ? confirm('Create the folder if it does not already exist?') : true;

    const resp = await fetch('/api/workspaces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            display_name: displayName,
            root_path: requestedPath || null,
            create_if_missing: createIfMissing,
        }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        alert(data.detail || `Workspace create failed (HTTP ${resp.status})`);
        return;
    }
    await loadWorkspaceCatalog({ workspaceId: data.id || '' });
    await selectWorkspace(data.id || '', { skipConfirm: true });
}

async function renameWorkspace(workspaceId, currentTitle) {
    const title = String(prompt('Rename workspace', currentTitle || '') || '').trim();
    if (!title) return;
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(workspaceId)}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        alert(data.detail || `Workspace rename failed (HTTP ${resp.status})`);
        return;
    }
    await loadWorkspaceCatalog({ workspaceId });
}

async function removeWorkspace(workspaceId) {
    if (!confirm('Remove this workspace from the catalog? Existing chats must be moved or deleted first.')) return;
    const removedActiveWorkspace = workspaceId === currentWorkspaceId;
    const resp = await fetch(`/api/workspaces/${encodeURIComponent(workspaceId)}`, { method: 'DELETE' });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        alert(data.detail || `Workspace delete failed (HTTP ${resp.status})`);
        return;
    }
    await loadWorkspaceCatalog();
    if (removedActiveWorkspace) {
        setCurrentWorkspaceSelection(workspaceCatalog[0]?.id || '');
        newChat({ preserveWorkspaceSelection: true });
    } else {
        renderWorkspaceCatalog();
    }
}

// ==================== Search ====================

let searchTimeout = null;

async function handleSearch(query) {
    const searchResults = document.getElementById('searchResults');
    const conversations = document.getElementById('conversations');
    if (searchTimeout) clearTimeout(searchTimeout);

    if (!query || query.trim().length < 2) {
        searchResults.style.display = 'none';
        searchResults.innerHTML = '';
        conversations.style.display = 'block';
        return;
    }

    searchTimeout = setTimeout(async () => {
        try {
            const resp = await fetch(`/api/search?query=${encodeURIComponent(query.trim())}`);
            const data = await resp.json();
            conversations.style.display = 'none';
            searchResults.style.display = 'block';
            searchResults.innerHTML = '';

            if (data.results && data.results.length > 0) {
                const grouped = {};
                data.results.forEach(r => {
                    if (!grouped[r.conversation_id]) grouped[r.conversation_id] = [];
                    grouped[r.conversation_id].push(r);
                });
                Object.keys(grouped).forEach(convId => {
                    const results = grouped[convId];
                    const first = results[0];
                    const item = document.createElement('div');
                    item.className = 'search-result-item';
                    item.onclick = () => {
                        loadConversation(convId);
                        document.getElementById('searchInput').value = '';
                        searchResults.style.display = 'none';
                        conversations.style.display = 'block';
                        closeMenu();
                    };
                    const preview = first.content.substring(0, 100);
                    item.innerHTML = `
                        <div class="search-result-title">${first.conversation_title}</div>
                        <div class="search-result-preview">${preview}${first.content.length > 100 ? '...' : ''}</div>
                        <div class="search-result-meta">${results.length} match${results.length > 1 ? 'es' : ''}</div>
                    `;
                    searchResults.appendChild(item);
                });
            } else {
                searchResults.innerHTML = '<div class="search-no-results">No results found</div>';
            }
        } catch (e) {
            searchResults.innerHTML = '<div class="search-no-results">Search error</div>';
        }
    }, 300);
}

// ==================== Conversations ====================

async function loadConversations() {
    try {
        const resp = await fetch('/api/conversations');
        const data = await resp.json();
        const container = document.getElementById('conversations');
        container.innerHTML = '';

        data.conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conv-item' + (conv.id === currentConvId ? ' active' : '');

            let preview = (conv.last_message || '').replace(/[#*_`\[\]]/g, '').replace(/\n/g, ' ').trim();
            if (preview.length > 80) preview = preview.substring(0, 80) + '...';

            const ts = formatTimestamp(conv.last_message_timestamp || conv.updated_at);
            const workspaceLabel = String(conv.workspace_display_name || '').trim();

            item.innerHTML = `
                <div class="conv-title">${conv.title}</div>
                ${workspaceLabel ? `<div class="conv-workspace">${escapeHtml(workspaceLabel)}</div>` : ''}
                <div class="conv-preview">${preview || 'No messages yet'}</div>
                <div class="conv-timestamp">${ts}</div>
                <div class="conv-actions">
                    <button class="conv-btn" onclick="event.stopPropagation(); renameConv('${conv.id}', '${conv.title.replace(/'/g, "\\'")}')" title="Rename">&#9998;</button>
                    <button class="conv-btn delete" onclick="event.stopPropagation(); deleteConv('${conv.id}')" title="Delete">&#128465;</button>
                </div>
            `;
            item.onclick = () => { loadConversation(conv.id); closeMenu(); };
            container.appendChild(item);
        });
    } catch (e) {
        console.error('Failed to load conversations:', e);
    }
}

async function loadConversation(id, options = {}) {
    const preserveActivity = Boolean(options.preserveActivity);
    const preserveScroll = Boolean(options.preserveScroll);
    activeDraftForegroundJobId = '';
    currentConvId = id;
    syncToolApprovalToggle();
    currentRunId = null;
    currentAssistantTurnStartedAt = null;
    currentAssistantTurnArtifactPaths = new Set();
    latestAssistantTurnArtifactPaths = new Set();
    selectedWorkspaceFile = '';
    selectedSpreadsheetSheet = '';
    clearBuildSteps();
    dismissExecutionPlan();
    clearPendingPermissionRequest();
    if (!preserveActivity) clearWorkspaceActivity();
    closeInlineViewer();
    clearPendingAttachments();
    const resp = await fetch(`/api/conversation/${id}`);
    const data = await resp.json();
    if (data.workspace_id) {
        setCurrentWorkspaceSelection(data.workspace_id);
    }
    try {
        await loadWorkspaceCatalog({ workspaceId: currentWorkspaceId || data.workspace_id || '' });
    } catch (error) {
        console.error('Failed to refresh workspaces while loading conversation:', error);
    }
    const sessionPath = options.filePath
        || getDraftSessionPathForConversation(currentWorkspaceId, id)
        || defaultDraftPathForWorkspace(currentWorkspaceId);
    await loadDraftDocument(sessionPath, { preserveAgentState: preserveActivity });
    const messages = document.getElementById('messages');
    stopSpeaking();
    messages.innerHTML = '';
    exitWelcomeMode();
    data.messages.forEach(msg => {
        const el = addMessage(msg.content, msg.role, msg.timestamp, { messageId: msg.id, feedback: msg.feedback });
        if (msg.role === 'assistant') el.dataset.autoSpoken = 'true';
    });
    if (!preserveScroll) scrollToBottom();
    if (data.pending_plan?.execute_prompt) {
        storeExecutionPlan(
            data.pending_plan.summary || '',
            data.pending_plan.execute_prompt || '',
            data.pending_plan.builder_steps || [],
            { focusComposer: false },
        );
    } else {
        dismissExecutionPlan();
    }
    appendStreamingAssistantMessageIfVisible();
    setTimeout(() => renderMarkdown(), 100);
    loadConversations();
    refreshWorkspace(true);
}

function newChat(options = {}) {
    const preserveWorkspaceSelection = options.preserveWorkspaceSelection !== false;
    dismissMobileKeyboard(true);
    stopSpeaking();
    activeDraftForegroundJobId = '';
    clearAllowedCommandsForConversation(currentConvId);
    clearAllowedToolPermissionsForConversation(currentConvId);
    currentConvId = generateId();
    syncToolApprovalToggle();
    currentRunId = null;
    currentAssistantTurnStartedAt = null;
    currentAssistantTurnArtifactPaths = new Set();
    latestAssistantTurnArtifactPaths = new Set();
    selectedWorkspaceFile = '';
    selectedSpreadsheetSheet = '';
    clearBuildSteps();
    dismissExecutionPlan();
    clearPendingPermissionRequest();
    clearWorkspaceActivity();
    closeInlineViewer();
    clearPendingAttachments();
    document.getElementById('messages').innerHTML = buildWelcomeMarkup();
    enterWelcomeMode();
    if (!preserveWorkspaceSelection) {
        setCurrentWorkspaceSelection(workspaceCatalog[0]?.id || '');
    }
    loadDraftFileSessions(currentWorkspaceId)
        .catch(error => {
            console.error('Failed to refresh file sessions for the new chat:', error);
        })
        .finally(() => {
            loadDraftDocument(defaultDraftPathForWorkspace()).catch(error => {
                console.error('Failed to load the active draft for the new chat:', error);
            });
        });
    renderWorkspaceCatalog();
    loadConversations();
    refreshWorkspace(true);
    renderDraftFileExplorer();
    refreshDraftVersionsSidebar().catch(() => {});
    refreshDraftOutputPane().catch(() => {});
}

// ==================== Modals ====================

function renameConv(id, currentTitle) {
    renameConvId = id;
    document.getElementById('renameInput').value = currentTitle;
    document.getElementById('renameModal').classList.add('show');
    document.getElementById('renameInput').focus();
}
function closeRenameModal() {
    document.getElementById('renameModal').classList.remove('show');
    renameConvId = null;
}
async function confirmRename() {
    const newTitle = document.getElementById('renameInput').value.trim();
    if (!newTitle || !renameConvId) return;
    await fetch(`/api/conversation/${renameConvId}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle })
    });
    closeRenameModal();
    loadConversations();
}

async function deleteConv(id) {
    if (!confirm('Delete this conversation?')) return;
    await fetch(`/api/conversation/${id}`, { method: 'DELETE' });
    clearAllowedCommandsForConversation(id);
    clearAllowedToolPermissionsForConversation(id);
    clearToolAutoApproveForConversation(id);
    clearRememberedTurnFeatures(id);
    if (id === currentConvId) newChat({ preserveWorkspaceSelection: true }); else loadConversations();
}

// System Prompt
function openSystemPromptEditor() {
    document.getElementById('systemPromptInput').value = localStorage.getItem('customSystemPrompt') || '';
    document.getElementById('systemPromptModal').classList.add('show');
    document.getElementById('systemPromptInput').focus();
}
function closeSystemPromptModal() {
    document.getElementById('systemPromptModal').classList.remove('show');
}
function saveSystemPrompt() {
    const prompt = document.getElementById('systemPromptInput').value.trim();
    if (prompt) localStorage.setItem('customSystemPrompt', prompt);
    else localStorage.removeItem('customSystemPrompt');
    closeSystemPromptModal();
}
function resetSystemPrompt() {
    document.getElementById('systemPromptInput').value = '';
    localStorage.removeItem('customSystemPrompt');
}

// ==================== Escape Key ====================

document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'a' && inlineViewerEditable && inlineViewerPath && inlineViewerView === 'edit') {
        const selection = getInlineSelection().trim();
        if (selection) {
            e.preventDefault();
            sendInlineSelectionToChat();
            return;
        }
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's' && inlineViewerEditable && inlineViewerPath && inlineViewerView === 'edit') {
        e.preventDefault();
        saveInlineViewer().catch(() => {});
        return;
    }
    if (e.key !== 'Escape') return;
    if (document.getElementById('renameModal').classList.contains('show')) { closeRenameModal(); return; }
    if (document.getElementById('systemPromptModal').classList.contains('show')) { closeSystemPromptModal(); return; }
    if (document.getElementById('aboutOverlay').classList.contains('show')) { closeAbout(); return; }
    if (document.getElementById('settingsOverlay').classList.contains('show')) { closeSettings(); return; }
    if (document.getElementById('menuOverlay').classList.contains('show')) { closeMenu(); return; }
    if (document.getElementById('fileExplorerOverlay').classList.contains('show')) { closeFileExplorer(); return; }
    if (workspacePanelOpen) { closeWorkspacePanel(); }
});

document.addEventListener('click', (event) => {
    if (!workspacePanelOpen) return;
    const ideWorkspace = document.getElementById('ideWorkspace');
    const toggle = document.getElementById('workspaceToggle');
    const target = event.target;
    if (!(target instanceof Node)) return;
    const eventPath = typeof event.composedPath === 'function' ? event.composedPath() : [];
    const insideWorkspace = eventPath.length
        ? [ideWorkspace, toggle].some(node => node && eventPath.includes(node))
        : Boolean(ideWorkspace?.contains(target) || toggle?.contains(target));
    if (insideWorkspace) return;
    closeWorkspacePanel();
});

// ==================== Init ====================

updateStatus('loading');
connectWS();
applyFeatureSettingsToUI();
syncMobileReasoningBadge();
syncDraftHeader(defaultDraftPathForWorkspace());
setDraftView(draftView);
renderDraftPreview();
renderDraftFileExplorer();
loadWorkspaceCatalog()
    .catch(error => {
        console.error('Failed to load workspaces:', error);
    })
    .finally(() => {
        loadConversations();
        refreshWorkspace(true);
        loadDraftDocument(defaultDraftPathForWorkspace()).catch(error => {
            console.error('Failed to load the active draft on startup:', error);
        });
        refreshDraftVersionsSidebar().catch(() => {});
        refreshDraftOutputPane().catch(() => {});
    });
startWelcomeHintRotation();

setTimeout(() => {
    fetch('/health').then(r => r.json()).then(health => {
        if (health.model_available) {
            updateStatus('connected');
        } else {
            if (!modelAvailable) updateStatus('loading');
            startHealthPolling();
        }
    }).catch(() => {
        if (!modelAvailable) updateStatus('loading');
        startHealthPolling();
    });
}, 1000);

// Safety-net: if the UI is stuck in a non-connected state for more than a few
// seconds, re-check health and restart polling if needed.  This covers edge
// cases where the one-shot health checks and WS-triggered polls all missed the
// model becoming available (e.g. WS idle-disconnect race, transient fetch
// failures, or applyModelRuntime overwriting a ready state with stale data).
setInterval(() => {
    if (modelAvailable) return;
    fetch('/health').then(r => r.json()).then(health => {
        if (health.model_available) {
            updateStatus('connected');
        } else if (!healthPollInterval) {
            startHealthPolling();
        }
    }).catch(() => {});
}, 10000);

const _input = document.getElementById('input');
if (_input) {
    observeMobileComposerSize();
    handleInputChange(_input);
    window.addEventListener('resize', () => {
        autoResizeTextarea(_input);
        syncMobileViewportState();
    });
    window.addEventListener('resize', positionSlashMenu);
    window.addEventListener('resize', syncChatShellLayout);
    window.addEventListener('scroll', positionSlashMenu, true);
    if (!isMobileViewport()) _input.focus();
    _input.addEventListener('focus', () => {
        updateSlashMenu(_input);
        syncMobileViewportState();
    });
    _input.addEventListener('blur', () => {
        window.setTimeout(syncMobileViewportState, 30);
        setTimeout(() => {
            if (document.activeElement?.closest?.('#slashMenu')) return;
            hideSlashMenu();
        }, 120);
    });

    // Collapse long pastes (>10 lines) into a placeholder; actual text is sent on submit
    _input.addEventListener('paste', (e) => {
        const text = (e.clipboardData || window.clipboardData).getData('text');
        const lines = text.split('\n');
        if (lines.length > 10) {
            e.preventDefault();
            const placeholder = `[${lines.length} lines pasted]`;
            pastedBlocks.push({ placeholder, actual: text });

            const start = _input.selectionStart;
            const end = _input.selectionEnd;
            const before = _input.value.substring(0, start);
            const after = _input.value.substring(end);
            _input.value = before + placeholder + after;
            _input.selectionStart = _input.selectionEnd = start + placeholder.length;
            handleInputChange(_input);
        }
    });
}

if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', () => {
        syncMobileViewportState();
        if (_input) autoResizeTextarea(_input);
    });
    window.visualViewport.addEventListener('scroll', syncMobileViewportState);
}

syncReasoningSelector();
renderWorkspaceActivity();
renderPermissionPanel();
syncToolApprovalToggle();
applyViewerMetadata('inlineViewer', '');
syncWorkspaceViewMode();
syncMobileViewportState();
syncChatShellLayout();
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopSpeaking();
        if (dictationActive) stopDictationAndUpload();
    }
});
document.addEventListener('pointerdown', (event) => {
    if (!isMobileViewport()) return;
    const target = event.target;
    if (!(target instanceof Node)) return;
    const inputArea = document.getElementById('inputArea');
    if (inputArea?.contains(target)) return;
    dismissMobileKeyboard();
});
