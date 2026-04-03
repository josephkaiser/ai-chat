// AI Chat Application

// ==================== State ====================

let ws = null;
let logWs = null;
let currentConvId = generateId();
const uiSessionId = generateId();
let isGenerating = false;
let activeStreamConversationId = null;
let streamingAssistantMessage = null;
let renameConvId = null;
let modelAvailable = false;
let markedReady = false;
let healthPollInterval = null;
let pastedBlocks = []; // tracks collapsed long pastes: [{placeholder, actual}]
let deepMode = localStorage.getItem('deepMode') === 'true';
let availableModelProfiles = [];
let selectedModelProfileKey = '';
let activeModelProfileKey = '';
let modelSwitchInFlight = false;
let uiReadyReportInFlight = false;
let lastReportedUiReadyKey = '';
let dashboardRefreshTimer = null;
let chatCommandAllowlists = {};
let workspaceEntries = [];
let workspaceTree = null;
let workspaceStats = { files: 0, directories: 0 };
let workspaceActivity = [];
let terminalEntries = [];
let terminalLiveBuffer = '';
let terminalWs = null;
let terminalEmulator = null;
let terminalFitAddon = null;
let terminalUnavailableReason = '';
let buildSteps = [];
let currentBuildStepIndex = null;
let selectedWorkspaceFile = '';
let selectedSpreadsheetSheet = '';
let workspaceRefreshTimer = null;
let workspacePanelOpen = localStorage.getItem('workspacePanelOpen') !== 'false';
let collapsedWorkspaceDirs = new Set();
let featureSettings = null;
let voiceSettings = null;
let pendingAttachments = [];
let inlineViewerPath = '';
let inlineViewerKind = 'text';
let inlineViewerEditable = false;
let inlineViewerView = 'edit';
let inlineEditor = null;
let inlineEditorReady = false;
let inlineViewerAutosaveTimer = null;
let inlineViewerLastSavedContent = '';
let inlineViewerSaveState = 'idle';
let inlineViewerUndoStacks = {};
let inlineViewerApplyingRemoteContent = false;
let inlineViewerPerformingUndo = false;
let currentFileSymbols = [];
let symbolGroupState = {};
let currentAssistantTurnStartedAt = null;
let currentAssistantTurnArtifactPaths = new Set();
let latestAssistantTurnArtifactPaths = new Set();
let slashMenuItems = [];
let slashMenuSelectedIndex = 0;
let petProfile = null;
let petExists = false;
let welcomeMascotRuntime = null;
let welcomeHintTimer = null;
let currentRunId = null;
let pendingExecutionPlan = null;
let dictationActive = false;
let currentAudio = null;
let speechQueue = [];
let speechQueueBusy = false;
let activeSpeechTask = null;
let speechQueueVersion = 0;
let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
const MOBILE_WORKSPACE_MEDIA_QUERY = '(max-width: 768px)';
const SEND_BUTTON_ICON = '&#10148;';
const STOP_BUTTON_ICON = '&#9632;';
const MIC_BUTTON_ICON = '&#127908;';
const MIC_ACTIVE_BUTTON_ICON = '&#9209;';
let voiceRuntime = {
    tts_available: false,
    stt_available: false,
    tts_backend: 'unavailable',
    stt_backend: 'unavailable',
    tts_voice: '',
};
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
const LEGACY_COMPY_THEME = Object.freeze({
    theme_primary: '#2563eb',
    theme_secondary: '#0f172a',
    theme_accent: '#dbeafe',
});
const WOLFY_THEME_PALETTES = Object.freeze({
    light: Object.freeze({
        theme_primary: '#d97706',
        theme_secondary: '#6b4f2a',
        theme_accent: '#f5e6c8',
    }),
    dark: Object.freeze({
        theme_primary: '#f59e0b',
        theme_secondary: '#f3d7ad',
        theme_accent: '#5b3a12',
    }),
});
const DISCOVERY_HINTS = Object.freeze([
    'Ask for structured thinking: "Work through this step by step and show the conclusion clearly."',
    'I can help with logic, math, and careful reasoning from the information you give me.',
    'Try: "Compare these options and recommend one with tradeoffs."',
    'Prompt idea: "Summarize this and pull out the key decisions or risks."',
    'I can analyze pasted text, notes, logs, screenshots, and attached files.',
    'Try: "Turn this rough idea into a clear plan, checklist, or draft."',
    'Prompt idea: "Question my assumptions and point out what I may be missing."',
    'I can help write, revise, explain, brainstorm, and organize complex information.',
    'Ask for depth control directly: "Give me the short version" or "Go deep on this."',
    'When useful, I can use tools to inspect files, run checks, or verify details in the workspace.',
]);

// Must match thinking_stream.py THINK_TAG_PAIRS (redacted_thinking + think)
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
});

const WORKSPACE_ACTIVITY_LIMIT = 80;
const INLINE_VIEWER_AUTOSAVE_DELAY_MS = 700;
const INLINE_VIEWER_UNDO_LIMIT = 40;
const PLAN_EXECUTION_MARKERS = Object.freeze([
    'execute approved plan',
    'execute this approved plan',
    'execute the approved plan',
    'execute plan now',
    'execute this plan',
    'run this plan',
    'run the plan',
]);
const APPROVED_PLAN_EXECUTION_MESSAGE = 'Execute approved plan.';

featureSettings = loadFeatureSettings();
voiceSettings = loadVoiceSettings();

// ==================== Theme ====================

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
    if (icon) icon.innerHTML = mode === 'dark' ? '&#9728;' : '&#127769;';

    applyPetTheme(petProfile, mode);
}

function normalizeThemeColor(value) {
    return String(value || '').trim().toLowerCase();
}

function resolveDefaultWolfyPalette(mode) {
    return mode === 'dark' ? WOLFY_THEME_PALETTES.dark : WOLFY_THEME_PALETTES.light;
}

function paletteMatchesProfile(profile, palette) {
    return (
        normalizeThemeColor(profile?.theme_primary) === palette.theme_primary &&
        normalizeThemeColor(profile?.theme_secondary) === palette.theme_secondary &&
        normalizeThemeColor(profile?.theme_accent) === palette.theme_accent
    );
}

function shouldUseThemeResponsivePetPalette(profile) {
    if (!profile) return true;
    return (
        paletteMatchesProfile(profile, LEGACY_COMPY_THEME) ||
        paletteMatchesProfile(profile, WOLFY_THEME_PALETTES.light) ||
        paletteMatchesProfile(profile, WOLFY_THEME_PALETTES.dark)
    );
}

function applyPetTheme(profile, mode) {
    const root = document.documentElement;
    const currentMode = mode || root.dataset.theme || localStorage.getItem('theme') || 'light';
    const fallback = resolveDefaultWolfyPalette(currentMode);
    const colors = shouldUseThemeResponsivePetPalette(profile)
        ? fallback
        : {
            theme_primary: profile?.theme_primary || fallback.theme_primary,
            theme_secondary: profile?.theme_secondary || fallback.theme_secondary,
            theme_accent: profile?.theme_accent || fallback.theme_accent,
        };
    root.style.setProperty('--pet_primary', colors.theme_primary);
    root.style.setProperty('--pet_secondary', colors.theme_secondary);
    root.style.setProperty('--pet_accent', colors.theme_accent);
}

function toggleTheme() {
    const current = localStorage.getItem('theme') || 'light';
    applyTheme(current === 'light' ? 'dark' : 'light');
    // Re-render markdown for new theme
    document.querySelectorAll('.message.assistant[data-rendered="true"]').forEach(msg => {
        msg.dataset.needsMarkdown = 'true';
        msg.dataset.rendered = 'false';
    });
    renderMarkdown();
}

// Apply theme immediately
applyTheme(localStorage.getItem('theme') || 'light');

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

function generateId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

function setModelNote(modelName) {
    const note = document.querySelector('.model-note');
    if (note && modelName) note.textContent = modelName;
    if (modelName) window.MODEL_NAME = modelName;
}

function syncReasoningSelector() {
    const select = document.getElementById('reasoningSelect');
    if (!select) return;
    select.value = deepMode ? 'high' : 'low';
    select.title = deepMode ? 'High reasoning effort enabled' : 'Low reasoning effort enabled';
}

function handleReasoningSelectChange(value) {
    deepMode = value === 'high';
    localStorage.setItem('deepMode', deepMode ? 'true' : 'false');
    syncReasoningSelector();
    syncMobileReasoningBadge();
}

function toggleDeepModeMobile() {
    handleReasoningSelectChange(deepMode ? 'low' : 'high');
}

function syncMobileReasoningBadge() {
    const badge = document.getElementById('mobileReasoningBadge');
    if (!badge) return;
    badge.textContent = deepMode ? 'High' : 'Low';
    badge.classList.toggle('is-high', deepMode);
}

function syncModelSelector() {
    const select = document.getElementById('modelSelect');
    if (!select) return;

    const profiles = Array.isArray(availableModelProfiles) ? availableModelProfiles : [];
    const nextValue = selectedModelProfileKey || activeModelProfileKey || select.value || '';

    select.innerHTML = '';
    if (!profiles.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = modelSwitchInFlight ? 'Switching model...' : (window.MODEL_NAME || 'Loading model...');
        select.appendChild(option);
        select.disabled = true;
        return;
    }

    profiles.forEach(profile => {
        const option = document.createElement('option');
        option.value = profile.key;
        const status = profile.active ? 'Active' : (profile.selected ? 'Selected' : '');
        option.textContent = status ? `${profile.label} (${status})` : profile.label;
        select.appendChild(option);
    });

    select.value = profiles.some(profile => profile.key === nextValue)
        ? nextValue
        : (profiles.find(profile => profile.selected)?.key || profiles[0].key);
    select.disabled = modelSwitchInFlight || isGenerating || !modelAvailable;
}

function extractProfileKey(profileValue) {
    if (!profileValue) return '';
    if (typeof profileValue === 'string') return profileValue;
    if (typeof profileValue === 'object' && typeof profileValue.key === 'string') return profileValue.key;
    return '';
}

function applyModelRuntime(data = {}) {
    availableModelProfiles = Array.isArray(data.available_profiles) ? data.available_profiles : [];
    selectedModelProfileKey = extractProfileKey(data.selected_profile) || extractProfileKey(data.selected_profile_key) || selectedModelProfileKey;
    activeModelProfileKey = extractProfileKey(data.active_profile) || extractProfileKey(data.active_profile_key) || activeModelProfileKey;
    if (!selectedModelProfileKey && Array.isArray(availableModelProfiles)) {
        selectedModelProfileKey = availableModelProfiles.find(profile => profile.selected)?.key || '';
    }
    if (!activeModelProfileKey && Array.isArray(availableModelProfiles)) {
        activeModelProfileKey = availableModelProfiles.find(profile => profile.active)?.key || selectedModelProfileKey;
    }
    setModelNote(data.selected_model_name || data.model_name || data.loaded_model_name || window.MODEL_NAME);
    syncModelSelector();
    maybeReportUiModelReady();
}

async function loadComposerRuntime() {
    try {
        const resp = await fetch('/api/dashboard');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        applyModelRuntime(data);
    } catch (error) {
        syncModelSelector();
    }
}

async function handleModelSelectChange(profileKey) {
    const previousKey = selectedModelProfileKey || activeModelProfileKey;
    if (!profileKey || profileKey === previousKey || modelSwitchInFlight) {
        syncModelSelector();
        return;
    }

    const profile = availableModelProfiles.find(item => item.key === profileKey);
    const confirmed = confirm(
        `Switch to ${profile?.label || profileKey}?\n\nThis will restart the model and the chat may be unavailable for a few minutes while it loads.`
    );
    if (!confirmed) {
        syncModelSelector();
        return;
    }

    modelSwitchInFlight = true;
    selectedModelProfileKey = profileKey;
    syncModelSelector();

    try {
        const resp = await fetch('/api/model/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile: profileKey }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.message || 'Switch failed');
        setModelNote(data.model_name || profile?.name || window.MODEL_NAME);
        updateStatus('loading');
        startHealthPolling();
        await loadComposerRuntime();
    } catch (error) {
        selectedModelProfileKey = previousKey;
        syncModelSelector();
        alert(`Model switch failed: ${error.message}`);
    } finally {
        modelSwitchInFlight = false;
        syncModelSelector();
    }
}

function loadFeatureSettings() {
    return {
        agent_tools: true,
        workspace_panel: true,
        local_rag: localStorage.getItem('feature.local_rag') !== 'false',
        web_search: localStorage.getItem('feature.web_search') === 'true',
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

function updateStatus(status) {
    const dot = document.getElementById('statusDot');
    if (!dot) return;

    if (status === 'connected') {
        dot.className = 'status-dot connected';
        modelAvailable = true;
        syncSendButton();
        maybeReportUiModelReady();
    } else if (status === 'loading') {
        dot.className = 'status-dot loading';
        modelAvailable = false;
        syncSendButton();
    } else {
        dot.className = 'status-dot';
        modelAvailable = false;
        syncSendButton();
    }
}

function setLoadingText(text) {
    const loadingText = document.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = text;
}

function isComposerAvailable() {
    const input = document.getElementById('input');
    return Boolean(modelAvailable && input && !input.disabled && ws && ws.readyState === WebSocket.OPEN);
}

async function maybeReportUiModelReady() {
    const profile = selectedModelProfileKey || activeModelProfileKey || '';
    const modelName = window.MODEL_NAME || '';
    if (!isComposerAvailable() || !profile || !modelName || uiReadyReportInFlight) return;

    const readyKey = `${uiSessionId}:${profile}:${modelName}`;
    if (readyKey === lastReportedUiReadyKey) return;

    uiReadyReportInFlight = true;
    try {
        const resp = await fetch('/api/ui/model-ready', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                profile,
                composer_available: true,
                websocket_connected: Boolean(ws && ws.readyState === WebSocket.OPEN),
            }),
        });
        if (resp.ok) {
            lastReportedUiReadyKey = readyKey;
        }
    } catch (error) {
        console.debug('Failed to report UI model readiness:', error);
    } finally {
        uiReadyReportInFlight = false;
    }
}

function applyWorkspacePanelState() {
    const panel = document.getElementById('workspacePanel');
    const toggle = document.getElementById('workspaceToggle');
    const shell = document.querySelector('.chat-shell');
    const root = document.getElementById('chatRoot');
    if (!panel || !toggle) return;
    const agentToolsEnabled = featureSettings.agent_tools;
    const workspacePanelEnabled = featureSettings.workspace_panel;
    const mobileViewport = window.matchMedia(MOBILE_WORKSPACE_MEDIA_QUERY).matches;
    const workspaceAllowed = agentToolsEnabled && workspacePanelEnabled && !mobileViewport;
    if (!workspaceAllowed) {
        workspacePanelOpen = false;
    }
    toggle.style.display = workspaceAllowed ? '' : 'none';
    const open = workspaceAllowed && workspacePanelOpen;
    panel.classList.toggle('is-open', open);
    toggle.classList.toggle('active', open);
    if (shell) shell.classList.toggle('ide-mode', open);
    if (shell) shell.classList.toggle('workspace-open', open);
    if (root) {
        root.classList.toggle('mobile-chat-mode', mobileViewport);
    }
    localStorage.setItem('workspacePanelOpen', open ? 'true' : 'false');
}

function toggleWorkspacePanel() {
    if (!featureSettings.agent_tools || !featureSettings.workspace_panel || window.matchMedia(MOBILE_WORKSPACE_MEDIA_QUERY).matches) return;
    workspacePanelOpen = !workspacePanelOpen;
    applyWorkspacePanelState();
}

function syncFeatureControls() {
    const agentTools = document.getElementById('settingAgentTools');
    const workspacePanel = document.getElementById('settingWorkspacePanel');
    const localRag = document.getElementById('settingLocalRag');
    const webSearch = document.getElementById('settingWebSearch');
    if (agentTools) agentTools.checked = featureSettings.agent_tools;
    if (workspacePanel) {
        workspacePanel.checked = featureSettings.workspace_panel;
        workspacePanel.disabled = !featureSettings.agent_tools;
    }
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
    if (!featureSettings.agent_tools || !featureSettings.workspace_panel) {
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
    if (name === 'agent_tools' && !enabled) {
        featureSettings.workspace_panel = false;
    }
    persistFeatureSettings();
    applyFeatureSettingsToUI();
    if (name === 'agent_tools' || name === 'workspace_panel') refreshWorkspace(true);
}

function downloadWorkspaceZip() {
    if (!currentConvId) return;
    window.open(`/api/workspace/${encodeURIComponent(currentConvId)}/download`, '_blank');
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

function messageRequestsPlanExecution(message) {
    const text = String(message || '').trim().toLowerCase().replace(/\s+/g, ' ');
    return PLAN_EXECUTION_MARKERS.some(marker => text.includes(marker));
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
}

function clearAllowedCommandsForConversation(conversationId) {
    const key = String(conversationId || '').trim();
    if (!key) return;
    delete chatCommandAllowlists[key];
}

function resolveTurnFeatures(message, attachmentPaths = [], slashCommand = null) {
    const permissions = inferTurnPermissions(message, attachmentPaths);
    const executionRequest = messageRequestsPlanExecution(message);
    const allowedCommands = getAllowedCommandsForConversation(currentConvId);
    const slashName = slashCommand?.name || '';
    const slashWantsWrite = slashName === 'code';
    const slashWantsRun = slashName === 'code';
    let workspaceWrite = false;
    const needsWriteApproval = permissions.wantsWrite || slashWantsWrite || executionRequest;
    let workspaceRunCommands = permissions.wantsRun || permissions.wantsWrite || slashWantsRun || executionRequest || allowedCommands.length > 0;

    if (needsWriteApproval && (!deepMode || executionRequest)) {
        workspaceWrite = window.confirm('Allow the assistant to create or edit files in the workspace for this request?');
    }

    return {
        ...featureSettings,
        agent_tools: true,
        workspace_panel: true,
        web_search: featureSettings.web_search || slashName === 'search',
        workspace_write: workspaceWrite,
        workspace_run_commands: workspaceRunCommands,
        allowed_commands: allowedCommands,
    };
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
    return Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder && voiceRuntime.stt_available);
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
        micButton.disabled = !supportsDictation();
        micButton.classList.toggle('is-active', dictationActive);
        micButton.innerHTML = dictationActive ? MIC_ACTIVE_BUTTON_ICON : MIC_BUTTON_ICON;
        micButton.setAttribute('aria-label', dictationActive ? 'Stop dictation' : 'Start dictation');
        micButton.title = dictationActive ? 'Stop dictation' : (supportsDictation() ? 'Record audio for server transcription' : 'Server dictation is unavailable');
    }

    const autoSpeak = document.getElementById('settingAutoSpeak');
    if (autoSpeak) autoSpeak.disabled = !supportsSpeechSynthesis();
    const speechSpeed = document.getElementById('settingSpeechSpeed');
    if (speechSpeed) speechSpeed.disabled = !supportsSpeechSynthesis();

    if (!supportsDictation() && !supportsSpeechSynthesis()) {
        updateVoiceNote('Server voice is unavailable. Install native STT/TTS tools on the server, then reload.');
    } else if (!supportsDictation()) {
        updateVoiceNote('Server transcription is unavailable. Reply playback is still available.');
    } else if (!supportsSpeechSynthesis()) {
        updateVoiceNote('Server speech playback is unavailable. Dictation is still available.');
    } else {
        updateVoiceNote('');
    }
}

function appendDictationTranscript(transcript) {
    const text = String(transcript || '').trim();
    const input = document.getElementById('input');
    if (!input) return;
    const existing = input.value || '';
    const separator = existing && text && !/\s$/.test(existing) ? ' ' : '';
    input.value = text ? `${existing}${separator}${text}` : existing;
    handleInputChange(input);
}

function mediaRecorderMimeType() {
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus', 'audio/ogg'];
    return candidates.find(type => window.MediaRecorder?.isTypeSupported?.(type)) || '';
}

async function transcribeRecordedAudio(blob) {
    const formData = new FormData();
    const extension = blob.type.includes('ogg') ? 'ogg' : (blob.type.includes('mp4') ? 'mp4' : 'webm');
    formData.append('file', blob, `dictation.${extension}`);
    const resp = await fetch('/api/voice/transcribe', { method: 'POST', body: formData });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
    return data;
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
    if (!voiceRuntime.stt_available) {
        updateVoiceNote('Server transcription is unavailable right now.');
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
            mediaRecorder = null;
            mediaStream?.getTracks?.().forEach(track => track.stop());
            mediaStream = null;
            syncVoiceSupportUI();
            updateVoiceNote('Recording failed before transcription could start.');
        };
        mediaRecorder.onstop = async () => {
            dictationActive = false;
            syncVoiceSupportUI();
            const chunks = recordedChunks.slice();
            recordedChunks = [];
            mediaRecorder = null;
            mediaStream?.getTracks?.().forEach(track => track.stop());
            mediaStream = null;
            if (!chunks.length) return;
            updateVoiceNote('Uploading audio for server transcription...');
            try {
                const blob = new Blob(chunks, { type: mediaRecorder?.mimeType || mimeType || 'audio/webm' });
                const result = await transcribeRecordedAudio(blob);
                appendDictationTranscript(result.transcript || '');
                updateVoiceNote('');
            } catch (error) {
                updateVoiceNote(`Transcription failed: ${error.message}`);
            }
        };
        mediaRecorder.start();
        dictationActive = true;
        syncVoiceSupportUI();
    } catch (error) {
        dictationActive = false;
        syncVoiceSupportUI();
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
    const resp = await fetch('/api/voice/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
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
                modelSwitchInFlight = false;
                loadComposerRuntime();
                clearInterval(healthPollInterval);
                healthPollInterval = null;
            } else {
                updateStatus('loading');
            }
        } catch (e) {
            updateStatus('loading');
        }
    }, 3000);
}

// ==================== WebSocket ====================

function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

    ws.onopen = () => {
        fetch('/health').then(r => r.json()).then(health => {
            if (health.model_available) {
                updateStatus('connected');
                loadComposerRuntime();
            } else {
                updateStatus('loading');
                startHealthPolling();
            }
        }).catch(() => {
            updateStatus('loading');
            startHealthPolling();
        });
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'start') {
            removeLoading();
            ensureStreamingAssistantMessage();
            clearTerminalOutput();
            currentAssistantTurnStartedAt = Date.now();
            currentAssistantTurnArtifactPaths = new Set();
            recordWorkspaceActivity('Turn', 'Turn started.');
        } else if (data.type === 'reasoning_note') {
            recordWorkspaceActivity('Think', 'Reasoning noted.');
        } else if (data.type === 'assistant_note') {
            ensureStreamingAssistantMessage();
            replaceAssistantAnswer(data.content || '');
            recordWorkspaceActivity('Note', data.content || 'Updated the working draft.');
        } else if (data.type === 'plan_ready') {
            storeExecutionPlan(data.plan || '', data.execute_prompt || '');
            recordWorkspaceActivity('Plan', 'Execution plan ready for approval.');
        } else if (data.type === 'build_steps') {
            updateBuildSteps(data.steps || [], data.active_index ?? null, data.completed_count || 0);
        } else if (data.type === 'think_start') {
            onAssistantThinkStart();
        } else if (data.type === 'think_token') {
            onAssistantThinkToken(data.content);
        } else if (data.type === 'think_end') {
            onAssistantThinkEnd();
        } else if (data.type === 'token') {
            appendAssistantAnswerToken(data.content);
        } else if (data.type === 'status') {
            setLoadingText(data.content);
        } else if (data.type === 'activity') {
            if (data.content) setLoadingText(data.content);
            recordWorkspaceActivity(
                data.label || 'Activity',
                data.content || 'Working...',
                {
                    phase: data.phase || 'status',
                    stepLabel: data.step_label || '',
                },
            );
        } else if (data.type === 'tool_start') {
            setLoadingText(data.content || `Using ${data.name || 'tool'}...`);
            if (data.name === 'workspace.run_command') {
                const command = Array.isArray(data.arguments?.command) ? data.arguments.command.join(' ') : '';
                recordTerminalCommandStart(command, data.arguments?.cwd || '.');
            }
        } else if (data.type === 'tool_result') {
            const prefix = data.ok === false ? 'Tool failed' : 'Tool finished';
            setLoadingText(data.content ? `${prefix}: ${data.content}` : prefix);
            if (data.ok !== false) noteAssistantArtifactsFromToolResult(data);
            if (data.name === 'workspace.run_command') {
                const command = Array.isArray(data.arguments?.command) ? data.arguments.command.join(' ') : '';
                recordTerminalCommandResult(command, data.payload || {});
            }
            scheduleWorkspaceRefresh();
        } else if (data.type === 'tool_error') {
            setLoadingText(data.content || 'Tool error');
        } else if (data.type === 'command_approval_required') {
            const command = Array.isArray(data.command) ? data.command : [];
            const commandKey = String(data.command_key || command[0] || 'command').trim().toLowerCase();
            const commandPreview = command.length ? command.join(' ') : commandKey;
            const approved = window.confirm(
                `Allow '${commandKey}' in this chat?\n\nRequested command:\n${commandPreview}\n\nThis approval will be remembered for this chat only.`
            );
            if (approved) {
                rememberAllowedCommand(currentConvId, commandKey);
                recordWorkspaceActivity('Approve', `Allowed '${commandKey}' for this chat.`);
                setLoadingText(`Approved '${commandKey}'. Resuming the command...`);
            } else {
                recordWorkspaceActivity('Blocked', `Denied '${commandKey}' for this chat.`);
                setLoadingText(`Command blocked: '${commandKey}' was not approved.`);
            }
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'command_approval',
                    conversation_id: currentConvId,
                    command_key: commandKey,
                    approved,
                }));
            }
        } else if (data.type === 'final_replace') {
            replaceAssistantAnswer(data.content);
            recordWorkspaceActivity('Draft', 'Draft updated.');
        } else if (data.type === 'done') {
            const finishedConvId = activeStreamConversationId;
            isGenerating = false;
            syncSendButton();
            document.getElementById('input').focus();
            renderMarkdown();
            loadConversations();
            finalizeAssistantTurnArtifacts();
            refreshWorkspace(true);
            recordWorkspaceActivity('Done', 'Turn complete.');
            activeStreamConversationId = null;
            streamingAssistantMessage = null;
            if (finishedConvId && currentConvId === finishedConvId) {
                loadConversation(finishedConvId, { preserveActivity: true });
            }
        } else if (data.type === 'canceled') {
            removeLoading();
            isGenerating = false;
            syncSendButton();
            recordWorkspaceActivity('Canceled', data.content || 'Stopped');
        } else if (data.type === 'error') {
            removeLoading();
            isGenerating = false;
            syncSendButton();
            recordWorkspaceActivity('Error', data.content || 'The assistant hit an error.', { error: true });
            const errorMsg = streamingAssistantMessage || createMessageElement('', 'assistant');
            const contentDiv = errorMsg.querySelector('.message-content');
            if (contentDiv) {
                contentDiv.style.color = '#ef4444';
                contentDiv.textContent = data.content;
            }
            streamingAssistantMessage = errorMsg;
            appendStreamingAssistantMessageIfVisible();
            if (currentConvId === activeStreamConversationId) {
                activeStreamConversationId = null;
                streamingAssistantMessage = null;
            }
        }
    };

    ws.onerror = () => {
        isGenerating = false;
        updateStatus('disconnected');
    };
    ws.onclose = () => {
        isGenerating = false;
        updateStatus('disconnected');
        setTimeout(connectWS, 2000);
    };
}

function connectLogWS() {
    if (logWs && logWs.readyState === WebSocket.OPEN) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    logWs = new WebSocket(`${protocol}//${window.location.host}/ws/logs`);
    logWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
            const el = document.getElementById('dashLogs');
            if (el) {
                el.textContent += data.content;
                el.scrollTop = el.scrollHeight;
            }
        }
    };
    logWs.onerror = () => { logWs = null; };
    logWs.onclose = () => { logWs = null; };
}

// ==================== Menu ====================

function toggleMenu() {
    document.getElementById('menuOverlay').classList.toggle('show');
}
function closeMenu() {
    document.getElementById('menuOverlay').classList.remove('show');
}

// ==================== Dashboard ====================

function showDashboard() {
    document.getElementById('dashboardOverlay').classList.add('show');
    loadDashboard();
    connectLogWS();
}
function closeDashboard() {
    document.getElementById('dashboardOverlay').classList.remove('show');
    if (dashboardRefreshTimer) {
        clearTimeout(dashboardRefreshTimer);
        dashboardRefreshTimer = null;
    }
}

function scheduleDashboardRefresh(delay = 5000) {
    if (dashboardRefreshTimer) clearTimeout(dashboardRefreshTimer);
    if (!document.getElementById('dashboardOverlay').classList.contains('show')) return;
    dashboardRefreshTimer = setTimeout(() => {
        dashboardRefreshTimer = null;
        loadDashboard();
    }, delay);
}

function formatDashboardDate(value) {
    if (!value) return '--';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
    });
}

function formatDurationSeconds(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
    const total = Math.max(0, Math.round(Number(value)));
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    if (minutes <= 0) return `${seconds}s`;
    return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
}

async function loadDashboard() {
    const el = document.getElementById('dashboardContent');
    try {
        const [runtimeResp, libraryResp] = await Promise.all([
            fetch('/api/dashboard'),
            fetch('/api/models/library'),
        ]);
        const data = await runtimeResp.json();
        const library = await libraryResp.json();

        const statusClass = data.model_available ? 'connected' : 'loading';
        const statusText = data.model_available ? 'Connected' : 'Loading / Unavailable';
        const containerStatus = data.container ? (data.container.running ? 'Running' : data.container.status || 'Stopped') : 'Unknown';
        const cache = data.cache || {};
        const loading = data.loading || {};
        const cacheStatus = cache.status || 'unknown';
        const cacheSize = cache.size_display || '--';
        const cacheDate = formatDashboardDate(cache.last_modified);
        const profiles = Array.isArray(data.available_profiles) ? data.available_profiles : [];
        const jobs = Array.isArray(library.jobs) ? library.jobs : [];
        const models = Array.isArray(library.models) ? library.models : [];
        const dockerControlAvailable = Boolean(data.docker_control_available);
        const switchButtons = dockerControlAvailable
            ? profiles.filter(profile => profile.key !== 'custom').map(profile => `
                <button class="dash-btn${profile.active ? ' active' : ''}" onclick='dashSwitchModel(${JSON.stringify(profile.key)}, ${JSON.stringify(profile.label)}, ${JSON.stringify(profile.name)})'>
                    Switch to ${profile.label}
                    <div class="btn-desc">${profile.active ? 'Currently active' : profile.name}</div>
                </button>
            `).join('')
            : '';
        const actionBody = dockerControlAvailable
            ? `
                <button class="dash-btn" onclick="dashRestart()">
                    Restart vLLM
                    <div class="btn-desc">Restart the inference server (keeps cached model)</div>
                </button>
                ${switchButtons}
                <button class="dash-btn danger" onclick="dashRedownload()">
                    Redownload Model
                    <div class="btn-desc">Delete cache and re-download from HuggingFace</div>
                </button>
            `
            : `
                <div class="dash-card">
                    <div class="dash-row"><span class="label">Runtime control</span><span class="value">Disabled</span></div>
                    <div style="color:var(--text_secondary);font-size:13px;margin-top:10px;">
                        Docker-backed restart, profile switch, and re-download actions are disabled unless the server enables Docker control explicitly.
                    </div>
                </div>
            `;
        const modelCards = models.length
            ? models.map(model => `
                <div class="model-library-card">
                    <div class="model-library-head">
                        <div>
                            <div class="model-library-title">${escapeHtml(model.model_id || 'Unknown model')}</div>
                            <div class="model-library-meta">
                                ${escapeHtml(model.size_display || '--')} • ${escapeHtml(model.status || 'unknown')} • Updated ${escapeHtml(formatDashboardDate(model.last_modified))}
                            </div>
                        </div>
                        <div class="model-library-badges">
                            ${model.active_profile ? '<span class="model-badge active">Active</span>' : ''}
                            ${model.managed_by_profile ? '<span class="model-badge">Profile</span>' : ''}
                            ${model.download_job && ['queued', 'downloading'].includes(model.download_job.status) ? '<span class="model-badge">Downloading</span>' : ''}
                        </div>
                    </div>
                    <div class="model-library-actions">
                        <button class="dash-btn dash-btn-compact" onclick='window.open(${JSON.stringify(model.download_url)}, "_blank", "noopener")'>View</button>
                        <button class="dash-btn dash-btn-compact" onclick='dashActivateModel(${JSON.stringify(model.model_id)})' ${dockerControlAvailable ? '' : 'disabled'}>${model.active_profile ? 'Active' : 'Use Model'}</button>
                        <button class="dash-btn dash-btn-compact danger" onclick='dashDeleteModel(${JSON.stringify(model.model_id)})' ${model.managed_by_profile ? 'disabled' : ''}>Delete</button>
                    </div>
                </div>
            `).join('')
            : '<div class="dash-empty">No Hugging Face model caches found yet.</div>';
        const jobCards = jobs.length
            ? jobs.map(job => `
                <div class="model-job ${job.status === 'error' ? 'is-error' : ''}">
                    <div class="model-job-head">
                        <span class="model-job-id">${escapeHtml(job.model_id || 'Unknown model')}</span>
                        <span class="model-job-status">${escapeHtml(job.status || 'queued')}</span>
                    </div>
                    <div class="model-job-meta">Started ${escapeHtml(formatDashboardDate(job.started_at || job.created_at))}</div>
                    ${job.error ? `<div class="model-job-error">${escapeHtml(job.error)}</div>` : ''}
                </div>
            `).join('')
            : '<div class="dash-empty">No active or recent downloads yet.</div>';

        el.innerHTML = `
            <div class="dash-section">
                <h4>Model</h4>
                <div class="dash-card">
                    <div class="dash-row"><span class="label">Name</span><span class="value">${data.model_name}</span></div>
                    <div class="dash-row"><span class="label">Selected</span><span class="value">${data.selected_model_name || data.model_name}</span></div>
                    <div class="dash-row"><span class="label">Status</span><span class="value"><span class="status-dot ${statusClass}"></span>${statusText}</span></div>
                    <div class="dash-row"><span class="label">Container</span><span class="value">${containerStatus}</span></div>
                    <div class="dash-row"><span class="label">Load Phase</span><span class="value">${escapeHtml(loading.phase || (data.model_available ? 'ready' : 'loading'))}</span></div>
                    <div class="dash-row"><span class="label">Progress</span><span class="value">${loading.progress !== null && loading.progress !== undefined ? `${Math.round(Number(loading.progress) * 100)}%` : '--'}</span></div>
                    <div class="dash-row"><span class="label">Elapsed</span><span class="value">${escapeHtml(formatDurationSeconds(loading.elapsed_seconds))}</span></div>
                    <div class="dash-row"><span class="label">ETA</span><span class="value">${escapeHtml(formatDurationSeconds(loading.eta_seconds))}</span></div>
                    <div class="dash-row"><span class="label">Avg Load</span><span class="value">${escapeHtml(formatDurationSeconds(loading.history?.average_seconds))}</span></div>
                    <div class="dash-row"><span class="label">Last Load</span><span class="value">${escapeHtml(formatDurationSeconds(loading.history?.last_seconds))}</span></div>
                    <div class="dash-row"><span class="label">History</span><span class="value">${escapeHtml(String(loading.history?.sample_count || 0))} samples</span></div>
                </div>
            </div>
            <div class="dash-section">
                <h4>Cache</h4>
                <div class="dash-card">
                    <div class="dash-row"><span class="label">Status</span><span class="value">${cacheStatus}</span></div>
                    <div class="dash-row"><span class="label">Size</span><span class="value">${cacheSize}</span></div>
                    <div class="dash-row"><span class="label">Last Updated</span><span class="value">${cacheDate}</span></div>
                </div>
            </div>
            <div class="dash-section">
                <h4>Actions</h4>
                <div class="dash-actions">
                    <button class="dash-btn" onclick="dashValidate()">
                        Validate Cache
                        <div class="btn-desc">Check model files are present and complete</div>
                    </button>
                    ${actionBody}
                </div>
            </div>
            <div class="dash-section">
                <h4>Model Library</h4>
                <div class="dash-card model-library-shell">
                    <div class="model-library-intro">Paste a Hugging Face model URL or repo id to download it into the shared cache used by this app.</div>
                    <div class="model-library-form">
                        <input id="modelLibraryInput" class="dash-input" placeholder="e.g. Qwen/Qwen3-8B or https://huggingface.co/Qwen/Qwen3-8B">
                        <button class="dash-btn" onclick="dashDownloadModel()">Download Model</button>
                    </div>
                    <div class="model-library-subhead">
                        <span>Cached Models</span>
                        <button class="dash-link-btn" onclick="loadDashboard()">Refresh</button>
                    </div>
                    <div class="model-library-list">${modelCards}</div>
                    <div class="model-library-subhead">
                        <span>Downloads</span>
                    </div>
                    <div class="model-jobs">${jobCards}</div>
                </div>
            </div>
            <div class="dash-section">
                <h4>Logs</h4>
                <div class="dash-logs" id="dashLogs"></div>
            </div>
        `;
        if (jobs.some(job => ['queued', 'downloading'].includes(job.status))) {
            scheduleDashboardRefresh(3000);
        } else {
            scheduleDashboardRefresh(12000);
        }
    } catch (e) {
        el.innerHTML = `<div style="padding:20px;color:var(--text_secondary)">Failed to load dashboard: ${e.message}</div>`;
        scheduleDashboardRefresh(12000);
    }
}

async function dashValidate() {
    const btn = event.target.closest('.dash-btn');
    btn.disabled = true;
    btn.textContent = 'Validating...';
    try {
        const resp = await fetch('/api/dashboard');
        const data = await resp.json();
        const cache = data.cache || {};
        if (cache.status === 'valid') {
            btn.textContent = 'Cache is valid';
            btn.style.borderColor = 'var(--status_connected)';
        } else if (cache.status === 'incomplete') {
            btn.textContent = 'Cache incomplete — try redownloading';
            btn.style.borderColor = '#f59e0b';
        } else {
            btn.textContent = 'No cache found — try redownloading';
            btn.style.borderColor = 'var(--status_disconnected)';
        }
    } catch (e) {
        btn.textContent = 'Validation failed: ' + e.message;
    }
    setTimeout(() => { btn.disabled = false; loadDashboard(); }, 3000);
}

async function dashRestart() {
    if (!confirm('Restart vLLM? The model will reload (takes a few minutes).')) return;
    const btn = event.target.closest('.dash-btn');
    btn.disabled = true;
    btn.textContent = 'Restarting...';
    try {
        const resp = await fetch('/api/vllm/restart', { method: 'POST' });
        const data = await resp.json();
        btn.textContent = data.message || 'Restarting...';
        updateStatus('loading');
        startHealthPolling();
    } catch (e) {
        btn.textContent = 'Restart failed: ' + e.message;
    }
    setTimeout(() => { btn.disabled = false; }, 5000);
}

async function dashSwitchModel(profileKey, profileLabel, modelName) {
    const confirmed = confirm(`Switch to ${profileLabel}?\n\nThis will hard-restart vLLM and load ${modelName}. The chat will be unavailable for a few minutes while the model loads.`);
    if (!confirmed) return;

    const btn = event.target.closest('.dash-btn');
    btn.disabled = true;
    btn.textContent = `Switching to ${profileLabel}...`;
    try {
        const resp = await fetch('/api/model/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile: profileKey }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.message || 'Switch failed');
        btn.textContent = data.message || `Switching to ${profileLabel}...`;
        setModelNote(data.model_name || modelName);
        updateStatus('loading');
        startHealthPolling();
        loadDashboard();
    } catch (e) {
        btn.textContent = 'Switch failed: ' + e.message;
    }
    setTimeout(() => { btn.disabled = false; loadDashboard(); }, 3000);
}

async function dashRedownload() {
    if (!confirm('This will delete the cached model and re-download it from HuggingFace. This may take a while. Continue?')) return;
    const btn = event.target.closest('.dash-btn');
    btn.disabled = true;
    btn.textContent = 'Clearing cache and restarting...';
    try {
        const resp = await fetch('/api/model/redownload', { method: 'POST' });
        const data = await resp.json();
        btn.textContent = data.message || 'Redownloading...';
        updateStatus('loading');
        startHealthPolling();
    } catch (e) {
        btn.textContent = 'Failed: ' + e.message;
    }
    setTimeout(() => { btn.disabled = false; }, 10000);
}

async function dashDownloadModel() {
    const input = document.getElementById('modelLibraryInput');
    const source = (input?.value || '').trim();
    if (!source) {
        alert('Enter a Hugging Face model URL or repo id first.');
        return;
    }
    const btn = event.target.closest('.dash-btn');
    btn.disabled = true;
    btn.textContent = 'Starting download...';
    try {
        const resp = await fetch('/api/models/library/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.message || 'Download failed');
        input.value = '';
        btn.textContent = data.message || 'Download started';
        loadDashboard();
    } catch (e) {
        btn.textContent = 'Failed: ' + e.message;
    }
    setTimeout(() => {
        btn.disabled = false;
        btn.textContent = 'Download Model';
    }, 3000);
}

async function dashDeleteModel(modelId) {
    const confirmed = confirm(`Delete the cached files for ${modelId}?\n\nThis only removes the local Hugging Face cache for that model.`);
    if (!confirmed) return;

    const btn = event.target.closest('.dash-btn');
    btn.disabled = true;
    btn.textContent = 'Deleting...';
    try {
        const resp = await fetch('/api/models/library/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.message || 'Delete failed');
        loadDashboard();
    } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Failed: ' + e.message;
    }
}

async function dashActivateModel(modelId) {
    const confirmed = confirm(`Switch to ${modelId}?\n\nThis will restart vLLM and load that cached model.`);
    if (!confirmed) return;

    const btn = event.target.closest('.dash-btn');
    btn.disabled = true;
    btn.textContent = 'Activating...';
    try {
        const resp = await fetch('/api/models/library/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.message || 'Activation failed');
        setModelNote(data.model_name || modelId);
        updateStatus('loading');
        startHealthPolling();
        loadDashboard();
    } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Failed: ' + e.message;
    }
}

// ==================== Settings ====================

function showSettings() {
    syncFeatureControls();
    showSettingsHome();
    document.getElementById('settingsOverlay').classList.add('show');
}
function closeSettings() {
    document.getElementById('settingsOverlay').classList.remove('show');
}

function showSettingsHome() {
    const main = document.getElementById('settingsMainView');
    const about = document.getElementById('settingsAboutView');
    const title = document.getElementById('settingsTitle');
    if (main) main.classList.remove('settings-view-hidden');
    if (about) about.classList.add('settings-view-hidden');
    if (title) title.textContent = 'Settings';
}

function showAboutPage() {
    const main = document.getElementById('settingsMainView');
    const about = document.getElementById('settingsAboutView');
    const title = document.getElementById('settingsTitle');
    if (main) main.classList.add('settings-view-hidden');
    if (about) about.classList.remove('settings-view-hidden');
    if (title) title.textContent = 'About';
}

function setResetButtonState(isBusy) {
    const button = document.getElementById('resetAppButton');
    if (!button) return;
    button.disabled = isBusy;
    button.innerHTML = isBusy
        ? '<span class="settings-icon">&#8635;</span><span>Resetting...</span>'
        : '<span class="settings-icon">&#128465;</span><span>Delete All Data</span>';
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
        'Delete all chats, workspaces, artifacts, pet memory, and saved browser preferences?\n\nThis also removes packages and files installed inside conversation workspaces.'
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

async function loadPet() {
    try {
        const resp = await fetch('/api/pet');
        const data = await resp.json();
        petExists = Boolean(data.exists);
        petProfile = data.pet || null;
        applyPetTheme(petProfile);
    } catch (e) {
        console.error('Failed to load agent profile:', e);
    }
}

async function promoteWorkspaceFile(path) {
    if (!petExists || !currentRunId || !path) return;
    const name = prompt('Capability name:', basename(path));
    if (!name) return;
    const description = prompt('Capability description:', `Promoted from ${path}`) || '';
    try {
        const resp = await fetch('/api/pet/capabilities/promote', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                run_id: currentRunId,
                source_path: path,
                name,
                kind: 'artifact',
                description,
            }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
        await loadPet();
    } catch (e) {
        alert(`Failed to promote artifact: ${e.message}`);
    }
}

function buildWelcomeMarkup() {
    return `<div class="welcome">
            <div class="welcome-brand" aria-label="${escapeHtml(window.APP_TITLE || '')}">
                <button class="welcome-mascot" type="button" aria-label="Wolf mascot. Click to pet." data-pose="sit">
                    <svg class="welcome-logo welcome-logo-dog welcome-logo-dog-pixel" viewBox="0 0 192 160" role="img" aria-hidden="true">
                        <g class="dog-drawing" shape-rendering="crispEdges">
                            <g class="dog-bark-lines" aria-hidden="true">
                                <rect class="px-bark" x="156" y="32" width="4" height="12"/>
                                <rect class="px-bark" x="164" y="40" width="4" height="12"/>
                                <rect class="px-bark" x="156" y="52" width="4" height="8"/>
                            </g>
                            <path class="dog-shadow px-shadow" d="M56 132h72v4H56zM48 136h88v4H48zM56 140h72v4H56z"/>
                            <g class="dog-tail">
                                <rect class="px-outline" x="36" y="76" width="8" height="12"/>
                                <rect class="px-outline" x="44" y="68" width="12" height="24"/>
                                <rect class="px-outline" x="56" y="64" width="12" height="28"/>
                                <rect class="px-outline" x="68" y="68" width="8" height="20"/>
                                <rect class="px-outline" x="72" y="80" width="8" height="16"/>
                                <rect class="px-outline" x="64" y="92" width="8" height="12"/>
                                <rect class="px-outline" x="52" y="96" width="12" height="8"/>
                                <rect class="px-fur-mid" x="40" y="80" width="8" height="8"/>
                                <rect class="px-fur-light" x="48" y="72" width="8" height="16"/>
                                <rect class="px-fur-light" x="56" y="68" width="8" height="20"/>
                                <rect class="px-fur-mid" x="64" y="72" width="8" height="16"/>
                                <rect class="px-cream" x="60" y="64" width="8" height="8"/>
                                <rect class="px-cream" x="52" y="92" width="8" height="8"/>
                            </g>
                            <g class="dog-body-group">
                                <rect class="px-outline" x="64" y="76" width="48" height="8"/>
                                <rect class="px-outline" x="56" y="84" width="64" height="12"/>
                                <rect class="px-outline" x="48" y="96" width="72" height="16"/>
                                <rect class="px-outline" x="52" y="112" width="68" height="8"/>
                                <rect class="px-outline" x="60" y="120" width="52" height="8"/>
                                <rect class="px-outline" x="72" y="128" width="28" height="4"/>
                                <rect class="px-outline" x="108" y="80" width="12" height="32"/>
                                <rect class="px-fur-dark" x="64" y="80" width="48" height="8"/>
                                <rect class="px-fur-mid" x="56" y="88" width="60" height="16"/>
                                <rect class="px-fur-mid" x="60" y="104" width="56" height="8"/>
                                <rect class="px-fur-dark" x="64" y="112" width="52" height="8"/>
                                <rect class="px-fur-light" x="68" y="120" width="36" height="4"/>
                                <rect class="px-cream" x="76" y="88" width="24" height="8"/>
                                <rect class="px-cream" x="72" y="96" width="32" height="16"/>
                                <rect class="px-cream" x="76" y="112" width="28" height="8"/>
                                <rect class="px-cream" x="92" y="84" width="12" height="8"/>
                            </g>
                            <g class="dog-leg-group dog-leg-back-group">
                                <rect class="px-outline" x="60" y="108" width="12" height="24"/>
                                <rect class="px-outline" x="76" y="112" width="12" height="20"/>
                                <rect class="px-fur-mid" x="64" y="112" width="4" height="16"/>
                                <rect class="px-fur-mid" x="80" y="116" width="4" height="12"/>
                                <rect class="px-cream" x="56" y="128" width="20" height="8"/>
                                <rect class="px-cream" x="72" y="128" width="20" height="8"/>
                            </g>
                            <g class="dog-head-group">
                                <g class="dog-ear-left">
                                    <rect class="px-outline" x="92" y="16" width="12" height="12"/>
                                    <rect class="px-outline" x="96" y="12" width="8" height="8"/>
                                    <rect class="px-ear" x="96" y="20" width="4" height="8"/>
                                    <rect class="px-ear" x="100" y="24" width="4" height="4"/>
                                </g>
                                <g class="dog-ear-right">
                                    <rect class="px-outline" x="120" y="12" width="8" height="8"/>
                                    <rect class="px-outline" x="124" y="16" width="12" height="16"/>
                                    <rect class="px-ear" x="124" y="20" width="4" height="8"/>
                                    <rect class="px-ear" x="120" y="24" width="4" height="4"/>
                                </g>
                                <rect class="px-outline" x="92" y="28" width="32" height="8"/>
                                <rect class="px-outline" x="84" y="36" width="52" height="12"/>
                                <rect class="px-outline" x="80" y="48" width="60" height="16"/>
                                <rect class="px-outline" x="84" y="64" width="56" height="12"/>
                                <rect class="px-outline" x="92" y="76" width="40" height="8"/>
                                <rect class="px-outline" x="100" y="84" width="20" height="4"/>
                                <rect class="px-fur-dark" x="92" y="32" width="32" height="4"/>
                                <rect class="px-fur-mid" x="88" y="40" width="44" height="20"/>
                                <rect class="px-fur-light" x="100" y="32" width="20" height="8"/>
                                <rect class="px-fur-light" x="96" y="40" width="28" height="8"/>
                                <rect class="px-fur-mid" x="92" y="60" width="40" height="12"/>
                                <rect class="px-cream" x="104" y="56" width="24" height="8"/>
                                <rect class="px-cream" x="100" y="64" width="28" height="12"/>
                                <rect class="px-cream" x="108" y="76" width="12" height="4"/>
                                <rect class="px-blush" x="100" y="68" width="4" height="4"/>
                                <rect class="px-blush" x="120" y="64" width="4" height="4"/>
                                <g class="dog-eyes-open">
                                    <rect class="px-eye" x="104" y="48" width="8" height="8"/>
                                    <rect class="px-eye" x="120" y="44" width="8" height="8"/>
                                    <rect class="px-eye-shine" x="108" y="48" width="4" height="4"/>
                                    <rect class="px-eye-shine" x="124" y="44" width="4" height="4"/>
                                </g>
                                <g class="dog-eye-blink">
                                    <rect class="px-eye" x="104" y="52" width="8" height="4"/>
                                    <rect class="px-eye" x="120" y="48" width="8" height="4"/>
                                </g>
                                <rect class="px-nose" x="128" y="60" width="8" height="4"/>
                                <rect class="px-nose" x="124" y="64" width="8" height="4"/>
                                <g class="dog-mouth-neutral">
                                    <rect class="px-mouth" x="124" y="68" width="8" height="4"/>
                                    <rect class="px-mouth" x="128" y="72" width="4" height="4"/>
                                </g>
                                <g class="dog-mouth-bark">
                                    <rect class="px-nose" x="124" y="60" width="8" height="4"/>
                                    <rect class="px-mouth" x="120" y="64" width="16" height="12"/>
                                    <rect class="px-tongue" x="124" y="72" width="8" height="4"/>
                                    <rect class="px-cream" x="116" y="76" width="16" height="4"/>
                                </g>
                            </g>
                            <g class="dog-leg-group dog-leg-front-group">
                                <rect class="px-outline" x="96" y="104" width="12" height="28"/>
                                <rect class="px-outline" x="112" y="100" width="12" height="32"/>
                                <rect class="px-fur-light" x="100" y="108" width="4" height="20"/>
                                <rect class="px-fur-light" x="116" y="104" width="4" height="24"/>
                                <rect class="px-cream" x="92" y="128" width="20" height="8"/>
                                <rect class="px-cream" x="108" y="128" width="20" height="8"/>
                            </g>
                        </g>
                    </svg>
                    <span class="bond-reaction" id="bondReaction" hidden></span>
                </button>
                <h1 class="welcome-title">${escapeHtml(window.APP_TITLE || '')}</h1>
                <span class="bond-note" id="bondNote"></span>
            </div>
        </div>`;
}

function destroyWelcomeMascot() {
    if (!welcomeMascotRuntime) return;
    window.clearInterval(welcomeMascotRuntime.cycleTimer);
    window.clearTimeout(welcomeMascotRuntime.poseTimer);
    welcomeMascotRuntime = null;
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

let currentBond = { affection: 50, pets_today: 0, streak: 0, mood: 'content', capped: false, max_pets_per_day: 12 };

function updateBondNote() {
    const el = document.getElementById('bondNote');
    if (!el) return;
    const b = currentBond;
    if (b.mood === 'happy') el.textContent = `Streak ${b.streak}d`;
    else if (b.mood === 'content') el.textContent = b.streak > 0 ? `Streak ${b.streak}d` : '';
    else if (b.mood === 'lonely') el.textContent = 'Lonely...';
    else el.textContent = 'Neglected';
}

function getMoodCycle(mood) {
    switch (mood) {
        case 'happy': return { poses: ['walk', 'sit', 'bark', 'sit', 'walk', 'sit'], interval: 2800, durations: { walk: 1200, bark: 800 } };
        case 'content': return { poses: ['sit', 'sit', 'walk', 'sit'], interval: 5000, durations: { walk: 1400 } };
        case 'lonely': return { poses: ['rest', 'rest', 'sit', 'rest'], interval: 6000, durations: { sit: 2000 } };
        case 'neglected': return { poses: ['dig', 'rest', 'dig', 'rest', 'rest'], interval: 4500, durations: { dig: 1800 } };
        default: return { poses: ['sit'], interval: 5000, durations: {} };
    }
}

async function loadBondState() {
    try {
        const resp = await fetch('/api/pet/bond');
        if (resp.ok) currentBond = await resp.json();
    } catch (e) { /* offline */ }
    updateBondNote();
}

function showBondReaction(text) {
    const el = document.getElementById('bondReaction');
    if (!el) return;
    el.textContent = text;
    el.hidden = false;
    el.classList.remove('pop');
    void el.offsetWidth;
    el.classList.add('pop');
    setTimeout(() => { el.hidden = true; }, 900);
}

function initWelcomeMascot() {
    destroyWelcomeMascot();

    const mascot = document.querySelector('.welcome-mascot');
    if (!mascot) return;

    const runtime = { cycleTimer: null, poseTimer: null, cycleIndex: 0, petting: false };

    const setPose = (pose) => { mascot.dataset.pose = pose; };

    const queuePose = (pose, duration, nextPose = 'sit') => {
        window.clearTimeout(runtime.poseTimer);
        setPose(pose);
        if (duration) {
            runtime.poseTimer = window.setTimeout(() => setPose(nextPose), duration);
        }
    };

    function startCycle() {
        window.clearInterval(runtime.cycleTimer);
        const mc = getMoodCycle(currentBond.mood);
        runtime.cycleIndex = 0;
        runtime.cycleTimer = window.setInterval(() => {
            if (runtime.petting) return;
            const pose = mc.poses[runtime.cycleIndex % mc.poses.length];
            runtime.cycleIndex++;
            const dur = mc.durations[pose] || mc.interval * 0.6;
            queuePose(pose, dur, currentBond.mood === 'lonely' || currentBond.mood === 'neglected' ? 'rest' : 'sit');
        }, mc.interval);
    }

    mascot.addEventListener('click', async () => {
        runtime.petting = true;
        if (currentBond.capped) {
            queuePose('bark', 800, 'sit');
            setTimeout(() => { runtime.petting = false; }, 1000);
            return;
        }
        queuePose('pet', 600, 'bark');
        try {
            const resp = await fetch('/api/pet/bond/pet', { method: 'POST' });
            if (resp.ok) {
                const prev = currentBond.mood;
                currentBond = await resp.json();
                updateBondNote();
                showBondReaction('+');
                if (currentBond.mood !== prev) startCycle();
            }
        } catch (e) { /* offline */ }
        setTimeout(() => {
            runtime.petting = false;
            queuePose('sit', 0);
        }, 1200);
    });

    loadBondState().then(() => {
        const defaultPose = currentBond.mood === 'lonely' || currentBond.mood === 'neglected' ? 'rest' : 'sit';
        setPose(defaultPose);
        startCycle();
    });

    welcomeMascotRuntime = runtime;
}

// ==================== Chat ====================

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.max(textarea.scrollHeight, 72) + 'px';
}

function exitWelcomeMode() {
    destroyWelcomeMascot();
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
    if (input) input.style.height = '';
    initWelcomeMascot();
    startWelcomeHintRotation();
}

function toggleDeepMode() {
    deepMode = !deepMode;
    localStorage.setItem('deepMode', deepMode);
    syncReasoningSelector();
}

function focusInput() {
    const input = document.getElementById('input');
    if (input) input.focus();
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
        send.disabled = false;
        const hasDraft = Boolean(input && input.value.trim());
        send.innerHTML = hasDraft ? SEND_BUTTON_ICON : STOP_BUTTON_ICON;
        send.classList.add('is-stop');
        send.setAttribute('aria-label', hasDraft ? 'Interrupt with prompt' : 'Stop response');
        send.title = hasDraft ? 'Interrupt with prompt' : 'Stop response';
        syncModelSelector();
        syncInterveneButton();
        renderExecutionPlanApproval();
        return;
    }

    send.innerHTML = SEND_BUTTON_ICON;
    send.classList.remove('is-stop');
    send.setAttribute('aria-label', 'Send message');
    send.title = 'Send message';
    send.disabled = !modelAvailable || !input || !input.value.trim();
    syncModelSelector();
    syncInterveneButton();
    renderExecutionPlanApproval();
}

function handleInputChange(textarea) {
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

function noteAssistantArtifactsFromToolResult(data) {
    const payload = data?.payload || {};
    if (!payload || typeof payload !== 'object') return;
    if (typeof payload.path === 'string' && payload.path) {
        noteAssistantTurnArtifactPath(payload.path);
    }
    if (Array.isArray(payload.items)) {
        payload.items.forEach(item => {
            if (item && typeof item.path === 'string') noteAssistantTurnArtifactPath(item.path);
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

function normalizeWorkspacePath(path) {
    const value = String(path || '.').trim();
    return value === '' || value === '.' ? '.' : value.replace(/^\.\/+/, '').replace(/\/+/g, '/');
}

function compareWorkspacePaths(a, b) {
    return normalizeWorkspacePath(a) === normalizeWorkspacePath(b);
}

function isWorkspaceEntryRecentlyChanged(entry) {
    if (!entry || entry.type !== 'file') return false;
    if (latestAssistantTurnArtifactPaths.has(entry.path)) return true;
    if (!currentAssistantTurnStartedAt) return false;
    const modified = new Date(entry.modified_at || 0).getTime() || 0;
    return modified >= currentAssistantTurnStartedAt - 1500;
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

    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'workspace-tree-row';
    row.style.setProperty('--workspace-depth', String(options.depth || 0));
    row.onclick = () => {
        if (isDirectory) {
            toggleWorkspaceDirectory(entry.path);
            return;
        }
        openWorkspaceFile(entry.path);
    };

    const caret = document.createElement('span');
    caret.className = `workspace-tree-caret${isDirectory ? '' : ' is-placeholder'}${collapsed ? ' is-collapsed' : ''}`;
    caret.innerHTML = isDirectory ? '&#9656;' : '&#8226;';
    row.appendChild(caret);

    const icon = document.createElement('span');
    icon.className = 'workspace-tree-icon';
    icon.innerHTML = isDirectory ? (collapsed ? '&#128193;' : '&#128194;') : '&#128196;';
    row.appendChild(icon);

    const copy = document.createElement('span');
    copy.className = 'workspace-tree-copy';
    copy.innerHTML = `
        <span class="workspace-tree-name">${escapeHtml(entry.name || basename(entry.path) || 'workspace')}</span>
        <span class="workspace-tree-meta">${escapeHtml(workspaceTreeNodeMeta(entry))}</span>
    `;
    row.appendChild(copy);

    if (isChanged) {
        const badge = document.createElement('span');
        badge.className = 'workspace-tree-status';
        badge.textContent = 'Live';
        row.appendChild(badge);
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
        container.innerHTML = '<div class="workspace-empty">Choose a chat action that creates or uploads files, and the workspace tree will appear here.</div>';
        return;
    }

    container.innerHTML = '';
    container.appendChild(renderWorkspaceTreeNode(workspaceTree, { depth: 0, ...options }));
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
    return !/\.(xlsx|xls|xlsm)$/i.test(path || '');
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
    const panel = document.getElementById('buildStepsPanel');
    const list = document.getElementById('buildStepsList');
    if (!panel || !list) return;
    panel.hidden = true;
    list.innerHTML = '';
}

function clearBuildSteps() {
    buildSteps = [];
    currentBuildStepIndex = null;
    renderBuildSteps();
    syncInterveneButton();
}

function updateBuildSteps(steps, activeIndex = null, completedCount = 0) {
    buildSteps = (steps || []).map((text, index) => {
        let state = 'pending';
        if (index < completedCount) state = 'complete';
        else if (activeIndex !== null && index === activeIndex) state = 'active';
        return { text, state };
    });
    currentBuildStepIndex = activeIndex !== null ? activeIndex : null;
    renderBuildSteps();
    syncInterveneButton();
}

function renderExecutionPlanApproval() {
    const panel = document.getElementById('planApprovalPanel');
    const summary = document.getElementById('planApprovalSummary');
    const runButton = document.getElementById('planApprovalRunButton');
    const editButton = document.getElementById('planApprovalEditButton');
    if (!panel || !summary) return;

    const hasPlan = Boolean(pendingExecutionPlan?.executePrompt);
    panel.hidden = !hasPlan;
    summary.textContent = hasPlan ? (pendingExecutionPlan.summary || 'Execution plan ready.') : '';
    if (runButton) {
        runButton.disabled = !hasPlan || isGenerating || !modelAvailable || !ws || ws.readyState !== WebSocket.OPEN;
    }
    if (editButton) {
        editButton.disabled = !hasPlan;
    }
}

function storeExecutionPlan(summary, executePrompt) {
    const cleanedPrompt = String(executePrompt || '').trim();
    if (!cleanedPrompt) return;
    pendingExecutionPlan = {
        summary: String(summary || '').trim(),
        executePrompt: cleanedPrompt,
    };
    renderExecutionPlanApproval();
}

function dismissExecutionPlan() {
    pendingExecutionPlan = null;
    renderExecutionPlanApproval();
}

function loadExecutionPlanDraft(executePrompt) {
    const draft = String(executePrompt || '').trim();
    const input = document.getElementById('input');
    if (!draft || !input) return;
    setInputValue(draft, draft.length);
    focusInput();
}

function editExecutionPlanDraft() {
    if (!pendingExecutionPlan?.executePrompt) return;
    loadExecutionPlanDraft(pendingExecutionPlan.executePrompt);
    recordWorkspaceActivity('Plan', 'Copied the execution draft into the composer.');
}

function approveExecutionPlan() {
    if (!pendingExecutionPlan?.executePrompt || isGenerating || !ws || ws.readyState !== WebSocket.OPEN || !modelAvailable) {
        return;
    }

    const turnFeatures = resolveTurnFeatures(APPROVED_PLAN_EXECUTION_MESSAGE, [], null);
    if (!turnFeatures.workspace_write) return;

    hideSlashMenu();
    document.querySelector('.welcome')?.remove();
    exitWelcomeMode();
    addMessage(APPROVED_PLAN_EXECUTION_MESSAGE, 'user');
    showLoading();
    clearBuildSteps();
    clearWorkspaceActivity();
    recordWorkspaceActivity('Request', APPROVED_PLAN_EXECUTION_MESSAGE);

    isGenerating = true;
    activeStreamConversationId = currentConvId;
    streamingAssistantMessage = null;
    dismissExecutionPlan();
    syncSendButton();
    ws.send(JSON.stringify({
        message: APPROVED_PLAN_EXECUTION_MESSAGE,
        attachments: [],
        conversation_id: currentConvId,
        system_prompt: localStorage.getItem('customSystemPrompt') || null,
        mode: deepMode ? 'deep' : 'normal',
        features: turnFeatures,
        slash_command: null,
    }));
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
    return `Step ${currentBuildStepIndex + 1}: ${step.text}`;
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
}

function renderTerminalOutput() {
    const outputEl = document.getElementById('terminalOutput');
    const statusEl = document.getElementById('terminalStatusLabel');
    if (!outputEl) return;
    if (terminalUnavailableReason) {
        outputEl.innerHTML = `<div class="workspace-empty terminal-empty">${escapeHtml(terminalUnavailableReason)}</div>`;
        if (statusEl) statusEl.textContent = 'Interactive workspace shell unavailable';
        return;
    }
    if (terminalEmulator) {
        const emptyEl = outputEl.querySelector('.workspace-empty');
        if (emptyEl) emptyEl.hidden = true;
        if (statusEl) {
            const connected = terminalWs && terminalWs.readyState === WebSocket.OPEN;
            statusEl.textContent = connected ? 'Interactive workspace shell connected' : 'Interactive workspace shell disconnected';
        }
        return;
    }
    if (!terminalEntries.length && !terminalLiveBuffer) {
        outputEl.innerHTML = '<div class="workspace-empty terminal-empty">Workspace command output will appear here during a turn.</div>';
        if (statusEl) statusEl.textContent = 'Interactive workspace shell';
        return;
    }

    outputEl.innerHTML = '';
    if (terminalLiveBuffer) {
        const live = document.createElement('div');
        live.className = 'terminal-live';
        live.textContent = terminalLiveBuffer;
        outputEl.appendChild(live);
    }
    terminalEntries.forEach(entry => {
        const row = document.createElement('div');
        row.className = 'terminal-entry';
        const stdout = entry.stdout ? `<div class="terminal-stream">${escapeHtml(entry.stdout)}</div>` : '';
        const stderr = entry.stderr ? `<div class="terminal-stream is-error">${escapeHtml(entry.stderr)}</div>` : '';
        row.innerHTML = `
            <div class="terminal-command">$ ${escapeHtml(entry.command || '')}</div>
            ${stdout}
            ${stderr}
            <div class="terminal-meta">${escapeHtml(entry.meta || '')}</div>
        `;
        outputEl.appendChild(row);
    });
    if (statusEl) {
        const connected = terminalWs && terminalWs.readyState === WebSocket.OPEN;
        statusEl.textContent = connected ? 'Interactive workspace shell connected' : 'Interactive workspace shell disconnected';
    }
    outputEl.scrollTop = outputEl.scrollHeight;
}

function clearTerminalOutput() {
    terminalEntries = [];
    terminalLiveBuffer = '';
    if (terminalEmulator) terminalEmulator.clear();
    renderTerminalOutput();
}

function recordTerminalCommandStart(command, cwd) {
    if (!command) return;
    const meta = [`Running in ${cwd || '.'}`];
    terminalEntries = terminalEntries.concat({
        id: Date.now() + Math.random(),
        command,
        stdout: '',
        stderr: '',
        meta: meta.join(' • '),
        pending: true,
    }).slice(-WORKSPACE_ACTIVITY_LIMIT);
    renderTerminalOutput();
}

function connectTerminalWS() {
    if (!featureSettings.agent_tools || !featureSettings.workspace_panel || !currentConvId) return;
    if (terminalUnavailableReason) return;
    if (terminalWs && (terminalWs.readyState === WebSocket.OPEN || terminalWs.readyState === WebSocket.CONNECTING)) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    terminalWs = new WebSocket(`${protocol}//${window.location.host}/ws/terminal/${encodeURIComponent(currentConvId)}`);
    terminalWs.onopen = () => {
        terminalUnavailableReason = '';
        initializeTerminalEmulator();
        if (terminalEmulator) {
            terminalEmulator.clear();
            fitTerminalEmulator();
            sendTerminalResize();
        }
        renderTerminalOutput();
    };
    terminalWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'terminal_output') {
            terminalLiveBuffer += data.content || '';
            if (terminalEmulator) terminalEmulator.write(data.content || '');
            renderTerminalOutput();
        } else if (data.type === 'terminal_status') {
            const statusEl = document.getElementById('terminalStatusLabel');
            if (statusEl) statusEl.textContent = data.content || 'Interactive workspace shell';
        } else if (data.type === 'terminal_unavailable') {
            terminalUnavailableReason = data.content || 'Interactive terminal is unavailable on this server.';
            if (terminalWs) {
                try { terminalWs.close(); } catch (e) {}
            }
            renderTerminalOutput();
        } else if (data.type === 'terminal_cleared') {
            terminalLiveBuffer = '';
            if (terminalEmulator) terminalEmulator.clear();
            renderTerminalOutput();
        }
    };
    terminalWs.onclose = () => {
        renderTerminalOutput();
        setTimeout(() => {
            if (!terminalUnavailableReason && currentConvId && featureSettings.agent_tools && featureSettings.workspace_panel) connectTerminalWS();
        }, 1500);
    };
    terminalWs.onerror = () => {
        renderTerminalOutput();
    };
}

function closeTerminalWS() {
    if (terminalWs) {
        try { terminalWs.close(); } catch (e) {}
    }
    terminalWs = null;
}

function initializeTerminalEmulator() {
    if (terminalEmulator) return terminalEmulator;
    const mount = document.getElementById('terminalCanvas');
    if (!mount || !window.Terminal) return null;
    terminalEmulator = new window.Terminal({
        cursorBlink: true,
        convertEol: true,
        fontFamily: '"IBM Plex Mono", "Courier New", monospace',
        fontSize: 12,
        theme: {
            background: '#0b1220',
            foreground: '#dce7f6',
        },
    });
    if (window.FitAddon && typeof window.FitAddon.FitAddon === 'function') {
        terminalFitAddon = new window.FitAddon.FitAddon();
        terminalEmulator.loadAddon(terminalFitAddon);
    }
    terminalEmulator.open(mount);
    terminalEmulator.onData((data) => {
        sendTerminalMessage({ type: 'input', content: data });
    });
    terminalEmulator.onResize((size) => {
        sendTerminalMessage({ type: 'resize', cols: size.cols, rows: size.rows });
    });
    fitTerminalEmulator();
    return terminalEmulator;
}

function fitTerminalEmulator() {
    if (terminalFitAddon && typeof terminalFitAddon.fit === 'function') {
        try { terminalFitAddon.fit(); } catch (e) {}
    }
}

function sendTerminalResize() {
    if (!terminalEmulator) return;
    sendTerminalMessage({ type: 'resize', cols: terminalEmulator.cols, rows: terminalEmulator.rows });
}

function sendTerminalMessage(payload) {
    if (!terminalWs || terminalWs.readyState !== WebSocket.OPEN) return false;
    terminalWs.send(JSON.stringify(payload));
    return true;
}

function interruptTerminal() {
    sendTerminalMessage({ type: 'signal', signal: 'interrupt' });
}

function restartTerminal() {
    sendTerminalMessage({ type: 'restart' });
}

function recordTerminalCommandResult(command, payload = {}) {
    if (!command && !terminalEntries.length) return;
    const returncode = payload.returncode;
    const cwd = payload.cwd || '.';
    const next = {
        command: command || terminalEntries[terminalEntries.length - 1]?.command || '',
        stdout: payload.stdout || '',
        stderr: payload.stderr || payload.error || '',
        meta: returncode == null ? `Command failed • ${cwd}` : `Exit ${returncode} • ${cwd}`,
        pending: false,
    };
    const lastEntry = terminalEntries[terminalEntries.length - 1];
    if (lastEntry?.pending && (!command || lastEntry.command === command)) {
        terminalEntries[terminalEntries.length - 1] = { ...lastEntry, ...next };
    } else {
        terminalEntries = terminalEntries.concat({ id: Date.now() + Math.random(), ...next }).slice(-WORKSPACE_ACTIVITY_LIMIT);
    }
    renderTerminalOutput();
}

function renderFileList() {
    const listEl = document.getElementById('workspaceFileList');
    const pathEl = document.getElementById('workspacePath');
    if (!listEl || !pathEl) return;

    if (!workspaceTree) {
        pathEl.textContent = 'No files yet';
        listEl.innerHTML = '<div class="workspace-empty">Choose a chat action that creates or uploads files, and the workspace tree will appear here.</div>';
        return;
    }

    pathEl.textContent = `${workspaceStats.files} file${workspaceStats.files === 1 ? '' : 's'} • ${workspaceStats.directories} folder${workspaceStats.directories === 1 ? '' : 's'}`;
    renderWorkspaceTrees();
}

function getCodeMirrorModeForPath(path) {
    if (!(window.CodeMirror && Array.isArray(window.CodeMirror.modeInfo))) return null;
    return window.CodeMirror.findModeByFileName(path || '') || null;
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
    if (prefix === 'inlineViewer') syncInlineViewerMode();
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
    const viewer = document.querySelector('.ide-viewer');
    if (!viewer) return;
    viewer.classList.toggle('is-hidden', !inlineViewerPath);
}

function toggleViewerEmptyState(prefix, hasFile) {
    const emptyEl = document.getElementById(`${prefix}Empty`);
    const editPane = document.getElementById(`${prefix}EditPane`);
    const previewPane = document.getElementById(`${prefix}PreviewPane`);
    if (!emptyEl) return;
    emptyEl.hidden = hasFile;
    if (!hasFile) {
        if (editPane) editPane.classList.add('file-modal-pane-hidden');
        if (previewPane) previewPane.classList.add('file-modal-pane-hidden');
    }
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

function renderHtmlPreview(targetId, content) {
    const previewEl = document.getElementById(targetId);
    if (!previewEl) return;
    previewEl.classList.remove('file-modal-markdown');
    previewEl.innerHTML = '';
    const iframe = document.createElement('iframe');
    iframe.className = 'file-modal-html';
    iframe.setAttribute('sandbox', 'allow-same-origin');
    iframe.srcdoc = content || '';
    previewEl.appendChild(iframe);
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
    if (state.kind === 'html') renderHtmlPreview(previewId, content);
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
        const resp = await fetch(`/api/workspace/${encodeURIComponent(currentConvId)}/file`, {
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
    if (titleEl) titleEl.textContent = path ? basename(path) : 'File Viewer';
    if (subtitleEl) subtitleEl.textContent = path || 'Select a workspace file to inspect or edit it here.';
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
    toggleViewerEmptyState(prefix, Boolean(path));
    syncInlineViewerMode();
    syncInlineViewerVisibility();
}

function syncViewerControls(prefix, editable, defaultView = 'edit') {
    const editTab = document.getElementById(`${prefix}EditTab`);
    const saveButton = document.getElementById(`${prefix}SaveButton`);
    if (editTab) editTab.hidden = !editable;
    if (saveButton) saveButton.hidden = !editable;
    setViewerView(prefix, editable ? defaultView : 'preview');
    if (!getViewerState(prefix).path) toggleViewerEmptyState(prefix, false);
    syncInlineUndoButton();
    syncInlineSelectionAction();
    syncInlineViewerMode();
    syncInlineViewerVisibility();
}

function closeInlineViewer() {
    if (inlineViewerAutosaveTimer) {
        clearTimeout(inlineViewerAutosaveTimer);
        inlineViewerAutosaveTimer = null;
    }
    selectedWorkspaceFile = '';
    selectedSpreadsheetSheet = '';
    inlineViewerPath = '';
    inlineViewerKind = 'text';
    inlineViewerEditable = false;
    inlineViewerView = 'edit';
    currentFileSymbols = [];
    applyViewerMetadata('inlineViewer', '');
    renderCurrentFileSymbols();
    syncViewerControls('inlineViewer', false, 'preview');
    renderFileList();
}

async function openWorkspaceFile(path, options = {}) {
    if (!featureSettings.agent_tools || !path) return;
    const preserveSheet = Boolean(options.preserveSheet);
    if (!preserveSheet) selectedSpreadsheetSheet = '';
    selectedWorkspaceFile = path;
    inlineViewerPath = path;
    renderFileList();
    applyViewerMetadata('inlineViewer', path);
    setViewerContent('inlineViewer', '');
    renderViewerSheetTabs('inlineViewer', [], '');

    try {
        if (/\.(xlsx|xls|xlsm)$/i.test(path)) {
            setViewerState('inlineViewer', { kind: 'spreadsheet', editable: false, path });
            const params = new URLSearchParams({ path });
            if (selectedSpreadsheetSheet) params.set('sheet', selectedSpreadsheetSheet);
            const resp = await fetch(`/api/workspace/${encodeURIComponent(currentConvId)}/spreadsheet?${params.toString()}`);
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
            selectedSpreadsheetSheet = data.sheet || '';
            renderViewerSheetTabs('inlineViewer', data.sheet_names || [], selectedSpreadsheetSheet);
            renderSpreadsheetPreview('inlineViewerPreview', data);
            syncViewerControls('inlineViewer', false, 'preview');
            return;
        }

        const resp = await fetch(`/api/workspace/${encodeURIComponent(currentConvId)}/file?path=${encodeURIComponent(path)}`);
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);

        const editable = isEditableTextPath(path);
        const kind = isHtmlPath(path) ? 'html' : (isMarkdownPath(path) ? 'markdown' : (/\.(csv|tsv)$/i.test(path) ? 'csv' : 'text'));
        setViewerState('inlineViewer', { editable, kind, path });
        setViewerContent('inlineViewer', data.content || '');
        inlineViewerLastSavedContent = data.content || '';
        setInlineViewerSaveState('idle');
        syncInlineUndoButton();

        if (kind === 'html') {
            renderHtmlPreview('inlineViewerPreview', data.content || '');
        }
        else if (kind === 'markdown') {
            renderMarkdownPreview('inlineViewerPreview', data.content || '');
        }
        else if (kind === 'csv') {
            const summaryResp = await fetch(`/api/workspace/${encodeURIComponent(currentConvId)}/spreadsheet?path=${encodeURIComponent(path)}`);
            const summaryData = await summaryResp.json();
            if (summaryResp.ok) {
                renderSpreadsheetPreview('inlineViewerPreview', summaryData);
            } else {
                renderDelimitedPreview('inlineViewerPreview', data.content || '', path);
            }
        } else {
            renderCodePreview('inlineViewerPreview', data.content || '', path);
        }

        syncViewerControls('inlineViewer', editable, inlineViewerView || 'edit');
        configureInlineEditorLanguage(path);
        updateCurrentFileSymbols(path, data.content || '');
        updateInlineViewerPreview();
    } catch (e) {
        alert(`Failed to open ${path}: ${e.message}`);
        closeInlineViewer();
    }
}

function renderPendingAttachments() {
    const bar = document.getElementById('attachmentBar');
    if (!bar) return;

    if (!pendingAttachments.length) {
        bar.hidden = true;
        bar.innerHTML = '';
        return;
    }

    bar.hidden = false;
    bar.innerHTML = '';
    pendingAttachments.forEach((file, index) => {
        const chip = document.createElement('div');
        chip.className = 'attachment-chip';
        chip.innerHTML = `
            <div class="attachment-chip-copy">
                <span class="attachment-chip-name">${escapeHtml(file.name || file.path)}</span>
                <span class="attachment-chip-meta">${escapeHtml(file.kind || 'file')} • ${escapeHtml(formatBytes(file.size || 0))}</span>
            </div>
            <button type="button" class="attachment-chip-remove" title="Remove attachment" aria-label="Remove attachment">&times;</button>
        `;
        chip.querySelector('.attachment-chip-remove').onclick = () => {
            pendingAttachments.splice(index, 1);
            renderPendingAttachments();
        };
        bar.appendChild(chip);
    });
}

function clearPendingAttachments() {
    pendingAttachments = [];
    const input = document.getElementById('attachmentInput');
    if (input) input.value = '';
    renderPendingAttachments();
}

function openAttachmentPicker() {
    document.getElementById('attachmentInput')?.click();
}

async function handleAttachmentSelection(event) {
    const input = event.target;
    const files = Array.from(input.files || []);
    if (!files.length) return;
    if (!featureSettings.agent_tools) {
        featureSettings.agent_tools = true;
        persistFeatureSettings();
        applyFeatureSettingsToUI();
    }

    const formData = new FormData();
    files.slice(0, 8).forEach(file => formData.append('files', file));
    formData.append('target_path', '.');

    const attachButton = document.getElementById('attachButton');
    if (attachButton) attachButton.disabled = true;

    try {
        const resp = await fetch(`/api/workspace/${encodeURIComponent(currentConvId)}/upload`, {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);

        pendingAttachments = pendingAttachments.concat(data.files || []).slice(0, 8);
        renderPendingAttachments();
        if (data.files?.length) {
            selectedWorkspaceFile = data.files[0].path || '';
        }
        refreshWorkspace(true);
    } catch (e) {
        alert(`Attachment upload failed: ${e.message}`);
    } finally {
        if (attachButton) attachButton.disabled = false;
        input.value = '';
    }
}

function sendMessage() {
    const input = document.getElementById('input');
    const displayText = input.value.trim();
    if (!modelAvailable || !displayText || !ws || ws.readyState !== WebSocket.OPEN || isGenerating) return;
    dismissExecutionPlan();
    hideSlashMenu();
    const slashCommand = parseDirectSlashCommandInput(displayText);

    // Resolve any collapsed paste placeholders to actual content
    let message = displayText;
    for (const block of pastedBlocks) {
        message = message.replace(block.placeholder, block.actual);
    }

    document.querySelector('.welcome')?.remove();
    exitWelcomeMode();
    const attachmentNote = pendingAttachments.length
        ? `\n\nAttached files:\n${pendingAttachments.map(file => `- ${file.path || file.name}`).join('\n')}`
        : '';
    addMessage(displayText + attachmentNote, 'user');
    input.value = '';
    pastedBlocks = [];
    const attachmentPaths = pendingAttachments.map(file => file.path).filter(Boolean);
    const turnFeatures = resolveTurnFeatures(message, attachmentPaths, slashCommand);
    clearPendingAttachments();
    autoResizeTextarea(input);
    showLoading();
    clearBuildSteps();
    clearWorkspaceActivity();
    recordWorkspaceActivity('Request', displayText);

    isGenerating = true;
    activeStreamConversationId = currentConvId;
    streamingAssistantMessage = null;
    syncSendButton();
    ws.send(JSON.stringify({
        message,
        attachments: attachmentPaths,
        conversation_id: currentConvId,
        system_prompt: localStorage.getItem('customSystemPrompt') || null,
        mode: deepMode ? 'deep' : 'normal',
        features: turnFeatures,
        slash_command: slashCommand ? {
            name: slashCommand.name,
            raw_name: slashCommand.rawName,
            args: slashCommand.args,
        } : null,
    }));
}

function sendInterruptMessage(messageText) {
    const text = String(messageText || '').trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    dismissExecutionPlan();
    const slashCommand = parseDirectSlashCommandInput(text);
    const attachmentPaths = pendingAttachments.map(file => file.path).filter(Boolean);
    const turnFeatures = resolveTurnFeatures(text, attachmentPaths, slashCommand);
    const attachmentNote = pendingAttachments.length
        ? `\n\nAttached files:\n${pendingAttachments.map(file => `- ${file.path || file.name}`).join('\n')}`
        : '';
    addMessage(text + attachmentNote, 'user');
    clearPendingAttachments();
    document.querySelector('.welcome')?.remove();
    exitWelcomeMode();
    recordWorkspaceActivity('Interrupt', text);
    ws.send(JSON.stringify({
        type: 'interrupt',
        message: text,
        attachments: attachmentPaths,
        conversation_id: currentConvId,
        system_prompt: localStorage.getItem('customSystemPrompt') || null,
        mode: deepMode ? 'deep' : 'normal',
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
    if (isGenerating) {
        if (draft) {
            input.value = '';
            autoResizeTextarea(input);
            syncSendButton();
            sendInterruptMessage(draft);
        } else {
            stopMessage();
        }
        return;
    }
    sendMessage();
}

function stopMessage() {
    if (!isGenerating || !ws || ws.readyState !== WebSocket.OPEN) return;
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
    messages.appendChild(streamingAssistantMessage);
    scrollToBottom();
}

function ensureStreamingAssistantMessage() {
    if (!streamingAssistantMessage) {
        streamingAssistantMessage = createMessageElement('', 'assistant');
    }
    appendStreamingAssistantMessageIfVisible();
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
    stack.appendChild(box);
    if (msg.isConnected) scrollToBottom();
}

function onAssistantThinkToken(text) {
    const msg = currentStreamingAssistantMessage();
    const box = msg?.querySelector('.think-box--streaming');
    const body = box?.querySelector('.think-body');
    if (!body) return;
    body.appendChild(document.createTextNode(text));
    if (msg.isConnected) scrollToBottom();
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
    if (msg.isConnected) scrollToBottom();
}

function appendAssistantAnswerToken(text) {
    const msg = currentStreamingAssistantMessage();
    if (!msg) return;
    const contentDiv = msg.querySelector('.message-content');
    if (!contentDiv) return;
    msg.dataset.originalContent = (msg.dataset.originalContent || '') + text;
    contentDiv.textContent = msg.dataset.originalContent;
    if (msg.isConnected) scrollToBottom();
}

function replaceAssistantAnswer(content) {
    const msg = currentStreamingAssistantMessage();
    if (!msg) return;
    const stack = msg.querySelector('.think-stack');
    const contentDiv = msg.querySelector('.message-content');
    if (!stack || !contentDiv) return;
    stack.innerHTML = '';
    msg.dataset.originalContent = '';
    msg.dataset.autoSpoken = 'false';
    hydrateAssistantFromRaw(msg, content || '');
    if (msg.isConnected) scrollToBottom();
}

function addMessage(content, role, timestamp = null, options = {}) {
    const msg = createMessageElement(content, role, timestamp, options);
    document.getElementById('messages').appendChild(msg);
    scrollToBottom();
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
    document.getElementById('messages').appendChild(loading);
    scrollToBottom();
}

function removeLoading() {
    document.getElementById('loadingIndicator')?.remove();
}

function scrollToBottom() {
    const messages = document.getElementById('messages');
    messages.scrollTop = messages.scrollHeight;
}

function handleKeyDown(event) {
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
    document.querySelectorAll('.message.assistant[data-needs-markdown="true"]').forEach(msg => {
        const contentDiv = msg.querySelector('.message-content');
        if (!contentDiv) return;
        // Read raw content from data attribute (answer only); thinking lives in .think-stack
        const content = msg.dataset.originalContent || contentDiv.textContent || contentDiv.innerText;
        const hasThink = msg.querySelector('.think-stack .think-box');
        if ((!content || !content.trim()) && !hasThink) return;

        const tsDiv = msg.querySelector('.message-timestamp');
        try {
            const originalContent = msg.dataset.originalContent || content;
            const html = (content && content.trim()) ? marked.parse(content) : '';
            contentDiv.innerHTML = html;
            msg.dataset.needsMarkdown = 'false';
            msg.dataset.rendered = 'true';

            if (tsDiv && !msg.querySelector('.message-timestamp')) msg.appendChild(tsDiv);

            // Action bar below assistant message (like Claude)
            if (!msg.querySelector('.message-actions')) {
                const copyText = (originalContent || '').trim();
                const actions = document.createElement('div');
                actions.className = 'message-actions';

                const copyBtn = document.createElement('button');
                copyBtn.className = 'action-btn';
                copyBtn.innerHTML = '&#128203; Copy';
                copyBtn.onclick = (e) => { e.stopPropagation(); copyToClipboard(copyText, copyBtn); };
                actions.appendChild(copyBtn);

                const addBtn = document.createElement('button');
                addBtn.className = 'action-btn';
                addBtn.innerHTML = '&#128190; Add to WS';
                addBtn.title = 'Recover this assistant reply into staged workspace files and notes';
                addBtn.onclick = async (e) => {
                    e.stopPropagation();
                    await recoverAssistantMessage(msg, addBtn);
                };
                actions.appendChild(addBtn);

                const tsDiv = msg.querySelector('.message-timestamp');
                if (tsDiv) msg.insertBefore(actions, tsDiv);
                else msg.appendChild(actions);
            }

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
        }
    });
}

function copyToClipboard(text, button) {
    const onSuccess = () => {
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
    const resp = await fetch(`/api/workspace/${encodeURIComponent(currentConvId)}/files?path=${encodeURIComponent(path)}`);
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
        renderBuildSteps();
        renderWorkspaceActivity();
        return;
    }

    if (force) pathEl.textContent = 'Refreshing workspace...';

    try {
        const tree = await buildWorkspaceTree('.');
        workspaceTree = tree;
        workspaceEntries = flattenWorkspaceFiles(tree, []);
        workspaceStats = collectWorkspaceStats(tree);
        currentRunId = tree.run_id || null;
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
        workspaceStats = { files: 0, directories: 0 };
        pathEl.textContent = 'Workspace unavailable';
        listEl.innerHTML = `<div class="workspace-empty">Workspace is unavailable: ${escapeHtml(e.message)}</div>`;
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

            item.innerHTML = `
                <div class="conv-title">${conv.title}</div>
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
    currentConvId = id;
    currentRunId = null;
    currentAssistantTurnStartedAt = null;
    currentAssistantTurnArtifactPaths = new Set();
    latestAssistantTurnArtifactPaths = new Set();
    selectedWorkspaceFile = '';
    selectedSpreadsheetSheet = '';
    clearBuildSteps();
    dismissExecutionPlan();
    if (!preserveActivity) clearWorkspaceActivity();
    if (!preserveActivity) clearTerminalOutput();
    closeInlineViewer();
    clearPendingAttachments();
    const resp = await fetch(`/api/conversation/${id}`);
    const data = await resp.json();
    const messages = document.getElementById('messages');
    stopSpeaking();
    messages.innerHTML = '';
    exitWelcomeMode();
    data.messages.forEach(msg => {
        const el = addMessage(msg.content, msg.role, msg.timestamp, { messageId: msg.id });
        if (msg.role === 'assistant') el.dataset.autoSpoken = 'true';
    });
    appendStreamingAssistantMessageIfVisible();
    setTimeout(() => renderMarkdown(), 100);
    loadConversations();
    refreshWorkspace(true);
}

function newChat() {
    stopSpeaking();
    clearAllowedCommandsForConversation(currentConvId);
    currentConvId = generateId();
    currentRunId = null;
    currentAssistantTurnStartedAt = null;
    currentAssistantTurnArtifactPaths = new Set();
    latestAssistantTurnArtifactPaths = new Set();
    selectedWorkspaceFile = '';
    selectedSpreadsheetSheet = '';
    clearBuildSteps();
    dismissExecutionPlan();
    clearWorkspaceActivity();
    clearTerminalOutput();
    closeInlineViewer();
    clearPendingAttachments();
    document.getElementById('messages').innerHTML = buildWelcomeMarkup();
    enterWelcomeMode();
    loadConversations();
    refreshWorkspace(true);
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
    if (id === currentConvId) newChat(); else loadConversations();
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
    if (document.getElementById('settingsOverlay').classList.contains('show')) { closeSettings(); return; }
    if (document.getElementById('dashboardOverlay').classList.contains('show')) { closeDashboard(); return; }
    if (document.getElementById('menuOverlay').classList.contains('show')) { closeMenu(); return; }
});

// ==================== Init ====================

updateStatus('loading');
connectWS();
loadPet();
loadConversations();
applyFeatureSettingsToUI();
syncMobileReasoningBadge();
refreshVoiceRuntime();
loadComposerRuntime();
refreshWorkspace(true);
initWelcomeMascot();
startWelcomeHintRotation();

setTimeout(() => {
    fetch('/health').then(r => r.json()).then(health => {
        if (health.model_available) {
            updateStatus('connected');
            loadComposerRuntime();
        } else {
            updateStatus('loading');
            startHealthPolling();
        }
    }).catch(() => { updateStatus('loading'); startHealthPolling(); });
}, 1000);

const _input = document.getElementById('input');
if (_input) {
    handleInputChange(_input);
    window.addEventListener('resize', () => autoResizeTextarea(_input));
    window.addEventListener('resize', positionSlashMenu);
    window.addEventListener('scroll', positionSlashMenu, true);
    _input.focus();
    _input.addEventListener('focus', () => updateSlashMenu(_input));
    _input.addEventListener('blur', () => {
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

syncReasoningSelector();
syncModelSelector();
renderWorkspaceActivity();
renderTerminalOutput();
applyViewerMetadata('inlineViewer', '');
window.addEventListener('resize', () => {
    fitTerminalEmulator();
    sendTerminalResize();
});
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopSpeaking();
        if (dictationActive) stopDictationAndUpload();
    }
});
