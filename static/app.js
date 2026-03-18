// AI Chat Application

// ==================== State ====================

let ws = null;
let logWs = null;
let currentConvId = generateId();
let isGenerating = false;
let renameConvId = null;
let modelAvailable = false;
let markedReady = false;
let healthPollInterval = null;

// ==================== Theme ====================

function applyTheme(mode) {
    const themes = window.THEMES || {};
    const colors = themes[mode] || themes.light || {};
    const root = document.documentElement;
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
        const send = document.getElementById('send');
        if (send) send.disabled = false;
    } else if (status === 'loading') {
        dot.className = 'status-dot loading';
        modelAvailable = false;
        const send = document.getElementById('send');
        if (send) send.disabled = true;
    } else {
        dot.className = 'status-dot';
        modelAvailable = false;
        const send = document.getElementById('send');
        if (send) send.disabled = true;
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
            addMessage('', 'assistant');
        } else if (data.type === 'token') {
            appendToLastMessage(data.content);
        } else if (data.type === 'done') {
            isGenerating = false;
            document.getElementById('send').disabled = false;
            document.getElementById('input').focus();
            renderMarkdown();
            loadConversations();
        } else if (data.type === 'set_system_prompt') {
            if (data.content) {
                localStorage.setItem('customSystemPrompt', data.content);
            } else {
                localStorage.removeItem('customSystemPrompt');
            }
        } else if (data.type === 'set_temperature') {
            localStorage.setItem('temperature', data.value);
        } else if (data.type === 'error') {
            removeLoading();
            isGenerating = false;
            document.getElementById('send').disabled = false;
            const errorMsg = document.createElement('div');
            errorMsg.className = 'message assistant';
            errorMsg.innerHTML = `<div class="message-content" style="color:#ef4444;">${data.content}</div>`;
            document.getElementById('messages').appendChild(errorMsg);
            scrollToBottom();
        }
    };

    ws.onerror = () => updateStatus('disconnected');
    ws.onclose = () => {
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
}

async function loadDashboard() {
    const el = document.getElementById('dashboardContent');
    try {
        const resp = await fetch('/api/dashboard');
        const data = await resp.json();

        const statusClass = data.model_available ? 'connected' : 'loading';
        const statusText = data.model_available ? 'Connected' : 'Loading / Unavailable';
        const containerStatus = data.container ? (data.container.running ? 'Running' : data.container.status || 'Stopped') : 'Unknown';
        const cache = data.cache || {};
        const cacheStatus = cache.status || 'unknown';
        const cacheSize = cache.size_display || '--';
        const cacheDate = cache.last_modified ? new Date(cache.last_modified).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : '--';

        el.innerHTML = `
            <div class="dash-section">
                <h4>Model</h4>
                <div class="dash-card">
                    <div class="dash-row"><span class="label">Name</span><span class="value">${data.model_name}</span></div>
                    <div class="dash-row"><span class="label">Status</span><span class="value"><span class="status-dot ${statusClass}"></span>${statusText}</span></div>
                    <div class="dash-row"><span class="label">Container</span><span class="value">${containerStatus}</span></div>
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
                    <button class="dash-btn" onclick="dashRestart()">
                        Restart vLLM
                        <div class="btn-desc">Restart the inference server (keeps cached model)</div>
                    </button>
                    <button class="dash-btn danger" onclick="dashRedownload()">
                        Redownload Model
                        <div class="btn-desc">Delete cache and re-download from HuggingFace</div>
                    </button>
                </div>
            </div>
            <div class="dash-section">
                <h4>Logs</h4>
                <div class="dash-logs" id="dashLogs"></div>
            </div>
        `;
    } catch (e) {
        el.innerHTML = `<div style="padding:20px;color:var(--text_secondary)">Failed to load dashboard: ${e.message}</div>`;
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

// ==================== Settings ====================

function showSettings() {
    document.getElementById('settingsOverlay').classList.add('show');
}
function closeSettings() {
    document.getElementById('settingsOverlay').classList.remove('show');
}

// ==================== Chat ====================

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    const maxH = window.innerHeight * 0.5;
    textarea.style.height = Math.min(Math.max(textarea.scrollHeight, 24), maxH) + 'px';
}

function exitWelcomeMode() {
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
}

function focusInput() {
    const input = document.getElementById('input');
    if (input) input.focus();
}

function sendMessage() {
    const input = document.getElementById('input');
    const message = input.value.trim();
    if (!modelAvailable || !message || !ws || ws.readyState !== WebSocket.OPEN || isGenerating) return;

    document.querySelector('.welcome')?.remove();
    exitWelcomeMode();
    addMessage(message, 'user');
    input.value = '';
    autoResizeTextarea(input);
    showLoading();

    isGenerating = true;
    document.getElementById('send').disabled = true;

    const payload = {
        message: message,
        conversation_id: currentConvId,
        system_prompt: localStorage.getItem('customSystemPrompt') || null
    };
    const temp = localStorage.getItem('temperature');
    if (temp !== null) payload.temperature = parseFloat(temp);
    ws.send(JSON.stringify(payload));
}

function addMessage(content, role, timestamp = null) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    msg.appendChild(contentDiv);

    const tsDiv = document.createElement('div');
    tsDiv.className = 'message-timestamp';
    tsDiv.textContent = formatTimestamp(timestamp || new Date().toISOString());
    msg.appendChild(tsDiv);

    if (role === 'assistant') {
        msg.dataset.needsMarkdown = 'true';
        msg.dataset.originalContent = content;
    }
    document.getElementById('messages').appendChild(msg);
    scrollToBottom();
    return msg;
}

function appendToLastMessage(content) {
    const messages = document.getElementById('messages');
    const lastMsg = messages.lastElementChild;
    if (!lastMsg?.classList.contains('assistant')) return;

    const contentDiv = lastMsg.querySelector('.message-content');
    if (!contentDiv) return;

    // Accumulate raw content in data attribute (source of truth)
    const rawContent = (lastMsg.dataset.originalContent || '') + content;
    lastMsg.dataset.originalContent = rawContent;

    // Display based on <think> tag state
    const hasThinkOpen = rawContent.includes('<think>');
    const hasThinkClose = rawContent.includes('</think>');

    if (hasThinkOpen && !hasThinkClose) {
        // Still inside thinking block — show thinking indicator
        const thinkStart = rawContent.indexOf('<think>') + 7;
        const thinkText = rawContent.substring(thinkStart);
        contentDiv.innerHTML =
            '<div class="thinking-block streaming">' +
            '<div class="thinking-header">Thinking...</div>' +
            '<div class="thinking-content">' + escapeHtml(thinkText) + '</div>' +
            '</div>';
    } else if (hasThinkOpen && hasThinkClose) {
        // Think block complete — show collapsed think + streaming response
        const thinkMatch = rawContent.match(/<think>([\s\S]*?)<\/think>([\s\S]*)/);
        if (thinkMatch) {
            let html = '<details class="thinking-block"><summary>Thinking</summary>' +
                '<div class="thinking-content">' + escapeHtml(thinkMatch[1].trim()) + '</div></details>';
            const afterThink = thinkMatch[2];
            if (afterThink.trim()) {
                html += '<div class="streaming-text">' + escapeHtml(afterThink) + '</div>';
            }
            contentDiv.innerHTML = html;
        } else {
            contentDiv.textContent = rawContent;
        }
    } else {
        // No think tags — show raw text
        contentDiv.textContent = rawContent;
    }

    scrollToBottom();
}

function showLoading() {
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
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
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
        // Read raw content from data attribute (source of truth), fallback to textContent
        const content = msg.dataset.originalContent || contentDiv.textContent || contentDiv.innerText;
        if (!content || !content.trim()) return;

        const tsDiv = msg.querySelector('.message-timestamp');
        try {
            let html = '';

            // Handle <think> blocks — extract and render as collapsible sections
            const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
            let lastIndex = 0;
            let match;
            let hasThinkBlocks = false;

            while ((match = thinkRegex.exec(content)) !== null) {
                hasThinkBlocks = true;
                // Render content before the think block
                const before = content.substring(lastIndex, match.index);
                if (before.trim()) html += marked.parse(before);
                // Render think block as collapsible section with markdown inside
                const thinkHtml = marked.parse(match[1].trim());
                html += '<details class="thinking-block"><summary>Thinking</summary>' +
                    '<div class="thinking-content">' + thinkHtml + '</div></details>';
                lastIndex = match.index + match[0].length;
            }

            // Render remaining content after last think block
            const remaining = content.substring(lastIndex);
            if (remaining.trim()) html += marked.parse(remaining);

            // If no think blocks, just parse everything as markdown
            if (!hasThinkBlocks) html = marked.parse(content);

            const originalContent = msg.dataset.originalContent || content;
            contentDiv.innerHTML = html;
            msg.dataset.needsMarkdown = 'false';
            msg.dataset.rendered = 'true';

            if (tsDiv && !msg.querySelector('.message-timestamp')) msg.appendChild(tsDiv);

            // Action bar below assistant message (like Claude)
            if (!msg.querySelector('.message-actions')) {
                const copyText = originalContent.replace(/<think>[\s\S]*?<\/think>\s*/g, '').trim();
                const actions = document.createElement('div');
                actions.className = 'message-actions';

                const copyBtn = document.createElement('button');
                copyBtn.className = 'action-btn';
                copyBtn.innerHTML = '&#128203; Copy';
                copyBtn.onclick = (e) => { e.stopPropagation(); copyToClipboard(copyText, copyBtn); };
                actions.appendChild(copyBtn);

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

async function loadConversation(id) {
    currentConvId = id;
    const resp = await fetch(`/api/conversation/${id}`);
    const data = await resp.json();
    const messages = document.getElementById('messages');
    messages.innerHTML = '';
    exitWelcomeMode();
    data.messages.forEach(msg => addMessage(msg.content, msg.role, msg.timestamp));
    setTimeout(() => renderMarkdown(), 100);
    loadConversations();
}

function newChat() {
    currentConvId = generateId();
    const modelName = window.MODEL_NAME || '';
    document.getElementById('messages').innerHTML = '<div class="welcome">' + (modelName ? '<h1 style="font-weight:700; margin:0 0 0.25em;">' + modelName + '</h1>' : '') + '<h2>Welcome</h2><p>Start a conversation</p></div>';
    enterWelcomeMode();
    loadConversations();
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
loadConversations();

setTimeout(() => {
    fetch('/health').then(r => r.json()).then(health => {
        if (health.model_available) updateStatus('connected');
        else { updateStatus('loading'); startHealthPolling(); }
    }).catch(() => { updateStatus('loading'); startHealthPolling(); });
}, 1000);

const _input = document.getElementById('input');
if (_input) {
    autoResizeTextarea(_input);
    window.addEventListener('resize', () => autoResizeTextarea(_input));
    _input.focus();
}
