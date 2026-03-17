// AI Chat Application - Main JavaScript
// Simplified single-model setup (vLLM + Qwen)

const CONFIG = window.APP_CONFIG || {};

let ws = null;
let logWs = null;
let currentConvId = generateId();
let isGenerating = false;
let renameConvId = null;
let modelAvailable = false;
let markedReady = false;
let healthPollInterval = null;

// Initialize marked library when ready
function initMarked() {
    if (typeof marked !== 'undefined') {
        markedReady = true;
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false,
            pedantic: false,
            sanitize: false,
            smartLists: true,
            smartypants: true
        });
        console.log('Marked library loaded');
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

function generateId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c == 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

function updateStatus(status, progress = null, message = null) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const progressContainer = document.getElementById('statusProgressContainer');
    const progressBar = document.getElementById('statusProgressBar');
    const progressText = document.getElementById('statusProgressText');

    if (!statusDot || !statusText) return;

    if (status === true || status === 'connected') {
        statusDot.className = 'status-dot';
        statusText.textContent = 'Connected';
        if (progressContainer) progressContainer.style.display = 'none';
        modelAvailable = true;
        const input = document.getElementById('input');
        const send = document.getElementById('send');
        if (input) input.disabled = false;
        if (send) send.disabled = false;
    } else if (status === false || status === 'disconnected') {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = message || 'Disconnected';
        modelAvailable = false;
    } else if (status === 'loading') {
        statusDot.className = 'status-dot booting';
        statusText.textContent = message || 'Loading model...';
        if (progressContainer && progress !== null) {
            progressContainer.style.display = 'block';
            if (progressBar) progressBar.style.setProperty('--progress', progress + '%');
            if (progressText) progressText.textContent = message || '';
        }
        modelAvailable = false;
        const input = document.getElementById('input');
        const send = document.getElementById('send');
        if (input) input.disabled = true;
        if (send) send.disabled = true;
    }
}

// Poll /health until model is available
function startHealthPolling() {
    if (healthPollInterval) return;

    healthPollInterval = setInterval(async () => {
        try {
            const resp = await fetch('/health');
            const health = await resp.json();
            if (health.model_available) {
                modelAvailable = true;
                updateStatus(true);
                clearInterval(healthPollInterval);
                healthPollInterval = null;
            } else {
                updateStatus('loading', null, health.message || 'Loading model...');
            }
        } catch (e) {
            updateStatus('loading', null, 'Connecting...');
        }
    }, 3000);
}

// ==================== WebSocket ====================

function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

    ws.onopen = () => {
        console.log('WebSocket connected');
        // Check health on connect
        fetch('/health').then(r => r.json()).then(health => {
            if (health.model_available) {
                modelAvailable = true;
                updateStatus(true);
            } else {
                updateStatus('loading', null, health.message || 'Model loading...');
                startHealthPolling();
            }
        }).catch(() => {
            updateStatus('loading', null, 'Checking model...');
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
        } else if (data.type === 'message_id') {
            // Could store for feedback
        } else if (data.type === 'done') {
            isGenerating = false;
            document.getElementById('send').disabled = false;
            document.getElementById('input').disabled = false;
            document.getElementById('input').focus();
            renderMarkdown();
            loadConversations();
        } else if (data.type === 'error') {
            removeLoading();
            isGenerating = false;
            document.getElementById('send').disabled = false;
            document.getElementById('input').disabled = false;

            const errorMsg = document.createElement('div');
            errorMsg.className = 'message assistant';
            errorMsg.innerHTML = `<div class="message-content" style="color: #ef4444;">${data.content}</div>`;
            document.getElementById('messages').appendChild(errorMsg);
            scrollToBottom();
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus(false, null, 'Connection error');
    };

    ws.onclose = () => {
        console.log('WebSocket closed, reconnecting...');
        updateStatus(false, null, 'Reconnecting...');
        setTimeout(connectWS, 2000);
    };
}

function connectLogWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    logWs = new WebSocket(`${protocol}//${window.location.host}/ws/logs`);

    logWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
            const logContent = document.getElementById('logContent');
            if (logContent) {
                logContent.textContent += data.content;
                logContent.scrollTop = logContent.scrollHeight;
            }
        }
    };

    logWs.onerror = () => { logWs = null; };
    logWs.onclose = () => { logWs = null; };
}

// ==================== Theme & Settings ====================

function toggleTheme() {
    const currentMode = localStorage.getItem('theme') || 'light';
    const newMode = currentMode === 'light' ? 'dark' : 'light';
    localStorage.setItem('theme', newMode);
    document.getElementById('settingsMenuContent').classList.remove('show');
    window.location.href = `/?mode=${newMode}`;
}

document.addEventListener('DOMContentLoaded', function() {
    const currentMode = localStorage.getItem('theme') || 'light';
    const themeIcon = document.getElementById('themeIcon');
    if (themeIcon) {
        themeIcon.textContent = currentMode === 'light' ? '\ud83c\udf19' : '\u2600\ufe0f';
    }
});

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('sidebarToggle');
    const overlay = document.getElementById('sidebarOverlay');

    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('show');
        if (sidebar.classList.contains('show')) {
            if (overlay) { overlay.style.display = 'block'; overlay.classList.add('show'); }
        } else {
            if (overlay) { overlay.style.display = 'none'; overlay.classList.remove('show'); }
        }
    } else {
        sidebar.classList.toggle('collapsed');
        if (toggle) {
            toggle.textContent = sidebar.classList.contains('collapsed') ? '\ud83d\udcd5' : '\ud83d\udcd6';
        }
        if (sidebar.classList.contains('collapsed')) {
            if (overlay) { overlay.style.display = 'none'; overlay.classList.remove('show'); }
        } else {
            if (overlay) { overlay.style.display = 'block'; overlay.classList.add('show'); }
        }
    }
}

// Start collapsed on desktop
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    const toggle = document.getElementById('sidebarToggle');

    if (window.innerWidth > 768) {
        sidebar.classList.add('collapsed');
        if (overlay) { overlay.style.display = 'none'; overlay.classList.remove('show'); }
        if (toggle) toggle.textContent = '\ud83d\udcd5';
    }
});

function toggleLogViewer() {
    const viewer = document.getElementById('logViewer');
    viewer.classList.toggle('show');
    document.getElementById('settingsMenuContent').classList.remove('show');
    if (viewer.classList.contains('show') && !logWs) {
        connectLogWS();
    }
}

function toggleSettingsMenu(event) {
    event.stopPropagation();
    const content = document.getElementById('settingsMenuContent');
    content.classList.toggle('show');
}

document.addEventListener('click', function(event) {
    const settingsMenu = document.getElementById('settingsMenu');
    const settingsContent = document.getElementById('settingsMenuContent');
    if (settingsContent && settingsContent.classList.contains('show')) {
        if (!settingsMenu.contains(event.target)) {
            settingsContent.classList.remove('show');
        }
    }
});

// ==================== Web Search ====================

function toggleWebSearch() {
    const modal = document.getElementById('webSearchModal');
    modal.classList.toggle('show');
    document.getElementById('settingsMenuContent').classList.remove('show');
    if (modal.classList.contains('show')) {
        document.getElementById('webSearchInput').focus();
    }
}

async function performWebSearch() {
    const query = document.getElementById('webSearchInput').value.trim();
    if (!query) return;

    const resultsDiv = document.getElementById('webSearchResults');
    resultsDiv.innerHTML = '<div style="text-align: center; padding: 20px;">Searching...</div>';

    const sources = [];
    document.querySelectorAll('.web-search-source-checkbox input:checked').forEach(cb => {
        sources.push(cb.value);
    });

    try {
        const resp = await fetch('/api/web-search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query, sources, max_results: 10 })
        });
        const data = await resp.json();

        resultsDiv.innerHTML = '';
        if (data.results && data.results.length > 0) {
            data.results.forEach(result => {
                const item = document.createElement('div');
                item.className = 'web-search-result';
                item.innerHTML = `
                    <div class="web-search-result-title">
                        <a href="${result.url}" target="_blank">${result.title}</a>
                    </div>
                    <div class="web-search-result-url">${result.url}</div>
                    <div class="web-search-result-snippet">${result.snippet}</div>
                    <div class="web-search-result-source">${result.source}</div>
                `;
                resultsDiv.appendChild(item);
            });
        } else {
            resultsDiv.innerHTML = '<div style="text-align: center; padding: 20px;">No results found</div>';
        }
    } catch (e) {
        resultsDiv.innerHTML = `<div style="text-align: center; padding: 20px; color: #ef4444;">Search failed: ${e.message}</div>`;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('webSearchInput');
    if (searchInput) {
        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                performWebSearch();
            }
        });
    }
});

// ==================== Chat ====================

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    const maxHeight = window.innerHeight * 0.66;
    const newHeight = Math.min(Math.max(textarea.scrollHeight, 24), maxHeight);
    textarea.style.height = newHeight + 'px';
}

function focusInput() {
    const input = document.getElementById('input');
    if (input && !input.disabled) {
        input.focus();
        const len = input.value.length;
        input.setSelectionRange(len, len);
    }
}

function sendMessage() {
    const input = document.getElementById('input');
    const message = input.value.trim();

    if (!modelAvailable) return;
    if (!message || !ws || ws.readyState !== WebSocket.OPEN || isGenerating) return;

    document.querySelector('.welcome')?.remove();

    addMessage(message, 'user');
    input.value = '';
    autoResizeTextarea(input);

    showLoading();

    isGenerating = true;
    document.getElementById('send').disabled = true;
    document.getElementById('input').disabled = true;

    const customSystemPrompt = localStorage.getItem('customSystemPrompt') || '';

    ws.send(JSON.stringify({
        message: message,
        conversation_id: currentConvId,
        system_prompt: customSystemPrompt || null
    }));
}

function addMessage(content, role, timestamp = null) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    msg.appendChild(contentDiv);

    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'message-timestamp';
    timestampDiv.textContent = formatTimestamp(timestamp || new Date().toISOString());
    msg.appendChild(timestampDiv);

    if (role === 'assistant') {
        msg.dataset.needsMarkdown = 'true';
        msg.dataset.originalContent = content;
    }
    document.getElementById('messages').appendChild(msg);
    scrollToBottom();
    return msg;
}

function formatTimestamp(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const messageDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());

    const timeStr = date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });

    if (messageDate.getTime() === today.getTime()) {
        return timeStr;
    } else {
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        if (messageDate.getTime() === yesterday.getTime()) {
            return `Yesterday ${timeStr}`;
        } else {
            const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            return `${dateStr} ${timeStr}`;
        }
    }
}

function formatFullTimestamp(isoString) {
    const date = new Date(isoString);
    const dateStr = date.toLocaleDateString('en-US', {
        weekday: 'short', year: 'numeric', month: 'short', day: 'numeric'
    });
    const timeStr = date.toLocaleTimeString('en-US', {
        hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true
    });
    return `${dateStr} ${timeStr}`;
}

function downloadAsJSON(content, filename = 'chat-response.json') {
    const data = { content, timestamp: new Date().toISOString(), type: 'chat_response' };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.innerHTML;
        button.innerHTML = '\u2713 Copied';
        button.classList.add('copied');
        setTimeout(() => {
            button.innerHTML = originalText;
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.opacity = '0';
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            const originalText = button.innerHTML;
            button.innerHTML = '\u2713 Copied';
            button.classList.add('copied');
            setTimeout(() => {
                button.innerHTML = originalText;
                button.classList.remove('copied');
            }, 2000);
        } catch (e) {
            console.error('Fallback copy failed:', e);
        }
        document.body.removeChild(textArea);
    });
}

function renderMarkdown() {
    if (!markedReady || typeof marked === 'undefined') {
        setTimeout(renderMarkdown, 100);
        return;
    }

    const messages = document.querySelectorAll('.message.assistant[data-needs-markdown="true"]');
    messages.forEach(msg => {
        const contentDiv = msg.querySelector('.message-content');
        if (!contentDiv) return;

        const content = contentDiv.textContent || contentDiv.innerText;
        if (!content || content.trim() === '') return;

        const timestampDiv = msg.querySelector('.message-timestamp');

        try {
            const html = marked.parse(content);
            const originalContent = msg.dataset.originalContent || content;
            contentDiv.innerHTML = html;
            msg.dataset.needsMarkdown = 'false';
            msg.dataset.rendered = 'true';

            if (timestampDiv && !msg.querySelector('.message-timestamp')) {
                msg.appendChild(timestampDiv);
            }

            if (!msg.querySelector('.copy-btn.message-copy-btn')) {
                const copyBtn = document.createElement('button');
                copyBtn.className = 'copy-btn message-copy-btn';
                copyBtn.innerHTML = '\ud83d\udccb Copy';
                copyBtn.title = 'Copy to clipboard';
                copyBtn.onclick = (e) => {
                    e.stopPropagation();
                    copyToClipboard(originalContent, copyBtn);
                };
                msg.appendChild(copyBtn);
            }

            if (typeof hljs !== 'undefined' && hljs.highlightElement) {
                msg.querySelectorAll('pre code').forEach(block => {
                    try {
                        if (block.textContent && block.textContent.trim()) {
                            hljs.highlightElement(block);
                        }
                    } catch (e) {
                        console.warn('Syntax highlighting skipped:', e.message || e);
                    }
                });
            }

            msg.querySelectorAll('pre').forEach(preBlock => {
                try {
                    if (preBlock.querySelector('.copy-btn')) return;
                    const codeText = preBlock.querySelector('code')?.textContent || preBlock.textContent;
                    if (codeText) {
                        const copyBtn = document.createElement('button');
                        copyBtn.className = 'copy-btn';
                        copyBtn.innerHTML = '\ud83d\udccb Copy';
                        copyBtn.title = 'Copy code to clipboard';
                        copyBtn.onclick = (e) => {
                            e.stopPropagation();
                            copyToClipboard(codeText, copyBtn);
                        };
                        preBlock.appendChild(copyBtn);
                    }
                } catch (e) {
                    console.warn('Error adding copy button:', e);
                }
            });
        } catch (e) {
            console.error('Markdown render error:', e);
            contentDiv.textContent = content;
        }
    });
}

function showLoading() {
    const loading = document.createElement('div');
    loading.className = 'loading';
    loading.id = 'loadingIndicator';
    loading.innerHTML = `
        <div class="loading-spinner">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
        <div class="loading-text">Thinking...</div>
    `;
    document.getElementById('messages').appendChild(loading);
    scrollToBottom();
}

function removeLoading() {
    document.getElementById('loadingIndicator')?.remove();
}

function appendToLastMessage(content) {
    const messages = document.getElementById('messages');
    const lastMsg = messages.lastElementChild;
    if (lastMsg?.classList.contains('assistant')) {
        const contentDiv = lastMsg.querySelector('.message-content');
        if (contentDiv) {
            const currentText = contentDiv.textContent || '';
            contentDiv.textContent = currentText + content;
            if (lastMsg.dataset.originalContent !== undefined) {
                lastMsg.dataset.originalContent = currentText + content;
            }
        }
        scrollToBottom();
    }
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

// ==================== Search ====================

let searchTimeout = null;

async function handleSearch(query) {
    const searchResults = document.getElementById('searchResults');
    const conversations = document.getElementById('conversations');

    if (searchTimeout) clearTimeout(searchTimeout);

    if (!query || query.trim().length < 2) {
        searchResults.style.display = 'none';
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
                data.results.forEach(result => {
                    if (!grouped[result.conversation_id]) grouped[result.conversation_id] = [];
                    grouped[result.conversation_id].push(result);
                });

                Object.keys(grouped).forEach(convId => {
                    const results = grouped[convId];
                    const firstResult = results[0];

                    const item = document.createElement('div');
                    item.className = 'search-result-item';
                    item.onclick = () => {
                        loadConversation(convId);
                        document.getElementById('searchInput').value = '';
                        searchResults.style.display = 'none';
                        conversations.style.display = 'block';
                    };

                    const preview = firstResult.content.substring(0, 100);
                    const timestamp = formatTimestamp(firstResult.timestamp);

                    item.innerHTML = `
                        <div class="search-result-title">${firstResult.conversation_title}</div>
                        <div class="search-result-preview">${preview}${firstResult.content.length > 100 ? '...' : ''}</div>
                        <div class="search-result-meta">${results.length} match${results.length > 1 ? 'es' : ''} \u2022 ${timestamp}</div>
                    `;

                    searchResults.appendChild(item);
                });
            } else {
                searchResults.innerHTML = '<div class="search-no-results">No results found</div>';
            }
        } catch (e) {
            console.error('Search error:', e);
            searchResults.innerHTML = '<div class="search-no-results">Error searching</div>';
        }
    }, 300);
}

// ==================== Conversations ====================

async function loadConversations() {
    const resp = await fetch('/api/conversations');
    const data = await resp.json();

    const container = document.getElementById('conversations');
    container.innerHTML = '';

    data.conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'conv-item conversation-item' + (conv.id === currentConvId ? ' active' : '');

        const lastTimestamp = conv.last_message_timestamp || conv.updated_at;
        const formattedTimestamp = formatFullTimestamp(lastTimestamp);

        let previewText = conv.last_message || '';
        previewText = previewText.replace(/[#*_`\[\]]/g, '').replace(/\n/g, ' ').trim();
        const maxPreviewLength = 150;
        const shortPreview = previewText.length > maxPreviewLength
            ? previewText.substring(0, maxPreviewLength) + '...'
            : previewText;

        if (window.innerWidth <= 768) {
            item.innerHTML = `
                <div class="conv-content">
                    <div class="conv-title">${conv.title}</div>
                    <div class="conv-preview">${shortPreview || 'No messages yet'}</div>
                    <div class="conv-timestamp">${formattedTimestamp}</div>
                </div>
                <div class="swipe-actions">
                    <button onclick="event.stopPropagation(); renameConv('${conv.id}', '${conv.title.replace(/'/g, "\\'")}')">Rename</button>
                    <button onclick="event.stopPropagation(); deleteConv('${conv.id}')" style="background: #fd7589;">Delete</button>
                </div>
            `;
        } else {
            item.innerHTML = `
                <div class="conv-content">
                    <div class="conv-title">${conv.title}</div>
                    <div class="conv-preview" data-full-preview="${previewText.replace(/"/g, '&quot;')}">${shortPreview || 'No messages yet'}</div>
                    <div class="conv-timestamp">${formattedTimestamp}</div>
                </div>
                <div class="conv-actions">
                    <button class="conv-btn" onclick="event.stopPropagation(); renameConv('${conv.id}', '${conv.title.replace(/'/g, "\\'")}')">&#9998;</button>
                    <button class="conv-btn delete" onclick="event.stopPropagation(); deleteConv('${conv.id}')">&#128465;</button>
                </div>
            `;
        }

        item.onclick = () => loadConversation(conv.id);

        if (window.innerWidth <= 768) {
            let startX = 0, currentX = 0, isDragging = false;

            item.addEventListener('touchstart', (e) => { startX = e.touches[0].clientX; isDragging = false; });
            item.addEventListener('touchmove', (e) => {
                currentX = e.touches[0].clientX;
                const diff = startX - currentX;
                if (Math.abs(diff) > 10) isDragging = true;
                if (isDragging && diff > 0) {
                    e.preventDefault();
                    item.style.transform = `translateX(-${Math.min(diff, 150)}px)`;
                }
            });
            item.addEventListener('touchend', () => {
                const diff = startX - currentX;
                if (isDragging && diff > 50) {
                    item.classList.add('swipe-left');
                    item.style.transform = 'translateX(-150px)';
                } else {
                    item.classList.remove('swipe-left');
                    item.style.transform = '';
                }
                isDragging = false;
            });
        }

        container.appendChild(item);
    });
}

async function loadConversation(id) {
    currentConvId = id;
    const resp = await fetch(`/api/conversation/${id}`);
    const data = await resp.json();

    const messages = document.getElementById('messages');
    messages.innerHTML = '';

    data.messages.forEach(msg => {
        addMessage(msg.content, msg.role, msg.timestamp);
    });
    setTimeout(() => renderMarkdown(), 100);
    loadConversations();
}

function newChat() {
    currentConvId = generateId();
    document.getElementById('messages').innerHTML =
        '<div class="welcome"><h2>New Conversation</h2><p>Start chatting!</p></div>';
    loadConversations();
}

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
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({title: newTitle})
    });

    closeRenameModal();
    loadConversations();
}

async function deleteConv(id) {
    if (!confirm('Delete this conversation?')) return;

    await fetch(`/api/conversation/${id}`, {method: 'DELETE'});

    if (id === currentConvId) {
        newChat();
    } else {
        loadConversations();
    }
}

// ==================== System Prompt ====================

function openSystemPromptEditor() {
    const modal = document.getElementById('systemPromptModal');
    const textarea = document.getElementById('systemPromptInput');
    textarea.value = localStorage.getItem('customSystemPrompt') || '';
    modal.classList.add('show');
    textarea.focus();
    document.getElementById('settingsMenuContent').classList.remove('show');
}

function closeSystemPromptModal() {
    document.getElementById('systemPromptModal').classList.remove('show');
}

function saveSystemPrompt() {
    const prompt = document.getElementById('systemPromptInput').value.trim();
    if (prompt) {
        localStorage.setItem('customSystemPrompt', prompt);
    } else {
        localStorage.removeItem('customSystemPrompt');
    }
    closeSystemPromptModal();
}

function resetSystemPrompt() {
    document.getElementById('systemPromptInput').value = '';
    localStorage.removeItem('customSystemPrompt');
}

// ==================== Escape Key ====================

document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const webSearchModal = document.getElementById('webSearchModal');
        if (webSearchModal && webSearchModal.classList.contains('show')) { toggleWebSearch(); return; }

        const renameModal = document.getElementById('renameModal');
        if (renameModal && renameModal.classList.contains('show')) { closeRenameModal(); return; }

        const systemPromptModal = document.getElementById('systemPromptModal');
        if (systemPromptModal && systemPromptModal.classList.contains('show')) { closeSystemPromptModal(); return; }

        const logViewer = document.getElementById('logViewer');
        if (logViewer && logViewer.classList.contains('show')) { toggleLogViewer(); return; }
    }
});

// ==================== Initialization ====================

updateStatus('loading', null, 'Connecting...');
connectWS();
loadConversations();

// Initial health check
setTimeout(() => {
    fetch('/health').then(r => r.json()).then(health => {
        if (health.model_available) {
            modelAvailable = true;
            updateStatus(true);
        } else {
            updateStatus('loading', null, health.message || 'Loading model...');
            startHealthPolling();
        }
    }).catch(() => {
        updateStatus('loading', null, 'Checking connection...');
        startHealthPolling();
    });
}, 1000);

// Auto-resize textarea
const input = document.getElementById('input');
if (input) {
    autoResizeTextarea(input);
    window.addEventListener('resize', () => autoResizeTextarea(input));
}

document.getElementById('input').focus();
