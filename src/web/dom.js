// Generated from src/web/dom.ts by scripts/build_frontend.mjs
function query(selector) {
    const element = document.querySelector(selector);
    if (!element) {
        throw new Error(`Missing required element: ${selector}`);
    }
    return element;
}
export function isMobileViewport() {
    return window.matchMedia("(max-width: 720px)").matches;
}
export const sidebarToggle = query("#sidebarToggle");
export const viewerToggle = query("#viewerToggle");
export const refreshWorkspaceButton = query("#refreshWorkspaceButton");
export const openInTabButton = query("#openInTabButton");
export const downloadFileButton = query("#downloadFileButton");
export const workspaceSettingsButton = query("#workspaceSettingsButton");
export const refreshContextEvalButton = query("#refreshContextEvalButton");
export const newChatButton = query("#newChatButton");
export const conversationList = query("#conversationList");
export const connectionBadge = query("#connectionBadge");
export const chatMessages = query("#chatMessages");
export const scrollToBottomButton = query("#scrollToBottomButton");
export const composerForm = query("#composerForm");
export const composerAttachments = query("#composerAttachments");
export const composerInput = query("#composerInput");
export const composerHint = query("#composerHint");
export const sendButton = query("#sendButton");
export const viewerTitle = query("#viewerTitle");
export const viewerMeta = query("#viewerMeta");
export const viewerModeButton = query("#viewerModeButton");
export const viewerCloseButton = query("#viewerCloseButton");
export const directoryPath = query("#directoryPath");
export const fileList = query("#fileList");
export const filePreview = query("#filePreview");
export const contextEvalReport = query("#contextEvalReport");
export const settingsOverlay = query("#settingsOverlay");
export const closeSettingsButton = query("#closeSettingsButton");
export const settingsSummary = query("#settingsSummary");
export const fixtureReviewFilter = query("#fixtureReviewFilter");
export const refreshFixtureReviewButton = query("#refreshFixtureReviewButton");
export const fixtureReviewList = query("#fixtureReviewList");
export const fixtureReviewDetail = query("#fixtureReviewDetail");
export const resetAppButton = query("#resetAppButton");


//# sourceURL=/Users/joe/dev/ai-chat/src/web/dom.ts