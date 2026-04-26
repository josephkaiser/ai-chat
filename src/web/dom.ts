function query<T extends Element>(selector: string): T {
    const element = document.querySelector<T>(selector);
    if (!element) {
        throw new Error(`Missing required element: ${selector}`);
    }
    return element;
}

export function isMobileViewport(): boolean {
    return window.matchMedia("(max-width: 720px)").matches;
}

export const sidebarToggle = query<HTMLButtonElement>("#sidebarToggle");
export const viewerToggle = query<HTMLButtonElement>("#viewerToggle");
export const refreshWorkspaceButton = query<HTMLButtonElement>("#refreshWorkspaceButton");
export const openInTabButton = query<HTMLButtonElement>("#openInTabButton");
export const downloadFileButton = query<HTMLButtonElement>("#downloadFileButton");
export const workspaceSettingsButton = query<HTMLButtonElement>("#workspaceSettingsButton");
export const refreshContextEvalButton = query<HTMLButtonElement>("#refreshContextEvalButton");
export const newChatButton = query<HTMLButtonElement>("#newChatButton");
export const conversationList = query<HTMLDivElement>("#conversationList");
export const connectionBadge = query<HTMLSpanElement>("#connectionBadge");
export const chatMessages = query<HTMLDivElement>("#chatMessages");
export const scrollToBottomButton = query<HTMLButtonElement>("#scrollToBottomButton");
export const composerForm = query<HTMLFormElement>("#composerForm");
export const composerAttachments = query<HTMLDivElement>("#composerAttachments");
export const composerInput = query<HTMLTextAreaElement>("#composerInput");
export const composerHint = query<HTMLSpanElement>("#composerHint");
export const sendButton = query<HTMLButtonElement>("#sendButton");
export const viewerTitle = query<HTMLHeadingElement>("#viewerTitle");
export const viewerMeta = query<HTMLParagraphElement>("#viewerMeta");
export const viewerModeButton = query<HTMLButtonElement>("#viewerModeButton");
export const viewerCloseButton = query<HTMLButtonElement>("#viewerCloseButton");
export const directoryPath = query<HTMLDivElement>("#directoryPath");
export const fileList = query<HTMLDivElement>("#fileList");
export const filePreview = query<HTMLDivElement>("#filePreview");
export const contextEvalReport = query<HTMLDivElement>("#contextEvalReport");
export const settingsOverlay = query<HTMLDivElement>("#settingsOverlay");
export const closeSettingsButton = query<HTMLButtonElement>("#closeSettingsButton");
export const settingsSummary = query<HTMLDivElement>("#settingsSummary");
export const reasoningOverlay = query<HTMLDivElement>("#reasoningOverlay");
export const closeReasoningButton = query<HTMLButtonElement>("#closeReasoningButton");
export const reasoningTitle = query<HTMLElement>("#reasoningTitle");
export const reasoningScope = query<HTMLParagraphElement>("#reasoningScope");
export const reasoningContent = query<HTMLDivElement>("#reasoningContent");
export const fixtureReviewFilter = query<HTMLSelectElement>("#fixtureReviewFilter");
export const refreshFixtureReviewButton = query<HTMLButtonElement>("#refreshFixtureReviewButton");
export const fixtureReviewList = query<HTMLDivElement>("#fixtureReviewList");
export const fixtureReviewDetail = query<HTMLDivElement>("#fixtureReviewDetail");
export const resetAppButton = query<HTMLButtonElement>("#resetAppButton");
