// Generated from src/web/state.ts by scripts/build_frontend.mjs
export const state = {
    appTitle: document.body.dataset.appTitle || "AI Chat",
    ws: null,
    reconnectTimer: 0,
    connectionState: "offline",
    generating: false,
    modelAvailable: false,
    modelName: "",
    workspaces: [],
    currentWorkspaceId: localStorage.getItem("lastWorkspaceId") || "",
    conversations: [],
    currentConversationId: "",
    currentConversationTitle: "New chat",
    messages: [],
    activeAssistantIndex: -1,
    currentDirectoryPath: ".",
    fileItems: [],
    selectedFilePath: "",
    selectedFileContentKind: "",
    latestWorkedFilePath: "",
    leftSidebarOpen: false,
    viewerMode: "closed",
    contextEvalReport: null,
    contextEvalLoading: false,
    contextEvalError: "",
    selectedContextEvalBucketKey: "",
    selectedContextEvalExampleIndex: 0,
    contextEvalFixtures: [],
    fixtureReviewLoading: false,
    fixtureReviewFilter: "all",
    selectedFixturePath: "",
    compareFixturePath: "",
    selectedFixtureDetail: null,
    compareFixtureDetail: null,
    fixtureDetailLoading: false,
    thinkingStatus: null,
    stickChatToBottom: true,
    handledArtifactKeys: new Set(),
    conversationRefreshTimer: 0,
    healthPollTimer: 0,
    pendingAttachments: [],
    composerUploadingCount: 0,
    composerDropActive: false,
    activeStreamConversationId: "",
    activePreviewNonce: ""
};
export const PASTE_ATTACH_CHAR_THRESHOLD = 1200;
export const PASTE_ATTACH_LINE_THRESHOLD = 28;
export const FIRST_TURN_COMPOSER_MAX_HEIGHT = 260;
export const LATER_TURN_COMPOSER_MAX_HEIGHT = 152;


//# sourceURL=/Users/joe/dev/ai-chat/src/web/state.ts