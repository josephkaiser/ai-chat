export interface ConversationSummary {
    id: string;
    title: string;
    updated_at?: string;
    last_message?: string;
    seed_message?: string;
    workspace_id?: string;
}

export interface WorkspaceSummary {
    id: string;
    display_name: string;
    root_path?: string;
}

export interface ChatMessage {
    id?: string;
    role: "user" | "assistant" | "system";
    content: string;
    timestamp?: string;
    error?: boolean;
    reasoning_notes?: ReasoningNote[];
}

export interface ReasoningNote {
    text: string;
    phase?: string;
    step_label?: string;
    timestamp?: string;
}

export interface WorkspaceItem {
    name: string;
    path: string;
    type: "file" | "directory";
    size?: number | null;
    modified_at?: string;
    content_kind?: string;
    kind?: string;
}

export interface WorkspaceFilePayload {
    path: string;
    content: string;
    content_kind: string;
    editable?: boolean;
    default_view?: string;
    title?: string;
    file_type?: string;
    extractor?: string;
    page_count?: number | null;
    chunk_count?: number | null;
    line_count?: number | null;
    opening_preview?: string;
    summary?: string;
    section_titles?: string[];
    preview_chunks?: Array<{
        chunk_index: number;
        page_start?: number | null;
        page_end?: number | null;
        section_title?: string;
        text?: string;
    }>;
    entries?: Array<{
        path: string;
        is_dir: boolean;
        size: number;
        compressed_size: number;
    }>;
    media_type?: string;
}

export interface UploadedWorkspaceFile {
    name: string;
    path: string;
    size?: number | null;
    content_type?: string;
    kind?: string;
}

export interface WorkspaceUploadResponse {
    workspace_id: string;
    workspace_path: string;
    target_path: string;
    files: UploadedWorkspaceFile[];
    count: number;
    conversation_id?: string;
}

export interface PendingComposerAttachment {
    name: string;
    path: string;
    size?: number | null;
    contentType?: string;
    kind?: string;
}

export interface SpreadsheetPreviewPayload {
    path: string;
    file_type: string;
    sheet?: string;
    sheet_names?: string[];
    workbook_sheets?: Array<{
        name: string;
        row_count: number;
        column_count: number;
        columns: string[];
    }>;
    row_count: number;
    column_count: number;
    columns: Array<string | {
        name: string;
        dtype?: string;
        non_null?: number;
        nulls?: number;
        sample_values?: unknown[];
        stats?: Record<string, unknown>;
    }>;
    preview_rows?: Array<Record<string, unknown>>;
}

export interface ChatEvent {
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

export interface ThinkingStatus {
    text: string;
    phase: string;
    error: boolean;
    persistent: boolean;
}

export interface KatexRenderOptions {
    displayMode?: boolean;
    throwOnError?: boolean;
    strict?: string;
}

export interface KatexNamespace {
    renderToString(source: string, options?: KatexRenderOptions): string;
}

export interface ContextEvalRecentFailure {
    source_path: string;
    name: string;
    score: number;
    trigger: string;
    failed_checks: string[];
    selected_keys: string[];
}

export interface ContextEvalExampleCase {
    name: string;
    source_path: string;
    trigger: string;
    phase: string;
    failed_checks: string[];
    selected_keys: string[];
}

export interface ContextEvalTriageBucket {
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

export interface ContextEvalReport {
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

export interface ContextEvalPromotionResponse {
    status: string;
    created?: boolean;
    fixture_name: string;
    fixture_path: string;
    source_path: string;
    review_status: string;
}

export interface ContextEvalFixtureRecord {
    name: string;
    path: string;
    review_status: string;
    promoted_at: string;
    updated_at: string;
    source_path: string;
    source_trigger: string;
    source_conversation_id: string;
}

export interface ContextEvalFixtureListResponse {
    fixtures: ContextEvalFixtureRecord[];
}

export interface ContextEvalFixtureDetailRecord extends ContextEvalFixtureRecord {
    payload: Record<string, unknown>;
}

export interface ContextEvalFixtureDetailResponse extends ContextEvalFixtureDetailRecord {}

export interface ContextEvalFixtureReviewResponse {
    status: string;
    fixture_name: string;
    fixture_path: string;
    review_status: string;
}

export type ConnectionState = "offline" | "online" | "streaming";
export type ViewerMode = "closed" | "tree" | "file";
