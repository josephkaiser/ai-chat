from __future__ import annotations

from typing import Any


def register_workspace_routes(
    app: Any,
    *,
    list_files,
    read_file_content,
    get_workspaces,
    create_workspace,
    get_workspace_catalog_entry,
    rename_workspace,
    delete_workspace,
    list_workspace_files_by_workspace,
    read_workspace_file_by_workspace,
    list_file_sessions,
    ensure_file_session,
    set_file_session_focus_for_workspace,
    delete_file_session_for_workspace,
    get_file_session_bundle_for_workspace,
    list_file_session_jobs_for_workspace,
    create_file_session_job_for_workspace,
    update_file_session_job_for_workspace,
    view_workspace_file_by_workspace,
    render_workspace_file_by_workspace,
    download_workspace_file_by_workspace,
    write_workspace_file_by_workspace,
    read_workspace_spreadsheet_by_workspace,
    upload_workspace_files_by_workspace,
    extract_workspace_archive_by_workspace,
    download_workspace_by_workspace,
    get_workspace_info,
    upload_workspace_files,
    extract_workspace_archive,
    list_workspace_files,
    read_workspace_file,
    view_workspace_file,
    render_workspace_file,
    download_workspace_file,
    write_workspace_file,
    read_workspace_spreadsheet,
    download_workspace,
) -> None:
    app.add_api_route("/api/files/list", list_files, methods=["GET"])
    app.add_api_route("/api/files/read", read_file_content, methods=["GET"])
    app.add_api_route("/api/workspaces", get_workspaces, methods=["GET"])
    app.add_api_route("/api/workspaces", create_workspace, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}", get_workspace_catalog_entry, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/rename", rename_workspace, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}", delete_workspace, methods=["DELETE"])

    app.add_api_route("/api/workspaces/{workspace_id}/files", list_workspace_files_by_workspace, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/file", read_workspace_file_by_workspace, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/file-sessions", list_file_sessions, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/file-sessions/ensure", ensure_file_session, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}/file-sessions/focus", set_file_session_focus_for_workspace, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}/file-sessions/{file_session_id}", delete_file_session_for_workspace, methods=["DELETE"])
    app.add_api_route("/api/workspaces/{workspace_id}/file-sessions/{file_session_id}", get_file_session_bundle_for_workspace, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/file-sessions/{file_session_id}/jobs", list_file_session_jobs_for_workspace, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/file-session-jobs", create_file_session_job_for_workspace, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}/file-session-jobs/{job_id}/status", update_file_session_job_for_workspace, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}/file/view", view_workspace_file_by_workspace, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/file/render/{path:path}", render_workspace_file_by_workspace, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/file/download", download_workspace_file_by_workspace, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/file", write_workspace_file_by_workspace, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}/spreadsheet", read_workspace_spreadsheet_by_workspace, methods=["GET"])
    app.add_api_route("/api/workspaces/{workspace_id}/upload", upload_workspace_files_by_workspace, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}/archive/extract", extract_workspace_archive_by_workspace, methods=["POST"])
    app.add_api_route("/api/workspaces/{workspace_id}/download", download_workspace_by_workspace, methods=["GET"])

    app.add_api_route("/api/workspace/{conversation_id}", get_workspace_info, methods=["GET"])
    app.add_api_route("/api/workspace/{conversation_id}/upload", upload_workspace_files, methods=["POST"])
    app.add_api_route("/api/workspace/{conversation_id}/archive/extract", extract_workspace_archive, methods=["POST"])
    app.add_api_route("/api/workspace/{conversation_id}/files", list_workspace_files, methods=["GET"])
    app.add_api_route("/api/workspace/{conversation_id}/file", read_workspace_file, methods=["GET"])
    app.add_api_route("/api/workspace/{conversation_id}/file/view", view_workspace_file, methods=["GET"])
    app.add_api_route("/api/workspace/{conversation_id}/file/render/{path:path}", render_workspace_file, methods=["GET"])
    app.add_api_route("/api/workspace/{conversation_id}/file/download", download_workspace_file, methods=["GET"])
    app.add_api_route("/api/workspace/{conversation_id}/file", write_workspace_file, methods=["POST"])
    app.add_api_route("/api/workspace/{conversation_id}/spreadsheet", read_workspace_spreadsheet, methods=["GET"])
    app.add_api_route("/api/workspace/{conversation_id}/download", download_workspace, methods=["GET"])
