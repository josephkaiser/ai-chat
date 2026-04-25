from __future__ import annotations

from typing import Any


def register_chat_routes(
    app: Any,
    *,
    get_conversations,
    submit_feedback,
    retry_message,
    chat_http,
    get_conversation,
    rename_conversation,
    delete_conversation,
    reset_all_application_data,
    search_chats,
    chat_websocket,
) -> None:
    app.add_api_route("/api/conversations", get_conversations, methods=["GET"])
    app.add_api_route("/api/message/{message_id}/feedback", submit_feedback, methods=["POST"])
    app.add_api_route("/api/message/{message_id}/retry", retry_message, methods=["POST"])
    app.add_api_route("/api/chat", chat_http, methods=["POST"])
    app.add_api_route("/api/conversation/{conversation_id}", get_conversation, methods=["GET"])
    app.add_api_route("/api/conversation/{conversation_id}/rename", rename_conversation, methods=["POST"])
    app.add_api_route("/api/conversation/{conversation_id}", delete_conversation, methods=["DELETE"])
    app.add_api_route("/api/reset-all", reset_all_application_data, methods=["POST"])
    app.add_api_route("/api/search", search_chats, methods=["GET"])
    app.add_api_websocket_route("/ws/chat", chat_websocket)
