from __future__ import annotations

from typing import Any


def register_context_eval_routes(
    app: Any,
    *,
    get_context_eval_report,
    promote_context_eval_capture,
    auto_draft_context_eval_capture,
    list_context_eval_fixtures,
    get_context_eval_fixture_detail,
    review_context_eval_fixture,
) -> None:
    app.add_api_route("/api/context-evals/report", get_context_eval_report, methods=["GET"])
    app.add_api_route("/api/context-evals/promote", promote_context_eval_capture, methods=["POST"])
    app.add_api_route("/api/context-evals/auto-draft", auto_draft_context_eval_capture, methods=["POST"])
    app.add_api_route("/api/context-evals/fixtures", list_context_eval_fixtures, methods=["GET"])
    app.add_api_route("/api/context-evals/fixtures/detail", get_context_eval_fixture_detail, methods=["GET"])
    app.add_api_route("/api/context-evals/fixtures/review", review_context_eval_fixture, methods=["POST"])
