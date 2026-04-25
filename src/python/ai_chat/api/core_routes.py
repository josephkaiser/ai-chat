from __future__ import annotations

from typing import Any


def register_core_routes(
    app: Any,
    *,
    global_exception_handler,
    home,
    health,
    startup_event,
) -> None:
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_api_route("/", home, methods=["GET"])
    app.add_api_route("/health", health, methods=["GET"])
    app.add_event_handler("startup", startup_event)
