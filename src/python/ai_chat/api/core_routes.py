from __future__ import annotations

from typing import Any


def _register_startup_event(app: Any, startup_event) -> None:
    add_event_handler = getattr(app, "add_event_handler", None)
    if callable(add_event_handler):
        add_event_handler("startup", startup_event)
        return

    router = getattr(app, "router", None)
    router_add_event_handler = getattr(router, "add_event_handler", None)
    if callable(router_add_event_handler):
        router_add_event_handler("startup", startup_event)
        return

    router_on_startup = getattr(router, "on_startup", None)
    if hasattr(router_on_startup, "append"):
        router_on_startup.append(startup_event)
        return

    on_event = getattr(app, "on_event", None)
    if callable(on_event):
        on_event("startup")(startup_event)
        return

    raise AttributeError("App object does not support startup event registration")


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
    _register_startup_event(app, startup_event)
