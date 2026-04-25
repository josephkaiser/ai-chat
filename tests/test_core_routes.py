import unittest

from src.python.ai_chat.api.core_routes import register_core_routes


class _RouterWithAddEventHandler:
    def __init__(self):
        self.calls = []

    def add_event_handler(self, event_name, handler):
        self.calls.append((event_name, handler))


class _AppWithAddEventHandler:
    def __init__(self):
        self.exception_handlers = []
        self.routes = []
        self.event_handlers = []

    def add_exception_handler(self, exc_type, handler):
        self.exception_handlers.append((exc_type, handler))

    def add_api_route(self, path, handler, methods):
        self.routes.append((path, handler, methods))

    def add_event_handler(self, event_name, handler):
        self.event_handlers.append((event_name, handler))


class _AppWithRouterEventHandler:
    def __init__(self):
        self.exception_handlers = []
        self.routes = []
        self.router = _RouterWithAddEventHandler()

    def add_exception_handler(self, exc_type, handler):
        self.exception_handlers.append((exc_type, handler))

    def add_api_route(self, path, handler, methods):
        self.routes.append((path, handler, methods))


class _AppWithOnEvent:
    def __init__(self):
        self.exception_handlers = []
        self.routes = []
        self.decorated = []

    def add_exception_handler(self, exc_type, handler):
        self.exception_handlers.append((exc_type, handler))

    def add_api_route(self, path, handler, methods):
        self.routes.append((path, handler, methods))

    def on_event(self, event_name):
        def decorator(handler):
            self.decorated.append((event_name, handler))
            return handler

        return decorator


class _AppWithRouterStartupList:
    def __init__(self):
        self.exception_handlers = []
        self.routes = []
        self.router = type("Router", (), {"on_startup": []})()

    def add_exception_handler(self, exc_type, handler):
        self.exception_handlers.append((exc_type, handler))

    def add_api_route(self, path, handler, methods):
        self.routes.append((path, handler, methods))


class CoreRoutesTests(unittest.TestCase):
    def test_register_core_routes_uses_app_add_event_handler_when_available(self):
        app = _AppWithAddEventHandler()

        def startup():
            return None

        register_core_routes(
            app,
            global_exception_handler=lambda *_args: None,
            home=lambda *_args: None,
            health=lambda *_args: None,
            startup_event=startup,
        )

        self.assertEqual(app.event_handlers, [("startup", startup)])

    def test_register_core_routes_falls_back_to_router_add_event_handler(self):
        app = _AppWithRouterEventHandler()

        def startup():
            return None

        register_core_routes(
            app,
            global_exception_handler=lambda *_args: None,
            home=lambda *_args: None,
            health=lambda *_args: None,
            startup_event=startup,
        )

        self.assertEqual(app.router.calls, [("startup", startup)])

    def test_register_core_routes_falls_back_to_on_event(self):
        app = _AppWithOnEvent()

        def startup():
            return None

        register_core_routes(
            app,
            global_exception_handler=lambda *_args: None,
            home=lambda *_args: None,
            health=lambda *_args: None,
            startup_event=startup,
        )

        self.assertEqual(app.decorated, [("startup", startup)])

    def test_register_core_routes_falls_back_to_router_startup_list(self):
        app = _AppWithRouterStartupList()

        def startup():
            return None

        register_core_routes(
            app,
            global_exception_handler=lambda *_args: None,
            home=lambda *_args: None,
            health=lambda *_args: None,
            startup_event=startup,
        )

        self.assertEqual(app.router.on_startup, [startup])


if __name__ == "__main__":
    unittest.main()
