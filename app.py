#!/usr/bin/env python3
"""Entrypoint. Serves the lean helper app by default; LEGACY_HARNESS=1 serves the old workspace harness."""

import os
import sys

if os.getenv("LEGACY_HARNESS", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from src.python import harness as _module
    _app = _module.app
else:
    from src.python.helper import server as _module
    _app = _module.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(_app, **_module.build_uvicorn_run_kwargs())
else:
    sys.modules[__name__] = _module
