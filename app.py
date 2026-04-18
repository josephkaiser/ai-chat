#!/usr/bin/env python3
"""Compatibility entrypoint for the Python harness.

Primary backend source now lives in `src/python/harness.py`.
"""

from src.python.harness import *  # noqa: F401,F403
from src.python.harness import app as fastapi_app
from src.python.harness import build_uvicorn_run_kwargs


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(fastapi_app, **build_uvicorn_run_kwargs())
