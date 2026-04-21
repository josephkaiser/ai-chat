#!/usr/bin/env python3
"""Compatibility entrypoint that aliases the backend harness module."""

import sys

from src.python import harness as _harness


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(_harness.app, **_harness.build_uvicorn_run_kwargs())
else:
    sys.modules[__name__] = _harness
