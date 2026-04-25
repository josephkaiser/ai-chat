from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import time
from dataclasses import dataclass
from logging import Logger


@dataclass(frozen=True)
class FrontendAssets:
    repo_root: pathlib.Path
    web_root: pathlib.Path
    frontend_source_path: pathlib.Path
    frontend_bundle_path: pathlib.Path
    frontend_build_script_path: pathlib.Path
    frontend_package_path: pathlib.Path


def compute_static_asset_version(web_root: pathlib.Path) -> str:
    """Build a cheap cache-busting token from local static asset mtimes."""
    hasher = hashlib.sha1()
    for rel_path in ("index.html", "app.js", "style.css"):
        target = web_root / rel_path
        try:
            stat = target.stat()
        except OSError:
            continue
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    digest = hasher.hexdigest()[:12]
    return digest or str(int(time.time()))


def frontend_bundle_is_stale(assets: FrontendAssets) -> bool:
    """Return True when the generated browser bundle should be rebuilt."""
    try:
        bundle_stat = assets.frontend_bundle_path.stat()
    except OSError:
        return True
    for candidate in (
        assets.frontend_source_path,
        assets.frontend_build_script_path,
        assets.frontend_package_path,
    ):
        try:
            if candidate.stat().st_mtime_ns > bundle_stat.st_mtime_ns:
                return True
        except OSError:
            continue
    return False


def ensure_frontend_bundle(assets: FrontendAssets, logger: Logger) -> bool:
    """Build the generated frontend bundle when local sources are newer."""
    if not frontend_bundle_is_stale(assets):
        return False

    node_bin = shutil.which("node")
    if not node_bin:
        if assets.frontend_bundle_path.is_file():
            logger.warning(
                "Frontend sources are newer than %s, but node is unavailable; serving the existing bundle.",
                assets.frontend_bundle_path,
            )
            return False
        raise RuntimeError(
            "Frontend bundle is missing and node is unavailable. Run `npm run build:frontend` first."
        )

    command = [node_bin, str(assets.frontend_build_script_path)]
    try:
        result = subprocess.run(
            command,
            cwd=str(assets.repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(f"Frontend build failed: {detail}") from exc

    build_output = (result.stdout or "").strip()
    logger.info(build_output or "Built frontend bundle from %s", assets.frontend_source_path)
    return True
