from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shlex


MODEL_DEFAULTS_FILE = pathlib.Path(__file__).resolve().parents[3] / "config" / "model-defaults.env"
MODEL_OVERRIDES_FILE = pathlib.Path(__file__).resolve().parents[3] / "config" / "model-overrides.local.env"
ORIGINAL_ENV_KEYS = set(os.environ.keys())

HOSTNAME_ADJECTIVES = (
    "Bright", "Clever", "Cosmic", "Daring", "Electric", "Golden", "Lucky",
    "Merry", "Nova", "Radiant", "Rocket", "Sonic", "Spark", "Starlit",
    "Sunny", "Swift", "Velvet", "Wild",
)
HOSTNAME_NOUNS = (
    "Anchor", "Atlas", "Beacon", "Camp", "Canvas", "Comet", "Cove", "Den",
    "Forge", "Garden", "Grove", "Harbor", "Hideout", "Lab", "Loft", "Nest",
    "Orbit", "Outpost", "Studio", "Trail",
)
HOSTNAME_SKIP_TOKENS = {
    "local", "localhost", "home", "host", "hostname", "node", "server", "srv",
    "desktop", "laptop", "computer", "machine", "mac", "macbook", "macbookpro",
    "macbook-air", "macmini", "imac", "pc", "pro", "air", "mini",
    "linux", "ubuntu", "debian", "fedora", "arch", "windows",
}


def load_env_defaults_file(path: pathlib.Path, *, allow_override: bool = False) -> None:
    """Populate environment variables from a simple KEY=VALUE defaults file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if allow_override:
            if key in ORIGINAL_ENV_KEYS:
                continue
            os.environ[key] = value.strip()
            continue
        if key in os.environ:
            continue
        os.environ[key] = value.strip()


def env_truthy(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable using a small truthy set."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def first_env_value(*names: str) -> str:
    """Return the first non-empty environment value from the provided keys."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def build_model_args_from_env(prefix: str) -> str:
    """Build one vLLM argument string from line-by-line env settings."""
    parts: list[str] = []

    gpu_memory_utilization = os.getenv(f"{prefix}_GPU_MEMORY_UTILIZATION", "").strip()
    max_model_len = os.getenv(f"{prefix}_MAX_MODEL_LEN", "").strip()
    max_num_seqs = os.getenv(f"{prefix}_MAX_NUM_SEQS", "").strip()
    quantization = os.getenv(f"{prefix}_QUANTIZATION", "").strip()
    swap_space = os.getenv(f"{prefix}_SWAP_SPACE", "").strip()
    extra_args = os.getenv(f"{prefix}_EXTRA_ARGS", "").strip()

    if gpu_memory_utilization:
        parts.extend(["--gpu-memory-utilization", gpu_memory_utilization])
    if max_model_len:
        parts.extend(["--max-model-len", max_model_len])
    if env_truthy(f"{prefix}_ENABLE_PREFIX_CACHING"):
        parts.append("--enable-prefix-caching")
    if max_num_seqs:
        parts.extend(["--max-num-seqs", max_num_seqs])
    if env_truthy(f"{prefix}_ENABLE_CHUNKED_PREFILL"):
        parts.append("--enable-chunked-prefill")
    if quantization:
        parts.extend(["--quantization", quantization])
    if env_truthy(f"{prefix}_TRUST_REMOTE_CODE"):
        parts.append("--trust-remote-code")
    if env_truthy(f"{prefix}_ENFORCE_EAGER"):
        parts.append("--enforce-eager")
    if swap_space:
        parts.extend(["--swap-space", swap_space])
    if extra_args:
        try:
            parts.extend(shlex.split(extra_args))
        except ValueError:
            parts.append(extra_args)

    return " ".join(parts)


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _title_case_token(token: str) -> str:
    """Render a hostname token as a human-readable title fragment."""
    if token.isupper() and len(token) <= 4:
        return token
    if token.isdigit():
        return token
    return token[:1].upper() + token[1:].lower()


def build_dynamic_app_title(hostname: str) -> str:
    """Turn a raw hostname into a friendlier, deterministic codename."""
    raw = (hostname or "").strip()
    if not raw:
        return "Lucky Harbor"

    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    adjective = HOSTNAME_ADJECTIVES[digest[0] % len(HOSTNAME_ADJECTIVES)]
    fallback_noun = HOSTNAME_NOUNS[digest[1] % len(HOSTNAME_NOUNS)]

    parts = [part for part in re.split(r"[^A-Za-z0-9]+", raw) if part]
    meaningful_parts = []
    numeric_parts = []
    for part in parts:
        lowered = part.lower()
        if part.isdigit():
            numeric_parts.append(part)
            continue
        if len(part) <= 2 or lowered in HOSTNAME_SKIP_TOKENS:
            continue
        if re.fullmatch(r"[a-f0-9]{6,}", lowered):
            continue
        meaningful_parts.append(_title_case_token(part))

    anchor = " ".join(meaningful_parts[:2]).strip() or fallback_noun
    suffix = numeric_parts[-1][-2:] if numeric_parts else f"{digest[2] % 100:02d}"
    return f"{adjective} {anchor} {suffix}"


def ensure_runtime_root(raw_path: str, fallback_relative: str, *, repo_root: pathlib.Path) -> pathlib.Path:
    """Resolve a writable runtime directory, falling back to repo-local data in dev/test."""
    candidate = pathlib.Path(raw_path).resolve()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except PermissionError:
        fallback = (repo_root / fallback_relative).resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def ensure_runtime_file_path(raw_path: str, fallback_relative: str, *, repo_root: pathlib.Path) -> pathlib.Path:
    """Resolve a writable runtime file path, falling back to repo-local data in dev/test."""
    candidate = pathlib.Path(raw_path).resolve()
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    except PermissionError:
        fallback = (repo_root / fallback_relative).resolve()
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback
