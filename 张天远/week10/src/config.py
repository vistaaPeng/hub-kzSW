"""
Runtime configuration helpers.

Values can be supplied through environment variables or a local .env file.
The checked-in defaults match the original local development setup so existing
machines keep working while new environments can override paths cleanly.
"""

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding env vars."""
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


HF_ENDPOINT = env_str("RAG_HF_ENDPOINT", "https://hf-mirror.com")
HF_CACHE_DIR = env_str("RAG_HF_CACHE_DIR", "M:/huggingface_cache")
HF_OFFLINE = env_bool("RAG_HF_OFFLINE", True)
