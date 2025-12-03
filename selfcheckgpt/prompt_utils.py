"""
Prompt cost/latency reduction scaffold.

Agents should extend this module with:
- Deterministic cache keys for (prompt, model, client_type, etc.)
- File-based or in-memory cache helpers
- Optional batching helpers for API/open-source prompt scorers
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json


@dataclass
class PromptCacheConfig:
    cache_dir: Path
    enabled: bool = False


def default_cache_dir() -> Path:
    return Path(".cache/selfcheckgpt")


def make_cache_key(payload: Dict[str, Any]) -> str:
    """
    Create a stable cache key from a prompt payload.
    """
    norm = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def read_cache(cache_dir: Path, key: str) -> Optional[str]:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_cache(cache_dir: Path, key: str, value: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(value, encoding="utf-8")
