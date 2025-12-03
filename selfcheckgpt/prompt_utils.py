"""
Prompt cost/latency reduction scaffold.

Agents should extend this module with:
- Deterministic cache keys for (prompt, model, client_type, etc.)
- File-based or in-memory cache helpers
- Optional batching helpers for API/open-source prompt scorers
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import hashlib
import json


@dataclass
class PromptCacheConfig:
    cache_dir: Path
    enabled: bool = False

    @classmethod
    def from_kwargs(
        cls,
        *,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = False,
    ) -> "PromptCacheConfig":
        cache_path = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        return cls(cache_dir=cache_path, enabled=use_cache)

    def override(
        self,
        *,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: Optional[bool] = None,
    ) -> "PromptCacheConfig":
        cache_path = Path(cache_dir) if cache_dir is not None else self.cache_dir
        enabled = self.enabled if use_cache is None else bool(use_cache)
        return PromptCacheConfig(cache_dir=cache_path, enabled=enabled)


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


def load_cached_json(
    cache_config: PromptCacheConfig,
    payload: Dict[str, Any],
) -> Optional[Any]:
    """
    Attempt to load a JSON payload from cache.
    """
    if not cache_config.enabled:
        return None
    key = make_cache_key(payload)
    cached = read_cache(cache_config.cache_dir, key)
    if cached is None:
        return None
    try:
        return json.loads(cached)
    except json.JSONDecodeError:
        return None


def store_cached_json(
    cache_config: PromptCacheConfig,
    payload: Dict[str, Any],
    value: Any,
) -> None:
    """
    Persist a JSON-serializable value into the cache.
    """
    if not cache_config.enabled:
        return
    key = make_cache_key(payload)
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=True)
    write_cache(cache_config.cache_dir, key, serialized)
