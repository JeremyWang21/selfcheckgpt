"""
Pluggable dataset loader scaffold for multi-domain evaluation.

Agents should extend the registry below with concrete loaders for events,
places, organizations, etc. Loaders should be lightweight and avoid
network access by default (or make it optional).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol


class Example(Mapping):
    """
    Minimal example protocol for downstream scorers.
    Required keys: 'id', 'prompt', 'reference' (ground truth/context), 'samples' (list of model outputs).
    """

    id: str
    prompt: str
    reference: str
    samples: List[str]

    def __getitem__(self, key):
        raise NotImplementedError

    def __iter__(self) -> Iterable[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Loader(Protocol):
    def __call__(self, split: str = "train", limit: Optional[int] = None) -> List[Example]:
        ...


LOADER_REGISTRY: Dict[str, Loader] = {}


@dataclass
class _DomainConfig:
    env_local_path: str
    env_hf_dataset: str
    env_hf_subset: str
    default_dir: Path


@dataclass
class _LoaderOverride:
    local_path: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_subset: Optional[str] = None


_LOADER_OVERRIDES: Dict[str, _LoaderOverride] = {}


def register_loader(name: str, loader: Loader) -> None:
    """
    Register a loader under a domain name.
    """
    LOADER_REGISTRY[name] = loader


def get_loader(name: str) -> Loader:
    if name not in LOADER_REGISTRY:
        raise KeyError(f"Loader '{name}' not found. Available: {list(LOADER_REGISTRY)}")
    return LOADER_REGISTRY[name]


def list_loaders() -> List[str]:
    return sorted(LOADER_REGISTRY)


def load(domain: str, split: str = "train", limit: Optional[int] = None) -> List[Example]:
    """
    Convenience wrapper to fetch examples.
    """
    loader = get_loader(domain)
    return loader(split=split, limit=limit)


def configure_loader(
    name: str,
    *,
    local_path: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_subset: Optional[str] = None,
) -> None:
    """
    Configure runtime overrides for a loader.

    Useful for demos/CLIs that expose custom JSONL files or Hugging Face datasets
    without mutating the registry itself.
    """
    override = _LOADER_OVERRIDES.get(name, _LoaderOverride())
    if local_path is not None:
        override.local_path = local_path
    if hf_dataset is not None:
        override.hf_dataset = hf_dataset
    if hf_subset is not None:
        override.hf_subset = hf_subset
    _LOADER_OVERRIDES[name] = override


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_multidomain_dir() -> Path:
    return _project_root() / "demo" / "experiments" / "multidomain" / "data"


def _default_domain_configs() -> Dict[str, _DomainConfig]:
    base_dir = _default_multidomain_dir()
    return {
        "events": _DomainConfig(
            env_local_path="SCG_EVENTS_PATH",
            env_hf_dataset="SCG_EVENTS_HF_DATASET",
            env_hf_subset="SCG_EVENTS_HF_SUBSET",
            default_dir=base_dir / "events",
        ),
        "places": _DomainConfig(
            env_local_path="SCG_PLACES_PATH",
            env_hf_dataset="SCG_PLACES_HF_DATASET",
            env_hf_subset="SCG_PLACES_HF_SUBSET",
            default_dir=base_dir / "places",
        ),
        "organizations": _DomainConfig(
            env_local_path="SCG_ORGS_PATH",
            env_hf_dataset="SCG_ORGS_HF_DATASET",
            env_hf_subset="SCG_ORGS_HF_SUBSET",
            default_dir=base_dir / "organizations",
        ),
    }


def _loader_override(name: str) -> _LoaderOverride:
    return _LOADER_OVERRIDES.get(name, _LoaderOverride())


def _candidate_paths(base_path: Path, split: str) -> List[Path]:
    if base_path.is_file():
        return [base_path]
    candidates = [
        base_path / f"{split}.jsonl",
        base_path / f"{split}.json",
        base_path / "data.jsonl",
        base_path / "data.json",
    ]
    seen = set()
    ordered: List[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _normalize_example(record: Mapping[str, Any]) -> Example:
    required = {"id", "prompt", "reference", "samples"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"Example missing keys {missing}: {record}")
    sample_list = record["samples"]
    if not isinstance(sample_list, list):
        raise ValueError(f"'samples' must be a list: {record}")
    normalized_samples = [str(sample) for sample in sample_list]
    return {
        "id": str(record["id"]),
        "prompt": str(record["prompt"]),
        "reference": str(record["reference"]),
        "samples": normalized_samples,
    }


def _load_examples_from_path(path: Path, limit: Optional[int]) -> List[Example]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records: List[Mapping[str, Any]] = []
    if path.suffix == ".json":
        records = json.loads(path.read_text())
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    examples = [_normalize_example(record) for record in records]
    if limit is not None:
        return examples[:limit]
    return examples


def _load_examples_from_hf(
    dataset_id: str,
    split: str,
    limit: Optional[int],
    subset: Optional[str] = None,
) -> List[Example]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is not installed. "
            "Install it with `pip install datasets` or use a local JSONL file."
        ) from exc
    dataset = load_dataset(dataset_id, subset, split=split)
    records: List[Mapping[str, Any]] = []
    for row in dataset:
        records.append(dict(row))
        if limit is not None and len(records) >= limit:
            break
    return [_normalize_example(record) for record in records]


def _resolve_local_path(
    domain: str,
    split: str,
    config: _DomainConfig,
) -> Optional[Path]:
    override = _loader_override(domain)
    path_candidates: List[Path] = []
    if override.local_path:
        path_candidates.extend(_candidate_paths(Path(override.local_path).expanduser(), split))
    env_path = os.environ.get(config.env_local_path)
    if env_path:
        path_candidates.extend(_candidate_paths(Path(env_path).expanduser(), split))
    path_candidates.extend(_candidate_paths(config.default_dir, split))
    for candidate in path_candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_hf_dataset(domain: str, config: _DomainConfig) -> Optional[str]:
    override = _loader_override(domain)
    if override.hf_dataset:
        return override.hf_dataset
    return os.environ.get(config.env_hf_dataset)


def _resolve_hf_subset(domain: str, config: _DomainConfig) -> Optional[str]:
    override = _loader_override(domain)
    if override.hf_subset:
        return override.hf_subset
    return os.environ.get(config.env_hf_subset)


def _jsonl_loader_factory(domain: str, config: _DomainConfig) -> Loader:
    def _loader(split: str = "train", limit: Optional[int] = None) -> List[Example]:
        local_path = _resolve_local_path(domain, split, config)
        if local_path:
            return _load_examples_from_path(local_path, limit)
        dataset_id = _resolve_hf_dataset(domain, config)
        if dataset_id:
            subset = _resolve_hf_subset(domain, config)
            return _load_examples_from_hf(dataset_id, split, limit, subset=subset)
        raise FileNotFoundError(
            f"No data source configured for domain '{domain}' and split '{split}'. "
            "Provide a local JSONL via configure_loader(), environment variable, or --*-path flag."
        )

    return _loader


def _register_default_domain_loaders() -> None:
    for domain, cfg in _default_domain_configs().items():
        if domain not in LOADER_REGISTRY:
            register_loader(domain, _jsonl_loader_factory(domain, cfg))


_register_default_domain_loaders()
