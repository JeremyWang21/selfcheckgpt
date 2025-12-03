"""
Pluggable dataset loader scaffold for multi-domain evaluation.

Agents should extend the registry below with concrete loaders for events,
places, organizations, etc. Loaders should be lightweight and avoid
network access by default (or make it optional).
"""

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol


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
