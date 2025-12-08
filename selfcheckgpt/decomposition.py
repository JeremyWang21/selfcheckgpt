"""
Sentence decomposition utilities used by fine-grained scoring flows.

When spaCy and the requested language model are available we try to split
input sentences into clause/phrase-like segments using dependency cues.
If spaCy (or the model) is missing we fall back to a lightweight heuristic
splitter so downstream code always receives at least one chunk.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Optional, Tuple, TYPE_CHECKING

try:
    import spacy
except ImportError:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spacy.language import Language
    from spacy.tokens import Doc

LOGGER = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = "en_core_web_sm"
CLAUSE_BOUNDARY_PUNCT = {",", ";", ":", "—", "–", "--"}
COORDINATING_CONJ = {"and", "but", "or", "so", "yet"}
CLAUSAL_DEP_LABELS = {"advcl", "ccomp", "conj", "pcomp", "acl", "relcl", "xcomp", "parataxis"}
FALLBACK_PATTERN = re.compile(
    r"(?:,|;|:|--|—|–|\.|\?|!|\b(?:and|but|or|so|yet)\b)", re.IGNORECASE
)


def simple_clause_split(sentence: str, model_name: str = DEFAULT_SPACY_MODEL) -> List[str]:
    """
    Split a sentence into clause/phrase-like sub spans.

    Args:
        sentence: Raw sentence string to split.
        model_name: spaCy model name to use when available.

    Returns:
        Ordered list of clause-like chunks (at least one entry when text exists).
    """
    sentence = sentence.strip()
    if not sentence:
        return []

    doc = _build_doc(sentence, model_name)
    if doc is not None:
        chunks = _spacy_clause_chunks(doc)
        if chunks:
            return chunks

    return _heuristic_split(sentence)


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str = DEFAULT_SPACY_MODEL) -> Optional["Language"]:
    if spacy is None:
        LOGGER.debug("spaCy is not installed; falling back to heuristic clause splitting.")
        return None
    try:
        return spacy.load(model_name, disable=("ner",))
    except OSError:
        LOGGER.warning(
            "spaCy model '%s' is not available. "
            "Install it via `python -m spacy download %s` for clause-level splitting.",
            model_name,
            model_name,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to load spaCy model '%s': %s", model_name, exc)
    return None


def _build_doc(sentence: str, model_name: str) -> Optional["Doc"]:
    nlp = _load_spacy_model(model_name)
    if nlp is None:
        return None
    try:
        return nlp(sentence)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("spaCy parsing failed (%s); falling back to heuristics.", exc)
        return None


def _spacy_clause_chunks(doc: "Doc") -> List[str]:
    if not doc:
        return []

    boundaries = [0]
    for token in doc:
        idx_after = token.i + 1
        if token.text in CLAUSE_BOUNDARY_PUNCT:
            boundaries.append(idx_after)
        elif token.pos_ == "CCONJ" and token.lower_ in COORDINATING_CONJ:
            boundaries.append(token.i)
        elif token.dep_ in CLAUSAL_DEP_LABELS and token.i > boundaries[-1]:
            boundaries.append(token.i)

    boundaries.append(len(doc))
    normalized = sorted({max(0, min(len(doc), value)) for value in boundaries})

    chunks = [
        doc[start:end].text.strip()
        for start, end in zip(normalized, normalized[1:])
        if end - start > 0 and doc[start:end].text.strip()
    ]
    if len(chunks) > 1:
        return chunks

    # If we did not find multiple clause spans, attempt noun/verb chunk based fallback.
    noun_verb_chunks = _noun_and_verb_chunks(doc)
    if noun_verb_chunks:
        return noun_verb_chunks

    return [doc.text.strip()] if doc.text.strip() else []


def _heuristic_split(sentence: str) -> List[str]:
    parts = [part.strip() for part in FALLBACK_PATTERN.split(sentence) if part and part.strip()]
    if parts:
        return parts
    return [sentence]


def _noun_and_verb_chunks(doc: "Doc") -> List[str]:
    spans: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()

    if doc.has_annotation("DEP"):
        try:
            for chunk in doc.noun_chunks:
                _append_span(spans, seen, chunk.start, chunk.end)
        except ValueError:
            # Noun chunks unavailable if the parser is missing; ignore silently.
            pass

    for token in doc:
        if token.pos_ in {"VERB", "AUX"}:
            subtree = list(token.subtree)
            if not subtree:
                continue
            start = subtree[0].i
            end = subtree[-1].i + 1
            _append_span(spans, seen, start, end)

    spans.sort(key=lambda rng: (rng[0], rng[1]))
    chunks = [doc[start:end].text.strip() for start, end in spans if doc[start:end].text.strip()]
    return chunks


def _append_span(
    spans: List[Tuple[int, int]],
    seen: set[Tuple[int, int]],
    start: int,
    end: int,
) -> None:
    if start >= end:
        return
    key = (start, end)
    if key in seen:
        return
    seen.add(key)
    spans.append(key)
