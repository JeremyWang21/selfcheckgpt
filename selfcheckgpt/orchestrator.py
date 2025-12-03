"""
Fine-grained scoring orchestrator.

This module exposes utilities that:
- Optionally decompose sentences into clause/phrase sub-spans.
- Score sub-spans with existing SelfCheckGPT scorers (unchanged internals).
- Aggregate sub-span scores back to sentence-level representations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from .decomposition import simple_clause_split

LOGGER = logging.getLogger(__name__)

Aggregation = Union[str, Callable[[List[float]], float]]


class Scorer(Protocol):
    def predict(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> Any:
        ...


def score_with_decomposition(
    scorer: Scorer,
    sentences: List[str],
    sampled_passages: List[str],
    use_decomposition: bool = False,
    aggregation: Aggregation = "mean",
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a scorer with optional clause-level decomposition and aggregation.

    Args:
        scorer: Any object exposing a ``predict`` method compatible with the
            SelfCheckGPT scorers.
        sentences: List of sentence strings to evaluate.
        sampled_passages: Evidence passages (passed straight through).
        use_decomposition: When True, split each sentence into clauses before
            scoring and aggregate the outputs back to sentence level.
        aggregation: Aggregation strategy for chunk scores (``"mean"`` or
            ``"max"``) or a callable accepting a list of floats.
        **kwargs: Forwarded directly to the scorer.

    Returns:
        Dictionary containing aggregated sentence scores plus chunk metadata.
        When ``use_decomposition`` is False the return value mirrors the scorer
        output (apart from the added ``mapping`` key for compatibility).
    """
    if not use_decomposition:
        scores = scorer.predict(sentences, sampled_passages, **kwargs)
        return {
            "scores": scores,
            "mapping": list(range(len(sentences))),
            "chunks_by_sentence": [[sent] for sent in sentences],
            "aggregation": None,
        }

    decomposition: List[List[str]] = []
    expanded: List[str] = []
    mapping: List[int] = []

    for idx, sent in enumerate(sentences):
        chunks = simple_clause_split(sent)
        if not chunks:
            stripped = sent.strip()
            chunks = [stripped] if stripped else []

        decomposition.append(chunks)
        expanded.extend(chunks)
        mapping.extend([idx] * len(chunks))

    if not expanded:
        LOGGER.warning(
            "All sentences were empty after decomposition; falling back to scorer output."
        )
        scores = scorer.predict(sentences, sampled_passages, **kwargs)
        return {
            "scores": scores,
            "mapping": list(range(len(sentences))),
            "chunks_by_sentence": decomposition,
            "aggregation": None,
        }

    chunk_predictions = scorer.predict(expanded, sampled_passages, **kwargs)
    chunk_scores = _coerce_float_list(chunk_predictions)

    if len(chunk_scores) != len(expanded):
        raise ValueError(
            "Scorer returned %s chunk scores but %s chunks were provided."
            % (len(chunk_scores), len(expanded))
        )

    aggregation_fn, aggregation_label = _resolve_aggregation(aggregation)
    sentence_scores, chunk_scores_by_sentence = _aggregate_scores(
        chunk_scores, mapping, len(sentences), aggregation_fn
    )

    return {
        "scores": sentence_scores,
        "chunk_scores": chunk_scores,
        "chunks": expanded,
        "mapping": mapping,
        "chunks_by_sentence": decomposition,
        "chunk_scores_by_sentence": chunk_scores_by_sentence,
        "aggregation": aggregation_label,
    }


def _coerce_float_list(predictions: Any) -> List[float]:
    if predictions is None:
        raise TypeError("Scorer returned None; cannot aggregate scores.")

    if isinstance(predictions, (int, float)):
        return [float(predictions)]

    if isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes, dict)):
        try:
            return [float(value) for value in predictions]
        except (TypeError, ValueError) as exc:
            raise TypeError("Scorer returned a non-numeric sequence.") from exc

    if hasattr(predictions, "tolist"):
        return _coerce_float_list(predictions.tolist())

    raise TypeError(f"Cannot convert scorer output of type {type(predictions)} to floats.")


def _resolve_aggregation(
    aggregation: Aggregation,
) -> Tuple[Callable[[List[float]], float], str]:
    if callable(aggregation):
        name = getattr(aggregation, "__name__", "<custom>")

        def wrapper(values: List[float]) -> float:
            return float(aggregation(values))

        wrapper.__name__ = name  # type: ignore[attr-defined]
        return wrapper, name

    normalized = str(aggregation).lower()
    if normalized == "mean":

        def _mean(values: List[float]) -> float:
            return float(sum(values) / len(values))

        _mean.__name__ = "mean"  # type: ignore[attr-defined]
        return _mean, "mean"
    if normalized == "max":

        def _max(values: List[float]) -> float:
            return float(max(values))

        _max.__name__ = "max"  # type: ignore[attr-defined]
        return _max, "max"

    raise ValueError(f"Unsupported aggregation '{aggregation}'.")


def _aggregate_scores(
    chunk_scores: List[float],
    mapping: List[int],
    num_sentences: int,
    aggregation_fn: Callable[[List[float]], float],
) -> Tuple[List[Optional[float]], List[List[float]]]:
    per_sentence: List[List[float]] = [[] for _ in range(num_sentences)]
    for score, sent_idx in zip(chunk_scores, mapping):
        per_sentence[sent_idx].append(score)

    aggregated: List[Optional[float]] = []
    for chunk_scores_for_sentence in per_sentence:
        if chunk_scores_for_sentence:
            aggregated.append(aggregation_fn(chunk_scores_for_sentence))
        else:
            aggregated.append(None)
    return aggregated, per_sentence
