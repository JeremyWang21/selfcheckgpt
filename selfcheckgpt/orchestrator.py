"""
Fine-grained scoring orchestrator scaffold.

Intended flow (to be implemented by the assigned agent):
- Optionally decompose sentences into subspans (clauses/phrases).
- Score subspans with existing SelfCheckGPT scorers.
- Aggregate subspan scores back to sentence-level summaries.
"""

from typing import Any, Dict, List, Protocol

from .decomposition import simple_clause_split


class Scorer(Protocol):
    def predict(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> Any:
        ...


def score_with_decomposition(
    scorer: Scorer,
    sentences: List[str],
    sampled_passages: List[str],
    use_decomposition: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Placeholder aggregation: currently passes sentences directly.
    Agents should extend to decompose -> score -> aggregate.
    """
    if use_decomposition:
        expanded: List[str] = []
        mapping: List[int] = []
        for idx, sent in enumerate(sentences):
            chunks = simple_clause_split(sent)
            expanded.extend(chunks)
            mapping.extend([idx] * len(chunks))
        scores = scorer.predict(expanded, sampled_passages, **kwargs)
        return {"scores": scores, "mapping": mapping}
    else:
        scores = scorer.predict(sentences, sampled_passages, **kwargs)
        return {"scores": scores, "mapping": list(range(len(sentences)))}
