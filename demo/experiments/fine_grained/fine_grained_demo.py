#!/usr/bin/env python3
"""
Fine-grained clause decomposition demo.

This script keeps the scoring logic intentionally simple so you can inspect
how ``score_with_decomposition`` expands sentences into sub-spans, gathers
raw chunk scores, and aggregates them back to the sentence level.
"""

from __future__ import annotations

from typing import List

from selfcheckgpt.orchestrator import score_with_decomposition


class LengthRatioScorer:
    """
    Toy scorer used for the demo.

    Each clause receives a score based on its token count; shorter clauses yield
    lower "risk" scores, longer clauses trend higher. Real scorers can drop in
    without any modification to this orchestrator layer.
    """

    def predict(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> List[float]:
        scaling = len(sampled_passages) + 5  # keep the output in [0, 1]
        return [min(1.0, len(sentence.split()) / scaling) for sentence in sentences]


def run_demo() -> None:
    sentences = [
        "Saturn is the sixth planet from the sun, and it is known for its expansive ring system.",
        "Astronomers have catalogued at least 82 moons orbiting Saturn, but the number keeps changing as new observations arrive.",
        "The gas giant weighs about 100 kilograms.",
    ]
    sampled_passages = [
        "Saturn, the sixth planet from the Sun, is famous for its rings and dozens of moons.",
        "Researchers debate the exact moon count, though current totals exceed 80.",
    ]

    scorer = LengthRatioScorer()
    result = score_with_decomposition(
        scorer=scorer,
        sentences=sentences,
        sampled_passages=sampled_passages,
        use_decomposition=True,
        aggregation="mean",
    )

    print("=== Fine-grained scoring ===")
    print(f"Aggregation: {result['aggregation']}")
    for idx, sentence in enumerate(sentences):
        aggregated_score = result["scores"][idx]
        chunk_texts = result["chunks_by_sentence"][idx]
        chunk_scores = result["chunk_scores_by_sentence"][idx]

        print(f"\nSentence {idx + 1}: {sentence}")
        if aggregated_score is None:
            print("  Aggregated score: n/a (no textual content)")
        else:
            print(f"  Aggregated score: {aggregated_score:.3f}")

        if not chunk_texts:
            print("  (no chunks generated)")
            continue

        for chunk, chunk_score in zip(chunk_texts, chunk_scores):
            print(f"    - {chunk_score:.3f} :: {chunk}")

    baseline = score_with_decomposition(
        scorer=scorer,
        sentences=sentences,
        sampled_passages=sampled_passages,
        use_decomposition=False,
    )
    print("\n=== Baseline sentence-level scoring ===")
    print(baseline["scores"])


if __name__ == "__main__":
    run_demo()

