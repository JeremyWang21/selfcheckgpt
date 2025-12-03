"""
Sentence decomposition utilities scaffold.

Agents can extend this with clause/phrase-level splitting using spaCy
(e.g., noun/verb chunks) to enable fine-grained hallucination detection.
"""

from typing import List


def simple_clause_split(sentence: str) -> List[str]:
    """
    Placeholder splitter: returns the sentence as a single chunk.
    Replace with clause/chunk logic (e.g., spaCy noun/verb chunks).
    """
    sentence = sentence.strip()
    return [sentence] if sentence else []
