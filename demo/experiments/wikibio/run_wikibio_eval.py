#!/usr/bin/env python3
"""
WikiBio Benchmark Evaluation Script.

Runs SelfCheckGPT scorers on the wiki_bio_gpt3_hallucination dataset
and computes standard metrics (AUC-PR, correlation) for comparison
with the original paper results.

Usage:
    python run_wikibio_eval.py --limit 50 --scorers nli ngram bertscore --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from selfcheckgpt.metrics import (
    compute_all_metrics,
    aggregate_passage_scores,
    format_metrics_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WikiBio SelfCheckGPT Evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of passages to evaluate (default: 50)",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        default=["nli"],
        choices=["nli", "ngram", "bertscore"],
        help="Scorers to run (default: nli)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for torch-based scorers (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--ngram-n",
        type=int,
        default=1,
        help="N-gram order (1=unigram, 2=bigram, etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "wikibio_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    return parser.parse_args()


def load_wikibio_dataset(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load WikiBio GPT-3 hallucination dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "datasets library required. Install with: pip install datasets"
        )
    
    print("Loading wiki_bio_gpt3_hallucination dataset from HuggingFace...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    
    examples = []
    for idx, item in enumerate(dataset["evaluation"]):
        if limit is not None and idx >= limit:
            break
        examples.append(dict(item))
    
    print(f"Loaded {len(examples)} passages")
    return examples


def get_sentence_labels(annotations: List[str]) -> List[int]:
    """
    Convert annotation strings to binary labels.
    
    WikiBio annotations: "major_inaccurate", "minor_inaccurate", "accurate"
    Returns: 1 for non-factual (major/minor inaccurate), 0 for factual
    """
    labels = []
    for ann in annotations:
        if ann in ["major_inaccurate", "minor_inaccurate"]:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def run_nli_scorer(
    examples: List[Dict[str, Any]],
    device: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run SelfCheck-NLI on all passages."""
    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
    
    print(f"\n=== Running NLI Scorer (device={device}) ===")
    scorer = SelfCheckNLI(device=device)
    
    all_scores = []
    all_labels = []
    passage_scores_list = []
    
    for ex in tqdm(examples, desc="NLI scoring", disable=not verbose):
        sentences = ex["gpt3_sentences"]
        samples = ex["gpt3_text_samples"]
        annotations = ex["annotation"]
        
        if not sentences or not samples:
            continue
        
        scores = scorer.predict(sentences=sentences, sampled_passages=samples)
        labels = get_sentence_labels(annotations)
        
        # Ensure lengths match
        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]
        
        all_scores.extend(scores.tolist())
        all_labels.extend(labels)
        passage_scores_list.append(scores.tolist())
    
    # Compute metrics
    metrics = compute_all_metrics(all_scores, all_labels)
    
    # Also compute passage-level metrics
    passage_scores = aggregate_passage_scores(passage_scores_list, method="mean")
    passage_labels = [
        1 if any(get_sentence_labels(ex["annotation"])) else 0 
        for ex in examples if ex["gpt3_sentences"]
    ][:len(passage_scores)]
    
    passage_metrics = compute_all_metrics(passage_scores, passage_labels)
    
    return {
        "sentence_level": metrics,
        "passage_level": passage_metrics,
        "num_sentences": len(all_scores),
        "num_passages": len(passage_scores),
    }


def run_ngram_scorer(
    examples: List[Dict[str, Any]],
    n: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run SelfCheck-Ngram on all passages."""
    from selfcheckgpt.modeling_selfcheck import SelfCheckNgram
    
    print(f"\n=== Running {n}-gram Scorer ===")
    scorer = SelfCheckNgram(n=n)
    
    all_scores = []
    all_labels = []
    passage_scores_list = []
    
    for ex in tqdm(examples, desc=f"{n}-gram scoring", disable=not verbose):
        sentences = ex["gpt3_sentences"]
        passage = ex["gpt3_text"]
        samples = ex["gpt3_text_samples"]
        annotations = ex["annotation"]
        
        if not sentences or not samples:
            continue
        
        result = scorer.predict(
            sentences=sentences,
            passage=passage,
            sampled_passages=samples,
        )
        
        # Use avg_neg_logprob as scores
        scores = result["sent_level"]["avg_neg_logprob"]
        labels = get_sentence_labels(annotations)
        
        # Ensure lengths match
        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]
        
        all_scores.extend(scores)
        all_labels.extend(labels)
        passage_scores_list.append(scores)
    
    # Normalize scores to [0, 1] range for fair comparison
    all_scores = np.array(all_scores)
    if all_scores.max() > all_scores.min():
        all_scores_norm = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())
    else:
        all_scores_norm = all_scores
    
    metrics = compute_all_metrics(all_scores_norm.tolist(), all_labels)
    
    # Passage-level
    passage_scores = aggregate_passage_scores(passage_scores_list, method="mean")
    passage_labels = [
        1 if any(get_sentence_labels(ex["annotation"])) else 0 
        for ex in examples if ex["gpt3_sentences"]
    ][:len(passage_scores)]
    
    # Normalize passage scores
    passage_scores = np.array(passage_scores)
    if passage_scores.max() > passage_scores.min():
        passage_scores_norm = (passage_scores - passage_scores.min()) / (passage_scores.max() - passage_scores.min())
    else:
        passage_scores_norm = passage_scores
    
    passage_metrics = compute_all_metrics(passage_scores_norm.tolist(), passage_labels)
    
    return {
        "sentence_level": metrics,
        "passage_level": passage_metrics,
        "num_sentences": len(all_scores),
        "num_passages": len(passage_scores),
    }


def run_bertscore_scorer(
    examples: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run SelfCheck-BERTScore on all passages."""
    from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
    
    print("\n=== Running BERTScore Scorer ===")
    scorer = SelfCheckBERTScore(rescale_with_baseline=True)
    
    all_scores = []
    all_labels = []
    passage_scores_list = []
    
    for ex in tqdm(examples, desc="BERTScore scoring", disable=not verbose):
        sentences = ex["gpt3_sentences"]
        samples = ex["gpt3_text_samples"]
        annotations = ex["annotation"]
        
        if not sentences or not samples:
            continue
        
        scores = scorer.predict(sentences=sentences, sampled_passages=samples)
        labels = get_sentence_labels(annotations)
        
        # Ensure lengths match
        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]
        
        all_scores.extend(scores.tolist())
        all_labels.extend(labels)
        passage_scores_list.append(scores.tolist())
    
    metrics = compute_all_metrics(all_scores, all_labels)
    
    # Passage-level
    passage_scores = aggregate_passage_scores(passage_scores_list, method="mean")
    passage_labels = [
        1 if any(get_sentence_labels(ex["annotation"])) else 0 
        for ex in examples if ex["gpt3_sentences"]
    ][:len(passage_scores)]
    
    passage_metrics = compute_all_metrics(passage_scores, passage_labels)
    
    return {
        "sentence_level": metrics,
        "passage_level": passage_metrics,
        "num_sentences": len(all_scores),
        "num_passages": len(passage_scores),
    }


def main() -> None:
    args = parse_args()
    
    # Load dataset
    examples = load_wikibio_dataset(limit=args.limit)
    
    if not examples:
        print("No examples loaded!")
        return
    
    # Run scorers
    results = {
        "dataset": "wiki_bio_gpt3_hallucination",
        "num_passages": len(examples),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "limit": args.limit,
            "scorers": args.scorers,
            "device": args.device,
            "ngram_n": args.ngram_n,
        },
        "scorers": {},
    }
    
    start_time = time.time()
    
    for scorer_name in args.scorers:
        scorer_start = time.time()
        
        if scorer_name == "nli":
            scorer_results = run_nli_scorer(examples, args.device, args.verbose)
        elif scorer_name == "ngram":
            scorer_results = run_ngram_scorer(examples, args.ngram_n, args.verbose)
        elif scorer_name == "bertscore":
            scorer_results = run_bertscore_scorer(examples, args.verbose)
        else:
            print(f"Unknown scorer: {scorer_name}")
            continue
        
        scorer_results["runtime_seconds"] = time.time() - scorer_start
        results["scorers"][scorer_name] = scorer_results
        
        # Print summary
        sent_metrics = scorer_results["sentence_level"]
        print(f"\n{scorer_name.upper()} Results (sentence-level):")
        print(f"  NonFact AUC-PR: {sent_metrics['nonfact_auc_pr']*100:.2f}")
        print(f"  Factual AUC-PR: {sent_metrics['factual_auc_pr']*100:.2f}")
        print(f"  Pearson: {sent_metrics['pearson']*100:.2f}")
        print(f"  Spearman: {sent_metrics['spearman']*100:.2f}")
    
    results["total_runtime_seconds"] = time.time() - start_time
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"Total runtime: {results['total_runtime_seconds']:.1f}s")
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON WITH ORIGINAL PAPER")
    print("="*60)
    
    # Format for table
    table_results = {}
    for scorer_name, scorer_data in results["scorers"].items():
        table_results[scorer_name] = scorer_data["sentence_level"]
    
    print(format_metrics_table(table_results))


if __name__ == "__main__":
    main()

