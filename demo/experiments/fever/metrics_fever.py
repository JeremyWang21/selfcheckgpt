#!/usr/bin/env python3
"""
Metrics computation for FEVER expansion experiment.

Computes:
- Non-Fact AUC-PR: How well we catch false sentences (REFUTES as positive class)
- Factual AUC-PR: How well we avoid flagging true sentences (SUPPORTS as positive class)
- Pearson & Spearman correlation between scores and binary labels

Usage:
    python metrics_fever.py --results results/nli_results.jsonl
    python metrics_fever.py --results results/nli_results.jsonl --output metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score


def load_results(results_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def extract_scores_and_labels(results: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract claim-level scores and binary labels from results.
    
    Returns:
        scores: Array of claim-level NLI scores (higher = more likely hallucination)
        labels: Binary labels (1 = REFUTES/non-fact, 0 = SUPPORTS/fact)
    """
    scores = []
    labels = []
    
    for result in results:
        # Aggregate sentence-level scores to claim-level using max
        nli_scores = result.get("nli_scores", [])
        if nli_scores:
            claim_score = max(nli_scores)  # max aggregation
        else:
            claim_score = 0.5  # neutral default
        
        scores.append(claim_score)
        
        # Binary label: 1 for REFUTES (non-fact), 0 for SUPPORTS (fact)
        label = result.get("label", "")
        if label == "REFUTES":
            labels.append(1)
        elif label == "SUPPORTS":
            labels.append(0)
        else:
            # NEI or unknown - skip or treat as 0.5
            labels.append(0)
    
    return np.array(scores), np.array(labels)


def compute_nonfact_auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Non-Fact AUC-PR: REFUTES as positive class.
    Higher scores should correspond to REFUTES (label=1).
    """
    try:
        return float(average_precision_score(labels, scores))
    except Exception as e:
        print(f"Warning: Could not compute Non-Fact AUC-PR: {e}")
        return 0.0


def compute_factual_auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Factual AUC-PR: SUPPORTS as positive class.
    Lower scores should correspond to SUPPORTS (label=0).
    We invert both scores and labels.
    """
    try:
        inverted_labels = 1 - labels
        inverted_scores = 1 - scores
        return float(average_precision_score(inverted_labels, inverted_scores))
    except Exception as e:
        print(f"Warning: Could not compute Factual AUC-PR: {e}")
        return 0.0


def compute_correlations(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute Pearson and Spearman correlations between scores and labels.
    """
    correlations = {}
    
    try:
        pearson_r, pearson_p = stats.pearsonr(scores, labels)
        correlations["pearson_r"] = float(pearson_r)
        correlations["pearson_p"] = float(pearson_p)
    except Exception as e:
        print(f"Warning: Could not compute Pearson correlation: {e}")
        correlations["pearson_r"] = 0.0
        correlations["pearson_p"] = 1.0
    
    try:
        spearman_r, spearman_p = stats.spearmanr(scores, labels)
        correlations["spearman_r"] = float(spearman_r)
        correlations["spearman_p"] = float(spearman_p)
    except Exception as e:
        print(f"Warning: Could not compute Spearman correlation: {e}")
        correlations["spearman_r"] = 0.0
        correlations["spearman_p"] = 1.0
    
    return correlations


def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all metrics from results."""
    scores, labels = extract_scores_and_labels(results)
    
    # Count labels
    n_refutes = int(np.sum(labels == 1))
    n_supports = int(np.sum(labels == 0))
    
    metrics = {
        "n_claims": len(results),
        "n_refutes": n_refutes,
        "n_supports": n_supports,
        "nonfact_auc_pr": compute_nonfact_auc_pr(scores, labels),
        "factual_auc_pr": compute_factual_auc_pr(scores, labels),
        **compute_correlations(scores, labels),
        "score_stats": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        "refutes_score_mean": float(np.mean(scores[labels == 1])) if n_refutes > 0 else 0.0,
        "supports_score_mean": float(np.mean(scores[labels == 0])) if n_supports > 0 else 0.0,
    }
    
    return metrics


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print("FEVER Expansion - SelfCheck-NLI Metrics Report")
    print("=" * 60)
    
    print(f"\nDataset Summary:")
    print(f"  Total claims: {metrics['n_claims']}")
    print(f"  REFUTES (non-fact): {metrics['n_refutes']}")
    print(f"  SUPPORTS (factual): {metrics['n_supports']}")
    
    print(f"\nAUC-PR Metrics:")
    print(f"  Non-Fact AUC-PR: {metrics['nonfact_auc_pr']:.4f}")
    print(f"  Factual AUC-PR:  {metrics['factual_auc_pr']:.4f}")
    
    print(f"\nCorrelation Metrics:")
    print(f"  Pearson ρ:  {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4f})")
    print(f"  Spearman ρ: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4f})")
    
    print(f"\nScore Statistics:")
    print(f"  Mean: {metrics['score_stats']['mean']:.4f}")
    print(f"  Std:  {metrics['score_stats']['std']:.4f}")
    print(f"  Min:  {metrics['score_stats']['min']:.4f}")
    print(f"  Max:  {metrics['score_stats']['max']:.4f}")
    
    print(f"\nScore by Label:")
    print(f"  REFUTES mean score:  {metrics['refutes_score_mean']:.4f}")
    print(f"  SUPPORTS mean score: {metrics['supports_score_mean']:.4f}")
    
    print("\n" + "-" * 60)
    print("Comparison with WikiBio (Original SelfCheckGPT Paper):")
    print("-" * 60)
    print(f"{'Metric':<20} {'FEVER':<12} {'WikiBio':<12} {'Diff':<10}")
    print("-" * 60)
    
    wikibio = {"nonfact_auc_pr": 0.65, "factual_auc_pr": 0.66, "pearson_r": 0.62, "spearman_r": 0.70}
    
    for metric, wiki_val in wikibio.items():
        fever_val = metrics.get(metric, 0.0)
        diff = fever_val - wiki_val
        print(f"{metric:<20} {fever_val:<12.4f} {wiki_val:<12.4f} {diff:+.4f}")
    
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics for FEVER expansion experiment")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/nli_results.jsonl"),
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printed output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        print("Run the experiment first: python run.py or python run_groq.py")
        return 1
    
    results = load_results(args.results)
    if not results:
        print("Error: No results found in file")
        return 1
    
    metrics = compute_all_metrics(results)
    
    if not args.quiet:
        print_metrics_report(metrics)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

