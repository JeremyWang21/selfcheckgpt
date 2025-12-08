"""
Evaluation metrics for SelfCheckGPT.

Provides functions to compute:
- AUC-PR (Area Under Precision-Recall Curve) for hallucination detection
- Pearson and Spearman correlation for ranking quality
- Accuracy at various thresholds
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from scipy.stats import pearsonr, spearmanr


def compute_auc_pr(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
) -> Tuple[float, float]:
    """
    Compute AUC-PR for both NonFact and Factual classes.
    
    Args:
        scores: Inconsistency scores (higher = more likely hallucinated)
        labels: Binary labels (1 = non-factual/hallucinated, 0 = factual)
    
    Returns:
        Tuple of (nonfact_auc_pr, factual_auc_pr)
        - nonfact_auc_pr: AUC-PR for detecting non-factual sentences
        - factual_auc_pr: AUC-PR for detecting factual sentences (inverted scores)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for AUC-PR computation. Install with: pip install scikit-learn")
    
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels must have same length, got {len(scores)} and {len(labels)}")
    
    if len(scores) == 0:
        return 0.0, 0.0
    
    # AUC-PR for non-factual class (higher score = non-factual)
    nonfact_auc_pr = average_precision_score(labels, scores)
    
    # AUC-PR for factual class (invert scores and labels)
    inverted_labels = 1 - labels
    inverted_scores = -scores  # or max(scores) - scores
    factual_auc_pr = average_precision_score(inverted_labels, inverted_scores)
    
    return float(nonfact_auc_pr), float(factual_auc_pr)


def compute_correlation(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[float], np.ndarray],
) -> Tuple[float, float]:
    """
    Compute Pearson and Spearman correlation between scores and labels.
    
    Args:
        scores: Inconsistency scores
        labels: Ground truth labels (can be continuous or binary)
    
    Returns:
        Tuple of (pearson_r, spearman_r)
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels must have same length, got {len(scores)} and {len(labels)}")
    
    if len(scores) < 2:
        return 0.0, 0.0
    
    # Handle constant arrays
    if np.std(scores) == 0 or np.std(labels) == 0:
        return 0.0, 0.0
    
    pearson_r, _ = pearsonr(scores, labels)
    spearman_r, _ = spearmanr(scores, labels)
    
    return float(pearson_r), float(spearman_r)


def compute_accuracy_at_threshold(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: float = 0.5,
) -> float:
    """
    Compute accuracy at a given threshold.
    
    Args:
        scores: Inconsistency scores
        labels: Binary labels (1 = non-factual, 0 = factual)
        threshold: Score threshold for classification
    
    Returns:
        Accuracy as float between 0 and 1
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    predictions = (scores >= threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    
    return float(accuracy)


def compute_optimal_threshold(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
) -> Tuple[float, float]:
    """
    Find the optimal threshold that maximizes F1 score.
    
    Args:
        scores: Inconsistency scores
        labels: Binary labels
    
    Returns:
        Tuple of (optimal_threshold, best_f1)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
    
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Compute F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find best threshold
    best_idx = np.argmax(f1_scores[:-1])  # Exclude last point
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    best_f1 = f1_scores[best_idx]
    
    return float(best_threshold), float(best_f1)


def aggregate_passage_scores(
    sentence_scores: List[List[float]],
    method: str = "mean",
) -> List[float]:
    """
    Aggregate sentence-level scores to passage-level.
    
    Args:
        sentence_scores: List of score lists, one per passage
        method: Aggregation method ("mean", "max", "median")
    
    Returns:
        List of passage-level scores
    """
    passage_scores = []
    
    for scores in sentence_scores:
        if not scores:
            passage_scores.append(0.0)
            continue
            
        scores_arr = np.asarray(scores)
        
        if method == "mean":
            passage_scores.append(float(np.mean(scores_arr)))
        elif method == "max":
            passage_scores.append(float(np.max(scores_arr)))
        elif method == "median":
            passage_scores.append(float(np.median(scores_arr)))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    return passage_scores


def compute_all_metrics(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
) -> Dict[str, float]:
    """
    Compute all standard metrics in one call.
    
    Args:
        scores: Inconsistency scores
        labels: Binary labels (1 = non-factual, 0 = factual)
    
    Returns:
        Dictionary with all metrics
    """
    nonfact_auc, fact_auc = compute_auc_pr(scores, labels)
    pearson, spearman = compute_correlation(scores, labels)
    opt_threshold, best_f1 = compute_optimal_threshold(scores, labels)
    accuracy = compute_accuracy_at_threshold(scores, labels, opt_threshold)
    
    return {
        "nonfact_auc_pr": nonfact_auc,
        "factual_auc_pr": fact_auc,
        "pearson": pearson,
        "spearman": spearman,
        "optimal_threshold": opt_threshold,
        "best_f1": best_f1,
        "accuracy_at_optimal": accuracy,
    }


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    paper_baselines: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """
    Format metrics as a markdown table for display.
    
    Args:
        results: Dict mapping scorer name to metrics dict
        paper_baselines: Optional dict with original paper results for comparison
    
    Returns:
        Markdown-formatted table string
    """
    if paper_baselines is None:
        paper_baselines = {
            "nli": {"nonfact_auc_pr": 92.50, "factual_auc_pr": 66.08, "pearson": 74.14},
            "bertscore": {"nonfact_auc_pr": 81.96, "factual_auc_pr": 44.23, "pearson": 58.18},
            "unigram": {"nonfact_auc_pr": 85.63, "factual_auc_pr": 58.47, "pearson": 64.71},
        }
    
    lines = [
        "| Method | NonFact AUC-PR | Factual AUC-PR | Pearson | Paper NonFact | Paper Factual | Paper Pearson |",
        "|--------|----------------|----------------|---------|---------------|---------------|---------------|",
    ]
    
    for scorer_name, metrics in results.items():
        paper = paper_baselines.get(scorer_name, {})
        
        nonfact = metrics.get("nonfact_auc_pr", 0) * 100
        factual = metrics.get("factual_auc_pr", 0) * 100
        pearson = metrics.get("pearson", 0) * 100
        
        paper_nonfact = paper.get("nonfact_auc_pr", "-")
        paper_factual = paper.get("factual_auc_pr", "-")
        paper_pearson = paper.get("pearson", "-")
        
        lines.append(
            f"| {scorer_name} | {nonfact:.2f} | {factual:.2f} | {pearson:.2f} | "
            f"{paper_nonfact} | {paper_factual} | {paper_pearson} |"
        )
    
    return "\n".join(lines)

