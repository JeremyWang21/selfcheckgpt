#!/usr/bin/env python3
"""
FEVER Benchmark Evaluation Script.

Runs SelfCheck-NLI on FEVER claims using Groq API for LLM generation,
and computes metrics comparing SUPPORTS vs REFUTES claims.

Usage:
    export GROQ_API_KEY=your-key
    python run_fever_eval.py --input fever_50.jsonl --num-samples 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import spacy
import torch
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

# Load environment variables
load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FEVER SelfCheck-NLI Evaluation")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "fever_50.jsonl",
        help="Input JSONL file with claims",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "fever_eval_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of LLM samples per claim",
    )
    parser.add_argument(
        "--model",
        default="llama-3.1-8b-instant",
        help="Groq model to use",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for NLI model (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of claims to process",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def load_fever_claims(input_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load FEVER claims from JSONL file."""
    claims = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))
                if limit and len(claims) >= limit:
                    break
    return claims


class GroqClient:
    """Wrapper for Groq API calls."""
    
    def __init__(self, model: str):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a response from the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {e}")
            return ""


def process_claim(
    claim_data: Dict[str, str],
    groq_client: GroqClient,
    nli_checker: SelfCheckNLI,
    nlp: Any,
    num_samples: int,
) -> Optional[Dict[str, Any]]:
    """Process a single claim and return results."""
    claim = claim_data["claim"]
    label = claim_data["label"]
    
    prompt = f"Is this statement true? Explain briefly.\n\n{claim}"
    
    # Generate samples in parallel
    with ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = [executor.submit(groq_client.generate, prompt, 0.7) for _ in range(num_samples)]
        responses = [f.result() for f in futures]
    
    # Filter empty responses
    responses = [r for r in responses if r]
    if len(responses) < 2:
        return None
    
    main_response = responses[0]
    samples = responses[1:]
    
    # Split into sentences
    sentences = [s.text.strip() for s in nlp(main_response).sents if s.text.strip()]
    if not sentences:
        return None
    
    # Run NLI scoring
    nli_scores = nli_checker.predict(sentences=sentences, sampled_passages=samples)
    
    return {
        "claim": claim,
        "label": label,
        "response": main_response,
        "samples": samples,
        "sentences": sentences,
        "nli_scores": [float(s) for s in nli_scores],
        "avg_nli_score": float(np.mean(nli_scores)),
        "max_nli_score": float(np.max(nli_scores)),
    }


def compute_fever_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute metrics from FEVER evaluation results."""
    supports_scores = []
    refutes_scores = []
    
    for r in results:
        if r["label"] == "SUPPORTS":
            supports_scores.append(r["avg_nli_score"])
        else:
            refutes_scores.append(r["avg_nli_score"])
    
    metrics = {
        "supports": {
            "count": len(supports_scores),
            "mean_nli_score": float(np.mean(supports_scores)) if supports_scores else 0,
            "std_nli_score": float(np.std(supports_scores)) if supports_scores else 0,
            "median_nli_score": float(np.median(supports_scores)) if supports_scores else 0,
        },
        "refutes": {
            "count": len(refutes_scores),
            "mean_nli_score": float(np.mean(refutes_scores)) if refutes_scores else 0,
            "std_nli_score": float(np.std(refutes_scores)) if refutes_scores else 0,
            "median_nli_score": float(np.median(refutes_scores)) if refutes_scores else 0,
        },
    }
    
    # Compute separation metrics
    all_scores = supports_scores + refutes_scores
    all_labels = [0] * len(supports_scores) + [1] * len(refutes_scores)
    
    if len(all_scores) > 0:
        # AUC-ROC style metric: can we separate SUPPORTS from REFUTES?
        from sklearn.metrics import roc_auc_score
        try:
            metrics["separation_auc"] = float(roc_auc_score(all_labels, all_scores))
        except:
            metrics["separation_auc"] = 0.5
        
        # Effect size (Cohen's d)
        if supports_scores and refutes_scores:
            pooled_std = np.sqrt(
                (np.var(supports_scores) + np.var(refutes_scores)) / 2
            )
            if pooled_std > 0:
                metrics["cohens_d"] = float(
                    (np.mean(refutes_scores) - np.mean(supports_scores)) / pooled_std
                )
            else:
                metrics["cohens_d"] = 0.0
        else:
            metrics["cohens_d"] = 0.0
    
    return metrics


def main() -> None:
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    print("Loading NLI model...")
    nli_checker = SelfCheckNLI(device=device)
    
    print(f"Initializing Groq client (model={args.model})...")
    groq_client = GroqClient(args.model)
    
    # Load claims
    claims = load_fever_claims(args.input, args.limit)
    print(f"Loaded {len(claims)} claims")
    
    # Count labels
    supports_count = sum(1 for c in claims if c["label"] == "SUPPORTS")
    refutes_count = len(claims) - supports_count
    print(f"  SUPPORTS: {supports_count}, REFUTES: {refutes_count}")
    
    # Process claims
    results = []
    start_time = time.time()
    
    for claim_data in tqdm(claims, desc="Processing claims"):
        result = process_claim(
            claim_data, groq_client, nli_checker, nlp, args.num_samples
        )
        if result:
            results.append(result)
            
            if args.verbose:
                print(f"\n  Claim: {result['claim'][:60]}...")
                print(f"  Label: {result['label']}, Avg NLI: {result['avg_nli_score']:.3f}")
    
    # Compute metrics
    metrics = compute_fever_metrics(results)
    
    # Save results
    output_data = {
        "dataset": "fever",
        "input_file": str(args.input),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "num_samples": args.num_samples,
            "device": str(device),
        },
        "metrics": metrics,
        "runtime_seconds": time.time() - start_time,
        "num_processed": len(results),
        "results": results,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FEVER EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Processed: {len(results)}/{len(claims)} claims")
    print(f"Runtime: {output_data['runtime_seconds']:.1f}s")
    print(f"\nSUPPORTS claims (true facts):")
    print(f"  Count: {metrics['supports']['count']}")
    print(f"  Mean NLI Score: {metrics['supports']['mean_nli_score']:.4f}")
    print(f"  Std: {metrics['supports']['std_nli_score']:.4f}")
    print(f"\nREFUTES claims (false facts):")
    print(f"  Count: {metrics['refutes']['count']}")
    print(f"  Mean NLI Score: {metrics['refutes']['mean_nli_score']:.4f}")
    print(f"  Std: {metrics['refutes']['std_nli_score']:.4f}")
    print(f"\nSeparation Metrics:")
    print(f"  AUC (SUPPORTS vs REFUTES): {metrics.get('separation_auc', 'N/A'):.4f}")
    print(f"  Cohen's d: {metrics.get('cohens_d', 'N/A'):.4f}")
    print(f"\nResults saved to: {args.output}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    if metrics['supports']['mean_nli_score'] < metrics['refutes']['mean_nli_score']:
        print("✓ SUPPORTS claims have LOWER NLI scores (more consistent)")
        print("✓ REFUTES claims have HIGHER NLI scores (more contradictory)")
        print("→ SelfCheck-NLI shows expected behavior on FEVER")
    else:
        print("! Unexpected: SUPPORTS claims have higher scores than REFUTES")
        print("  This may indicate noise or domain differences")


if __name__ == "__main__":
    main()

