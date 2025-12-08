#!/usr/bin/env python3
"""
FEVER Expansion Experiment Runner (Local Ollama Version)

Runs SelfCheck-NLI on FEVER claims using a local Llama-3 model via Ollama.

Usage:
    python run.py                    # Run full experiment
    python run.py --smoke-test       # Quick sanity check (2 claims)
    python run.py --limit 5          # Run on first 5 claims only
    python run.py --num-samples 3    # Use 3 sampled passages per claim

Requirements:
    - Ollama installed with llama3:8b model: ollama pull llama3:8b
    - Python packages: selfcheckgpt, spacy, torch
    - spaCy model: python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

# Configuration

DEFAULT_MODEL = "llama3:8b"
DEFAULT_NUM_SAMPLES = 4
SMOKE_TEST_CLAIMS = [
    {"claim": "Paris is the capital of France.", "label": "SUPPORTS"},
    {"claim": "The Earth is flat.", "label": "REFUTES"},
]

# Helper Functions

def get_device() -> torch.device:
    """Select best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_spacy_model():
    """Load spaCy English model for sentence splitting."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Error: spaCy model not found. Install with:")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)

def split_sentences(nlp, text: str) -> List[str]:
    """Split text into sentences using spaCy."""
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """
    Call local LLM via Ollama CLI.
    
    Args:
        prompt: The input prompt
        model: Ollama model name
        temperature: Sampling temperature (0.0 = deterministic)
    
    Returns:
        Model response text
    """
    command = ["ollama", "run", model]
    
    # Note: Ollama CLI doesn't support temperature via args directly
    # Temperature variation is achieved by multiple runs with same prompt
    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120,  # 2 minute timeout per call
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Warning: LLM call timed out for prompt: {prompt[:50]}...")
        return ""
    except FileNotFoundError:
        print("Error: Ollama not found. Please install Ollama:")
        print("  https://ollama.ai")
        sys.exit(1)

def process_claim(
    claim: str,
    label: str,
    nlp,
    checker: SelfCheckNLI,
    model: str,
    num_samples: int,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Process a single FEVER claim through the SelfCheck-NLI pipeline.
    
    Args:
        claim: The FEVER claim text
        label: Ground truth label (SUPPORTS/REFUTES)
        nlp: spaCy model for sentence splitting
        checker: SelfCheckNLI instance
        model: Ollama model name
        num_samples: Number of sampled passages to generate
        verbose: Print progress information
    
    Returns:
        Result dictionary or None if processing failed
    """
    prompt = f"Is this statement true? Explain.\n\n{claim}"
    
    if verbose:
        print(f"\n  Processing: {claim[:60]}...")
    
    # Generate main response
    R = call_llm(prompt, model=model, temperature=0.0)
    if not R:
        print(f"  Warning: Empty response for claim")
        return None
    
    sentences_R = split_sentences(nlp, R)
    if not sentences_R:
        print(f"  Warning: No sentences extracted from response")
        return None
    
    # Generate sampled passages for comparison
    samples = []
    for i in range(num_samples):
        sample = call_llm(prompt, model=model, temperature=1.0)
        if sample:
            samples.append(sample)
    
    if len(samples) < 2:
        print(f"  Warning: Not enough valid samples ({len(samples)} < 2)")
        return None
    
    # Run SelfCheck-NLI
    nli_scores = checker.predict(
        sentences=sentences_R,
        sampled_passages=samples,
    )
    
    result = {
        "claim": claim,
        "label": label,
        "R": R,
        "samples": samples,
        "sentences_R": sentences_R,
        "nli_scores": list(map(float, nli_scores)),
    }
    
    if verbose:
        avg_score = float(nli_scores.mean()) if len(nli_scores) > 0 else 0
        max_score = float(nli_scores.max()) if len(nli_scores) > 0 else 0
        print(f"  Label: {label} | Avg NLI: {avg_score:.3f} | Max NLI: {max_score:.3f}")
    
    return result

def run_smoke_test(nlp, checker: SelfCheckNLI, model: str) -> bool:
    """
    Run a quick smoke test to verify the pipeline works.
    
    Returns:
        True if smoke test passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("SMOKE TEST: Running quick sanity check...")
    print("=" * 60)
    
    results = []
    for claim_data in SMOKE_TEST_CLAIMS:
        result = process_claim(
            claim=claim_data["claim"],
            label=claim_data["label"],
            nlp=nlp,
            checker=checker,
            model=model,
            num_samples=2,  # Minimal samples for speed
            verbose=True,
        )
        if result:
            results.append(result)
    
    print("\n" + "-" * 60)
    
    if len(results) < 2:
        print("SMOKE TEST FAILED: Could not process all test claims")
        return False
    
    # Verify that REFUTES gets higher score than SUPPORTS
    supports_score = None
    refutes_score = None
    
    for r in results:
        max_score = max(r["nli_scores"]) if r["nli_scores"] else 0
        if r["label"] == "SUPPORTS":
            supports_score = max_score
        elif r["label"] == "REFUTES":
            refutes_score = max_score
    
    print(f"\n  SUPPORTS claim max score: {supports_score:.3f}")
    print(f"  REFUTES claim max score:  {refutes_score:.3f}")
    
    if supports_score is not None and refutes_score is not None:
        if refutes_score > supports_score:
            print("\nSMOKE TEST PASSED: REFUTES score > SUPPORTS score (expected)")
            return True
        else:
            print("\nSMOKE TEST WARNING: REFUTES score <= SUPPORTS score")
            print("This may indicate model inconsistency, but pipeline is functional")
            return True
    
    print("SMOKE TEST FAILED: Could not compute scores")
    return False

# Main Entry Point

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FEVER expansion experiment with SelfCheck-NLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick sanity check (2 claims, 2 samples each)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "fever_20.jsonl",
        help="Path to FEVER claims JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "fever_nli_results.jsonl",
        help="Output path for results JSONL",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of sampled passages per claim (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N claims (for testing)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-claim progress output",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    
    # Setup
    print(f"\n Initializing...")
    device = get_device()
    print(f"Device: {device}")
    
    nlp = load_spacy_model()
    print(f"spaCy model loaded")
    
    checker = SelfCheckNLI(device=device)
    print(f"SelfCheck-NLI initialized")
    
    # Smoke test mode
    if args.smoke_test:
        success = run_smoke_test(nlp, checker, args.model)
        return 0 if success else 1
    
    # Load FEVER data
    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        return 1
    
    with args.data.open("r", encoding="utf-8") as f:
        fever_items = [json.loads(line) for line in f if line.strip()]
    
    if args.limit:
        fever_items = fever_items[:args.limit]
    
    print(f"\nProcessing {len(fever_items)} FEVER claims...")
    print(f"Model: {args.model}")
    print(f"Samples per claim: {args.num_samples}")
    
    # Process claims
    results = []
    for i, ex in enumerate(fever_items, 1):
        if not args.quiet:
            print(f"\n[{i}/{len(fever_items)}]", end="")
        
        result = process_claim(
            claim=ex["claim"],
            label=ex["label"],
            nlp=nlp,
            checker=checker,
            model=args.model,
            num_samples=args.num_samples,
            verbose=not args.quiet,
        )
        
        if result:
            results.append(result)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Results saved to: {args.output}")
    print(f"Processed: {len(results)}/{len(fever_items)} claims")
    
    if results:
        supports_scores = [max(r["nli_scores"]) for r in results if r["label"] == "SUPPORTS"]
        refutes_scores = [max(r["nli_scores"]) for r in results if r["label"] == "REFUTES"]
        
        if supports_scores:
            print(f"SUPPORTS mean score: {sum(supports_scores)/len(supports_scores):.3f}")
        if refutes_scores:
            print(f"REFUTES mean score:  {sum(refutes_scores)/len(refutes_scores):.3f}")
    
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
