"""
Runnable multidomain evaluation entry point.

Features:
- Loader overrides via env vars or CLI flags
- Pluggable scorer dispatch (nli, ngram, bertscore, prompt_stub)
- JSONL results per example plus aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from selfcheckgpt import data as data_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multidomain SelfCheckGPT evaluation")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["events", "places", "organizations"],
        help="Registered domains to evaluate",
    )
    parser.add_argument("--split", default="dev", help="Dataset split to load")
    parser.add_argument("--limit", type=int, default=None, help="Max examples per domain")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of completions sampled per example (uses list order)",
    )
    parser.add_argument("--output", type=Path, default=Path("multidomain_results.jsonl"))
    parser.add_argument("--metrics", type=Path, default=Path("multidomain_metrics.json"))
    parser.add_argument(
        "--scorers",
        nargs="+",
        default=["ngram"],
        choices=["nli", "ngram", "bertscore", "prompt_stub"],
        help="Scorers to run",
    )
    parser.add_argument("--device", default="cpu", help="Device string for torch-based scorers")
    parser.add_argument("--ngram-n", type=int, default=2, help="n parameter for ngram scorer")
    parser.add_argument("--bertscore-model", default="en", help="BERTScore language/model alias")
    parser.add_argument(
        "--no-bertscore-rescale",
        action="store_true",
        help="Disable baseline rescaling inside SelfCheckBERTScore",
    )
    # Loader overrides
    parser.add_argument("--events-path", type=Path, help="Override events JSONL directory/file")
    parser.add_argument("--places-path", type=Path, help="Override places JSONL directory/file")
    parser.add_argument("--organizations-path", type=Path, help="Override organizations JSONL")
    parser.add_argument("--events-hf-dataset", help="Hugging Face dataset ID for events")
    parser.add_argument("--events-hf-subset", help="Optional HF subset for events")
    parser.add_argument("--places-hf-dataset", help="Hugging Face dataset ID for places")
    parser.add_argument("--places-hf-subset", help="Optional HF subset for places")
    parser.add_argument("--organizations-hf-dataset", help="Hugging Face dataset ID for orgs")
    parser.add_argument("--organizations-hf-subset", help="Optional HF subset for orgs")
    return parser.parse_args()


@dataclass
class ExampleResult:
    domain: str
    split: str
    example: Mapping[str, Any]
    samples: List[MutableMapping[str, object]]


class BaseScorer:
    name: str

    def score(self, example: Mapping[str, str], candidate: str, sampled_passages: Sequence[str]) -> float:
        raise NotImplementedError


class NgramScorer(BaseScorer):
    def __init__(self, n: int, lowercase: bool = True):
        from selfcheckgpt.modeling_selfcheck import SelfCheckNgram

        self.name = "ngram"
        self.impl = SelfCheckNgram(n=n, lowercase=lowercase)

    def score(self, example: Mapping[str, str], candidate: str, sampled_passages: Sequence[str]) -> float:
        sentences = _split_sentences(candidate)
        if not sentences:
            return 1.0
        scores = self.impl.predict(
            sentences=sentences,
            passage=example["reference"],
            sampled_passages=list(sampled_passages),
        )
        return _mean(scores)


class NLIScorer(BaseScorer):
    def __init__(self, device: str = "cpu"):
        from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

        self.name = "nli"
        self.impl = SelfCheckNLI(device=device)

    def score(self, example: Mapping[str, str], candidate: str, sampled_passages: Sequence[str]) -> float:
        sentences = _split_sentences(candidate)
        if not sentences:
            return 1.0
        scores = self.impl.predict(sentences=sentences, sampled_passages=list(sampled_passages))
        return _mean(scores)


class BertScoreScorer(BaseScorer):
    def __init__(self, model: str = "en", rescale_with_baseline: bool = True):
        from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

        self.name = "bertscore"
        self.impl = SelfCheckBERTScore(default_model=model, rescale_with_baseline=rescale_with_baseline)

    def score(self, example: Mapping[str, str], candidate: str, sampled_passages: Sequence[str]) -> float:
        sentences = _split_sentences(candidate)
        if not sentences:
            return 1.0
        scores = self.impl.predict(sentences=sentences, sampled_passages=list(sampled_passages))
        return _mean(scores)


class PromptStubScorer(BaseScorer):
    def __init__(self):
        self.name = "prompt_stub"

    def score(self, example: Mapping[str, str], candidate: str, sampled_passages: Sequence[str]) -> float:
        reference_tokens = _token_set(example["reference"])
        candidate_tokens = _token_set(candidate)
        if not candidate_tokens:
            return 1.0
        overlap = len(reference_tokens & candidate_tokens)
        return 1.0 - (overlap / len(candidate_tokens))


def build_scorers(args: argparse.Namespace) -> List[BaseScorer]:
    builders = {
        "ngram": lambda: NgramScorer(n=args.ngram_n),
        "nli": lambda: NLIScorer(device=args.device),
        "bertscore": lambda: BertScoreScorer(
            model=args.bertscore_model,
            rescale_with_baseline=not args.no_bertscore_rescale,
        ),
        "prompt_stub": lambda: PromptStubScorer(),
    }
    scorers: List[BaseScorer] = []
    for name in args.scorers:
        if name not in builders:
            raise ValueError(f"Unknown scorer '{name}'")
        scorers.append(builders[name]())
    return scorers


def apply_loader_overrides(args: argparse.Namespace) -> None:
    overrides = {
        "events": {
            "local_path": args.events_path,
            "hf_dataset": args.events_hf_dataset,
            "hf_subset": args.events_hf_subset,
        },
        "places": {
            "local_path": args.places_path,
            "hf_dataset": args.places_hf_dataset,
            "hf_subset": args.places_hf_subset,
        },
        "organizations": {
            "local_path": args.organizations_path,
            "hf_dataset": args.organizations_hf_dataset,
            "hf_subset": args.organizations_hf_subset,
        },
    }
    for domain, params in overrides.items():
        cleaned = {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items() if v}
        if cleaned:
            data_utils.configure_loader(domain, **cleaned)


def _split_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences or [stripped]


def _token_set(text: str) -> set:
    return {token for token in re.findall(r"[A-Za-z0-9']+", text.lower())}


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(statistics.fmean(seq))


def evaluate_example(
    example: Mapping[str, Any],
    *,
    domain: str,
    split: str,
    scorers: Sequence[BaseScorer],
    max_samples: Optional[int],
) -> ExampleResult:
    samples = list(example["samples"])
    if max_samples is not None:
        samples = samples[:max_samples]
    sample_payloads: List[MutableMapping[str, object]] = []
    for idx, candidate in enumerate(samples):
        # Build sampled passages: include reference and other completions for context
        sampled_passages = [example["reference"]] + [samples[j] for j in range(len(samples)) if j != idx]
        score_map: Dict[str, float] = {}
        for scorer in scorers:
            score_map[scorer.name] = scorer.score(example, candidate, sampled_passages)
        sample_payloads.append(
            {
                "rank": idx,
                "text": candidate,
                "scores": score_map,
            }
        )
    return ExampleResult(domain=domain, split=split, example=example, samples=sample_payloads)


def main() -> None:
    args = parse_args()
    apply_loader_overrides(args)
    scorers = build_scorers(args)
    if not scorers:
        raise ValueError("At least one scorer must be specified")

    results: List[ExampleResult] = []
    metrics_accumulator: Dict[str, Dict[str, Any]] = {
        scorer.name: {"overall": [], "domains": defaultdict(list)} for scorer in scorers
    }

    total_examples = 0
    for domain in args.domains:
        examples = data_utils.load(domain, split=args.split, limit=args.limit)
        for example in examples:
            total_examples += 1
            example_result = evaluate_example(
                example,
                domain=domain,
                split=args.split,
                scorers=scorers,
                max_samples=args.max_samples,
            )
            results.append(example_result)
            for sample in example_result.samples:
                for scorer_name, value in sample["scores"].items():
                    metrics_accumulator[scorer_name]["overall"].append(float(value))
                    metrics_accumulator[scorer_name]["domains"][domain].append(float(value))

    if not results:
        raise RuntimeError("No examples were loaded; verify domains, split, and overrides.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for result in results:
            payload = {
                "domain": result.domain,
                "split": result.split,
                "id": result.example["id"],
                "prompt": result.example["prompt"],
                "reference": result.example["reference"],
                "samples": result.samples,
            }
            handle.write(json.dumps(payload) + "\n")

    finalized_metrics: Dict[str, Dict[str, object]] = {}
    for scorer_name, sections in metrics_accumulator.items():
        domain_means = {
            domain: _mean(values) for domain, values in sections["domains"].items() if values
        }
        finalized_metrics[scorer_name] = {
            "overall_mean": _mean(sections["overall"]),
            "domain_mean": domain_means,
            "count": len(sections["overall"]),
        }

    metrics_payload = {
        "domains": args.domains,
        "split": args.split,
        "examples_evaluated": total_examples,
        "scorers": finalized_metrics,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Processed {total_examples} examples across {len(args.domains)} domain(s).")
    print(f"Results: {args.output.resolve()}")
    print(f"Metrics: {args.metrics.resolve()}")


if __name__ == "__main__":
    main()
