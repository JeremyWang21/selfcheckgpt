"""
Placeholder runner for evaluating SelfCheckGPT scorers across domains.

Agents should implement:
- Loader registration in selfcheckgpt.data
- Wiring to call desired scorer(s) based on CLI flags

This script is intentionally lightweight and CPU-only by default.
"""

import argparse
import json
from pathlib import Path
from typing import List

from selfcheckgpt import modeling_selfcheck  # noqa: F401 - for future extension
from selfcheckgpt import data as data_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multidomain SelfCheckGPT evaluation scaffold")
    parser.add_argument("--domain", required=True, help="Registered domain name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on examples")
    parser.add_argument("--output", type=Path, default=Path("multidomain_results.jsonl"))
    parser.add_argument(
        "--scorers",
        nargs="+",
        default=["nli"],
        help="Scorers to run (agent should implement dispatcher)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = data_utils.load(args.domain, split=args.split, limit=args.limit)

    # TODO: Add scorer dispatch; currently just writes stubbed examples.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"id": ex["id"], "prompt": ex["prompt"]}) + "\n")

    print(f"Wrote placeholder results to {args.output}")


if __name__ == "__main__":
    main()
