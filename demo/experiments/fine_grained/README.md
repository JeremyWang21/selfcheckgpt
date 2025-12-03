# Fine-Grained Decomposition Demo

This experiment folder contains a lightweight script that demonstrates how to
use `selfcheckgpt.orchestrator.score_with_decomposition` for clause/phrase level
scoring and aggregation.

## Requirements

- Python 3.9+
- `selfcheckgpt` installed (editable/develop mode recommended)
- spaCy English model so the clause splitter can leverage dependency parses:

```bash
python -m spacy download en_core_web_sm
```

Without the model the splitter falls back to a heuristic tokenizer.

## Usage

Run the demo with:

```bash
python demo/experiments/fine_grained/fine_grained_demo.py
```

The script uses a dummy scorer so you can inspect the decomposition and
aggregation flow without pulling down heavy models. Swap in a real SelfCheckGPT
scorer to experiment with fine-grained hallucination checks.
