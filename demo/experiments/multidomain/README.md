Multidomain Evaluation Scaffold
===============================

- Purpose: host loaders/runners for evaluating SelfCheckGPT across new domains (events, places, organizations, etc.).
- Files:
  - `run.py`: CLI scaffold; expand to dispatch to scorers and write metrics.
- Next steps for the assigned agent:
  - Implement loaders in `selfcheckgpt/data.py` and register them.
  - Extend `run.py` to instantiate scorers based on `--scorers` and log outputs/metrics.
  - Keep defaults CPU-only and avoid network access unless explicitly requested.
