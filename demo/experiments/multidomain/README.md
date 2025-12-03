Multidomain Evaluation Runner
=============================

This experiment bundles lightweight JSONL datasets for multiple factuality domains
(`events`, `places`, `organizations`) and a flexible runner that can dispatch to
several SelfCheckGPT scorers. Everything is CPU-friendly by default while still
allowing optional Hugging Face / custom data overrides when needed.

Dataset Layout
--------------

```
demo/experiments/multidomain/data/<domain>/<split>.jsonl
```

Each line is a JSON object with the following schema:

- `id`: unique identifier per example.
- `prompt`: instruction or question provided to the model.
- `reference`: ground-truth context paragraph.
- `samples`: list of model completions to be scored.

Default files avoid network access and cover a handful of curated examples per domain.
Loaders can be overridden via:

- Environment variables: `SCG_EVENTS_PATH`, `SCG_PLACES_PATH`, `SCG_ORGS_PATH` (folders or files),
  plus the analogous `*_HF_DATASET` and `*_HF_SUBSET` for Hugging Face sources.
- CLI flags on `run.py` (see below) which internally call `selfcheckgpt.data.configure_loader`.

Runner Usage
------------

Basic invocation over all domains with the fast `ngram` scorer:

```
python demo/experiments/multidomain/run.py \
  --domains events places organizations \
  --split dev \
  --scorers ngram \
  --output demo/experiments/multidomain/results/dev_ngram.jsonl \
  --metrics demo/experiments/multidomain/results/metrics_ngram.json
```

Choose specific scorers or add the prompt-based stub:

```
python demo/experiments/multidomain/run.py \
  --domains events places \
  --split dev \
  --scorers ngram prompt_stub \
  --max-samples 1
```

Enable heavier scorers (requires `torch`, `transformers`, `spacy`, etc.) and send them to GPU:

```
python demo/experiments/multidomain/run.py \
  --domains events \
  --scorers nli bertscore \
  --device cuda \
  --bertscore-model en \
  --output demo/experiments/multidomain/results/events_nli_bertscore.jsonl
```

Override datasets without editing code:

```
# Custom JSONL file for events and Hugging Face dataset for organizations
python demo/experiments/multidomain/run.py \
  --domains events organizations \
  --events-path ~/datasets/events/data.jsonl \
  --organizations-hf-dataset my-org/selfcheck-orgs \
  --organizations-hf-subset clean \
  --split train
```

Outputs
-------

- `--output`: JSONL with one record per example containing the prompt, reference,
  sampled completions, and per-scorer scores.
- `--metrics`: JSON summary with overall/domain means per scorer and counts.

Refer to `python demo/experiments/multidomain/run.py --help` for the complete list of options
including n-gram order, BERTScore settings, and sample caps.
