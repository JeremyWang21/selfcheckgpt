SelfCheckGPT
=====================================================
[![arxiv](https://img.shields.io/badge/arXiv-2303.08896-b31b1b.svg)](https://arxiv.org/abs/2303.08896)
[![PyPI version selfcheckgpt](https://badge.fury.io/py/selfcheckgpt.svg?kill_cache=1)](https://pypi.python.org/pypi/selfcheckgpt/)
[![Downloads](https://pepy.tech/badge/selfcheckgpt)](https://pepy.tech/project/selfcheckgpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A zero-resource, black-box framework for detecting hallucinations in LLM outputs.

**How it works:** Generate multiple passages from the same LLM, then check for consistency. Sentences that contradict most samples are flagged as potentially hallucinated.

- Project page for our paper "[SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896)"
- [Oct 2023] The paper is accepted and to appear at EMNLP 2023 [\[Poster\]](https://drive.google.com/file/d/1EzQ3MdmrF0gM-83UV2OQ6_QR1RuvhJ9h/view?usp=drive_link)

![](demo/selfcheck_qa_prompt.png)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scoring Methods](#scoring-methods)
  - [NLI (Recommended)](#selfcheckgpt-usage-nli-recommended)
  - [LLM Prompt](#selfcheckgpt-usage-llm-prompt)
  - [BERTScore, QA, N-gram](#selfcheckgpt-usage-bertscore-qa-n-gram)
- [New Features](#new-features)
  - [Fine-Grained Clause-Level Scoring](#fine-grained-clause-level-scoring)
  - [Prompt Caching](#prompt-caching)
  - [Multi-Domain Data Loaders](#multi-domain-data-loaders)
- [Experiments](#experiments)
  - [FEVER Experiment](#fever-experiment)
  - [Multi-Domain Evaluation](#multi-domain-evaluation)
  - [Fine-Grained Demo](#fine-grained-demo)
- [Running Tests](#running-tests)
- [Dataset](#dataset)
- [Citation](#citation)

---

## Installation

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

**Optional:** For FEVER experiments with Groq API:
```bash
pip install python-dotenv
```

---

## Quick Start

Verify your installation works:

```bash
# Run all tests (31 tests, no model downloads required)
pytest tests/ -v

# Run the fine-grained demo (uses toy scorer)
python demo/experiments/fine_grained/fine_grained_demo.py
```

Basic usage with NLI scorer:

```python
import torch
import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm")
selfcheck_nli = SelfCheckNLI(device=device)

# Text to evaluate
passage = "Paris is the capital of France. It has a population of 10 million."
sentences = [sent.text.strip() for sent in nlp(passage).sents]

# Sampled passages from the same LLM (with temperature > 0)
samples = [
    "Paris is the capital of France. It has a population of 2 million.",
    "Paris is the capital of France. The city has about 12 million people.",
]

# Get inconsistency scores (higher = more likely hallucinated)
scores = selfcheck_nli.predict(sentences=sentences, sampled_passages=samples)
print(scores)
# [0.01, 0.45] -- second sentence is inconsistent across samples
```

---

## Scoring Methods

### SelfCheckGPT Usage: NLI (Recommended)

Uses DeBERTa-v3-large fine-tuned on Multi-NLI. Returns P(contradiction) as the inconsistency score.

```python
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device)

sent_scores_nli = selfcheck_nli.predict(
    sentences = sentences,
    sampled_passages = [sample1, sample2, sample3],
)
print(sent_scores_nli)
# [0.334014 0.975106] -- higher means more likely hallucinated
```

### SelfCheckGPT Usage: LLM Prompt

Prompts an LLM to assess whether each sentence is supported by the sampled passages.

```python
# Option 1: Open-source model (HuggingFace)
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_prompt = SelfCheckLLMPrompt("mistralai/Mistral-7B-Instruct-v0.2", device)

# Option 2: API access (OpenAI or Groq)
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")
# Or with Groq:
selfcheck_prompt = SelfCheckAPIPrompt(client_type="groq", model="llama3-70b-8192", api_key="your-key")

sent_scores_prompt = selfcheck_prompt.predict(
    sentences = sentences,
    sampled_passages = [sample1, sample2, sample3],
    verbose = True,
)
# Yes -> 0.0, No -> 1.0, N/A -> 0.5
```

### SelfCheckGPT Usage: BERTScore, QA, N-gram

```python
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram

# BERTScore variant
selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
sent_scores_bertscore = selfcheck_bertscore.predict(
    sentences = sentences,
    sampled_passages = [sample1, sample2, sample3],
)

# N-gram variant (scores not bounded to [0,1])
selfcheck_ngram = SelfCheckNgram(n=1)  # n=1 is Unigram, n=2 is Bigram
sent_scores_ngram = selfcheck_ngram.predict(
    sentences = sentences,
    passage = passage,
    sampled_passages = [sample1, sample2, sample3],
)
# Returns: {'sent_level': {'avg_neg_logprob': [...], 'max_neg_logprob': [...]}, 'doc_level': {...}}

# MQAG (Question-Answering) variant
selfcheck_mqag = SelfCheckMQAG(device=device)
sent_scores_mqag = selfcheck_mqag.predict(
    sentences = sentences,
    passage = passage,
    sampled_passages = [sample1, sample2, sample3],
    num_questions_per_sent = 5,
    scoring_method = 'bayes_with_alpha',
    beta1 = 0.8, beta2 = 0.8,
)
```

---

## New Features

### Fine-Grained Clause-Level Scoring

Split sentences into clauses/phrases before scoring, then aggregate back to sentence level. Useful for catching partial hallucinations within long sentences.

```python
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from selfcheckgpt.orchestrator import score_with_decomposition

scorer = SelfCheckNLI(device="cpu")

sentences = [
    "Saturn is the sixth planet from the sun, and it is known for its rings.",
    "The gas giant weighs about 100 kilograms.",  # Partially wrong!
]
samples = ["Saturn is the sixth planet with famous rings and enormous mass."]

# Score with clause decomposition
result = score_with_decomposition(
    scorer=scorer,
    sentences=sentences,
    sampled_passages=samples,
    use_decomposition=True,   # Enable clause splitting
    aggregation="mean",       # or "max"
)

print(result["scores"])              # Aggregated sentence scores
print(result["chunks_by_sentence"])  # Clauses per sentence
print(result["chunk_scores_by_sentence"])  # Scores per clause
```

**Example output:**
```
=== Fine-grained scoring ===
Sentence 1: Saturn is the sixth planet from the sun, and it is known for its rings.
  Aggregated score: 0.762
    - 1.000 :: Saturn is the sixth planet from the sun,
    - 0.429 :: and it is
    - 0.857 :: known for its expansive ring system.

Sentence 2: The gas giant weighs about 100 kilograms.
  Aggregated score: 0.619
    - 0.429 :: The gas giant
    - 1.000 :: The gas giant weighs about 100 kilograms.
```

### Prompt Caching

Reduce API costs and latency by caching prompt responses to disk.

```python
# Open-source LLM with caching
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

scorer = SelfCheckLLMPrompt(
    model="meta-llama/Llama-2-7b-chat-hf",
    use_cache=True,
    cache_dir=".cache/selfcheckgpt/llm",
    prompt_batch_size=4,  # Batch tokenization + generation
)

scores = scorer.predict(sentences, samples)
# Subsequent runs with same inputs will use cached results

# API-based with caching
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

scorer = SelfCheckAPIPrompt(
    client_type="openai",
    model="gpt-4o-mini",
    use_cache=True,
    cache_dir=".cache/selfcheckgpt/api",
)

scores = scorer.predict(
    sentences,
    samples,
    prompt_batch_size=5,  # Concurrent API requests
)
```

Cache files are stored as `<cache_dir>/<sha256>.json` where the hash covers the prompt, model, and input text.

### Multi-Domain Data Loaders

Load evaluation datasets from local JSONL files or HuggingFace with pluggable loaders.

```python
from selfcheckgpt import data

# List available domains
print(data.list_loaders())
# ['events', 'organizations', 'places']

# Load examples
examples = data.load("events", split="dev", limit=10)
for ex in examples:
    print(ex["id"], ex["prompt"])
    print("Reference:", ex["reference"])
    print("Samples:", ex["samples"])

# Override data source at runtime
data.configure_loader(
    "events",
    local_path="~/my_data/events.jsonl",
    # Or use HuggingFace:
    # hf_dataset="my-org/events-dataset",
    # hf_subset="v1",
)
```

**Expected JSONL schema:**
```json
{"id": "event_1", "prompt": "Describe Apollo 11.", "reference": "Apollo 11 launched...", "samples": ["Apollo 11...", "NASA's mission..."]}
```

---

## Experiments

### FEVER Experiment

Evaluate SelfCheck-NLI on FEVER claims (fact verification dataset).

**Pre-computed results** are available in `demo/experiments/fever/results/nli_results.jsonl`.

**Sample Results (20 FEVER claims):**

| Claim | Label | Avg NLI Score |
|-------|-------|---------------|
| Paris is the capital of France. | SUPPORTS | 0.067 |
| The Earth is flat. | REFUTES | 0.158 |
| Water boils at 100°C at sea level. | SUPPORTS | 0.191 |
| Albert Einstein developed relativity. | SUPPORTS | 0.291 |

**Interpretation:**
- **Low scores (< 0.1)**: LLM responses are consistent across samples → likely factual
- **High scores (> 0.3)**: LLM responses contradict each other → potential hallucination or ambiguity

**Run the experiment yourself:**

```bash
# Using Groq API (fast, requires API key)
export GROQ_API_KEY=your-key
cd demo/experiments/fever
python run_groq.py

# Using local Ollama (requires ollama installed)
cd demo/experiments/fever
python run.py
```

### Multi-Domain Evaluation

Run SelfCheck scorers across multiple domains (events, places, organizations).

```bash
# Basic run with n-gram scorer (CPU-friendly, no model downloads)
python demo/experiments/multidomain/run.py \
  --domains events places organizations \
  --split dev \
  --scorers ngram \
  --output results.jsonl \
  --metrics metrics.json

# With NLI scorer (requires model download)
python demo/experiments/multidomain/run.py \
  --domains events places \
  --scorers nli \
  --device cuda \
  --limit 5

# Override data sources
python demo/experiments/multidomain/run.py \
  --domains events \
  --events-path ~/my_data/events/ \
  --scorers ngram prompt_stub
```

**Available scorers:** `ngram`, `nli`, `bertscore`, `prompt_stub`

### Fine-Grained Demo

Demonstrates clause-level decomposition without heavy models:

```bash
python demo/experiments/fine_grained/fine_grained_demo.py
```

---

## Running Tests

The test suite includes 31 tests covering all scorers and new modules:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=selfcheckgpt --cov-report=term-missing
```

**Test breakdown:**
- `test_smoke_placeholders.py`: 6 mocked tests for core scorers (MQAG, BERTScore, N-gram, NLI, LLM Prompt, API Prompt)
- `test_new_modules.py`: 25 tests for decomposition, orchestrator, prompt caching, data loaders

All tests run on CPU without downloading models.

---

## Dataset

The `wiki_bio_gpt3_hallucination` dataset consists of 238 annotated passages. 

### Load via HuggingFace
```python
from datasets import load_dataset
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
```

### Manual Download
Download from [Google Drive](https://drive.google.com/file/d/1AyQ7u9nYlZgUZLm5JBDx6cFFWB__EsNv/view?usp=share_link).

Each instance contains:
- `gpt3_text`: GPT-3 generated passage
- `wiki_bio_text`: Actual Wikipedia passage
- `gpt3_sentences`: Sentences split using spaCy
- `annotation`: Human annotation at sentence level
- `gpt3_text_samples`: Sampled passages (temperature=1.0)

---

## Experimental Results

Results on the `wiki_bio_gpt3_hallucination` dataset:

| Method               |  NonFact (AUC-PR)  |  Factual (AUC-PR)  |   Ranking (PCC)   |
|----------------------|:------------------:|:------------------:|:-----------------:|
| Random Guessing      |        72.96       |        27.04       |         -         |
| GPT-3 Avg(-logP)     |        83.21       |        53.97       |       57.04       |
| SelfCheck-BERTScore  |        81.96       |        44.23       |       58.18       |
| SelfCheck-QA         |        84.26       |        48.14       |       61.07       |
| SelfCheck-Unigram    |        85.63       |        58.47       |       64.71       |
| SelfCheck-NLI        |        92.50       |        66.08       |       74.14       |
| SelfCheck-Prompt (Llama2-7B)     |   89.05   |   63.06   |   61.52   |
| SelfCheck-Prompt (Llama2-13B)    |   91.91   |   64.34   |   75.44   |
| SelfCheck-Prompt (Mistral-7B)    |   91.31   |   62.76   |   74.46   |
| **SelfCheck-Prompt (gpt-3.5-turbo)** | **93.42** | **67.09** | **78.32** |

---

## Project Structure

```
selfcheckgpt/
├── modeling_selfcheck.py      # Core scorers (MQAG, BERTScore, N-gram, NLI, LLM Prompt)
├── modeling_selfcheck_apiprompt.py  # API-based prompt scorer (OpenAI, Groq)
├── modeling_mqag.py           # Question generation & answering
├── modeling_ngram.py          # N-gram models
├── decomposition.py           # Clause-level splitting (spaCy + heuristics)
├── orchestrator.py            # Fine-grained scoring with aggregation
├── prompt_utils.py            # Prompt caching utilities
├── data.py                    # Multi-domain data loaders
└── utils.py                   # Helper functions

demo/experiments/
├── fever/                     # FEVER claim evaluation
├── multidomain/               # Multi-domain runner + data
├── fine_grained/              # Clause decomposition demo
└── prompt_caching/            # Caching documentation

tests/
├── test_smoke_placeholders.py # Core scorer tests (mocked)
└── test_new_modules.py        # New module tests
```

---

## Acknowledgements

This work is supported by Cambridge University Press & Assessment (CUP&A), a department of The Chancellor, Masters, and Scholars of the University of Cambridge, and the Cambridge Commonwealth, European & International Trust.

---

## Citation

```bibtex
@article{manakul2023selfcheckgpt,
  title={Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark JF},
  journal={arXiv preprint arXiv:2303.08896},
  year={2023}
}
```
