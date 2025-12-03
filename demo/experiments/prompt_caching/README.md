# Prompt caching & batching

This branch wires optional caching/batching into the prompt-based scorers so you can trade disk space for lower latency and cost. All knobs are opt-in and default to the original behavior.

## Cache key strategy

Each prompt response is stored as `<cache_dir>/<sha256>.json` where the hash covers:

- scorer name (`SelfCheckLLMPrompt` or `SelfCheckAPIPrompt`)
- model identifier and client type (`hf` vs API provider)
- sentence/sample indices and their raw text
- the exact prompt string delivered to the model

Because sentence/sample IDs and prompt text are part of the key, retries that stem from different batches or calling code still re-use the cached artifact.

## Example usage

### Open-source prompts

```
python - <<'PY'
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

sentences = ["Paris is the capital of France."]
samples = ["Paris is the capital of France and home to the Eiffel Tower."]

scorer = SelfCheckLLMPrompt(
    model="meta-llama/Llama-2-7b-chat-hf",
    use_cache=True,
    cache_dir=".cache/selfcheckgpt/llm",
    prompt_batch_size=4,   # batches tokenization+generate calls
)

scores = scorer.predict(
    sentences,
    samples,
)
print(scores)
PY
```

### API prompts (OpenAI/Groq)

```
export OPENAI_API_KEY=sk-...
python - <<'PY'
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

sentences = ["Paris is the capital of France."]
samples = ["Paris is the capital of France and home to the Eiffel Tower."]

scorer = SelfCheckAPIPrompt(
    client_type="openai",
    model="gpt-4o-mini",
    use_cache=True,
    cache_dir=".cache/selfcheckgpt/api",
)

scores = scorer.predict(
    sentences,
    samples,
    prompt_batch_size=5,   # threads 5 concurrent API requests
)
print(scores)
PY
```

### Overriding per call

Both scorers also accept the same kwargs on `predict`, which lets you toggle caching or change the directory for a single run:

```
scores = scorer.predict(
    sentences,
    samples,
    use_cache=True,
    cache_dir="/tmp/prompt-cache",
)
```

To clear cached responses, delete the configured cache directory (default `.cache/selfcheckgpt/`).
