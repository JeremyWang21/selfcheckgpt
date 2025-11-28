import json
import spacy
import torch
import subprocess

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check for Torch GPU avail
nlp = spacy.load("en_core_web_sm") # Use spacy nlp model for sentence splitting
checker = SelfCheckNLI(device=device) # Creates NLI checker obj
 
# Return clean sentences for llm output
def split_sentences(text):
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

# Method to call LLM
def call_llm(prompt, temperature=0.0):
    command = ["ollama", "run", "llama3:8b"]
    result = subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",   # 👈 force UTF-8
        errors="ignore"     # 👈 drop any dodgy bytes instead of crashing
    )
    return result.stdout.strip()

# Load FEVER slice
# e.g. Fever Example: {"claim": "Paris is the capital of France.", "label": "SUPPORTS"}
with open("fever_20.jsonl", "r") as f:
    fever_items = [json.loads(l) for l in f]

# Result Analysis
results = []

for ex in fever_items:
    claim = ex["claim"]
    label = ex["label"]

    # Builds prompt for clami judging
    prompt = f"Is this statement true? Explain.\n\n{claim}"
    R = call_llm(prompt, temperature=0.0)
    sentences_R = split_sentences(R)

    # Comparison baselines
    samples = [
        call_llm(prompt, temperature=1.0)
        for _ in range(4)
    ]

    # run NLI SelfCheck
    nli_scores = checker.predict(
        sentences=sentences_R,
        sampled_passages=samples,
    )

    results.append({
        "claim": claim,
        "label": label,
        "R": R,
        "samples": samples,
        "sentences_R": sentences_R,
        "nli_scores": list(map(float, nli_scores)),
    })

# Save results
with open("results/nli_results.jsonl", "w") as out:
    for r in results:
        out.write(json.dumps(r) + "\n")

print("results saved")
