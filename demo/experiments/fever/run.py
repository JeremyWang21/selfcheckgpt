import json
import spacy
import torch
import subprocess
import time

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
        encoding="utf-8",   
        errors="ignore"     
    )
    return result.stdout.strip()

# Load FEVER slice
# e.g. Fever Example: {"claim": "Paris is the capital of France.", "label": "SUPPORTS"}
with open("fever_dev_20.jsonl", "r", encoding="utf-8") as f:
    fever_items = [json.loads(l) for l in f]

# DROPS NOT ENOUGH INFORMATION CLAIMS
fever_items = [ex for ex in fever_items if ex["label"] in ("SUPPORTS", "REFUTES")]

# Result Analysis
results = []

start_time = time.time()
total_sentences = 0

for ex in fever_items:
    claim = ex["claim"]
    label = ex["label"]

    # Builds prompt for clami judging
    prompt = f"Is this statement true? Explain.\n\n{claim}"
    R = call_llm(prompt, temperature=0.0)
    sentences_R = split_sentences(R)
    total_sentences += len(sentences_R)

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

elapsed = time.time() - start_time

print(f"Processed {len(fever_items)} claims, {total_sentences} sentences in {elapsed:.1f} sec")
if total_sentences > 0:
    print(f"Runtime per 100 sentences: {elapsed / total_sentences * 100:.1f} sec")

# Save results
with open("results/nli_results_fever.jsonl", "w", encoding="utf-8") as out:
    for r in results:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")

print("results saved")
