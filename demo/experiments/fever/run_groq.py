import json
import os
import spacy
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

# Load environment variables
load_dotenv()

# Setup Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"  # Fast model, or "llama-3.1-70b-versatile" for better accuracy

# Device selection with MPS support for Apple Silicon
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

nlp = spacy.load("en_core_web_sm")
checker = SelfCheckNLI(device=device)

def split_sentences(text):
    """Return clean sentences for LLM output"""
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

def call_llm(prompt, temperature=0.7):
    """Call Groq API - much faster than local Ollama"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error: {e}")
        return ""

def process_claim(claim, label):
    """Process a single claim with parallel API calls"""
    prompt = f"Is this statement true? Explain.\n\n{claim}"
    
    # Make 4 parallel API calls (1 main response + 3 samples)
    num_samples = 4
    with ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = [executor.submit(call_llm, prompt, 0.7) for _ in range(num_samples)]
        responses = [f.result() for f in futures]
    
    # Filter out empty responses
    responses = [r for r in responses if r]
    if len(responses) < 2:
        print(f"Warning: Not enough valid responses for claim: {claim[:50]}...")
        return None
    
    R = responses[0]  # Main response
    samples = responses[1:]  # Sampled responses for comparison
    sentences_R = split_sentences(R)
    
    if not sentences_R:
        print(f"Warning: No sentences extracted from response for claim: {claim[:50]}...")
        return None
    
    # Run NLI SelfCheck
    nli_scores = checker.predict(
        sentences=sentences_R,
        sampled_passages=samples,
    )
    
    return {
        "claim": claim,
        "label": label,
        "R": R,
        "samples": samples,
        "sentences_R": sentences_R,
        "nli_scores": list(map(float, nli_scores)),
    }

def main():
    # Load FEVER slice
    with open("fever_20.jsonl", "r") as f:
        fever_items = [json.loads(l) for l in f if l.strip()]
    
    print(f"Processing {len(fever_items)} claims...")
    
    results = []
    for ex in tqdm(fever_items, desc="Processing claims"):
        result = process_claim(ex["claim"], ex["label"])
        if result:
            results.append(result)
            # Print intermediate result
            avg_score = sum(result["nli_scores"]) / len(result["nli_scores"])
            print(f"\n  Claim: {result['claim'][:60]}...")
            print(f"  Label: {result['label']}, Avg NLI Score: {avg_score:.3f}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/nli_results.jsonl", "w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")
    
    print(f"\n✅ Results saved to results/nli_results.jsonl")
    print(f"   Processed {len(results)}/{len(fever_items)} claims successfully")

if __name__ == "__main__":
    main()
