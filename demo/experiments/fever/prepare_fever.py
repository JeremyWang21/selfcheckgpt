import json
import random

OUT = "fever_dev_20.jsonl"  

NUM_SAMPLES = 20 # Run time considerations

# Creates a random sample of 200 fever claims for use
def main():
    claims = []

    with open("fever_dev.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            claim = ex.get("claim")
            label = ex.get("label")

            if not claim or not label:
                continue

            label = label.upper()
            if label not in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
                continue

            claims.append({"claim": claim, "label": label})

    print(f"Loaded {len(claims)} FEVER items")

    random.seed(42)
    subset = random.sample(claims, min(NUM_SAMPLES, len(claims)))

    with open(OUT, "w", encoding="utf-8") as out:
        for ex in subset:
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(subset)} items to {OUT}")

if __name__ == "__main__":
    main()
