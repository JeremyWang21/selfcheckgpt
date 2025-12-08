
# Benchmark Results Summary

## WikiBio (50 passages)

| Method | NonFact AUC-PR | Factual AUC-PR | Pearson | Paper NonFact | Paper Factual | Paper Pearson |
|--------|----------------|----------------|---------|---------------|---------------|---------------|
| nli | 92.25 | 59.24 | 49.98 | 92.5 | 66.08 | 74.14 |
| ngram | 83.02 | 38.01 | 19.84 | - | - | - |
| bertscore | 82.56 | 35.76 | 17.57 | 81.96 | 44.23 | 58.18 |

- Runtime: 3095.8s
- Sentences scored (NLI): 423
- Passages scored: 50

## FEVER (50 claims)

| Split | Mean NLI | Std NLI | Count |
|-------|----------|---------|-------|
| SUPPORTS | 0.1059 | 0.1787 | 24 |
| REFUTES  | 0.1831 | 0.2282 | 26 |

- Separation AUC (SUPPORTS vs REFUTES): 0.6795
- Cohen's d: 0.3770
- Runtime: 919.5s
- Processed: 50/50 claims

## Notes
- WikiBio NLI NonFact AUC-PR (ours): 92.25 vs paper 92.50
- FEVER shows expected behavior: SUPPORTS mean NLI < REFUTES mean NLI
- Some Groq rate-limit 429 warnings; still completed 50 claims
