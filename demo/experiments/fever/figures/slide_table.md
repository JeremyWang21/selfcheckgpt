# FEVER Expansion Results - Slide-Ready Table

## Metrics Comparison Table (Actual Results)

| Metric | FEVER (Ours) | WikiBio (Original) | Δ |
|--------|:------------:|:------------------:|:-:|
| Non-Fact AUC-PR | **0.59** | 0.65 | -0.06 |
| Factual AUC-PR | **0.54** | 0.66 | -0.12 |
| Pearson ρ | **~0** | 0.62 | -0.62 |
| Spearman ρ | **~0** | 0.70 | -0.70 |

## Key Takeaways for Speaking

1. **Non-Fact AUC-PR 0.59**: Close to random but slightly better at catching false claims
2. **Factual AUC-PR 0.54**: Slightly above random for identifying true claims  
3. **Near-zero correlation**: Llama-3 gives **equally confident, consistent explanations** for both true AND false claims

## Why Low Correlation?

This is actually an **important finding**:
- SelfCheck-NLI works by detecting *inconsistency* across samples
- Llama-3 is **consistently wrong** on false claims (always says "No, that's false...")
- Llama-3 is **consistently right** on true claims (always says "Yes, that's true...")
- Both produce **low NLI scores** (high consistency) → method can't distinguish!

## Example Claims from Your Run

### High score examples (flagged as potential hallucinations):
| Claim | Label | Score |
|-------|-------|-------|
| "Bananas grow on trees." | REFUTES ✓ | 0.98 |
| "Pluto is classified as a dwarf planet." | SUPPORTS ✗ | 0.96 |
| "Napoleon Bonaparte was unusually short." | REFUTES ✓ | 0.90 |

### Low score examples (flagged as factual):
| Claim | Label | Score |
|-------|-------|-------|
| "Mount Everest is the tallest mountain..." | SUPPORTS ✓ | 0.35 |
| "Paris is the capital of France." | SUPPORTS ✓ | 0.34 |
| "Bulls are angered by the color red." | REFUTES ✗ | 0.19 |

## Key Point for Presentation

> "SelfCheck-NLI detects **model uncertainty**, not factual accuracy.  
> When the LLM is **confidently wrong**, the method fails.  
> This explains the performance gap between FEVER and WikiBio."
