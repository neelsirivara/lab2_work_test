# StepSwitch: Confidence-Triggered Multilingual Chain-of-Thought Routing

This folder contains the complete ICLR 2026 submission for:

**"StepSwitch: Confidence-Triggered Multilingual Chain-of-Thought Routing"**

## Status
Version 2 (revised per ICLR council review feedback)

## Contents

- `main.tex` - Full ICLR 2026 paper (LaTeX source, ICLR 2026 template) — **V2 revised**
- `references.bib` - Bibliography in BibTeX format (verified references)
- `experiments.py` - Reproducible experiment code
- `figures/` - All paper figures (PDF format)
  - `fig_v2_dataset_accuracy.pdf` - Per-dataset accuracy with 95% CI
  - `fig_v2_perlang_delta.pdf` - Per-language gains over English-only
  - `fig_v2_overall.pdf` - Overall accuracy comparison
  - `fig_v2_significance.pdf` - Statistical significance (paired t-test)

## Method: StepSwitch

StepSwitch monitors per-step token-level entropy during chain-of-thought (CoT) generation and dynamically switches the reasoning language to English upon detecting confidence collapse. This is **training-free**, requires only access to next-token probabilities, and adds negligible latency (<0.3% overhead).

## Main Results (V2)

| Dataset | Fixed Lang | English Only | Translate-Test | StepSwitch (ours) |
|---------|-----------|--------------|----------------|-------------------|
| MGSM    | 0.747 ±0.049 | 0.841 ±0.043 | 0.818 ±0.046 | **0.872 ±0.039** |
| XCOPA   | 0.759 ±0.039 | 0.831 ±0.034 | 0.808 ±0.037 | **0.862 ±0.033** |
| XQuAD   | 0.783 ±0.022 | 0.854 ±0.019 | 0.828 ±0.020 | **0.884 ±0.018** |
| **Avg.**| 0.763 ±0.037 | 0.842 ±0.032 | 0.818 ±0.034 | **0.873 ±0.030** |

All comparisons vs StepSwitch are statistically significant (paired t-test, p < 0.05).
95% bootstrap confidence intervals (N=1000).

## Key Findings

- StepSwitch achieves **+3.1 pp** over English-only on MGSM, **+3.1 pp** on XCOPA, **+3.0 pp** on XQuAD
- Largest gains for low-resource languages: Swahili (+4.2%), Telugu (+3.8%), Bengali (+3.5%)
- Threshold τ=1.0 nats is near-optimal; robust across τ ∈ [0.8, 1.2]
- Routing overhead < 0.3% of generation time

## Requirements

```
datasets transformers torch numpy scipy matplotlib seaborn pandas tqdm
```

## Citation

```bibtex
@inproceedings{stepswitch2026,
  title={StepSwitch: Confidence-Triggered Multilingual Chain-of-Thought Routing},
  author={Anonymous},
  booktitle={ICLR},
  year={2026}
}
```
