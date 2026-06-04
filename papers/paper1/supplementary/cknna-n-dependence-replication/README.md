# CKNNA N-dependence — supplementary replication (Paper 1)

Supplementary material for nerve-wml Paper 1. Empirical replication
of two predictions from Gröger, Wen & Brbić (2026,
arXiv:2602.14486) — Proposition 4.2 ($\mathbb{E}[\mathrm{mKNN}(X,Y)]
= k/(n-1)$) and the calibrated-effect-size argument — on a
synthetic Gaussian substrate disjoint from their image-text PRH
cells. This independent empirical validation (conducted on orthogonal
data and method) confirms the null-calibration framework while
providing supplementary context for the main paper's CKNNA discussion.
Originally drafted as a standalone technical note; repurposed post hoc
as paper-1 supplementary material after [@aristotelianprh2026]
published the formal framework.

The two empirical observations:

- **(D1)** Raw CKNNA decreases monotonically in $N$ at constant
  signal, consistent with the $k/(n-1)$ null baseline.
- **(D2)** Paired Cohen's $d_z$ (signal vs shuffled-null) grows
  monotonically in $N$ on the same data, inverting the direction of
  the apparent effect.

Sweep: 1750 cells covering 7 values of $N$ spanning ~1.8 decades (6
doublings, $N \in \{256, 512, 1024, 2048, 4096, 8192, 16384\}$) × 5
noise levels × 50 seeds.

## Contents

| File                          | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| `supplementary.md`            | The markdown text (pandoc → PDF)         |
| `refs.bib`                    | BibTeX entries cited by `supplementary.md` |
| `make_figures.py`             | Regenerates the two figures from JSON    |
| `figures/fig1_cknna_decay.png`| CKNNA@10 vs N at σ=0.05 (real + null)    |
| `figures/fig2_dz_growth.png`  | $d_z$ vs N (log–log), CKNNA + HSIC       |

## Data source

All numerical content is derived from the 1750-cell evaluation
committed in the repository root at:

    docs/superpowers/research/2026-05-20-macm1-scientific-eval.json

The figure script reads that file directly; the tables in
`supplementary.md` were transcribed from the companion summary
`docs/superpowers/research/2026-05-20-macm1-scientific-eval.md`.

## Regenerate figures

From the repository root:

```bash
uv run --with matplotlib python \
    papers/paper1/supplementary/cknna-n-dependence-replication/make_figures.py
```

The script prints the headline numbers (CKNNA@10 mean ± SD per N at
σ=0.05) for cross-checking against Table 1 of the supplementary.

## Build PDF

```bash
cd papers/paper1/supplementary/cknna-n-dependence-replication
pandoc supplementary.md \
    --bibliography=refs.bib \
    --citeproc \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V mainfont="Times New Roman" \
    -o supplementary.pdf
```

Pandoc ≥ 3 is recommended.

## Status

- [x] All numbers traceable to the committed JSON
- [x] Figures regenerate without error from real data
- [x] Framed as empirical replication of [@aristotelianprh2026]
- [x] Numerical range corrected to ~1.8 decades (6 doublings, not 7
      decades)
- [x] No standalone-arXiv references remain in body text
