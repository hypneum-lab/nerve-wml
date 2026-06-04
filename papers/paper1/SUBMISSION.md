# arXiv submission bundle: nerve-wml paper

## Status

Final draft state at master `de05169` (post-supplementary-merge). Build
verified via `pdflatex + bibtex + 2x pdflatex` (TeX Live 2026) after
regenerating `figures/cycle_trace.pdf`.

- Output: **14 pages, 527,718 bytes** (`main.pdf`).
- Citations: 24/24 resolve; **zero** `Citation undefined` warnings.
- Boxes: zero Overfull/Underfull warnings.

## Pre-build figure regeneration

The repository's `.gitignore` excludes generated PDFs in
`papers/paper1/figures/`. Before any build, regenerate them via:

```bash
uv run --with matplotlib python -c "from scripts.render_paper_figures import render_cycle_trace; render_cycle_trace()"
```

(Add the other `render_*` calls per `scripts/render_paper_figures.py` if
those figures are also missing — the script regenerates each from
golden artefacts under `tests/golden/`.)

## Bundle composition for arXiv upload

Files to upload (tarball, top-level flat or with `figures/` subdir):

- `main.tex` (primary source)
- `refs.bib` (bibliography)
- `main.bbl` (compiled bibliography — arXiv accepts both `.bib` and
  `.bbl`; including `.bbl` avoids cross-platform pdflatex issues)
- `figures/*.{pdf,png}` (only those actually `\includegraphics`'d):
  - `figures/cycle_trace.pdf` (regenerate before upload — see above)
  - `figures/w4_forgetting.pdf`
  - `figures/p1_dead_curve.pdf`
  - `figures/w2_histogram.pdf`
  - `figures/w2_hard_scaling.pdf`
  - `figures/info_transmission.pdf`
  - `figures/mnist_scaling.pdf`
  - `figures/bigger_arch_scaling.pdf`
  - `figures/temporal_info_tx.pdf`

Files NOT in the bundle:

- `main.aux`, `main.log`, `main.toc`, `main.out` (auto-generated)
- `Makefile`, `README.md`, `SUBMISSION.md` (build/meta only)
- `figures/*.json` (raw experiment data, not figures)
- `figures/*.png` duplicates of the `.pdf` versions

## Build command (matches arXiv's pipeline)

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Local result: 12 pages, 515,695 bytes, zero `Citation undefined`.

## Cited references (24, alphabetical)

aer2025biohybrid, aristotelianprh2026, bastos2012canonical,
bastos2020routing, gabhart2025predictive, hinton2015distilling,
huh2024platonic, kornblith2019similarity, liu2021dvnc,
liu2024hybrid, morcos2018insights, moschella2022relative,
neftci2019surrogate, pedersen2024nir, peng2025gridlikevq,
platoscave2026, rao1999predictive, royvetterli2007erank,
ruffini2025comparator, saxe2024universality, vandenoord2017neural,
wei2024differank, zeghidour2022soundstream, zhao2025channelawarevq.

Zero orphan refs in `refs.bib`.

## Supplementary materials

A supplementary note `cknna-n-dependence-replication/` (empirical
replication of Gröger, Wen & Brbić 2026 Propositions 4.1/4.2 on a
synthetic Gaussian substrate, N up to 16,384) was prepared at commit
`8b5c634` on branch `origin/notes/cknna-scale-artefact` but has **not
yet been merged to master**. Decide before arXiv submission:

1. merge the branch and include `supplementary/` as a separate
   arXiv ancillary archive, or
2. submit paper-only now and add the supplementary in a v2 update.

## Submission metadata (for arXiv form)

- **Title**: Substrate-Agnostic Nerve Protocol for Inter-Module
  Communication in Hybrid Neural Systems
- **Author**: Saillant, Clément — L'Electron Rare, Grandris, France;
  Hypneum Lab — `clement@saillant.cc`
- **Primary category**: `cs.LG` (Machine Learning)
- **Secondary categories**: `cs.NE` (Neural and Evolutionary
  Computing), `q-bio.NC` (Neurons and Cognition)
- **License**: CC-BY-4.0 (suggested) or CC-BY-NC-4.0
- **Abstract**: copy-paste verbatim from `main.tex` `abstract`
  environment (lines 18–57).

## Pre-submission checklist

- [x] pdflatex builds clean (no `Citation undefined`)
- [ ] All figures present (one missing: `figures/cycle_trace.pdf`)
- [x] All `\cite{}` keys resolve in `refs.bib`
- [x] Limitations section explicit
- [x] Reproducibility section points to public commits + Zenodo DOIs
- [ ] Supplementary bundled separately (depends on merge decision)
- [ ] arXiv categories chosen
- [ ] License chosen
- [ ] Endorsement (if first arXiv submission)

## Known caveats to disclose in pre-submission review

- `spectral_entropy` 4-arm ordering is B-dependent (acknowledged in
  Limitations).
- AKOrN minimal flavour, not equivalent to Miyato et al. 2025.
- No real CL1/FinalSpark API key was used.
- `ml-explore/mlx#3568` documents `random.normal` cross-Apple-Silicon
  non-bit-exactness, relevant to the reproducibility section.
- `figures/cycle_trace.pdf` not committed; either regenerate before
  upload or drop the figure.
