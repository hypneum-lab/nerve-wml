# arXiv submission checklist — nerve-wml Paper 1

Step-by-step guide for submitting `papers/paper1/main.pdf` to arXiv
based on the v0.9 draft (tag `paper-v0.9-draft`).

## 1. Prepare the upload package

The arXiv submission should contain:

- `main.tex` (LaTeX source)
- `figures/w2_hard_scaling.pdf`
- `figures/info_transmission.pdf`
- any additional figures referenced by `\includegraphics`
- `main.bbl` (generated — arXiv does NOT run bibtex; ship the .bbl)

From the repo root:

```bash
cd papers/paper1
tectonic main.tex --keep-intermediates   # generates main.bbl
tar czf /tmp/nerve-wml-arxiv-v0.9.tar.gz main.tex main.bbl figures/
```

## 2. Upload + metadata

Go to <https://arxiv.org/submit>. Upload the tarball.

### Title

```
Substrate-Agnostic Nerve Protocol for Inter-Module Communication in Hybrid Neural Systems
```

### Authors

```
Clément Saillant (L'Electron Rare, Grandris, France)
```

Include ORCID if available.

### Abstract

Paste verbatim from `main.tex` lines 16–45 (abstract environment).
1500-char limit on arXiv — the v0.9 abstract is under 1400 chars.

### Primary category

`cs.NE` (Neural and Evolutionary Computing)

### Cross-listing (secondary)

- `cs.LG` (Machine Learning) — MI / transmission measurements
- `q-bio.NC` (Neurons and Cognition) — γ/θ multiplexing biological
  motivation

### License

`CC BY 4.0` (matches the code + docs licence combination: MIT code,
CC-BY-4.0 docs — arXiv's CC-BY-4.0 is the standard for the paper PDF).

### Comments field

```
15 pages, 2 figures. Code and reproducibility scripts archived at
Zenodo (DOI: <fill-in-after-zenodo-mint>), linked to the parent
dreamOfkiki programme pre-registration (OSF DOI 10.17605/OSF.IO/Q6JYN).
Source repository: https://github.com/hypneum-lab/nerve-wml (tag
v1.1.4). All numeric claims reproducible via uv run pytest with
explicit seeds.
```

### Journal-ref / DOI (if applicable)

Leave blank for preprint. Can be added via `arXiv Replace` after
peer-reviewed acceptance.

### Related identifiers

In the arXiv "Link external resources" section, add:

- `10.5281/zenodo.<code>` (nerve-wml v1.1.4 Zenodo DOI — fill after mint)
- `10.17605/OSF.IO/Q6JYN` (dreamOfkiki OSF pre-registration)
- `https://github.com/hypneum-lab/nerve-wml` (source)

## 3. Endorsement

For first-time submissions to `cs.NE`, arXiv requires endorsement.
Two paths:

- **If you have affiliated co-authors** with arXiv history in cs.NE,
  they can endorse you directly from the submission UI.
- **Otherwise**, find a recent cs.NE author whose work you cite and
  request endorsement via `arxiv.org/auth/request-endorsement`.

Paper 1 cites Rao & Ballard (1999), Bastos et al. (2012), van den
Oord et al. (2017), Zeghidour et al. (2022), Neftci et al. (2019) —
any of these groups' active members would be natural endorsers.

## 4. Post-submission

After arXiv assigns an ID (e.g. `arXiv:2604.XXXXX`):

1. Update `CITATION.cff` → add `preferred-citation.url` pointing to arXiv.
2. Update `.zenodo.json` `related_identifiers` → add arXiv DOI.
3. Update `README.md` DOI badge with the arXiv one alongside Zenodo.
4. Tag a `paper-v0.9-arxiv` to mark the version submitted.
5. Cut a GitHub Release mentioning the arXiv number.

## 5. Post-Sprint Status (2026-05-20)

**Pre-arXiv Blockers: ALL RESOLVED** ✓

- [x] Cycle trace figure regeneration (scripts/render_paper_figures.py)
- [x] Supplementary material merged (cknna-n-dependence-replication framed as orthogonal Gröger 2026 validation)
- [x] References audit (24/24 resolved, 0 undefined citations)
- [x] main.pdf compiles clean (pdflatex, 15 pages, 536 KB)
- [x] Both figures embedded and readable
- [x] Abstract under 1500 chars (~1390 chars)
- [x] Fact-check audit protocol implemented (CI scripts/factcheck_audit.py, 27 checks OK / 0 DIVERGENT)

**Claims Revision Record** (Gap-analysis Sprint, PR #28–#29 merged)

Six empirical claims revised per post-sprint validation:

1. **Synchrony metric invalidated** — top-PC anti-monotone (null 0.32 > GTM 0.20 > simple_gating 0.08); replaced with spectral_entropy
2. **CKNNA N-dependence artefact** — continuous-kernel nearest-neighbour decay (0.92→0.87, N=256→16384) at constant signal; cross-paper comparisons biased; scooped Gröger 2026 Prop 4.2
3. **HSIC debiased uninformative** — linear-CKA raw |r| ≤ 0.015 vs all metrics (1750 cells); replaced with spectral_entropy
4. **AKOrN well-parametrized outperforms GTM** — n_osc=64, lr=0.05: 0.45±0.16 vs GTM 0.20; accuracy bimodal (phase-aligner, not encoder)
5. **Relative repeatability ≈ learned** — non-dominant equality (p=0.627, d_z=−0.25 vs strict hierarchy hypothesis)
6. **Spectral entropy metric adopted** — monotone proxy (null 1.96 < akorn 1.97 < gtm 2.17 < simple_gating 3.46, p=1.78e-15); canonical ordering at batch=256

**Seven Key Findings** (research-notes & supplementary)

1. **MLX 2× PyTorch MPS** on M1 (1478ms vs 3059ms CKNNA Gram matrix)
2. **M3 Pro 2–3× M1 Max** per-core throughput (Python+torch multi-proc)
3. **Cross-host bit-exact on 3/4 substrates** (macm1 ≡ macM3: GTM, simple_gating, null; AKOrN diverges Δacc=+0.16)
4. **Signal/null ratio metric** cross-paper-comparable, monotone in N (inverse of CKNNA artefact)
5. **MLX issue #3568** (mx.random.normal M1 g13s vs M3/M5 g15+ divergence, erfinv FMA branch B |log(1-a²)|>6.125)
6. **Spectral entropy B-sensitive** (canonical ordering only batch=256; paper caveat Renf-13)
7. **AKOrN sub-parameterization risk** (n_osc=64 recovery after minimal sweep)

## 6. Checklist

- [x] `paper-v0.9-draft` tag exists on master
- [x] `main.pdf` compiles clean (pdflatex, no errors)
- [x] Both figures embedded and readable
- [x] Abstract under 1500 chars
- [x] Zenodo DOI minted (v1.1.4, post-sprint)
- [x] arXiv package tar creatable (main.tex + main.bbl + figures/)
- [ ] Endorsement obtained (if required, post-submission)
- [ ] `related_identifiers` in .zenodo.json link to OSF
- [ ] `preferred-citation` in CITATION.cff updated post-arXiv-ID
