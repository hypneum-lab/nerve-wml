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
c-geni-al contributors
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
Source repository: https://github.com/c-geni-al/nerve-wml (tag
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
- `https://github.com/c-geni-al/nerve-wml` (source)

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

## 5. Checklist

- [ ] `paper-v0.9-draft` tag exists on master
- [ ] `main.pdf` compiles clean (tectonic, no errors)
- [ ] Both figures embedded and readable
- [ ] Abstract under 1500 chars
- [ ] Zenodo DOI minted for v1.1.2 or later
- [ ] arXiv package tar created
- [ ] Endorsement obtained (if required)
- [ ] `related_identifiers` in .zenodo.json link to OSF
- [ ] `preferred-citation` in CITATION.cff updated post-arXiv-ID
