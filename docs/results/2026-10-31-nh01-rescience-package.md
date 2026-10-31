# NH-01 ReScience C submission package and independent review (2026-10-31)

**Package:** [hypneum-lab/rescience-nh01](https://github.com/hypneum-lab/rescience-nh01) — article (ReScience C template, Markdown source + XeLaTeX build, 10 pages, 8 references), verbatim runners at tag `prereg-nh01-correlated-20261027`, the three frozen pre-registrations, the three result sets (`cells.jsonl.gz`, `summary.json`, `run_meta.json`, `figure.png`, SHA-256 manifest) and the 2026-10-30 provenance re-runs of the two N = 1 000 grids. Upstream code archive: `nerve-wml` v1.10.0, Zenodo concept DOI [10.5281/zenodo.19656342](https://doi.org/10.5281/zenodo.19656342).

**Scope:** partial replication of Fachechi, Agliari & Barra (2019), *Dreaming neural networks: forgetting spurious memories and reinforcing pure ones*, restricted to the zero-temperature retrieval claims; the finite-temperature and basin-of-attraction experiments of the original are not reproduced (see the article's deviation list).

## Independent review (2026-10-31)

Two automated review passes (Codex CLI, reasoning effort `xhigh`) were run before any human review; every finding was verified against the cells, the summaries, the runner code and the original paper before a change was made, and the PI decided the open points. Briefs, reports, verdict table and the pre-review article are published in `rescience-nh01/review/`.

| Pass | Model | Input | Output | Outcome |
|---|---|---|---|---|
| 1, blind | `gpt-6.1-sol` | repositories only | 10 596 words | 1 blocking statement (conclusion claimed every pattern is a fixed point up to load 0.95; pilot cells contradict it), 30 findings; every number re-derived independently and confirmed |
| 2, cross | `gpt-6-astra` | corrected package + verdict table (not the blind report) | 6 791 words | 20/30 corrections judged right, 8 insufficient, 0 wrong; 1 new error (Hebb spurious final states at load 0.14 = 51/320, not "almost only"); argued for reporting the registered m = 0.30 ratio next to its reservation |

Commits in `rescience-nh01`: `1f7c6b6` (pass 1), `d78590c` (pass 2), then the `review/` folder.

## What changes for the NH-01 reading in this repository

Nothing in `results/`, `docs/milestones/` or the runners. The wording of the m = 0.30 level is refined, not reversed:

- **C1 at T = 0 (registered):** the ratio 44.38 [42.56, 46.97] is the pre-registered statistic and is reported as **registered C1 confirmed**, with the reservation that every Hebbian crossing lies inside the lowest grid interval (loads 0.01–0.02), so the magnitude is grid-sensitive (30–65 under a bracketing reading). The earlier phrasing "magnitude not robust" stands; "non calculable" applies to C3′ at T = 0.2, not to C1.
- **C3′ at T = 0.2:** undefined on 5/10 seeds (Hebbian capacity below the grid); the zero-coded five-seed statistic 0.6 [0.2, 1.0] has two shrink numerators also below the grid. The `summary.json` label `confirme` for the level is the output of the tagged code and is overridden in the article: **post-analysis assessment inconclusive**, shown next to the registered label (commit `5884927` in the package).
- **Spurious states at T = 0, confirmatory grid:** Hebb 51/320 at α = 0.14 and 318/320 at α = 0.20; dreaming 0/320 at both. The earlier "almost only spurious" wording for α = 0.14 was wrong.
- **Pilot f = 0 arm:** right-censored at t ∈ {10, 30, 100} for N ≤ 256 (30/30 cells at the 0.95 ceiling), so no "in agreement" claim is made for that arm.

Next step for the PI: proofread the PDF (`article/article.pdf`, built locally) and submit to ReScience C.
