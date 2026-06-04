# Fact-check audit report — 2026-05-20

## Purpose

Every numerical claim in the nerve-wml paper, in research notes, and in
PR bodies must be **traceable to a JSON cell** in
`docs/superpowers/research/`. This audit verifies that mechanically.

The rule, adopted after the "7 décades" miscount (the dataset only
spans 1.81 decades — six doublings — yet a draft claimed seven), is:

> Every number in a draft must point to a code cell or executed log
> line in the same session.

`scripts/factcheck_audit.py` is the machine-checkable enforcement of
that rule.

## Protocol

For each headline claim, the script:

1. Reads the source JSON in `docs/superpowers/research/`.
2. Recomputes the claimed quantity from the raw cells.
3. Compares against the value asserted in the paper / research note.
4. Reports `OK`, `DIVERGENT` (with the recomputed value), or
   `ORPHAN` (no source JSON available yet).

Numerical comparisons use an absolute tolerance defaulting to
`±0.005`. Specific claims override the tolerance and the override
is annotated in the claim label (e.g. `tol=0.01` for the CKNNA
1-percentage-point headline rounding).

## Coverage

| # | Claim | Source JSON | Tolerance |
|---|-------|-------------|-----------|
| 1 | Renf 6 — 1750 cells = 50 seeds × 5 σ × 7 N | `2026-05-20-macm1-scientific-eval.json` | exact |
| 2 | Renf 7 — spectral_entropy means and arm ordering | `2026-05-20-synchrony-replacement.json` | 0.005 |
| 3 | Renf 7 cross-host — macM1 CPU 254.7 s, MPS 253.3 s | `2026-05-20-synchrony-replacement-macm1-{cpu,mps}.json` | 0.005 |
| 4 | Renf 8 — transducer 50 seeds, tie p = 0.627 (M5 + macM1) | `2026-05-20-transducer-anchors64-50s{,-macm1}.json` | 0.005 |
| 5 | Renf 5/5b/9 — MLX 2.07× MPS M5 at N=16384 | `2026-05-20-gpu-backend-bench-mlx.json` | informational |
| 6 | Renf 4 v3 — transducer 67.8 s, gtm 359.6 s, scale 2.4 s | `2026-05-20-extended-eval-v3.json` | 0.005 |
| 7 | Renf 6 — CKNNA 0.92 → 0.87 between N=256 and 16384 (σ=0.05) | `2026-05-20-macm1-scientific-eval.json` | 0.01 (1 pp) |
| 8 | Renf 1 — AKOrN top cell synchrony 0.4542 ± 0.1624 | `2026-05-20-akorn-sweep.md` (table) | string match |
| 9 | Renf 10 — spectral_entropy B-sweep ordering, gtm > null, gtm < simple_gating, akorn_best < null at B=128 | `2026-05-20-renf10-batch-sensitivity.json` (not on master) | exact / boolean |
| 10 | Renf 11 — seed-window stability (Mann-Whitney p > 0.01 for A/B/C pairs) | `2026-05-20-renf11-seed-window.json` (not on master) | p > 0.01 |
| 11 | Renf 12 — Renf 1 mean inside Renf 12 CI95 (50-seed corroboration) | `2026-05-20-renf12-akorn-top-50s.json` (not on master) | CI containment |
| 12 | Renf 13 — harder-routing arm separation (accuracy / MI / SE) | `2026-05-20-renf13-harder-routing.json` (not on master) | informational |
| 13 | 4-host MLX blindage (sha256 cross-host) | `2026-05-20-blindage*.json` (not on master) | exact |

## Results — master @ run 2026-05-20

```
=== AUDIT SUMMARY ===
  OK:        27
  DIVERGENT: 0
  ORPHAN:    5
```

| Status | Count | Details |
|--------|-------|---------|
| OK | 27 | Claims 1, 2, 3, 4, 5 (subchecks), 6, 7, 8 |
| DIVERGENT | 0 | — |
| ORPHAN | 5 | Claims 9, 10, 11, 12, 13 (source JSONs not on master) |

### Verdict

**No divergence detected on master**. Every numerical claim that has a
source JSON on master matches the value asserted in the paper /
research notes within tolerance.

The five `ORPHAN` claims correspond to Renf 10–13 plus the 4-host MLX
blindage — these JSONs live on `feat/deep-verifications` and other
working branches, and will be picked up automatically by the audit
once they land on master (the script uses an `_maybe()` reader and
will switch from `ORPHAN` to `OK`/`DIVERGENT` without code changes).

## Limitations

Claims **not** yet covered by the audit:

- GPU bench wall-clocks on M5 MPS (the `2.07×` ratio is recomputed
  but the 3058.6 ms reference value is hard-coded — the source MPS
  JSON should be wired in once the schema is normalised).
- The d_z(real vs null) headline numbers (135 → 1311) are recomputed
  and printed but not yet asserted with a `check()` — they are
  surface-level diagnostics for now.
- Cross-host MLX sha256 divergence (blindage) — JSON schema is not
  yet stable; the audit detects presence of the file but does not
  yet check the per-host hashes.
- Paper-side cross-reference: claims that appear *only* in
  `papers/paper1/main.tex` and not in a research note are not
  enumerated. A future ORPHAN scrape (see below) will close that
  gap.

## CI integration

```bash
python scripts/factcheck_audit.py --ci
```

Exits non-zero iff any `DIVERGENT` was reported. `ORPHAN` does *not*
fail the build — orphans are an expected state for unmerged research
branches.

The workflow at `.github/workflows/factcheck.yml` runs the audit on
every push to `master`/`main` and every PR that touches
`docs/superpowers/research/**` or the audit script itself, on Python
3.12, with `numpy + scipy` (both pure-import dependencies; the audit
itself uses only stdlib).

## Future work

1. **Orphan scrape.** A `grep -E '[0-9]+\.[0-9]+'`-style pass over
   `docs/superpowers/research/*.md` (and `papers/paper1/main.tex`)
   would surface every numerical literal. Cross-referencing that set
   against the claims audited here would flag any number in a draft
   that is *not* covered by `factcheck_audit.py` — the proper
   `ORPHAN` semantics.
2. **Tolerance registry.** Move the per-claim tolerances out of the
   script into a small YAML so reviewers can argue about tolerance
   in PR review without touching Python.
3. **Paper-tex integration.** Parse `\num{...}` macros in
   `papers/paper1/main.tex` and reconcile each against the audit.
4. **Diff mode.** `factcheck_audit.py --diff PREVIOUS.log` to flag
   any claim whose recomputed value changed between two runs (e.g.
   after a re-run of a Renf script).
