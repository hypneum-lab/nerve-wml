# Scripts

Build, dev, and automation scripts. 61 Python + 3 Bash scripts post-sprint (Renf 1–13 pipeline + backend bench + paper figures + utilities).

## Conventions

- Executable: `chmod +x script.sh`
- Header comment with usage, e.g. `# Usage: ./script.sh --flag value`
- Bash: start with `set -euo pipefail`
- Exit codes: 0 success, non-zero with a clear message

## Adding new scripts

- Live in this directory
- Add an alias in the project's standard task runner if used often
- Keep them single-purpose; chain instead of mixing concerns
- **Before adding numerical claims** to docs/superpowers/research/ or papers/: ensure the corresponding JSON output is in `docs/superpowers/research/` so `scripts/factcheck_audit.py --ci` covers it.

## Anti-patterns

- Don't hardcode absolute paths — use `$HOME`, env vars, or compute relative
- Don't skip error handling — `set -e` + explicit checks at boundaries
- Don't assume tools exist — `command -v <tool>` first
- Don't write non-idempotent operations without `--dry-run` support

## Post-sprint script catalog (61 Python + 3 Bash)

### Core protocol + scientific eval (2 scripts)
- `factcheck_audit.py` — audit numerical claims against JSON cells in `docs/superpowers/research/`, enforce CI via `.github/workflows/factcheck.yml`.
- `multi_seed.py` — n=50 seed ensemble wrapper (Renf 4+ baseline).

### Renf research pipeline 1–13 (14 scripts)
- `renf1_akorn_sweep.py` — AKOrN top-cell synchrony sweep (0.4542 ± 0.1624).
- `renf2_akorn_scale.py` — AKOrN scaling study.
- `renf3_gtm_baseline.py` — GammaThetaMultiplexer baseline (deprecated, replaced by Renf 7).
- `renf4_extended_eval.py` — transducer + GTM + scale benchmarks (50 seeds, 50s/67.8s/359.6s/2.4s wall-clocks).
- `renf5_gpu_backend_bench.py` — MLX vs MPS: 2.07× speedup at N=16384.
- `renf6_macm1_scientific_eval.py` — CKNNA 0.92→0.87 (N=256→16384, σ=0.05); 1750 cells (50 seeds × 5 σ × 7 N).
- `renf7_synchrony_replacement.py` — spectral_entropy replaces synchrony; cross-host macM1 CPU 254.7s/MPS 253.3s.
- `renf8_transducer_anchors64.py` — transducer 50-seed tie p=0.627 (M5+macM1).
- `renf10_batch_sensitivity.py` — spectral_entropy B-sweep: gtm > null > simple_gating > akorn_best (boolean assertions).
- `renf11_seed_window.py` — seed-window stability Mann-Whitney p > 0.01 (A/B/C pair-wise).
- `renf12_akorn_top_50s.py` — Renf 1 mean inside Renf 12 CI95 (50-seed corroboration).
- `renf13_harder_routing.py` — harder-routing arm separation (accuracy/MI/spectral_entropy, informational).
- `renf_cross_host_comparison.py` — macm1 + macM3 + M5 validation; bit-exact frontier at PyTorch determinism boundary.
- `renf_macm3_audit.py` — post-reboot artefact audit (Kuramoto-Euler divergence vs PyTorch).

### Backend benchmarks + bio substrate (4 scripts)
- `cknna_mlx_bench.py` — CKNNA MLX inference w/ quantization variants.
- `synchrony_replacement_eval.py` — spectral_entropy wall-clock vs synchrony (benchmark harness).
- `mlx_blindage.py` — 4-host MLX sha256 cross-host validation (MLX#3568 randomness divergence).
- `bio_substrate_access.py` — bio-substrate model availability + access latency probes.

### Paper figures (2 groups, 4 scripts)
**Group 1: Cycle traces + Renf 1 replication**
- `make_cycle_trace.py` — reproduce paper Figure 2 (Kuramoto cycle + Renf 1 overlay).
- `make_renf1_figure.py` — synchrony violin plot (Renf 1, top-cell).

**Group 2: Spectral entropy + GTM comparison**
- `make_spectral_entropy_figure.py` — spectral_entropy vs synchrony comparison.
- `make_gtm_comparison_figure.py` — GTM vs null baseline (Renf 7).

### Ablations + diagnostics (5 scripts)
- `akorn_cascade_ablate.py` — AKOrN cascade depth + adapter width ablation.
- `gating_mechanism_compare.py` — simple_gating vs GTM vs akorn_best (diagnostic).
- `vq_codebook_analysis.py` — VQ codebook utilization + collapse risk.
- `routing_entropy_diagnostic.py` — router entropy per domain (detectability diagnostic).
- `invariant_violation_detector.py` — scan Renf artefacts for N-1..N-5, W-1..W-4 violations.

### Data preprocessing + utilities (8 groups, 14+ scripts)
**Group 1: Multi-seed utilities**
- `collect_seed_ensemble.py` — gather n=50 seeds, compute mean/std/ci95.
- `seed_window_validator.py` — Mann-Whitney stability checker.

**Group 2: Dataset tools**
- `prepare_neurocluster_data.py` — NeuroCluster ETL (train/val/test splits).
- `augment_spikes.py` — spike jitter + time warping.
- `normalize_features.py` — z-score normalization per recording.

**Group 3: JSON artefact tools**
- `json_to_csv_export.py` — convert Renf session JSONs to CSV (for paper tables).
- `artefact_validator.py` — schema check + hash consistency.
- `artefact_merge.py` — combine multiple seed runs into ensemble artefact.

**Group 4: Checkpoint utilities**
- `checkpoint_loader.py` — load Renf intermediate checkpoints (for ablation restarts).
- `state_dict_audit.py` — verify parameter counts match spec.

**Group 5: CI helpers (3 Bash)**
- `run_fast_tests.sh` — invoke L1–L2 pytest suite (gate before push).
- `run_slow_tests.sh` — L3–L4 suite (gate before merge).
- `validate_docs.sh` — spellcheck + markdown link validation.
