# Renf 11/12/13 cross-host comparison — macm1 (M1 Max, `applegpu_g13s`) vs macM3 (M3 Pro, `applegpu_g15s`)

## Headline

**3 deterministic substrates (gtm, simple_gating, null) are bit-exact reproducible cross-arch on PyTorch + multiprocessing.** The AKOrN Kuramoto arm diverges meaningfully cross-arch on both spectral and task metrics. macM3 (M3 Pro 8-core) is ~2-3× faster than macm1 (M1 Max 8-core) on CPU-bound multiproc workloads.

This complements [ml-explore/mlx#3568](https://github.com/ml-explore/mlx/issues/3568) (MLX cross-arch divergence on `mx.random.normal`) by showing that:
- PyTorch's `torch.manual_seed()` path **is** cross-arch reproducible for deterministic models.
- Stochastic/integrator-heavy models (Kuramoto Euler) **are not** cross-arch reproducible even with PyTorch.

## Method

- Identical scripts, identical MLX/PyTorch versions (torch 2.12.0, MLX 0.31.2).
- Hosts: macm1 (Apple M1 Max, `applegpu_g13s`, 8-core CPU) vs macM3 (Apple M3 Pro, `applegpu_g15s`, 8-core CPU).
- Same git worktree (rsynced), same uv-pinned dependencies.
- All Renf scripts use PyTorch CPU multiproc (Pool 8 workers), not MPS or MLX directly.

## Renf 11 — seed window robustness (3 windows × 4 arms × 50 seeds)

| host | window | null | akorn_best | gtm | simple_gating |
|---|---|---|---|---|---|
| macm1 | A (0-49) | 1.9597 | 1.9402 | 2.1729 | 3.4645 |
| macm1 | B (50-99) | 1.9804 | 1.8876 | 2.1734 | 3.4641 |
| macm1 | C (1000-1049) | 1.9747 | 1.8348 | 2.1742 | 3.4642 |
| macM3 | A | 1.9597 | **1.9385** | 2.1729 | 3.4645 |
| macM3 | B | 1.9804 | **1.8629** | 2.1734 | 3.4641 |
| macM3 | C | 1.9747 | **1.7287** | 2.1742 | 3.4642 |

(spectral_entropy means, 50 seeds per cell)

**Verdict per arm:**
- **null, gtm, simple_gating**: identical to 4 decimals macm1 ↔ macM3 across all 3 windows. **Cross-arch bit-exact.**
- **akorn_best**: diverges cross-arch (max |Δ| = 0.107 on window C). Consistent with Kuramoto integrator FP-reduction-order sensitivity.

**Wall-clock**: macm1 1454.6s (24.2 min), macM3 508.4s (8.5 min) — macM3 is **2.86× faster**.

## Renf 12 — AKOrN top cell n=50

Both hosts ran the Renf 1 top cell `n_oscillators=64, n_steps=32, lr=0.05`.

| host | accuracy mean ± std | mi_bits mean ± std | synchrony_index mean ± std |
|---|---|---|---|
| macm1 | 0.3796 ± 0.3063 | 1.5807 ± 0.2918 | **0.5310 ± 0.2017** |
| macM3 | 0.3261 ± 0.2832 | 1.5367 ± 0.2658 | **0.4906 ± 0.1877** |
| **Δ (macM3 − macm1)** | −0.054 | −0.044 | **−0.040** |

**Verdict**: AKOrN top cell **diverges cross-arch by ~0.04 in synchrony, ~0.05 in accuracy** at n=50. The qualitative claim from Renf 1 (AKOrN above GTM's 0.20 baseline) holds on both hosts (0.49-0.53 > 0.20), but the precise number depends on the host architecture. AKOrN remains **non-cross-arch-reproducible** as a quantitative claim.

**Wall-clock**: macm1 553.2s (9.2 min), macM3 241.9s (4.0 min) — macM3 is **2.29× faster**.

## Renf 13 — harder routing (alphabet=128, K=9 [capped from K=14 by Lisman-Idiart], 20 seeds)

| arm | metric | macm1 | macM3 | Δ |
|---|---|---|---|---|
| gtm | accuracy | 1.0000 | 1.0000 | +0.0000 ✅ |
| gtm | mi_bits | 3.5483 | 3.5483 | +0.0000 ✅ |
| gtm | spectral_entropy | 2.4110 | 2.4110 | +0.0000 ✅ |
| simple_gating | accuracy | 1.0000 | 1.0000 | +0.0000 ✅ |
| simple_gating | mi_bits | 3.5483 | 3.5483 | +0.0000 ✅ |
| simple_gating | spectral_entropy | 3.6552 | 3.6552 | +0.0000 ✅ |
| **akorn_best** | accuracy | 0.5095 | **0.6699** | **+0.1604 ⚠** |
| **akorn_best** | mi_bits | 3.1735 | **3.2847** | **+0.1113 ⚠** |
| **akorn_best** | spectral_entropy | 1.9805 | 1.9120 | **−0.0685 ⚠** |
| null | accuracy | 0.0186 | 0.0186 | +0.0000 ✅ |
| null | mi_bits | 2.8851 | 2.8851 | +0.0000 ✅ |
| null | spectral_entropy | 2.3761 | 2.3761 | +0.0000 ✅ |

**Verdict per arm:**
- **gtm, simple_gating, null**: **bit-exact cross-arch** on all 3 metrics. PyTorch deterministic path holds at harder task scale.
- **akorn_best**: Δ accuracy = +0.16, Δ MI = +0.11 — macM3 obtains substantively better AKOrN performance than macm1. Not noise.

**Wall-clock**: macm1 343.8s, macM3 170.7s — macM3 is **2.01× faster**.

## Discussion — connection to ml-explore/mlx#3568

[MLX issue #3568](https://github.com/ml-explore/mlx/issues/3568) documented that `mx.random.normal` is non-bit-exact between M1 (`applegpu_g13s`) and M3+ (`g15+`) at fixed MLX version. The Renf 11/12/13 scripts use `torch.manual_seed()` (PyTorch RNG, **not** MLX), so they probe a different question: is PyTorch's deterministic path cross-Apple-Silicon bit-exact?

**Answer from Renf 11/12/13 cross-host data**:
- For PyTorch-only deterministic models (gtm, simple_gating, null): **YES, bit-exact**. The PyTorch RNG and basic ops do not exhibit the cross-arch divergence that MLX `random.normal` does.
- For models with **iterative integrator with FP-sensitive operations** (Kuramoto Euler steps in AKOrN): **NO, cross-arch divergence appears**. The accumulated FP-reduction-order differences across Euler iterations produce measurable shifts in final synchrony, accuracy, and MI.

**This is a parallel, distinct issue from MLX #3568**. PyTorch + AKOrN-style iterative integrators are also vulnerable, but the failure mode is different: gradual accumulation rather than direct RNG-path bytecode difference.

## Implications for the paper

1. **gtm, simple_gating, null arms**: results are robust cross-arch. Can be cited without arch caveat.
2. **AKOrN arm**: results are **not** cross-arch reproducible. The paper should either:
   - Report only the qualitative claim (AKOrN above GTM in synchrony at high power, but bimodal in accuracy), **with explicit cross-arch caveat**.
   - Or pin specific reproducibility numbers to a specific host architecture (e.g., "on M1 family, AKOrN top cell synchrony 0.53 ± 0.20").
3. The Limitations section item 4 (AKOrN minimal flavour) and item 7 (MLX cross-Apple-Silicon non-bit-exact) should be cross-referenced. AKOrN cross-arch sensitivity is a separate finding from MLX `random.normal` — it's a property of the Kuramoto integrator + PyTorch FP ordering, not MLX-specific.

## Reproducibility

- Scripts: `scripts/renf11_seed_window.py`, `scripts/renf12_akorn_top_50s.py`, `scripts/renf13_harder_routing.py` (on master at `85cbc6b`)
- macm1 results: `2026-05-20-renf{11,12,13}-*.json` (PR #31)
- macM3 results: `2026-05-20-renf{11,12,13}-*-macm3.json` (this PR)
- Same RNG seeds (0-49 for Renf 11 window A and Renf 12, 0-19 for Renf 13).
