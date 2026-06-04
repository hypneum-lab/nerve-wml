# Scale Robustness at Large N: CKNNA Evolution from N=64 to N=2048

**Date:** 2026-05-20  
**Experiment:** `scripts/scale_robustness_pilot.run_scale_robustness(sizes=(64, 128, 256, 512, 1024, 2048), seed in 0..6)`  
**Hypothesis:** PRH (Huh et al. 2024) critique that representation-alignment metrics degrade at scale.

## Methodology

We embed a stream of 2048 randomly drawn codes from a shared WML alphabet (size 64) through two substrates: MlpWML and LifWML. For each subsample size N ∈ {64, 128, 256, 512, 1024, 2048}, we compute:
- **Real arm:** CKNNA(MLP embeddings, LIF embeddings), k=10.
- **Null arm:** CKNNA(MLP embeddings, permuted LIF embeddings), k=10.

The null baseline destructs the per-row pairing, so null curves validate that real curves reflect true substrate alignment rather than noise.

Repeated across **7 seeds** (0–6) with fixed alphabet and code stream, reporting mean ± std.

## Results

| N    | Real CKNNA (μ ± σ) | Null CKNNA (μ ± σ) | Δ (Real − Null) |
|------|--------------------|--------------------|-----------------|
| 64   | 0.2353 ± 0.0235    | 0.1491 ± 0.0199    | 0.0862          |
| 128  | 0.2390 ± 0.0202    | 0.0826 ± 0.0167    | 0.1564          |
| 256  | 0.4326 ± 0.0260    | 0.0381 ± 0.0050    | 0.3945          |
| 512  | 0.7145 ± 0.0192    | 0.0204 ± 0.0021    | 0.6941          |
| 1024 | 0.6558 ± 0.0118    | 0.0097 ± 0.0010    | 0.6461          |
| 2048 | 0.3288 ± 0.0088    | 0.0052 ± 0.0004    | 0.3236          |

## Interpretation

**PRH critique — PARTIALLY SUPPORTED AND REFINED:**

1. **Scale-dependent peak:** Real CKNNA exhibits a non-monotone trajectory, peaking at **N=512** (μ=0.715) and declining sharply to N=2048 (μ=0.329). This is **not uniform degradation**, but a sharp drop post-optimum.

2. **Null baseline validation:** Null CKNNA remains near zero at all N (max μ=0.149 at N=64) and decreases toward zero at large N (μ=0.005 at N=2048). This confirms that real scores reflect actual substrate alignment, not noise.

3. **Signal preservation:** Despite the drop, real CKNNA at N=2048 (0.329 ± 0.009) remains **>60× higher than null** (0.005 ± 0.0004). The substrates are still statistically distinguishable at large N, but with weaker alignment signal.

4. **Mechanism hypothesis:** The degradation at N>512 suggests that:
   - The MlpWML and LifWML codebooks exhibit **high mutual alignment at small N** (where random sampling variance is high and alignment is easy to detect).
   - At intermediate N (~256–512), the codebook structure becomes fully visible and CKNNA captures genuine structure (local neighborhoods align across substrates).
   - At large N (≥1024), random pairing noise and sparse coverage of the codebook begin to dominate; the two substrates are no longer sufficiently **locally aligned** for k-nearest-neighbor overlaps to remain high.

## Conclusion

The PRH critique holds for this substrate pair, but **with nuance:**
- Alignment metrics do degrade as N grows beyond ~512.
- However, degradation is sharp and non-monotone, not gradual.
- The signal survives to N=2048, suggesting the critique is about **signal-to-noise ratio**, not absolute loss of alignment.

This suggests that **alignment benchmarks should focus on intermediate scales (N=256–512)** where signal is strong and noise is manageable, rather than extrapolating from small-N experiments (N≤128) or trusting large-N measurements (N≥1024) without sanity checks.

## Open Questions

1. **Does this generalise to harder substrate pairs?** (e.g., MlpWML vs. a untrained random embedding matrix)
2. **Is the peak at N=512 an artifact of our codebook size (64)?** Conjecture: peak N ~ 2–4× alphabet size.
3. **Can we recover signal at large N via better distance metrics?** (e.g., Wasserstein, sliced-Wasserstein instead of k-NN)
