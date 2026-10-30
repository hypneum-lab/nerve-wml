"""NH-01 pilot: dreaming Hopfield kernel vs Frobenius-matched uniform shrink.

Pre-registration (binding, L1):
``docs/milestones/2026-10-25-nh01-hopfield-dream-vs-downscale.md``
(tag ``prereg-nh01-hopfield-dream-20261025``). Every constant below is the
pre-registered value; changing one requires an amendment tag.

numpy only. Deterministic per seed: every random draw comes from
``numpy.random.default_rng([ROOT_SEED, seed, N, <stream id>])``.

Usage (on the pre-registered host only)::

    python -m experiments.nh01_hopfield_dream.run --out results/nh01 --workers 4
    python -m experiments.nh01_hopfield_dream.run --analyze-only --out results/nh01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Pre-registered constants
# --------------------------------------------------------------------------- #

ROOT_SEED = 20261025
ALPHA_GRID: tuple[float, ...] = (
    0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
)
T_GRID: tuple[float, ...] = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
N_GRID: tuple[int, ...] = (64, 128, 256)
SEEDS: tuple[int, ...] = tuple(range(10))
MAX_PROBES = 32
T_PRIMARY = 10.0
N_PRIMARY = 256
FLIPS_PRIMARY = 0.10
# (temperature, cue flip fraction) conditions. T = 0 is the primary dynamics.
CONDITIONS: tuple[tuple[float, float], ...] = ((0.0, 0.10), (0.0, 0.0), (0.2, 0.10))
RETRIEVED_THRESH = {0.0: 0.9, 0.2: 0.8}
MAX_SWEEPS_T0 = 100
TIE_REL = 1e-12  # zero-field tolerance relative to N * max|J| (see run_zero_temperature)
SWEEPS_GLAUBER = 40
AVG_LAST_GLAUBER = 10
GRID_STEP_TOL = 0.05  # one grid step, C2 monotonicity tolerance
SPURIOUS_ALPHAS = (0.10, 0.14)
BOOT_SEED = 20261025
BOOT_N = 10_000
WALL_CAP_SECONDS = 2 * 3600

# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #


def hebbian_kernel(xi: np.ndarray) -> np.ndarray:
    """Hebbian couplings J = (1/N) sum_mu xi^mu xi^mu^T, diagonal included.

    ``xi`` has shape (P, N) with entries in {-1, +1}.
    """
    xi = np.asarray(xi, dtype=np.float64)
    n = xi.shape[1]
    return xi.T @ xi / n


def dreaming_kernel(j: np.ndarray, t: float) -> np.ndarray:
    """Fachechi–Agliari–Barra dreamed kernel J(t) = J (1+t) (1 + tJ)^-1.

    Computed by spectral decomposition of the symmetric Hebbian J: each
    eigenvalue lam >= 0 maps to lam (1+t) / (1 + t lam). ``t = 0`` returns J.
    """
    if t == 0.0:
        return np.array(j, dtype=np.float64, copy=True)
    w, v = np.linalg.eigh(np.asarray(j, dtype=np.float64))
    w = np.clip(w, 0.0, None)  # J is PSD; clip numerical negatives
    wt = w * (1.0 + t) / (1.0 + t * w)
    return (v * wt) @ v.T


def dreaming_kernel_pattern_space(xi: np.ndarray, t: float) -> np.ndarray:
    """Closed form in pattern space: (1/N) xi^T [(1+t)(1 + tC)^-1] xi.

    Reference implementation used by the tests (C = xi xi^T / N is P x P).
    """
    xi = np.asarray(xi, dtype=np.float64)
    p, n = xi.shape
    c = xi @ xi.T / n
    m = (1.0 + t) * np.linalg.inv(np.eye(p) + t * c)
    return xi.T @ m @ xi / n


def offdiag(j: np.ndarray) -> np.ndarray:
    """Copy of J with a zero diagonal (no self-coupling in the dynamics)."""
    out = np.array(j, dtype=np.float64, copy=True)
    np.fill_diagonal(out, 0.0)
    return out


def matched_shrink_factors(
    j0_off: np.ndarray, jt_off: np.ndarray, spectral: bool = True
) -> tuple[float, float]:
    """Return (c_frobenius, c_spectral) such that c * J0 matches Jt.

    Frobenius matching is the pre-registered control; spectral is descriptive
    and may be skipped (NaN) when the two dense eigen-solves are too costly.
    """
    c_fro = float(np.linalg.norm(jt_off) / np.linalg.norm(j0_off))
    if not spectral:
        return c_fro, math.nan
    s0 = float(np.max(np.abs(np.linalg.eigvalsh(j0_off))))
    st = float(np.max(np.abs(np.linalg.eigvalsh(jt_off))))
    return c_fro, st / s0


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #


def make_cues(
    xi: np.ndarray, probe_idx: np.ndarray, flips: float, rng: np.random.Generator
) -> np.ndarray:
    """Noisy cues: each probed pattern with exactly round(flips * N) bits flipped."""
    n = xi.shape[1]
    k = int(round(flips * n))
    cues = np.array(xi[probe_idx], dtype=np.float64, copy=True)
    for b in range(cues.shape[0]):
        if k > 0:
            pos = rng.choice(n, size=k, replace=False)
            cues[b, pos] *= -1.0
    return cues


def run_zero_temperature(
    j_off: np.ndarray,
    s0: np.ndarray,
    rng: np.random.Generator,
    max_sweeps: int = MAX_SWEEPS_T0,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Asynchronous sign dynamics, random unit order per sweep (shared by the batch).

    Ties (h = 0) keep the current spin. An exact zero field is detected up
    to a tolerance proportional to the scale of the kernel (TIE_REL * N *
    max|J|), so that the rule is invariant under a rescaling J -> cJ despite
    floating-point rounding (Hebbian fields are exact multiples of 1/N and
    tie exactly; c/N multiples do not). Stops at the first sweep with no
    flip in the whole batch. Returns (final states, sweeps run, per-probe
    converged).
    """
    s = np.array(s0, dtype=np.float64, copy=True)
    jt = np.ascontiguousarray(j_off.T)  # jt[i] = column i of J
    n = s.shape[1]
    tol = TIE_REL * n * float(np.abs(j_off).max())
    flipped_in_sweep = np.zeros(s.shape[0], dtype=bool)
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        order = rng.permutation(n)
        flipped_in_sweep[:] = False
        for i in order:
            h = s @ jt[i]
            col = s[:, i]
            new = np.where(h > tol, 1.0, np.where(h < -tol, -1.0, col))
            changed = new != col
            if changed.any():
                flipped_in_sweep |= changed
                s[:, i] = new
        if not flipped_in_sweep.any():
            return s, sweeps, np.ones(s.shape[0], dtype=bool)
    return s, sweeps, ~flipped_in_sweep


def run_glauber(
    j_off: np.ndarray,
    s0: np.ndarray,
    beta: float,
    xi: np.ndarray,
    rng: np.random.Generator,
    sweeps: int = SWEEPS_GLAUBER,
    avg_last: int = AVG_LAST_GLAUBER,
) -> tuple[np.ndarray, np.ndarray]:
    """Glauber dynamics at inverse temperature beta, fixed number of sweeps.

    Returns (final states, time-averaged overlaps with all patterns over the
    last ``avg_last`` sweeps, shape (B, P)).
    """
    s = np.array(s0, dtype=np.float64, copy=True)
    jt = np.ascontiguousarray(j_off.T)
    n = s.shape[1]
    m_acc = np.zeros((s.shape[0], xi.shape[0]), dtype=np.float64)
    n_acc = 0
    for sweep in range(1, sweeps + 1):
        order = rng.permutation(n)
        u = rng.random((n, s.shape[0]))
        for k, i in enumerate(order):
            h = s @ jt[i]
            p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
            s[:, i] = np.where(u[k] < p_up, 1.0, -1.0)
        if sweep > sweeps - avg_last:
            m_acc += s @ xi.T / n
            n_acc += 1
    return s, m_acc / n_acc


def energy(j_off: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E(s) = -1/2 s^T J s / N per state (diagonal-free kernel)."""
    return -0.5 * np.einsum("bi,ij,bj->b", s, j_off, s) / s.shape[1]


# --------------------------------------------------------------------------- #
# Cell evaluation
# --------------------------------------------------------------------------- #


def stream(seed: int, n: int, *ids: int) -> np.random.Generator:
    return np.random.default_rng([ROOT_SEED, seed, n, *ids])


def classify(
    m_cued: np.ndarray, m_all: np.ndarray, cued_idx: np.ndarray, thresh: float
) -> dict[str, float]:
    """Fractions retrieved / other_stored / spurious from overlaps (B,) and (B, P)."""
    b = m_cued.shape[0]
    retrieved = m_cued >= thresh
    m_other = np.abs(m_all).copy()
    m_other[np.arange(b), cued_idx] = 0.0
    other = (~retrieved) & (m_other.max(axis=1) >= thresh)
    spurious = ~(retrieved | other)
    return {
        "retrieved_frac": float(retrieved.mean()),
        "other_stored_frac": float(other.mean()),
        "spurious_frac": float(spurious.mean()),
    }


def evaluate_task(args: tuple) -> list[dict]:
    """All cells (alpha x arms x conditions) for one (N, seed). Paired design.

    ``args`` is ``(n, seed)`` for the pilot grid, or ``(n, seed, cfg)`` where
    ``cfg`` overrides ``alpha_grid``, ``t_grid``, ``conditions``,
    ``max_probes`` and ``spectral`` (confirmatory run). Without ``cfg`` the
    module constants apply and the pilot output is unchanged.
    """
    n, seed = args[0], args[1]
    cfg = args[2] if len(args) > 2 else {}
    alpha_grid = tuple(cfg.get("alpha_grid", ALPHA_GRID))
    t_grid = tuple(cfg.get("t_grid", T_GRID))
    conditions = tuple(cfg.get("conditions", CONDITIONS))
    max_probes = int(cfg.get("max_probes", MAX_PROBES))
    spectral = bool(cfg.get("spectral", True))
    t_start = time.time()
    records: list[dict] = []
    p_max = max(1, int(round(alpha_grid[-1] * n)))
    xi_all = np.where(stream(seed, n, 1).random((p_max, n)) < 0.5, -1.0, 1.0)

    for ai, alpha in enumerate(alpha_grid):
        p = max(1, int(round(alpha * n)))
        xi = xi_all[:p]
        j_full = hebbian_kernel(xi)
        j0_off = offdiag(j_full)
        # eigen-decomposition once per (N, seed, alpha); reused for all t
        w, v = np.linalg.eigh(j_full)
        w = np.clip(w, 0.0, None)

        arms: list[tuple[str, float, np.ndarray, float, float]] = [
            ("hebb", 0.0, j0_off, 1.0, 1.0)
        ]
        for t in t_grid[1:]:
            wt = w * (1.0 + t) / (1.0 + t * w)
            jt_off = offdiag((v * wt) @ v.T)
            c_fro, c_spec = matched_shrink_factors(j0_off, jt_off, spectral=spectral)
            arms.append(("dream", t, jt_off, c_fro, c_spec))
            arms.append(("shrink", t, c_fro * j0_off, c_fro, c_spec))

        probe_idx = np.arange(min(p, max_probes))
        for ci, (temp, flips) in enumerate(conditions):
            cues = make_cues(xi, probe_idx, flips, stream(seed, n, 2, ai, ci))
            m0 = float(np.mean(np.einsum("bi,bi->b", cues, xi[probe_idx]) / n))
            for arm, t, j_arm, c_fro, c_spec in arms:
                rng_dyn = stream(seed, n, 3, ai, ci)  # identical across arms
                rec = {
                    "N": n, "seed": seed, "alpha": alpha, "P": p, "arm": arm, "t": t,
                    "c_fro": c_fro, "c_spec": c_spec, "temp": temp, "flips": flips,
                    "n_probes": int(probe_idx.size), "cue_overlap": m0,
                }
                if temp == 0.0:
                    s, sweeps, conv = run_zero_temperature(j_arm, cues, rng_dyn)
                    m_all = s @ xi.T / n
                    m_cued = m_all[np.arange(probe_idx.size), probe_idx]
                    # non-converged probes count as not retrieved
                    m_cued_eff = np.where(conv, m_cued, -np.inf)
                    rec.update(classify(m_cued_eff, m_all, probe_idx, RETRIEVED_THRESH[0.0]))
                    rec["nonconverged_frac"] = float((~conv).mean())
                    rec["sweeps"] = int(sweeps)
                else:
                    s, m_avg = run_glauber(j_arm, cues, 1.0 / temp, xi, rng_dyn)
                    m_cued = m_avg[np.arange(probe_idx.size), probe_idx]
                    rec.update(classify(m_cued, m_avg, probe_idx, RETRIEVED_THRESH[temp]))
                    rec["nonconverged_frac"] = 0.0
                    rec["sweeps"] = SWEEPS_GLAUBER
                rec["mean_overlap"] = float(np.mean(m_cued))
                rec["energy_own"] = float(np.mean(energy(j_arm, s)))
                rec["energy_hebb"] = float(np.mean(energy(j0_off, s)))
                records.append(rec)
    for r in records:
        r["task_seconds"] = round(time.time() - t_start, 3)
    return records


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def capacity_from_curve(alphas: np.ndarray, r: np.ndarray) -> tuple[float, str]:
    """Pre-registered estimator: 0.5 crossing after the last alpha with r >= 0.5."""
    above = np.where(r >= 0.5)[0]
    if above.size == 0:
        return 0.0, "below_grid"
    k = int(above[-1])
    if k == len(alphas) - 1:
        return float(alphas[-1]), "censored"
    r0, r1 = r[k], r[k + 1]
    a0, a1 = alphas[k], alphas[k + 1]
    return float(a0 + (r0 - 0.5) / (r0 - r1) * (a1 - a0)), "ok"


def arm_key(rec: dict) -> str:
    return "hebb" if rec["arm"] == "hebb" else f"{rec['arm']}:{rec['t']:g}"


def bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = BOOT_N, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _collect(
    records: list[dict],
    spurious_alphas: tuple[float, ...] = SPURIOUS_ALPHAS,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
) -> tuple[dict, dict]:
    """Capacities per (condition, arm, seed) and spurious fractions at ``spurious_alphas``."""
    alphas = np.array(alpha_grid)
    curves: dict[tuple, dict[str, dict[int, np.ndarray]]] = {}
    spur: dict[tuple, dict[str, dict[int, dict[float, float]]]] = {}
    for rec in records:
        cond = (rec["temp"], rec["flips"], rec["N"])
        k = arm_key(rec)
        by_seed = curves.setdefault(cond, {}).setdefault(k, {})
        curve = by_seed.setdefault(rec["seed"], np.full(len(alphas), np.nan))
        curve[alpha_grid.index(rec["alpha"])] = rec["retrieved_frac"]
        if rec["alpha"] in spurious_alphas:
            sp_seed = spur.setdefault(cond, {}).setdefault(k, {}).setdefault(rec["seed"], {})
            sp_seed[rec["alpha"]] = rec["spurious_frac"]
    caps: dict[tuple, dict[str, dict[int, tuple[float, str]]]] = {}
    for cond, by_arm in curves.items():
        caps[cond] = {}
        for k, by_seed in by_arm.items():
            caps[cond][k] = {s: capacity_from_curve(alphas, r) for s, r in by_seed.items()}
    return caps, spur


def _condition_entry(
    cond: tuple,
    by_arm: dict,
    sp: dict,
    t_grid: tuple[float, ...] = T_GRID,
    t_primary: float = T_PRIMARY,
    spurious_alphas: tuple[float, ...] = SPURIOUS_ALPHAS,
) -> dict:
    temp, flips, n = cond
    seeds = sorted(by_arm["hebb"].keys())
    hebb = np.array([by_arm["hebb"][s][0] for s in seeds])
    entry: dict = {
        "temp": temp, "flips": flips, "N": n, "seeds": seeds,
        "hebb": {
            "alpha_c_per_seed": hebb.tolist(),
            "mean_ci": bootstrap_mean_ci(hebb),
            "flags": [by_arm["hebb"][s][1] for s in seeds],
        },
        "dream": {}, "shrink": {},
    }
    for t in t_grid[1:]:
        for arm in ("dream", "shrink"):
            k = f"{arm}:{t:g}"
            vals = np.array([by_arm[k][s][0] for s in seeds])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(hebb > 0, vals / hebb, np.nan)
            entry[arm][f"{t:g}"] = {
                "alpha_c_per_seed": vals.tolist(),
                "flags": [by_arm[k][s][1] for s in seeds],
                "mean_ci": bootstrap_mean_ci(vals),
                "median": float(np.median(vals)),
                "ratio_per_seed": ratio.tolist(),
                "ratio_mean_ci": bootstrap_mean_ci(ratio),
                "diff_mean_ci": bootstrap_mean_ci(vals - hebb),
                "ratio_of_means": (
                    float(vals.mean() / hebb.mean()) if hebb.mean() > 0 else math.nan
                ),
            }
    entry["spurious"] = {}
    k_dr = f"dream:{t_primary:g}"
    k_sh = f"shrink:{t_primary:g}"
    for a in spurious_alphas:
        h = np.array([sp["hebb"][s][a] for s in seeds]) if "hebb" in sp else np.array([])
        d = np.array([sp[k_dr][s][a] for s in seeds]) if k_dr in sp else np.array([])
        s_ = np.array([sp[k_sh][s][a] for s in seeds]) if k_sh in sp else np.array([])
        rel = (h.mean() - d.mean()) / h.mean() if h.size and h.mean() > 0 else math.nan
        entry["spurious"][f"{a:g}"] = {
            "hebb_mean_ci": bootstrap_mean_ci(h),
            "dream_t10_mean_ci": bootstrap_mean_ci(d),
            "shrink_c10_mean_ci": bootstrap_mean_ci(s_),
            "relative_reduction_dream": float(rel),
        }
    return entry


def _verdict(e: dict) -> dict:
    """Binding decision rule C1 / C2 / C3 and the P4 label on the primary cell."""
    tk = f"{T_PRIMARY:g}"
    r_mean, r_lo, r_hi = e["dream"][tk]["ratio_mean_ci"]
    d_mean, d_lo, d_hi = e["dream"][tk]["diff_mean_ci"]
    if r_lo >= 3.0:
        c1 = "confirmed"
    elif d_hi < 0.1:
        c1 = "refuted"
    else:
        c1 = "inconclusive"
    medians = [float(np.median(e["hebb"]["alpha_c_per_seed"]))]
    medians += [e["dream"][f"{t:g}"]["median"] for t in T_GRID[1:]]
    drops = [medians[i] - medians[i + 1] for i in range(len(medians) - 1)]
    c2 = "holds" if max(drops) <= GRID_STEP_TOL else "violated"
    s_mean, s_lo, s_hi = e["shrink"][tk]["ratio_mean_ci"]
    s_vals = np.array(e["shrink"][tk]["ratio_per_seed"], dtype=np.float64)
    s_exact = bool(np.all(np.isclose(s_vals[np.isfinite(s_vals)], 1.0)))
    if not s_exact:
        c3 = "implementation_failure"
    elif s_hi < 1.1:
        c3 = "holds"
    elif s_lo >= 2.0:
        c3 = "refutes"
    else:
        c3 = "inconclusive"
    if c1 == "confirmed" and c2 == "holds" and c3 == "holds":
        overall = "confirme"
    elif c1 == "refuted" or c2 == "violated" or c3 == "refutes":
        overall = "refute"
    else:
        overall = "non_concluant"
    p4 = {a: e["spurious"][f"{a:g}"]["relative_reduction_dream"] for a in SPURIOUS_ALPHAS}
    t_labels = [f"{t:g}" for t in T_GRID]
    return {
        "t_primary": T_PRIMARY,
        "C1_gain": {
            "label": c1,
            "ratio_mean_ci": [r_mean, r_lo, r_hi],
            "diff_mean_ci": [d_mean, d_lo, d_hi],
        },
        "C2_monotone": {
            "label": c2,
            "median_alpha_c_by_t": dict(zip(t_labels, medians, strict=True)),
            "max_drop": max(drops),
        },
        "C3_shrink": {
            "label": c3,
            "ratio_mean_ci": [s_mean, s_lo, s_hi],
            "exact_one_at_T0": s_exact,
        },
        "P4_spurious": {
            "label": "holds" if all(v >= 0.5 for v in p4.values()) else "fails",
            "relative_reduction_by_alpha": {f"{a:g}": v for a, v in p4.items()},
        },
        "overall": overall,
    }


def analyze(records: list[dict]) -> dict:
    caps, spur = _collect(records)
    out: dict = {"conditions": {}}
    for cond, by_arm in caps.items():
        temp, flips, n = cond
        out["conditions"][f"T{temp:g}_f{flips:g}_N{n}"] = _condition_entry(
            cond, by_arm, spur.get(cond, {})
        )
    key = f"T0_f{FLIPS_PRIMARY:g}_N{N_PRIMARY}"
    out["verdict"] = {"cell": key, **_verdict(out["conditions"][key])}
    return out


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def _mean_curve(records: list[dict], arm: str, t: float, field: str) -> np.ndarray:
    curve = np.zeros(len(ALPHA_GRID))
    for r in records:
        if (
            r["N"] == N_PRIMARY and r["temp"] == 0.0 and r["flips"] == FLIPS_PRIMARY
            and r["arm"] == arm and r["t"] == t
        ):
            curve[ALPHA_GRID.index(r["alpha"])] += r[field] / len(SEEDS)
    return curve


def make_figure(records: list[dict], summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = np.array(ALPHA_GRID)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ts = np.array(T_GRID[1:])
    x = np.log10(ts)
    conds = summary["conditions"]

    # (a) alpha_c vs t, T = 0, f = 0.10, all N
    ax = axes[0, 0]
    for n, color in zip(N_GRID, ("#8da0cb", "#66c2a5", "#fc8d62"), strict=True):
        e = conds[f"T0_f{FLIPS_PRIMARY:g}_N{n}"]
        d = np.array([e["dream"][f"{t:g}"]["mean_ci"] for t in ts])
        s = np.array([e["shrink"][f"{t:g}"]["mean_ci"] for t in ts])
        h = e["hebb"]["mean_ci"]
        ax.errorbar(
            x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]],
            color=color, marker="o", label=f"dream N={n}",
        )
        ax.plot(x, s[:, 0], color=color, marker="s", ls="--", label=f"shrink c(t) N={n}")
        ax.axhline(h[0], color=color, ls=":", lw=1)
    ax.set_xlabel("log10 dreaming time t")
    ax.set_ylabel("alpha_c (mean, 95% bootstrap CI over seeds)")
    ax.set_title("(a) Capacity vs t, T=0, cue 10% flips (dotted: Hebb)")
    ax.legend(fontsize=7, ncol=2)

    sel = {
        ("hebb", 0.0): ("Hebb", "k"),
        ("dream", T_PRIMARY): (f"dream t={T_PRIMARY:g}", "#1b9e77"),
        ("shrink", T_PRIMARY): (f"shrink c({T_PRIMARY:g})", "#d95f02"),
    }
    # (b) retrieval curves r(alpha) at N_PRIMARY
    ax = axes[0, 1]
    for (arm, t), (label, color) in sel.items():
        curve = _mean_curve(records, arm, t, "retrieved_frac")
        ax.plot(alphas, curve, marker="o", ms=3, color=color, label=label,
                ls="--" if arm == "shrink" else "-")
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set_xlabel("load alpha = P/N")
    ax.set_ylabel("retrieved fraction (mean over seeds)")
    ax.set_title(f"(b) Retrieval curves, N={N_PRIMARY}, T=0")
    ax.legend(fontsize=8)

    # (c) spurious fraction vs alpha at N_PRIMARY
    ax = axes[1, 0]
    for (arm, t), (label, color) in sel.items():
        curve = _mean_curve(records, arm, t, "spurious_frac")
        ax.plot(alphas, curve, marker="o", ms=3, color=color, label=label,
                ls="--" if arm == "shrink" else "-")
    ax.set_xlabel("load alpha = P/N")
    ax.set_ylabel("spurious final states (fraction)")
    ax.set_title(f"(c) Spurious attractors, N={N_PRIMARY}, T=0")
    ax.legend(fontsize=8)

    # (d) ratios vs t at T = 0 and T = 0.2 (descriptive) for N_PRIMARY
    ax = axes[1, 1]
    for temp, ls in ((0.0, "-"), (0.2, "--")):
        e = conds[f"T{temp:g}_f{FLIPS_PRIMARY:g}_N{N_PRIMARY}"]
        d = np.array([e["dream"][f"{t:g}"]["ratio_mean_ci"] for t in ts])
        s = np.array([e["shrink"][f"{t:g}"]["ratio_mean_ci"] for t in ts])
        ax.errorbar(
            x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]],
            color="#1b9e77", ls=ls, marker="o", label=f"dream / Hebb, T={temp:g}",
        )
        ax.errorbar(
            x, s[:, 0], yerr=[s[:, 0] - s[:, 1], s[:, 2] - s[:, 0]],
            color="#d95f02", ls=ls, marker="s", label=f"shrink / Hebb, T={temp:g}",
        )
    ax.axhline(3.0, color="grey", lw=0.8, ls=":")
    ax.axhline(1.1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("log10 dreaming time t")
    ax.set_ylabel("alpha_c ratio to Hebb (mean, 95% CI)")
    ax.set_title(f"(d) Ratios, N={N_PRIMARY}: T=0 (solid) vs T=0.2 (dashed)")
    ax.legend(fontsize=7)

    v = summary["verdict"]
    fig.suptitle(
        f"NH-01 pilot — verdict {v['overall']} (C1 {v['C1_gain']['label']}, "
        f"C2 {v['C2_monotone']['label']}, C3 {v['C3_shrink']['label']})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/nh01"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument(
        "--smoke", action="store_true", help="tiny grid for implementation checks (not a run)"
    )
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"
    meta_path = args.out / "run_meta.json"

    if not args.analyze_only:
        tasks = [(n, s) for n in N_GRID for s in SEEDS]
        if args.smoke:
            tasks = [(64, 0)]
        t0 = time.time()
        records: list[dict] = []
        with Pool(processes=max(1, args.workers)) as pool:
            for recs in pool.imap_unordered(evaluate_task, tasks):
                records.extend(recs)
                done = {(r["N"], r["seed"]) for r in records}
                print(
                    f"[{time.time() - t0:7.1f}s] {len(done)}/{len(tasks)} tasks, "
                    f"{len(records)} cells",
                    flush=True,
                )
                if time.time() - t0 > WALL_CAP_SECONDS:
                    print("wall-time cap reached: stopping (pre-registered rule)", flush=True)
                    pool.terminate()
                    break
        wall = time.time() - t0
        with cells_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        meta = {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpus": os.cpu_count(),
            "workers": args.workers,
            "wall_seconds": round(wall, 1),
            "git_commit": git_commit(),
            "n_cells": len(records),
            "tasks_done": sorted({(r["N"], r["seed"]) for r in records}),
            "smoke": args.smoke,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        if args.smoke:
            return 0

    records = [json.loads(line) for line in cells_path.read_text().splitlines() if line.strip()]
    summary = analyze(records)
    summary["run_meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figure(records, summary, args.out / "figure.png")
    print(json.dumps(summary["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
