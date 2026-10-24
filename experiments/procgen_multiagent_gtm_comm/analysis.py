"""Frozen N10-C estimation analysis (numpy/scipy only).

Implements, and nothing beyond, the "Estimation plan" of the 2026-10-23
amendment: Welch / one-sample t / Fieller intervals, the pilot category,
secondary contrasts, normalized improvement and the message-disruption
(S) sensitivity report. No hypothesis tests, no p-values, no bootstrap,
no verdict labels beyond the registered pilot categories.

Input format of :func:`primary_report`
--------------------------------------
``results`` is a ``dict[(arm, condition)] -> dict`` with

* ``arm`` in ``{"gtm", "mlp", "atoc", "dial", "eng", "nocomm"}``,
* ``condition`` in ``{"registered", "closed"}`` (the closed condition
  has gtm, dial, eng, nocomm),
* the inner dict holding ``"seeds"`` (exactly the six registered seeds,
  in order) and one length-6 array per seed (same seed order in every
  cell) for the metrics

  - ``"R"``: held-out team reward (success rate, range 1),
  - ``"time_to_half"``: sample efficiency endpoint, already capped at
    50M upstream,
  - ``"bandwidth_bits"``: plug-in message entropy in bits,
  - ``"S"``: message-shuffle sensitivity (drop in R),

  and the REQUIRED ``"reached_threshold"`` (bool per seed, True when the
  0.5 training-success threshold was reached within the budget). It
  disambiguates runs reaching it at the final 50M evaluation (also
  time 50M) from runs never reaching it (time 50M, flag False).

Every array must be 1-D, of exactly 6 finite values inside the metric's
range (R in [0, 1], S in [-1, 1], bandwidth in [0, 16], time in [0, 50M]);
malformed cells raise ``ValueError``.

Output of :func:`primary_report` (all intervals are
``{"estimate"|"mean", "low", "high", ...}``; unavailable intervals have
``low = high = None`` and keep the point estimate) is a dict with keys
``primary``, ``secondary``, ``normalized_improvement``, ``S``,
``threshold_counts`` and ``by_host`` (descriptive per-host summaries of the
fixed 3-seed blocks: no interval, no label). A normalized improvement whose
Fieller interpretability condition fails has ``estimate = None`` and
``interpretable = False``; ``numerator`` / ``denominator`` stay reported.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .harness.codes import HOSTS, SEED_HOST, SEEDS

DELTA = 0.10
RANGE = 1.0
CAP = 50_000_000
BASELINES = ("mlp", "atoc", "dial")
METRICS = ("R", "time_to_half", "bandwidth_bits")
N_PRIMARY = 3
N_SECONDARY = 9
N_ENV = 1


N_SEEDS = 6  # amendment 2026-10-23 (second round): seeds 0,17,42,73,101,131
ARMS_REGISTERED = ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")
ARMS_CLOSED = ("gtm", "dial", "eng", "nocomm")
CELLS = tuple((a, "registered") for a in ARMS_REGISTERED) + tuple(
    (a, "closed") for a in ARMS_CLOSED)
CROSS_PAIRS = tuple((s, SEEDS[(k + 1) % len(SEEDS)]) for k, s in enumerate(SEEDS))
BANDWIDTH_MAX = 16.0  # bits: entropy of a 16-bit message


def _var(x):
    """Sample variance; exactly 0 for exactly-constant arrays."""
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == x[0]):
        return 0.0
    return float(np.var(x, ddof=1))


def _welch_parts(x, y):
    """Return (diff, se, df) or None when the df is undefined."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return None
    v1 = _var(x) / n1
    v2 = _var(y) / n2
    tot = v1 + v2
    if not tot > 0:
        return None
    df = tot**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    return float(x.mean() - y.mean()), float(np.sqrt(tot)), float(df)


def welch_ci(x, y, level):
    """Welch CI of mean(x) - mean(y): (diff, low, high), None if undefined."""
    parts = _welch_parts(x, y)
    if parts is None:
        return None
    d, se, df = parts
    h = stats.t.ppf(1 - (1 - level) / 2, df) * se
    return d, d - h, d + h


def one_sample_ci(x, level):
    """One-sample t CI of the mean: (mean, low, high), None if s == 0."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return None
    s = np.sqrt(_var(x))
    if not s > 0:
        return None
    m = float(x.mean())
    h = stats.t.ppf(1 - (1 - level) / 2, n - 1) * s / np.sqrt(n)
    return m, m - h, m + h


def fieller_interpretable(a_arm, nocomm, eng, level=0.95):
    """Interpretability rule of the normalized improvement.

    True if the denominator interval b +- t sqrt(Var b) lies entirely above
    0, False if it does not, None if it cannot be computed (undefined df).
    """
    pa = _welch_parts(a_arm, nocomm)
    pb = _welch_parts(eng, nocomm)
    if pa is None or pb is None:
        return None
    t = stats.t.ppf(1 - (1 - level) / 2, min(pa[2], pb[2]))
    return bool(pb[0] - t * pb[1] > 0)


def fieller_ci(a_arm, nocomm, eng, level=0.95):
    """Fieller CI of (arm - NOCOMM) / (ENG - NOCOMM).

    Cov(a, b) = v_NOCOMM; t at the smaller Welch df of the two
    differences. Returns (ratio, low, high), or None unless the
    denominator interval b +- t sqrt(Var b) lies entirely above 0.
    """
    pa = _welch_parts(a_arm, nocomm)
    pb = _welch_parts(eng, nocomm)
    if pa is None or pb is None:
        return None
    a, se_a, df_a = pa
    b, se_b, df_b = pb
    t = stats.t.ppf(1 - (1 - level) / 2, min(df_a, df_b))
    var_a, var_b = se_a**2, se_b**2
    cov = _var(nocomm) / len(nocomm)
    if not b - t * se_b > 0:
        return None
    t2 = t * t
    qa = b * b - t2 * var_b  # > 0 by the rule above
    qb = a * b - t2 * cov
    qc = a * a - t2 * var_a
    disc = qb * qb - qa * qc
    if disc < 0:
        return None
    r = np.sqrt(disc)
    return float(a / b), float((qb - r) / qa), float((qb + r) / qa)


def pilot_category(intervals, delta=DELTA):
    """Pilot category from the three primary intervals, in registered order."""
    if any(iv is None for iv in intervals):
        return "pilot-interval-unavailable"
    if any(iv[2] < delta for iv in intervals):
        return "pilot-excludes-SESOI"
    if all(iv[1] >= delta for iv in intervals):
        return "pilot-positive"
    return "pilot-inconclusive"


def _contrast(x, y, level):
    """Interval dict for mean(x) - mean(y); point estimate kept if unavailable."""
    ci = welch_ci(x, y, level)
    est = float(np.mean(x) - np.mean(y))
    if ci is None:
        return {"estimate": est, "low": None, "high": None, "level": level}
    return {"estimate": ci[0], "low": ci[1], "high": ci[2], "level": level}


def _position(c):
    if c["low"] is None:
        return "unavailable"
    if c["low"] > 0:
        return "above_0"
    if c["high"] < 0:
        return "below_0"
    return "includes_0"


def _metric(results, arm, cond, metric):
    return np.asarray(results[(arm, cond)][metric], float)


def _primary(results):
    level = 1 - 0.05 / N_PRIMARY
    gtm = _metric(results, "gtm", "registered", "R")
    contrasts, ivs = {}, []
    for b in BASELINES:
        base = _metric(results, b, "registered", "R")
        c = _contrast(gtm / RANGE, base / RANGE, level)
        contrasts[b] = c
        ivs.append(None if c["low"] is None else (c["estimate"], c["low"], c["high"]))
    return {
        "level": level,
        "delta": DELTA,
        "range": RANGE,
        "contrasts": contrasts,
        "category": pilot_category(ivs, DELTA),
    }


def _secondary(results):
    level = 1 - 0.05 / N_SECONDARY
    out, counts = [], {"above_0": 0, "below_0": 0, "includes_0": 0, "unavailable": 0}
    for b in BASELINES:
        for m in METRICS:
            g = _metric(results, "gtm", "registered", m)
            o = _metric(results, b, "registered", m)
            # sample efficiency: comparator - GTM (positive favours GTM);
            # team reward and bandwidth: GTM - comparator (bandwidth signed).
            c = _contrast(o, g, level) if m == "time_to_half" else _contrast(g, o, level)
            c.update({"baseline": b, "metric": m})
            counts[_position(c)] += 1
            out.append(c)
    return {"level": level, "contrasts": out, "counts": counts}


def _ni(results):
    out = {}
    for cond in ("registered", "closed"):
        cell = {}
        if ("eng", cond) in results:
            cell["eng"] = {"estimate": 1.0, "fixed": True}
        if ("nocomm", cond) in results:
            cell["nocomm"] = {"estimate": 0.0, "fixed": True}
        have = ("eng", cond) in results and ("nocomm", cond) in results
        for (arm, c), _ in results.items():
            if c != cond or arm in ("eng", "nocomm"):
                continue
            if not have:
                cell[arm] = {"estimate": None, "low": None, "high": None}
                continue
            r = lambda a, cond=cond: _metric(results, a, cond, "R")  # noqa: E731
            num = float(r(arm).mean() - r("nocomm").mean())
            den = float(r("eng").mean() - r("nocomm").mean())
            ok = fieller_interpretable(r(arm), r("nocomm"), r("eng"), 0.95)
            ci = fieller_ci(r(arm), r("nocomm"), r("eng"), 0.95)
            if ok is False or (ok is None and not den > 0):
                # interpretability rule fails: NI is not reported (a property
                # of the data); numerator and denominator stay available
                est = None
            else:
                est = num / den if ci is None else ci[0]
            cell[arm] = {
                "estimate": est,
                "low": None if ci is None else ci[1],
                "high": None if ci is None else ci[2],
                "level": 0.95,
                "interpretable": ok,
                "numerator": num,
                "denominator": den,
            }
        out[cond] = cell
    return out


def _s_report(results):
    level = 1 - 0.05 / N_ENV
    intervals = {}
    for (arm, cond), cell in results.items():
        if arm == "nocomm":
            continue
        s = np.asarray(cell["S"], float)
        ci = one_sample_ci(s, level)
        intervals[(arm, cond)] = {
            "mean": float(s.mean()),
            "low": None if ci is None else ci[1],
            "high": None if ci is None else ci[2],
            "level": level,
        }
    pc = intervals.get(("eng", "closed"))
    if pc is None or pc["low"] is None:
        status = "uninterpretable"
    elif pc["low"] > 0:
        status = "sensitive"
    else:
        status = "insensitive"
    zero = {
        cond: bool(np.all(np.asarray(results[("nocomm", cond)]["S"], float) == 0))
        for cond in ("registered", "closed")
        if ("nocomm", cond) in results
    }
    return {
        "level": level,
        "intervals": intervals,
        "positive_control": status,
        "interpretable": status == "sensitive",
        "nocomm_zero_check": zero,
    }


def _threshold_counts(results):
    out = {}
    for (arm, cond), cell in results.items():
        if cond != "registered":
            continue
        reached = np.asarray(cell["reached_threshold"], bool)
        t = np.asarray(cell["time_to_half"], float)
        out[arm] = {
            "reached_before_final_eval": int(np.sum(reached & (t < CAP))),
            "reached_at_final_eval": int(np.sum(reached & (t >= CAP))),
            "never_reached": int(np.sum(~reached)),
        }
    return out


RANGES = {  # metric -> (low, high) inclusive
    "R": (0.0, 1.0),
    "S": (-1.0, 1.0),
    "bandwidth_bits": (0.0, BANDWIDTH_MAX),
    "time_to_half": (0.0, float(CAP)),
}


def _validate(results):
    req = ("seeds", "R", "time_to_half", "bandwidth_bits", "S", "reached_threshold")
    for key, cell in results.items():
        for m in req:
            if m not in cell:
                raise ValueError(f"cell {key}: missing metric {m!r}")
        if [int(x) for x in np.asarray(cell["seeds"]).ravel()] != list(SEEDS):
            raise ValueError(
                f"cell {key}: seeds must be exactly {list(SEEDS)} in this order")
        for m in req[1:]:
            a = np.asarray(cell[m])
            if a.ndim != 1 or len(a) != N_SEEDS:
                raise ValueError(
                    f"cell {key}, {m}: need a 1-D array of {N_SEEDS} seeds, "
                    f"got shape {a.shape}"
                )
            if m == "reached_threshold":
                if a.dtype != bool:
                    raise ValueError(f"cell {key}: reached_threshold must be bool")
            elif not np.all(np.isfinite(a.astype(float))):
                raise ValueError(f"cell {key}, {m}: non-finite values")
            elif m in RANGES:
                lo, hi = RANGES[m]
                af = a.astype(float)
                if np.any(af < lo) or np.any(af > hi):
                    raise ValueError(f"cell {key}, {m}: value outside range [{lo}, {hi}]")
        reached = np.asarray(cell["reached_threshold"], bool)
        t = np.asarray(cell["time_to_half"], float)
        if np.any(~reached & (t != CAP)):
            raise ValueError(f"cell {key}: never-reaching runs must carry the cap")
        if np.any(t > CAP):
            raise ValueError(f"cell {key}: time_to_half exceeds the cap")


def _by_host(results):
    """Descriptive per-host summaries on the fixed 3-seed host blocks.

    Per host: per-cell arm means of every metric, the point estimate of
    every primary and secondary contrast (same sign conventions as the
    pooled report) and S per cell. No interval, level or label.
    """
    out = {}
    for host in HOSTS:
        idx = [i for i, sd in enumerate(SEEDS) if SEED_HOST[sd] == host]

        def blk(arm, cond, metric, idx=idx):
            return _metric(results, arm, cond, metric)[idx]

        arm_means = {
            key: {m: float(blk(*key, m).mean()) for m in ("R", "time_to_half",
                                                          "bandwidth_bits", "S")}
            for key in results
        }
        primary, secondary = {}, []
        if ("gtm", "registered") in results:
            for b in BASELINES:
                if (b, "registered") not in results:
                    continue
                primary[b] = float(blk("gtm", "registered", "R").mean()
                                   - blk(b, "registered", "R").mean())
                for m in METRICS:
                    g = blk("gtm", "registered", m).mean()
                    o = blk(b, "registered", m).mean()
                    est = o - g if m == "time_to_half" else g - o
                    secondary.append({"baseline": b, "metric": m,
                                      "estimate": float(est)})
        out[host] = {
            "seeds": [SEEDS[i] for i in idx],
            "arm_means": arm_means,
            "primary": primary,
            "secondary": secondary,
            "S": {key: arm_means[key]["S"] for key in results if key[0] != "nocomm"},
        }
    return out


def primary_report(results):
    """Build the full frozen report from per-(arm, condition) seed arrays."""
    _validate(results)
    return {
        "primary": _primary(results),
        "secondary": _secondary(results),
        "normalized_improvement": _ni(results),
        "S": _s_report(results),
        "threshold_counts": _threshold_counts(results),
        "by_host": _by_host(results),
    }


def cross_play_report(results):
    """Descriptive cross-seed cross-play summary (amendment 2026-10-23).

    ``results`` must hold ALL ten (arm, condition) cells (6 registered, 4
    closed). Each cell: ``"seeds"`` (= the six registered seeds in order),
    ``"pairs"`` (= the six fixed (role-0 seed, role-1 seed) pairs in order),
    ``"R_intact"`` (one per seed), ``"R_cross"`` (one per pair, in [0, 1])
    and ``"untrained"`` (bool per pair: a failed/untrained model took part).
    Per cell: the cross values, the pair identities, the untrained flags and
    their count, the means and the gap ``mean_intact - mean_cross``. No
    interval, p-value, category or between-arm comparison is computed, and a
    gap is not a loss of communication.
    """
    missing = [c for c in CELLS if c not in results]
    extra = [k for k in results if k not in CELLS]
    if missing or extra:
        raise ValueError(f"cross-play cells must be exactly the {len(CELLS)} "
                         f"registered cells; missing {missing}, unexpected {extra}")
    out = {}
    for key in CELLS:
        cell = results[key]
        for m in ("seeds", "pairs", "R_intact", "R_cross", "untrained"):
            if m not in cell:
                raise ValueError(f"cell {key}: missing metric {m!r}")
        if [int(x) for x in cell["seeds"]] != list(SEEDS):
            raise ValueError(f"cell {key}: seeds must be {list(SEEDS)}")
        if [tuple(int(x) for x in p) for p in cell["pairs"]] != list(CROSS_PAIRS):
            raise ValueError(f"cell {key}: pairs must be the six fixed pairs "
                             f"{list(CROSS_PAIRS)} in order")
        arrs = {}
        for m in ("R_intact", "R_cross"):
            a = np.asarray(cell[m], float)
            if a.ndim != 1 or len(a) != N_SEEDS or not np.all(np.isfinite(a)):
                raise ValueError(
                    f"cell {key}, {m}: need {N_SEEDS} finite values, one each")
            if np.any(a < 0) or np.any(a > RANGE):
                raise ValueError(f"cell {key}, {m}: value outside range [0, {RANGE}]")
            arrs[m] = a
        un = np.asarray(cell["untrained"])
        if un.dtype != bool or un.ndim != 1 or len(un) != N_SEEDS:
            raise ValueError(f"cell {key}: untrained must be {N_SEEDS} booleans")
        mi, mc = float(arrs["R_intact"].mean()), float(arrs["R_cross"].mean())
        out[key] = {
            "R_cross": [float(x) for x in arrs["R_cross"]],
            "pairs": [list(p) for p in CROSS_PAIRS],
            "untrained": [bool(x) for x in un],
            "n_untrained": int(un.sum()),
            "mean_cross": mc,
            "mean_intact": mi,
            "gap": mi - mc,
        }
    return out
