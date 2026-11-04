"""Pre-registered analysis for P1bis-ext and P8c-nocap (pure NumPy + SciPy).

Usage::

    python -m experiments.hopfield_dream_conf.analyse p1bis <out_dir>
    python -m experiments.hopfield_dream_conf.analyse p8c   <out_dir>

Writes ``verdict.json`` next to the raw files.  Every rule implemented here is
stated in the corresponding pre-registration; the labels it prints are the only
verdict vocabulary allowed in the results note.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys

import numpy as np

from experiments.hopfield_dream_conf import stats

MIN_SEEDS = 16          # a test is analysed only if >= 16 of 20 seeds are measured

# ----------------------------------------------------------------- P1bis-ext
P1_FRACS = (0.10, 0.20, 0.30)
P1_NS = (1000, 2000, 4000)
P1_CTRL = ("d", "g")
P1_MIN_EFFECT = 0.02
P1_HEBB_WINDOW = (0.035, 0.060)     # gate: mean Hebb rho at alpha = 0.15, every N
P1_FRESH_MAX = 0.005                # gate: rho on fresh patterns, every row


def _load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def p1bis_diffs(rows, metric):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["N"], r["f"], r["arm"])][r["seed"]] = r

    def diff(n, f, ctrl):
        b, o = by.get((n, f, "b"), {}), by.get((n, f, ctrl), {})
        seeds = sorted(s for s in b if s in o and b[s].get("ok") and o[s].get("ok")
                       and b[s].get(metric) is not None and o[s].get(metric) is not None)
        if len(seeds) < MIN_SEEDS:
            return []
        return [b[s][metric] - o[s][metric] for s in seeds]

    return diff, by


def analyse_p1bis(out_dir):
    rows = _load(os.path.join(out_dir, "grid.jsonl"))
    diff_rho, by = p1bis_diffs(rows, "rho")
    diff_rec, _ = p1bis_diffs(rows, "recall_q0.1")
    gates = {}
    hebb = {n: [r["rho"] for (nn, f, arm), d in by.items() if nn == n and arm == "a"
                for r in d.values()] for n in P1_NS}
    gates["hebb_rho_mean"] = {str(n): float(np.mean(v)) if v else None for n, v in hebb.items()}
    gates["hebb_in_window"] = all(v and P1_HEBB_WINDOW[0] <= np.mean(v) <= P1_HEBB_WINDOW[1]
                                  for v in hebb.values())
    fresh = [r["rho_fresh"] for r in rows if r.get("rho_fresh") is not None]
    gates["max_rho_fresh"] = max(fresh) if fresh else None
    gates["fresh_ok"] = bool(fresh) and max(fresh) <= P1_FRESH_MAX
    gates["pass"] = bool(gates["hebb_in_window"] and gates["fresh_ok"])

    prim = stats.family_decisions(
        {f"N{n}_b-{c}_f0.20": diff_rho(n, 0.20, c) for n in P1_NS for c in P1_CTRL},
        sup_margin=0.0, inf_margin=P1_MIN_EFFECT, min_effect=P1_MIN_EFFECT)
    sec1 = stats.family_decisions(
        {f"N{n}_b-{c}_f{f:.2f}": diff_rho(n, f, c) for n in P1_NS for c in P1_CTRL
         for f in (0.10, 0.30)},
        sup_margin=0.0, inf_margin=P1_MIN_EFFECT, min_effect=P1_MIN_EFFECT)
    sec2 = stats.family_decisions(
        {f"N{n}_b-{c}_f0.20_recall0.1": diff_rec(n, 0.20, c) for n in P1_NS for c in P1_CTRL},
        sup_margin=0.0, inf_margin=0.0, min_effect=0.0)
    labels = [v["label"] for v in prim.values()]
    if not gates["pass"]:
        overall = "void_gate_failed"
    elif all(lab == "success" for lab in labels):
        overall = "supported"
    elif any(lab == "below" for lab in labels):
        overall = "not_supported"
    else:
        overall = "inconclusive"
    return {"study": "hopfield-p1bis-ext", "gates": gates, "primary": prim,
            "secondary_rho_f010_f030": sec1, "secondary_recall_q01": sec2,
            "overall_primary": overall}


# ----------------------------------------------------------------- P8c-nocap
P8_NS = (4096, 16384)
P8_ALPHAS = (0.3, 0.4, 0.5)
P8_T = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, "inf")
P8_R = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
P8_Q_PRIMARY = (0.2, 0.3)
P8_Q_SECONDARY = (0.1,)
P8_FLOOR = 0.05             # both selected arms below this mean recovery on B -> "floor"
P8_MIN_EFFECT = 0.10
P8_GAP_TOL = 0.01
P8_GATE_KERNEL_MIN = 0.90   # alpha = 0.3, N = 4096, q = 0.1: mean best-kernel recovery on B
P8_GATE_HEBB_MAX = 0.05     # alpha = 0.3, N = 4096, q = 0.1: mean Hebb recovery on B


def _tkey(t):
    return "inf" if t in ("inf", math.inf) else float(t)


def p8c_seed_tables(rows):
    """Map (N, alpha, seed) -> {"hebb": row, "kernel": {t: row}, "grid": {r: row},
    "matched": {t: row}}; stochastic rows outside the 1 % budget gap are dropped."""
    tab = collections.defaultdict(lambda: {"hebb": None, "kernel": {}, "grid": {}, "matched": {}})
    for r in rows:
        key = (r["N"], r["alpha"], r["seed"])
        if r["arm"] == "hebb":
            tab[key]["hebb"] = r
        elif r["arm"] == "kernel":
            tab[key]["kernel"][_tkey(r["t"])] = r
        elif r["arm"] == "stoch":
            ok = r.get("status") != "not_measured" and r.get("gap", 1.0) <= P8_GAP_TOL
            if not ok:
                continue
            if r["kind"] == "grid":
                tab[key]["grid"][round(r["r"], 6)] = r
            else:
                tab[key]["matched"][_tkey(r["t"])] = r
    return tab


def _argmax_first(vals):
    """Index of the maximum, ties -> first (smallest t or r in grid order)."""
    best, arg = -1.0, None
    for i, v in enumerate(vals):
        if v > best:
            best, arg = v, i
    return arg


def p8c_contrasts(seed_tab, q):
    kq_a, kq_b = f"rec_q{q:.1f}_A", f"rec_q{q:.1f}_B"
    k = seed_tab["kernel"]
    g = seed_tab["grid"]
    m = seed_tab["matched"]
    best = match = None
    if all(_tkey(t) in k for t in P8_T) and all(round(r, 6) in g for r in P8_R):
        ka = [k[_tkey(t)][kq_a] for t in P8_T]
        sa = [g[round(r, 6)][kq_a] for r in P8_R]
        it, ir = _argmax_first(ka), _argmax_first(sa)
        best = {"delta": k[_tkey(P8_T[it])][kq_b] - g[round(P8_R[ir], 6)][kq_b],
                "t_hat": P8_T[it], "r_hat": P8_R[ir],
                "kernel_B": k[_tkey(P8_T[it])][kq_b], "stoch_B": g[round(P8_R[ir], 6)][kq_b]}
    if all(_tkey(t) in k for t in P8_T) and all(_tkey(t) in m for t in P8_T):
        da = [k[_tkey(t)][kq_a] - m[_tkey(t)][kq_a] for t in P8_T]
        it = _argmax_first(da)
        t = _tkey(P8_T[it])
        match = {"delta": k[t][kq_b] - m[t][kq_b], "t_tilde": P8_T[it]}
    return best, match


def analyse_p8c(out_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "**", "cells.jsonl"), recursive=True))
    rows = [r for p in paths for r in _load(p) if r.get("arm") != "meta"]
    tab = p8c_seed_tables(rows)
    best_d, match_d, detail = {}, {}, {}
    for n in P8_NS:
        for a in P8_ALPHAS:
            for q in P8_Q_PRIMARY + P8_Q_SECONDARY:
                key = f"N{n}_a{a}_q{q}"
                bs, ms = [], []
                det = []
                for (nn, aa, seed), st in sorted(tab.items()):
                    if nn != n or aa != a:
                        continue
                    b, m = p8c_contrasts(st, q)
                    if b is not None:
                        bs.append(b["delta"])
                        det.append(dict(seed=seed, **b))
                    if m is not None:
                        ms.append(m["delta"])
                best_d[key] = bs if len(bs) >= MIN_SEEDS else []
                match_d[key] = ms if len(ms) >= MIN_SEEDS else []
                detail[key] = det
    # gates (alpha = 0.3, N = 4096, q = 0.1)
    kb, hb = [], []
    for (n, a, _seed), st in tab.items():
        if n == 4096 and a == 0.3 and st["hebb"] is not None and st["kernel"]:
            hb.append(st["hebb"]["rec_q0.1_B"])
            ka = [st["kernel"][_tkey(t)]["rec_q0.1_A"] for t in P8_T if _tkey(t) in st["kernel"]]
            tb = [st["kernel"][_tkey(t)]["rec_q0.1_B"] for t in P8_T if _tkey(t) in st["kernel"]]
            kb.append(tb[_argmax_first(ka)])
    gates = {"kernel_best_mean": float(np.mean(kb)) if kb else None,
             "hebb_mean": float(np.mean(hb)) if hb else None}
    gates["pass"] = bool(kb and hb and np.mean(kb) >= P8_GATE_KERNEL_MIN
                         and np.mean(hb) <= P8_GATE_HEBB_MAX)
    prim_keys = [f"N{n}_a{a}_q{q}" for n in P8_NS for a in P8_ALPHAS for q in P8_Q_PRIMARY]
    sec_keys = [f"N{n}_a{a}_q{q}" for n in P8_NS for a in P8_ALPHAS for q in P8_Q_SECONDARY]
    prim = stats.family_decisions({k: best_d[k] for k in prim_keys}, 0.0, P8_MIN_EFFECT,
                                  P8_MIN_EFFECT)
    for k in prim_keys:          # floor rule: no arm recovers -> uninformative, not "below"
        det = detail[k]
        if prim[k]["label"] != "not_measured" and det:
            mk = float(np.mean([d["kernel_B"] for d in det]))
            ms = float(np.mean([d["stoch_B"] for d in det]))
            prim[k]["mean_kernel_B"], prim[k]["mean_stoch_B"] = mk, ms
            if max(mk, ms) < P8_FLOOR and prim[k]["label"] != "success":
                prim[k]["label"] = "floor"
    sec_match = stats.family_decisions({k: match_d[k] for k in prim_keys}, 0.0, P8_MIN_EFFECT,
                                       P8_MIN_EFFECT)
    sec_q1 = stats.family_decisions({k: best_d[k] for k in sec_keys}, 0.0, P8_MIN_EFFECT,
                                    P8_MIN_EFFECT)
    labels = [v["label"] for v in prim.values()]
    informative = [lab for lab in labels if lab != "floor"]
    if not gates["pass"]:
        overall = "void_gate_failed"
    elif any(lab == "success" for lab in labels):
        overall = "supported_in_cells:" + ",".join(k for k in prim_keys
                                                     if prim[k]["label"] == "success")
    elif informative and all(lab == "below" for lab in informative):
        overall = "not_supported"
    else:
        overall = "inconclusive"
    return {"study": "hopfield-p8c-nocap", "gates": gates, "primary_best_vs_best": prim,
            "secondary_matched_budget": sec_match, "secondary_q01_best_vs_best": sec_q1,
            "selection_detail": detail, "overall_primary": overall}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    which, out_dir = argv[0], argv[1]
    res = analyse_p1bis(out_dir) if which == "p1bis" else analyse_p8c(out_dir)
    path = os.path.join(out_dir, "verdict.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "selection_detail"}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
