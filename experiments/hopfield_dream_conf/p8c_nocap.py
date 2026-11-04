"""P8c-nocap (confirmatory, L0 draft): Fachechi-Agliari-Barra dreaming kernel
versus Hebbian unlearning (HU, Marinari/Benedetti) WITHOUT the dose cap of the
exploratory pilot.

Pre-registration (binding once frozen): ``docs/prereg/hopfield-p8c-nocap/prereg.md``.

Network: P = round(alpha N) independent +-1 patterns, couplings scaled 1/sqrt(N),
no clipping, zero diagonal (as in the pilot ``gb300-p8c-dream-kernel``).

Arms (one cell = one (N, alpha, seed)):
  hebb    J0 = X^T X / sqrt(N)
  kernel  J(t) = X^T (1+t)(I + tC)^-1 X / sqrt(N), C = X X^T / N, by the exact
          eigendecomposition of C (float64); t in T_GRID (t = inf: C^-1)
  stoch   HU from J0: random states relaxed to fixed points by asynchronous
          zero-temperature dynamics on a frozen J (batches of at most
          DREAM_BATCH states), then J <- J - (1/(tau_d sqrt N)) sum_a s_a s_a^T
          (zero diagonal), tau_d = 100.  ONE trajectory per cell, checkpointed
          at the first crossing of each relative budget r = ||J - J0||_F /
          ||J0||_F in R_GRID and at each kernel budget r_kernel(t) ("matched").
          No 0.5 cap: the trajectory runs to max(R_GRID) unless the per-cell
          wall-time ceiling is reached (remaining checkpoints "not_measured").

Measures at every checkpoint: recovery from noisy cues (q in CUE_QS, exactly
round(q N) flipped spins, asynchronous relaxation, recovered if overlap >= 0.95)
on two disjoint pattern sets A (selection) and B (evaluation), 200 patterns
each; identical cues and visiting orders for every arm of a cell.

Dynamics: asynchronous sign dynamics, random visiting order per sweep, flip iff
s_i h_i < 0 (h_i = 0 keeps the spin), local fields updated incrementally - the
rule of the vendored Marinari module, implemented here for a batch of
independent states sharing the visiting order (each state follows ordinary
asynchronous dynamics; the batch only shares row reads of J).

Determinism: each cell is a pure function of (seed, N, alpha); BLAS and numba
are single-threaded; the dream trajectory re-seeds its stream per batch, so
evaluations do not perturb it.  Outputs (timing excluded) do not depend on the
number of workers, the scheduling order or the core type.
"""

from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

STUDY = "hopfield-p8c-nocap"
ROOT_SEED = 20261104
STUDY_ID = 2
TAU_D = 100.0
NS = (4096, 16384)
ALPHAS = (0.3, 0.4, 0.5)
T_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, math.inf)
R_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CUE_QS = (0.1, 0.2, 0.3)
CUE_THR = 0.95
N_CUE = 200            # per set (A and B)
DREAM_BATCH = 64
MAX_SWEEPS = 2000
N_STAB = 512           # patterns used for the descriptive stability statistics
CONF_SEEDS = tuple(range(32001, 32021))
SMOKE_MIN_SEED = 90000


def t_label(t: float):
    return "inf" if math.isinf(t) else float(t)


def streams(seed: int, n: int, alpha: float):
    """Independent generators and integer seeds derived from (seed, N, alpha)."""
    ss = np.random.SeedSequence([ROOT_SEED, STUDY_ID, seed, n, int(round(alpha * 100))])
    pat, cue, dream, order = ss.spawn(4)
    return (np.random.default_rng(pat), np.random.default_rng(cue),
            int(dream.generate_state(1, dtype=np.uint32)[0] % (2**31 - 2**20)),
            int(order.generate_state(1, dtype=np.uint32)[0] % (2**31 - 2**20)))


# ----------------------------------------------------------------- dynamics
@njit(cache=False)
def _seed(s):
    np.random.seed(s)


@njit(cache=False)
def relax_batch(S, H, J, max_sweeps):
    """Asynchronous zero-T dynamics for every row of S (in place).

    H must hold S @ J on entry (J symmetric, zero diagonal).  Each sweep visits
    the neurons in a fresh random order (numba global stream); a spin flips iff
    s_i h_i < 0.  Returns (number of rows not converged, sweeps used)."""
    b, n = S.shape
    active = np.ones(b, dtype=np.bool_)
    n_active = b
    sweeps = 0
    while n_active > 0 and sweeps < max_sweeps:
        sweeps += 1
        flipped = np.zeros(b, dtype=np.bool_)
        order = np.random.permutation(n)
        for ii in range(n):
            i = order[ii]
            row = J[i]
            for a in range(b):
                if active[a] and S[a, i] * H[a, i] < 0.0:
                    old = S[a, i]
                    S[a, i] = -old
                    d = np.float32(-2.0) * old
                    flipped[a] = True
                    Ha = H[a]
                    for k in range(n):
                        Ha[k] += row[k] * d
        n_active = 0
        for a in range(b):
            if active[a] and not flipped[a]:
                active[a] = False
            if active[a]:
                n_active += 1
    return n_active, sweeps


def relax(S, J, order_seed, return_fields=False):
    S = np.array(S, dtype=np.float32, order="C", copy=True)   # never relax the caller's array
    H = S @ J
    _seed(order_seed)
    nc, sw = relax_batch(S, H, J, MAX_SWEEPS)
    if return_fields:
        return S, int(nc), int(sw), H
    return S, int(nc), int(sw)


# ----------------------------------------------------------------- couplings
def hebb(X, n):
    J = X.T @ X                      # exact integers in float32 (|sum| <= P < 2**24)
    J *= np.float32(1.0 / math.sqrt(n))
    np.fill_diagonal(J, 0.0)
    return J


def kernel_factor(X, n):
    """Eigendecomposition of C = X X^T / N (float64) and Y = X^T V (float32)."""
    G = (X @ X.T).astype(np.float64) / n
    lam, V = np.linalg.eigh(G)
    del G
    Y = X.T @ V.astype(np.float32)
    return lam, Y


def kernel_coupling(lam, Y, n, t):
    g = 1.0 / lam if math.isinf(t) else (1.0 + t) / (1.0 + t * lam)
    Yg = Y * np.sqrt(g).astype(np.float32)[None, :]
    J = Yg @ Yg.T
    del Yg
    J *= np.float32(1.0 / math.sqrt(n))
    np.fill_diagonal(J, 0.0)
    return J


def fro(J, rows=1024):
    s = 0.0
    for r0 in range(0, J.shape[0], rows):
        blk = J[r0:r0 + rows].astype(np.float64)
        s += float(np.einsum("ij,ij->", blk, blk))
    return math.sqrt(s)


def fro_diff(J, J0, rows=1024):
    s = 0.0
    for r0 in range(0, J.shape[0], rows):
        blk = J[r0:r0 + rows].astype(np.float64) - J0[r0:r0 + rows]
        s += float(np.einsum("ij,ij->", blk, blk))
    return math.sqrt(s)


def unlearn(J, S, c, rows=1024):
    """J <- J - c S^T S with zero diagonal, in row blocks."""
    for r0 in range(0, J.shape[0], rows):
        r1 = min(J.shape[0], r0 + rows)
        J[r0:r1] -= np.float32(c) * (S[:, r0:r1].T @ S)
    np.fill_diagonal(J, 0.0)


# ----------------------------------------------------------------- measures
def make_cues(X, idx, q, rng):
    n = X.shape[1]
    k = int(round(q * n))
    C = X[idx].copy()
    for r in range(C.shape[0]):
        C[r, rng.choice(n, size=k, replace=False)] *= -1
    return C


def evaluate(J, ctx):
    X, n = ctx["X"], ctx["N"]
    out = {}
    nonconv = 0
    for qi, q in enumerate(CUE_QS):
        for si, name in enumerate(("A", "B")):
            idx = ctx["sets"][name]
            S, nc, _ = relax(ctx["cues"][(q, name)], J, ctx["order_seed"] + 10 * qi + si)
            nonconv += nc
            ov = (S * X[idx]).sum(axis=1) / n
            out[f"rec_q{q:.1f}_{name}"] = float(np.mean(ov >= CUE_THR))
    # descriptive stability on a fixed subset of stored patterns
    Xs = X[ctx["stab_idx"]]
    Hs = Xs @ J
    rown = np.empty(n)
    for r0 in range(0, n, 1024):
        blk = J[r0:r0 + 1024].astype(np.float64)
        rown[r0:r0 + 1024] = np.sqrt(np.einsum("ij,ij->i", blk, blk))
    stab = (Xs * Hs) / np.maximum(rown, 1e-30)[None, :]
    out["stable_frac_sub"] = float(np.mean(np.all(stab > 0, axis=1)))
    out["delta_min_sub"] = float(stab.min())
    out["cue_nonconv"] = int(nonconv)
    return out


# ----------------------------------------------------------------- cell
def run_cell(n, alpha, seed, max_stoch_secs, stopfile):
    t0 = time.time()
    P = round(alpha * n)
    if P < 2 * N_CUE:
        raise ValueError(f"P = {P} < {2 * N_CUE}: cue sets A and B need distinct patterns")
    rng_pat, rng_cue, dream_seed, order_seed = streams(seed, n, alpha)
    X = (rng_pat.integers(0, 2, size=(P, n), dtype=np.int8) * 2 - 1).astype(np.float32)
    perm = rng_cue.permutation(P)
    sets = {"A": np.sort(perm[:N_CUE]), "B": np.sort(perm[N_CUE:2 * N_CUE])}
    cues = {(q, name): make_cues(X, sets[name], q, rng_cue) for q in CUE_QS for name in "AB"}
    stab_idx = np.sort(rng_cue.choice(P, size=min(P, N_STAB), replace=False))
    ctx = {"X": X, "N": n, "sets": sets, "cues": cues, "order_seed": order_seed,
           "stab_idx": stab_idx}
    common = {"study": STUDY, "N": n, "alpha": alpha, "P": P, "seed": seed}
    rows = []

    J0 = hebb(X, n)
    n0 = fro(J0)
    rows.append(dict(common, arm="hebb", t=0.0, r=0.0, kind="ref", **evaluate(J0, ctx),
                     secs=round(time.time() - t0, 2)))

    lam, Y = kernel_factor(X, n)
    r_kernel = {}
    for t in T_GRID:
        if os.path.exists(stopfile):
            return None
        Jt = kernel_coupling(lam, Y, n, t)
        r = fro_diff(Jt, J0) / n0
        r_kernel[t] = r
        rows.append(dict(common, arm="kernel", t=t_label(t), r=r, kind="grid",
                         lam_min=float(lam[0]), lam_max=float(lam[-1]), **evaluate(Jt, ctx),
                         secs=round(time.time() - t0, 2)))
        del Jt
    del Y

    # stochastic HU trajectory, no dose cap
    targets = [(r, "grid", None) for r in R_GRID]
    targets += [(r_kernel[t], "matched", t_label(t)) for t in T_GRID]
    targets.sort(key=lambda x: (x[0], x[1] != "matched"))
    c = 1.0 / (TAU_D * math.sqrt(n))
    J = J0.copy()
    done, batch_i, cur, sq = 0, 0, 0.0, 0.0
    nonconv_dream, ovl_acc, ovl_n = 0, 0.0, 0
    t_s = time.time()
    status = "complete"
    ti = 0
    Xs = X[stab_idx]
    while ti < len(targets):
        if os.path.exists(stopfile):
            return None
        if time.time() - t_s > max_stoch_secs:
            status = "ceiling"
            break
        target = targets[ti][0] * n0
        if done == 0 or cur <= 0.0:
            b = 1
        else:
            est = done * (target / cur) - done
            b = int(max(1, min(DREAM_BATCH, math.ceil(max(1.0, est) / 2))))
        rng_b = np.random.default_rng([dream_seed, batch_i])
        S0 = (rng_b.integers(0, 2, size=(b, n), dtype=np.int8) * 2 - 1).astype(np.float32)
        S, nc, _, H = relax(S0, J, (dream_seed + 7919 * batch_i) % (2**31 - 1),
                            return_fields=True)
        nonconv_dream += nc
        ovl_acc += float(np.abs(Xs @ S.T).max(axis=0).sum() / n)
        ovl_n += b
        # incremental ||J - J0||^2 (scheduling only; checkpoints use the exact norm):
        # <dJ, S^T S> = sum_a s_a^T J s_a - s_a^T J0 s_a, s^T J0 s = (|X s|^2 - P N)/sqrt(N)
        sJs = np.einsum("ij,ij->i", S.astype(np.float64), H.astype(np.float64))
        XS = (X @ S.T).astype(np.float64)
        sJ0s = (np.einsum("ij,ij->j", XS, XS) - P * n) / math.sqrt(n)
        G = (S @ S.T).astype(np.float64)
        sq = sq - 2.0 * c * float(np.sum(sJs - sJ0s)) + c * c * float(np.sum(G * G) - b * b * n)
        unlearn(J, S, c)
        done += b
        batch_i += 1
        cur = math.sqrt(max(sq, 0.0))
        if cur >= targets[ti][0] * n0 or batch_i % 64 == 0:
            cur = fro_diff(J, J0)       # exact
            sq = cur * cur
        while ti < len(targets) and cur >= targets[ti][0] * n0:
            r_t, kind, t_m = targets[ti]
            rows.append(dict(common, arm="stoch", t=t_m, r=r_t, kind=kind, r_realised=cur / n0,
                             gap=abs(cur / n0 - r_t) / r_t, D_steps=done,
                             dream_nonconv=nonconv_dream,
                             visited_ovl_mean=ovl_acc / max(1, ovl_n),
                             **evaluate(J, ctx), secs=round(time.time() - t0, 2)))
            ti += 1
    for r_t, kind, t_m in targets[ti:]:
        rows.append(dict(common, arm="stoch", t=t_m, r=r_t, kind=kind, status="not_measured",
                         reason="ceiling", D_steps=done, r_reached=cur / n0))
    meta = dict(common, arm="meta", status=status, normJ0=n0, D_steps=done,
                r_kernel={str(t_label(t)): v for t, v in r_kernel.items()},
                secs=round(time.time() - t0, 2), stoch_secs=round(time.time() - t_s, 2),
                host=os.uname().nodename, pid=os.getpid())
    return rows, meta


def _worker(task):
    n, alpha, seed, max_stoch_secs, stopfile = task
    res = run_cell(n, alpha, seed, max_stoch_secs, stopfile)
    return task, res


def done_cells(path):
    s = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                if r.get("arm") == "meta":
                    s.add((r["N"], r["alpha"], r["seed"]))
            except (ValueError, KeyError):
                pass
    return s


def parse_list(text, cast):
    out = []
    for part in text.split(","):
        if "-" in part and cast is int:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(cast(part))
    return out


def main(argv=None):
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="confirmatory")
    ap.add_argument("--ns", default=",".join(str(x) for x in NS))
    ap.add_argument("--alphas", default=",".join(str(x) for x in ALPHAS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-stoch-secs", type=float, default=6 * 3600.0)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--confirmatory", action="store_true")
    a = ap.parse_args(argv)
    seeds = list(CONF_SEEDS) if a.seeds == "confirmatory" else parse_list(a.seeds, int)
    if any(s < SMOKE_MIN_SEED for s in seeds) and not a.confirmatory:
        sys.exit("refusing confirmatory seeds without --confirmatory (freeze tag required)")
    if a.confirmatory and any(s >= SMOKE_MIN_SEED for s in seeds):
        sys.exit("--confirmatory with smoke seeds is not allowed")
    ns = parse_list(a.ns, int)
    alphas = parse_list(a.alphas, float)
    os.makedirs(a.out, exist_ok=True)
    stopfile = a.stop_file or os.path.join(a.out, "STOP")
    path = os.path.join(a.out, "cells.jsonl")
    done = done_cells(path)
    # largest N and alpha first (longest tasks first); results do not depend on order
    tasks = [(n, al, s, a.max_stoch_secs, stopfile) for n in sorted(ns, reverse=True)
             for al in sorted(alphas, reverse=True) for s in seeds if (n, al, s) not in done]
    with open(os.path.join(a.out, "run_info.json"), "a") as fi:
        fi.write(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), "argv": sys.argv,
                             "n_tasks": len(tasks), "workers": a.workers,
                             "numpy": np.__version__, "host": os.uname().nodename}) + "\n")
    fout = open(path, "a")
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for task, res in pool.imap_unordered(_worker, tasks, chunksize=1):
            n, al, s = task[:3]
            if res is None:
                print(time.strftime("%FT%TZ", time.gmtime()), STUDY, n, al, s, "STOPPED",
                      flush=True)
                continue
            rows, meta = res
            for row in rows:
                fout.write(json.dumps(row) + "\n")
            fout.write(json.dumps(meta) + "\n")
            fout.flush()
            print(time.strftime("%FT%TZ", time.gmtime()), STUDY, n, al, s, meta["status"],
                  f"D={meta['D_steps']}", f"{meta['secs']:.0f}s", flush=True)
            if os.path.exists(stopfile):
                pool.terminate()
                break
    print(time.strftime("%FT%TZ", time.gmtime()), STUDY, "END", flush=True)


if __name__ == "__main__":
    main()
