"""P1bis-ext (confirmatory, L0 draft): Marinari dreaming vs two budget-matched
controls in a clipped Hopfield network, budget = fraction f of ||J_Hebb||_F.

Pre-registration (binding once frozen): ``docs/prereg/hopfield-p1bis-ext/prereg.md``.

Arms (identical definitions to the exploratory pilot
``pilots/cpu-p1bis-hopfield-fracbudget/pilot_p1bis.py``, GB300 slot 2026-11-01):
  a  Hebb, clipped at A = 0.4 (tau_l = 1)
  b  Marinari dreaming (tau_d = 100, clipped), one trajectory per cell;
     checkpoint at the first crossing of each target f * ||J_Hebb||_F, the
     last step's scale bisected to land within +-0.5 % of the target
  d' random-state unlearning: m random (non-relaxed) states, m = number of
     dream steps of b at that checkpoint, scale bisected (no cap)
  g  symmetric Gaussian noise (zero diagonal) added then clipped, scale bisected

Changes relative to the pilot (declared in the pre-registration):
  * f grid {0.10, 0.20, 0.30}; N in {1000, 2000, 4000}; alpha = 0.15 only;
  * seeds from a new namespace, cell RNG derived with SeedSequence;
  * new measure: recall from a noisy cue (q in {0.1, 0.2} flipped spins,
    recovered if final overlap >= 0.95), same cues and same relaxation-order
    stream for every arm of a cell (common random numbers).

The dynamics and learning/dreaming steps are the vendored upstream module
``vendor/hopfield.py`` (Marinari, Rossi, Zamponi, MIT, commit 7e42574, sha256
in the manifest), imported unchanged.

Determinism: one cell = one pure function of (seed, N, alpha) run in one
process with single-threaded BLAS/numba; outputs (timing fields excluded) do
not depend on the number of workers, scheduling order or core type.
"""

from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

from experiments.hopfield_dream_conf.vendor import hopfield as hf  # noqa: E402

STUDY = "hopfield-p1bis-ext"
ROOT_SEED = 20261104
STUDY_ID = 1
TAU_L, TAU_D, A_CLIP = 1.0, 100.0, 0.4
FRACS = (0.10, 0.20, 0.30)
NS = (1000, 2000, 4000)
ALPHA = 0.15
CONF_SEEDS = tuple(range(31001, 31021))     # confirmatory namespace (20 seeds)
SMOKE_MIN_SEED = 90000                      # smoke/pilot seeds must be >= this
CUE_QS = (0.1, 0.2)
CUE_THR = 0.95
N_CUE = 200
TOL = 0.005          # bisection target |realised/target - 1| < TOL
OK_TOL = 0.02        # "measured" rule: |realised/target - 1| <= OK_TOL
SIGN = np.array([-1, 1], dtype=np.float32)


def cell_base(seed: int, n: int, alpha: float) -> int:
    ss = np.random.SeedSequence([ROOT_SEED, STUDY_ID, seed, n, int(round(alpha * 1000))])
    return int(ss.generate_state(1, dtype=np.uint32)[0] % (2**31 - 64))


@njit(cache=False)
def fro_diff(A, B):
    s = 0.0
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            d = np.float64(A[i, j]) - np.float64(B[i, j])
            s += d * d
    return np.sqrt(s)


@njit(cache=False)
def lin_clip(Jb, M, c, A, out):
    """out = clip(Jb + c*M, -A, A), zero diagonal."""
    n = Jb.shape[0]
    for i in range(n):
        for j in range(n):
            v = Jb[i, j] + c * M[i, j]
            if v > A:
                v = A
            elif v < -A:
                v = -A
            out[i, j] = v
        out[i, i] = 0.0


@njit(cache=False)
def dream_step(J, s, coef, A):
    """In place: J <- clip(J - coef s s^T) (Marinari Eq. 3), zero diagonal."""
    n = J.shape[0]
    for i in range(n):
        for j in range(n):
            v = J[i, j] - coef * s[i] * s[j]
            if v > A:
                v = A
            elif v < -A:
                v = -A
            J[i, j] = v
        J[i, i] = 0.0


def rho(J, X, seed):
    hf.seed_numba(seed)
    return float(hf.recognition_rate(J, X))


def make_cues(X, idx, q, rng):
    """Cues: stored patterns idx with exactly round(q N) spins flipped (no replacement)."""
    n = X.shape[1]
    k = int(round(q * n))
    C = X[idx].copy()
    for r in range(C.shape[0]):
        flip = rng.choice(n, size=k, replace=False)
        C[r, flip] *= -1
    return C


def recall(J, X, idx, cues, seed):
    """Fraction of cues whose async fixed point has overlap >= CUE_THR with its pattern."""
    hf.seed_numba(seed)
    S = hf.relax(cues.copy(), J)
    ov = (S * X[idx]).sum(axis=1) / X.shape[1]
    return float(np.mean(ov >= CUE_THR))


def match_scale(J0, M, target, A, c0=1.0):
    """Find c with ||clip(J0+c*M)-J0|| = target (+-TOL); no cap on c."""
    out = np.empty_like(J0)

    def dl(c):
        lin_clip(J0, M, np.float32(c), A, out)
        return fro_diff(out, J0)

    lo, hi = 0.0, c0
    for _ in range(60):
        if dl(hi) >= target:
            break
        lo, hi = hi, hi * 2.0
    c = hi
    for _ in range(80):
        c = 0.5 * (lo + hi)
        d = dl(c)
        if abs(d / target - 1.0) < TOL:
            break
        if d < target:
            lo = c
        else:
            hi = c
    d = dl(c)
    return float(c), float(d), out


def cell(args):
    n, alpha, seed, stopfile = args
    if os.path.exists(stopfile):
        return None
    t_cell = time.time()
    A = A_CLIP
    base = cell_base(seed, n, alpha)
    np.random.seed(base)
    hf.seed_numba(base + 1)
    P = round(alpha * n)
    rs = base + 2
    X = np.random.choice(SIGN, size=(P, n))
    Xf = np.random.choice(SIGN, size=(P, n))
    J0 = np.zeros((n, n), dtype=np.float32)
    hf.learn(J0, X, A, TAU_L)
    nJ0 = float(fro_diff(J0, np.zeros_like(J0)))
    rng = np.random.default_rng(base + 3)
    # cue sets: fixed per cell and q, shared by every arm (common random numbers)
    rng_cue = np.random.default_rng(base + 4)
    idx = np.sort(rng_cue.choice(P, size=min(P, N_CUE), replace=False))
    cues = {q: make_cues(X, idx, q, rng_cue) for q in CUE_QS}
    rs_cue = base + 5

    def measures(J):
        out = {"rho": rho(J, X, rs), "rho_fresh": rho(J, Xf, rs)}
        for q in CUE_QS:
            out[f"recall_q{q:.1f}"] = recall(J, X, idx, cues[q], rs_cue)
        return out

    coef = np.float32(1.0 / (TAU_D * np.sqrt(n)))
    inc_max = np.sqrt(n * (n - 1.0)) / (TAU_D * np.sqrt(n))
    common = dict(study=STUDY, N=n, alpha=alpha, seed=seed, P=P, normJ0=nJ0, n_cue=int(idx.size))
    rows = []
    t0 = time.time()
    rows.append(dict(common, f=0.0, arm="a", target=0.0, realised=0.0, ratio=None, n_steps=0,
                     scale=None, ok=True, **measures(J0), secs=time.time() - t0))
    # ---- arm b: one dreaming trajectory, checkpoints at first crossing of each target
    t0 = time.time()
    Jb = J0.copy()
    Jprev = J0.copy()
    delta = 0.0
    k = 0
    kmax = 30 * n
    ovl = []
    ks = 0
    ckpts = {}
    targets = [f * nJ0 for f in FRACS]
    ti = 0
    while ti < len(FRACS) and k < kmax:
        need_copy = delta + inc_max >= targets[ti]
        if need_copy:
            Jprev[:] = Jb
        s = np.random.choice(SIGN, size=(1, n))
        s = hf.relax(s, Jb)[0]
        dream_step(Jb, s, coef, A)
        k += 1
        ovl.append(float(np.abs(X @ s).max() / n))
        delta = float(fro_diff(Jb, J0))
        while ti < len(FRACS) and delta >= targets[ti]:
            assert need_copy
            M = (-coef * np.outer(s, s)).astype(np.float32)
            out = np.empty_like(J0)
            lo, hi = 0.0, 1.0
            for _ in range(60):
                c = 0.5 * (lo + hi)
                lin_clip(Jprev, M, np.float32(c), A, out)
                d = float(fro_diff(out, J0))
                if abs(d / targets[ti] - 1.0) < TOL:
                    break
                if d < targets[ti]:
                    lo = c
                else:
                    hi = c
            ckpts[ti] = dict(n_steps=k, scale=float(c), realised=d, J=out.copy(),
                             omean=float(np.mean(ovl[ks:])), omax=float(np.max(ovl[ks:])))
            ks = len(ovl)
            ti += 1
    for ti, f in enumerate(FRACS):
        t1 = time.time()
        if ti not in ckpts:
            rows.append(dict(common, f=f, arm="b", target=targets[ti], realised=delta,
                             ratio=delta / targets[ti], n_steps=k, scale=None, ok=False,
                             note="unreached", secs=0.0))
            continue
        ck = ckpts[ti]
        rows.append(dict(common, f=f, arm="b", target=targets[ti], realised=ck["realised"],
                         ratio=ck["realised"] / targets[ti], n_steps=ck["n_steps"],
                         scale=ck["scale"], ok=abs(ck["realised"] / targets[ti] - 1) <= OK_TOL,
                         ovl_mean=ck["omean"], ovl_max=ck["omax"], **measures(ck["J"]),
                         secs=time.time() - t1))
    t_b = time.time() - t0
    # ---- arm d' (random-state unlearning, m = n_steps of b) and arm g (Gaussian noise)
    Z = rng.standard_normal((n, n)).astype(np.float32)
    Z = np.triu(Z, 1)
    Z = (Z + Z.T).astype(np.float32)
    acc = np.zeros((n, n), dtype=np.float32)
    nacc = 0
    for ti, f in enumerate(FRACS):
        tg = targets[ti]
        if ti in ckpts:
            m = ckpts[ti]["n_steps"]
            while nacc < m:
                mm = min(2000, m - nacc)
                S = rng.choice(SIGN, size=(mm, n))
                acc += S.T @ S
                nacc += mm
            Md = -(acc - np.diag(np.diag(acc))) * coef
            t1 = time.time()
            c, d, Jd = match_scale(J0, Md.astype(np.float32), tg, A)
            rows.append(dict(common, f=f, arm="d", target=tg, realised=d, ratio=d / tg,
                             n_steps=nacc, scale=c, ok=abs(d / tg - 1) <= OK_TOL,
                             **measures(Jd), secs=time.time() - t1))
        else:
            rows.append(dict(common, f=f, arm="d", target=tg, realised=None, ratio=None,
                             n_steps=None, scale=None, ok=False, note="b_unreached", secs=0.0))
        t1 = time.time()
        c, d, Jg = match_scale(J0, Z, tg, A, c0=tg / max(1.0, n))
        rows.append(dict(common, f=f, arm="g", target=tg, realised=d, ratio=d / tg, n_steps=0,
                         scale=c, ok=abs(d / tg - 1) <= OK_TOL, **measures(Jg),
                         secs=time.time() - t1))
    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    ctrl = dict(common, kind="ctrl",
                max_rho_fresh=max(r["rho_fresh"] for r in rows if r.get("rho_fresh") is not None),
                min_ratio=min(ratios), max_ratio=max(ratios), all_ok=all(r["ok"] for r in rows),
                rho_hebb=rows[0]["rho"], secs=time.time() - t_cell, secs_b=t_b,
                host=os.uname().nodename, pid=os.getpid())
    return dict(rows=rows, ctrl=ctrl)


def done_cells(path):
    s = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                s.add((r["N"], r["alpha"], r["seed"]))
            except (ValueError, KeyError):
                pass
    return s


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="confirmatory",
                    help="'confirmatory' (31001-31020) or a list/range of smoke seeds >= 90000")
    ap.add_argument("--ns", default=",".join(str(x) for x in NS))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("POOL", "20")))
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--confirmatory", action="store_true",
                    help="required to run confirmatory seeds (after the freeze tag only)")
    a = ap.parse_args(argv)
    seeds = list(CONF_SEEDS) if a.seeds == "confirmatory" else parse_seeds(a.seeds)
    if any(s < SMOKE_MIN_SEED for s in seeds) and not a.confirmatory:
        sys.exit("refusing confirmatory seeds without --confirmatory (freeze tag required)")
    if a.confirmatory and any(s >= SMOKE_MIN_SEED for s in seeds):
        sys.exit("--confirmatory with smoke seeds is not allowed")
    ns = [int(x) for x in a.ns.split(",")]
    os.makedirs(a.out, exist_ok=True)
    stopfile = a.stop_file or os.path.join(a.out, "STOP")
    done = done_cells(os.path.join(a.out, "control.jsonl"))
    # largest N first (shorter tail); results do not depend on the order
    tasks = [(n, ALPHA, s, stopfile) for n in sorted(ns, reverse=True) for s in seeds
             if (n, ALPHA, s) not in done]
    with open(os.path.join(a.out, "run_info.json"), "a") as fi:
        fi.write(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), "argv": sys.argv,
                             "n_tasks": len(tasks), "workers": a.workers,
                             "numpy": np.__version__, "host": os.uname().nodename}) + "\n")
    fg = open(os.path.join(a.out, "grid.jsonl"), "a")
    fc = open(os.path.join(a.out, "control.jsonl"), "a")
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(cell, tasks, chunksize=1):
            if r is None:
                continue
            for row in r["rows"]:
                fg.write(json.dumps(row) + "\n")
            fg.flush()
            fc.write(json.dumps(r["ctrl"]) + "\n")
            fc.flush()
            c = r["ctrl"]
            print(time.strftime("%FT%TZ", time.gmtime()), STUDY, c["N"], c["alpha"], c["seed"],
                  round(c["secs"]), "ratio[{:.3f},{:.3f}]".format(c["min_ratio"], c["max_ratio"]),
                  "ok" if c["all_ok"] else "BUDGET-OFF", flush=True)
            if os.path.exists(stopfile):
                pool.terminate()
                break
    print(time.strftime("%FT%TZ", time.gmtime()), STUDY, "END", flush=True)


if __name__ == "__main__":
    main()
