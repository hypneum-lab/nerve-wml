"""Stage-2 configuration manifest writer (N10-C, pre-registered).

Builds ``stage2-manifest-DRAFT.md`` from the code (parameter counts via
``policy.param_count``, PPO constants from ``train.py``, GTM adapter from
``comm.py``) and from the measured ``throughput.json`` / ``smoke_runs.json``
found next to the output file. Reads training-level measurements only.
Run it on macM1 so the toolchain versions are collected there.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import mlx.core as mx

from . import comm, train
from .codes import ARM_CODES, CONDITION_CODES, HOST_SETTINGS, SEED_HOST, SEEDS
from .gtm_mlx import GTMConfigMLX
from .policy import PairPolicy, param_count

PKG = Path(__file__).resolve().parent.parent  # experiments/procgen_multiagent_gtm_comm
REPO = PKG.parent.parent
PATCH = PKG / "bench" / "procgen-0.10.7-arm64-build.patch"
PROCGEN_COMMIT = "5e1dbf3"
QT_VERSION = "5.15.19"
DEFAULT_SDKROOT = ("/Applications/Xcode.app/Contents/Developer/Platforms/"
                   "MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk")
PLANNING_H_PER_RUN = 17.6  # macM1, amendment (second round): ~790 pair-steps/s
PLANNING_TOTAL_H = 528.0  # macM1: 30 runs x 17.6 h (the bottleneck host)
TOTAL_PAIR_STEPS = 50_000_000
N_RUNS = 60  # 6 seeds x (6 registered + 4 closed arms)
RUNS_PER_HOST = 30  # 3 seeds per host x 10 runs per seed
GTM_RUNS_PER_HOST = 6  # 3 seeds x (registered + closed)
N_EVAL_MODELS = 60
# KXKM-AI (RTX 4090), micro_batch_pairs 512 + clear_cache: measured
# 1,286-1,295 pair-steps/s (bench/kxkm/README.md, branch feat/n10c-grad-accum,
# PR #53); the amendment uses 1,290. Evaluation ~6.7 min per model on KXKM.
KXKM_PAIR_STEPS_PER_S = 1290.0
KXKM_EVAL_S_PER_MODEL = 402.0
PROCGEN_X86_WHEEL = "procgen-0.10.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
PROCGEN_X86_WHEEL_SHA256 = (
    "4b594d14e42f0f2166e59a9e294477b906eb99c90c3343b570b7124cbc865f53")
KXKM_VERSIONS = {  # venv310 on KXKM-AI, recorded 2026-10-23; re-verify at deploy
    "python": "3.10.18", "mlx": "0.31.1", "mlx-cuda-12": "0.31.1",
    "numpy": "1.26.4", "gym3": "0.3.3", "gym": "0.26.2", "procgen": "0.10.7",
}
def _fmt_settings(st: dict) -> dict[str, str]:
    mb = st["micro_batch_pairs"]
    return {"micro_batch_pairs": "none (full minibatch)" if mb is None else str(mb),
            "clear_cache": "on" if st["clear_cache"] else "off",
            "n_workers": str(st["n_workers"])}


# Display form of codes.HOST_SETTINGS (single source of truth).
# n_workers 8 on KXKM: +8 % end to end vs 1 (bench/kxkm/README.md, nworkers/)
HOST_TRAIN_SETTINGS = {h: _fmt_settings(s) for h, s in HOST_SETTINGS.items()}
# Commitment rule (amendment 2026-10-23, item 13): the manifest is committed
# by an annotated git tag pushed to GitHub, OSF is filed afterwards. DRAFT
# until then; the frozen text never contains the word DRAFT (campaign guard).
STATUS = ("DRAFT — not frozen (see freeze rule: committed by a frozen git tag "
          "before any run)")
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def status_text(freeze_tag: str | None = None) -> str:
    if freeze_tag is None:
        return STATUS
    if not _TAG_RE.match(freeze_tag):
        raise ValueError(f"invalid freeze tag {freeze_tag!r}")
    return f"FROZEN (tag {freeze_tag})"

RULINGS = [
    "R1: the harness venv (Python 3.10) has no scipy (uv.lock pins scipy 1.17, "
    "which needs Python >= 3.11); the analysis code runs in the repo environment.",
    "R2: GTM message over the 16-bit bus = 4-bit binary code (MSB first) of each "
    "of the 4 symbol indices (the bus carries and validates these 16 hard bits). "
    "In training the receiver gets the GTMBridge straight-through one-hot "
    "[K=4, A=16] directly (value identical to the hard symbols, gradient through "
    "the Gumbel-softmax); at evaluation and in the cross-play and shuffle probes "
    "it rebuilds the same one-hot from the 16 bits as prod_j (b_j if CODE_TABLE[a,j] "
    "= 1 else 1 - b_j), exact for hard bits (identical forward).",
    "R3: train_run takes keyword arguments log_every_pair_steps and "
    "ckpt_every_pair_steps (defaults 1,000,000).",
    "Task 4: the IMPALA trunk is duplicated between policy.py (live) and "
    "bench/bench_ppo_mlx.py (frozen archived evidence); never run together.",
    "Task 6: advantage normalisation is applied per minibatch (as in common "
    "Procgen PPO baselines); this is a PPO detail not in the registered list.",
    "Task 6: time_to_half requires a full window of 100 completed episodes.",
    "Task 6: the budget is exactly 50,000,000 pair-steps (final rollout truncated "
    "and updated on the partial rollout); the training success rate is measured "
    "at the exact boundaries k x 1M pair-steps; checkpoints are written after the "
    "first update ending at or after each 1M boundary (labelled with the actual "
    "step, < 8,192 pair-steps from the boundary) and once at the exact end.",
    "Task 6: the engineered right-minus-left counter counts only actions the "
    "environment applies (an agent that already finished is frozen; its actions "
    "are ignored).",
    "Task 6: GTM and DIAL message noise is stored per pair-step in the rollout and "
    "replayed in the PPO update (so the PPO ratio is 1 for an unchanged policy).",
    "Task 7: evaluation action sampling = one uniform per agent per step (agent 0 "
    "then agent 1) from default_rng([20261023, j]), inverse CDF on the float64 "
    "softmax; identical across passes and models. Evaluation episodes are capped "
    "at 5,000 steps (an error, not an outcome). Evaluation messages are "
    "deterministic (GTM argmax symbols).",
    "R4: gradient accumulation (micro_batch_pairs, clear_cache; PR #53) and "
    "n_workers (PR #54) are merged into the harness branch; the freeze commit "
    "must contain both. The KXKM-AI settings above refer to them.",
    "Task 9: smoke runs = 1 per arm in the registered condition plus gtm in the "
    "closed condition, 200,000 pair-steps each (7 runs), training levels only. "
    "The amendment allows up to 3 per arm of up to 1M steps; more may be added "
    "before the freeze within that cap.",
]


PROCGEN_EVIDENCE_DIR = "experiments/procgen_multiagent_gtm_comm/bench/procgen-builds"


def _procgen_evidence() -> dict[str, str]:
    d = REPO / PROCGEN_EVIDENCE_DIR
    files = sorted(p for p in d.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"no archived Procgen build evidence in {d}")
    return {p.name: sha256(p) for p in files}


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc.__class__.__name__})"


def _ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    env_commit = os.environ.get("N10C_COMMIT")
    head = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    ok = len(head) == 40 and all(c in "0123456789abcdef" for c in head)
    return {"commit": head if ok else (env_commit or "unknown"),
            "source": "git" if ok else ("N10C_COMMIT env" if env_commit else "none")}


def collect_build_record() -> dict:
    qt = _run(["brew", "list", "--versions", "qt@5"])
    return {
        "patch_file": str(PATCH.relative_to(REPO)),
        "patch_sha256": sha256(PATCH),
        "procgen_commit": PROCGEN_COMMIT,
        "procgen_package": _ver("procgen"),
        "qt_version_registered": QT_VERSION,
        "qt_brew_observed": qt,
        "clang": _run(["clang", "--version"]).splitlines()[0:2],
        "xcode": _run(["xcodebuild", "-version"]).splitlines(),
        "sdkroot": os.environ.get("SDKROOT", DEFAULT_SDKROOT),
        "sdkroot_source": "SDKROOT env" if os.environ.get("SDKROOT")
        else "documented build value (SDKROOT not set in this shell)",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx": _ver("mlx"),
        "mlx_runtime": getattr(mx, "__version__", "n/a"),
        "numpy": _ver("numpy"),
        "gym3": _ver("gym3"),
        "gym": _ver("gym"),
    }


def _pc(m) -> int:
    return param_count(m)


def arm_params() -> dict[str, dict]:
    out = {}
    for arm in ARM_CODES:
        model = PairPolicy(arm, mx.random.key(0))
        out[arm] = {
            "total": _pc(model),
            "comm": _pc(model.comm),
            "trunk": _pc(model.trunk),
            "to_msg": _pc(model.to_msg),
            "fuse_pi_v": _pc(model.fuse) + _pc(model.pi) + _pc(model.v),
        }
    return out


def compute_estimate(results: dict) -> dict:
    """Two-host estimate: 60 runs x 50M pair-steps, 30 per host.

    macM1 (seeds 0, 17, 42): 6 gtm runs at the measured gtm rate, 24 others
    at the measured nocomm rate (proxy), training only. KXKM-AI (seeds 73,
    101, 131): 30 runs at the KXKM rate, plus the evaluation of all 60
    models (macM1 checkpoints are copied to KXKM). The hosts run in
    parallel; the wall time is set by the slower host.
    """
    g, n = results["gtm"], results["nocomm"]

    def train_h(r: dict) -> float:
        return TOTAL_PAIR_STEPS / r["pair_steps_per_s"] / 3600

    hg, hn = train_h(g), train_h(n)
    mac = GTM_RUNS_PER_HOST * hg + (RUNS_PER_HOST - GTM_RUNS_PER_HOST) * hn
    kx_run = TOTAL_PAIR_STEPS / KXKM_PAIR_STEPS_PER_S / 3600
    kx_train = RUNS_PER_HOST * kx_run
    kx_eval = N_EVAL_MODELS * KXKM_EVAL_S_PER_MODEL / 3600
    wall = max(mac, kx_train + kx_eval)
    return {
        "hours_per_run_gtm": hg, "hours_per_run_nocomm": hn,
        "macm1_hours": mac, "kxkm_hours_per_run": kx_run,
        "kxkm_train_hours": kx_train, "kxkm_eval_hours": kx_eval,
        "wall_hours": wall, "wall_days": wall / 24,
        "bottleneck_host": "macm1" if mac >= kx_train + kx_eval else "kxkm",
        "total_hours": mac + kx_train + kx_eval,
        "planning_h_per_run": PLANNING_H_PER_RUN,
        "planning_total_hours": PLANNING_TOTAL_H,
        "exceeds_planning": mac > PLANNING_TOTAL_H,
    }


def _check_levels(thr: dict) -> None:
    for arm, r in thr.get("results", {}).items():
        lv = r.get("eval_proxy", {}).get("levels")
        if lv and lv[1] > 1000:  # [start, stop): levels >= 1000 held out
            raise ValueError(f"{arm}: evaluation proxy must use training levels only")


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _fmt_throughput(thr: dict | None) -> list[str]:
    if not thr:
        return ["**NOT MEASURED** (throughput.json absent).", ""]
    lines = [f"Measured {thr.get('measured_on', '?')}; one job at a time, real "
             "Procgen CoinRun-coop (64 pairs, 128-step rollouts), policy, message "
             "bus and PPO update, untrained start, training levels only. Steady "
             "state discards the first 2 update cycles (mx.compile warm-up).", "",
             "| arm | pair-steps/s (steady) | agent-samples/s | s per update cycle "
             "| pair-steps/s incl. warm-up | eval s/model (proxy 2x200) |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, r in thr["results"].items():
        lines.append(
            f"| {arm} | {r['pair_steps_per_s']:.1f} | {r['agent_samples_per_s']:.1f} "
            f"| {r['steady_s_per_update_cycle']:.2f} "
            f"| {r['pair_steps_per_s_incl_warmup']:.1f} "
            f"| {r['eval_proxy']['seconds_per_model_2x200']:.0f} |")
    lines += ["", "Host load (`uptime`) recorded in throughput.json (macM1 is shared "
              "with the second-brain VM under OrbStack, so contention is possible):"]
    for arm, r in thr["results"].items():
        lines.append(f"- {arm}: before `{r.get('uptime_before')}`; after "
                     f"`{r.get('uptime_after')}`")
    lines += ["", "Evaluation cost is a **proxy**: an untrained policy, intact + "
              "shuffled passes over the 16 training levels (levels 0-15), scaled "
              "linearly to 2 x 200 episodes. Trained policies finish episodes "
              "earlier, and the 200 evaluation levels are different levels, so the "
              "real cost differs. Concurrent jobs were not measured.", ""]
    return lines


def _fmt_estimate(thr: dict | None) -> list[str]:
    if not thr or not {"gtm", "nocomm"} <= set(thr.get("results", {})):
        return ["**Compute estimate not available** (needs gtm and nocomm "
                "throughput).", ""]
    e = compute_estimate(thr["results"])
    lines = [
        f"Option D on two hosts: {N_RUNS} runs (36 registered + 24 closed), "
        f"each {TOTAL_PAIR_STEPS:,} pair-steps, {RUNS_PER_HOST} per host. macM1 "
        "uses the gtm rate for its 6 gtm runs and the nocomm rate as a proxy for "
        "its 24 other runs (mlp, atoc, dial, eng were not throughput-measured). "
        f"KXKM-AI uses {KXKM_PAIR_STEPS_PER_S:.0f} pair-steps/s (micro_batch_pairs "
        "= 512 + clear_cache; bench/kxkm/README.md, PR #53). ALL evaluations run on "
        "KXKM-AI (macM1 checkpoints are copied there), about "
        f"{KXKM_EVAL_S_PER_MODEL / 60:.1f} min per model.", "",
        f"- macM1: {RUNS_PER_HOST} runs, gtm {e['hours_per_run_gtm']:.1f} h and "
        f"others (nocomm proxy) {e['hours_per_run_nocomm']:.1f} h per run, total "
        f"**{e['macm1_hours']:.0f} h = {e['macm1_hours'] / 24:.1f} days**",
        f"- KXKM-AI: {RUNS_PER_HOST} runs x {e['kxkm_hours_per_run']:.1f} h = "
        f"**{e['kxkm_train_hours']:.0f} h** training, plus "
        f"{e['kxkm_eval_hours']:.1f} h for the {N_EVAL_MODELS} evaluations",
        f"- hosts run in parallel; wall time **{e['wall_hours']:.0f} h = "
        f"{e['wall_days']:.1f} days**, set by {e['bottleneck_host']} (the "
        "bottleneck)",
        f"- amendment planning figure: macM1 about {PLANNING_H_PER_RUN:.1f} h per "
        f"run, about {PLANNING_TOTAL_H:.0f} h (22 days); KXKM about 324 h", ""]
    if e["exceeds_planning"]:
        lines += [f"**The measured macM1 estimate EXCEEDS the amendment's planning "
                  f"figure ({e['macm1_hours']:.0f} h versus about "
                  f"{PLANNING_TOTAL_H:.0f} h).** The amendment text has not been "
                  "changed; whether to revise it or accept the longer run is the "
                  "PI's decision.", ""]
    else:
        lines += ["The measured macM1 estimate is within the amendment's planning "
                  "figure.", ""]
    return lines


def _fmt_smoke(sm: dict | None) -> list[str]:
    if not sm:
        return ["**No smoke runs recorded** (smoke_runs.json absent).", ""]
    lines = [f"{len(sm['runs'])} smoke runs of {sm.get('pair_steps_each'):,} "
             f"pair-steps each (cap: 3 per arm, 1M steps), training levels only "
             f"(start_level 0, num_levels 16), smoke seed {sm.get('smoke_seed')} "
             "(not a registered seed), 64 pairs, 128-step rollouts. Their results "
             "informed no registered quantity; outcomes are listed unfiltered.", "",
             "| arm | condition | outcome | pair-steps done | wall s | pair-steps/s "
             "(incl. warm-up) | episodes | success (last 100) | losses finite |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in sm["runs"]:
        sr = r.get("final_success_rate_last100")
        lines.append(
            f"| {r['arm']} | {r['condition']} | {r['outcome']} "
            f"| {r.get('pair_steps_done', '-')} | {r.get('wall_time_s', 0):.0f} "
            f"| {r.get('pair_steps_per_s_incl_warmup', 0):.0f} "
            f"| {r.get('completed_episodes', '-')} "
            f"| {'-' if sr is None else f'{sr:.2f}'} "
            f"| {r.get('all_losses_finite', '-')} |")
    for r in sm["runs"]:
        if r["outcome"] != "completed":
            lines += ["", f"Failure detail ({r['arm']}/{r['condition']}): "
                      f"{r.get('failure_reason') or r.get('error')}"]
    lines.append("")
    return lines


def write_manifest(out: Path, freeze_tag: str | None = None) -> None:
    status = status_text(freeze_tag)
    out = Path(out)
    thr = _load(out.parent / "throughput.json")
    sm = _load(out.parent / "smoke_runs.json")
    if thr:
        _check_levels(thr)
    git = git_state()
    build = collect_build_record()
    params = arm_params()
    cfg = GTMConfigMLX()
    files = {p.relative_to(REPO).as_posix(): sha256(p) for p in
             [PKG / "coop_env.py", *sorted((PKG / "harness").glob("*.py")),
              PKG / "harness" / "requirements.lock"]}

    lines: list[str] = [
        "# N10-C stage-2 configuration manifest", "",
        f"**Status: {status}.**", "",
        "Registration: N10-C amendment of 2026-10-23, configuration manifest "
        "(stage 2). Immutable once frozen. Runs started before the freeze tag are "
        "excluded from all analyses and reported.", "",
        "## 1. Commit, integer codes and bus", "",
        f"- Harness commit at manifest time: `{git['commit']}` (source: "
        f"{git['source']}). **The freeze commit must be re-recorded after "
        "merge**: this draft is committed in a later commit than the code it "
        "describes.",
        "- Code identity independent of git (SHA-256 of each file at manifest time):",
        "", "| file | SHA-256 |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in files.items()]
    lines += ["", "- Files covered: `coop_env.py` (environment), the MLX PPO-MA harness "
          "(`harness/`: vec_env, comm, policy, rollout, train, evaluate) and the "
          "message bus (`harness/bus.py`).",
          f"- `ARM_CODES = {ARM_CODES}`",
          f"- `CONDITION_CODES = {CONDITION_CODES}`",
          "- Run randomness: `numpy.random.SeedSequence([seed, arm_code, cond_code])`, "
          "seeds {0, 17, 42, 73, 101, 131} (the five registered seeds and 131, "
          "fixed by the amendment's outcome-blind rule), 6 seeds x 10 arm-condition "
          "cells = 60 runs. Seed-to-host mapping: macM1: seeds {0, 17, 42}; KXKM-AI: "
          "seeds {73, 101, 131} (host fixed per seed; see section 4).",
          "- Bus: all arms send m in {0,1}^16 per agent per step; shape and binary "
          "values asserted at evaluation for all arms and in training for all arms "
          "except DIAL (soft DRU outputs in training).", "",
          "## 2. Network, PPO and GTM-adapter sizes (as implemented)", "",
          "Per agent (parameters shared across the two agents of a pair): IMPALA "
          "CNN (16, 32, 32; 64x64x3 input scaled by 1/255) -> 256-d h; "
          "Linear(256 -> 64) -> comm.send (16 bits); partner swap; "
          "comm.receive(msg, h) -> 64; concat [h 256, recv 64, team summary 2] = 322 "
          "-> Linear(322 -> 256) -> ReLU -> policy Linear(256 -> 15) and value "
          "Linear(256 -> 1). Inputs: own image, received message, team summary "
          "(registered condition only; zeros in the closed condition). Never "
          "rewards, done flags or partner actions.", "",
          "| arm | total params | comm params | IMPALA trunk | to_msg 256->64 "
          "| fuse+pi+v |", "|---|---:|---:|---:|---:|---:|"]
    for arm, p in params.items():
        lines.append(f"| {arm} | {p['total']:,} | {p['comm']:,} | {p['trunk']:,} | "
                 f"{p['to_msg']:,} | {p['fuse_pi_v']:,} |")
    lines += ["", "Comm modules (encoder/decoder sizes):",
          "- mlp: encode Linear(64 -> 16), hard bit = (m > 0), straight-through "
          "(sigmoid) in training; decode Linear(16 -> 64).",
          "- atoc: same sender; receiver cross-attends over the 16 received bit "
          "tokens: bit embedding 2 x 32, positional embedding 16 x 32, query "
          "Linear(256 -> 32), output Linear(32 -> 64).",
          "- dial: encode Linear(64 -> 16); training message sigmoid(m + 2 * "
          "N(0, 1)) (noise sigma 2), evaluation hard threshold at 0; decode "
          "Linear(16 -> 64).",
          "- eng: no sender parameters; engineered 16-bit message (in-window flag, "
          "5-bit remaining steps, 10-bit clipped right-minus-left count); decode "
          "Linear(16 -> 64).",
          "- nocomm: zero 16-bit message; decode Linear(16 -> 64) (floor).",
          f"- gtm: see the adapter below; decode Linear({comm.MSG_DIM} + 2 x "
          f"{comm.K_SYMBOLS} -> {comm.MSG_DIM}).", "",
          "### GTM symbol adapter", "",
          f"- K = {comm.K_SYMBOLS} symbols per message from an alphabet A = "
          f"{comm.ALPHABET}; sender head Linear(64 -> {comm.K_SYMBOLS * comm.ALPHABET})"
          " reshaped to [4, 16] logits.",
          "- Training: Gumbel-softmax sample, tau = 1.0, hard one-hot with "
          "straight-through gradient; evaluation: argmax symbols (deterministic).",
          "- Bus mapping (ruling R2): each symbol index -> 4-bit binary code, MSB "
          "first, giving 4 x 4 = 16 bits (validated on the bus).",
          f"- Receiver: takes the straight-through one-hot directly in training "
          f"(rebuilt from the 16 bits at evaluation), multiplexes the carrier "
          f"with the GTM (gamma {cfg.gamma_hz} Hz, theta {cfg.theta_hz} Hz, sample "
          f"rate {cfg.sample_rate_hz} Hz, {cfg.symbols_per_theta} symbols per "
          "theta), adaptive mean-pool of the carrier to 64 samples (pool 64), "
          "concatenated with the 8 IQ coordinates of the 4 symbols, then "
          "Linear(72 -> 64).",
          "- Constellation: 16-PSK initialisation + 0.01 N(0, 1), trainable; "
          "cosine plasticity schedule 1.0 -> 0.1 over the run's total updates "
          "(multiplies the constellation gradient).", "",
          "### PPO and training settings", "",
          f"- lr {train.LR:g} (lr 5e-4), clip {train.CLIP}, GAE lambda "
          f"{train.LAMBDA}, gamma {train.GAMMA}, Adam, value coefficient "
          f"{train.VALUE_COEF}, entropy coefficient {train.ENTROPY_COEF}, max "
          f"gradient norm {train.MAX_GRAD_NORM}.",
          f"- {train.EPOCHS} epochs per update, minibatch {train.MINIBATCH_PAIRS} "
          "pair-steps (= 2,048 agent-samples; on KXKM-AI computed by gradient "
          "accumulation over micro-batches, same update), per-minibatch advantage "
          "normalisation, 64 parallel coop envs, 128-step rollouts (16,384 "
          "agent-samples per update).",
          f"- Budget {TOTAL_PAIR_STEPS:,} pair-steps per run (about 6,104 updates); "
          "training levels start_level 0, num_levels 16; distribution_mode easy; "
          "Delta t = 20. Success measured every 1M pair-steps as the mean over the "
          "last 100 completed episodes.",
          "- Non-finite loss or parameter: the run stops and the last finite "
          "checkpoint is kept (registered failure rule).", "",
          "## 3. Scope", "",
          "Caveflyer-coop and Maze-coop are **not run** under this registration "
          "(scope option D). N10-C makes no statement about them. 60 runs on two hosts "
          "(30 per host), see section 4.", "",
          "## 4. Measured throughput, compute estimate and Procgen build record", "",
          "### Measured training throughput (macM1)", ""]
    lines += _fmt_throughput(thr)
    lines += ["### Compute estimate including evaluation", ""]
    lines += _fmt_estimate(thr)
    lines += ["### Procgen arm64 build record", "",
          f"- Build patch: `{build['patch_file']}`, **SHA-256 "
          f"`{build['patch_sha256']}`**.",
          f"- Procgen 0.10.7, source commit `{build['procgen_commit']}`; installed "
          f"package version: {build['procgen_package']}.",
          f"- Qt {QT_VERSION} (Qt 5.15.19 via Homebrew `qt@5`, found through "
          f"`PROCGEN_CMAKE_PREFIX_PATH`); observed: `{build['qt_brew_observed']}`.",
          f"- clang: `{' | '.join(build['clang'])}`",
          f"- Xcode: `{' | '.join(build['xcode'])}`",
          f"- SDKROOT: `{build['sdkroot']}` ({build['sdkroot_source']}); "
          "`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`; "
          "`CMAKE_POLICY_VERSION_MINIMUM=3.5`.",
          f"- Python {build['python']} ({build['platform']}); MLX "
          f"{build['mlx']} (runtime {build['mlx_runtime']}); numpy "
          f"{build['numpy']}; gym3 {build['gym3']}; gym {build['gym']}.",
          "- The patch changes the build configuration only, not game code. This "
          "build is used for the macM1 training runs only (seeds 0, 17, 42).", "",
          "### Hosts, seed-to-host mapping and KXKM-AI record", "",
          "Host is a block factor fixed per seed; all arms and both conditions of a "
          "seed run on that seed's host.", ""]
    for host, label, hw in (
            ("macm1", "macM1", "Apple M1 Max, 32 GB, MLX Metal"),
            ("kxkm", "KXKM-AI", "RTX 4090 24 GB, MLX CUDA 12")):
        sd = [s for s in SEEDS if SEED_HOST[s] == host]
        st = HOST_TRAIN_SETTINGS[host]
        lines.append(
            "- " + label + ": seeds {" + ", ".join(map(str, sd)) + "} (" + hw
            + "); micro_batch_pairs = " + st["micro_batch_pairs"]
            + ", clear_cache = " + st["clear_cache"] + ", n_workers = "
            + st["n_workers"] + " (registered per-host implementation setting, "
            "`harness/codes.py` HOST_SETTINGS, applied by run_campaign to every run of "
            "this host and recorded in each run.json"
            + ("; n_workers 8 measured +8 % end to end vs 1, bench/kxkm/README.md)."
               if host == "kxkm" else "; n_workers 1 is the default, not measured)."))
    kv = KXKM_VERSIONS
    lines += [
          "- KXKM-AI software (venv310, recorded 2026-10-23, re-verify at deploy): "
          + ", ".join(f"{k} {v}" for k, v in kv.items())
          + "; `CUDA_HOME` must point at the CUDA headers for the JIT kernels. "
          "Version pins: mlx 0.31.1 and mlx-cuda-12 0.31.1 on KXKM-AI; MLX 0.31.1 "
          "(Metal) on macM1.",
          f"- KXKM-AI Procgen build: the official Procgen 0.10.7 x86-64 wheel "
          f"`{PROCGEN_X86_WHEEL}`, PyPI SHA-256 `{PROCGEN_X86_WHEEL_SHA256}`.",
          "- ALL evaluations (200 held-out levels, intact and shuffled passes, "
          "cross-seed probe) run on KXKM-AI with the x86-64 build; macM1 "
          "checkpoints are copied there. **The two Procgen builds are NOT "
          "bit-identical**: on 20 of 20 CoinRun levels tested the frame / reward "
          "streams differ between macM1 (arm64) and KXKM-AI (x86-64), although "
          "each machine is deterministic across reruns; frames agree early in an "
          "episode on some levels and diverge late. Evidence, procedure and "
          "result archived in "
          f"`{PROCGEN_EVIDENCE_DIR}/README.md`; SHA-256 of the archived "
          "files:", "",
          "| file | SHA-256 |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in _procgen_evidence().items()]
    lines += ["", "macM1-trained models are therefore evaluated "
          "on the x86 rendering.", "",
          "### Reproducibility (nondeterminism)", "",
          "On both Metal and CUDA, MLX 0.31.1 convolution backward in the IMPALA "
          "trunk is nondeterministic (Metal about 1e-7 relative, CUDA about 1.4e-5 "
          "to 4.5e-5 relative per gradient; initialisation, forward pass and the "
          "first rollout are bit-identical; no deterministic switch exists). Runs "
          "are statistically, not bitwise, reproducible from the seed; an "
          "infrastructure-failure restart with the same seed stream will not "
          "retrace the same trajectory. Evidence: "
          "`experiments/procgen_multiagent_gtm_comm/bench/kxkm/repeat/` and "
          "`bench/kxkm/README.md` (branch feat/n10c-grad-accum, PR #53).", "",
          "## Smoke runs and development disclosure", ""]
    lines += _fmt_smoke(sm)
    lines += ["Development used only the stub (unit tests with scripted environments), "
          "these smoke runs and throughput measurements on untrained policies. "
          "Levels 1000-1199 were never touched and no full-length run result was "
          "used.", "",
          "## Artefact archiving (PI decision 2026-10-23)", "",
          "After each run and each evaluation, the run directory (`run.json`, "
          "checkpoints, EvalRecord JSON, cross-play records) is stored on electron-server "
          "(Tailscale 100.78.191.52) at `electron@100.78.191.52:/home/electron/archives/n10c/`. "
          "KXKM-AI cannot reach electron-server, so electron-server PULLS the completed "
          "runs from KXKM-AI (`campaign/pull_archive.sh`, rsync over ssh to "
          "`electron@10.2.0.237:/ai/saisail/n10c-run/campaign`, completed runs only, "
          "idempotent, logged); macM1 can push with `--archive-to`. The copy is post-hoc "
          "storage only and changes no registered quantity. macM1 checkpoints are "
          "additionally copied to KXKM-AI for evaluation (campaign `sync` step), and "
          "are archived from that KXKM copy. (This item is outside items 1-4 of the "
          "amendment's manifest list.)", "",
          "## Implementation rulings disclosed", ""]
    lines += [f"- {r}" for r in RULINGS]
    lines += [""]
    text = "\n".join(lines)
    if freeze_tag is not None and "DRAFT" in text:
        raise ValueError("a frozen manifest must not contain the word DRAFT")
    out.write_text(text)


def main() -> None:
    p = argparse.ArgumentParser(description="Write the N10-C stage-2 manifest draft")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--freeze-tag", default=None,
                   help="write 'Status: FROZEN (tag <tag>)' instead of DRAFT; "
                        "use a file name without 'DRAFT'")
    a = p.parse_args()
    write_manifest(a.out, freeze_tag=a.freeze_tag)


if __name__ == "__main__":
    main()
