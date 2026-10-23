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
import subprocess
from pathlib import Path

import mlx.core as mx

from . import comm, train
from .codes import ARM_CODES, CONDITION_CODES
from .gtm_mlx import GTMConfigMLX
from .policy import PairPolicy, param_count

PKG = Path(__file__).resolve().parent.parent  # experiments/procgen_multiagent_gtm_comm
REPO = PKG.parent.parent
PATCH = PKG / "bench" / "procgen-0.10.7-arm64-build.patch"
PROCGEN_COMMIT = "5e1dbf3"
QT_VERSION = "5.15.19"
DEFAULT_SDKROOT = ("/Applications/Xcode.app/Contents/Developer/Platforms/"
                   "MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk")
PLANNING_H_PER_RUN = 16.0
PLANNING_TOTAL_H = 800.0
TOTAL_PAIR_STEPS = 50_000_000
N_GTM_RUNS, N_OTHER_RUNS = 10, 40  # gtm: 5 registered + 5 closed; others
STATUS = "DRAFT — to be filed on OSF by the PI as stage 2 before any run"

RULINGS = [
    "R1: the harness venv (Python 3.10) has no scipy (uv.lock pins scipy 1.17, "
    "which needs Python >= 3.11); the analysis code runs in the repo environment.",
    "R2: GTM message over the 16-bit bus = 4-bit binary code (MSB first) of each "
    "of the 4 symbol indices; in training the straight-through one-hot is mapped "
    "to bits by a fixed table (value = hard bits, gradient through the softmax), "
    "and the receiver rebuilds a differentiable one-hot per symbol as "
    "prod_j (b_j if CODE_TABLE[a,j] = 1 else 1 - b_j), exact for hard bits. This "
    "gradient path differs from the GTMBridge soft-IQ path of the earlier code.",
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
    "Task 9: smoke runs = 1 per arm in the registered condition plus gtm in the "
    "closed condition, 200,000 pair-steps each (7 runs), training levels only. "
    "The amendment allows up to 3 per arm of up to 1M steps; more may be added "
    "before filing within that cap.",
]


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
    """Option D estimate: 50 runs x 50M pair-steps + one evaluation each.

    gtm runs (10) use the measured gtm rate; the other 40 runs use the
    measured nocomm rate as the proxy (mlp/atoc/dial/eng are not
    throughput-measured; their smoke-run rates are listed separately).
    """
    g, n = results["gtm"], results["nocomm"]

    def per_run(r: dict) -> float:
        train_h = TOTAL_PAIR_STEPS / r["pair_steps_per_s"] / 3600
        return train_h + r["eval_proxy"]["seconds_per_model_2x200"] / 3600

    hg, hn = per_run(g), per_run(n)
    total = N_GTM_RUNS * hg + N_OTHER_RUNS * hn
    worst = (N_GTM_RUNS + N_OTHER_RUNS) * max(hg, hn)
    return {
        "hours_per_run_gtm": hg, "hours_per_run_nocomm": hn,
        "total_hours": total, "total_days": total / 24,
        "conservative_total_hours_all_runs_at_slowest": worst,
        "planning_h_per_run": PLANNING_H_PER_RUN,
        "planning_total_hours": PLANNING_TOTAL_H,
        "exceeds_planning": total > PLANNING_TOTAL_H,
    }


def _check_levels(thr: dict) -> None:
    for arm, r in thr.get("results", {}).items():
        lv = r.get("eval_proxy", {}).get("levels")
        if lv and max(lv) > 1000:
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
        "Option D: 50 runs (10 gtm: 5 registered + 5 closed; 40 others), each "
        f"{TOTAL_PAIR_STEPS:,} pair-steps, plus one evaluation per trained model. "
        "The gtm rate is used for gtm runs; the nocomm rate is used as a proxy for "
        "the other 40 runs (mlp, atoc, dial, eng were not throughput-measured; see "
        "the smoke-run rates).", "",
        f"- per run: gtm {e['hours_per_run_gtm']:.1f} h, others (nocomm proxy) "
        f"{e['hours_per_run_nocomm']:.1f} h",
        f"- total: **{e['total_hours']:.0f} h = {e['total_days']:.1f} days** of "
        "continuous single-job compute",
        f"- if all 50 runs ran at the slower measured rate: "
        f"{e['conservative_total_hours_all_runs_at_slowest']:.0f} h",
        f"- amendment planning figure: about {PLANNING_H_PER_RUN:.0f} h per run, "
        f"about {PLANNING_TOTAL_H:.0f} h (33 days)", ""]
    if e["exceeds_planning"]:
        lines += [f"**The measured estimate EXCEEDS the amendment's planning figure "
                  f"({e['total_hours']:.0f} h versus about {PLANNING_TOTAL_H:.0f} h, "
                  f"{e['total_hours'] / PLANNING_TOTAL_H:.2f} times).** The amendment "
                  "text has not been changed; whether to revise it, reduce scope, or "
                  "accept the longer run is the PI's decision.", ""]
    else:
        lines += ["The measured estimate is within the amendment's planning figure.",
                  ""]
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


def write_manifest(out: Path) -> None:
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

    L: list[str] = [
        "# N10-C stage-2 configuration manifest", "",
        f"**Status: {STATUS}.**", "",
        "Registration: N10-C amendment of 2026-10-23, configuration manifest "
        "(stage 2). Immutable once filed. Runs started before filing are excluded "
        "from all analyses and reported.", "",
        "## 1. Commit, integer codes and bus", "",
        f"- Harness commit at manifest time: `{git['commit']}` (source: "
        f"{git['source']}). **The final filing commit must be re-recorded after "
        "merge**: this draft is committed in a later commit than the code it "
        "describes.",
        "- Code identity independent of git (SHA-256 of each file at manifest time):",
        "", "| file | SHA-256 |", "|---|---|"]
    L += [f"| `{k}` | `{v}` |" for k, v in files.items()]
    L += ["", "- Files covered: `coop_env.py` (environment), the MLX PPO-MA harness "
          "(`harness/`: vec_env, comm, policy, rollout, train, evaluate) and the "
          "message bus (`harness/bus.py`).",
          f"- `ARM_CODES = {ARM_CODES}`",
          f"- `CONDITION_CODES = {CONDITION_CODES}`",
          "- Run randomness: `numpy.random.SeedSequence([seed, arm_code, cond_code])`, "
          "seeds {0, 17, 42, 73, 101}.",
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
        L.append(f"| {arm} | {p['total']:,} | {p['comm']:,} | {p['trunk']:,} | "
                 f"{p['to_msg']:,} | {p['fuse_pi_v']:,} |")
    L += ["", "Comm modules (encoder/decoder sizes):",
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
          "first, giving 4 x 4 = 16 bits.",
          f"- Receiver: rebuilds the one-hot from bits, multiplexes the carrier "
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
          "pair-steps (= 2,048 agent-samples), per-minibatch advantage "
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
          "(scope option D). N10-C makes no statement about them.", "",
          "## 4. Measured throughput, compute estimate and Procgen build record", "",
          "### Measured training throughput (macM1)", ""]
    L += _fmt_throughput(thr)
    L += ["### Compute estimate including evaluation", ""]
    L += _fmt_estimate(thr)
    L += ["### Procgen arm64 build record", "",
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
          "- The patch changes the build configuration only, not game code. "
          "Equivalence with the official x86-64 builds has not been checked. All "
          "N10-C runs use this single build.", "",
          "## Smoke runs and development disclosure", ""]
    L += _fmt_smoke(sm)
    L += ["Development used only the stub (unit tests with scripted environments), "
          "these smoke runs and throughput measurements on untrained policies. "
          "Levels 1000-1199 were never touched and no full-length run result was "
          "used.", "",
          "## Artefact archiving (PI decision 2026-10-23)", "",
          "After each run and each evaluation, the run directory (`run.json`, "
          "checkpoints, EvalRecord JSON) is copied to electron-server (Tailscale "
          "100.78.191.52) for storage. The copy is post-hoc storage only and "
          "changes no registered quantity. Destination path: **to be filled by "
          "the PI before filing**: `<PATH>`. (This item is outside items 1-4 of the "
          "amendment's manifest list; flagged for the PI.)", "",
          "## Implementation rulings disclosed", ""]
    L += [f"- {r}" for r in RULINGS]
    L += [""]
    out.write_text("\n".join(L))


def main() -> None:
    p = argparse.ArgumentParser(description="Write the N10-C stage-2 manifest draft")
    p.add_argument("--out", type=Path, required=True)
    write_manifest(p.parse_args().out)


if __name__ == "__main__":
    main()
