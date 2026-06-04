"""Multi-seed comparison: none / rehearsal / ewc on HardSplitTask (slow)."""
import statistics

import pytest

from scripts.track_w_pilot import run_w4_compare
from track_w.tasks.hard_split import HardSplitTask

SEEDS = list(range(5))
LAM   = 1.0  # EWC strength — swept and documented in research JSON


@pytest.mark.slow
def test_w4_all_methods_multi_seed():
    """n=5 seeds: rehearsal and ewc both strictly beat none on mean forgetting.

    Results traced to docs/superpowers/research/2026-05-30-w4-ewc-comparison.json
    keys "multi_seed_none_mean", "multi_seed_rehearsal_mean", "multi_seed_ewc_mean".
    """
    forgetting: dict[str, list[float]] = {"none": [], "rehearsal": [], "ewc": []}
    for seed in SEEDS:
        task = HardSplitTask(seed=seed)
        for method in ("none", "rehearsal", "ewc"):
            kw = {"lam": LAM} if method == "ewc" else {}
            r = run_w4_compare(method=method, task=task, steps=400, seed=seed, **kw)
            forgetting[method].append(r["forgetting"])

    mean_none      = statistics.mean(forgetting["none"])
    mean_rehearsal = statistics.mean(forgetting["rehearsal"])
    mean_ewc       = statistics.mean(forgetting["ewc"])

    assert mean_rehearsal < mean_none, (
        f"rehearsal mean forgetting ({mean_rehearsal:.3f}) did not beat "
        f"none ({mean_none:.3f})"
    )
    assert mean_ewc < mean_none, (
        f"ewc mean forgetting ({mean_ewc:.3f}) did not beat "
        f"none ({mean_none:.3f})"
    )
    # Which method wins: print for tracing, no assertion (honest reporting).
    winner = "rehearsal" if mean_rehearsal <= mean_ewc else "ewc"
    print(
        f"\n[multi-seed n={len(SEEDS)}] "
        f"none={mean_none:.3f}  rehearsal={mean_rehearsal:.3f}  "
        f"ewc={mean_ewc:.3f}  winner={winner}"
    )
