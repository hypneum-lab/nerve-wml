"""Trainer tests: scripted envs, tiny budgets (n_pairs 2, horizon 4)."""
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.harness.codes import (  # noqa: E402
    run_seed_sequence,
)
from experiments.procgen_multiagent_gtm_comm.harness.train import (  # noqa: E402
    EngCounters,
    _assert_not_stub,
    train_run,
)
from experiments.procgen_multiagent_gtm_comm.harness.vec_env import (  # noqa: E402
    VecCoopEnv,
)

NP, H = 2, 4
PER_UPDATE = NP * H


class _Scripted:
    def __init__(self, rand_seed: int, coin_steps: set[int]) -> None:
        self.coin_steps = coin_steps
        self.t = -1

    def act(self, actions) -> None:
        self.t += 1

    def observe(self):
        coin = self.t in self.coin_steps
        img = np.full((1, 64, 64, 3), (self.t + 1) % 256, dtype=np.uint8)
        return np.array([10.0 if coin else 0.0]), {"rgb": img}, np.array([coin])

    def close(self) -> None:
        pass


def _factory():
    created: list[_Scripted] = []

    def f(rand_seed: int) -> _Scripted:
        env = _Scripted(rand_seed, {0} if len(created) % 2 == 0 else {1})
        created.append(env)
        return env

    return f


def _run(tmp_path, updates, *, arm="mlp", seed=0, log=PER_UPDATE,
         ckpt=PER_UPDATE, hook=None, name="r", total=None, rhook=None):
    return train_run(
        arm, seed, "registered", total or updates * PER_UPDATE, tmp_path / name,
        env_factory=_factory(), n_pairs=NP, horizon=H,
        log_every_pair_steps=log, ckpt_every_pair_steps=ckpt, _fault_hook=hook,
        _rollout_hook=rhook,
    )


def test_train_run_writes_record_and_checkpoint(tmp_path):
    rec = _run(tmp_path, 2)
    d = tmp_path / "r"
    assert (d / "ckpt_8.safetensors").exists()
    assert (d / "ckpt_16.safetensors").exists()
    data = json.loads((d / "run.json").read_text())
    assert data["arm"] == "mlp" and data["seed"] == 0
    assert data["condition"] == "registered"
    assert data["codes"] == {"arm": 1, "condition": 0}
    assert data["total_pair_steps"] == 16 and data["pair_steps_done"] == 16
    assert data["using_stub"] is False and data["training_failure"] is False
    assert data["restarts"] == 0 and data["bits_logged"] > 0
    assert data["samples_per_s"] > 0 and data["wall_time_s"] > 0
    assert rec.bits_logged == 16 * 2 * 16  # 16 pair-steps * 2 agents * 16 bits


def test_success_curve_logged_every_interval(tmp_path):
    rec = _run(tmp_path, 6, log=2 * PER_UPDATE)
    assert [s for s, _ in rec.success_curve] == [16, 32, 48]
    assert all(r == 1.0 for _, r in rec.success_curve)  # scripted: all succeed
    assert rec.time_to_half is None  # window of 100 episodes never full


def test_nan_stops_at_last_finite_checkpoint(tmp_path):
    def hook(update, model):
        if update == 1:
            model.pi.weight = mx.full(model.pi.weight.shape, float("nan"))

    rec = _run(tmp_path, 4, hook=hook)
    assert rec.training_failure is True
    assert rec.n_updates == 2
    # failure detected before applying: only update 1's 3 steps happened
    assert rec.optimizer_steps == 3
    assert rec.last_checkpoint == "ckpt_8.safetensors"
    assert not (tmp_path / "r" / "ckpt_16.safetensors").exists()
    data = json.loads((tmp_path / "r" / "run.json").read_text())
    assert data["training_failure"] is True


@pytest.mark.parametrize("arm", ["gtm", "dial"])
def test_same_seed_same_first_update(tmp_path, arm):
    acts = {}

    def grab(name):
        return lambda u, buf: acts.setdefault(name, buf.actions.copy())

    a = _run(tmp_path, 1, arm=arm, seed=3, name="a", rhook=grab("a"))
    b = _run(tmp_path, 1, arm=arm, seed=3, name="b", rhook=grab("b"))
    c = _run(tmp_path, 1, arm=arm, seed=4, name="c", rhook=grab("c"))
    assert np.array_equal(acts["a"], acts["b"])  # rollout is deterministic
    assert not np.array_equal(acts["a"], acts["c"])
    # Metal reductions are not bit-exact (~1e-7 relative).
    assert a.losses[0] == pytest.approx(b.losses[0], rel=1e-5)
    assert abs(a.losses[0] - c.losses[0]) > 1e-3
    pa = mx.load(str(tmp_path / "a" / "ckpt_8.safetensors"))
    pb = mx.load(str(tmp_path / "b" / "ckpt_8.safetensors"))
    # Adam's step is ~lr*sign(g): a ~1e-7 gradient difference can flip the
    # sign only of near-zero gradients, so a parameter differs by at most
    # 2*lr per optimizer step (3 steps) and almost all entries agree.
    for k in ("pi.weight", "v.weight", "fuse.bias"):
        d = np.abs(np.array(pa[k]) - np.array(pb[k]))
        assert d.max() <= 3 * 2 * 5e-4 + 1e-6
        assert (d <= 1e-5).mean() > 0.95


@pytest.mark.parametrize("arm", ["gtm", "dial", "mlp", "eng"])
def test_unchanged_policy_ratio_is_one(tmp_path, arm):
    rec = _run(tmp_path, 2, arm=arm)
    assert len(rec.first_mb_ratio_dev) == 2
    assert max(rec.first_mb_ratio_dev) < 1e-4


def test_inexact_budget_and_boundaries(tmp_path):
    # 22 pair-steps = 11 env steps: rollouts of 4, 4, 3 env steps.
    rec = _run(tmp_path, 0, total=22, log=6, ckpt=10)
    assert rec.pair_steps_done == 22 and rec.n_updates == 3
    assert [s for s, _ in rec.success_curve] == [6, 12, 18]
    names = sorted(p.name for p in (tmp_path / "r").glob("ckpt_*"))
    assert names == ["ckpt_16.safetensors", "ckpt_22.safetensors"]
    assert rec.last_checkpoint == "ckpt_22.safetensors"
    assert (rec.time_to_half or 0) <= 22


def test_engineered_counters_reset_per_episode():
    c = EngCounters(2)
    c.update(np.array([[6, 0], [7, 8]]), np.array([False, False]))
    assert c.n.tolist() == [[1, -1], [1, 1]]
    c.update(np.array([[8, 1], [3, 3]]), np.array([False, True]))
    assert c.n.tolist() == [[2, -2], [0, 0]]  # pair 1 reset at done
    bits = c.bits(np.array([[True, False]] * 2), np.array([[5, 0]] * 2, np.int32))
    assert bits.shape == (2, 2, 16)
    assert bits[0, 0, 0] == 1 and bits[0, 1, 0] == 0
    assert bits[1, 0, 6:].sum() == 0  # n == 0 after reset


def test_engineered_counters_ignore_finished_agents():
    c = EngCounters(1)
    c.update(np.array([[6, 6]]), np.array([False]), np.array([[True, False]]))
    assert c.n.tolist() == [[1, 1]]
    c.update(np.array([[6, 6]]), np.array([False]), np.array([[True, False]]))
    assert c.n.tolist() == [[1, 2]]  # agent 0 finished: action not applied
    c.update(np.array([[6, 6]]), np.array([True]), np.array([[True, True]]))
    assert c.n.tolist() == [[0, 0]] and not c.finished.any()


def test_engineered_arm_trains(tmp_path):
    rec = _run(tmp_path, 1, arm="eng")
    assert not rec.training_failure and rec.last_checkpoint == "ckpt_8.safetensors"


def test_stub_is_refused():
    vec = VecCoopEnv(1, run_seed_sequence(0, "gtm", "registered"), True,
                     env_factory=_factory())
    _assert_not_stub(vec)
    vec.envs[0].using_stub = True
    with pytest.raises(RuntimeError, match="stub"):
        _assert_not_stub(vec)


def test_train_run_identical_first_rollout_across_worker_counts(tmp_path):
    from tests.unit.n10c.scripted_env import rand_factory

    acts, params = {}, {}
    for nw in (1, 2):
        def grab(u, buf, nw=nw):
            acts.setdefault(nw, buf.actions.copy())

        train_run(
            "gtm", 3, "registered", PER_UPDATE, tmp_path / f"w{nw}",
            env_factory=rand_factory, n_pairs=NP, horizon=H,
            log_every_pair_steps=PER_UPDATE, ckpt_every_pair_steps=PER_UPDATE,
            n_workers=nw, _rollout_hook=grab,
        )
        params[nw] = mx.load(str(tmp_path / f"w{nw}" / "ckpt_8.safetensors"))
    assert np.array_equal(acts[1], acts[2])
    for k in ("pi.weight", "v.weight", "fuse.bias"):
        d = np.abs(np.array(params[1][k]) - np.array(params[2][k]))
        assert d.max() <= 3 * 2 * 5e-4 + 1e-6
        assert (d <= 1e-5).mean() > 0.95
