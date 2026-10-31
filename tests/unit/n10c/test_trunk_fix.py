"""Amendment 2026-10-25 (trunk collapse): training options, finished-agent
mask, trunk-liveness guard, Adam epsilon, individual coin reward, trunk
pre-training, and a trivially learnable action-dependent task."""
import dataclasses
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.campaign import guards  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness import devledger  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness import train as tr  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness.policy import (  # noqa: E402
    PairPolicy,
)
from experiments.procgen_multiagent_gtm_comm.harness.pretrain import (  # noqa: E402
    pretrain_trunk,
)
from experiments.procgen_multiagent_gtm_comm.harness.rollout import (  # noqa: E402
    compute_gae,
)
from tests.unit.n10c import _backend  # noqa: E402

GOOD = 7  # the rewarded action of the bandit task


class _Coins:
    """Agent A reaches its coin at step 0, agent B at step 1 (per episode)."""

    def __init__(self, coin_step: int, rand_seed: int) -> None:
        self.coin_step, self.t, self.seed = coin_step, -1, rand_seed

    def act(self, actions) -> None:
        self.t += 1

    def observe(self):
        coin = self.t == self.coin_step
        img = np.random.default_rng([self.seed, self.t + 2]).integers(
            0, 256, (1, 64, 64, 3), dtype=np.uint8)
        return np.array([10.0 if coin else 0.0]), {"rgb": img}, np.array([coin])

    def close(self) -> None:
        pass


def _coins_factory():
    n = [0]

    def f(rand_seed):
        n[0] += 1
        return _Coins(0 if n[0] % 2 else 1, rand_seed)

    return f


class _Bandit:
    """One action per episode: GOOD reaches the coin, anything else dies."""

    def __init__(self, rand_seed: int) -> None:
        rng = np.random.default_rng(rand_seed)
        self.img = rng.integers(0, 256, (1, 64, 64, 3), dtype=np.uint8)
        self.a = None

    def act(self, actions) -> None:
        self.a = int(np.asarray(actions)[0])

    def observe(self):
        if self.a is None:
            return np.array([0.0]), {"rgb": self.img}, np.array([False])
        coin = self.a == GOOD
        return (np.array([10.0 if coin else 0.0]), {"rgb": self.img},
                np.array([True]))

    def close(self) -> None:
        pass


# -- presets / options --------------------------------------------------------
def test_frozen_preset_is_the_options_default_and_unchanged():
    o = tr.PRESETS[tr.FROZEN_PRESET]
    assert o == tr.TrainOptions()
    assert (o.coin_reward, o.adam_eps, o.mask_finished, o.liveness,
            o.pretrained_trunk) == (0.0, 1e-8, False, "off", False)
    # the registered preset is fix C (amendment 2026-10-25, section 5);
    # the frozen preset stays selectable to reproduce n10c-freeze-20261024
    assert tr.FROZEN_PRESET == "frozen-20261024"
    assert guards.REGISTERED_TRAIN_OPTIONS == "fix-c-coin-pretrain"


def test_fix_presets():
    a, b, c = (tr.PRESETS[k] for k in
               ("fix-a-coin", "fix-b-pretrain", "fix-c-coin-pretrain"))
    for o in (a, b, c):
        assert (o.adam_eps, o.mask_finished, o.liveness) == (1e-5, True, "abort")
    assert (a.coin_reward, a.pretrained_trunk) == (0.5, False)
    assert (b.coin_reward, b.pretrained_trunk) == (0.0, True)
    assert (c.coin_reward, c.pretrained_trunk) == (0.5, True)


def test_unregistered_options_and_trunk_mismatch_refused(tmp_path):
    odd = dataclasses.replace(tr.PRESETS["fix-a-coin"], coin_reward=0.7)
    with pytest.raises(ValueError, match="PRESETS"):
        tr.train_run("mlp", 0, "registered", 8, tmp_path, n_pairs=2, horizon=4,
                     env_factory=_coins_factory(), options=odd)
    with pytest.raises(ValueError, match="needs"):
        tr.train_run("mlp", 0, "registered", 8, tmp_path, n_pairs=2, horizon=4,
                     env_factory=_coins_factory(),
                     options=tr.PRESETS["fix-c-coin-pretrain"])


def test_registered_run_must_use_registered_preset(monkeypatch):
    ctx = guards.make_context_for_tests()
    monkeypatch.setattr(guards, "_runtime_probe", lambda: {
        "system": "Darwin", "machine": "arm64", "backend": "metal"})
    kw = dict(arm="gtm", seed=0, condition="registered",
              total_pair_steps=50_000_000,
              settings=dict(tr.HOST_SETTINGS["macm1"]),
              config=dict(guards.REGISTERED_TRAIN_CONFIG), has_env_factory=False)
    with pytest.raises(guards.GuardError, match="training options"):
        guards.check_train_boundary(ctx, **kw, options_preset="fix-a-coin")
    # development mode: any preset
    assert guards.check_train_boundary(
        None, **(kw | {"total_pair_steps": 64}), options_preset="fix-a-coin") is None


def test_gate_dev_runs_may_reach_ten_million_steps():
    kw = dict(arm="gtm", seed=9101, condition="registered",
              settings=dict(tr.HOST_SETTINGS["kxkm"]),
              config=dict(guards.REGISTERED_TRAIN_CONFIG), has_env_factory=False,
              options_preset="fix-c-coin-pretrain")
    assert guards.dev_max_pair_steps("gate_dev") == 10_000_000
    assert guards.dev_max_pair_steps("dev_train") == guards.DEV_MAX_PAIR_STEPS
    assert guards.check_train_boundary(
        None, total_pair_steps=10_000_000, dev_kind="gate_dev", **kw) is None
    with pytest.raises(guards.GuardError, match="gate_dev"):
        guards.check_train_boundary(
            None, total_pair_steps=10_000_001, dev_kind="gate_dev", **kw)
    with pytest.raises(guards.GuardError, match="dev_train"):
        guards.check_train_boundary(
            None, total_pair_steps=1_000_001, dev_kind="dev_train", **kw)


# -- Adam epsilon -------------------------------------------------------------
def test_adam_eps_per_preset():
    assert tr.make_optimizer(tr.PRESETS[tr.FROZEN_PRESET]).eps == 1e-8
    for k in ("fix-a-coin", "fix-b-pretrain", "fix-c-coin-pretrain"):
        assert tr.make_optimizer(tr.PRESETS[k]).eps == 1e-5


def test_adam_bias_correction_only_in_bc_preset():
    for k, o in tr.PRESETS.items():
        assert tr.make_optimizer(o).bias_correction is k.endswith("-bc")
    bc, c = tr.PRESETS["fix-c-coin-pretrain-bc"], tr.PRESETS["fix-c-coin-pretrain"]
    assert dataclasses.replace(bc, preset=c.preset,
                               adam_bias_correction=False) == c


def test_adam_eps_bounds_the_first_step_on_tiny_gradients():
    # first MLX Adam step (no bias correction): lr*(1-b1)g / (sqrt(1-b2)|g| + eps)
    g = 1e-6
    steps = {}
    for eps in (1e-8, 1e-5):
        opt = tr.make_optimizer(dataclasses.replace(
            tr.PRESETS[tr.FROZEN_PRESET], adam_eps=eps))
        p = {"w": mx.zeros((1,))}
        opt.init(p)
        new = opt.apply_gradients({"w": mx.full((1,), g)}, p)
        steps[eps] = abs(float(new["w"].item()))
        want = tr.LR * 0.1 * g / (np.sqrt(0.001) * g + eps)
        assert steps[eps] == pytest.approx(want, rel=1e-3)
    assert steps[1e-5] < steps[1e-8] / 5


# -- per-agent reward and live mask in the rollout ----------------------------
def test_gae_accepts_per_agent_rewards():
    rng = np.random.default_rng(0)
    r = rng.random((5, 3, 2)).astype(np.float32)
    v = rng.random((5, 3, 2)).astype(np.float32)
    d = rng.random((5, 3)) < 0.3
    last = rng.random((3, 2)).astype(np.float32)
    adv, ret = compute_gae(r, v, d, last)
    for i in range(2):
        a_i, r_i = compute_gae(r[..., i], v[..., i:i + 1], d, last[:, i:i + 1])
        np.testing.assert_allclose(adv[..., i], a_i[..., 0], rtol=1e-6)
        np.testing.assert_allclose(ret[..., i], r_i[..., 0], rtol=1e-6)


@pytest.mark.parametrize("preset,bonus", [("fix-a-coin", 0.5),
                                          (tr.FROZEN_PRESET, 0.0)])
def test_coin_reward_and_live_mask_recorded(tmp_path, preset, bonus):
    bufs = []
    rec = tr.train_run(
        "mlp", 0, "registered", 8, tmp_path / preset, n_pairs=2, horizon=4,
        env_factory=_coins_factory(), log_every_pair_steps=8,
        ckpt_every_pair_steps=8, options=tr.PRESETS[preset],
        _rollout_hook=lambda u, b: bufs.append(b))
    b = bufs[0]
    # episode = steps (0, 1): A coins at step 0 and is frozen at step 1
    assert b.live[0].all() and b.live[2].all()
    assert (~b.live[1, :, 0]).all() and b.live[1, :, 1].all()
    np.testing.assert_array_equal(b.team_reward[:, 0], [0, 1, 0, 1])
    np.testing.assert_array_equal(b.agent_reward[0, 0], [bonus, 0])
    np.testing.assert_array_equal(b.agent_reward[1, 0], [0, bonus])
    assert rec.coin_rate_curve == [[8, 1.0]]
    assert rec.entropy_curve and rec.entropy_curve[0][2] > 0


# -- finished-agent mask in the loss ------------------------------------------
def _mb(rng, n=6, arm="nocomm", nd=0):
    live = np.ones((n, 2), np.float32)
    live[1, 0] = live[3, 1] = live[4, 0] = 0
    return {
        "obs": mx.array(rng.integers(0, 256, (n, 2, 64, 64, 3), dtype=np.uint8)),
        "summary": mx.zeros((n, 2, 2)), "eng_bits": mx.zeros((n, 2, 16)),
        "actions": mx.array(rng.integers(0, 15, (n, 2))),
        "noise": mx.zeros((n, 2, nd)),
        "logp": mx.array(np.full((n, 2), -np.log(15), np.float32)),
        "advantages": mx.array(rng.normal(size=(n, 2)).astype(np.float32)),
        "returns": mx.array(rng.normal(size=(n, 2)).astype(np.float32)),
        "live": mx.array(live),
    }


def _grads(mask, mb, micro=None):
    model = PairPolicy("nocomm", mx.random.key(3))
    opt = tr.make_optimizer(tr.PRESETS["fix-a-coin"])
    opt.init(model.trainable_parameters())
    gs, _, _ = tr._make_steps(model, opt, False, plastic=False, mask_finished=mask)
    _, g, ok = tr.accumulate_minibatch(gs, mb, micro, mx.random.key(0), mask=mask)
    assert ok
    from mlx.utils import tree_flatten
    return {k: np.array(v) for k, v in tree_flatten(g)}


def _close(a, b, rel=1e-3):
    # conv backward is nondeterministic (MLX 0.31.1, ~1e-6 on CUDA):
    # compare relative to the tensor's scale
    return np.abs(a - b).max() <= rel * max(np.abs(a).max(), 1e-6)


def _perturb_dead(mb, rng):
    out = dict(mb)
    dead = np.array(mb["live"]) == 0
    a = np.array(mb["actions"])
    a[dead] = (a[dead] + 5) % 15
    adv = np.array(mb["advantages"])
    adv[dead] = 50.0
    out["actions"], out["advantages"] = mx.array(a), mx.array(adv)
    return out


def test_finished_agents_do_not_enter_policy_entropy_or_adv_stats():
    rng = np.random.default_rng(1)
    mb = _mb(rng)
    mb2 = _perturb_dead(mb, rng)
    g1, g2 = _grads(True, mb), _grads(True, mb2)
    assert all(_close(g1[k], g2[k]) for k in g1)
    # without the mask the same perturbation changes the gradient
    h1, h2 = _grads(False, mb), _grads(False, mb2)
    assert not all(_close(h1[k], h2[k], rel=5e-2) for k in h1)


def test_finished_agents_do_not_enter_the_entropy_bonus(monkeypatch):
    # isolate the entropy term: no value loss; perturb the dead samples'
    # observations (-> their logits and entropy) and their advantages only
    monkeypatch.setattr(tr, "VALUE_COEF", 0.0)
    rng = np.random.default_rng(4)
    mb = _mb(rng)
    mb["advantages"] = mx.zeros(mb["advantages"].shape)  # no policy term
    obs = np.array(mb["obs"])
    dead = np.array(mb["live"]) == 0
    obs2 = obs.copy()
    obs2[dead] = 255 - obs2[dead]
    mb2 = dict(mb, obs=mx.array(obs2))
    g1, g2 = _grads(True, mb), _grads(True, mb2)
    heads = [k for k in g1 if not k.startswith("trunk.")]
    assert all(_close(g1[k], g2[k], 1e-4) for k in heads)
    assert all(_close(g1[k], g2[k], 5e-3) for k in g1)
    h1, h2 = _grads(False, mb), _grads(False, mb2)
    assert not all(_close(h1[k], h2[k], 5e-2) for k in heads)


def test_masked_loss_is_invariant_to_micro_batching():
    mb = _mb(np.random.default_rng(2))
    full, micro = _grads(True, mb), _grads(True, mb, micro=2)
    # heads exact up to float rounding; trunk convs pick batch-size
    # dependent kernels on CUDA (~1e-3 relative)
    for k in full:
        assert _close(full[k], micro[k], 5e-3 if k.startswith("trunk.") else 1e-5), k


def test_masked_adv_normalisation_uses_live_samples_only():
    adv = mx.array([[1.0, 100.0], [3.0, -7.0]])
    live = mx.array([[1.0, 0.0], [1.0, 0.0]])
    out = np.array(tr.normalize_adv(adv, live))
    np.testing.assert_allclose(out[:, 0], [-1.0, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(out[:, 1], [0.0, 0.0])
    assert float(tr.live_scale(live).item()) == 2.0


# -- trunk-liveness guard -----------------------------------------------------
def _frames(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (n, 2, 64, 64, 3), dtype=np.uint8),
            np.zeros((n, 2, 2), np.float32))


def _kill_h(model):
    model.trunk.fc.weight = mx.zeros(model.trunk.fc.weight.shape)
    # far below zero: Adam momentum from earlier steps cannot revive it
    model.trunk.fc.bias = mx.full(model.trunk.fc.bias.shape, -100.0)


def test_liveness_probe_healthy_init_not_flagged():
    obs, summ = _frames()
    m = tr.liveness_probe(PairPolicy("nocomm", mx.random.key(0)), obs, summ,
                          None, mx.random.key(1))
    assert not m["flagged"]
    assert m["block3_pos_frac"] > 0.05 and m["live_h_frac"] > 0.5
    assert m["logit_std"] > 0 and m["value_std"] > 0


def test_liveness_probe_flags_dead_h_and_dead_block3():
    obs, summ = _frames()
    model = PairPolicy("nocomm", mx.random.key(0))
    _kill_h(model)
    m = tr.liveness_probe(model, obs, summ, None, mx.random.key(1))
    assert m["flagged"] and m["live_h_frac"] == 0.0
    assert {"live_h_frac", "logit_std", "value_std"} <= set(m["reasons"])
    model = PairPolicy("nocomm", mx.random.key(0))
    c2 = model.trunk.blocks[2].r2.c2
    c2.bias = mx.full(c2.bias.shape, -1e4)
    m = tr.liveness_probe(model, obs, summ, None, mx.random.key(1))
    assert m["flagged"] and "block3_pos_frac" in m["reasons"]


def test_liveness_guard_aborts_the_run(tmp_path, monkeypatch):
    fast = dataclasses.replace(tr.PRESETS["fix-a-coin"], liveness_every=8)
    monkeypatch.setitem(tr.PRESETS, "fix-a-coin", fast)

    def hook(update, model):
        if update == 1:
            _kill_h(model)

    rec = tr.train_run(
        "nocomm", 0, "registered", 32, tmp_path / "r", n_pairs=2, horizon=4,
        env_factory=_coins_factory(), log_every_pair_steps=8,
        ckpt_every_pair_steps=8, options=fast, _fault_hook=hook)
    assert rec.liveness_abort and rec.training_failure
    assert "liveness" in rec.failure_reason
    # the run stops at the FIRST flagged probe (the in-place kill reaches
    # the compiled optimizer state within one or two updates)
    flags = [m["flagged"] for m in rec.liveness]
    assert flags[0] is False and flags[-1] is True and not any(flags[:-1])
    assert rec.liveness[-1]["step"] == rec.pair_steps_done < 32
    assert rec.n_updates == rec.pair_steps_done // 8
    data = json.loads((tmp_path / "r" / "run.json").read_text())
    assert data["liveness_abort"] is True
    assert data["train_options"]["preset"] == "fix-a-coin"


def test_liveness_flag_mode_records_without_abort(tmp_path, monkeypatch):
    flag = dataclasses.replace(tr.PRESETS["fix-a-coin"], liveness="flag",
                               liveness_every=8)
    monkeypatch.setitem(tr.PRESETS, "fix-a-coin", flag)
    rec = tr.train_run(
        "nocomm", 0, "registered", 32, tmp_path / "r", n_pairs=2, horizon=4,
        env_factory=_coins_factory(), log_every_pair_steps=8,
        ckpt_every_pair_steps=8, options=flag,
        _fault_hook=lambda u, m: _kill_h(m) if u == 1 else None)
    assert rec.liveness_flagged and not rec.liveness_abort
    assert rec.pair_steps_done == 32 and not rec.training_failure


# -- trivially learnable action-dependent task --------------------------------
def test_coin_reward_learns_an_action_dependent_task(tmp_path):
    """With +0.5 per own coin the policy must learn the rewarded action of a
    one-step bandit (random: 1/15); a dead or inert learner cannot."""
    bufs = []
    n, h, upd = 16, 16, 10
    rec = tr.train_run(
        "nocomm", 0, "registered", n * h * upd, tmp_path / "b", n_pairs=n,
        horizon=h, env_factory=_Bandit, log_every_pair_steps=n * h,
        ckpt_every_pair_steps=n * h * upd, options=tr.PRESETS["fix-a-coin"],
        _rollout_hook=lambda u, b: bufs.append((b.actions == GOOD).mean()))
    assert not rec.training_failure
    assert bufs[0] < 0.2  # first rollout: initial policy
    assert np.mean(bufs[-3:]) > 0.8, bufs


# -- trunk pre-training -------------------------------------------------------
class _VecBandit:
    def __init__(self, n):
        self.n, self.a = n, None
        self.img = np.random.default_rng(0).integers(0, 256, (n, 64, 64, 3),
                                                     dtype=np.uint8)

    def act(self, a):
        self.a = np.asarray(a)

    def observe(self):
        if self.a is None:
            return np.zeros(self.n), {"rgb": self.img}, np.zeros(self.n, bool)
        return ((self.a == GOOD) * 10.0, {"rgb": self.img},
                np.ones(self.n, bool))

    def close(self):
        pass


def test_pretrain_writes_a_loadable_trunk(tmp_path):
    rec = pretrain_trunk(5, 4 * 8 * 3, tmp_path, n_envs=4, horizon=8,
                         liveness_every=32, env=_VecBandit(4))
    assert rec.usable and rec.steps_done == 96 and rec.n_updates == 3
    assert [m["step"] for m in rec.liveness] == [0, 32, 64, 96]
    path = tmp_path / rec.trunk_file
    model = PairPolicy("gtm", mx.random.key(0))
    before = np.array(model.fuse.weight)
    tr.load_trunk(model, path)
    saved = mx.load(str(path))
    np.testing.assert_array_equal(np.array(model.trunk.fc.weight),
                                  np.array(saved["fc.weight"]))
    np.testing.assert_array_equal(np.array(model.fuse.weight), before)
    assert tr.file_sha256(path) == rec.trunk_sha256
    assert json.loads((tmp_path / "pretrain.json").read_text())["usable"]


@pytest.mark.skipif(_backend.on_cuda(), reason=_backend.CUDA_NONDETERMINISM)
def test_pretrain_micro_batches_match_the_full_minibatch(tmp_path):
    # KXKM-AI pre-trains in micro-batches (HOST_SETTINGS): same update
    recs = {}
    for name, micro in (("full", None), ("micro", 4)):
        recs[name] = pretrain_trunk(5, 4 * 8 * 2, tmp_path / name, n_envs=4,
                                    horizon=8, liveness_every=32,
                                    env=_VecBandit(4), micro_batch_pairs=micro,
                                    clear_cache=micro is not None)
    assert recs["micro"].settings == {"micro_batch_pairs": 4, "clear_cache": True}
    assert recs["full"].settings == {"micro_batch_pairs": None, "clear_cache": False}
    full = mx.load(str(tmp_path / "full" / recs["full"].trunk_file))
    micro = mx.load(str(tmp_path / "micro" / recs["micro"].trunk_file))
    assert full.keys() == micro.keys()
    for k in full:
        np.testing.assert_allclose(np.array(micro[k]), np.array(full[k]),
                                   rtol=1e-4, atol=1e-6)


def test_pretrain_micro_slices():
    from experiments.procgen_multiagent_gtm_comm.harness.pretrain import micro_slices
    assert micro_slices(10, None) == [(0, 10, 1.0)]
    assert micro_slices(10, 16) == [(0, 10, 1.0)]
    sl = micro_slices(10, 4)
    assert [(a, b) for a, b, _ in sl] == [(0, 4), (4, 8), (8, 10)]
    assert abs(sum(w for *_, w in sl) - 1.0) < 1e-12


def test_pretrained_preset_records_trunk_hash(tmp_path):
    rec0 = pretrain_trunk(5, 32, tmp_path / "p", n_envs=4, horizon=8,
                          env=_VecBandit(4))
    trunk = tmp_path / "p" / rec0.trunk_file
    rec = tr.train_run(
        "mlp", 0, "registered", 8, tmp_path / "r", n_pairs=2, horizon=4,
        env_factory=_coins_factory(), log_every_pair_steps=8,
        ckpt_every_pair_steps=8, options=tr.PRESETS["fix-c-coin-pretrain"],
        init_trunk=trunk)
    assert rec.init_trunk_sha256 == rec0.trunk_sha256


# -- development ledger -------------------------------------------------------
def test_gate_ledger_kinds_are_separate_and_capped(tmp_path):
    led = tmp_path / "ledger.json"
    for _ in range(4):  # exhaust the smoke allowance of gtm
        devledger.reserve("gtm", "registered", 10, path=led)
    with pytest.raises(devledger.SmokeLimitError):
        devledger.reserve("gtm", "registered", 10, path=led)
    for arm, cap in (("gtm", 20), ("eng", 19), ("nocomm", 6), ("mlp", 4)):
        assert devledger.gate_allowance("gate_dev", arm) == cap
        for _ in range(cap):
            devledger.reserve(arm, "registered", 10, kind="gate_dev", path=led)
        with pytest.raises(devledger.SmokeLimitError, match="gate_dev"):
            devledger.reserve(arm, "registered", 10, kind="gate_dev", path=led)
    for _ in range(8):
        devledger.reserve("trunk", "single", 10, kind="pretrain_dev", path=led)
    with pytest.raises(devledger.SmokeLimitError):
        devledger.reserve("trunk", "single", 10, kind="pretrain_dev", path=led)
    with pytest.raises(ValueError):
        devledger.reserve("gtm", "registered", 10, kind="pretrain_dev", path=led)
    with pytest.raises(devledger.SmokeLimitError):
        devledger.reserve("eng", "registered", 10_000_001, kind="gate_dev",
                          path=led)
    with pytest.raises(devledger.SmokeLimitError):  # smoke cap unchanged
        devledger.reserve("eng", "registered", 1_000_001, path=led)
    with pytest.raises(devledger.SmokeLimitError):  # pretrain_dev cap 1M
        devledger.reserve("trunk", "single", 1_000_001, kind="pretrain_dev",
                          path=led)
    long = devledger.reserve("gtm", "registered", 10_000_000, kind="gate_dev",
                             path=tmp_path / "long.json")  # PI 13:33
    assert long.steps == 10_000_000


# -- registered paths: launcher preset / trunk, trunk hash, pre-training -----
def test_registered_pretrained_preset_requires_the_pinned_trunk(monkeypatch):
    ctx = guards.make_context_for_tests()
    monkeypatch.setattr(guards, "_runtime_probe", lambda: {
        "system": "Darwin", "machine": "arm64", "backend": "metal"})
    monkeypatch.setattr(guards, "REGISTERED_TRAIN_OPTIONS", "fix-c-coin-pretrain")
    kw = dict(arm="gtm", seed=0, condition="registered",
              total_pair_steps=50_000_000,
              settings=dict(tr.HOST_SETTINGS["macm1"]),
              config=dict(guards.REGISTERED_TRAIN_CONFIG), has_env_factory=False,
              options_preset="fix-c-coin-pretrain", pretrained_trunk=True)
    with pytest.raises(guards.GuardError, match="trunk sha256"):
        guards.check_train_boundary(ctx, **kw, init_trunk_sha256="ab" * 32)
    monkeypatch.setitem(guards.REGISTERED_TRUNK_SHA256, 0, "ab" * 32)
    with pytest.raises(guards.GuardError, match="trunk sha256"):
        guards.check_train_boundary(ctx, **kw, init_trunk_sha256="cd" * 32)
    monkeypatch.setattr(guards, "_procgen_probe", lambda: {})
    try:
        guards.check_train_boundary(ctx, **kw, init_trunk_sha256="ab" * 32)
    except guards.GuardError as exc:  # only the procgen-build probe may remain
        assert "trunk" not in str(exc) and "options" not in str(exc)


def test_launcher_passes_registered_preset_and_seed_trunk(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments.procgen_multiagent_gtm_comm.campaign import queue as q
    from experiments.procgen_multiagent_gtm_comm.campaign import run_campaign as rc

    seen = []

    def train_fn(**kw):
        seen.append(kw)
        kw["out_dir"].mkdir(parents=True, exist_ok=True)
        (kw["out_dir"] / f"ckpt_{q.TOTAL_PAIR_STEPS}.safetensors").write_text("x")
        return SimpleNamespace(training_failure=False,
                               pair_steps_done=q.TOTAL_PAIR_STEPS,
                               last_checkpoint=f"ckpt_{q.TOTAL_PAIR_STEPS}.safetensors",
                               wall_time_s=1.0, failure_reason=None)

    for preset in (tr.FROZEN_PRESET, "fix-c-coin-pretrain"):
        monkeypatch.setattr(guards, "REGISTERED_TRAIN_OPTIONS", preset)
        seen.clear()
        c = rc.Campaign(
            rc.Options(campaign_dir=tmp_path / preset, host="macm1",
                       registration=guards.make_context_for_tests()),
            train_fn=train_fn, eval_fn=lambda *a, **k: None,
            cross_fn=lambda *a, **k: None, gpu_fn=lambda: None)
        spec = q.runs_for_host("macm1")[0]
        c.train(spec, c.store.load(spec))
        kw = seen[0]
        assert kw["options"] == tr.PRESETS[preset]
        if preset == tr.FROZEN_PRESET:
            assert "init_trunk" not in kw
        else:
            assert kw["init_trunk"] == guards.trunk_path(tmp_path / preset, spec.seed)


def test_registered_pretraining_boundary(tmp_path):
    ctx = guards.make_context_for_tests()
    with pytest.raises(guards.GuardError, match="RegistrationContext"):
        guards.check_pretrain_boundary(object(), seed=0, total_steps=10**6,
                                       has_env=False)
    with pytest.raises(guards.GuardError, match="seed"):
        guards.check_pretrain_boundary(ctx, seed=9101, total_steps=10**6,
                                       has_env=False)
    with pytest.raises(guards.GuardError, match="budget"):
        guards.check_pretrain_boundary(ctx, seed=0, total_steps=500_000,
                                       has_env=False)
    with pytest.raises(guards.GuardError, match="real Procgen"):
        pretrain_trunk(0, 10**6, tmp_path, env=_VecBandit(4), registration=ctx)
    with pytest.raises(guards.GuardError, match="limited"):
        guards.check_pretrain_boundary(None, seed=9101, total_steps=10**6 + 1,
                                       has_env=False)


def test_pretrain_ledgers_a_failed_env_build(tmp_path, monkeypatch):
    from experiments.procgen_multiagent_gtm_comm.harness import pretrain as pt

    def boom(*a, **k):
        raise RuntimeError("no cmake")

    monkeypatch.setattr(pt, "make_env", boom)
    led = tmp_path / "ledger.json"
    with pytest.raises(RuntimeError, match="cmake"):
        pt.pretrain_trunk(9101, 1000, tmp_path / "o", dev_ledger=led)
    run = devledger.read(led)["runs"][-1]
    assert run["kind"] == "pretrain_dev" and run["outcome"] == "crashed"
    assert (tmp_path / "o" / "pretrain.json").exists()


def test_pretrain_ledger_survives_a_failing_close(tmp_path, monkeypatch):
    from experiments.procgen_multiagent_gtm_comm.harness import pretrain as pt

    class _BadClose(_VecBandit):
        def close(self):
            raise OSError("close failed")

    monkeypatch.setattr(pt, "make_env", lambda n, s: _BadClose(n))
    led = tmp_path / "ledger.json"
    with pytest.raises(OSError):
        pt.pretrain_trunk(9101, 32, tmp_path / "o", n_envs=4, horizon=8,
                          dev_ledger=led)
    assert devledger.read(led)["runs"][-1]["outcome"] == "completed"
    assert json.loads((tmp_path / "o" / "pretrain.json").read_text())["usable"]


def test_gate_report_thresholds_match_the_guard():
    from experiments.procgen_multiagent_gtm_comm.harness import gate_report
    assert gate_report.LIVENESS_MIN_FRAC == tr.LIVENESS_MIN_FRAC
    assert gate_report.LIVENESS_STD_EPS == tr.LIVENESS_STD_EPS
    assert gate_report.GATE_PROBE_EVERY == (
        tr.PRESETS[gate_report.GATE_PRESET].liveness_every)


# -- per-run-pretraining control (PI 2026-10-25 19:12) ------------------------
def test_per_run_pretraining_seeding_and_record(tmp_path):
    from experiments.procgen_multiagent_gtm_comm.harness import pretrain as pt
    from experiments.procgen_multiagent_gtm_comm.harness.codes import (
        ARM_CODES,
        CONDITION_CODES,
    )

    assert pt.pretrain_seed_entropy(101) == [101, 99]
    assert pt.pretrain_seed_entropy(101, ("eng", "closed")) == [
        101, 99, ARM_CODES["eng"], CONDITION_CODES["closed"]]
    with pytest.raises(ValueError):
        pt.pretrain_seed_entropy(101, ("nope", "closed"))
    shared = pretrain_trunk(101, 32, tmp_path / "s", n_envs=4, horizon=8,
                            env=_VecBandit(4))
    own = pretrain_trunk(101, 32, tmp_path / "o", n_envs=4, horizon=8,
                         env=_VecBandit(4), unit=("gtm", "registered"))
    again = pretrain_trunk(101, 32, tmp_path / "a", n_envs=4, horizon=8,
                           env=_VecBandit(4), unit=("gtm", "registered"))
    other = pretrain_trunk(101, 32, tmp_path / "x", n_envs=4, horizon=8,
                           env=_VecBandit(4), unit=("gtm", "closed"))
    assert shared.unit is None and shared.seed_sequence == [101, 99]
    assert own.unit == ["gtm", "registered"]
    assert own.seed_sequence == pt.pretrain_seed_entropy(101, ("gtm", "registered"))
    # (trunk bytes are not compared across runs: MLX training is not
    # bit-reproducible; the seeding is what the control specifies)
    assert again.seed_sequence == own.seed_sequence != other.seed_sequence
    assert len({shared.trunk_sha256, own.trunk_sha256, other.trunk_sha256}) == 3
    rec = tr.train_run(
        "gtm", 101, "registered", 8, tmp_path / "r", n_pairs=2, horizon=4,
        env_factory=_coins_factory(), log_every_pair_steps=8,
        ckpt_every_pair_steps=8, options=tr.PRESETS["fix-c-coin-pretrain"],
        init_trunk=tmp_path / "o" / own.trunk_file, per_run_trunk=True)
    assert rec.trunk_mode == "per_run"
    assert rec.init_trunk_sha256 == own.trunk_sha256
    with pytest.raises(ValueError, match="own init_trunk"):
        tr.train_run("gtm", 101, "registered", 8, tmp_path / "r2", n_pairs=2,
                     horizon=4, env_factory=_coins_factory(),
                     log_every_pair_steps=8, ckpt_every_pair_steps=8,
                     per_run_trunk=True)


def test_prefreeze_and_registered_pretraining_enforce_the_procedure(tmp_path):
    for kw in (dict(n_envs=4), dict(horizon=8), dict(liveness_every=10**7)):
        with pytest.raises(ValueError, match="registered pre-training uses"):
            pretrain_trunk(73, 10**6, tmp_path, pre_freeze=True, **kw)
        with pytest.raises(ValueError, match="registered pre-training uses"):
            pretrain_trunk(0, 10**6, tmp_path,
                           registration=guards.make_context_for_tests(), **kw)
