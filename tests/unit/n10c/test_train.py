"""Trainer tests: scripted envs, tiny budgets (n_pairs 2, horizon 4)."""
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map  # noqa: E402

from experiments.procgen_multiagent_gtm_comm.harness.codes import (  # noqa: E402
    run_seed_sequence,
)
from experiments.procgen_multiagent_gtm_comm.harness.policy import (  # noqa: E402
    PairPolicy,
)
from experiments.procgen_multiagent_gtm_comm.harness.train import (  # noqa: E402
    MAX_GRAD_NORM,
    EngCounters,
    _assert_not_stub,
    _make_steps,
    accumulate_minibatch,
    finalize_grads,
    micro_slices,
    normalize_adv,
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


# ---- Task 11: gradient accumulation over micro-batches ----------------

def _rand_mb(n, seed=0, adv_scale=1.0, ret_scale=1.0):
    r = np.random.default_rng(seed)
    return {
        "obs": mx.array(r.integers(0, 256, (n, 2, 64, 64, 3), dtype=np.uint8)),
        "summary": mx.array(r.normal(size=(n, 2, 2)).astype(np.float32)),
        "eng_bits": mx.zeros((n, 2, 16)),
        "actions": mx.array(r.integers(0, 15, (n, 2))),
        "noise": mx.zeros((n, 2, 0)),
        "logp": mx.array(-2.7 + 0.3 * r.normal(size=(n, 2)).astype(np.float32)),
        "advantages": mx.array(
            adv_scale * r.normal(size=(n, 2)).astype(np.float32)
        ),
        "returns": mx.array(ret_scale * r.normal(size=(n, 2)).astype(np.float32)),
    }


def _steps(arm="mlp"):
    model = PairPolicy(arm, mx.random.key(1))
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=5e-4)
    opt.init(model.trainable_parameters())
    return model, _make_steps(model, opt, False, plastic=False)[0]


def _gnorm(grads):
    return float(mx.sqrt(sum((g * g).sum() for _, g in tree_flatten(grads))).item())


def _max_rel(a, b):
    worst = 0.0
    for (_, x), (_, y) in zip(tree_flatten(a), tree_flatten(b), strict=True):
        x, y = np.array(x), np.array(y)
        worst = max(worst, float(np.abs(x - y).max() / (np.abs(y).max() + 1e-12)))
    return worst


def test_uneven_last_minibatch_weights():
    sl = micro_slices(10, 4)
    assert [(a, b) for a, b, _ in sl] == [(0, 4), (4, 8), (8, 10)]
    w = [x for _, _, x in sl]
    assert w == pytest.approx([0.4, 0.4, 0.2]) and sum(w) == pytest.approx(1.0)
    assert micro_slices(3, None) == [(0, 3, 1.0)]
    assert micro_slices(3, 8) == [(0, 3, 1.0)]  # micro larger than minibatch
    # accumulated gradient of an uneven minibatch == full-batch gradient
    _, gs = _steps()
    mb = _rand_mb(7)
    k = mx.random.key(0)
    _, g_full, ok1 = accumulate_minibatch(gs, mb, None, k)
    _, g_mic, ok2 = accumulate_minibatch(gs, mb, 3, k)  # 3 + 3 + 1
    assert ok1 and ok2
    assert _max_rel(g_mic, g_full) < 1e-5


def test_adv_norm_over_full_minibatch():
    adv = mx.array(np.concatenate([np.full(4, 10.0), np.full(4, -10.0)])
                   .reshape(4, 2).astype(np.float32) + np.arange(8).reshape(4, 2))
    full = normalize_adv(adv)
    assert float(full.mean().item()) == pytest.approx(0.0, abs=1e-5)
    assert float(full.std().item()) == pytest.approx(1.0, abs=1e-3)
    # micro-batches whose advantages sit at very different offsets: the
    # accumulated gradient must still equal the full-batch one (per-micro
    # normalisation would erase the offsets and differ).
    _, gs = _steps()
    mb = _rand_mb(8, seed=2)
    off = np.repeat([0.0, 5.0, -5.0, 9.0], 2).reshape(8, 1)
    mb["advantages"] = mb["advantages"] + mx.array(off.astype(np.float32))
    k = mx.random.key(0)
    _, g_full, _ = accumulate_minibatch(gs, mb, None, k)
    _, g_mic, _ = accumulate_minibatch(gs, mb, 2, k)
    assert _max_rel(g_mic, g_full) < 1e-5
    parts = []
    for a, b, w in micro_slices(8, 2):  # per-micro normalisation (wrong)
        sub = {k: v[a:b] for k, v in mb.items()}
        g = accumulate_minibatch(gs, sub, None, k)[1]
        parts.append(tree_map(lambda x, w=w: x * w, g))
    g_wrong = tree_map(lambda *x: sum(x), *parts)
    assert _max_rel(g_wrong, g_full) > 1e-3  # the test is sensitive


def test_clip_applied_to_accumulated_grad():
    _, gs = _steps()
    mb = _rand_mb(8, seed=3, ret_scale=30.0)
    k = mx.random.key(0)
    _, g_mic, ok = accumulate_minibatch(gs, mb, 2, k)
    assert ok
    raw = _gnorm(g_mic)
    assert raw > MAX_GRAD_NORM  # accumulation itself must not clip
    clipped, norm = finalize_grads(g_mic, scale=None)
    assert float(norm.item()) == pytest.approx(raw, rel=1e-5)
    assert _gnorm(clipped) == pytest.approx(MAX_GRAD_NORM, rel=1e-4)
    _, g_full, _ = accumulate_minibatch(gs, mb, None, k)
    assert _max_rel(clipped, finalize_grads(g_full, scale=None)[0]) < 1e-4


def test_nonfinite_micro_stops_before_apply(tmp_path):
    _, gs = _steps()
    mb = _rand_mb(6, seed=4)
    ret = np.array(mb["returns"])
    ret[3, 0] = np.nan  # lands in the second micro of size 2
    mb["returns"] = mx.array(ret)
    loss, grads, ok = accumulate_minibatch(gs, mb, 2, mx.random.key(0))
    assert ok is False and grads is None
    # end to end: single minibatch per epoch -> failure before any update
    def poison(update, buf):
        buf.advantages[0, 0, 0] = np.nan

    rec = train_run(
        "mlp", 0, "registered", PER_UPDATE, tmp_path / "nf",
        env_factory=_factory(), n_pairs=NP, horizon=H,
        log_every_pair_steps=PER_UPDATE, ckpt_every_pair_steps=PER_UPDATE,
        minibatch_pairs=PER_UPDATE, micro_batch_pairs=2, _rollout_hook=poison,
    )
    assert rec.training_failure is True
    assert rec.optimizer_steps == 0
    assert rec.last_checkpoint is None


def _run_params(tmp_path, arm, micro, name, *, total=PER_UPDATE, mbp=4):
    """(params after update 1, params after the last update, RunRecord)."""
    snaps = {}
    rollouts = {}

    def hook(update, model):
        snaps[update] = {k: np.array(v) for k, v in
                         tree_flatten(model.parameters())}

    def rhook(update, buf):
        if update == 1:
            rollouts.update({n: np.array(getattr(buf, n)) for n in (
                "obs", "actions", "logp", "value", "returns")})

    rec = train_run(
        arm, 5, "registered", total, tmp_path / name, env_factory=_factory(),
        n_pairs=NP, horizon=H, log_every_pair_steps=PER_UPDATE,
        ckpt_every_pair_steps=PER_UPDATE, minibatch_pairs=mbp,
        micro_batch_pairs=micro, _fault_hook=hook, _rollout_hook=rhook,
    )
    _run_params.rollouts = rollouts  # first rollout of the last call
    return snaps[1], snaps[max(snaps)], rec


LR_ = 5e-4


def _max_abs_diff(a, b):
    return max(float(np.abs(a[k] - b[k]).max()) for k in a)


@pytest.mark.parametrize("arm", ["gtm", "dial", "mlp", "nocomm"])
def test_same_mode_repeat_control(tmp_path, arm):
    """Same seed, same mode, run twice: inputs are bit-identical, the
    parameters after one update are NOT (see the localisation test below),
    and accumulation does not add error beyond the same-mode noise floor."""
    p1, _, _ = _run_params(tmp_path, arm, None, "rep1")
    r1 = dict(_run_params.rollouts)
    p2, _, _ = _run_params(tmp_path, arm, None, "rep2")
    r2 = dict(_run_params.rollouts)
    for n in r1:  # identical init + rollout inputs (forward is bit-exact)
        assert np.array_equal(r1[n], r2[n]), n
    pa, _, _ = _run_params(tmp_path, arm, 2, "acc")
    # 3 epochs x 2 minibatches = 6 Adam steps; a ~1e-7 trunk-gradient
    # difference flips near-zero entries by up to 2*lr per step.
    bound = 6 * 2 * LR_ + 1e-6
    floor = _max_abs_diff(p1, p2)  # same-mode noise floor (not hidden)
    assert floor <= bound
    assert _max_abs_diff(p1, pa) <= bound


@pytest.mark.parametrize("arm", ["mlp", "gtm"])
def test_nondeterminism_is_localised_to_conv_backward(arm):
    # Measured finding (Metal, MLX 0.31.1): init and forward are
    # bit-identical across repeats; so are the gradients of every tensor
    # outside the IMPALA conv stack. Only conv weight/bias gradients
    # (trunk.blocks.*) vary run to run, by ~1e-9 absolute (~1e-7 rel),
    # with or without mx.compile (conv backward reductions are not
    # order-deterministic). Not the GTM path, not accumulation.
    m1 = PairPolicy(arm, mx.random.key(1))
    m2 = PairPolicy(arm, mx.random.key(1))
    f1, f2 = tree_flatten(m1.parameters()), tree_flatten(m2.parameters())
    assert all(np.array_equal(np.array(a), np.array(b))
               for (_, a), (_, b) in zip(f1, f2, strict=True))
    opt = optim.Adam(learning_rate=LR_)
    opt.init(m1.trainable_parameters())
    gs = _make_steps(m1, opt, False, plastic=False)[0]
    mb = _rand_mb(8, seed=3)
    mb["noise"] = (m1.sample_noise(8, mx.random.key(5)) if m1.noise_dim
                   else mb["noise"])
    k = mx.random.key(0)
    out = [m1(mb["obs"], mb["summary"], None, train=True, key=k,
              noise=mb["noise"] if m1.noise_dim else None) for _ in range(2)]
    assert np.array_equal(np.array(out[0].logits), np.array(out[1].logits))
    gr = [dict(tree_flatten(accumulate_minibatch(gs, mb, None, k)[1]))
          for _ in range(4)]
    for g in gr[1:]:
        for name, a in gr[0].items():
            a, b = np.array(a), np.array(g[name])
            if not name.startswith("trunk.blocks"):
                assert np.array_equal(a, b), name
            else:
                assert np.abs(a - b).max() <= 1e-5 * (np.abs(a).max() + 1e-12)


def test_optimizer_equivalence_vs_independent_adam():
    """Same params AND Adam state, one fixed minibatch through both paths;
    each resulting update is compared with an independent numpy Adam
    (MLX Adam default: no bias correction, b1 .9, b2 .999, eps 1e-8)
    applied to that path's own finalized gradient."""
    def build():
        m = PairPolicy("mlp", mx.random.key(1))
        mx.eval(m.parameters())
        o = optim.Adam(learning_rate=LR_)
        o.init(m.trainable_parameters())
        gs_, ap_, _ = _make_steps(m, o, False, plastic=False)
        return m, o, gs_, ap_

    (ma, oa, gsa, apa), (mb_, ob, gsb, apb) = build(), build()
    k = mx.random.key(0)
    warm = _rand_mb(8, seed=1)
    # ONE warm gradient applied to both clones (recomputing it per clone
    # would already differ at ~1e-9: conv backward is not bit-repeatable).
    _, g, _ = accumulate_minibatch(gsa, warm, None, k)
    g = finalize_grads(g, None)[0]
    mx.eval(g)
    for m, o, ap in ((ma, oa, apa), (mb_, ob, apb)):
        ap(g)
        mx.eval(m.state, o.state)
    p0 = {n: np.array(v, np.float64) for n, v in tree_flatten(ma.parameters())}
    p0b = {n: np.array(v, np.float64) for n, v in tree_flatten(mb_.parameters())}
    assert all(np.array_equal(p0[n], p0b[n]) for n in p0)  # cloned start

    def adam_state(o):
        st = dict(tree_flatten(o.state))
        return ({n: np.array(st[n + ".m"], np.float64) for n in p0},
                {n: np.array(st[n + ".v"], np.float64) for n in p0})

    m0, v0 = adam_state(oa)
    assert all(np.abs(m0[n]).max() > 0 for n in p0 if "weight" in n)
    mbx = _rand_mb(8, seed=2, ret_scale=10.0)
    for m, o, gs, ap, micro in ((ma, oa, gsa, apa, None),
                                (mb_, ob, gsb, apb, 3)):
        _, acc, ok = accumulate_minibatch(gs, mbx, micro, k)
        assert ok
        g, _ = finalize_grads(acc, None)
        mx.eval(g)
        gn = {n: np.array(v, np.float64) for n, v in tree_flatten(g)}
        ap(g)
        mx.eval(m.state, o.state)
        got = {n: np.array(v, np.float64) for n, v in tree_flatten(m.parameters())}
        for n in p0:
            mm = 0.9 * m0[n] + 0.1 * gn[n]
            vv = 0.999 * v0[n] + 0.001 * gn[n] ** 2
            exp = p0[n] - LR_ * mm / (np.sqrt(vv) + 1e-8)
            # float32 state/param arithmetic: abs error <= ~1e-7 on
            # parameters of magnitude <= ~1 (eps32 = 6e-8), rel 1e-5 on
            # the update itself for the sqrt/division chain.
            assert np.allclose(got[n], exp, rtol=1e-5, atol=2e-7), (micro, n)


@pytest.mark.parametrize("arm", ["gtm", "dial", "mlp", "nocomm"])
def test_accumulated_grad_matches_full_batch_all_arms(arm):
    # The acceptance quantity itself, before Adam's sign-like step:
    # accumulated vs full-batch gradient agree to float summation order,
    # with replayed NONZERO per-sample message noise (gumbel for GTM,
    # gaussian for DIAL), after plasticity scaling and after clipping.
    model = PairPolicy(arm, mx.random.key(1))
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=5e-4)
    opt.init(model.trainable_parameters())
    plastic = arm == "gtm"
    gs = _make_steps(model, opt, False, plastic=plastic)[0]
    mb = _rand_mb(8, seed=7, ret_scale=10.0)
    nz = model.sample_noise(8, mx.random.key(11))
    if model.noise_dim:
        assert nz is not None and float(mx.abs(nz).min().item()) > 0.0
        mb["noise"] = nz
    else:
        mb["noise"] = mx.zeros((8, 2, 0))
    k = mx.random.key(0)
    l_full, g_full, ok1 = accumulate_minibatch(gs, mb, None, k)
    l_mic, g_mic, ok2 = accumulate_minibatch(gs, mb, 3, k)  # 3 + 3 + 2
    assert ok1 and ok2 and np.isfinite(l_full) and np.isfinite(l_mic)
    assert l_mic == pytest.approx(l_full, rel=1e-5)
    assert np.isfinite(_gnorm(g_full)) and np.isfinite(_gnorm(g_mic))
    assert _max_rel(g_mic, g_full) < 1e-5
    scale = mx.array(0.37, dtype=mx.float32)
    c_full, n_full = finalize_grads(g_full, scale, plastic)
    c_mic, n_mic = finalize_grads(g_mic, scale, plastic)
    assert float(n_full.item()) > MAX_GRAD_NORM  # clipping is active
    assert float(n_mic.item()) == pytest.approx(float(n_full.item()), rel=1e-5)
    assert _max_rel(c_mic, c_full) < 1e-5


def test_uneven_last_minibatch_end_to_end(tmp_path):
    # exact-budget final rollout: 20 pair-steps = 2 full updates (8 pairs,
    # minibatches 3+3+2) + a truncated 3rd rollout (2 env steps = 4 pairs,
    # minibatches 3+1 with micro 2).
    kw = dict(total=20, mbp=3)
    _, ref, r_ref = _run_params(tmp_path, "mlp", None, "u_ref", **kw)
    _, acc, r_acc = _run_params(tmp_path, "mlp", 2, "u_acc", **kw)
    for r in (r_ref, r_acc):
        assert r.pair_steps_done == 20 and r.n_updates == 3
        assert r.training_failure is False
        assert r.optimizer_steps == 24  # 2 * (3 mb * 3 ep) + 2 mb * 3 ep
    for k in ref:
        assert np.abs(ref[k] - acc[k]).max() <= 24 * 2 * 5e-4 + 1e-6
