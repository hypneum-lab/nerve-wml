"""N10-C-bis trainer and evaluation on the scripted stub (tiny budgets):
``gtm_mis`` deranged delivery and replay, ``gtm_warm`` warm start, ``vq``
auxiliary loss, ``R_primary`` designation. No Procgen, no GPU."""
import hashlib
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from experiments.procgen_multiagent_gtm_comm.harness import bus as bus_mod  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness.bus import is_derangement  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness.codes import (  # noqa: E402
    ARM_CODES,
    BIS_ARMS,
    init_key_int,
)
from experiments.procgen_multiagent_gtm_comm.harness.evaluate import (  # noqa: E402
    evaluate_policy,
)
from experiments.procgen_multiagent_gtm_comm.harness.policy import (  # noqa: E402
    PairPolicy,
)
from experiments.procgen_multiagent_gtm_comm.harness.rollout import (  # noqa: E402
    deranged_recv,
    derangement,
)
from experiments.procgen_multiagent_gtm_comm.harness.train import (  # noqa: E402
    WARM_EXCLUDE_DEFAULT,
    _make_steps,
    accumulate_minibatch,
    load_warm_start,
    train_run,
)
from tests.unit.n10c.scripted_env import rand_factory  # noqa: E402

NP, H = 2, 4
PER_UPDATE = NP * H


def _run(tmp_path, arm, updates=1, *, seed=0, name="r", n_pairs=NP, **kw):
    per = n_pairs * H
    return train_run(
        arm, seed, "registered", updates * per, tmp_path / name,
        env_factory=rand_factory, n_pairs=n_pairs, horizon=H,
        log_every_pair_steps=per, ckpt_every_pair_steps=per, **kw)


def _mb(n, model, seed=0):
    r = np.random.default_rng(seed)
    mb = {
        "obs": mx.array(r.integers(0, 256, (n, 2, 64, 64, 3), dtype=np.uint8)),
        "summary": mx.array(r.normal(size=(n, 2, 2)).astype(np.float32)),
        "eng_bits": mx.array(r.integers(0, 2, (n, 2, 16)).astype(np.float32)),
        "actions": mx.array(r.integers(0, 15, (n, 2))),
        "logp": mx.array(-2.7 + 0.3 * r.normal(size=(n, 2)).astype(np.float32)),
        "advantages": mx.array(r.normal(size=(n, 2)).astype(np.float32)),
        "returns": mx.array(r.normal(size=(n, 2)).astype(np.float32)),
        "recv_bits": mx.array(r.integers(0, 2, (n, 2, 16)).astype(np.float32)),
    }
    nz = model.sample_noise(n, mx.random.key(5))
    mb["noise"] = nz if nz is not None else mx.zeros((n, 2, 0))
    return mb


def _grads(model, mb, *, plastic, use_recv=False):
    opt = optim.Adam(learning_rate=5e-4)
    opt.init(model.trainable_parameters())
    gs = _make_steps(model, opt, False, plastic=plastic, use_recv=use_recv)[0]
    loss, g, ok = accumulate_minibatch(gs, mb, None, mx.random.key(0))
    assert ok
    return loss, {k: np.array(v) for k, v in tree_flatten(g)}


# -- every control arm trains on the stub --------------------------------------
@pytest.mark.parametrize("arm", BIS_ARMS)
def test_bis_arms_train_with_ratio_one_and_full_bit_accounting(tmp_path, arm):
    rec = _run(tmp_path, arm, updates=2, name=arm)
    assert rec.training_failure is False and rec.pair_steps_done == 2 * PER_UPDATE
    assert rec.last_checkpoint == f"ckpt_{2 * PER_UPDATE}.safetensors"
    assert rec.codes == {"arm": ARM_CODES[arm], "condition": 0}
    assert len(rec.first_mb_ratio_dev) == 2 and max(rec.first_mb_ratio_dev) < 1e-4
    assert rec.bits_logged == 2 * PER_UPDATE * 2 * 16  # 16 bits per agent per step
    assert all(np.isfinite(rec.losses))
    data = json.loads((tmp_path / arm / "run.json").read_text())
    assert data["arm"] == arm
    assert (data["derangement_stream"] is not None) == (arm == "gtm_mis")
    assert data["warm_source_untrained"] == (arm == "gtm_warm")


# -- gtm_mis ----------------------------------------------------------------
def test_gtm_mis_delivers_deranged_messages_through_the_bus_and_replays_them(
        tmp_path, monkeypatch):
    seen = []
    orig = bus_mod.MessageBus.deliver

    def spy(self, bits, *, phase, pairs=None):
        seen.append((np.array(bits).copy(),
                     None if pairs is None else np.array(pairs).copy()))
        return orig(self, bits, phase=phase, pairs=pairs)

    monkeypatch.setattr(bus_mod.MessageBus, "deliver", spy)
    bufs = {}
    n = 3  # two possible derangements of three pairs
    rec = _run(tmp_path, "gtm_mis", updates=2, n_pairs=n,
               _rollout_hook=lambda u, buf: bufs.setdefault(u, buf.recv_bits.copy()))
    assert rec.training_failure is False
    assert len(seen) == 2 * H  # one bus delivery per acted step (bootstrap: none)
    for bits, sigma in seen:
        assert sigma is not None and is_derangement(sigma, n)
        assert bits.shape == (n, 2, 16)
    for u in (1, 2):  # one derangement per rollout
        assert len({tuple(s) for _, s in seen[(u - 1) * H:u * H]}) == 1
        for t in range(H):
            bits, sigma = seen[(u - 1) * H + t]
            np.testing.assert_array_equal(bufs[u][t], deranged_recv(bits, sigma))
            for e in range(n):  # agent i of pair e got role 1 - i of pair sigma(e)
                np.testing.assert_array_equal(bufs[u][t][e, 0], bits[sigma[e], 1])
                np.testing.assert_array_equal(bufs[u][t][e, 1], bits[sigma[e], 0])
    assert rec.bits_logged == 2 * H * n * 2 * 16
    assert rec.derangement_stream == {
        "seed_sequence": [0, ARM_CODES["gtm_mis"], 0], "spawn_index": 4,
        "draws": "one derangement of the pairs per rollout"}
    assert max(rec.first_mb_ratio_dev) < 1e-4  # replay: PPO ratio is 1


def test_gtm_mis_rollout_is_deterministic_from_the_seed_stream(tmp_path):
    acts = {}

    def grab(name):
        return lambda u, buf: acts.setdefault(name, (buf.actions.copy(), buf.recv_bits.copy()))

    _run(tmp_path, "gtm_mis", seed=3, name="a", _rollout_hook=grab("a"))
    _run(tmp_path, "gtm_mis", seed=3, name="b", _rollout_hook=grab("b"))
    _run(tmp_path, "gtm_mis", seed=4, name="c", _rollout_hook=grab("c"))
    assert np.array_equal(acts["a"][0], acts["b"][0])
    assert np.array_equal(acts["a"][1], acts["b"][1])
    assert not np.array_equal(acts["a"][0], acts["c"][0])


def test_gtm_mis_sender_head_gets_no_gradient_receiver_and_trunk_do():
    model = PairPolicy("gtm_mis", mx.random.key(1))
    mb = _mb(8, model)
    _, g = _grads(model, mb, plastic=True, use_recv=True)
    assert np.abs(g["comm.encode_logits.weight"]).max() == 0.0  # disclosed
    assert np.abs(g["comm.encode_logits.bias"]).max() == 0.0
    assert np.abs(g["to_msg.weight"]).max() == 0.0  # feeds the sender only
    assert np.abs(g["comm.constellation"]).max() > 0.0
    assert np.abs(g["comm.decode.weight"]).max() > 0.0
    assert np.abs(g["trunk.fc.weight"]).max() > 0.0 and np.abs(g["pi.weight"]).max() > 0.0
    # contrast: the gtm arm (partner swap, ST one-hot) trains its sender
    ref = PairPolicy("gtm", mx.random.key(1))
    _, g_ref = _grads(ref, mb, plastic=True)
    assert np.abs(g_ref["comm.encode_logits.weight"]).max() > 0.0


def test_encode_then_heads_equals_call_for_replay_and_partner_swap():
    model = PairPolicy("gtm_mis", mx.random.key(2))
    rng = np.random.default_rng(0)
    obs = mx.array(rng.integers(0, 256, (4, 2, 64, 64, 3)).astype(np.float32))
    summary = mx.array(rng.random((4, 2, 2)).astype(np.float32))
    k = mx.random.key(3)
    nz = model.sample_noise(4, mx.random.key(4))
    enc = model.encode(obs, None, train=True, key=k, noise=nz)
    mx.eval(enc.hard_bits)
    sigma = derangement(4, np.random.default_rng(1))
    recv = mx.array(deranged_recv(np.array(enc.hard_bits), sigma))
    a = model.heads(enc, summary, recv, train=True)
    b = model(obs, summary, None, train=True, key=k, noise=nz, recv_bits=recv)
    for name in ("logits", "value", "hard_bits"):
        np.testing.assert_allclose(np.array(getattr(a, name)),
                                   np.array(getattr(b, name)), atol=1e-6)
    msg_p = enc.msg.reshape(4, 2, -1)
    swap = mx.stack([msg_p[:, 1], msg_p[:, 0]], axis=1)
    c = model.heads(enc, summary, swap, train=True)
    d = model(obs, summary, None, train=True, key=k, noise=nz)
    np.testing.assert_allclose(np.array(c.logits), np.array(d.logits), atol=1e-6)
    # deranged delivery changes what the receiver sees (the test is live)
    assert not np.allclose(np.array(a.logits), np.array(d.logits), atol=1e-6)


# -- gtm_warm ---------------------------------------------------------------
def test_warm_start_loads_everything_but_the_excluded_parameters(tmp_path):
    src = _run(tmp_path, "eng_gtmrx", seed=5, name="src")
    ck = tmp_path / "src" / src.last_checkpoint
    ckd = {k: np.array(v) for k, v in mx.load(str(ck)).items()}
    model = PairPolicy("gtm_warm", mx.random.key(init_key_int(5, "gtm_warm", "registered")))
    fresh = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    info = load_warm_start(model, ck)
    after = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    excluded = {"comm.encode_logits.weight", "comm.encode_logits.bias",
                "to_msg.weight", "to_msg.bias"}
    assert set(info["excluded"]) == excluded
    assert info["exclude_prefixes"] == list(WARM_EXCLUDE_DEFAULT)
    assert info["loaded"] == len(after) - len(excluded) and info["source_pair_steps"] == 8
    assert set(after) == set(ckd)
    for k in after:
        if k in excluded:  # the warm run's own seed-stream initialisation
            assert np.array_equal(after[k], fresh[k]), k
            assert not np.array_equal(after[k], ckd[k]), k
        else:  # bit-identical to the source checkpoint
            assert np.array_equal(after[k], ckd[k]), k
    assert any(not np.array_equal(fresh[k], ckd[k]) for k in after if k not in excluded)
    with pytest.raises(ValueError, match="does not match"):
        load_warm_start(PairPolicy("vq", mx.random.key(0)), ck)
    with pytest.raises(ValueError, match="warm_start_from"):
        _run(tmp_path, "gtm", name="not_warm", warm_start_from=ck)


def test_gtm_warm_run_records_source_digest_and_cumulative_budget(tmp_path):
    src = _run(tmp_path, "eng_gtmrx", updates=2, seed=3, name="src")
    ck = tmp_path / "src" / src.last_checkpoint
    assert ck.name == "ckpt_16.safetensors"
    rec = _run(tmp_path, "gtm_warm", seed=3, name="warm", warm_start_from=ck)
    assert rec.training_failure is False and rec.warm_source_untrained is False
    assert rec.warm_from["file"] == "ckpt_16.safetensors"
    assert rec.warm_from["source_pair_steps"] == 16
    assert rec.warm_ckpt_sha256 == hashlib.sha256(ck.read_bytes()).hexdigest()
    assert rec.cumulative_pair_steps == 16 + PER_UPDATE  # source + own budget
    data = json.loads((tmp_path / "warm" / "run.json").read_text())
    for key in ("warm_from", "warm_ckpt_sha256", "cumulative_pair_steps",
                "warm_source_untrained"):
        assert key in data
    assert data["warm_from"]["path"] == str(ck)
    # no source (failed eng_gtmrx without a finite checkpoint): initialisation,
    # flagged and kept
    cold = _run(tmp_path, "gtm_warm", seed=3, name="cold")
    assert cold.warm_source_untrained is True and cold.warm_from is None
    assert cold.warm_ckpt_sha256 is None and cold.cumulative_pair_steps == PER_UPDATE


# -- vq -----------------------------------------------------------------------
def test_vq_aux_loss_is_added_to_the_ppo_loss_and_trains_the_codebook():
    model = PairPolicy("vq", mx.random.key(1))
    mb = _mb(6, model)
    loss, g = _grads(model, mb, plastic=False)
    assert np.abs(g["comm.codebook"]).max() > 0.0  # only reachable via aux_loss
    assert np.abs(g["comm.encode.weight"]).max() > 0.0
    # the same model with the auxiliary term removed: the loss drops by it
    enc = model.encode(mb["obs"].astype(mx.float32), None, train=True,
                       key=mx.random.key(0))
    aux = float(model.comm.aux_loss(enc.x_msg))
    assert aux > 0.0
    plain = PairPolicy("vq", mx.random.key(1))
    plain.comm.aux_loss = lambda x: mx.array(0.0, dtype=mx.float32)
    loss0, g0 = _grads(plain, mb, plastic=False)
    assert loss - loss0 == pytest.approx(aux, rel=1e-4, abs=1e-6)
    assert np.abs(g0["comm.codebook"]).max() == 0.0


# -- evaluation designations -------------------------------------------------
class _Scripted:
    def __init__(self, c0: int) -> None:
        self.c0, self.t = c0, -1

    def act(self, actions) -> None:
        self.t += 1

    def observe(self):
        ended = self.t == self.c0
        img = np.full((1, 64, 64, 3), (7 * self.t + 3) % 251, dtype=np.uint8)
        return np.array([10.0 if ended else 0.0]), {"rgb": img}, np.array([ended])

    def close(self) -> None:
        pass


def _factory(c0s):
    made = []

    def f(rand_seed: int):
        env = _Scripted(c0s[(len(made) // 2) % len(c0s)])
        made.append(env)
        return env

    return f


@pytest.mark.parametrize("arm", BIS_ARMS)
def test_bis_arms_evaluate_with_the_registered_designations(arm):
    model = PairPolicy(arm, mx.random.key(init_key_int(0, arm, "registered")))
    rec = evaluate_policy(model, arm, "registered", range(3),
                          env_factory=_factory([0, 2, 1]))
    assert len(rec.returns_intact) == len(rec.returns_shuffled) == 3
    assert 0.0 <= rec.bandwidth_bits <= 16.0
    if arm == "gtm_mis":  # the shuffled pass is the in-distribution pass (R_MIS)
        assert rec.primary_pass == "shuffled" and rec.R_primary == rec.R_shuffled
    else:
        assert rec.primary_pass == "intact" and rec.R_primary == rec.R_intact
    if arm == "vq":
        assert len(rec.codebook_usage) == 16
        assert sum(rec.codebook_usage) == 4 * 2 * sum(rec.lengths_intact)
    else:
        assert rec.codebook_usage is None
    d = json.loads(rec.to_json())
    assert d["R_primary"] == rec.R_primary and d["primary_pass"] == rec.primary_pass
    assert d["S"] == rec.R_intact - rec.R_shuffled


# -- alignment with fix C (amendment 2026-10-25, PI decision 19:50) ----------
def _trunk_file(path):
    m = PairPolicy("gtm", mx.random.key(11))
    mx.eval(m.parameters())
    mx.save_safetensors(str(path), dict(tree_flatten(m.trunk.parameters())))
    return path


@pytest.mark.parametrize("arm", BIS_ARMS)
def test_bis_arms_train_under_fix_c_from_the_seed_trunk(tmp_path, arm):
    from experiments.procgen_multiagent_gtm_comm.harness.options import PRESETS

    trunk = _trunk_file(tmp_path / "trunk.safetensors")
    rec = _run(tmp_path, arm, updates=2, name=arm,
               options=PRESETS["fix-c-coin-pretrain"], init_trunk=trunk)
    assert rec.training_failure is False and not rec.liveness_abort
    assert rec.train_options["preset"] == "fix-c-coin-pretrain"
    assert rec.init_trunk_sha256 == hashlib.sha256(trunk.read_bytes()).hexdigest()
    assert rec.trunk_mode == "shared"
    assert rec.liveness and rec.liveness[0]["step"] == 0
    assert max(rec.first_mb_ratio_dev) < 1e-4


def test_gtm_warm_under_fix_c_records_the_warm_source_trunk(tmp_path):
    from experiments.procgen_multiagent_gtm_comm.harness.options import PRESETS

    opts = PRESETS["fix-c-coin-pretrain"]
    trunk = _trunk_file(tmp_path / "trunk.safetensors")
    src = _run(tmp_path, "eng_gtmrx", updates=2, seed=3, name="src",
               options=opts, init_trunk=trunk)
    ck = tmp_path / "src" / src.last_checkpoint
    warm = _run(tmp_path, "gtm_warm", seed=3, name="warm", options=opts,
                init_trunk=trunk, warm_start_from=ck)
    assert warm.trunk_mode == "warm_source" and warm.warm_from is not None
    cold = _run(tmp_path, "gtm_warm", seed=3, name="cold", options=opts,
                init_trunk=trunk)
    assert cold.trunk_mode == "shared" and cold.warm_source_untrained


def test_vq_aux_loss_is_masked_for_finished_agents_under_fix_c():
    """PI decision 2026-10-26: with mask_finished the VQ loss of finished
    senders is excluded (masked mean over the full minibatch), as the
    policy and entropy terms; the frozen expression is unchanged."""
    model = PairPolicy("vq", mx.random.key(1))
    mb = _mb(6, model)
    enc = model.encode(mb["obs"].astype(mx.float32), None, train=True,
                       key=mx.random.key(0))
    rows = model.comm.aux_loss_rows(enc.x_msg)
    assert float(rows.mean()) == pytest.approx(
        float(model.comm.aux_loss(enc.x_msg)), rel=1e-6)

    def codebook_grad(mask, live):
        opt = optim.Adam(learning_rate=5e-4)
        opt.init(model.trainable_parameters())
        gs = _make_steps(model, opt, False, plastic=False, mask_finished=mask)[0]
        m = dict(mb, live=mx.array(live.astype(np.float32)))
        _, g, ok = accumulate_minibatch(gs, m, None, mx.random.key(0), mask=mask)
        assert ok
        return np.abs(np.array(dict(tree_flatten(g))["comm.codebook"])).max()

    none_live = np.zeros((6, 2))
    assert codebook_grad(True, none_live) == 0.0  # every sender finished
    assert codebook_grad(True, np.ones((6, 2))) > 0.0
    assert codebook_grad(False, none_live) > 0.0  # frozen path: unmasked


def test_vq_mask_is_per_agent_row_and_micro_batch_invariant():
    model = PairPolicy("vq", mx.random.key(2))
    mb = _mb(6, model, seed=3)
    live = np.ones((6, 2), np.float32)
    live[:, 1] = 0.0  # agent 1 finished in every pair
    live[4, 0] = 0.0  # and agent 0 of pair 4

    def grads(m, micro):
        opt = optim.Adam(learning_rate=5e-4)
        opt.init(model.trainable_parameters())
        gs = _make_steps(model, opt, False, plastic=False, mask_finished=True)[0]
        _, g, ok = accumulate_minibatch(gs, m, micro, mx.random.key(0), mask=True)
        assert ok
        return {k: np.array(v) for k, v in tree_flatten(g)}

    m = dict(mb, live=mx.array(live))
    full, micro = grads(m, None), grads(m, 2)  # 2 + 2 + 2 pairs
    for k in full:
        np.testing.assert_allclose(micro[k], full[k], rtol=2e-4, atol=1e-7)
    # the codebook gets gradient only from the (masked) VQ loss: changing the
    # observations of finished rows leaves it unchanged
    obs = np.array(mb["obs"])
    obs[:, 1] = 255 - obs[:, 1]
    obs[4, 0] = 255 - obs[4, 0]
    other = grads(dict(m, obs=mx.array(obs)), None)
    np.testing.assert_allclose(other["comm.codebook"], full["comm.codebook"],
                               rtol=1e-5, atol=1e-8)
    assert np.abs(full["comm.codebook"]).max() > 0.0
