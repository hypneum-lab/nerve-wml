"""Evaluation / shuffle-probe tests: scripted envs only (never held-out)."""
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.harness.codes import (  # noqa: E402
    init_key_int,
)
from experiments.procgen_multiagent_gtm_comm.harness.evaluate import (  # noqa: E402
    bandwidth_entropy,
    derangement,
    donor_received,
    evaluate_model,
    evaluate_policy,
    level_rng,
)
from experiments.procgen_multiagent_gtm_comm.harness.policy import (  # noqa: E402
    PairPolicy,
)


class _Scripted:
    """Both agents reach their coin at step c0 (team success
    at step c0) unless ``no_coin``: then it finishes at step 0 w/o coin."""

    def __init__(self, c0: int, no_coin: bool, log: list) -> None:
        self.c0, self.no_coin, self.t, self.acts = c0, no_coin, -1, []
        log.append(self)

    def act(self, actions) -> None:
        self.t += 1
        self.acts.append(int(actions[0]))

    def observe(self):
        if self.no_coin:
            ended = self.t == 0
            r = 0.0
        else:
            ended = self.t == self.c0
            r = 10.0 if ended else 0.0
        img = np.full((1, 64, 64, 3), (7 * self.t + 3) % 251, dtype=np.uint8)
        return np.array([r]), {"rgb": img}, np.array([ended])

    def close(self) -> None:
        pass


def _factory(c0s, *, no_coin=False):
    """Env k of a pass: level index (k // 2) % len(c0s); also logs envs."""
    log: list[_Scripted] = []

    def f(rand_seed: int):
        idx = (len(log) // 2) % len(c0s)
        return _Scripted(c0s[idx], no_coin, log)

    f.log = log
    return f


def _model(arm="mlp", seed=0):
    return PairPolicy(arm, mx.random.key(init_key_int(seed, arm, "registered")))


def test_derangement_has_no_fixed_points_and_is_deterministic():
    a = derangement(200, np.random.default_rng(20261023))
    b = derangement(200, np.random.default_rng(20261023))
    np.testing.assert_array_equal(a, b)
    assert sorted(a.tolist()) == list(range(200))
    assert not (a == np.arange(200)).any()
    with pytest.raises(ValueError):
        derangement(1, np.random.default_rng(0))


def test_level_rng_is_registered_stream():
    np.testing.assert_array_equal(
        level_rng(1003).random(5),
        np.random.default_rng([20261023, 1003]).random(5),
    )


def test_donor_received_uses_last_message_when_donor_shorter():
    donor = np.arange(3 * 2 * 16, dtype=np.float32).reshape(3, 2, 16)  # T=3
    # agent i receives the *other* role's message from the donor
    np.testing.assert_array_equal(donor_received(donor, 1)[0], donor[1, 1])
    np.testing.assert_array_equal(donor_received(donor, 1)[1], donor[1, 0])
    for t in (2, 3, 10):  # donor has T=3: clamp to its last step
        np.testing.assert_array_equal(donor_received(donor, t), donor_received(donor, 2))
        np.testing.assert_array_equal(donor_received(donor, t)[0], donor[2, 1])


def test_shuffle_delivers_donor_messages_end_to_end():
    c0s = [0, 2, 5, 1]  # episode lengths 1, 3, 6, 2
    n = len(c0s)
    model = _model("mlp")
    calls = []

    class Spy:
        def __call__(self, *a, **kw):
            out = model(*a, **kw)
            calls.append((kw.get("recv_bits"), np.array(out.hard_bits)))
            return out

    rec = evaluate_policy(Spy(), "mlp", "registered", range(n),
                          env_factory=_factory(c0s))
    lens = rec.lengths_intact
    assert lens == [c + 1 for c in c0s]
    n_intact = sum(lens)
    assert all(c[0] is None for c in calls[:n_intact])
    intact, off = [], 0
    for length in lens:
        intact.append(np.stack([c[1][0] for c in calls[off:off + length]]))
        off += length
    pi = derangement(n, np.random.default_rng(20261023))
    k = n_intact
    for j in range(n):
        for t in range(rec.lengths_shuffled[j]):
            recv = np.array(calls[k][0])[0]
            exp = donor_received(intact[pi[j]], t)
            np.testing.assert_array_equal(recv, exp)
            k += 1
    assert k == len(calls)
    assert rec.S == rec.R_intact - rec.R_shuffled


def test_nocomm_s_is_zero():
    f = _factory([0, 2, 5, 1])
    rec = evaluate_policy(_model("nocomm"), "nocomm", "registered", range(4),
                          env_factory=f)
    assert rec.S == 0.0
    assert rec.returns_intact == rec.returns_shuffled == [1.0] * 4
    assert rec.lengths_intact == rec.lengths_shuffled
    assert rec.bandwidth_bits == 0.0


def test_eval_rng_identical_across_passes():
    c0s = [0, 2, 5, 1]
    f = _factory(c0s)
    evaluate_policy(_model("nocomm"), "nocomm", "registered", range(4),
                    env_factory=f)
    envs = f.log
    assert len(envs) == 2 * 2 * 4  # 4 levels x 2 agents x 2 passes
    for j in range(4):
        for agent in range(2):
            a = envs[2 * j + agent].acts
            b = envs[2 * 4 + 2 * j + agent].acts
            assert a == b and len(a) == c0s[j] + 1


def test_completion_time_none_without_success():
    rec = evaluate_policy(_model("mlp"), "mlp", "registered", range(3),
                          env_factory=_factory([0, 0, 0], no_coin=True))
    assert rec.R_intact == 0.0 and rec.completion_time is None
    assert rec.lengths_intact == [1, 1, 1]
    rec2 = evaluate_policy(_model("mlp"), "mlp", "registered", range(3),
                           env_factory=_factory([0, 2, 5]))
    assert rec2.R_intact == 1.0
    assert rec2.completion_time == pytest.approx(np.mean([1, 3, 6]))


def test_bandwidth_entropy_bounds():
    const = np.zeros((10, 2, 16), np.float32)
    assert bandwidth_entropy(const) == 0.0
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 2, size=(4000, 2, 16)).astype(np.float32)
    h = bandwidth_entropy(rand)
    assert 0.0 < h <= 16.0 and h > 11.0
    two = np.zeros((4, 1, 16), np.float32)
    two[:2, 0, 0] = 1.0  # two equiprobable messages -> exactly 1 bit
    assert bandwidth_entropy(two) == pytest.approx(1.0)


def test_untrained_when_checkpoint_none_and_roundtrip(tmp_path):
    f = _factory([0, 2])
    rec = evaluate_model(None, "mlp", "registered", range(2), seed=17,
                         env_factory=f)
    assert rec.untrained is True and rec.ckpt is None
    # untrained init is exactly the run's seed-stream init
    m = _model("mlp", seed=17)
    path = tmp_path / "ckpt_1.safetensors"
    from mlx.utils import tree_flatten
    mx.save_safetensors(str(path), dict(tree_flatten(m.parameters())))
    rec2 = evaluate_model(path, "mlp", "registered", range(2), seed=17,
                          env_factory=_factory([0, 2]))
    assert rec2.untrained is False
    assert rec2.returns_intact == rec.returns_intact
    assert rec2.bandwidth_bits == rec.bandwidth_bits
    d = json.loads(rec2.to_json())
    assert d["S"] == rec2.S and d["levels"] == [0, 2]


@pytest.mark.parametrize("arm", ["gtm", "dial", "atoc", "eng"])
def test_all_arms_evaluate(arm):
    rec = evaluate_policy(_model(arm), arm, "closed", range(3),
                          env_factory=_factory([0, 2, 1]))
    assert len(rec.returns_intact) == 3 and 0.0 <= rec.bandwidth_bits <= 16.0
