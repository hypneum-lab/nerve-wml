from dataclasses import dataclass, field

import pytest

from nerve_core.invariants import (
    assert_n1_silence_legal,
    assert_n2_send_idempotent,
    assert_n3_role_phase_consistent,
    assert_n4_routing_weight_valid,
    assert_n5_codebook_owned,
    assert_w1_step_uses_nerve_only,
    assert_w2_parameters_include_codebook,
    assert_w3_no_routing_access_in_step,
    assert_w4_id_stable,
)
from nerve_core.neuroletter import Neuroletter, Phase, Role


def test_n1_empty_listen_is_legal():
    assert_n1_silence_legal([])  # does not raise


def test_n3_prediction_must_be_gamma_in_strict_mode():
    ok = Neuroletter(5, Role.PREDICTION, Phase.GAMMA, 1, 2, 0.0)
    assert_n3_role_phase_consistent(ok, strict=True)

    bad = Neuroletter(5, Role.PREDICTION, Phase.THETA, 1, 2, 0.0)
    with pytest.raises(AssertionError, match="N-3"):
        assert_n3_role_phase_consistent(bad, strict=True)


def test_n3_error_must_be_theta_in_strict_mode():
    bad = Neuroletter(5, Role.ERROR, Phase.GAMMA, 1, 2, 0.0)
    with pytest.raises(AssertionError, match="N-3"):
        assert_n3_role_phase_consistent(bad, strict=True)


def test_n4_routing_weight_range():
    assert_n4_routing_weight_valid(0.0, pruned=True)
    assert_n4_routing_weight_valid(1.0, pruned=True)
    assert_n4_routing_weight_valid(0.42, pruned=False)  # continuous during training
    with pytest.raises(AssertionError, match="N-4"):
        assert_n4_routing_weight_valid(0.5, pruned=True)  # must be {0, 1} once pruned


# ---------------------------------------------------------------------------
# N-2 — send() idempotency on (src, dst, code, timestamp)
# ---------------------------------------------------------------------------


def test_n2_unique_sends_pass():
    sent = [
        Neuroletter(1, Role.PREDICTION, Phase.GAMMA, 0, 1, 0.0),
        Neuroletter(2, Role.PREDICTION, Phase.GAMMA, 0, 1, 0.1),
        Neuroletter(1, Role.PREDICTION, Phase.GAMMA, 0, 2, 0.0),
    ]
    assert_n2_send_idempotent(sent)


def test_n2_duplicate_send_raises():
    a = Neuroletter(7, Role.PREDICTION, Phase.GAMMA, 0, 1, 0.5)
    b = Neuroletter(7, Role.PREDICTION, Phase.GAMMA, 0, 1, 0.5)
    with pytest.raises(AssertionError, match="N-2"):
        assert_n2_send_idempotent([a, b])


# ---------------------------------------------------------------------------
# N-5 — each WML owns its codebook (no shared object identity)
# ---------------------------------------------------------------------------


@dataclass
class _StubWML:
    id: int
    codebook: object
    _params: list = field(default_factory=list)
    _mutation_log: list = field(default_factory=list)

    def parameters(self):
        return list(self._params)


def test_n5_distinct_codebooks_pass():
    a = _StubWML(id=0, codebook=object())
    b = _StubWML(id=1, codebook=object())
    assert_n5_codebook_owned(a, [a, b])


def test_n5_shared_codebook_raises():
    shared = object()
    a = _StubWML(id=0, codebook=shared)
    b = _StubWML(id=1, codebook=shared)
    with pytest.raises(AssertionError, match="N-5"):
        assert_n5_codebook_owned(a, [b])


# ---------------------------------------------------------------------------
# W-1 — step() must not mutate other WMLs
# ---------------------------------------------------------------------------


def test_w1_no_foreign_mutation_passes():
    a = _StubWML(id=0, codebook=object())
    b = _StubWML(id=1, codebook=object())
    # Foreign mutation by a non-actor is fine for W-1 from a's perspective.
    b._mutation_log.append((99, "set_state"))
    assert_w1_step_uses_nerve_only(a, [b])


def test_w1_foreign_mutation_by_actor_raises():
    a = _StubWML(id=0, codebook=object())
    b = _StubWML(id=1, codebook=object())
    b._mutation_log.append((0, "overwrite_codebook"))
    with pytest.raises(AssertionError, match="W-1"):
        assert_w1_step_uses_nerve_only(a, [b])


# ---------------------------------------------------------------------------
# W-2 — parameters() includes codebook
# ---------------------------------------------------------------------------


def test_w2_codebook_in_parameters_passes():
    cb = object()
    wml = _StubWML(id=0, codebook=cb, _params=[cb, object()])
    assert_w2_parameters_include_codebook(wml)


def test_w2_codebook_missing_raises():
    cb = object()
    other = object()
    wml = _StubWML(id=0, codebook=cb, _params=[other])
    with pytest.raises(AssertionError, match="W-2"):
        assert_w2_parameters_include_codebook(wml)


# ---------------------------------------------------------------------------
# W-3 — step() must not access routing_weight
# ---------------------------------------------------------------------------


def test_w3_clean_access_log_passes():
    assert_w3_no_routing_access_in_step(["nerve.listen", "nerve.send"])


def test_w3_routing_weight_access_raises():
    with pytest.raises(AssertionError, match="W-3"):
        assert_w3_no_routing_access_in_step(["nerve.routing_weight(0,1)"])


# ---------------------------------------------------------------------------
# W-4 — id is stable across a WML's lifetime
# ---------------------------------------------------------------------------


def test_w4_stable_id_passes():
    wml = _StubWML(id=7, codebook=object())
    assert_w4_id_stable(wml, expected_id=7)


def test_w4_id_drift_raises():
    wml = _StubWML(id=7, codebook=object())
    wml.id = 8  # drift
    with pytest.raises(AssertionError, match="W-4"):
        assert_w4_id_stable(wml, expected_id=7)
