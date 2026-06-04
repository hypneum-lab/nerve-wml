"""Runtime guards for the N-1..N-5 and W-1..W-4 invariants in the spec §4.5.

These are assert_*() helpers — cheap in prod, strict in tests.

Canonical definitions (from `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`):
- N-1: `len(listen(wml_id)) == 0` is valid (silence is legitimate).
- N-2: `send()` is idempotent on `(src, dst, code, timestamp)`.
- N-3: `role == ERROR` ⟺ `phase == THETA` in strict mode.
- N-4: `routing_weight(i, j) ∈ {0, 1}` post-pruning ; continuous Gumbel during training.
- N-5: Each WML owns its codebook ; cross-codebook mapping via transducers.
- W-1: `step()` never mutates another WML — isolation via Nerve only.
- W-2: `parameters()` includes `codebook` and all internal weights.
- W-3: No access to `routing_weight` from inside `step()` — router is global (Track-P).
- W-4: `id` is stable across a WML's lifetime — Nerves index by id.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .neuroletter import Neuroletter, Phase, Role


def assert_n1_silence_legal(inbound: list[Neuroletter]) -> None:
    """N-1: listen() returning [] is always valid (silence is information)."""
    # Nothing to assert — the contract is that empty is fine.
    return


def assert_n2_send_idempotent(sent: Iterable[Neuroletter]) -> None:
    """N-2: send() is idempotent on `(src, dst, code, timestamp)`.

    Given the full transcript of letters sent on a nerve, no two letters may
    share the same `(src, dst, code, timestamp)` key. Required for bit-stable
    golden tests.
    """
    seen: set[tuple[int, int, int, float]] = set()
    for letter in sent:
        key = (letter.src, letter.dst, letter.code, letter.timestamp)
        if key in seen:
            raise AssertionError(
                f"N-2 violated: duplicate send key {key!r} — "
                "send() must be idempotent on (src, dst, code, timestamp)"
            )
        seen.add(key)


def assert_n3_role_phase_consistent(
    letter: Neuroletter,
    *,
    strict: bool = True,
) -> None:
    """N-3 strict mode: PREDICTION↔GAMMA and ERROR↔THETA."""
    if not strict:
        return
    expected_phase = Phase.GAMMA if letter.role is Role.PREDICTION else Phase.THETA
    assert letter.phase is expected_phase, (
        f"N-3 violated: role={letter.role.name} with phase={letter.phase.name} "
        f"(expected {expected_phase.name} in strict mode)"
    )


def assert_n4_routing_weight_valid(weight: float, *, pruned: bool) -> None:
    """N-4: post-pruning weights are {0, 1}; pre-pruning they are continuous in [0, 1]."""
    if pruned:
        assert weight in (0.0, 1.0), (
            f"N-4 violated: post-pruning weight must be 0 or 1, got {weight}"
        )
    else:
        assert 0.0 <= weight <= 1.0, (
            f"N-4 violated: pre-pruning weight must be in [0, 1], got {weight}"
        )


def assert_n5_codebook_owned(wml: Any, other_wmls: Iterable[Any]) -> None:
    """N-5: each WML owns its codebook — codebooks are not shared object-identity-wise.

    Cross-codebook mapping must go through transducers, not via aliasing the
    same tensor across WMLs.
    """
    own_cb = getattr(wml, "codebook", None)
    if own_cb is None:
        raise AssertionError(
            f"N-5 violated: wml {wml!r} has no `codebook` attribute"
        )
    for other in other_wmls:
        if other is wml:
            continue
        other_cb = getattr(other, "codebook", None)
        if other_cb is None:
            continue
        if other_cb is own_cb:
            raise AssertionError(
                f"N-5 violated: wml id={getattr(wml, 'id', '?')} and "
                f"id={getattr(other, 'id', '?')} share the same codebook object — "
                "each WML must own its codebook (use a transducer for cross-mapping)"
            )


def assert_w1_step_uses_nerve_only(wml: Any, foreign_wmls: Iterable[Any]) -> None:
    """W-1: `step()` must never mutate another WML — isolation via Nerve only.

    Convention: foreign WMLs may expose a `_mutation_log` list of `(actor_id, op)`
    tuples. We assert no entry was authored by `wml.id`.
    """
    actor_id = getattr(wml, "id", None)
    for other in foreign_wmls:
        if other is wml:
            continue
        log = getattr(other, "_mutation_log", None)
        if log is None:
            continue
        for entry in log:
            actor = entry[0] if isinstance(entry, tuple) and entry else None
            if actor == actor_id:
                raise AssertionError(
                    f"W-1 violated: wml id={actor_id} mutated foreign wml "
                    f"id={getattr(other, 'id', '?')} with op {entry!r} — "
                    "step() must communicate exclusively via the Nerve"
                )


def assert_w2_parameters_include_codebook(wml: Any) -> None:
    """W-2: `parameters()` must include the codebook and all internal weights.

    The VQ commitment loss has to flow through the codebook; if the codebook is
    not in the parameter set, gradients are silently dropped.
    """
    codebook = getattr(wml, "codebook", None)
    if codebook is None:
        raise AssertionError(
            f"W-2 violated: wml {wml!r} has no `codebook` attribute"
        )
    params = getattr(wml, "parameters", None)
    if params is None or not callable(params):
        raise AssertionError(
            f"W-2 violated: wml {wml!r} has no callable `parameters()`"
        )
    param_ids = {id(p) for p in params()}
    if id(codebook) not in param_ids:
        raise AssertionError(
            f"W-2 violated: wml id={getattr(wml, 'id', '?')} parameters() does not "
            "include the codebook — VQ commitment loss will not backpropagate"
        )


def assert_w3_no_routing_access_in_step(step_access_log: Iterable[str]) -> None:
    """W-3: `step()` must not access `routing_weight` — router is global (Track-P).

    Convention: instrument `step()` to record a string log of attribute paths it
    accessed on the nerve / router. Any entry containing `routing_weight` is a
    violation.
    """
    for path in step_access_log:
        if "routing_weight" in path:
            raise AssertionError(
                f"W-3 violated: step() accessed {path!r} — router is global, "
                "owned by Track-P; WML.step() must not query routing_weight"
            )


def assert_w4_id_stable(wml: Any, expected_id: int) -> None:
    """W-4: `id` is stable across a WML's lifetime — Nerves index by id.

    A drift between the WML's current `id` and its registered id breaks every
    nerve route that targets it.
    """
    current = getattr(wml, "id", None)
    if current is None:
        raise AssertionError(
            f"W-4 violated: wml {wml!r} has no `id` attribute"
        )
    if current != expected_id:
        raise AssertionError(
            f"W-4 violated: wml id drifted from {expected_id!r} to {current!r} — "
            "ids must be stable for the lifetime of the WML"
        )
