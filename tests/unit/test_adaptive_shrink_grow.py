"""Tests for AdaptiveCodebook.shrink and .grow."""

from track_p.adaptive_codebook import AdaptiveCodebook


def test_shrink_retires_unused_codes():
    cb = AdaptiveCodebook(size=16, dim=8)
    # Simulate usage: only codes 0..9 saw samples.
    cb.storage.usage_counter[:10] = 100
    cb.storage.usage_counter[10:] = 0

    assert cb.current_size() == 16
    remaining = cb.shrink(min_usage_frac=0.01, min_codes=8)
    assert cb.current_size() == 10
    assert len(remaining) == 10


def test_shrink_honours_min_codes_floor():
    cb = AdaptiveCodebook(size=16, dim=8)
    # Everything below threshold → would retire all.
    cb.storage.usage_counter.zero_()
    cb.storage.usage_counter[0] = 1000  # one heavy user

    cb.shrink(min_usage_frac=0.5, min_codes=4)
    # Floor keeps at least 4 codes alive.
    assert cb.current_size() >= 4


def test_shrink_noop_when_no_usage():
    cb = AdaptiveCodebook(size=16, dim=8)
    # No usage recorded.
    before = cb.current_size()
    cb.shrink(min_usage_frac=0.01)
    assert cb.current_size() == before


def test_grow_seeds_vacant_slots_from_hot_parents():
    cb = AdaptiveCodebook(size=16, dim=8)
    # Retire half so there's vacancy.
    cb.active_mask[12:] = False
    cb.storage.usage_counter[:12] = 10
    cb.storage.usage_counter[0] = 100  # hot parent

    new_indices = cb.grow(top_k_to_split=2, perturb_scale=0.01, seed=0)
    assert len(new_indices) == 2
    # Newly active slots are now marked active.
    for idx in new_indices:
        assert cb.active_mask[idx]


def test_grow_noop_when_no_vacancy():
    cb = AdaptiveCodebook(size=16, dim=8)
    cb.storage.usage_counter[:] = 10
    new_indices = cb.grow(top_k_to_split=4)
    assert new_indices == []


def test_shrink_then_grow_preserves_invariant():
    cb = AdaptiveCodebook(size=16, dim=8)
    cb.storage.usage_counter[:10] = 100
    cb.storage.usage_counter[10:] = 0

    cb.shrink(min_usage_frac=0.01, min_codes=4)
    size_after_shrink = cb.current_size()
    new = cb.grow(top_k_to_split=3, seed=0)

    # Growth respects available vacancy.
    assert cb.current_size() == size_after_shrink + len(new)
