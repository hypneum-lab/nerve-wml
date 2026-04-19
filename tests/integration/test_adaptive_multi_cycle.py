"""Multi-cycle stability — repeated shrink/grow must not collapse the alphabet."""
import torch

from track_p.adaptive_codebook import AdaptiveCodebook


def test_three_shrink_grow_cycles_keep_alphabet_alive():
    """After 3 shrink/grow cycles, the alphabet must have ≥ 4 codes and
    training losses must stay finite (no NaN)."""
    torch.manual_seed(0)
    cb = AdaptiveCodebook(size=16, dim=8)

    # Seed some usage.
    cb.storage.usage_counter[:8] = 50
    cb.storage.usage_counter[8:] = 5

    for cycle in range(3):
        cb.shrink(min_usage_frac=0.05, min_codes=4)
        # Simulate further usage: make the remaining codes hot.
        active = cb.active_indices()
        cb.storage.usage_counter[active] += 20
        cb.grow(top_k_to_split=2, seed=cycle)

    # After 3 cycles.
    assert cb.current_size() >= 4, (
        f"alphabet collapsed to {cb.current_size()} codes after 3 cycles"
    )

    # Verify embeddings are finite (no NaN accumulated from perturbations).
    assert torch.isfinite(cb.storage.embeddings).all()


def test_alternating_shrink_grow_is_stable():
    """Alternating shrink→grow→shrink→grow keeps size bounded."""
    torch.manual_seed(0)
    cb = AdaptiveCodebook(size=16, dim=8)
    cb.storage.usage_counter[:8] = 50
    cb.storage.usage_counter[8:] = 5

    sizes = [cb.current_size()]
    for _ in range(2):
        cb.shrink(min_usage_frac=0.05, min_codes=4)
        sizes.append(cb.current_size())
        # Re-bump usage so next shrink has signal.
        cb.storage.usage_counter[cb.active_indices()] += 30
        cb.grow(top_k_to_split=2)
        sizes.append(cb.current_size())

    # Alphabet stays in a reasonable band across cycles.
    assert min(sizes) >= 4
    assert max(sizes) <= 16
