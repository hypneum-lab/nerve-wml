"""MLX backend of the test run, for the strict reproducibility tests.

MLX 0.31.1 on CUDA (KXKM-AI): the convolution backward is
non-deterministic and selects batch-size-dependent kernels, with no
deterministic switch (amendment 2026-10-25, section 4.2; N10-C
Reproducibility statement). Tests that compare two trainings, two
gradient computations on different batch splits or a ratio of exactly 1
are therefore strict on Metal (macM1) and CPU only; on CUDA they are
skipped or relaxed with this reason, as in commit a31ae96.
"""
import mlx.core as mx

CUDA_NONDETERMINISM = (
    "MLX 0.31.1 on CUDA: non-deterministic, batch-size-dependent conv "
    "backward (no deterministic switch); strict on Metal (macM1) and CPU")


def on_cuda() -> bool:
    """True when MLX runs on its CUDA backend (KXKM-AI)."""
    cuda = getattr(mx, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_available"):
        try:
            return bool(cuda.is_available())
        except Exception:  # noqa: BLE001 - a broken probe is not CUDA
            return False
    return False
