"""Software (PyTorch) vs neuromorphic (mock runner) accuracy delta."""
from __future__ import annotations

import torch

from neuromorphic.mock_runner import MockNeuromorphicRunner


def compare_software_vs_neuromorphic(
    wml,
    inputs: torch.Tensor,
    artefact: dict,
) -> dict:
    """Compute the codes each path produces on the same inputs.

    Returns:
        dict with pytorch_codes, neuromorphic_codes, agreement (fraction
        of inputs where both paths picked the same code).
    """
    from track_w._surrogate import spike_with_surrogate

    # PyTorch reference.
    with torch.no_grad():
        i_in = wml.input_proj(inputs)
        spikes = spike_with_surrogate(i_in, v_thr=wml.v_thr)
        norms = wml.codebook.norm(dim=-1) + 1e-6
        spikes_norm = spikes.norm(dim=-1, keepdim=True) + 1e-6
        sims = spikes @ wml.codebook.T / (norms * spikes_norm)
        pytorch_codes = sims.argmax(dim=-1).cpu().numpy()

    # Mock runner.
    runner = MockNeuromorphicRunner(artefact)
    neuromorphic_codes = runner.forward(inputs.cpu().numpy())

    agreement = float((pytorch_codes == neuromorphic_codes).mean())
    return {
        "pytorch_codes":      pytorch_codes,
        "neuromorphic_codes": neuromorphic_codes,
        "agreement":          agreement,
        "delta":              1.0 - agreement,
    }
