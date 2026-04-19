"""Tests for neuromorphic.export — quantization + artefact save/load."""
import numpy as np

from neuromorphic.export import (
    load_neuromorphic_artefact,
    quantize_lif_wml,
    save_neuromorphic_artefact,
)
from track_w.lif_wml import LifWML


def test_quantize_returns_expected_keys():
    wml = LifWML(id=0, n_neurons=16, seed=0)
    export = quantize_lif_wml(wml)
    for k in ("codebook_int8", "codebook_scale",
              "input_proj_int8", "input_proj_scale", "input_proj_bias",
              "v_thr", "tau_mem", "n_neurons", "alphabet_size", "bits"):
        assert k in export


def test_quantize_int8_dtype_and_range():
    wml = LifWML(id=0, n_neurons=16, seed=0)
    export = quantize_lif_wml(wml)
    assert export["codebook_int8"].dtype == np.int8
    assert export["input_proj_int8"].dtype == np.int8
    assert abs(export["codebook_int8"]).max() <= 127
    assert abs(export["input_proj_int8"]).max() <= 127


def test_save_load_round_trip_is_bit_stable(tmp_path):
    wml = LifWML(id=0, n_neurons=16, seed=0)
    export = quantize_lif_wml(wml)

    path = tmp_path / "artefact"
    save_neuromorphic_artefact(export, path)
    loaded = load_neuromorphic_artefact(path)

    # Arrays equal bit-for-bit.
    np.testing.assert_array_equal(loaded["codebook_int8"],   export["codebook_int8"])
    np.testing.assert_array_equal(loaded["input_proj_int8"], export["input_proj_int8"])
    np.testing.assert_array_equal(loaded["input_proj_bias"], export["input_proj_bias"])

    # Scalars equal.
    assert loaded["codebook_scale"]   == export["codebook_scale"]
    assert loaded["input_proj_scale"] == export["input_proj_scale"]
    assert loaded["v_thr"]            == export["v_thr"]
    assert loaded["tau_mem"]          == export["tau_mem"]
    assert loaded["n_neurons"]        == export["n_neurons"]
    assert loaded["alphabet_size"]    == export["alphabet_size"]
    assert loaded["bits"]             == export["bits"]


def test_quantize_preserves_approximate_values():
    """Dequantizing should recover values within the quantization error."""
    wml = LifWML(id=0, n_neurons=16, seed=0)
    export = quantize_lif_wml(wml)

    cb_approx = export["codebook_int8"].astype(np.float32) * export["codebook_scale"]
    original  = wml.codebook.detach().cpu().numpy()
    max_err = np.abs(cb_approx - original).max()
    # INT8 symmetric quant error bounded by scale.
    assert max_err <= export["codebook_scale"]
