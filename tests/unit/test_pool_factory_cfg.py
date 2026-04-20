from track_w.configs.wml_config import WmlConfig
from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML
from track_w.pool_factory import build_pool_cfg


def test_build_pool_cfg_uses_config_dims():
    cfg = WmlConfig(
        input_dim=32, d_hidden=64, n_neurons=64,
        d_model=64, n_heads=4, n_tokens=4, alphabet_size=128,
    )
    pool = build_pool_cfg(n_wmls=4, cfg=cfg, seed=0)
    assert len(pool) == 4
    mlps = [w for w in pool if isinstance(w, MlpWML)]
    lifs = [w for w in pool if isinstance(w, LifWML)]
    assert len(mlps) == 2 and len(lifs) == 2
    assert mlps[0].codebook.shape == (128, 64)
    assert lifs[0].codebook.shape == (128, 64)
    assert lifs[0].input_proj.in_features == 32
    assert lifs[0].input_proj.out_features == 64
