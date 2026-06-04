import torch

from track_p.transducer_baselines import ProcrustesTransducer, RelativeRepTransducer


def test_procrustes_fits_and_maps_to_valid_codes():
    torch.manual_seed(0)
    src_cb = torch.randn(64, 32)
    # dst codebook is a rotation of src + small noise — recoverable.
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = src_cb @ q + 0.01 * torch.randn(64, 32)
    t = ProcrustesTransducer(src_codebook=src_cb, dst_codebook=dst_cb)
    src_code = torch.tensor([5, 17, 42])
    dst_code = t.forward(src_code)
    assert dst_code.shape == src_code.shape
    assert (dst_code >= 0).all() and (dst_code < 64).all()


def test_procrustes_recovers_known_rotation():
    torch.manual_seed(1)
    src_cb = torch.randn(64, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    # dst is a pure rotation of src, aligned by code index.
    # This is the scenario Procrustes is designed for: well-paired codebooks.
    dst_cb = src_cb @ q
    t = ProcrustesTransducer(src_codebook=src_cb, dst_codebook=dst_cb)
    # Verify the fitted rotation is orthogonal (key property of Procrustes).
    identity_check = torch.eye(32, dtype=t.rotation.dtype, device=t.rotation.device)
    assert torch.allclose(
        t.rotation @ t.rotation.T, identity_check, atol=1e-4
    ), "Fitted rotation should be orthogonal"
    # Test the forward mapping: it should produce valid code indices.
    mapped = t.forward(torch.arange(64))
    # Check that mapped codes are within valid range and have the right shape.
    assert mapped.shape == (64,)
    assert (mapped >= 0).all() and (mapped < 64).all()


def test_relative_rep_zero_shot_maps_valid_codes():
    torch.manual_seed(2)
    src_cb = torch.randn(64, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = src_cb @ q
    t = RelativeRepTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb, n_anchors=16, seed=0
    )
    dst_code = t.forward(torch.tensor([3, 9, 60]))
    assert dst_code.shape == (3,)
    assert (dst_code >= 0).all() and (dst_code < 64).all()


def test_relative_rep_invariant_to_rotation():
    torch.manual_seed(3)
    src_cb = torch.randn(64, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = src_cb @ q  # pure rotation, anchors shared by index
    t = RelativeRepTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb, n_anchors=32, seed=1
    )
    # Cosine-to-anchors is rotation-invariant -> identity recovered.
    assert torch.equal(t.forward(torch.arange(64)), torch.arange(64))


def test_vec2vec_trains_and_maps_valid_codes():
    from track_p.transducer_baselines import Vec2VecTransducer
    torch.manual_seed(4)
    src_cb = torch.randn(64, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = src_cb @ q
    t = Vec2VecTransducer(src_codebook=src_cb, dst_codebook=dst_cb, seed=0)
    history = t.fit(steps=200)
    assert len(history) == 200
    # Cycle-consistency loss should decrease over training.
    assert history[-1] < history[0]
    dst_code = t.forward(torch.tensor([1, 30, 63]))
    assert dst_code.shape == (3,)
    assert (dst_code >= 0).all() and (dst_code < 64).all()
