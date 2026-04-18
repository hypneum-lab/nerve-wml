from harness.run_registry import compute_run_id


def test_run_id_is_stable_for_same_inputs():
    a = compute_run_id(c_version="0.1.0", topology=[(0, 1), (1, 2)],
                       seed=42, commit_sha="deadbeef")
    b = compute_run_id(c_version="0.1.0", topology=[(0, 1), (1, 2)],
                       seed=42, commit_sha="deadbeef")
    assert a == b


def test_run_id_differs_for_different_seeds():
    a = compute_run_id(c_version="0.1.0", topology=[(0, 1)], seed=42,  commit_sha="abc")
    b = compute_run_id(c_version="0.1.0", topology=[(0, 1)], seed=43,  commit_sha="abc")
    assert a != b


def test_run_id_is_16_hex_chars():
    run_id = compute_run_id(c_version="0.1.0", topology=[(0, 1)],
                            seed=0, commit_sha="abc")
    assert len(run_id) == 16
    assert all(c in "0123456789abcdef" for c in run_id)


def test_run_id_is_order_independent_for_topology():
    a = compute_run_id(c_version="0.1.0", topology=[(0, 1), (2, 3)],
                       seed=0, commit_sha="abc")
    b = compute_run_id(c_version="0.1.0", topology=[(2, 3), (0, 1)],
                       seed=0, commit_sha="abc")
    assert a == b
