from track_w.bio_wml import BioWML
from track_w.pool_factory import build_pool_with_bio


def test_build_pool_with_bio_includes_one_bio_wml():
    pool = build_pool_with_bio(n=4, seed=0)
    assert len(pool) == 4
    assert sum(isinstance(w, BioWML) for w in pool) >= 1


def test_build_pool_with_bio_ids_are_contiguous():
    pool = build_pool_with_bio(n=5, seed=0)
    assert [w.id for w in pool] == [0, 1, 2, 3, 4]


def test_build_pool_with_bio_is_deterministic():
    a = build_pool_with_bio(n=4, seed=7)
    b = build_pool_with_bio(n=4, seed=7)
    assert [type(w).__name__ for w in a] == \
           [type(w).__name__ for w in b]
