"""Unit tests for HardSplitTask."""
from track_w.tasks.hard_split import HardSplitTask


def test_hard_split_shapes():
    task = HardSplitTask(seed=0)
    x0, y0 = task.subtasks[0].sample(batch=32)
    x1, y1 = task.subtasks[1].sample(batch=32)
    assert x0.shape == (32, 16)
    assert y0.shape == (32,)
    assert x1.shape == (32, 16)
    assert y1.shape == (32,)


def test_hard_split_shared_label_space():
    """Both sub-tasks emit labels in 0..11 (same 12-class head)."""
    task = HardSplitTask(seed=0)
    for subtask in task.subtasks:
        _, y = subtask.sample(batch=256)
        assert y.min().item() >= 0
        assert y.max().item() <= 11


def test_hard_split_subtasks_different_distributions():
    """The two sub-tasks must be non-identical (different centroids)."""
    task = HardSplitTask(seed=0)
    x0, _ = task.subtasks[0].sample(batch=128)
    x1, _ = task.subtasks[1].sample(batch=128)
    # Means should differ by at least 0.1 in L2 norm.
    assert (x0.mean(0) - x1.mean(0)).norm().item() > 0.1
