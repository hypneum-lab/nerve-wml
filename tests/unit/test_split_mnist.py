from track_w.tasks.split_mnist import SplitMnistLikeTask


def test_split_mnist_like_has_two_subtasks():
    task = SplitMnistLikeTask(seed=0)
    assert len(task.subtasks) == 2


def test_split_mnist_like_subtasks_have_disjoint_labels():
    task = SplitMnistLikeTask(seed=0)
    labels_a = set()
    labels_b = set()
    for _ in range(32):
        _, ya = task.subtasks[0].sample(batch=16)
        _, yb = task.subtasks[1].sample(batch=16)
        labels_a.update(ya.tolist())
        labels_b.update(yb.tolist())
    assert labels_a.isdisjoint(labels_b)
