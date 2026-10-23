"""Localise gradient nondeterminism: repeat forward and grad_step on one
fixed minibatch (bs 256 pairs), per-tensor max |diff| across 5 repeats."""
import collections
import sys

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from experiments.procgen_multiagent_gtm_comm.harness.policy import PairPolicy
from experiments.procgen_multiagent_gtm_comm.harness.train import _make_steps, accumulate_minibatch

arm = sys.argv[1]
n = 256
m = PairPolicy(arm, mx.random.key(1))
mx.eval(m.parameters())
r = np.random.default_rng(3)
mb = {
    "obs": mx.array(r.integers(0, 256, (n, 2, 64, 64, 3), dtype=np.uint8)),
    "summary": mx.array(r.normal(size=(n, 2, 2)).astype(np.float32)),
    "eng_bits": mx.zeros((n, 2, 16)), "actions": mx.array(r.integers(0, 15, (n, 2))),
    "noise": m.sample_noise(n, mx.random.key(5)) if m.noise_dim else mx.zeros((n, 2, 0)),
    "logp": mx.array(-2.7 + 0.3 * r.normal(size=(n, 2)).astype(np.float32)),
    "advantages": mx.array(r.normal(size=(n, 2)).astype(np.float32)),
    "returns": mx.array(r.normal(size=(n, 2)).astype(np.float32)),
}
k = mx.random.key(0)
outs = []
for _ in range(2):
    o = m(mb["obs"], mb["summary"], None, train=True, key=k,
          noise=mb["noise"] if m.noise_dim else None)
    mx.eval(o.logits, o.value)
    outs.append((np.array(o.logits), np.array(o.value)))
print(arm, "forward bit-identical:", all(np.array_equal(a, b) for a, b in zip(*outs)))
opt = optim.Adam(5e-4)
opt.init(m.trainable_parameters())
gs = _make_steps(m, opt, False, plastic=False)[0]
for micro in (None, 128):
    gr = [dict(tree_flatten(accumulate_minibatch(gs, mb, micro, k)[1])) for _ in range(5)]
    grp = collections.defaultdict(float)
    nd = 0
    for g in gr[1:]:
        for name, a in gr[0].items():
            d = float(np.abs(np.array(a) - np.array(g[name])).max())
            if d > 0:
                nd += 1
                key = "trunk.blocks" if name.startswith("trunk.blocks") else name.rsplit(".", 1)[0]
                grp[key] = max(grp[key], d / (float(np.abs(np.array(a)).max()) + 1e-30))
    print(arm, "micro", micro, "non-identical tensor-pairs:", nd, "max rel diff by group:",
          {k_: f"{v:.1e}" for k_, v in sorted(grp.items())})
