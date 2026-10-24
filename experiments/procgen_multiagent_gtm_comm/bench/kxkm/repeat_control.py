"""Same-seed repeat control: dump init/forward/rollout hashes and params
after update 1 and 8 for one (arm, micro) config. Usage:
python repeat_control.py ARM MICRO|none TAG OUTDIR"""
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from experiments.procgen_multiagent_gtm_comm.harness.codes import init_key_int
from experiments.procgen_multiagent_gtm_comm.harness.policy import PairPolicy
from experiments.procgen_multiagent_gtm_comm.harness.train import train_run

arm, micro, tag, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
micro = None if micro == "none" else int(micro)
out.mkdir(parents=True, exist_ok=True)
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]  # noqa: E731
info = {}
m = PairPolicy(arm, mx.random.key(init_key_int(5, arm, "registered")))
mx.eval(m.parameters())
info["init"] = h(np.concatenate([np.array(v).ravel() for _, v in tree_flatten(m.parameters())]))
r = np.random.default_rng(0)
x = mx.array(r.integers(0, 256, (8, 2, 64, 64, 3), dtype=np.uint8))
o = m(x, mx.zeros((8, 2, 2)), None, train=False, key=mx.random.key(0))
mx.eval(o.logits, o.value)
info["forward"] = h(np.array(o.logits)) + h(np.array(o.value))
snaps, roll = {}, {}


def fault(u, model):
    if u in (1, 8):
        snaps[u] = {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def rhook(u, buf):
    if u in (1, 8):
        roll[u] = {n: h(getattr(buf, n)) for n in ("obs", "actions", "logp", "value", "returns")}


rec = train_run(arm, 5, "registered", 65536, out / tag, n_pairs=64, horizon=128,
                log_every_pair_steps=65536, ckpt_every_pair_steps=65536,
                micro_batch_pairs=micro, _fault_hook=fault, _rollout_hook=rhook)
info["rollout"] = roll
info["failure"] = rec.training_failure
flat = {f"u{u}/{k}": v for u, d in snaps.items() for k, v in d.items()}
np.savez(out / f"{tag}.params.npz", **flat)
(out / f"{tag}.info.json").write_text(json.dumps(info, indent=1))
