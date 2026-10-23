import hashlib, json, sys, numpy as np
from procgen import ProcgenGym3Env
out = {}
for L in list(range(16)) + list(range(2000, 2004)):
    env = ProcgenGym3Env(num=1, env_name="coinrun", distribution_mode="easy",
                         num_levels=1, start_level=L, rand_seed=0)
    rng = np.random.default_rng(L)
    h = hashlib.sha256()
    for _ in range(300):
        env.act(rng.integers(0, 15, size=1).astype(np.int32))
        rew, obs, first = env.observe()
        h.update(np.ascontiguousarray(obs["rgb"]).tobytes())
        h.update(np.ascontiguousarray(rew).tobytes())
        h.update(np.ascontiguousarray(first).tobytes())
    out[str(L)] = h.hexdigest()
    env.close() if hasattr(env, "close") else None
json.dump(out, sys.stdout, sort_keys=True)
