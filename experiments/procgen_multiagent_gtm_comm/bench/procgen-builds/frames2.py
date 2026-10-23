import hashlib, numpy as np
from procgen import ProcgenGym3Env
for L in (0, 1, 5):
    env = ProcgenGym3Env(num=1, env_name="coinrun", distribution_mode="easy", num_levels=1, start_level=L, rand_seed=0)
    rng = np.random.default_rng(L); R=[];F=[];I=[]
    for t in range(300):
        env.act(rng.integers(0, 15, size=1).astype(np.int32))
        rew, obs, first = env.observe()
        R.append(float(rew[0])); F.append(bool(first[0])); I.append(hashlib.sha256(obs["rgb"].tobytes()).hexdigest()[:8])
    print("RES", L, "rew", sum(R), "firsts", sum(F), "rgb0", I[0], "rgb1", I[1], "rgb299", I[299], "dtype", obs["rgb"].dtype, obs["rgb"].shape, "sum", int(obs["rgb"].astype(np.int64).sum()))
