"""Smoke test: exercise ProcgenCoopEnv with random policy for 100 steps.

Reports team reward distribution + observation shape sanity. Confirms
the cooperative wrapper works end-to-end.

Plan: HYPNEUM-PLANS/2026-05-11-niveau10-scaling-up.md (N10-C kickoff).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from experiments.procgen_multiagent_gtm_comm.coop_env import ProcgenCoopEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = ProcgenCoopEnv(env_name="coinrun", n_agents=2, seed=args.seed)
    obs = env.reset()
    print(
        f"reset OK. n_agents={len(obs)}, obs[0].shape={obs[0].shape}, "
        f"dtype={obs[0].dtype}"
    )

    rng = np.random.default_rng(args.seed)
    rewards = []
    coin_events = 0
    for _step in range(args.steps):
        # Random action per agent. Procgen actions: 0..14 (15 discrete).
        actions = [int(rng.integers(0, 15)) for _ in range(env.n_agents)]
        result = env.step(actions)
        rewards.append(result.team_reward)
        if any(result.info["coin_flags"]):
            coin_events += 1
        if result.done:
            obs = env.reset()
    env.close()

    arr = np.array(rewards)
    print(f"\n=== Smoke result ({args.steps} steps) ===")
    print(
        f"Team reward: min={arr.min():+.3f} mean={arr.mean():+.3f} "
        f"max={arr.max():+.3f}"
    )
    print(f"Coin events (any agent): {coin_events}")
    print(
        f"Distribution: -0.1 (step penalty)={int((arr == -0.1).sum())}, "
        f"+1.0 (team coin)={int((arr == 1.0).sum())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
