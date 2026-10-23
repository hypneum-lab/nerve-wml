# Procgen build-difference evidence (macM1 arm64 vs KXKM x86-64)

Evidence for the disclosure in the 2026-10-23 amendment ("the two
Procgen builds are not bit-identical"). Measured 2026-10-23. Nothing
here is a training or evaluation result.

## Hosts and builds

| Host | Build |
|---|---|
| macM1 (Apple M1 Max, arm64, macOS 27) | Procgen 0.10.7 built from source with the local build patch `../procgen-0.10.7-arm64-build.patch` (see `../README.md`) |
| KXKM-AI (x86-64 Linux, RTX 4090 host) | official Procgen 0.10.7 x86-64 wheel `procgen-0.10.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` |

## Procedure

`frames.py`: CoinRun, `distribution_mode="easy"`, `num=1`,
`num_levels=1`, `rand_seed=0`, `start_level` L for L in 0-15 and
2000-2003 (20 levels). Per level, 300 steps with actions drawn from
`numpy.random.default_rng(L)` (`integers(0, 15)`); SHA-256 of the
concatenated bytes of the rgb observation, reward and first flag at
every step. The script prints one JSON object, level -> hash.

`frames2.py`: diagnostic on levels 0, 1 and 5: reward and first-flag
totals, rgb hash prefixes at steps 0, 1 and 299, and the total pixel
sum at step 299.

Each host ran `frames.py` twice (`*_kxkm.json` / `*_kxkmb.json`,
`*_macm1.json` / `*_macm1b.json`). `h_kxkm.txt` and `h_macm1.txt` are
the same per-level hashes, one per line. The macM1 JSON files start
with the line `building procgen...done` printed by the local build on
import; the JSON object follows it. The files are archived as
produced.

## Result

- 0 of 20 levels have identical stream hashes between macM1 and KXKM
  (20 distinct hashes on each host).
- Each host is deterministic: the two reruns on each host are
  identical (`frames_kxkm.json` equals `frames_kxkmb.json`;
  `frames_macm1.json` equals `frames_macm1b.json`, byte for byte).
- The divergence appears late in an episode, not at the initial
  render. From `frames2.py` (output quoted from
  `reports/bench-m3-kxkm-es.md`, raw stdout not archived): levels 0
  and 1 have identical per-frame rgb hashes at steps 0, 1 and 299 and
  identical pixel sums; level 5 is identical at steps 0 and 1 and
  differs at step 299 (pixel sum 1886007 vs 1886144, after an episode
  reset). The whole-stream hash also covers reward and first bytes,
  which were not separated.

Verdict: the builds are not bit-identical; seed-for-seed
reproducibility of Procgen across the two hosts cannot be assumed. It
is not known whether the difference affects results.

## Files

`frames.py`, `frames2.py` (procedure), `frames_kxkm.json`,
`frames_kxkmb.json`, `frames_macm1.json`, `frames_macm1b.json`,
`h_kxkm.txt`, `h_macm1.txt` (hashes). The stage-2 manifest records the
SHA-256 of each file.
