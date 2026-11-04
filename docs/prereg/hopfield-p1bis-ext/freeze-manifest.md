# Freeze manifest (DRAFT) — P1bis-ext and P8c-nocap

**Status:** DRAFT at L0. Not a freeze. The hashes below describe the branch
`prereg/p1bis-p8c-20261104` at drafting (2026-11-04, commit 13f2092) and
will be regenerated with `scripts/hopfield_conf/make_manifest.sh` on the
merge commit before tagging. Fields marked "at freeze" are filled then.

- Studies: `hopfield-p1bis-ext` (`docs/prereg/hopfield-p1bis-ext/prereg.md`),
  `hopfield-p8c-nocap` (`docs/prereg/hopfield-p8c-nocap/prereg.md`)
- Tag: `prereg-hopfield-p1bis-p8c-<YYYYMMDD>` — at freeze
- Merge commit, tagger date, SWHID (revision `swh:1:rev:…`) — at freeze
- SHA-256 of this manifest: recorded in the tag message at freeze
- Host: kxkm-ai, CPU only, `/ai/saisail/p1bis-p8c-20261104`
- Environment: Python 3.12.15 installed by uv 0.13.5 (binary sha256
  `8a70db72685b7138121bfcda71fa1a00cb709ed9f1fdc4ecf1dd871e8e167e52`),
  `experiments/hopfield_dream_conf/requirements-hopfield-conf.txt`
- Vendored upstream: SaverioRo/hopfield-dreaming-clipping
  @`7e4257489efd763560a6df0e4456da0fa428cc9d`, `hopfield.py` (MIT)

## Files (draft hashes; this manifest excluded)

| File | SHA-256 |
|---|---|
| `docs/prereg/hopfield-p1bis-ext/prereg.md` | `147cc0ae1de712caf57f5a0eea09e91d84f995fc495d9eee12d9f0edd6a678d7` |
| `docs/prereg/hopfield-p8c-nocap/prereg.md` | `e7610623ecd526f4aea9d3d3e0cead12a44860bcb30046aaf6d65d100557835d` |
| `experiments/hopfield_dream_conf/__init__.py` | `caa682e51adb80d8ff49bad201ede95d15e0b4557d52cc8a90d885cda680e02f` |
| `experiments/hopfield_dream_conf/p1bis_ext.py` | `3bff8cb057084f9051262dd168704b81eabda46b2385aee3bf2a96eacb5f4b74` |
| `experiments/hopfield_dream_conf/p8c_nocap.py` | `a80fc61d2997529a04287f87392be98a207e409e55386044c965efcbfd24ff09` |
| `experiments/hopfield_dream_conf/stats.py` | `bc1094dacdf8957053a5e9133fb7be898480d111030fb31a0abac01941b2d649` |
| `experiments/hopfield_dream_conf/analyse.py` | `fcdb791c7d88b387fa721788657e37a45968b9db6aedb21acbe24c760ea8d679` |
| `experiments/hopfield_dream_conf/power.py` | `cabcd7724a152e27697f681baf7a9d40f3ae31d4659b86c6988aaf996257f344` |
| `experiments/hopfield_dream_conf/requirements-hopfield-conf.txt` | `2d93299dec8dbd4c0f6a60c8e4f2e9431a121792588dc256adaf9c321ca1596a` |
| `experiments/hopfield_dream_conf/vendor/__init__.py` | `3294a344a0523a850ea20b1986c24890c96042bb8b728746c33792040e2ce9fe` |
| `experiments/hopfield_dream_conf/vendor/hopfield.py` | `dfaaeb7c1904a0eb2ab2ffa94b7211b36214dd7eef0182eb0d61959dcd9162af` |
| `experiments/hopfield_dream_conf/vendor/LICENSE` | `f11a28932be4f9edb32b7406f0284b8801e390d392d3392246e89e631813e775` |
| `tests/unit/test_hopfield_dream_conf_stats.py` | `8725ba5f20f31e29fcfa56344cf92885b8a8a241416b20dc45758ee4031fff1e` |
| `tests/unit/test_hopfield_dream_conf_drivers.py` | `adeb920d6383bc79694a810f62801e334f7d65c19b7d7e7658c3aa0436a4e0b3` |
| `scripts/hopfield_conf/kxkm_launch.sh` | `5952cc7bb21256876ca2b3e913cff5dc58e733e993a7bb0e38960662e7df0eee` |
| `scripts/hopfield_conf/make_manifest.sh` | `c3d59c2df2e47c907de49d24fbc850eeb0b07279385140c95c9784e458e062a5` |
| `.ruff.toml` | `96282cb312aae7be7ed2f7ecde7cc736c0f413a9ae7516e803f42c6652b0552c` |

## Exploratory inputs (not part of the frozen tree)

- Pilot archive: `electron-server:~/archives/gb300-20261101/hypneum-20261101/pilots/`
  (`cpu-p1-hopfield-bounded`, `cpu-p1bis-hopfield-fracbudget`,
  `gb300-p8c-dream-kernel`, `rtx-p8-hopfield-mixtures`); P1bis code sha256 in
  its `SHA256SUMS`, P8c in its `deps-record.txt`.
- Smoke runs (seeds ≥ 90 000): `kxkm-ai:/ai/saisail/p1bis-p8c-20261104/smoke/`
  (`SHA256SUMS`, `README-SMOKE.txt`).
