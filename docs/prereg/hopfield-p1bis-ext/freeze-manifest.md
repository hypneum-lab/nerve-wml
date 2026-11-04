# Freeze manifest — P1bis-ext and P8c-nocap

**Status:** frozen at L1 by the annotated tag
`prereg-hopfield-p1bis-p8c-freeze-20261104` on the commit that adds this file
(its parent is the merge commit below; this commit changes nothing but this
manifest). Any later change is an amendment
(`docs/prereg/<study>/amendment-YYYY-MM-DD.md`), and a new dated tag replaces
this one; the tag is never moved.

- Studies: `hopfield-p1bis-ext` (`docs/prereg/hopfield-p1bis-ext/prereg.md`),
  `hopfield-p8c-nocap` (`docs/prereg/hopfield-p8c-nocap/prereg.md`)
- PR: hypneum-lab/nerve-wml#106, merged on green CI
- Merge commit: `4c7ab1bb9d7b8b4ba169bd3d31dddb8822ad8ae4`
- Tag: `prereg-hopfield-p1bis-p8c-freeze-20261104`; tagger date and SWHID
  (revision) recorded below in a follow-up commit after the Software Heritage
  save
- SHA-256 of this manifest: in the tag message
- PI decisions closed before the tag (2026-11-04): N = 16 384 restricted to
  α = 0.5 in P8c-nocap (cost); all other drafted defaults kept; independent
  review ACCEPT, four minor documentation points integrated. No open
  "TBD by PI" item.
- Host: kxkm-ai, CPU only, `/ai/saisail/p1bis-p8c-20261104`
- Environment: Python 3.12.15 installed by uv 0.13.5 (binary sha256
  `8a70db72685b7138121bfcda71fa1a00cb709ed9f1fdc4ecf1dd871e8e167e52`),
  `experiments/hopfield_dream_conf/requirements-hopfield-conf.txt`
- Vendored upstream: SaverioRo/hopfield-dreaming-clipping
  @`7e4257489efd763560a6df0e4456da0fa428cc9d`, `hopfield.py` (MIT)
- Confirmatory seeds: 31001–31020 (P1bis-ext), 32001–32020 (P8c-nocap)

## Files at the merge commit (this manifest excluded)

| File | SHA-256 |
|---|---|
| `docs/prereg/hopfield-p1bis-ext/prereg.md` | `0a96ddf8243e43d1cf15a5604728859bae21a871e4b89342268c6843536beef4` |
| `docs/prereg/hopfield-p8c-nocap/prereg.md` | `d0530d7b7f8a07fe3f73bef5166e1b7354ce4687932087f4d738a83aaf79360b` |
| `experiments/hopfield_dream_conf/__init__.py` | `caa682e51adb80d8ff49bad201ede95d15e0b4557d52cc8a90d885cda680e02f` |
| `experiments/hopfield_dream_conf/p1bis_ext.py` | `3bff8cb057084f9051262dd168704b81eabda46b2385aee3bf2a96eacb5f4b74` |
| `experiments/hopfield_dream_conf/p8c_nocap.py` | `10f55e835e1479d9b174745210bb5790e7f25728c6ddce03e92537b5cc878254` |
| `experiments/hopfield_dream_conf/stats.py` | `bc1094dacdf8957053a5e9133fb7be898480d111030fb31a0abac01941b2d649` |
| `experiments/hopfield_dream_conf/analyse.py` | `f814bde4ccaca531a5638c70ad462b9a8aa6b836df35bddbb9fc6ec960d93dc4` |
| `experiments/hopfield_dream_conf/power.py` | `e6bcc42443970011c027f2e1559c0686f6f55ec9eaab15f74460519900fa1b3b` |
| `experiments/hopfield_dream_conf/requirements-hopfield-conf.txt` | `2d93299dec8dbd4c0f6a60c8e4f2e9431a121792588dc256adaf9c321ca1596a` |
| `experiments/hopfield_dream_conf/vendor/__init__.py` | `3294a344a0523a850ea20b1986c24890c96042bb8b728746c33792040e2ce9fe` |
| `experiments/hopfield_dream_conf/vendor/hopfield.py` | `dfaaeb7c1904a0eb2ab2ffa94b7211b36214dd7eef0182eb0d61959dcd9162af` |
| `experiments/hopfield_dream_conf/vendor/LICENSE` | `f11a28932be4f9edb32b7406f0284b8801e390d392d3392246e89e631813e775` |
| `tests/unit/test_hopfield_dream_conf_stats.py` | `e6dd868773bfdc3423381390599554b788a8bd4da273b9ccad0c27021c801cee` |
| `tests/unit/test_hopfield_dream_conf_drivers.py` | `adeb920d6383bc79694a810f62801e334f7d65c19b7d7e7658c3aa0436a4e0b3` |
| `scripts/hopfield_conf/kxkm_launch.sh` | `278fc8290bd5e2f394453ede84c624597e9392dbd73f80e72aa6f4eedf072d15` |
| `scripts/hopfield_conf/make_manifest.sh` | `c3d59c2df2e47c907de49d24fbc850eeb0b07279385140c95c9784e458e062a5` |
| `.ruff.toml` | `96282cb312aae7be7ed2f7ecde7cc736c0f413a9ae7516e803f42c6652b0552c` |

Git tree `experiments/hopfield_dream_conf` at the merge commit: `902f4e4e54c40a9bcf97ce5ef750b0f9d6f69c2e`

## Exploratory inputs (not part of the frozen tree)

- Pilot archive: `electron-server:~/archives/gb300-20261101/hypneum-20261101/pilots/`
  (`cpu-p1-hopfield-bounded`, `cpu-p1bis-hopfield-fracbudget`,
  `gb300-p8c-dream-kernel`, `rtx-p8-hopfield-mixtures`); P1bis code sha256 in
  its `SHA256SUMS`, P8c in its `deps-record.txt`.
- Smoke runs (seeds ≥ 90 000, pre-freeze working tree):
  `kxkm-ai:/ai/saisail/p1bis-p8c-20261104/smoke/` (`SHA256SUMS`,
  `README-SMOKE.txt`).

## Freeze record

- Tagger date: to be recorded
- SWHID (revision): to be recorded
