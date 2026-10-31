# Freeze manifest — G7 GB300 indicative pilot (FROZEN)

**Status:** FROZEN (tag `n10c-bis-g7-gb300-freeze-20261031`). PI decisions 2026-10-31
22:20–22:26 CET, PI approval of the hashed requirements 22:37 CET. PR #101 (Procgen kit,
head `ac1bed2`) merged as `36fd117`, PR #99 (head `2a61c46`) merged on green CI as
`8d9c125d353569aba77f07c5cabb9fee298dbff3`; the hashes below are those of that merge commit. This freeze covers the
indicative GB300 pilot only; it is not the N10-C-bis freeze (`n10c-bis-freeze-*`), which
still depends on the decisional KXKM-AI gate. Deployed N10-C clones stay at their freeze
tag.

| file | SHA-256 |
|---|---|
| `docs/prereg/n10c-bis-gate-gb300/g7-host-annex.md` | `4e9c5b05206832c6e17bb47eb6aca785400646419c9b6ea5994aea2eefe79b98` |
| `docs/prereg/n10c-bis-gate-gb300/gate_g7_lane.sh` | `62c6bc0e65c09fde725e8396e1d656284b503f53eea66790f4357af272b4dcf0` |
| `docs/prereg/n10c-bis-gate-gb300/requirements-gb300-n10c-bis-gate-cuda12.txt` | `4a10e6c6417bf6a9445cf82f53dc6f8e927ca26584ff4539895d865ad8444038` |
| `docs/prereg/n10c-bis-gate-gb300/requirements-gb300-n10c-bis-gate-cuda13.txt` | `929339cd815b49244640f31c09944d45d10d89ab826a83dcab36613ee7db841a` |
| `docs/prereg/n10c-bis-gate-gb300/requirements-gb300-build-tools.txt` | `70705ec0dbb8897aa502d421767cdc02909ccd673857018d93d838003b03b358` |
| `docs/prereg/n10c-bis-gate-gb300/requirements-review.md` | `4863d22cb4c4d4f0bfcda503c9f5e3fd01228e5848fed9658fe296fb32aaf7e7` |
| `docs/milestones/2026-10-25-n10c-bis-control-arms.md` | `70531796b889e622a8a678614cd802b19ca1ba6f4c6ed904d8d279bfea056a43` |
| `experiments/procgen_multiagent_gtm_comm/harness/devledger.py` | `ee6e7cd01e96b03f7f6de44a11cc6a94222ecad2f42ab79eb66a66ae74fe8a9b` |
| `tests/unit/n10c/test_trunk_fix.py` | `f23523babe3932a994ac7f2007a52c61201f67efb242b91a189a1bc7ed78148d` |
| `tools/procgen-aarch64/README-build.md` | `d2af988948d8ad94599958bd94693ef986f5e19848f81548a27d7ab36de192c0` |
| `tools/procgen-aarch64/build_procgen_aarch64.sh` | `6ed3a4297c9bd42477b3432da8cac9da9bc722e85bb554130ed0392df8cb234d` |
| `tools/procgen-aarch64/patches.sha256` | `a27739a9a44699b4ba8da04fd6e4020340ec19a7861b6fe6116592a3766d5e1d` |
| `tools/procgen-aarch64/patches/0001-fix-setup-allow-filelock-4.x.patch` | `7abc70e39c54f42fb624d98b661434fd188b7738805631fe25f50a2b3ecef474` |
| `tools/procgen-aarch64/patches/0002-fix-cmake-gate-march-ivybridge-to-x86.patch` | `6a890e7e326a537d7f22dfb4b205076f58a00cd3a14359c1cf00f55ac67aa5b9` |
| `tools/procgen-aarch64/patches/0003-fix-cmake-match-x86-char-and-FP-semantics.patch` | `d4446421a36f943df4c92bcd45e199a87a9e530ba7683db9f912c787936d4ac2` |
| `tools/procgen-aarch64/requirements-aarch64-cp312.lock` | `b49aba5ae95cb2ce825a7118bcbcea7eb68f9db627b1abae1b3aef98064ebea1` |

Tag: `n10c-bis-g7-gb300-freeze-20261031` (annotated, on the commit that adds this frozen
manifest) · merge commit: `8d9c125d353569aba77f07c5cabb9fee298dbff3` · SWHID (revision): pending Software Heritage.
