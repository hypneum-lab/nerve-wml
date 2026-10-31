# Freeze manifest — G7 GB300 indicative pilot (PREPARED, NOT FROZEN)

**Status:** prepared 2026-10-31; **not tagged, not merged**. The freeze waits for the PI's review of the hashed requirement files. At the freeze: merge this PR on green CI, recompute this table on the merge commit (any change of a file requires a new table), annotate the tag `n10c-bis-gb300-pilot-freeze-YYYYMMDD` (never `n10c-bis-freeze-*`) with this file's SHA-256, save on Software Heritage and record the revision SWHID here. The Procgen kit (PR #101) and the build record are pinned by their own commit and SHA-256 (annex section 4).

| file | SHA-256 |
|---|---|
| `docs/prereg/n10c-bis-gate-gb300/g7-host-annex.md` | `3fa94ec42a8b69bb3831656db5aecaca3a727bc797fa8e30cf5507159fb68618` |
| `docs/prereg/n10c-bis-gate-gb300/gate_g7_lane.sh` | `f331d474be3a933cde3759814a839cf7d5a444991e90ec3ae4573ea53eef3d87` |
| `docs/prereg/n10c-bis-gate-gb300/requirements-gb300-n10c-bis-gate-cuda12.txt` | `4a10e6c6417bf6a9445cf82f53dc6f8e927ca26584ff4539895d865ad8444038` |
| `docs/prereg/n10c-bis-gate-gb300/requirements-gb300-n10c-bis-gate-cuda13.txt` | `929339cd815b49244640f31c09944d45d10d89ab826a83dcab36613ee7db841a` |
| `docs/prereg/n10c-bis-gate-gb300/requirements-gb300-build-tools.txt` | `70705ec0dbb8897aa502d421767cdc02909ccd673857018d93d838003b03b358` |
| `docs/prereg/n10c-bis-gate-gb300/requirements-review.md` | `3e9b69dbc3f30317f7bdd817c15449fd6daee84dbb4a2de92f527c501ca4f334` |
| `docs/milestones/2026-10-25-n10c-bis-control-arms.md` | `70531796b889e622a8a678614cd802b19ca1ba6f4c6ed904d8d279bfea056a43` |
| `experiments/procgen_multiagent_gtm_comm/harness/devledger.py` | `ee6e7cd01e96b03f7f6de44a11cc6a94222ecad2f42ab79eb66a66ae74fe8a9b` |
| `tests/unit/n10c/test_trunk_fix.py` | `f23523babe3932a994ac7f2007a52c61201f67efb242b91a189a1bc7ed78148d` |

Tag: `<TBD at freeze>` · merge commit: `<TBD>` · SWHID (revision): `<TBD>` · SHA-256 of this file: recorded in the tag message.
