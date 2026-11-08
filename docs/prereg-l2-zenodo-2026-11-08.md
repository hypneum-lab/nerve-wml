# Zenodo L2 deposit — nerve-wml pre-registrations (2026-11-08)

The restricted Zenodo record
[10.5281/zenodo.24053908](https://doi.org/10.5281/zenodo.24053908) (concept DOI
[10.5281/zenodo.24053907](https://doi.org/10.5281/zenodo.24053907)) holds the
seven frozen pre-registration zips for this repository. Access is restricted;
metadata are public; authenticated users may request access. The record is
licensed CC-BY-4.0 and was published on 2026-11-08 as a PI act
(Clément Saillant, ORCID 0000-0002-8414-185X). It constitutes level L2 of the
Hypneum Lab evidence chain defined in
[`hypneum-lab/.github` `docs/PROJECT-MANAGEMENT.md`](https://github.com/hypneum-lab/.github/blob/main/docs/PROJECT-MANAGEMENT.md)
section 4 (L1 = annotated tag + Software Heritage archive; L2 = verifiable
deposit date via a third-party repository; L3 = journal registration). Each zip
is a snapshot of this repository at the corresponding annotated tag, produced by
`git show <tag>:<path>` for every document frozen at that tag.

## Frozen pre-registrations

| Study | Tag | Commit | SWHID | Zip | SHA-256 | Documents in the zip |
|---|---|---|---|---|---|---|
| NH-01 pilot (dreaming kernel vs uniform shrink) | `prereg-nh01-hopfield-dream-20261025` | `bc2f838919756a8dc2249e577cc5fac0f21cf0d2` | `swh:1:rev:bc2f838919756a8dc2249e577cc5fac0f21cf0d2` | `prereg-nh01-hopfield-dream-20261025.zip` | `91b53d196634b27b02bcf8ad4bf304594b4ea233e688c10ad4e4526162480d29` | `docs/milestones/2026-10-25-nh01-hopfield-dream-vs-downscale.md` |
| NH-01 confirmatory N=1000 | `prereg-nh01-confirmatory-20261025` | `a000f73addde8ff8ac0f28e72a038ad886848f48` | `swh:1:rev:a000f73addde8ff8ac0f28e72a038ad886848f48` | `prereg-nh01-confirmatory-20261025.zip` | `09d1f8bb9f12950003930b58ef1a42b02d24df0dd8c3399a3ea5a9559375b633` | `docs/milestones/2026-10-25-nh01-confirmatory-n1000.md` |
| NH-01 correlated patterns N=1000 | `prereg-nh01-correlated-20261027` | `df9a1e7634113599c9b38cb4259068a29a5bdd2a` | `swh:1:rev:df9a1e7634113599c9b38cb4259068a29a5bdd2a` | `prereg-nh01-correlated-20261027.zip` | `229e3a7a24e9e0f728df0f8b0f2b5726b9689acc434679e05a95bb08c4ee4598` | `docs/milestones/2026-10-27-nh01-correlated-patterns.md` |
| NH-01 diagonal ablation N=1000 | `prereg-nh01-diagonal-20261031` | `6e1b0069c49750f7cb2a01dff832a257c6157a6e` | `swh:1:rev:6e1b0069c49750f7cb2a01dff832a257c6157a6e` | `prereg-nh01-diagonal-20261031.zip` | `4a6a307c81167fdee1c2a5ba388943dbcdac630fb29a4c75401208ec0634877a` | `docs/milestones/2026-10-31-nh01-diagonal-ablation.md` |
| P1bis-ext (dreaming vs two budget-matched controls, clipped Hopfield) | `prereg-hopfield-p1bis-p8c-freeze-20261104` | `257e4698cadd6786d2381cf61d2eef4cdbeb4475` | `swh:1:rev:257e4698cadd6786d2381cf61d2eef4cdbeb4475` | `prereg-hopfield-p1bis-p8c-freeze-20261104.zip` | `fae1997be8957b6bcc063c12ffbf24fc8dc7b5a162a65c2365da8c2bc1b89772` | `docs/prereg/hopfield-p1bis-ext/prereg.md`, `docs/prereg/hopfield-p1bis-ext/freeze-manifest.md` |
| P8c-nocap (dreaming kernel vs Hebbian unlearning without dose cap, relaunch) | `prereg-hopfield-p8c-relaunch-20261105` (also covers `prereg-hopfield-p1bis-p8c-freeze-20261104`) | `db4eb5ba5644df8dc5622f53f249c19d52a9d82b` (freeze: `257e4698cadd6786d2381cf61d2eef4cdbeb4475`) | `swh:1:rev:db4eb5ba5644df8dc5622f53f249c19d52a9d82b` | `prereg-hopfield-p8c-relaunch-20261105.zip` | `b561ce5b4a8b40b1688113be2e2212f3c5921fea4052dc613cfb1f3937b6fb75` | `docs/prereg/hopfield-p1bis-ext/freeze-manifest.md`, `docs/prereg/hopfield-p8c-nocap/prereg.md`, `docs/prereg/hopfield-p8c-nocap/amendment-2026-11-05.md`, `docs/prereg/hopfield-p8c-nocap/amendment-2026-11-05b.md` |
| N10-C-bis G7 indicative pilot (GB300) | `n10c-bis-g7-gb300-freeze-20261031` | `44364dc9820056d808a5a88c9e454bce10c91484` | `swh:1:rev:44364dc9820056d808a5a88c9e454bce10c91484` | `n10c-bis-g7-gb300-freeze-20261031.zip` | `3457c8625432bb00645d53d2cab4ae529f48ee905932eb091ca4974de1de2b6f` | `docs/prereg/n10c-bis-gate-gb300/freeze-manifest.md`, `docs/prereg/n10c-bis-gate-gb300/g7-host-annex.md`, `docs/prereg/n10c-bis-gate-gb300/requirements-review.md` |

## Files not in the deposit (created after the tag)

The following files exist on `master` but were committed after the freeze tag and
are therefore absent from the deposit.

| Tag | File absent from the deposit |
|---|---|
| `prereg-hopfield-p1bis-p8c-freeze-20261104` | `docs/prereg/hopfield-p1bis-ext/amendment-2026-11-06-reading-gate.md` |
| `prereg-hopfield-p8c-relaunch-20261105` | `docs/prereg/hopfield-p8c-nocap/amendment-2026-11-06-reading-gate.md` |
