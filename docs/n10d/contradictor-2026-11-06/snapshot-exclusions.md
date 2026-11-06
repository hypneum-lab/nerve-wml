# Instantané N10-D — exclusions (2026-11-06)

Dépôt : `hypneum-lab/nerve-wml` (GitHub), clone frais `--no-tags`, branche de la PR #100
(`prereg/n10d-skeleton`) extraite puis HEAD détaché.

Commit de l'instantané : `6e682cc0bb67b14d0854d2eb10242a04e0607d12`
(« style(n10d): wrap long test line for ruff E501 », 2026-10-31).

## Refs retirées

- Branches locales `master` et `prereg/n10d-skeleton` supprimées (HEAD détaché sur le commit ci-dessus).
- Remote `origin` supprimé : toutes les refs distantes (dont `origin/docs/n10d-reading-note`,
  `origin/docs/n10c-closure-20261105`, etc.) ont disparu ; aucun tag (clone `--no-tags`).
- `git reflog expire --expire=now --all` + `git gc --prune=now` : objets inaccessibles purgés
  (`git fsck --unreachable` vide ; `git log --all -- '*reading*' '*brainstorm*'` vide).

## Fichiers retirés de l'arbre de travail (grep -ril reading-note / brainstorm / catalogue)

Supprimés (documents qui renvoient au catalogue d'hypothèses ou à un brainstorm) :
- `docs/results/2026-10-25-nh01-pilot.md`
- `docs/results/2026-10-25-nh01-confirmatory.md`
- `docs/milestones/2026-10-25-nh01-hopfield-dream-vs-downscale.md`
- `docs/milestones/2026-10-25-nh01-confirmatory-n1000.md`
- `docs/milestones/2026-10-27-nh01-correlated-patterns.md`
- `docs/superpowers/specs/2026-04-18-nerve-wml-design.md` (« brainstorm complete »)

Pas de `docs/research/` dans ce dépôt (il existe `docs/research-notes/` et
`docs/superpowers/research/`, notes techniques anciennes sans rapport avec N10-D, conservés).

## Correspondances conservées volontairement (et pourquoi)

- `docs/prereg/n10d/protocol.md`, `docs/prereg/n10d/nh96-null.md` : objet même de l'attaque.
  Ils citent une phrase du catalogue (énoncé de NH-96 : « Nul qui peut échouer… ») et un
  chemin vers `.github/.../priorites-hypotheses.md` (non fourni). Le protocole contient aussi
  l'étiquette « VARIANT (expected result, Lowe 2019) » : c'est un énoncé du protocole, pas une
  conclusion sur des données N10-D (aucun run N10-D n'existe).
- `experiments/nh01_hopfield_dream/confirmatory.py`, `tests/unit/test_paper_v3_integrations.py` :
  code/tests, occurrence incidente du mot « catalogue » (NH-01, sans rapport avec N10-D).
- Docs N10-C (`docs/milestones/2026-10-25-n10c-bis-control-arms.md`, amendement du 2026-10-30) :
  résultats/protocoles de la campagne précédente qui définissent l'origine de N10-D (porte de
  nécessité) ; ce sont des documents de protocole, pas des conclusions sur N10-D.

## Limite connue

Les fichiers supprimés restent accessibles dans l'historique git du commit (`git show HEAD:<path>`) ;
ils concernent NH-01, pas N10-D. Les messages de commit de la branche (`docs(n10d): …`) sont visibles.
