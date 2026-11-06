# Vérification en source des deux rapports de contradicteur (2026-11-06, 13:20-13:35 CET)

Méthode : un vérificateur (Claude Opus, contexte vierge) par rapport, sur l'instantané `6e682cc` (worktree HEAD détaché), avec accès cette fois à la note de lecture et au brainstorm de la PR #102 (fusionnée) pour dire ce qui était déjà connu. Chaque constat est rouvert à la ligne citée ; tout ce qui est recalculable a été recalculé sur macM1 (`~/scratch/n10d-verif-0611/`, clone jetable). Détail : `verification-claude-report.md`, `verification-codex-report.md`.

## Décompte

| rapport | constats | confirmés | nuancés | réfutés | sévérité retenue |
|---|---:|---:|---:|---:|---|
| Claude Opus | 14 | 12 | 2 (1.1 déclassé BLOQUANT→MAJEUR ; 2.2 k = 9 et non 18 si σ_D est pris directement) | 0 | 1 BLOQUANT (2.1), 7 MAJEUR, 4 MINEUR |
| Codex sol | 17 + tableau §3 | 14 | 3 (B2 BLOQUANT→MAJEUR ; 1.1 et 1.2 MAJEUR→MINEUR) | 0 | 1 BLOQUANT (B1), 7 MAJEUR, 3 MINEUR |

Les 9 tests N10-D passent à `6e682cc` (`uv sync --all-extras`, 9 passed). Ils ne couvrent que `asym`.

Chiffres reproduits à l'identique par les deux vérificateurs : contraste optimal 0,75 (`asym`) / 0 (`sym`) ; P99 du nul à σ = 0,05, k = 8 : 0,0625 (`asym`), 0,035 (`sym`), 0 à σ = 0 ; `required_seeds` k = 5 jusqu'à σ = 0,15, 18 à σ = 0,30, 48 si `n_eval` = 1 ; puissance du test des signes à k = 5 : q⁵ = 0,59 à q = 0,9 ; σ̂ = 0 dans 57 % des pilotes (P(χ²₇ < 7) = 0,571) ; bug CLI `sigma_seed.py:55` reproduit (k 5 → 48 avec une ligne `eng` à `n_eval` = 1 en tête) ; `simple_reference.py` JaxMARL = 3 landmarks, `dim_c` = 10, `local_ratio` = 0,5.

## Un seul BLOQUANT, commun aux deux rapports

Le contrôle 3 / porte d'apprentissage (`protocol.md:64,68-69`, `spec.py:40`) exige `eng > blind ceiling + P99` sur **tous** les bras, donc `eng_sym > 1`, impossible. L'issue A (`protocol.md:86`) est inatteignable à la lettre. Correction triviale (restreindre à `asym`, porte propre pour `sym`), mais à faire avant tout gel.

## Déjà décidé par le PI et non encore reporté dans la PR #100

- « `sym` sort de H1 » (brainstorm #102 l. 13-15) : règle la moitié de B1/2.1, 2.3, M3.
- Nul apparié par graine (brainstorm #102 l. 16-19) : règle B2/2.2 dans son principe ; le nul de `nh96-null.md:40-43` tire encore les deux bras indépendamment.
- « nécessité tranchée par construction » (1.1) : déjà écrit dans `protocol.md:35-36` et note #102 l. 35-40, 218-219.

## Apports nets des contradicteurs (ni la note ni le brainstorm ne les avaient)

Porte impossible en `sym` (B1/2.1) ; métrique « 5 logs × 100 épisodes » non binomiale (M2/2.7) ; test des signes à k = 5 et puissance (M1/2.5) ; règle ponctuelle contre règle IC (M3/2.4) ; P99 de D réutilisé pour des seuils sur des moyennes (M4) ; bug CLI ; écart `simple_reference` (TBD-11) ; contre-exemple `sym` à 0,625 ; un succès N10-D ne remonte pas à N10-C sans bras exécuté dans le harnais N10-C (2.6) ; marge 0,30 non reliée à l'oracle 0,75 (2.9).

## Trouvé par la vérification seulement

- `msg_shuffle` à « fixed permutation » (`protocol.md:57`) : si la permutation porte sur les identités de cible, le message reste une fonction déterministe de la cible, un auditeur l'apprend jusqu'à 1,0 et le contrôle 2 échoue sans fuite. Il faut un dérangement **d'épisodes** dans le lot (`interfaces.py:33-35`).
- H0 de `nh96-null.md:12-13` ne définit le nul que pour `asym`.
- `required_seeds` double la variance (`sigma_seed.py:47`) en supposant deux bras indépendants ; avec un σ_D mesuré directement, k = 9 et non 18 pour un D bimodal.
- Lowe 2019 arXiv cite Lazaridou en [25], pas [23].

## Corrections à porter dans la PR #100 (fusion des deux listes, 19 points)

`protocol.md` : §3 contrôles 2-3 restreints à `asym`, porte propre `sym` ; §3 l. 57 nature de la permutation ; §3 l. 59-61 et TBD-11 évaluation définie (politique figée, `n_eval` indépendants) ; §3 contrôle 4 audit a priori du générateur ; §4 règle de k alignée sur le test utilisé et σ pris sur `S_eng`/`asym` ; §5 A sans `D_sym`, « plumbing and trunk learn the code » au lieu de « necessity confirmed », exiger le rejet du test ; §5 D « fuite possible ou effet d'optimisation » ; §5 A/B une seule règle (IC ou ponctuelle) ; §2 l. 42-44 conditionner la lecture « Procgen n'a pas besoin du message » ; §7 TBD-10 statuts par backend ; TBD-11 jeu JaxMARL spécifié.
`nh96-null.md` : §1 H1 sans `sym`, H0 explicite ; §2 nul apparié ; §3 P99 `sym` ≠ `asym` ; §4 un seul jeu de critères ; §5 « 0 by construction » = politiques optimales seulement.
`sigma_seed.py` : l. 55 `n_eval` des lignes filtrées ; l. 29 et 39-49 écart-type de `eng` et formule sans facteur 2 si σ_D direct.
`interfaces.py:33-35` : dérangement d'épisodes.
`tests/` : nul `sym` (P99 = 0 à σ = 0) ; chemin CLI avec `n_eval` hétérogène.

Rien n'est appliqué dans la PR #100 à ce stade.
