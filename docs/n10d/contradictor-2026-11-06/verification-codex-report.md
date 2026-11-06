# Vérification en source — rapport contradicteur N10-D (Codex gpt-6.1-sol), 2026-11-06

Vérificateur : agent Claude Opus, 2026-11-06 après-midi. Source auditée :
`snapshot-6e682cc/` (HEAD détaché `6e682cc`, branche PR #100). Chemins `chemin:ligne`
relatifs à ce snapshot sauf mention. `MS` = `docs/milestones/2026-10-25-n10c-bis-control-arms.md`.
Note #102 = `master:docs/prereg/n10d/reading-note.md` ; brainstorm #102 =
`master:docs/prereg/n10d/brainstorm-2026-11-01.md` (PR #102 fusionnée le 2026-11-06 06:06 UTC).
Rien n'a été modifié dans aucun dépôt ; aucun commit.

## Calculs refaits

- **Simulations du nul et `required_seeds`** : lancées sur macM1, clone
  `~/scratch/n10d-verif-0611/nerve-wml` @ `6e682cc`, `.venv/bin/python -B` (pas de
  bytecode, aucun fichier écrit dans le dépôt ; `git status` n'y montre que `M uv.lock`,
  causé par le `uv sync` de l'agent qui a lancé les tests, pas par moi).
  `simulate_null(TaskSpec(c), 8, 500, σ, n_draws=10 000, rng_seed=20261101)` :

  | condition | σ | médiane | P95 | P99 |
  |---|---|---:|---:|---:|
  | asym | 0 | 0,000 | 0,01575 | 0,02275 |
  | asym | 0,05 | −0,00025 | 0,0445 | **0,0625** |
  | sym | 0 | 0 | 0 | **0** |
  | sym | 0,05 | 0 | 0,02425 | **0,0350** |

  k = 5, N = 2 000, σ = 0,05 : P99 asym 0,0764, sym 0,0468. Tout coïncide avec le §4 du
  rapport Codex ; la ligne C = 4 de `nh96-null.md:34-36` (0,023 / 0,062) coïncide aussi.
  `required_seeds(σ, 0.25, 500)` = 5, 5, 5, 5, 18 pour σ = 0 ; 0,02 ; 0,05 ; 0,15 ; 0,30
  (valeur continue 17,22 à σ = 0,30) ; `required_seeds(0, 0.25, 1)` = 48 (valeur continue 47,58).
- **Bug CLI `sigma_seed.py`** reproduit sur macM1 (`python -B -m experiments.n10d.sigma_seed <json>`,
  JSON écrit hors dépôt dans `~/scratch/n10d-verif-0611/codex-cli-*.json`) : 8 lignes
  `nocomm/asym` constantes à 0,25, `n_eval=500` → `sigma_seed=0.0000 k=5` ; mêmes lignes
  précédées d'une ligne `eng` à `n_eval=1` → `sigma_seed=0.0000 k=48`.
- **Arithmétique** (`python3 -I -c`, local) : q⁵ = 0,3277 / 0,5905 / 0,7738 pour
  q = 0,80 / 0,90 / 0,95 ; 0,9^(1/5) = 0,97915 ; contre-exemple M1 : E[D] = 0,30, sd = 0,275 ;
  moyenne latente sym tronquée 1 − 0,05/√(2π) = 0,98005 ; contre-exemple sym 8/16·1 + 8/16·0,25 = 0,625.
- **Tests** : `~/scratch/n10d-verif-0611/tests-6e682cc.txt` (lancé par l'autre agent,
  2026-11-06 12:19 UTC, `uv sync --all-extras` + `uv run pytest tests/test_n10d_sigma.py
  tests/test_n10d_skeleton.py -v --no-cov`, Python 3.14.4, pytest 9.0.3) : **9 passed, exit=0**.
  Le rapport Codex (Python 3.14.8, pytest 9.0.2) disait aussi 9 passants.

## Tableau des constats

| n° | constat (10 mots) | sévérité rapport | verdict | sévérité retenue | preuve |
|---|---|---|---|---|---|
| 1.1 | Nécessité informationnelle idéale déjà tranchée : contraste optimal 0,75 / 0 | MAJEUR | **confirmé** (calcul exact) ; déjà reconnu par le protocole et par #102 | MINEUR | `spec.py:33-40` (plafonds 1/C et 1) ; asym : 0,25 sans message, 1 avec code injectif (16 symboles = 16 cibles, `spec.py:24-26`) ; sym : 1 et 1. Le protocole le dit lui-même : `protocol.md:35-36` (« test of the task and of the channel plumbing », « VARIANT (expected result) ») et `nh96-null.md:60` (sym = 0 par construction). Note #102 l. 37-40 et l. 218-219 : « Ce qui n'est pas ouvert : eng − nocomm sur asym ». Sévérité abaissée : c'est une limite déjà déclarée. La remarque de capacité (4 symboles ne suffiraient pas sans contexte) est exacte. |
| 1.2 | Antériorités directes : Lazaridou, Lowe, Chauhan & Bera | MAJEUR | **confirmé** pour les identifiants et pour Lazaridou/Lowe en texte ; Chauhan & Bera non vérifié en texte par moi, déjà lu par #102 | MINEUR (portée) | Lazaridou p. 4 (§4 « Learning to communicate ») : « agents almost perfectly coordinate », tableau 1 p. 5 : 99-100 % (lu dans `library-biblio/references/arxiv/1612.07182.pdf`) ; NB : jeu à **2** images, pas C = 4. Lowe : phrase « communication may act as an alternate pathway for optimization » à la p. 8 du PDF arXiv (= p. 700 des actes AAMAS, 693-701), §6.2 confirmé. Chauhan & Bera 0,482 contre 1,000 : cohérent avec note #102 l. 79 (lue pp. 1-9). L'antériorité ne change pas le dessin ; déjà intégrée par #102 (§3, §4.4). |
| 1.3 | EGG `signal_game` existe ; pas résultat N10-D | (info) | **confirmé** | OK | `gh api` facebookresearch/EGG @ `54eb11c6` (2025-05-06) : `egg/zoo/signal_game/{README,train,features,archs}.py` ; `train.py:44-45` mode `rf`/`gs`, `train.py:74 get_game`, `train.py:91-94` `ReinforceWrapper`/`SymbolGameReinforce` ; `features.py:15,33` `_BatchIterator.get_batch`, `features.py:72-75` permutation des candidats pour le récepteur. |
| B1 | Porte d'apprentissage impossible sur `sym` (`eng > 1 + P99`) | BLOQUANT | **confirmé** sur la lettre du texte | BLOQUANT avant gel (correction triviale) | `protocol.md:64` « reported next to every result » ; `protocol.md:68-69` « eng above blind ceiling + P99 » ; `spec.py:40` sym → 1,0 ; P99 sym σ = 0 = 0 (recalcul ci-dessus) → exige `eng > 1`. `protocol.md:67` exige aussi que `msg_shuffle` « tombe » au plafond, vide de sens en sym. Partiellement anticipé : `nh96-null.md:60` qualifie déjà sym de « contrôle non informatif », et le PI a décidé « sym sort de H1 » (brainstorm #102 l. 13-15, l. 178) — mais aucune des deux sources ne restreint explicitement la **porte** à asym, et le snapshot garde sym dans H1 (`nh96-null.md:10`, `protocol.md:86`). |
| B2 | Nul simulé mal calibré : effet latent gaussien, sym « quasi parfait » | BLOQUANT | **confirmé** sur les faits et le calcul ; la conséquence est moins forte que BLOQUANT une fois sym retiré de H1 | MAJEUR | `nh96-null.md:12-14,18` ; `null.py:27-29` (clip(p0 + N(0, σ)) puis binomiale) et `null.py:40-42` (deux tirages indépendants par graine) ; σ issu de `nocomm/asym` seul (`sigma_seed.py:4-6,29`). Sous uniformité exacte, toute politique asym aveugle vaut 0,25 : l'effet latent n'a pas de justification dans le modèle du jeu (Codex admet lui-même qu'il est plutôt conservateur en asym). En sym, le nul centré sur 1 tronqué (moyenne 0,98005) suppose deux auditeurs presque parfaits ; P99 sym 0,035 ≠ P99 asym 0,0625 à σ = 0,05 (recalculé). Le critère H1 asym passe par l'IC bootstrap contre 0,30 (`nh96-null.md:50`), pas par ce nul ; le nul ne sert donc qu'à D_sym et aux contrôles. Déjà vu en partie : l'indépendance des bras (covariance de graine) est relevée par note #102 §4.1 l. 136-138 et brainstorm l. 16-19 (correction PI n° 2) ; l'angle « effet latent au-dessus de la borne » et « sym ≠ deux auditeurs parfaits » est nouveau. |
| M1 | Variance `nocomm` et test des signes ne dimensionnent pas D | MAJEUR | **confirmé** | MAJEUR | `sigma_seed.py:39-49` (formule normale de moyenne) contre `protocol.md:75-76` (test des signes unilatéral + bootstrap). Recalcul : k = 5 pour σ ≤ 0,15, 18 pour 0,30. À k = 5, seul 5/5 rejette (p = 1/32 = 0,03125) → puissance q⁵ (0,328 / 0,590 / 0,774) ; 90 % exige q ≥ 0,97915. Contre-exemple : E[D] = 0,30, sd 0,275, puissance 0,32768 — exact. Var(D) = Var(S_eng) + Var(S_nocomm) − 2 Cov : σ_seed de `nocomm` seul ne la détermine pas (`protocol.md:80-81`). Partiellement vu : note #102 l. 259-262 (25 graines appariées chez Chauhan & Bera « à comparer à la puissance visée par P100 ») ; le désaccord test des signes / formule normale n'y est pas. |
| M2 | « Cinq derniers logs » n'est pas une binomiale d'évaluation | MAJEUR | **confirmé** | MAJEUR | `protocol.md:59-61` ; `MS:337` « last 100 completed episodes at each log » ; `null.py:29` une seule binomiale par bras-graine ; `sigma_seed.py:35` soustrait p(1−p)/n_eval. Aucun champ ne relie `n_eval` aux fenêtres de 100 épisodes chevauchantes et non stationnaires ; `interfaces.py:39-40` renvoie « last-5-logs mean » sans `n_eval`. Lié à TBD-11 (`protocol.md:28`). Non vu par #102. |
| M3 | Redondance, nullité du contraste et équivalence confondues | MAJEUR | **confirmé** ; en partie neutralisé par « sym sort de H1 » | MAJEUR (pour l'audit de fuite) | `nh96-null.md:60` (« 0 by construction ») ; `protocol.md:91-92` (D_sym > P99 ⇒ « not symmetric ») ; `nh96-null.md:51-52` admet « or the channel is used when redundant ». Contre-exemple sym 0,625 contre 1 → D_sym = 0,375 sans fuite : arithmétique exacte. Règle unilatérale `D_sym <= P99` (`nh96-null.md:10,51`) : accepte tout effet négatif. Cas A au seuil ponctuel `D_asym >= 0,30` (`protocol.md:86`) contre rejet si borne haute de l'IC < 0,30 (`nh96-null.md:50`) : les deux règles ne coïncident pas (zone D̂ ≥ 0,30 avec IC haut ≥ 0,30 mais IC bas < 0,30 non tranchée). Lakens 2017 vérifié en texte (XML Europe PMC, 1er paragraphe du corps, fpage 355). |
| M4 | Permutation, audit de fuite et seuils insuffisamment définis | MAJEUR | **confirmé** | MAJEUR | `interfaces.py:33-35` (« permutation, fixed seed », rien d'autre) ; `backends.py:8-15` lèvent ; `protocol.md:67,68,70-71` réutilisent « P99 of the null » (quantile d'une **différence moyenne entre bras**) pour des seuils sur **une** moyenne (`msg_shuffle`, `eng`, `nocomm`). La remarque « coïncidence de cible 1/16 n'est pas une fuite » est exacte (K = 16 uniforme). Partiellement vu : note #102 l. 247-250 (dérangement conservant la marginale, Cheng et al.). |
| TBD-10 | Verdict par backend ; conjonction ≠ Bonferroni ; statuts à définir | MAJEUR | **confirmé** (raisonnement correct, aucune erreur de fait) | MAJEUR | `protocol.md:27,109-111`. « Deux non-rejets ne prouvent pas un nul commun » : correct. La règle actuelle ne définit que « agree / disagree → non concluant ». |
| TBD-11 | `simple_reference` JaxMARL ≠ K = 16 / C = 4, récompense mixte | MAJEUR | **confirmé** | MAJEUR | `gh api` bold-lab-ai/JaxMARL (ex-FLAIROx, redirection) @ `31b0a2309d75` (2026-07-20) `jaxmarl/environments/mpe/simple_reference.py` : `:25 num_landmarks=3`, `:31 assert num_landmarks == 3`, `:26 local_ratio=0.5`, `:37 dim_c = 10`, `:161` récompense `global*(1−local_ratio) + agent*local_ratio`. `backends.py:9` dit « simple_reference-like or custom Lewis env ». Le pin `jaxmarl 0.2.0` (`requirements-to-hash.md:10`, 2026-09-10) n'a pas été audité (Codex le dit). `interfaces.py:20` `Sequence[int]` sans longueur : exact. |
| MIN | CLI `sigma_seed.py` lit `n_eval` dans `recs[0]` : k 5 → 48 | MINEUR | **confirmé et reproduit** sur macM1 | MINEUR | `sigma_seed.py:53-56` (`n_eval = recs[0]["n_eval"]`, non filtré) ; reproduction : k = 5 → k = 48. `tests/test_n10d_sigma.py:39-40` ne teste que la fonction `sigma_seed`, pas le `__main__`. Aussi : la ligne extraite n'est pas forcément `nocomm/asym`, et rien ne vérifie que les `n_eval` filtrés sont égaux. |
| OK | Namespaces de graines, non-mélange, refus du sauvetage post hoc | OK | **confirmé** | OK | `protocol.md:21` (pilote 9600+, confirmatoire 600+), `:26`, `:44-45`, `:81-82`, `:107-108` (pas de mélange backends/hôtes), `:23` et `:88-90` (TBD-6), `:96-98` (stubs annoncés), `requirements-to-hash.md:5-6,21` (hashes à produire). |
| R1 | Lazaridou 2017, arXiv:1612.07182 | MAJEUR | **identifiant vérifié** (arXiv API : titre, Lazaridou/Peysakhovich/Baroni, 2016-12-21) ; **page vérifiée** (p. 4 citation, tableau 1 p. 5) | MINEUR | PDF bibliothèque `references/arxiv/1612.07182.pdf`. |
| R2 | Lowe 2019, arXiv:1903.05168 | MAJEUR | **identifiant vérifié** ; **page vérifiée** (§6.2, p. 8 arXiv = p. 700 AAMAS) ; **nuancé** pour « référence [23] p. 701 » : dans le PDF arXiv la référence à Lazaridou 2016 est **[25]** (p. 9) ; numérotation AAMAS non vérifiée | MAJEUR (sur la redondance) | PDF bibliothèque `references/arxiv/1903.05168.pdf`. |
| R3 | Chauhan & Bera 2026, arXiv:2609.34373v2 | MAJEUR | **identifiant vérifié** (arXiv API, v2, 2026-09-28) ; **non vérifié en texte** par moi (pas dans la bibliothèque) ; chiffres 0,482 / 1,000 cohérents avec note #102 l. 79 | MINEUR | — |
| R4 | Zhang, COLING 2025, arXiv:2410.18806 | MAJEUR (portée) | **identifiant arXiv vérifié** (v2) ; actes COLING pp. 1660-1666 non vérifiés par API ; **non vérifié en texte** | MINEUR | — |
| R5 | Kharitonov et al. 2019, EGG | MAJEUR (adaptation) | **vérifié** : Crossref 10.18653/v1/D19-3010, pp. 55-60 ; arXiv:1907.00852 ; **non vérifié en texte** (note 3 p. 56) | MINEUR | — |
| R6 | Hofbauer & Huttegger 2008 | MAJEUR | **vérifié** : Crossref 10.1016/j.jtbi.2008.07.010, JTB 254, 843-849 ; **non vérifié en texte** | MINEUR (contexte) | — |
| R7 | Eccles et al. 2019 | MAJEUR | **identifiant vérifié** (arXiv:1912.05676) ; **non vérifié en texte** par moi ; déjà lu pp. 1-5 par #102 (l. 99, §4.4) | MINEUR | — |
| R8 | Lakens 2017 | MAJEUR | **vérifié** : Crossref 10.1177/1948550617697177, SPPS 8, 355-362 ; citation « the absence of an effect can be rejected, but not statistically supported » trouvée dans le 1er paragraphe du corps (fpage 355) ; section « Testing for Equivalence » présente (page 356 non vérifiable dans le XML) | MAJEUR | `references/doi/10.1177_1948550617697177.xml`. |
| R9 | Bouchacourt & Baroni 2018 | MINEUR | **vérifié** : Crossref 10.18653/v1/D18-1119, pp. 981-985 ; arXiv:1808.10696 ; **non vérifié en texte** | MINEUR | — |
| R10 | Główka et al. 2024 | MAJEUR (spéc.) | **vérifié** : Crossref 10.1016/j.cogsys.2024.101293, CSR 88 ; arXiv:2305.05821 ; **non vérifié en texte** | MINEUR | — |
| R11 | Yamamoto et al. 2026 | MINEUR | **vérifié** : Crossref 10.3389/fcomp.2026.1873726, Front. Comput. Sci. 8 (2026-08-10) ; **non vérifié en texte** | MINEUR | — |
| R12 | Toquebiau 2024, thèse | MINEUR / piste | **vérifié** : API HAL `tel-05076852` (titre, auteur, soutenance 2024-12-13, DOI 10.70675/3385d0bb…) ; NNT non renvoyé par la requête ; résumé seul | MINEUR | Déjà vue et écartée par #102 (l. 328, « non gardée »). |

## Corrections du dessin N10-D que ce rapport justifie (à appliquer par le PI, rien n'est appliqué)

**`docs/prereg/n10d/protocol.md`**
1. §3, contrôles 2 et 3 (l. 67-69) : écrire que la porte d'apprentissage et le contrôle
   `msg_shuffle` portent sur `asym` seulement ; pour `sym`, dire ce qui est rapporté
   (contrôle de fuite, sans seuil « au-dessus du plafond »). (B1)
2. §3 l. 64 « reported next to every result » : préciser la condition de chaque contrôle. (B1)
3. §5 cas A (l. 86) : retirer « D_sym <= P99 » de la condition de confirmation, conformément
   à la décision PI « sym sort de H1 » (brainstorm #102 l. 13-15) ; cas D (l. 91-92) : un
   D_sym > P99 ne doit plus se lire « not symmetric » seul (contre-exemple 0,625) — écrire
   « fuite possible ou effet d'optimisation ; audit requis ». (B1, M3)
4. §5 cas A/B : aligner la règle ponctuelle `D_asym >= 0,30` sur le critère IC de
   `nh96-null.md:50` (dire si A exige la borne basse de l'IC ≥ 0,30, ou D̂ ≥ 0,30 avec
   IC excluant 0, et ce que devient la zone intermédiaire). (M3)
5. §3 contrôles 2-4 : donner à chaque seuil sur **une** moyenne (`eng`, `nocomm`, `msg_shuffle`)
   sa propre statistique et son propre quantile, au lieu du « P99 of the null » de D. (M4)
6. §3 métrique (l. 59-61) et TBD-11 : définir l'évaluation (politique figée, n_eval épisodes
   indépendants après entraînement) ou modéliser la dépendance des 5 logs × 100 épisodes ;
   ne pas laisser `n_eval` implicite. (M2)
7. §4 (l. 75-81) : faire correspondre la règle de k au test effectivement utilisé (test
   des signes → puissance q^k, ou test t/bootstrap apparié sur la moyenne) ; dire d'où vient
   la variance de D (pas seulement σ_seed de `nocomm`). (M1)
8. §7 / TBD-10 : énumérer les statuts par backend (effet, non concluant, non mesurable,
   invalide) et la table de combinaison. (TBD-10)
9. §7 / TBD-11 : remplacer « simple_reference-like » (`backends.py:9`) par une spécification
   explicite du jeu JaxMARL (K, C, longueur de message, ordre émission/action, récompense
   commune identique) ; `simple_reference` impose 3 landmarks, `dim_c=10`, `local_ratio=0,5`. (TBD-11)

**`docs/prereg/n10d/nh96-null.md`**
10. §1 H1 (l. 10) et §4 critère 2 (l. 51-52) : retirer D_sym de H1 (décision PI) ou, si
    gardé, le remplacer par un test d'équivalence à bornes explicites (Lakens 2017). (M3)
11. §2 (l. 18) : nul apparié (effet de graine commun aux deux bras ou permutation des bras
    dans la graine — déjà décidé, brainstorm #102 l. 16-19) ; justifier ou supprimer l'effet
    latent gaussien en asym ; ne pas réutiliser pour sym un nul « deux auditeurs à 1 ». (B2)
12. §3 : l'en-tête dit « asym » ; ajouter que P99 sym ≠ P99 asym (0,035 contre 0,0625 à
    σ = 0,05, k = 8) si sym reste rapporté. (B2)
13. §5 l. 60 : « 0 by construction » vaut pour des politiques optimales, pas pour des
    politiques entraînées à budget fini. (M3)

**`experiments/n10d/sigma_seed.py`**
14. l. 55 : prendre `n_eval` dans les lignes filtrées `nocomm/asym` et vérifier qu'elles
    portent toutes le même `n_eval` (sinon erreur). (MINEUR)
15. l. 39-49 : docstring / formule à aligner sur la correction 7. (M1)

**`experiments/n10d/interfaces.py`**
16. l. 33-35 : spécifier le dérangement (permutation sans point fixe **d'épisodes**, à
    l'intérieur du lot courant, conservant la marginale et la longueur des messages ; pas
    d'exclusion d'étiquettes identiques). (M4)

**`tests/test_n10d_sigma.py`**
17. Ajouter un test du chemin CLI (`__main__`) avec une ligne d'un autre bras à `n_eval`
    différent en tête. (MINEUR)

## Ce que le contradicteur n'a pas vu (déjà répondu par la note #102 ou le brainstorm #102)

- **1.1 / question déjà tranchée** : note #102 l. 35-40 et l. 218-219 dit déjà que
  `eng − nocomm` sur asym est « attendu par construction » et « pas ouvert » ; la question
  ouverte retenue est l'usage causal du bras `learned` (§4.1, §5 dessin D1/D2). Le
  brainstorm (l. 176-177) l'inscrit dans « ce qu'on ne fait pas ».
- **B1 / M3 (sym)** : décision PI « `sym` sort de H1 » (brainstorm #102 l. 13-15, l. 178 ;
  note #102 l. 233-234), relevée à l'origine en `RAP:94` (M9). Elle n'est pas encore
  reportée dans PR #100 ; le contradicteur, aveugle, ne pouvait pas la voir. Elle neutralise
  la moitié de B1 et de M3 mais pas la porte d'apprentissage écrite pour « every result ».
- **B2 (indépendance des bras)** : correction PI n° 2, nul apparié (brainstorm #102
  l. 16-19 ; note #102 §4.1 l. 136-138). L'angle propre à Codex (effet latent au-dessus de
  0,25 ; sym non parfait) reste nouveau.
- **1.2 / R3, R7 (canal inutilisé ≠ inutilisable)** : note #102 §4.4 (l. 201-216) et §5
  (l. 255-258) tirent déjà de Chauhan & Bera et d'Eccles la nécessité de séparer « canal
  inutilisé » et « canal inutilisable ».
- **M1 (puissance)** : note #102 l. 259-262 signale déjà 25 graines appariées chez
  Chauhan & Bera comme ordre de grandeur à comparer à la puissance de P100 ; le désaccord
  test des signes / formule normale, lui, n'y figure pas.
- **M4 (permutation)** : note #102 l. 247-250 recommande le dérangement qui conserve la
  marginale (Cheng et al.), et le contraste « zéro mêle présence et contenu ».
- **R12 Toquebiau** : déjà vue au résumé et écartée (note #102 l. 328).
- **Non vus par #102 non plus** (apports nets du rapport) : impossibilité littérale de la
  porte en sym (B1), dépendance des 5 logs × 100 épisodes (M2), test des signes à k = 5
  (M1), incohérence règle ponctuelle / règle IC (M3), réutilisation du P99 de D pour des
  seuils sur une moyenne (M4), bug CLI, écart `simple_reference` (TBD-11).

## Non vérifié

- Textes de Chauhan & Bera, Zhang (COLING), Kharitonov, Hofbauer & Huttegger, Eccles,
  Bouchacourt & Baroni, Główka, Yamamoto, Toquebiau : absents de `library-biblio` →
  pages citées **non vérifiées en texte** (identifiants vérifiés par API).
- Numérotation [23] de Lowe dans la version AAMAS ; actes COLING 2025 pp. 1660-1666.
- Requêtes de recherche négative du §4 Codex (OpenAlex, OpenReview 403, Semantic Scholar) :
  non rejouées.
