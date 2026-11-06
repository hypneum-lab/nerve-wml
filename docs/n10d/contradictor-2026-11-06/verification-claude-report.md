# Vérification en source du rapport contradicteur N10-D — passe Claude (06/11)

Vérificateur : agent Claude Opus 5.5, 2026-11-06 (après-midi).
Rapport vérifié : `n10d-verif/nerve-wml/docs/n10d/contradictor-2026-11-06/contradictor-report-n10d-2026-11-06-claude.md` (noté `RC`).
Source auditée : `n10d-verif/snapshot-6e682cc/` (HEAD `6e682cc`, branche PR #100). Abréviations :
`P` = `docs/prereg/n10d/protocol.md`, `N` = `docs/prereg/n10d/nh96-null.md`,
`SP` = `experiments/n10d/spec.py`, `NU` = `experiments/n10d/null.py`,
`SS` = `experiments/n10d/sigma_seed.py`, `MS` = `docs/milestones/2026-10-25-n10c-bis-control-arms.md`.
Note de lecture PR #102 (fusionnée 2026-11-06 06:06 UTC, `master` 5e7a809) : `RN` = `docs/prereg/n10d/reading-note.md`, `BS` = `docs/prereg/n10d/brainstorm-2026-11-01.md` (lus via `git show master:<chemin>`).

## 0. Exécutions

**Tests (point 3 du brief).** macM1 (joint via le LAN `192.168.0.176` parce que le chemin Tailscale ne répondait pas, rx 0 ; clé d'hôte vérifiée contre l'entrée connue `macm1.tail78ae15.ts.net`). Clone frais `~/scratch/n10d-verif-0611/nerve-wml`, `git checkout 6e682cc` (= `6e682cc0bb67b14d0854d2eb10242a04e0607d12`), `uv sync --all-extras` (comme `.github/workflows/test.yml:38`), puis `uv run pytest tests/test_n10d_sigma.py tests/test_n10d_skeleton.py -v --no-cov`.
Résultat : **9 passed in 0.10 s**, exit 0 (Python 3.14.4, pytest 9.0.3, uv 0.11.19). Journal : macM1 `~/scratch/n10d-verif-0611/tests-6e682cc.txt`.
Note : un premier `uv run python -m pytest` sans `--all-extras` échoue (« No module named pytest ») ; pytest est dans un extra. `uv sync` a réécrit `uv.lock` dans le clone jetable de macM1 (rien de poussé).
Fait : les tests ne couvrent que `asym` (`tests/test_n10d_skeleton.py:19-23`). Aucun test n'exerce le nul `sym`, qui est dégénéré (2.3).

**Recalculs.** Script `scratchpad/scripts-0611/recalc_n10d.py` (hors dépôt), exécuté sur macM1 avec le code de `6e682cc` ; sortie `~/scratch/n10d-verif-0611/recalc-6e682cc.txt`. Arithmétique courte en `python3 -I -c` en local.

## 1. Tableau de vérification

| n° | constat (10 mots) | sévérité rapport | verdict | sévérité retenue | preuve |
|---|---|---|---|---|---|
| 1.1 | Contraste primaire fixé par construction (0,75 / 0), sans entraînement | BLOQUANT (formulation) | **confirmé, déjà connu** | MAJEUR | `SP:33-40` (plafond 1/C ; 1,0 en `sym`). Recalcul : oracle − aveugle = 0,75 (`asym`) et 0,0 (`sym`). Eccles 2019 p. 6, §4.1 : la citation « calculating the optimal policies explicitly. The reward the listener can gain … Pl(πs) = R(πlc, πs) − R(πlnc, πs) » est exacte. Rita 2022 p. 4, §3 : la citation « merely corresponds to the optimization of its supervised task » est exacte. Mais `P:35-36` dit déjà « test of the task and of the channel plumbing … VARIANT (expected result) ». `RN:37-40` et `RN:218-219` (« ce qui n'est pas ouvert ») et `BS` §4 le disaient aussi. Seule l'étiquette « necessity confirmed » (`P:86`) dépasse ce que le dessin établit. C'est une correction de formulation, pas un défaut qui invalide le test, d'où MAJEUR. |
| 1.2 | Bras `learned` : résultats publiés dans les deux sens | MAJEUR | **confirmé** | MAJEUR | Lazaridou 2017 p. 4 (« almost perfectly coordinate ») et Table 1 p. 5 (succès 99-100 %, cible + 1 distracteur p. 3). Eccles 2019 Table 1 p. 7 : sans biais, 0,00 de bons runs, IC 0,00-0,07. La phrase « final reward of 0.1 is exactly what can be achieved … with no communication » est en p. 6. MADDPG p. 8 : « the listener learns to ignore the speaker », et AC/TRPO/DDPG « all fail ». `learned` est secondaire (`P:56`, `P:61`) et n'a aucun seuil en `P` §5 : constat exact. Le rapport ne le savait pas, mais `RN` §4-5 et `BS` D2 font déjà de `learned` contre `nocomm` la question ouverte. |
| 1.3 | EGG/JaxMARL fournissent le même jeu prêt à l'emploi | MINEUR | **confirmé** | MINEUR | API GitHub : EGG `54eb11c6b4ba…`, 2025-05-06, « Merge pull request #267 ». Le dossier `egg/zoo/basic_games/` existe, et README:3 décrit le « discrimination game » en ordre aléatoire. JaxMARL `jaxmarl/environments/mpe/simple_speaker_listener.py` et `simple_reference.py` existent. Les courbes des annexes JaxMARL ne sont pas vérifiées en texte. |
| 2.1 | Contrôle 3 impossible en `sym`, issue A inatteignable | BLOQUANT (correction triviale) | **confirmé** | BLOQUANT (pour le gel) | `P:68` : le contrôle 3 n'est pas restreint à une condition, et le plafond `sym` vaut 1,0 (`P:62`). `eng` ≤ 1 < 1 + P99 dès que P99 > 0. L'issue A exige une lecture de D_sym (`P:86`). Le contrôle 4 est, lui, explicitement restreint à `asym` (`P:70`), ce qui confirme l'oubli. La décision PI n° 1 (`BS:13-15`, « `sym` sort de H1 ») retire déjà `sym` de H1. Elle n'est pas appliquée à `6e682cc` (`P:86`, `N:10`), et le contrôle 3 doit de toute façon être restreint. |
| 2.2 | `sigma_seed` mesuré sur un bras sans variance structurelle | MAJEUR | **confirmé (chiffres reproduits), nuancé sur k = 18** | MAJEUR | `SS:29` ne lit que `nocomm`/`asym`. Une politique aveugle a une espérance 1/C exacte (`SP:36-37`) tant que l'ordre des candidats est aléatoire. Simulation (2 000 pilotes, 8 graines, n = 500, p = 0,25, σ = 0, estimateur `sigma_seed()`) : σ̂ = 0 dans **57,2 / 55,9 / 57,7 %** des cas (3 graines rng), P95 = 0,019. En théorie, P(χ²₇ < 7) = 0,571. `required_seeds` : k = 5 pour σ ≤ 0,15 (n = 500), 8 à 0,2, 12 à 0,25, 18 à 0,3. Pour n = 50 : 5 jusqu'à 0,1, puis 6, 9, 13, 19. Nuance : avec 20 % de graines en échec, l'écart-type de D par graine vaut 0,75·√0,16 = 0,30. C'est déjà σ_D, et la formule donne alors k = ⌈8,56⌉ = 9. Le rapport injecte 0,30 comme σ_seed, ce qui donne σ_D = 0,425 et k = 18. Dans les deux cas k > 5 : la conclusion tient. Le nul `asym` sous H0 n'est pas faux (sous H0 les deux bras sont aveugles). C'est le **calcul de puissance** qui est mal calibré, et `N:23-25` le reconnaît en partie. |
| 2.3 | Le nul de `sym` est dégénéré (P99 = 0) | MAJEUR | **confirmé exactement** | MAJEUR (BLOQUANT tant que `sym` reste dans H1) | Recalcul `simulate_null(TaskSpec("sym",16,4),8,500,σ)` : P99 = 0,0000 (σ = 0), 0,0140 (σ = 0,019), 0,0145 (σ = 0,02), 0,0350 (σ = 0,05). À σ = 0, un D_sym = 0,01 a p = 1/10 001 = 0,0001 : il déclenche à la fois l'issue D (`P:91`) et le critère `N:51-52` (p ≤ 0,01). Au P95 du pilote (σ = 0,019), D_sym = 0,01 a p = 0,046 et ne déclenche rien. Le transport de σ d'`asym` vers `sym` est confirmé : `N:21-22` mesure `nocomm`/`asym` et l'applique au nul `sym` (`P:86`). Fait supplémentaire : `N:12-13` ne définit H0 que pour `asym` (« B0 (asym 1/C) »). Le nul `sym` n'existe que dans le code (`NU:37`). |
| 2.4 | Critères `protocol.md` contre `nh96-null.md` incohérents ; puissance mal ciblée | MAJEUR | **confirmé sur le principe, exemple nuancé** | MAJEUR | `P:75-82` : test des signes et bootstrap, non-rejet = « non concluant ». `P:86-88` : issues A/B/C sur la valeur ponctuelle de D_asym. `N:50` : rejet si la borne supérieure de l'IC 95 % est < 0,30. `SS:44` : puissance pour D > 0 à D = 0,30. Si le vrai D vaut 0,30, P(D̂ ≥ 0,30) ≈ 0,5 (normale symétrique) : exact. L'exemple du rapport (D̂ = 0,20, borne sup. 0,28 → B et « rejetée ») n'est pas une contradiction franche : les deux disent « en dessous de la prédiction ». Deux cas sont plus nets. (a) D̂ = 0,32 avec 4/5 signes positifs : p = 6/32 = 0,19, donc test non concluant (`P:82`), mais issue A « necessity confirmed » (`P:86`). (b) D̂ = 0,25 avec borne sup. 0,40 : issue B « weaker » mais NH-96 non rejetée (`N:50`). |
| 2.5 | k = 5 fragilise test des signes et bootstrap | MAJEUR | **confirmé** | MAJEUR | À k = 5, le seul rejet possible à α = 0,05 est 5/5 positifs (p = 1/32 = 0,031 ; 4/5 donne 0,1875). Bootstrap : C(9,5) = **126** rééchantillons distincts. Avec 20 % d'échecs (signe aléatoire), P(5/5) = 0,9⁵ = **0,590**. Le rapport n'appuie la sous-couverture du percentile à n petit sur aucune source (il le dit). `RN:259-262` relevait déjà 25 graines appariées comme ordre de grandeur publié (Chauhan & Bera). |
| 2.6 | Retour vers N10-C non soutenu (autre harnais) | MAJEUR | **confirmé** | MAJEUR | `P:42-44` : l'échec de la porte est lu comme « the Procgen task does not need the message ». `P:103-108` : JaxMARL et EGG, tronc « adapted per backend », TBD-11 ouvert (`P:28`). Rien dans `P` n'exécute un bras dans le harnais N10-C. En partie anticipé : `BS` D7 (`BS:140-151`) propose de rejouer la porte à 3 graines dans Procgen, ce qui teste la tâche Procgen mais pas la tuyauterie via N10-D. |
| 2.7 | Métrique d'entraînement contre nul binomial n_eval | MINEUR | **confirmé, en partie connu** | MINEUR | `P:59-60` et `MS:337` : moyenne des 5 derniers logs, chacun sur les 100 derniers épisodes. `NU:29` : Binomial(n_eval). `n_eval` est explicitement ouvert (TBD-11, `P:28`). Le recouvrement des fenêtres dépend de l'intervalle de log, qui n'est pas fixé : il n'est ni vérifiable ni réfutable à `6e682cc`. |
| 2.8 | `msg_shuffle` ambigu ; audit de fuite à faire en amont | MINEUR | **confirmé** | MINEUR | `P:57` : « message from another episode (fixed permutation) », sans dire s'il s'agit d'un entraînement ou d'une évaluation. Espérance à l'évaluation avec un auditeur `eng` : 1/K + (1 − C/K)·q = 0,0625 + 0,75·q, soit 0,25 seulement si q = 1/C (recalculé). Lowe 2019 p. 8, §6.2 : la phrase « setting the messages to 0 … may cause them to fail » est exacte, et la mise en garde vaut pour un `msg_shuffle` évalué. EGG README:43 (@54eb11c6) : la position de la cible est **lue dans le fichier de données** ; le générateur doit donc être audité. Voir aussi §3 ci-dessous (ambiguïté « fixed permutation »). |
| 2.9 | Marge 0,30 non reliée à l'oracle 0,75 | MINEUR | **confirmé** | MINEUR | Un auditeur à 55 % donne D = 0,55 − 0,25 = 0,30, ce qui passe l'issue A (`P:86`). La marge vient de TBD-4 (`P:21`). |
| §3 | 19 identifiants d'antériorités résolus | — | **confirmé (identifiants)** | OK | API arXiv (06/11, 2 appels groupés) : les 16 identifiants arXiv concordent en titre et premier auteur, y compris `physics/0509075` (5 auteurs selon arXiv). Crossref : les 4 DOI concordent (Argiento 2009 SPA, Huttegger 2014 PNAS, Pawlowitsch 2008 GEB, Peters 2025 JAAMAS). Le contenu n'est vérifié en texte que pour Lowe 2019, Lazaridou 2017, Eccles 2019, Rita 2022 et MADDPG ; les autres sont « non vérifié en texte ». |
| §4 | Tests non lus ni exécutés | — | **comblé** | OK | 9/9 passent (§0). Ils ne testent ni `sym` ni la puissance (k), donc ils ne contredisent ni ne soutiennent 2.2-2.5. |

**Décompte retenu** : BLOQUANT 1 (2.1 ; 2.3 le devient tant que `sym` reste dans H1) ; MAJEUR 7 (1.1, 1.2, 2.2, 2.3, 2.4, 2.5, 2.6) ; MINEUR 4 (1.3, 2.7, 2.8, 2.9). Aucun constat réfuté. 1.1 est déclassé parce qu'il était déjà écrit dans `P:35-36` et dans `RN`. 2.2 est nuancé sur k = 18 (k = 9 si σ_D est pris directement). L'exemple de 2.4 est remplacé par deux cas plus nets.

### Littérature : provenance des PDF

| réf. | source | SHA-256 (préfixe) | page vérifiée |
|---|---|---|---|
| Lazaridou 2017, arXiv:1612.07182 | bibliothèque `library-biblio/references/arxiv/1612.07182.pdf` (INDEX l. 21) | — | p. 3 (2 images), p. 4, Table 1 p. 5 |
| Lowe 2019, arXiv:1903.05168 | bibliothèque `…/arxiv/1903.05168.pdf` (INDEX l. 36) | — | p. 8 §6.2 (« sufficient indicator » ; « setting the messages to 0 ») |
| Eccles 2019, arXiv:1912.05676 | arXiv, `library-dl/2026-11-06-n10d-verif/1912.05676.pdf`, 16 p. | `31ba309d86acd30a` | p. 6 §4.1 ; Table 1 p. 7 |
| Rita 2022, arXiv:2209.15342 | arXiv, même répertoire, 28 p. | `d84641c6f208a1e4` | p. 4 §3 |
| Lowe 2017 MADDPG, arXiv:1706.02275 | arXiv, même répertoire, 16 p. | `dcbf258875023a47` | p. 8 §6 |
| JaxMARL, arXiv:2311.10090 | identifiant seul (API arXiv) | — | non vérifié en texte (annexes) |

## 2. Corrections du dessin N10-D que ce rapport justifie (non appliquées)

**`docs/prereg/n10d/protocol.md`**
- §0 TBD-5 (`P:25`) et §5 issue A (`P:86`) : retirer « and D_sym <= P99 of its calibrated null » de l'issue A (décision PI n° 1, `BS:13`). L'issue D (`P:91-92`) devient un rapport de fuite, plus une issue de H1.
- §3 contrôle 3 (`P:68`) : « `eng` above `blind ceiling + P99 of the null` **on `asym`** ». Pour `sym`, mettre une porte propre, par exemple « `eng_sym` et `nocomm_sym` ≥ 1 − ε, ε fixé au manifeste ».
- §5 issue A (`P:86`) : remplacer « necessity confirmed on this task » par « channel plumbing and trunk learn the engineered code on this task » (1.1). Exiger aussi le rejet du test confirmatoire de §4 : « D_asym ≥ 0,30 **and the one-sided sign test rejects at α = 0,05** ». Ainsi A ne peut pas coexister avec « non concluant » (2.4 a).
- §4 (`P:78-81`) : mesurer σ pour la puissance sur l'écart-type de `S_eng` en `asym` (pilote, D non lu), ou fixer un plancher MIN_K plus haut (2.2, 2.5). Viser la puissance pour le seuil de décision réellement utilisé (D ≥ 0,30 ou borne de l'IC), pas pour D > 0 (2.4).
- §3 tableau, ligne `msg_shuffle` (`P:57`) : préciser s'il s'agit d'un entraînement ou d'une évaluation, et la nature de la « fixed permutation » (épisodes ou identités de cible ; voir §3 ci-dessous) (2.8).
- §3 contrôle 4 (`P:70-71`) : ajouter un audit **a priori** du générateur (position de la cible uniforme, indépendante de tout ce que voit l'auditeur ; EGG la lit dans le fichier) et dire que le critique centralisé, s'il voit la cible, n'est pas une fuite à l'exécution (2.8).
- §2 (`P:42-44`) : supprimer ou conditionner « read as "the Procgen task does not need the message" ». Sinon, ajouter un bras N10-D exécuté dans le harnais N10-C avec le même code de canal (2.6).
- §5 issue A / §0 TBD-4 : envisager une marge relative à l'oracle, ≥ 0,9 × (1 − 1/C) = 0,675 (2.9).
- §0 TBD-11 (`P:28`) : fixer `n_eval` effectif et l'intervalle de log, pour garantir des fenêtres de 100 épisodes sans recouvrement (2.7).

**`docs/prereg/n10d/nh96-null.md`**
- §1 H1 (`N:10`) : retirer la clause `sym` (décision PI n° 1).
- §1 H0 (`N:12-13`) : définir explicitement le nul `sym` (B0 = 1,0), ou dire qu'il n'y en a pas puisque `sym` sort de H1.
- §2 (`N:18-22`) : nul apparié avec composante de graine commune (décision PI n° 2, `BS:16-19`). Ne pas transporter σ mesuré sur `asym` vers `sym` (2.3).
- §4 (`N:50-53`) : aligner sur `P` §5 un seul jeu de critères (IC ou valeur ponctuelle, pas les deux) (2.4). Le critère 2 sur `sym` sort de la liste de rejet.

**`experiments/n10d/sigma_seed.py`**
- `SS:29` et la docstring `SS:4-6` : si la PI l'accepte, extraire aussi l'écart-type inter-graines de `eng`/`asym` (sans calculer D). Sinon, documenter que σ_seed(`nocomm`/`asym`) vaut 0 par construction, au bruit binomial près (57 % des pilotes, P(χ²₇ < 7) = 0,571).
- `SS:40-48` (`required_seeds`) : la formule double la variance (`2 *`) en supposant des bras indépendants. Pour un écart-type de D par graine mesuré directement, il faut σ_D sans facteur 2 (nuance de 2.2).

**`tests/`** : ajouter un test sur le nul `sym` (P99 = 0 à σ = 0), pour que la dégénérescence soit visible.

## 3. Ce que le contradicteur n'a pas vu

Ce que la note #102 et le brainstorm répondaient déjà :
- 1.1 est écrit dans `P:35-36`, et `RN:37-40`, `RN:50-57` (hukTsKYu8Y Prop. 1 p. 4 : déficit garanti quand le canal est la seule route) et `RN:218-219` le développent. `BS` §4 refuse de « lire `eng − nocomm` sur asym comme un résultat ».
- 2.1 et 2.3 sont en grande partie réglés par la décision PI n° 1, « `sym` sort de H1 » (`BS:13-15`, qui cite `RAP:94`, M9). Elle n'était pas encore reportée dans PR #100 à `6e682cc`.
- La décision PI n° 2 (`BS:16-19`) exige déjà un nul apparié par graine. Le contradicteur n'a pas vu que le nul de `NU:40-43` tire les deux bras indépendamment.
- 1.2 : `RN` §4-5 et `BS` D2 font déjà de `learned − nocomm` entraînés de zéro (Lowe §6.2) la question ouverte. Ils citent Chauhan & Bera 2026 (arXiv:2609.34373) pour un `learned` indistinguable du silence et Cheng et al. (2608.04893) pour « nécessaire par construction ». Ces deux résultats prolongent 1.2 au-delà de Lazaridou, Eccles et MADDPG.
- 2.5 : `RN:259-262` donne 25 graines appariées comme ordre de grandeur publié.
- 2.6 : `BS` D7 (porte N10-C-bis à 3 graines, dans Procgen) traite en partie la question « la tâche Procgen a-t-elle besoin du message ».

Ce que ni le contradicteur ni la note n'ont relevé (constats de cette vérification) :
- **« fixed permutation » peut rendre `msg_shuffle` informatif.** Si la permutation porte sur les identités de cible (bijection fixe π sur K), le message code π(t), qui reste une fonction déterministe de la cible. Un auditeur entraîné sur ce bras l'apprendrait jusqu'à 1,0, et le contrôle 2 (`P:67`) échouerait sans qu'il y ait de fuite. Il faut une permutation entre épisodes, ou une évaluation seulement, à écrire dans `P:57`.
- **H0 de `N` ne définit le nul que pour `asym`** (`N:12-13`). Le nul `sym` n'existe que dans `NU:37`.
- **Les tests couvrent `asym` seulement** (`tests/test_n10d_skeleton.py:19-23`). `test_zero_noise_gives_zero_after_binomial_correction` (`tests/test_n10d_sigma.py:27-29`) fige précisément le comportement σ̂ = 0 sous bruit binomial pur, sans relever qu'il est le cas typique du bras mesuré.
- **Nuance de 2.2** : pour un D bimodal (20 % d'échecs), k = 9 et non 18, si l'écart-type par graine est pris comme σ_D. Le facteur 2 de `SS:47` suppose deux bras indépendants.
- **Exemple de 2.4** : l'incohérence la plus nette est l'issue A sans rejet du test des signes (4/5 positifs, p = 0,19), pas le cas B contre « rejetée ».
