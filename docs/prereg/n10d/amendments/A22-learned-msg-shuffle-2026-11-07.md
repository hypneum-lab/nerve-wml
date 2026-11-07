# A22 — `msg_shuffle` étendu à la paire `learned` et rapport par graine (proposition, 2026-11-07)

- **Statut** : PROPOSITION. Les deux points sont **décidés par le PI** le
  2026-11-07 entre 09:52 et 09:58 CET (décision B-1 (a) de la relecture
  croisée, et rapport par graine) ; rien n'est appliqué à `protocol.md`, qui
  reste inchangé, comme la note de lecture et tout autre fichier existant. Ni
  gel, ni tag, ni run.
- **Base** : `prereg/n10d-skeleton` à `c0f6760` (PR #100) ; `protocol.md:N`
  renvoie à ce commit. Fichier compagnon :
  `A21-context-conditioned-perturbation-2026-11-07.md` (même branche,
  désigné « A21 »), dont la règle R0 (A21 § 4.1) et la condition de lecture
  A21-2 (A21 § 5) sont reprises ici.
- **Qui** : agent Claude (Opus 5.5) pour le PI. Aucun calcul nouveau : les
  chiffres de puissance sont ceux d'A21 § 4.3. Rien n'a été lu dans
  `docs/n10d/contradictor-*`.

## Résumé (10 lignes)

1. Fait : le protocole enregistre NWML-N1 sur la paire `eng` seulement et laisse `learned` non décidé (`protocol.md:106-107`, `protocol.md:495`).
2. Fait (rapporté par le coordinateur, non relu ici) : le catalogue (NH-113) décrit l'usage causal du canal par la paire `learned`. Le protocole et le catalogue divergent donc.
3. Décision PI (09:52-09:58) : `msg_shuffle` est étendu à la paire `learned`, en secondaire exploratoire, sans α, sans changer aucune issue A à D.
4. Lecture : condition A21-2. Un non-rejet de R0 sur `learned` ne se lit « pas d'usage détectable » que si R0 rejette sur `eng` dans le même backend ; sinon « non concluant ».
5. Cette extension **résout la divergence** protocole/catalogue ; NH-113 **reste tel qu'écrit**.
6. Coût : **zéro entraînement** (les paires `learned` sont déjà entraînées pour le secondaire `learned − nocomm`, `protocol.md:89,106`) ; 20 évaluations de 500 épisodes par backend, 40 en tout.
7. Le jumeau exploratoire d'A21-1 s'applique « par bras perturbé » : 40 évaluations de plus sur `learned`, toujours sans entraînement.
8. Rapport par graine obligatoire et descriptif : valeurs par graine et un histogramme ou diagramme en tiges de `learned − nocomm`, `eng − nocomm` et des chutes sous `msg_shuffle`, à côté des tests appariés, sans verdict.
9. Motif : distribution par graine bimodale rapportée par Chauhan & Bera (arXiv:2609.34373, n = 25 ; motif fourni par le PI, non relu ici) et hypothèse de planification en mélange (π, `protocol.md:150-156`).
10. Phrases d'amendement en anglais au § 4 (pour `protocol.md` § 3, § 8, § 9 et la ligne 495).

## 1. Faits

- Bras `learned` : « trained jointly, alphabet A », « emergent-protocol arm
  (secondary) » (`protocol.md:89`). Il est entraîné : le contraste secondaire
  `learned − nocomm` est déjà listé (`protocol.md:106`).
- Même apprenant, même métrique et **même liste de graines** pour tous les
  bras (`protocol.md:280-282`) ; k = 20 par backend (`protocol.md:170`).
  Inférence : 20 paires `learned` gelées par backend, 40 en tout.
- `msg_shuffle` : évaluation seule, « on the trained `eng` pair »
  (`protocol.md:90`) ; secondaire `eng − msg_shuffle` (`protocol.md:106-107`).
- NWML-N1 (`protocol.md:495`) : « applying it to the `learned` arm is not
  decided here » ; secondaire exploratoire, sans α, aucun verdict propagé,
  aucune issue A à D changée.
- Interface : `Channel.transmit` porte l'arm (`interfaces.py:38-54`) ; le
  docstring ne décrit `msg_shuffle` que sur la paire `eng`
  (`interfaces.py:48`). L'opération (dérangement des épisodes,
  `interfaces.py:57-74`) ne dépend pas du code appris : elle s'applique telle
  quelle à n'importe quel émetteur.
- Le protocole demande déjà la distribution des D_i par graine avec chaque
  issue A, B ou C (`protocol.md:230-232`), mais seulement pour D = `eng −
  nocomm`, sans forme imposée (ni valeurs, ni figure) et pas pour
  `learned − nocomm` ni pour les chutes.
- Hypothèse de planification bimodale : une fraction π des graines apprend,
  le reste reste aveugle (`protocol.md:150-156`).
- Catalogue NH-113 : décrit l'usage causal du canal par `learned`
  (rapporté par le coordinateur et par le tour 2, l. 254-264 de
  `veille/tour-2-2026-11-07.md`, branche `docs/n10d-veille-tour-1-2026-11-06`,
  qui cite « la chute message dérangé contre intact sur `learned` »). La
  mention « randomisation conditionnelle sur l'état complet » en a été retirée
  le 2026-11-07 à 09:17 CET (décision PI rapportée, A21 § 6 item 5).

## 2. `msg_shuffle` sur `learned`

### 2.1 Ce que dit le dérangement sur `learned` (inférences)

- Espérance sous dérangement : 1/C pour tout auditeur sous les hypothèses
  (H-a) à (H-e) d'A21 § 2.3, donc aussi pour l'auditeur `learned`, quel que
  soit le code appris. La chute par graine drop_i = S_learned,i −
  S_learned+shuffle,i a donc pour espérance S_learned,i − 1/C, soit
  `learned − nocomm` en espérance (`nocomm` vaut aussi 1/C).
- Différence avec `learned − nocomm` : `nocomm` est un **autre** entraînement
  (sans canal) ; la chute est mesurée sur la **même** paire gelée, seul le
  message change. Elle répond à « cette paire se sert-elle du message ? » et
  non à « un canal aide-t-il l'apprentissage ? » ; c'est la distinction que la
  note tient de Lowe 2019 (tour 2 § 3.2, l. 275-280).
- Code partiel : une graine qui n'a appris qu'une partie du code (cas
  documenté, `protocol.md:229-231`) donne une chute partielle ; une graine
  aveugle donne une chute nulle. D'où le rapport par graine du § 3.

### 2.2 Règle et lecture

- Statistique et règle : R0 d'A21 § 4.1 (test des signes unilatéral à
  α = 0,05, 15 graines positives sur 20, et borne basse de l'IC 95 % de la
  chute moyenne > 0), par backend.
- Condition de lecture (A21-2) : un non-rejet sur `learned` ne se lit « pas
  d'usage détectable » que si R0 rejette sur `eng` dans le même backend ;
  sinon « non concluant ». Un rejet sur `learned` se lit « usage du message
  détecté par cette paire », exploratoire.
- Tailles détectables (A21 § 4.3, puissance 0,9) : chute homogène ≈ 0,03
  (≈ 0,04 avec un écart-type entre graines de 0,03) ; en mélange, plus
  d'environ deux tiers des graines qui utilisent le canal. Un non-rejet ne
  dit rien d'un usage par moins de ≈ 2/3 des graines : c'est l'une des
  raisons du rapport par graine.
- Pas de combinaison entre backends : NWML-N1 ne propage aucun verdict
  (`protocol.md:495`), donc la règle de `combine_backends` ne s'applique pas ;
  les deux backends sont rapportés côte à côte.
- Appariement (A21-3) : `msg_shuffle` sur `learned` utilise les mêmes 500
  épisodes que l'évaluation intacte de `learned`, lot de dérangement = ces
  500 épisodes.

### 2.3 Divergence protocole / catalogue

Avant A22, le protocole n'enregistrait le test d'usage que sur `eng`
(`protocol.md:495`), où l'usage est garanti par construction (code fixe) et le
test ne sert que de témoin ; le catalogue (NH-113) vise l'usage par
`learned`. Avec A22, le protocole enregistre aussi le test sur `learned`, sous
la même règle et sans α : **la divergence est résolue côté protocole, NH-113
reste tel qu'écrit.**

### 2.4 Coût

- Entraînement : **zéro** (paires `learned` déjà entraînées, § 1).
- Évaluations `msg_shuffle` sur `learned` : 20 par backend, **40 en tout**,
  de 500 épisodes chacune.
- Jumeau exploratoire d'A21-1 (« 40 evaluations per perturbed arm ») :
  appliqué aussi à `learned`, **40 de plus**, sans entraînement.
- Heures : comme en A21 § 4.4, aucun coût d'évaluation N10-D n'existe
  (TBD-11) ; borne haute pessimiste tirée de N10-C (6,7 min par modèle sur
  KXKM, PR #114) ≈ 4,5 h KXKM par lot de 40 évaluations ; probablement des
  secondes par évaluation pour deux petits MLP (inférence, à mesurer avec
  TBD-11).

## 3. Rapport par graine (descriptif, obligatoire)

- Grandeurs : par backend et par graine, `learned − nocomm`, `eng − nocomm`
  (D_i), chute `eng − eng+msg_shuffle`, chute `learned −
  learned+msg_shuffle`.
- Forme : les valeurs par graine (tableau, avec la graine) et une figure par
  grandeur et par backend, histogramme ou diagramme en tiges, à côté des tests
  appariés.
- Statut : **descriptif, sans verdict** ; aucun seuil, aucune classification
  « graine qui apprend / n'apprend pas » n'en est tirée, aucune issue A à D ni
  lecture NWML-N1 n'en dépend.
- Motif : une moyenne et son IC ne distinguent pas « quelques graines
  apprennent tout » de « chaque graine apprend en partie »
  (`protocol.md:231-232`) ; la planification suppose un mélange
  (`protocol.md:150-156`) ; Chauhan & Bera (arXiv:2609.34373, n = 25 graines)
  rapportent une distribution par graine bimodale (motif fourni par le PI ;
  **non relu dans ce dépôt**, à vérifier dans la note de lecture avant
  report dans `protocol.md`).
- Rapport avec l'existant : étend `protocol.md:230-232` (distribution des D_i
  avec chaque issue) aux trois autres grandeurs et en fixe la forme.

## 4. Phrases d'amendement proposées (anglais ; NON appliquées)

Pour `protocol.md` § 3, tableau des bras, ligne `msg_shuffle`
(`protocol.md:90`) :

> | `msg_shuffle` | evaluation only, on the trained `eng` pair and on the
> trained `learned` pair (A22): message of another episode of the same
> evaluation batch (uniform episode derangement, no label exclusion,
> `derange_episodes`) | negative control on `eng`; secondary exploratory
> use test on `learned` |

Pour `protocol.md` § 3, phrase des secondaires (`protocol.md:106-107`) :

> Secondary: `learned - nocomm`, `eng - msg_shuffle` and `learned -
> msg_shuffle` (the last two are NWML-N1, secondary exploratory, PI
> 2026-11-06 ~20:50 and 2026-11-07 09:52-09:58, sections 8 and 9).

Pour `protocol.md` § 8 (liste des amendements) :

> - **A22-1** (`msg_shuffle` on `learned`; PI decision 2026-11-07
>   09:52-09:58 CET, cross-review B-1 option a). `msg_shuffle` is also
>   applied, evaluation only, to the trained `learned` pair of every
>   (backend, seed): per-seed drop = S_learned - S_learned+msg_shuffle, on the
>   same 500 evaluation episodes as the intact `learned` evaluation (A21-3).
>   It is secondary exploratory, spends no alpha, changes no outcome A-D of
>   section 5, is not combined across backends and is reported next to
>   NWML-N1 on `eng`. It is read by rule R0 under the condition of A21-2: a
>   non-rejection on `learned` is read as "no detectable use" only if R0
>   rejects on `eng` in the same backend; otherwise it is "non concluant".
>   Cost: no training; 20 evaluations of 500 episodes per backend (40 in
>   all), plus 40 for the exploratory twin evaluation of A21-1. This resolves
>   the divergence between this protocol (use test on `eng` only) and the
>   catalogue entry NH-113 (causal use of the channel by `learned`); NH-113
>   stays as written.
> - **A22-2** (per-seed report; PI decision 2026-11-07 09:52-09:58 CET). For
>   each backend, the per-seed values of `learned - nocomm`, `eng - nocomm`
>   and of the per-seed drops under `msg_shuffle` on `eng` and on `learned`
>   are reported, with the seed, as a table and as a histogram or stem plot,
>   next to the paired tests. This report is descriptive and mandatory: no
>   verdict, threshold or classification of seeds is attached to it, and no
>   outcome A-D or NWML-N1 reading depends on it. Motive: the planning
>   assumption is a mixture over seeds (section 4, pi), and bimodal per-seed
>   distributions are reported for agents trained from scratch
>   (arXiv:2609.34373, 25 seeds). It extends the per-seed distribution of D
>   already required with every outcome A, B or C (section 5).

Pour `protocol.md:495` (ligne NWML-N1 de la table du § 9, remplacer « applying
it to the `learned` arm is not decided here. ») :

> applying it to the `learned` arm is decided on 2026-11-07 09:52-09:58 CET
> (A22-1): same derangement, evaluation only, read by rule R0 under the
> condition of A21-2.

Pour `protocol.md` § 9 (nouvelle sous-section « 2026-11-07 09:52-09:58 CET
(A22) ») :

> | A22 | `msg_shuffle` extended to the trained `learned` pair, secondary
> exploratory, no alpha, read under A21-2 (A22-1); per-seed values and a
> histogram or stem plot of `learned - nocomm`, `eng - nocomm` and the
> `msg_shuffle` drops, descriptive and mandatory (A22-2). NH-113 unchanged |

## 5. Ce que le PI doit décider

Rien de neuf : les deux points sont décidés (09:52-09:58 CET). Restent à
faire, hors décision :

1. Vérifier dans la note de lecture la bimodalité par graine de Chauhan &
   Bera (n = 25) avant de reporter A22-2 dans `protocol.md` (§ 3 ci-dessus).
2. Reporter A21 et A22 dans `protocol.md` en un seul commit, au moment choisi
   par le PI.
