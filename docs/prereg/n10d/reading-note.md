# Note de lecture — N10-D (v0, 2026-11-01)

Statut : brouillon de lecture, verrou **non passé**. Ce document ne modifie ni
`protocol.md` (PR #100, brouillon L0) ni aucune préreg. Il ne contient aucun
verdict sur E1-small, N10-C run 2 ou Me6, aucun tag de gel, et n'a demandé
aucun calcul sur KXKM, macM1 ou GrosMac.

Conventions : « Fait » = constat lu dans une source ou dans le dépôt ;
« Inférence » = raisonnement de l'auteur de la note, dans une phrase séparée.
`[résumé moteur]` = seul le résumé ou une notice d'API a été lu.
« à recouper » = une seule source, ou deux sources de même nature.

Références internes (chemins relatifs au dépôt indiqué) :
- `MS` = `nerve-wml:docs/milestones/2026-10-25-n10c-bis-control-arms.md`
- `P100` = PR #100, `docs/prereg/n10d/protocol.md` et `nh96-null.md` (non fusionnés)
- `RAP` = rapport d'analyse profonde du 01/11, `synthese/rapport.md`
  (dossier `.github`, PR #23)
- `VEI` = `veille/veille-2026-11-01.md` ; `RDF` = `veille/revue-de-fond-2026-11-01.md`

## 1. Question que la campagne prétend trancher (une phrase)

Un message devient-il nécessaire et utilisé quand, par construction, l'auditeur
ne peut pas réussir sans une information que seul l'émetteur détient, autrement
dit `eng − nocomm` est-il grand sur la tâche asymétrique et nul sur sa jumelle
symétrique (`P100` §1) ?

Fait : `P100` §1 qualifie lui-même NH-96 de « VARIANT (expected result, Lowe
2019) » et de « test of the task and of the channel plumbing ».
Inférence : avec un émetteur `eng` à code idéal figé, D_asym ≥ 0,30 est attendu
par construction ; la question scientifiquement ouverte n'est pas celle-là mais
l'usage **causal** du canal par le bras `learned`, qui n'est que secondaire dans
`P100` §3.

## 2. Sources primaires lues en entier

| id résolu | auteurs, année | pages lues | ce qu'elle établit |
|---|---|---|---|
| arXiv:1903.05168 | Lowe, Foerster, Boureau, Pineau, Dauphin, 2019 (AAMAS) | 1-13, annexes comprises | Distingue positive signaling et positive listening ; la cohérence de l'émetteur (SC) et l'entropie ne prouvent pas l'usage ; le CIC mesure l'influence causale ; Table 2 : SC avec messages brouillés ≈ SC par défaut (2×2 : 0,198 contre 0,202) ; §6.2 : couper le canal en test est peu fiable (décalage de distribution), il faut entraîner de zéro avec et sans canal. |
| arXiv:1812.09755 | Singh, Jain, Sukhbaatar, 2019 (ICLR) | 1-15 | IC3Net : porte de communication apprise ; traffic junction à vision nulle « non soluble sans communication » : IRIC 29,8 / 3,4 / 35,0 contre IC3Net 93,0 / 89,3 / 72,4 (5 runs). |
| arXiv:2107.02367 | Liu, Lamb et al., 2021 (NeurIPS) | 1-29 (corps, annexe E ; preuves de l'annexe D survolées) | Communication discrète (VQ) entre spécialistes ; VQ > Gumbel ; β = 0,25 ; limite §5 : le gain n'apparaît que si la communication entre spécialistes est importante pour la tâche. |
| arXiv:2405.08550 | Hu, Shen, Zhang, Tao, 2024 (ICLR) | 1-16 | CommFormer : graphe de communication appris ; sparsité S = 0,4 ≈ graphe complet ; Fig. 1 : grande variance des architectures prédéfinies sur 10 graines. |
| arXiv:2609.27252 | You, Jang, Mo, Jung, 2026 | corps + annexe B (33 p.) | Grille structure × géométrie ; calibration par permutation tenant compte de l'agrégation (K = 500, BH) ; structure relationnelle convergente, géométrie faible ; Spearman 0,968 entre mKNN et H0. |
| arXiv:2610.08612 | Agliari, Lepre, Roscani, 2026 | corps (33 p.), preuves survolées | RBM modulaire ≡ Hopfield couplés ; Thm 1 : point fixe CD-1 à w = η̄, dérive bornée ; désenchevêtrement numérique (L = 3, N = 5000). |
| arXiv:2609.32605 | Agliari, Alessandrelli, Barra, Fachechi, 2026 | corps (31 p.), annexe A en partie | Mémoires associatives modulaires denses (P = 4, D = 8, K = Θ(N)) appliquées à un canal bruité ; code hebbien **fixe** (rien d'appris) ; tolère 80 % d'effacement et ≈ 25-30 % de bits inversés ; taux de code ≈ 0,034. |

Lue en résumé seulement :

| id | source | statut |
|---|---|---|
| OpenReview hukTsKYu8Y | « Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model », soumission ICLR 2027 | `[résumé moteur]` (PDF et `notes?id=` en 403 ChallengeRequired ; titre et résumé via `notes/search`). Résumé : couper le broadcast −41 à −52 % sur les questions à deux modules (≤ 1,4 pt sur un module) ; broadcast substitué → combinaison dans 61-73 % des cas ; restauration 2 047/2 048 ; protocole fixé d'avance. |

## 3. Déjà publié

| affirmation | sources (nature) | état |
|---|---|---|
| Sur une tâche où l'information est privée par construction, un canal appris bat nettement le silence. | 1812.09755 Tables (article, expérience) ; 1903.05168 §2 et §6 (article, analyse critique) ; `MS` G1 et `P100` §1 qui citent l'attendu (document interne) | recoupé (2 articles + dépôt) |
| Les métriques côté émetteur (SC, entropie) ne prouvent pas que l'auditeur utilise le message. | 1903.05168 Table 2 et §5 (article) ; 1912.05676 Eccles 2019, résumé et notice OpenReview l0S7QgBMjM (article, autre équipe) | recoupé ; Eccles lu en résumé seulement |
| Couper le canal en test est un test biaisé ; il faut des bras entraînés avec et sans canal. | 1903.05168 §6.2 (article) ; `RDF:685+` (lecture interne de Lowe) | **à recouper** (la 2e source dérive de la 1re) |
| Le dérangement / la substitution / la restauration de messages comme protocole causal d'usage du canal. | 1903.05168 (CIC, messages brouillés) ; hukTsKYu8Y `[résumé moteur]` ; 2608.04893 Cheng et al. `[résumé moteur]` ; 2607.26773 Zhang & Emu `[résumé moteur]` | recoupé en méthode ; trois des quatre sources en résumé seulement |
| Sous couplage rigide un canal appris peut ne rien apporter par rapport au silence, et le cross-play s'effondre. | 2609.34373 Chauhan & Bera 2026 `[résumé moteur]` (p = 0,982, n = 25 ; cross-play 0,980 → 0,144) ; `MS:326-328` (gtm_mis 0,81 contre gtm 0,54-0,73, une graine) | **à recouper** (résumé + run interne non répliqué) |
| Messages discrets appris entre modules spécialistes. | 2107.02367 (article) ; `VEI:70-105` (veille interne) | **à recouper** (la veille dérive de l'article) |
| Un code fixe décodé par attracteurs résiste à un fort bruit de canal sans aucun apprentissage. | 2609.32605 (article) ; 2610.08612 (article, même groupe) | **à recouper** (même groupe d'auteurs) |
| Positive listening partenaire contre étranger comme test d'usage. | 2610.37849 Denamganaï 2026 `[résumé moteur]` ; 1903.05168 §3 (définition) | recoupé sur la notion, pas sur le protocole |

## 4. Reste ouvert

1. **Usage causal du canal `learned` sur une tâche asymétrique nécessaire par
   construction, avec un nul qui tient compte de la covariance de graine.**
   Preuve de l'absence (partielle) : `P100` §2 simule les deux bras
   indépendamment par graine (`nh96-null.md` §2, « Draw per seed and arm ») ;
   `RAP:94` (M9) relève l'absence de covariance. Dans la littérature lue, Lowe
   2019 et IC3Net rapportent des moyennes sur graines sans nul apparié par
   graine (lecture intégrale). Inférence : la méthode existe, son application
   avec ce nul n'a pas été trouvée au tour 1. **à recouper** au tour 2 (canaux
   non interrogés, § 6).
2. **Le même test sur des neurolettres discrètes entre substrats hétérogènes
   (MLP/LIF) et au chemin de réception γ/θ.** Preuve de l'absence : aucune des
   7 sources ni des 668 citants en aval du snowball (`snowball-tour1.tsv`) n'a
   un titre portant sur ces substrats (filtre par titre seulement). **à recouper**.
3. **Claim B (accord de classifieurs) contre usage du canal.** Fait : `RAP:56`
   (B3) indique que les codes « post-VQ » sont l'argmax de la tête π et que
   MI(code ; XOR) ≈ 0,001 nat selon un rejeu d'agent non reproductible depuis
   le dépôt. Inférence : la question « le canal porte-t-il l'information
   nécessaire » n'est pas tranchée par Claim B. Question interne, pas de
   littérature attendue.

Ce qui n'est **pas** ouvert : `eng − nocomm` sur asym avec émetteur conçu
(`P100` §1 le dit lui-même).

## 5. Dessin minimal qui tranche la question ouverte

Détail et alternatives : `brainstorm-2026-11-01.md`. Dessin minimal retenu
pour discussion (rien ne part avant le verrou) :

- **Bras.** Sur la tâche `asym` de `P100` (K = 16, C = 4, A = 16), bras
  `learned` et `nocomm` **entraînés de zéro**, appariés par graine (Lowe §6.2) ;
  sur chaque agent `learned` entraîné, interventions en test : message permuté
  entre épisodes, message gelé (constant), canal mis à zéro, plus restauration.
  `eng` et `msg_shuffle` restent des contrôles de chaîne.
- **Nul.** Permutation des étiquettes de bras **à l'intérieur de chaque graine**
  (nul apparié qui conserve la covariance de graine), plus le nul substitut
  binomial de `P100` §2 recalibré avec une composante de graine commune aux
  deux bras. `sym` sort de H1 (décision PI) et reste un contrôle de fuite.
- **Critère.** Usage causal soutenu si la chute permuté-contre-intact sur
  `learned` dépasse le P99 du nul apparié **et** si la restauration ramène la
  performance intacte ; réfuté si l'IC 95 % apparié de la chute contient 0
  alors que `learned − nocomm` dépasse le P99 (gain sans usage du message).
- **Coût.** Non mesuré. Inférence : un jeu référentiel K = 16 sous JaxMARL ou
  EGG est d'un ordre de grandeur bien inférieur à un run Procgen N10-C
  (≈ 10,8 h par run sur KXKM, `MS:179`). Hypothèse de travail à mesurer au
  pilote : ≤ 1 h GPU par run, ≈ 10-20 h GPU au total ; ≈ 15-25 h humaines
  (environnements réels manquants, `P100` §6).

## 6. Trois tests du verrou

| test | état | ce qui manque |
|---|---|---|
| Saturation | **non passé.** Tour 1 seulement ; le tour a encore apporté 8 entrées gardées et changé la note (§ 1, § 4). | Au moins un tour de plus sans entrée qui change la note. |
| Recoupement | **partiel.** 3 affirmations de § 3 recoupées, 5 « à recouper ». | Lire en entier Eccles 2019, Chauhan & Bera 2026, Cheng et al., PoPLi ; obtenir hukTsKYu8Y en texte intégral. |
| Couverture | **non passé.** Interrogés : arXiv (5 requêtes), OpenReview (4 termes), snowball S2 sur 7 sources (955 lignes). Non interrogés : bioRxiv, Crossref-revues, thèses/HAL, code et données (GitHub, Hugging Face). | Ces quatre canaux ; tri du snowball au-delà du filtre par titre ; résolution de arXiv:2610.28905 (Gazzaev et al., cité en `RAP:242`, non résolu ici). |

## 7. Journal de lecture

| date | source | pages / sections |
|---|---|---|
| 2026-11-01 | arXiv:1903.05168 (PDF) | 1-13, annexes comprises |
| 2026-11-01 | arXiv:1812.09755 (PDF) | 1-15 |
| 2026-11-01 | arXiv:2107.02367 (PDF) | 1-29 ; annexe D survolée |
| 2026-11-01 | arXiv:2405.08550 (PDF) | 1-16 |
| 2026-11-01 | arXiv:2609.27252 (PDF) | corps + annexe B |
| 2026-11-01 | arXiv:2610.08612 (PDF) | corps ; preuves survolées |
| 2026-11-01 | arXiv:2609.32605 (PDF) | corps ; annexe A en partie |
| 2026-11-01 | OpenReview hukTsKYu8Y | titre + résumé (`notes/search`) `[résumé moteur]` |
| 2026-11-01 | 8 entrées du tour 1 (voir `veille-journal.md`) | résumés arXiv `[résumé moteur]` |
| 2026-11-01 | `MS` | l. 179-202 (plan de calcul), 300-352 (B3, porte G1-G7) |
| 2026-11-01 | `P100` | `protocol.md` et `nh96-null.md` en entier |
| 2026-11-01 | `RAP` | l. 56-57, 89-94, 148-151, 242, 285 |
| 2026-11-01 | `VEI` / `RDF` | `VEI:70-105` ; `RDF:113-168, 609-620, 685+` |
