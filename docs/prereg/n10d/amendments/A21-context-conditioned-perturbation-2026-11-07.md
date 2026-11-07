# A21 — dérangement conditionné au contexte et puissance de la statistique de chute (proposition, 2026-11-07)

- **Statut** : PROPOSITION, rien n'est appliqué à `protocol.md`. Trois points
  ont été tranchés par le PI le 2026-11-07 à 09:28 CET (§ 6) ; le texte
  d'amendement du § 5 en tient compte, mais il reste à reporter dans
  `protocol.md`. `protocol.md`, la note de
  lecture et tout autre fichier existant sont inchangés. Ni gel, ni tag, ni run.
- **Base** : `prereg/n10d-skeleton` à `c0f6760` (PR #100). Les numéros de ligne
  `protocol.md:N`, `spec.py:N`, etc. renvoient à ce commit ; le code est sous
  `experiments/n10d/`.
- **Origine** : décision PI du 2026-11-07 09:17 CET, « chiffrer avant gel » les
  deux contrôles suggérés par le tour de veille 2
  (`docs/prereg/n10d/veille/tour-2-2026-11-07.md`, branche
  `docs/n10d-veille-tour-1-2026-11-06`, commit `ca3f07e`, désigné ci-dessous
  « tour 2 »), § 3.2 l. 290-304 et § 7 item 2 l. 383-384. Ces deux contrôles
  sont des **inférences** du tour 2 à partir d'arXiv:2611.01666 (Xu & Chen),
  lu en partie (tour 2 l. 67-73).
- **Qui** : agent Claude (Opus 5.5) pour le PI. Calcul : un Monte-Carlo en
  Python pur (bibliothèque standard), 37 s sur GrosMac, sans aucune donnée de
  run ; script en annexe. Rien n'a été lu dans `docs/n10d/contradictor-*`.

## Résumé (10 lignes)

1. Fait (spécification) : l'émetteur ne voit que la cible t ; l'auditeur voit les C candidats mélangés (`asym`), plus t (`sym`) (`protocol.md:266-267`, `spec.py:25-26`, `backends.py:15-17`).
2. Fait : aucun code d'environnement n'existe encore (`backends.py:22,27` lèvent) ; ce qui précède est une spécification, pas une implémentation auditée.
3. Sur `asym`, la seule information privée de l'émetteur est t, c.-à-d. le contenu dont on veut mesurer l'usage ; sur `sym`, l'émetteur ne voit rien de plus que l'auditeur.
4. Dérivation (§ 2) : sous un dérangement qui tire le message d'un épisode indépendant, le succès attendu vaut exactement 1/C pour **tout** auditeur, quoi que voie l'émetteur ; le conditionnement au contexte ne change pas cette espérance.
5. Conclusion : le contrôle 1 (dérangement conditionné) est **sans objet** pour N10-D ; à la place, deux assertions d'audit (4a) qui garantissent les hypothèses de la dérivation (**décidé, PI 2026-11-07 09:28 CET**).
6. La version par groupes est dégénérée à n_eval = 500 (≈ 120 épisodes sur 500 groupables par ensemble, ≈ 6 par tuple ordonné) ; le « jumeau » à cible retirée parmi les candidats (§ 3.3) est **gardé comme évaluation exploratoire**, hors des verdicts de H1 et de NWML-N1 (décidé 09:28).
7. Puissance (§ 4, Monte-Carlo 2 000 simulations, k = 20, n_eval = 500, test des signes à α = 0,05, m_k = 15) : l'effet attendu sur `eng` (chute moyenne 0,60 à π = 0,8) est détecté avec une probabilité 0,993 par backend (≈ 0,986 pour les deux).
8. Effet minimal détectable à puissance 0,9 : chute homogène ≈ 0,028 (sans hétérogénéité entre graines) ou ≈ 0,04 (sd entre graines 0,03) ; en mélange bimodal, fraction de graines « qui utilisent » π ≥ ≈ 0,68 ; risque de première espèce sous chute nulle 0,011.
9. Coût : **zéro entraînement** en plus (le bras `eng` est déjà entraîné pour H1 ; `msg_shuffle` est déjà prévu en évaluation seule) ; la vérification de puissance elle-même est une simulation. Le jumeau coûte 40 évaluations de 500 épisodes par bras perturbé ; intact, `msg_shuffle` et jumeau sont évalués sur les mêmes 500 épisodes (décidé 09:28).
10. Décidé le 2026-11-07 09:28 CET (§ 6) : contrôle 1 sans objet + audit 4a + jumeau exploratoire ; rejet de R0 sur `eng` requis pour lire un non-rejet sur `learned` ; appariement sur 500 épisodes. Restent ouverts : `msg_shuffle` sur `learned` et le lien avec NH-113.

## 1. Ce que voient l'émetteur et l'auditeur (faits)

Tous les faits ci-dessous sont des **spécifications** : l'environnement n'est
implémenté sur aucun backend (`backends.py:22` JaxMARL et `backends.py:27` EGG
lèvent `NotImplementedError` ; `protocol.md:253-256` liste l'environnement réel
parmi ce qui manque avant le pilote).

| | émetteur (speaker) | auditeur (listener) |
|---|---|---|
| `asym` | la cible t, une identité parmi K = 16 (`protocol.md:266` ; `spec.py:25` ; `backends.py:16`) | les C = 4 candidats, dans un ordre mélangé uniformément par épisode, dont t (`protocol.md:263-267` ; `spec.py:25` ; `backends.py:15-17`) ; plus la sortie du canal (`protocol.md:268-270`) |
| `sym` | t (même texte) | les mêmes candidats **plus** t (`protocol.md:267` ; `spec.py:26` ; `backends.py:17`), plus la sortie du canal |

Autres faits utiles :

- Le message ne dépend, par interface, que de l'observation de l'émetteur :
  `Sender.emit(speaker_obs)` (`interfaces.py:34`) ; `reset` rend la paire
  `(speaker_obs, listener_obs)` (`interfaces.py:23-24`), sans autre champ.
- Un pas par épisode, un symbole d'alphabet A = 16 (`protocol.md:262,268-269` ;
  L = 1 est une valeur de travail de TBD-11, `protocol.md:44`).
- La récompense ne dépend que de t et de l'action : 1 si le candidat choisi est
  t (`protocol.md:272`) ; l'action est toujours un candidat valide
  (`protocol.md:271`).
- Sur `asym`, t est uniforme parmi les C positions et indépendant de tout ce que
  voit l'auditeur, d'où le plafond aveugle 1/C (`spec.py:39-46`) ; c'est ce que
  vérifie l'audit a priori 4a (`protocol.md:128-133`).
- Le critique centralisé qui voit la cible à l'entraînement n'est pas une fuite
  à l'exécution (`protocol.md:135-136`) ; il ne change pas ce que voit
  l'émetteur à l'évaluation.
- `msg_shuffle` : dérangement uniforme des **épisodes** d'un lot d'évaluation,
  par `random.Random(batch_seed)`, sans exclusion d'étiquette
  (`protocol.md:90-98` ; `interfaces.py:42-54,57-74`). Évaluation seule, sur la
  paire `eng` entraînée (`protocol.md:90`, `protocol.md:495`).
- Ce que **rien** n'énonce : (i) que l'observation de l'émetteur se réduit à t
  dans l'implémentation (l'audit 4a, `protocol.md:129-131`, ne porte que sur
  l'observation de l'auditeur) ; (ii) que l'auditeur traite chaque épisode
  d'évaluation indépendamment des autres épisodes du lot ; (iii) la taille du
  « lot d'évaluation » de `msg_shuffle` et si les épisodes dérangés sont les
  mêmes que ceux de l'évaluation intacte.

En deux lignes : **l'émetteur ne voit que t ; sur `asym` l'auditeur voit les C
candidats mélangés sans t, sur `sym` les candidats et t.** L'émetteur ne voit
donc rien de « contextuel » que l'auditeur ne voie pas ; sa seule information
privée (sur `asym`) est la cible elle-même.

## 2. Le contrôle 1 est-il nécessaire ?

### 2.1 La question du tour 2

Le tour 2 (l. 290-299) craint qu'un dérangement par lot rompe à la fois le lien
message → cible et un lien message → contexte de l'épisode, « si l'émetteur
voit autre chose que la cible » ; une chute mêlerait alors contenu privé et
contexte partagé. La condition est explicitement « à vérifier dans `spec.py` ».

### 2.2 Fait : la condition n'est pas remplie

D'après le § 1, l'émetteur voit t et rien d'autre. Sur `asym`, t est
précisément le contenu privé ; il n'y a pas de contexte vu par l'émetteur et
caché à l'auditeur. Sur `sym`, l'observation de l'émetteur est incluse dans
celle de l'auditeur : aucun contenu privé.

### 2.3 Dérivation : le conditionnement ne changerait rien en espérance

Inférence (calcul, à partir des seules spécifications citées). Soit, pour un
épisode i d'évaluation sur `asym`, o_i l'observation de l'auditeur (tuple
ordonné des C candidats) et t_i la cible. Hypothèses :

- (H-a) épisodes indépendants et de même loi dans le lot d'évaluation (cibles et
  ordres frais, `interfaces.py:92-94`) ;
- (H-b) t_i uniforme sur les C positions sachant o_i (`spec.py:42-45`, audit 4a
  `protocol.md:130-132`) ;
- (H-c) l'épisode source j ≠ i du message est choisi indépendamment des cibles
  (permutation tirée par un générateur à graine fixée, `interfaces.py:57-74`,
  `protocol.md:92-93`) ;
- (H-d) l'auditeur décide pour l'épisode i à partir de (o_i, message reçu) et
  de son propre aléa seulement (pas de couplage entre épisodes d'un lot) ;
- (H-e) l'action est toujours un candidat valide (`protocol.md:271`).

Le message reçu m_j est une fonction de l'épisode j (et de l'aléa de
l'émetteur), donc indépendant de t_i sachant o_i par (H-a) et (H-c). Par
(H-d), l'action a_i est indépendante de t_i sachant o_i. Par (H-b) et (H-e),
P(a_i = t_i | o_i) = 1/C. Donc **E[S_msg_shuffle] = 1/C pour tout auditeur**,
appris ou non, quel que soit le comportement devant un message hors des
candidats, et **quel que soit ce que voit l'émetteur**. Le docstring
d'`interfaces.py:62-65` donne déjà ce résultat pour un auditeur `eng` qui
choisit au hasard quand le message ne nomme aucun candidat ; la dérivation
l'étend à tout auditeur.

Un dérangement restreint aux épisodes de même contexte o (ou de même ensemble
de candidats) choisit j selon o_j = o_i, ce qui ne dépend pas de t_i : (H-c)
tient encore, l'espérance vaut encore 1/C. **La chute moyenne
S_intact − S_dérangé est donc la même, S_intact − 1/C, avec ou sans
conditionnement.** Le conditionnement ne changerait que la variance et la part
de messages « hors candidats » (≈ 3/4 sans conditionnement, 0 avec), pas la
grandeur testée.

Le piège de Xu & Chen (§7.2, p. 21, tour 2 l. 91-96 : un rééchantillonneur
marginal reproduit un faux effet) suppose que l'entrée perturbée porte une
information sur une variable non vue qui agit sur le résultat. Ici le résultat
ne dépend que de t et de a (`protocol.md:272`), et t est uniforme sachant tout
ce que voit l'auditeur (H-b) ; il n'existe pas de telle variable. Inférence :
le piège ne s'applique pas à la grandeur « chute sous dérangement » de N10-D.

### 2.4 Verdict

**Contrôle 1 sans objet** pour N10-D, pour deux raisons indépendantes :
(1) fait, l'émetteur ne voit que t (§ 1) ; (2) inférence, même s'il voyait
plus, l'espérance sous dérangement est 1/C sous (H-a) à (H-e), conditionnée ou
non. Ce qui compte n'est donc pas un nouveau contrôle mais que (H-a) à (H-e)
tiennent dans l'implémentation. Deux d'entre elles ne sont garanties par aucun
texte (§ 1, « ce que rien n'énonce ») :

- l'observation de l'émetteur réduite à t (sinon, sur `learned`, un émetteur qui
  voit les candidats pourrait coder la **position** de t, ce qui reste un
  contenu privé mais change la lecture d'un code « identité » ; et EGG lit ses
  entrées dans un fichier de données, `protocol.md:134-135`, où un champ
  supplémentaire passerait inaperçu) ;
- (H-d), l'absence de couplage entre épisodes du lot à l'évaluation : une
  normalisation par statistiques de lot ou une attention sur le lot laisserait
  passer le **multi-ensemble** des messages, que le dérangement conserve.
  Inférence : la fuite résultante serait faible à 500 épisodes, et le contrôle 2
  (moyenne `msg_shuffle` ≤ L99, `protocol.md:118`) la verrait si elle portait
  le niveau au-dessus du nul ; mieux vaut l'exclure par construction.

D'où la proposition A21 (§ 5) : deux assertions d'audit en 4a au lieu d'un
contrôle de plus.

## 3. Dessin du contrôle 1 et variante exploratoire retenue

La version par groupes (§ 3.2) n'est pas retenue. La variante « jumeau »
(§ 3.3) est gardée par le PI comme évaluation **exploratoire** (décision
2026-11-07 09:28 CET, option b) ; elle n'est pas un contrôle du dessin.

### 3.1 « Même contexte » opérationnel

- `asym` : contexte visible = tuple ordonné des C candidats (ce que voit
  l'auditeur, § 1). Variante plus lâche : ensemble non ordonné des candidats.
- `sym` : contexte visible = candidats **et** t. Deux épisodes de même contexte
  ont la même cible, donc le même message `eng` (code déterministe) : le
  dérangement conditionné est l'identité. **Dégénéré par construction sur
  `sym`**, où `msg_shuffle` n'est de toute façon pas lu (`protocol.md:118`).

### 3.2 Par groupes dans le lot de 500 : dégénéré

Calcul (exact, loi d'occupation, n_eval = 500, `protocol.md:101`) :

| définition du contexte | nombre de contextes | P(épisode seul de son contexte) | épisodes dérangeables (espérance) | groupes de taille ≥ 2 |
|---|---|---|---|---|
| ensemble non ordonné | C(16,4) = 1 820 | 0,760 | 119,9 / 500 | 57,2 |
| tuple ordonné | 16·15·14·13 = 43 680 | 0,989 | 5,7 / 500 | 2,8 |

Un dérangement exige au moins 2 épisodes par groupe (`interfaces.py:68-69`).
Les groupes seraient presque tous des paires (un échange forcé), sur un quart
des épisodes au mieux : statistique sur ≈ 120 épisodes non représentatifs du
lot, ou sur ≈ 6. Augmenter n_eval pour remplir les groupes ordonnés demanderait
de l'ordre de 10⁵ épisodes ; n_eval = 500 est fermé (A8, `protocol.md:44`).

### 3.3 Variante praticable : le jumeau à cible retirée (exploratoire, décidée)

« Retirée » veut dire **tirée à nouveau** parmi les candidats, pas ôtée de
l'ensemble : la cible du jumeau reste un des C candidats visibles. Pour chaque épisode i, garder o_i et tirer t*_i uniformément parmi les C
candidats de i (coïncidence t*_i = t_i permise, comme l'absence d'exclusion
d'étiquette de `interfaces.py:61-63`) ; l'auditeur reçoit le message que
l'émetteur gelé émet pour t*_i. C'est un dérangement « de même contexte » à
groupes infinis, tiré de la loi connue du plan (le générateur d'environnement,
randomisé, en boucle ouverte) : la loi de t sachant le contexte de l'auditeur,
ce que l'Hypothèse 1 de Xu & Chen demande pour l'entrée tirée (tour 2
l. 281-284). Espérance : 1/C pour tout auditeur (§ 2.3 avec j remplacé par le
jumeau). Exclure t*_i = t_i casserait cette propriété (l'auditeur `eng`
tomberait à 0, un auditeur aveugle resterait à 1/C).

Coût : une évaluation de 500 épisodes de plus par (backend, graine, bras
perturbé), sans entraînement : 2 × 20 = 40 évaluations pour `eng`, 40 de plus
si `msg_shuffle` est un jour appliqué à `learned` (point encore ouvert, § 6).
Évaluée sur les mêmes 500 épisodes que l'évaluation intacte et `msg_shuffle`
(décision d'appariement, § 5 c).

Statut (décidé 09:28) : **exploratoire**, rapporté à côté des résultats ; il
ne dépense pas d'α, n'entre ni dans les verdicts A à D de H1 ni dans la
lecture de NWML-N1, et n'est pas une porte.

Ce qu'il ajoute (inférence) : sous (H-a) à (H-e), son espérance est celle de
`msg_shuffle`, 1/C (§ 2.3). Il ne mesure donc rien de neuf en espérance ; il
sert de **vérification** que l'auditeur ne lit pas la cible dans l'ensemble
des candidats. Mécanisme : avec `msg_shuffle`, le message nomme une identité
absente des candidats dans ≈ 3/4 des épisodes, et l'auditeur retombe sur sa
règle de repli, qui ne dépend que des candidats ; avec le jumeau, le message
nomme toujours un candidat. Si (H-b) tient, les deux valent 1/C. Si
l'ensemble ou l'ordre des candidats trahit t (H-b violée), le repli peut
dépasser 1/C sous `msg_shuffle` et pas sous le jumeau, où l'auditeur `eng`
suit le message. Un écart `msg_shuffle` − jumeau au-dessus du bruit binomial
(≈ 0,027 par graine) pointe vers une lecture de la cible dans les candidats ;
un écart dans l'autre sens, vers une autre violation de (H-a) à (H-e). Dans
les deux cas, c'est un signal d'audit, pas un résultat.

## 4. Puissance de la statistique de chute sur `eng`

### 4.1 Statistique et règle

`protocol.md` ne donne pas de règle de décision à NWML-N1 : secondaire
exploratoire, sans α dépensé ni verdict propagé (`protocol.md:495`). Pour
vérifier la puissance « au α du dessin », on applique à la chute la machinerie
de la règle IC de H1 (`protocol.md:143-147`, `decision.py:49-65`) avec une
borne 0 au lieu de la marge :

- chute par graine : drop_i = S_eng,i − S_msg_shuffle,i, n_eval = 500
  épisodes chacun ; unité = graine, k = 20 par backend (`protocol.md:170`) ;
  pas de mise en commun des épisodes (le tour 2 l. 305-310 tire de Xu & Chen
  §3.3 la même raison que le nul apparié par graine) ;
- règle R0 : test des signes unilatéral à α = 0,05 qui rejette (au moins
  m_k = 15 graines positives sur 20, P(Bin(20, ½) ≥ 15) = 0,0207,
  `sigma_seed.py:70-75`) **et** borne basse de l'IC 95 % de la moyenne > 0 ;
  IC percentile bootstrap (`protocol.md:146-147`) et, en parallèle, t apparié
  (l'autre option ouverte de TBD-16, `protocol.md:198-201`).

### 4.2 Modèles d'effet

- Fait (§ 2.3) : E[S_msg_shuffle] = 1/C, comme le plafond aveugle de `nocomm`.
  Inférence : sous l'hypothèse de planification bimodale de `protocol.md:150-156`,
  drop_i suit la même loi que D_i : N(0,75 ; 0,03) pour une graine qui apprend
  (fraction π), N(0 ; 0,027) sinon.
- Effet homogène δ (utilisation partielle par chaque graine) :
  S_eng ~ Bin(500 ; 0,25 + δ_i)/500, δ_i ~ N(δ ; τ), τ ∈ {0 ; 0,03} ;
  S_msg_shuffle ~ Bin(500 ; 0,25)/500 indépendant (hypothèse prudente : des
  épisodes appariés entre intact et dérangé réduiraient la variance).

### 4.3 Résultats (Monte-Carlo, 2 000 simulations, 1 000 rééchantillons ; erreur type ≤ 0,011)

Mélange bimodal (puissance de R0 par backend ; percentile et t identiques à
10⁻³ près ; le test des signes est ce qui lie) :

| π | 0,50 | 0,60 | 0,65 | 0,70 | 0,75 | 0,80 |
|---|---|---|---|---|---|---|
| P(R0) | 0,609 | 0,813 | 0,872 | 0,938 | 0,970 | 0,993 |

Contrôle analytique : P(R0) ≈ P(Bin(20, q) ≥ 15) avec q = (1 + π)/2 ;
puissance 0,90 atteinte à q ≈ 0,834, soit π ≈ 0,67. Avec la marge 0,30 de H1
au lieu de 0 : 0,907 (percentile) / 0,893 (t) à π = 0,75 et 0,972 / 0,966 à
π = 0,80, cohérent avec le Monte-Carlo de H1 (0,899 et 0,966,
`protocol.md:187-191`), ce qui recoupe le modèle.

Effet homogène (puissance de R0 percentile) :

| δ | 0 | 0,01 | 0,015 | 0,02 | 0,025 | 0,03 | 0,04 | 0,05 |
|---|---|---|---|---|---|---|---|---|
| τ = 0 | 0,011 | 0,169 | 0,369 | 0,610 | 0,850 | 0,944 | 0,994 | 1,000 |
| τ = 0,03 | 0,007 | 0,088 | 0,175 | 0,338 | 0,489 | 0,648 | 0,897 | 0,983 |

Lecture (inférences) :

- **L'effet connu sur `eng` est détecté.** À l'hypothèse de planification
  π = 0,8 (chute moyenne 0,60), P(R0) = 0,993 par backend, ≈ 0,986 pour les
  deux ; à π = 0,75, 0,970 (≈ 0,94). Le contrôle négatif a donc de la puissance
  au sens de Xu & Chen §7.2 (tour 2 l. 94-96, l. 300-304), **sous l'hypothèse
  de planification** : c'est une simulation, pas une démonstration sur données.
- **Effet minimal détectable à puissance 0,9** (k = 20, n_eval = 500,
  α = 0,05) : δ ≈ 0,028 sans hétérogénéité, ≈ 0,04 avec τ = 0,03 (soit 4 à
  5 % du contraste oracle 0,75, `protocol.md:56-57`) ; en mélange, π ≥ ≈ 0,68.
  Risque de première espèce sous δ = 0 : 0,011 (le test des signes discret est
  conservateur, niveau nominal 0,0207).
- Conséquence pour `learned` (si le PI y applique `msg_shuffle`, non décidé,
  `protocol.md:495`) : un non-rejet n'exclut qu'une utilisation homogène
  ≥ 0,03-0,04 ou une utilisation par plus de deux tiers des graines ; il ne dit
  rien d'un canal utilisé par moins de ≈ 2/3 des graines. C'est la limite du
  test des signes sous bimodalité, pas de n_eval.
- Le choix de la méthode d'IC (TBD-16) n'a pas d'effet ici : avec une borne 0,
  le test des signes lie, percentile et t donnent la même puissance à 0,01 près.

### 4.4 Coût en runs et en heures

- **Entraînement : zéro run de plus.** Le bras `eng` est entraîné pour H1
  (k = 20 par backend, `protocol.md:170`) et `msg_shuffle` est déjà une
  évaluation seule de la paire `eng` gelée (`protocol.md:90`,
  `protocol.md:106-107`, `protocol.md:495`) : contrôle 2 et NWML-N1 la
  prévoient déjà.
- **Évaluation : zéro de plus** pour la vérification de puissance sur `eng` :
  les 2 × 20 = 40 évaluations `msg_shuffle` de 500 épisodes sont déjà dans le
  dessin. Elle ne demande que l'application de R0 aux chutes `eng` et le
  calcul ci-dessus (simulation, 37 s, aucune donnée).
- Ajouts éventuels : `msg_shuffle` sur `learned` = 40 évaluations ; jumeau
  (§ 3.3) = 40 évaluations par bras perturbé.
- **Heures** : aucun coût par run N10-D n'existe (TBD-11 ouvert, « no figure
  yet », `protocol.md:44,508-509`). Le seul chiffre d'évaluation disponible est
  celui de N10-C (Procgen) dans PR #114 : 6,7 min par modèle, KXKM seulement
  (`docs/prereg/n10c-bis/three-seeds-plan-2026-11-06.md:78`, branche
  `docs/n10c-bis-three-seeds-2026-11-06`). Borne haute pessimiste, d'un autre
  jeu : 40 évaluations × 6,7 min ≈ 4,5 h KXKM. Aucun chiffre d'évaluation
  macM1 n'existe ; par le rapport d'entraînement macM1/KXKM de la même source
  (3,64 h contre 2,07 h par 10 M de pas, l. 78), ≈ 7,9 h macM1 (inférence
  grossière). Inférence : un épisode N10-D est un seul passage avant de deux
  petits MLP (tronc 2 × 256 en valeur de travail, `protocol.md:44`) ; 500
  épisodes devraient prendre des secondes, à mesurer avec TBD-11.

## 5. Texte d'amendement proposé (pour `protocol.md` § 8 et § 9 ; NON appliqué)

Les trois phrases A21-1 à A21-3 traduisent les décisions du PI du 2026-11-07
09:28 CET ; A21-4 est le constat de puissance qui les justifie. À reporter
telles quelles en § 8 (liste des amendements) ; la ligne de § 9 suit.

> - **A21-1** (control 1, context-conditioned derangement; PI decision
>   2026-11-07 09:28 CET, option b). No context-conditioned derangement
>   control is added: the speaker observes the target identity only (section
>   7, item 2), and under independent episodes, a target uniform over the C
>   slots given the listener observation, a derangement drawn independently
>   of the targets, per-episode listener decisions and valid actions, the
>   expected `msg_shuffle` success is exactly 1/C for any listener, whatever
>   the speaker observes and whether or not the derangement is conditioned on
>   the visible context (per-context groups would be degenerate at n_eval =
>   500). Control 4a gains two assertions, committed with the environment
>   hash: (i) the speaker observation holds the target identity only (no
>   candidate, no order, no episode index; for EGG, the data-file fields read
>   by the sender); (ii) at evaluation the listener decides each episode from
>   its own observation and message only (no batch statistics, no state shared
>   across episodes). A twin evaluation is kept as an exploratory evaluation
>   only: for each evaluation episode the candidates are kept, a twin target
>   is redrawn uniformly among the C candidates (coincidence allowed), and the
>   listener receives the frozen speaker's message for the twin target; 40
>   evaluations per perturbed arm (k = 20 seeds x 2 backends), no training.
>   Its expectation equals that of `msg_shuffle` (1/C); it checks that the
>   listener does not read the target from the candidate set. It spends no
>   alpha, is not a gate, and enters neither the outcomes of section 5 nor the
>   reading of NWML-N1.
> - **A21-2** (reading condition of NWML-N1; PI decision 2026-11-07 09:28
>   CET). Rule R0 on the per-seed drop (one-sided sign test at alpha 0.05,
>   15 of 20 seeds positive, and lower bound of the 95 % CI of the mean drop
>   > 0) is the positive power control of the perturbation test: a
>   non-rejection of R0 on `learned` is read as "no detectable use" only if
>   R0 rejects on `eng` in the same backend; otherwise the `learned` result is
>   "non concluant". NWML-N1 stays secondary exploratory and spends no alpha.
> - **A21-3** (pairing of the perturbed evaluation, section 3 at
>   `protocol.md:92`; PI decision 2026-11-07 09:28 CET). The intact and every
>   perturbed evaluation (`msg_shuffle`, twin) of a frozen pair use the same
>   n_eval = 500 evaluation episodes (same episode seeds), and the
>   perturbation batch of `Channel.transmit` is the whole set of these 500
>   episodes.
> - **A21-4** (power, fact of simulation, no run data). Planning-model power
>   of R0 on `eng` at k = 20, n_eval = 500: 0.993 per backend at pi = 0.8,
>   0.970 at 0.75; minimal detectable homogeneous drop at power 0.9 about
>   0.03 (0.04 with a between-seed sd of 0.03); about two thirds of learning
>   seeds under the bimodal model; type I 0.011. No extra training run; the
>   40 `msg_shuffle` evaluations are already planned.

Ligne proposée pour `protocol.md` § 9 (nouvelle sous-section « 2026-11-07
09:28 CET (A21) ») :

> | A21 | Control 1 void, two control-4a assertions, twin evaluation kept as
> exploratory (A21-1); R0 on `eng` required to read a non-rejection on
> `learned` (A21-2); same 500 episodes and one 500-episode perturbation batch
> (A21-3). Open: `msg_shuffle` on `learned`; NH-113 wording |

## 6. Décisions du PI

### Décidé (2026-11-07 09:28 CET)

1. **Contrôle 1** : option **(b)**. Déclaré sans objet pour les raisons du
   § 2 ; deux assertions ajoutées à l'audit 4a (observation de l'émetteur
   réduite à t ; aucun couplage entre épisodes à l'évaluation) ; jumeau à cible
   retirée gardé comme évaluation exploratoire, 40 évaluations par bras
   perturbé, sans entraînement, hors des verdicts de H1 et de NWML-N1 (§ 3.3,
   A21-1).
2. **Condition de lecture de NWML-N1** : **oui**. Le rejet de R0 sur `eng`
   dans le même backend est le contrôle positif de puissance exigé pour lire
   un non-rejet sur `learned` comme « pas d'usage détectable » ; sinon le
   résultat est « non concluant » (A21-2).
4. **Appariement et lot** : **fixé**. Mêmes 500 épisodes pour l'évaluation
   intacte et les évaluations perturbées, lot de perturbation = les 500
   épisodes (A21-3, `protocol.md:92`).

### Ouvert pour le PI

3. **`msg_shuffle` sur `learned`** (non décidé, `protocol.md:495`) : les
   tailles détectables du § 4.3 disent ce qu'un non-rejet exclurait
   (utilisation homogène ≥ 0,03-0,04, ou par plus de ≈ 2/3 des graines). A21-2
   ne s'applique que si ce point est tranché par oui.
5. **Lien avec la décision 1 du tour 2** (l. 379-382, NH-113 et la
   « randomisation conditionnelle sur l'état complet ») : si le PI garde cette
   mention, le jumeau du § 3.3 en est la seule forme licenciée par le plan
   (tirage dans la loi connue de t sachant le contexte de l'auditeur), et le
   § 2.3 montre qu'elle a la même espérance que `msg_shuffle`.

Ces chiffres reposent sur l'hypothèse de planification bimodale
(`protocol.md:150-156`) et sur un article lu en partie (tour 2 l. 67-73) ; le
tour 2 demande de recouper 2611.01666 avant toute modification de la préreg
(l. 312-316).

## Annexe — script du Monte-Carlo (Python 3.14, bibliothèque standard)

Lancé sur GrosMac le 2026-11-07 : `python3 -I power.py 2000` (37 s). Le calcul
des groupes du § 3.2 est exact (formule d'occupation en tête de script).

```python
"""A21 power check of the msg_shuffle drop statistic on eng (stdlib only).
drop_i = S_eng,i - S_shuf,i per seed, k = 20 seeds, n_eval = 500.
Rule R0: one-sided sign test (m_k = 15) rejects AND lower bound of 95 %
CI of mean(drop) > 0. CI: percentile bootstrap (protocol section 4) or
paired t (TBD-16 alternative)."""
import random, statistics, math, sys, time

# Section 3.2: P(episode alone) = (1 - 1/G) ** (n - 1), n = 500,
# G = comb(16, 4) = 1820 (set) or 16 * 15 * 14 * 13 = 43680 (tuple).
K = 20; N = 500; M_K = 15
SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; BOOT = 1000
T975_19 = 2.093024
rng = random.Random(20261107)

def lower_pct(d):
    means = sorted(statistics.fmean(rng.choices(d, k=K)) for _ in range(BOOT))
    return means[min(BOOT - 1, int(0.025 * BOOT))]

def lower_t(d):
    return statistics.fmean(d) - T975_19 * statistics.stdev(d) / math.sqrt(K)

def bern(p):
    return rng.binomialvariate(N, min(1.0, max(0.0, p))) / N

def run(draw, margin=0.0):
    hp = ht = hs = 0
    for _ in range(SIMS):
        d = [draw() for _ in range(K)]
        s = sum(x > 0 for x in d) >= M_K
        hs += s
        if s and lower_t(d) > margin: ht += 1
        if s and lower_pct(d) > margin: hp += 1
    return hp / SIMS, ht / SIMS, hs / SIMS

t0 = time.time()
print("bimodal: pi, P(R0 pct), P(R0 t), P(sign)")
for pi in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
    f = lambda pi=pi: rng.gauss(0.75, 0.03) if rng.random() < pi else rng.gauss(0.0, 0.027)
    print(pi, *("%.3f" % x for x in run(f)))
print("bimodal, margin 0.30:")
for pi in (0.75, 0.8):
    f = lambda pi=pi: rng.gauss(0.75, 0.03) if rng.random() < pi else rng.gauss(0.0, 0.027)
    print(pi, *("%.3f" % x for x in run(f, 0.30)))
print("homogeneous: delta, tau, P(R0 pct), P(R0 t), P(sign)")
for tau in (0.0, 0.03):
    for delta in (0.0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        f = lambda delta=delta, tau=tau: bern(0.25 + (rng.gauss(delta, tau) if tau else delta)) - bern(0.25)
        print(delta, tau, *("%.3f" % x for x in run(f)))
print("elapsed %.1f s" % (time.time() - t0))
```

Sortie brute : celle des tableaux du § 4.3 (colonnes percentile, t, signes
seuls ; les trois colonnes coïncident à la marge 0 sauf aux faibles δ, à
0,01 près).
