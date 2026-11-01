# Brainstorm N10-D — dessins alternatifs (2026-11-01)

**Aucun de ces dessins ne part avant que le verrou soit passé** (saturation,
recoupement, couverture : voir `reading-note.md` § 6). Ce document ne modifie
ni `protocol.md` (PR #100) ni aucune préreg ; il prépare une discussion PI.

Références internes : `MS` = `docs/milestones/2026-10-25-n10c-bis-control-arms.md` ;
`P100` = PR #100 (`protocol.md`, `nh96-null.md`) ; `RAP` = `synthese/rapport.md`
du dossier `.github` (PR #23).

## 0. Corrections déjà décidées par le PI (à intégrer dans tout dessin)

1. **`sym` sort de H1.** Fait : `P100` déclare `sym` « contrôle non
   informatif » (`nh96-null.md` §5) mais l'inclut dans H1 (`nh96-null.md` §1) ;
   relevé en `RAP:94` (M9). `sym` reste un contrôle de fuite rapporté à côté.
2. **Le nul tient compte de la covariance de graine.** Fait : le nul de `P100`
   tire chaque bras indépendamment par graine (`nh96-null.md` §2). Tout
   dessin ci-dessous utilise un nul apparié (permutation des bras à
   l'intérieur de chaque graine, ou effet de graine commun aux deux bras).
3. **Porte N10-C-bis : 3 graines appariées, ou abandon de bis au profit de
   N10-D corrigé** (option (a) de la question Q8, `RAP:285`). Fait : la porte
   actuelle tient en 1 graine et 1 run par bras ; `RAP:93` (M8) estime ≈ 8 %
   de faux succès et ≈ 24 % de faux échecs à 1 graine, 0,8 % et 11 % à 3
   graines (inférence de l'agent, effet vrai supposé 0,30).

## 1. Hypothèses de coût (communes)

- Fait : un run N10-C sur KXKM ≈ 10,8 h (`MS:179`) ; la portée bis-A ≈ 526 h
  ≈ 21,9 jours de KXKM (`MS:199`, `MS:324`).
- Fait : le débit d'un jeu référentiel K = 16 sous JaxMARL ou EGG n'est pas
  mesuré ; les environnements réels n'existent pas encore (`P100` §6).
- Inférence : un run de jeu référentiel est d'un ou deux ordres de grandeur
  plus court qu'un run Procgen. Les coûts ci-dessous en heures GPU KXKM sont
  des **estimations à mesurer au pilote**, notées « est. ».

## 2. Dessins

### D1 — Bras « usage causal du canal » en test (message permuté / gelé / coupé)

- **Dessin.** Sur chaque agent `learned` entraîné en `asym`, quatre conditions
  d'évaluation : intact ; message permuté entre épisodes ; message gelé
  (constant, valeur du premier épisode) ; canal mis à zéro ; plus restauration
  (retour à l'intact) pour vérifier qu'on mesure le message et non un état
  abîmé. Métrique d'appoint : CIC de Lowe 2019.
- **Nul calibré.** Permutation des étiquettes de condition à l'intérieur de
  chaque graine et de chaque lot d'épisodes (nul apparié) ; P99 sur 10 000
  tirages.
- **Réfutation.** Usage causal réfuté si l'IC 95 % apparié de (intact −
  permuté) contient 0 alors que `learned − nocomm` dépasse le P99 : gain
  sans usage du message (cas Chauhan & Bera, `MS:328` gtm_mis).
- **Coût.** Évaluation seule sur des agents déjà entraînés : est. < 2 h GPU ;
  ≈ 6-8 h humaines (hooks d'intervention dans les deux backends).
- **Ce que ça tranche.** Si le message entre causalement dans la politique de
  l'auditeur — la question ouverte n° 1 de la note.
- **Antériorité la plus proche.** Lowe 2019 (arXiv:1903.05168, CIC et messages
  brouillés) ; OpenReview hukTsKYu8Y (couper / substituer / restaurer ;
  lu en entier pp. 1-38 le 01/11, voir § 5) ; Cheng et al. arXiv:2608.04893 `[résumé moteur]`.
- **Limite.** Lowe §6.2 : couper le canal en test crée un décalage de
  distribution ; d'où la restauration et D2 en complément.

### D2 — Bras appariés entraînés de zéro avec et sans canal (asym seule dans H1)

- **Dessin.** `learned` et `nocomm` entraînés de zéro, mêmes graines, `asym`
  seule dans H1 ; `eng` et `msg_shuffle` gardés comme contrôles de chaîne ;
  `sym` en contrôle de fuite. C'est la recommandation de Lowe §6.2.
- **Nul calibré.** Substitut binomial de `P100` §2 avec un effet de graine
  **commun** aux deux bras plus un effet bras × graine, ou permutation des
  bras à l'intérieur de chaque graine.
- **Réfutation.** IC 95 % apparié de `learned − nocomm` sous 0,10 avec la
  porte d'apprentissage passée ; « non mesurable » si `eng` échoue.
- **Coût.** k graines × 2 bras (k par la formule de `P100` §4, minimum 5) :
  est. 10-20 h GPU ; ≈ 15-25 h humaines (environnements réels, tronc PPO
  adapté, `P100` §6).
- **Ce que ça tranche.** Si un protocole appris émerge et aide sur une tâche
  où il est nécessaire ; ne dit pas à lui seul s'il est utilisé causalement.
- **Antériorité.** Lowe 2019 ; IC3Net arXiv:1812.09755 (vision nulle,
  IC3Net 93,0 contre IRIC 29,8).

### D3 — Récepteur démarré à chaud

- **Dessin.** Auditeur pré-entraîné à lire un code conçu (`eng`), puis
  entraînement conjoint avec un émetteur appris ; comparé à un départ à froid.
- **Nul calibré.** Nul apparié de D2 sur (chaud − froid), par graine.
- **Réfutation.** Pas d'écart chaud − froid au-delà du P99 : le goulot n'est
  pas la co-adaptation du récepteur.
- **Coût.** + k runs : est. 10-15 h GPU ; ≈ 6 h humaines.
- **Ce que ça tranche.** Si un échec de `learned` vient de l'optimisation
  (co-adaptation) et non de la tâche ; utile seulement si D2 échoue.
- **Antériorité.** Chauhan & Bera arXiv:2609.34373 `[résumé moteur]`
  (0,857 à chaud contre 0,562 à froid) ; `MS:302` gtm_warm (une graine).

### D4 — Cross-play entre graines

- **Dessin.** Émetteur de la graine i face à l'auditeur de la graine j ≠ i,
  sur les agents de D2.
- **Nul calibré.** Distribution des paires i ≠ j permutées contre paires
  i = i, appariée par auditeur.
- **Réfutation.** Effondrement du cross-play au niveau de `nocomm` : chaque
  paire a un code privé ; aucune réfutation de l'usage, mais de la
  convergence du code.
- **Coût.** Évaluation seule : est. < 1 h GPU ; ≈ 3 h humaines.
- **Ce que ça tranche.** Code partagé contre code idiosyncratique ; relie au
  thème « structure contre géométrie ».
- **Antériorité.** Chauhan & Bera (cross-play 0,980 → 0,144) `[résumé
  moteur]` ; You et al. arXiv:2609.27252 (convergence de structure, pas de
  géométrie, lu en entier) ; cross-play déjà prévu dans `MS:195`.

### D5 — Code hebbien fixe ou décodeur à attracteurs

- **Dessin.** Remplacer l'émetteur appris par un code fixe et l'auditeur par
  un décodeur à attracteurs (mémoire associative dense), à bruit de canal
  croissant.
- **Nul calibré.** Décodeur à attracteurs sur messages permutés ; nul apparié
  par graine de bruit.
- **Réfutation.** Le décodeur fixe égale `learned` à tout niveau de bruit :
  l'apprentissage du code n'apporte rien sur cette tâche.
- **Coût.** CPU majoritairement : est. < 2 h GPU ; ≈ 10 h humaines.
- **Ce que ça tranche.** Si l'intérêt de N10-D est l'apprentissage du code ou
  seulement la transmission ; donne la référence « Related work » demandée en
  `RAP:92` (M7).
- **Antériorité.** Agliari et al. arXiv:2609.32605 et arXiv:2610.08612 (même
  groupe, lus en entier) ; `RAP:92` note que 2609.32605 recouvre le Test 12,
  pas N10-C.

### D6 — Claim B testé sur le canal existant (CPU)

- **Dessin.** Sur les sorties déjà produites, MI(indices VQ ; cible) et
  MI conditionnelle MI(message ; cible | entrée de l'auditeur), en séparant
  argmax de π et indices VQ ; nul intra-cluster.
- **Nul calibré.** Permutation intra-cluster (≈ 0,06 selon `RAP:57`, B4,
  agent seul).
- **Réfutation.** MI conditionnelle dans le nul : le canal n'ajoute rien à ce
  que l'auditeur sait déjà.
- **Coût.** 0 h GPU ; ≈ 4-6 h humaines ; à lancer seulement sur un hôte
  autorisé, hors du présent chantier.
- **Ce que ça tranche.** Si Claim B mesure l'accord de deux classifieurs
  (inférence de `RAP:56`, B3) ou un usage du canal.
- **Antériorité.** Interne ; méthode MI conditionnelle standard.

### D7 — Porte N10-C-bis à 3 graines appariées, contre abandon

- **Dessin.** (a) rejouer la porte G2 (`eng − nocomm ≥ 0,20`, `MS:338`) sur
  3 graines appariées, ou (b) abandonner bis et faire N10-D corrigé.
- **Nul calibré.** Nul apparié par graine (correction n° 2) ; taux d'erreur
  écrits d'avance (`RAP:93`).
- **Réfutation.** Porte échouée à 3 graines → bis non gelé (G3), N10-D.
- **Coût.** (a) 3 graines × 2 bras × run de porte à 10 M pas (`MS:335`) ;
  durée par run de porte non consignée dans `MS` ; borne haute est. 6 ×
  10,8 h ≈ 65 h si un run de porte coûtait un run N10-C (`MS:179`) ;
  ≈ 4 h humaines. (b) 0 h GPU sur bis ; libère ≈ 526 h (`MS:199`).
- **Ce que ça tranche.** Si la tâche Procgen a besoin du message ; condition
  pour dépenser 526 h sur bis.
- **Antériorité.** `MS` G1-G7 ; Lowe 2019 ; DVNC arXiv:2107.02367 §5 (le gain
  n'apparaît que si la communication est importante pour la tâche).

## 3. Classement « ce que ça tranche / coût »

Inférence de l'auteur, à discuter avec le PI. Coût = GPU est. + humain.

| rang | dessin | tranche | coût | rapport |
|---|---|---|---|---|
| 1 | D6 Claim B sur canal existant | question interne bloquante (B3) | 0 GPU, ≈ 5 h | très élevé |
| 2 | D1 usage causal en test | question ouverte n° 1 | < 2 h GPU, ≈ 7 h (exige des agents D2) | très élevé |
| 3 | D2 bras appariés de zéro | nécessité + émergence, base de D1/D3/D4 | 10-20 h GPU, ≈ 20 h | élevé |
| 4 | D4 cross-play | code partagé ou privé | < 1 h GPU, ≈ 3 h | élevé (marginal sur D2) |
| 5 | D7 (b) abandon de bis | libère ≈ 526 h | 0 | élevé si la porte échoue |
| 6 | D5 code fixe / attracteurs | intérêt de l'apprentissage du code | < 2 h GPU, ≈ 10 h | moyen |
| 7 | D7 (a) porte à 3 graines | décision bis | ≤ 65 h GPU, ≈ 4 h | moyen |
| 8 | D3 démarrage à chaud | diagnostic d'un échec de D2 | 10-15 h GPU, ≈ 6 h | conditionnel |

Inférence : le paquet minimal cohérent est D2 + D1 + D4 (même jeu d'agents),
précédé de D6 (CPU), soit est. 12-23 h GPU et ≈ 30 h humaines.

## 4. Ce qu'on ne fait pas et pourquoi

- **Lire `eng − nocomm` sur asym comme un résultat.** Attendu par
  construction (`P100` §1, « VARIANT »).
- **Garder `sym` dans H1.** Décision PI (correction n° 1).
- **Couper le canal en test comme seul test d'usage.** Biaisé (Lowe §6.2) ;
  toujours avec restauration et D2.
- **Lancer bis-A sur une porte à 1 graine.** Taux d'erreur trop élevés
  (`RAP:93`) pour engager ≈ 526 h.
- **Étendre à MLP/LIF ou au chemin γ/θ dans N10-D.** Ouvert mais hors du
  dessin minimal ; à traiter après D1/D2, et seulement avec une capacité
  appariée (`RAP:151`, C4) et une correspondance π/ε relue (`RAP:89`, M5).
- **Toute exécution maintenant.** Verrou non passé ; aucun calcul sur KXKM,
  macM1 ou GrosMac dans ce chantier ; KXKM est d'ailleurs réservé après
  E1-small (`MS:348`, G5).

## 5. Après lecture de hukTsKYu8Y (01/11, pp. 1-38 lues en entier)

Rien n'est lancé ; aucun dessin ne part avant que le verrou soit passé.
Détail des faits et des pages : `reading-note.md` § 2 (complément v0.3).

- **D1 — en partie tombé.** Fait : hukTsKYu8Y fait déjà le canal coupé (OFF),
  le contenu substitué depuis un autre input (transplant à trois valeurs) et
  une restauration, sur des modèles entraînés de zéro (pp. 4-7, p. 23). Fait :
  il ne fait ni message gelé (constant), ni bras sans canal entraîné, ni nul
  apparié par graine, et son canal est un paquet continu entre modules
  homogènes supervisés par entropie croisée (p. 3, p. 18). Inférence : D1 ne
  peut plus être présenté comme une intervention nouvelle ; ce qui reste neuf
  est son cadre (renforcement, neurolettres discrètes, `nocomm` entraîné,
  nul apparié). L'antériorité de D1 passe de `[résumé moteur]` à lu en entier.
  Inférence : leur sélection de cas à trois valeurs distinctes (p. 5) est une
  amélioration possible de D1, à chiffrer.
- **D2 — reste neuf.** Fait : aucun modèle n'y est entraîné sans canal (le
  déficit OFF est garanti par construction, Prop. 1 p. 4).
- **D3 — non touché** (pas de démarrage à chaud).
- **D4 — non touché** (deux initialisations par corpus, pas de cross-play,
  p. 23).
- **D5 — non touché** (pas de code fixe ni d'attracteurs).
- **D6, D7 — internes, non touchés.**
- **Nouveau candidat, non classé :** un test de mécanisme à la manière de leur
  plan 2 × 2 (restauration des entrées de l'auditeur contre transplant
  d'entrées seules) suppose une voie de réception identifiable ; inférence :
  pour N10-D cela demanderait d'instrumenter le chemin de réception de
  l'auditeur, coût non estimé, hors du dessin minimal.
