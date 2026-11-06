# N10-C-bis à 3 graines appariées — plan (brouillon, 2026-11-06)

**Statut :** BROUILLON de plan. Ni préenregistrement, ni gel, ni tag, ni lancement. Aucun calcul n'a été lancé pour l'écrire.
**Origine :** décision du PI du 06/11 vers 13:45 (« corriger PR #100 et préparer N10-C-bis 3 graines »). Question Q8 de la roadmap v2 (`hypneum-lab/.github`, `docs/ROADMAP.md:117`) : « N10-C-bis à 3 graines appariées ou N10-D corrigé, avant tout gel ». Issue nerve-wml #62, reformulée « geler après verrou » (décision du 01/11, jalon M1, 29 nov).
**Ne modifie pas :** les préregs et amendements N10-C (gelés ou déclarés), l'addendum N10-C-bis `docs/milestones/2026-10-25-n10c-bis-control-arms.md` (abrégé `MS`), l'annexe G7 gelée, la PR #100.
**Statut d'aveuglement (obligatoire).** Ce plan est écrit après la troisième rupture d'aveugle (lecture des résultats held-out de graine 0 le 05/11, `…-amendment-2026-11-04-interim.md:94`). Toute décision qu'il prépare est **non aveugle vis-à-vis des résultats de graine 0** et doit être déclarée comme telle. Il ne lit aucun résultat au-delà de ce que la closure publie (`…-closure-2026-11-05.md:51-63`).

Abréviations : `CL` = closure du 05/11 ; `IA` = amendement intérimaire du 04/11 ; `DEV` = déclaration du 04/11 matin ; `A1030` = déclaration du 30/10 ; `A1023`, `A1025` = amendements gelés ; `BS` = `docs/prereg/n10d/brainstorm-2026-11-01.md` ; `VER` = `docs/n10d/contradictor-2026-11-06/VERIFICATION.md` (branche `docs/n10d-contradictor-2026-11-06`) ; `RM` = roadmap v2 du dépôt `.github`.

## 0. Deux lectures de « N10-C-bis à 3 graines »

Les sources donnent deux objets sous le même nom. Les deux sont chiffrés ici. Ils ne s'excluent pas : le premier est le préalable du second.

- **Objet A — porte de nécessité G2 rejouée à 3 graines appariées.** C'est le sens de D7 (a) (`BS:140-151`) : arms `eng` et `nocomm`, condition `closed`, 10 M pas de paire, graines de développement, critère `eng − nocomm ≥ 0,20` (`MS:332-339`). Exploratoire par construction (`MS:346`).
- **Objet B — campagne N10-C-bis réduite à 3 graines appariées.** Les bras de run 2 et de l'addendum, 50 M pas de paire par run, analyse appariée par graine.

Fait : G3 (`MS:341-344`) fait de A la condition de tout gel de B. Inférence : préparer B sans A n'a pas d'objet. Ce plan recommande donc A d'abord, B seulement si A passe et si le PI choisit B contre N10-D à M1.

## 1. Ce que 3 graines appariées peuvent trancher, et ce qu'elles ne peuvent pas

### 1.1 Calculs

Notation : D_s = différence de deux bras sur la graine s (même tronc, même hôte), k = 3.

- **Test des signes, ou permutation exacte appariée.** Avec 2 bras par graine, la permutation des étiquettes à l'intérieur de chaque graine n'a que 2³ = 8 configurations. La plus petite p-valeur atteignable, à 3 signes positifs sur 3, vaut 1/8 = 0,125. Fait arithmétique : à α = 0,05, aucun résultat ne rejette. La puissance est **nulle** quelle que soit la taille de l'effet. (À k = 5, le minimum est 1/32 = 0,031, `VER:14`.)
- **t apparié, df = 2.** Valeur critique unilatérale à 0,05 : 2,920. Demi-largeur d'un IC 95 % bilatéral = 4,303 · σ_D/√3 = 2,48 σ_D, soit 0,25 à σ_D = 0,10. Au niveau 98,33 % des contrastes primaires de `MS:141-147`, t = 7,649 et la demi-largeur vaut 4,42 σ_D, soit 0,44 à σ_D = 0,10. Puissance unilatérale (α = 0,05) : 0,32 à d_z = 1 ; 0,71 à d_z = 2 ; 0,93 à d_z = 3. La normalité des D_s ne peut pas être vérifiée sur 3 valeurs.
- **Équivalence (TOST, marge Δ = 0,10).** Elle exige 1,686 · σ_D < 0,10 − |D̄|. Même à D̄ = 0, il faut σ_D < 0,059. Fait : les répétitions de développement à même graine diffèrent déjà de 0,10 à 0,13 (`A1025:159`). Inférence : une équivalence à k = 3 est hors d'atteinte. Elle n'est pas préenregistrable avec un sens.
- **Porte à seuil sur la moyenne (règle G2 telle quelle).** σ_D = 0,142 est une **inférence** : c'est la valeur qui reproduit les taux de `RAP:93` cités en `BS:20-24`. Aucun N10-C ne peut servir de pilote de variance (`CL:44`, `IA:95`). Sous cette hypothèse, P(D̄ ≥ 0,20 | D = 0) vaut 7,9 % à k = 1, 0,74 % à k = 3 et 0,08 % à k = 5. P(D̄ < 0,20 | D = 0,30) vaut 24,1 %, 11,1 % et 5,8 %. Ces chiffres sont recalculés (`python3 -I`, loi normale).
- **Porte « au moins m graines sur 3 au-dessus du nul apparié ».** Le seuil par graine est le P99 du nul. Il vaut 0,33 si le nul est N(0 ; 0,142). Pour m = 1 : erreur de type I 2,97 % (1 − 0,99³) et puissance 0,80 à D = 0,30. Pour m = 2 : 0,03 % et 0,37. Pour m = 3 : environ 0 et 0,07. Inférence : « au moins 1 sur 3 » est une porte permissive, pas une preuve. « 3 sur 3 » au P99 est trop exigeant pour être utile.
- **Règle composite** (D̄ ≥ 0,20 **et** au moins 2 graines sur 3 avec D_s ≥ 0,20). Par simulation (200 000 tirages, σ_D = 0,142) : type I 0,49 % et puissance 0,83 à D = 0,30. Avec D̄ ≥ 0,20 seul : 0,71 % et 0,89.

### 1.2 Ce qui a un sens à k = 3

1. **Objet A — une porte de décision, pas un test.** Il faut écrire avant tout run une règle à seuil dont les taux d'erreur sont publiés à côté. Recommandation : garder la règle G2 (`D̄ ≥ 0,20`, moyenne des D_s appariés). Cela évite de changer un seuil après la lecture de graine 0, ce qu'interdit `IA:119-129`. On ajoute deux contrôles pouvant échouer (§1.3). Le compte de signes est rapporté sans test, puisqu'il ne peut pas rejeter. Ce que la porte tranche : sur cette tâche, dans le harnais N10-C, avec le canal N10-C, le message idéal (`eng`) laisse-t-il ou non une marge d'au moins 0,20 sur le silence, en condition `closed` et sur les niveaux d'entraînement ? C'est en partie le « bras dans le harnais N10-C » que demande le constat 2.6 (`VER:28`). Inférence : il teste la tâche et la tuyauterie de N10-C, ce que N10-D, dans un autre harnais, ne peut pas faire.
2. **Objet B — estimation descriptive seulement.** À k = 3, les intervalles appariés à 98,33 % ont une demi-largeur d'environ 0,44 à σ_D = 0,10. Le seul résultat atteignable serait « pilot-inconclusive », sauf effet supérieur à environ 0,5. Aucune catégorie pilote ne peut être lue comme un gain. Ce que B ne peut pas trancher : une supériorité, une équivalence, ni la catégorie primaire de `A1023` ou `MS`.
3. **Ce que ni A ni B ne tranchent :** la question « le GTM communique-t-il ». La règle de lecture R1-R5 de l'addendum reste en vigueur. Sous R3 (`MS:131`), N10-C reste ininterprétable comme preuve de communication.

### 1.3 Liste « nuls calibrés » appliquée à l'objet A

| Exigence | Mise en œuvre proposée |
|---|---|
| Porte d'apprentissage d'abord | Chaque run passe d'abord le critère B3 (`MS:290`) : pas de sonde de vivacité signalée après le pas 0, pas d'avortement, taux de pièce en hausse. Un run qui échoue est relancé une fois, selon `MS:339`. S'il échoue encore, la graine est « non mesurable ». D n'est calculé que sur des runs qui ont appris. |
| Contrôle positif pouvant échouer | `eng` en `closed` doit dépasser son propre point de départ et atteindre au moins 5 % de succès d'équipe (`A1025:90`). Si `eng` échoue sur au moins 2 graines, la porte est **non mesurable**, pas « échouée ». |
| Nul apparié par graine, même pipeline | Un troisième run par graine, `nocomm_rep` : `nocomm` sur un autre flux aléatoire, même tronc, même hôte, même code. D0_s = `nocomm_rep − nocomm` mesure le bruit de la mesure dans le pipeline exact. Contrôle de bruit : si |D0_s| ≥ 0,10 (la SESOI) sur au moins 2 graines, la porte est **non concluante, bruit trop élevé**. Fixé avant tout run. Coût : +3 runs. Variante à 0 run de plus : un nul de permutation intra-graine, mais il n'a que 8 états (§1.1) et ne calibre rien. |
| Non-rejet ≠ réfutation | Un échec de porte se lit : « sur cette tâche et ce budget, le message idéal n'a pas laissé une marge de 0,20 ». Il ne se lit pas : « la communication est inutile » ni « le canal GTM ne transmet pas ». La conséquence G3 (pas de gel de B, N10-D) reste une règle de décision, pas une conclusion scientifique. |
| Mesure | La moyenne des 5 derniers logs de succès sur 100 épisodes (`MS:337`) n'est pas binomiale (fenêtres qui se recouvrent, `VER:28`, constat 2.7). On la garde comme mesure de la porte. On ne lui attache aucun nul binomial. |

## 2. Bras, et ce que la pré-analyse de graine 0 impose de changer

### 2.1 Bras de run 2 et de l'addendum

- Run 2 (`A1023:113`, `A1023:109`) : 6 bras en condition `registered` (`gtm`, `nocomm`, `eng`, `dial`, `mlp`, `atoc`) et 4 bras en `closed` (`gtm`, `nocomm`, `eng`, `dial`), soit 10 cellules par graine. Seuls 9 runs de graine 0 ont été entraînés et évalués (`CL:20`). `registered-atoc-s0` n'a jamais été entraîné (`CL:30`).
- Addendum (`MS:84-93`) : `gtm_mis`, `eng_gtmrx`, `gtm_warm`, `vq`, chacun dans les deux conditions (8 cellules). Les 4 bras passent la porte d'apprentissage (`MS:318`). Le comparateur prévu était le groupe N10-C de la même condition (`MS:139`, `MS:266`). Ce groupe n'existe pas : N10-C est arrêté à une graine (`CL:43-45`). Fait : un N10-C-bis à 3 graines doit donc **ré-entraîner lui-même** `gtm`, `nocomm` et `eng` sur ses propres graines.

### 2.2 Ce que la closure publie, et ce qui en découle

Tout ce qui suit vient de `CL:58-61`. Ce sont des lectures **informées par les résultats**. `IA:121-129` interdit de changer, sur leur base, les bras, les cellules, les seuils et les métriques de N10-C-bis sans **nouvelle déviation déclarée comme informée par les résultats**. Chaque changement ci-dessous porte donc cette étiquette, sauf si un fondement antérieur au 05/11 est cité.

| Constat de la closure | Conséquence proposée pour B | Fondement antérieur à la lecture ? |
|---|---|---|
| `nocomm` trop bon : 0,94 de succès en entraînement, 0,415 en held-out, au niveau des bras qui communiquent (`CL:58` ; la condition n'est pas précisée) | La tâche telle qu'elle est ne laisse pas de place au message. B n'a de sens qu'après une porte A réussie. Sinon, application de G3 et passage à N10-D. | **Oui** : G1 (`MS:324-330`), avec `gtm_mis` à 0,81 et une politique scriptée à environ 43 %, sur des données de développement (`CL:53`) |
| `time_to_half` dégénéré : égal au premier point de log (1 M) pour les 9 runs (`CL:59`) | Remplacer la métrique d'efficacité (`A1023:197`) : logs dès le pas 0 tous les 100 k pas, et aire sous la courbe de 0 à 5 M, ou seuil fixé hors données. Changement de métrique = déviation déclarée. | Partiel : départ à chaud du tronc et récompense de pièce propre (`A1025:54-61`) le laissaient prévoir. Ce n'était pas écrit. |
| Écart de généralisation de 0,40 à 0,54 entre entraînement et held-out (`CL:60`) | Le held-out mesure surtout la généralisation de niveaux. Pour B : rapporter les deux mesures, et désigner avant le gel celle qui est primaire. La porte A utilise déjà les niveaux d'entraînement. | Non |
| « 0,96 » : succès final d'entraînement de `registered-gtm-s0`, lu le 30/10 (`CL:73`) | Effet plafond possible sur la mesure d'entraînement en `registered`. Raison de plus pour la condition `closed`. | Non (rupture déclarée, `A1030` §1) |
| Effets `gtm − nocomm` de +0,025 et +0,060 sur une graine ; S compatible avec 0 (`CL:61`) | Aucun : ne sert ni de seuil, ni de taille d'effet, ni de pilote (`CL:44`). | — |

Le brief cite aussi « `mlp` nul ». **La closure ne le publie pas** : `mlp` n'apparaît que dans la liste des runs évalués (`CL:20`). Ce plan ne s'en sert pas. Le PI est seul juge d'une lecture du rapport intérimaire au-delà de la closure.

### 2.3 Jeu de bras proposé pour B (seulement après une porte A réussie)

- **B' (recommandé si B est fait) : condition `closed` seule.** Bras `gtm`, `nocomm`, `eng`, `gtm_mis`, `vq`, plus `nocomm_rep` (nul apparié) : 6 cellules × 3 graines = 18 runs. Fondements antérieurs au 05/11 : `closed` est le régime discriminant (`MS:155`, Chauhan & Bera) ; `vq` remplace les baselines 2016-2018 (`MS:27`) ; `gtm_mis` porte la question « contenu ou présence » (`MS:121`). On retire `dial`, `mlp`, `atoc`, `eng_gtmrx` et `gtm_warm`. Retrait déclaré **informé par les résultats** au titre de `IA:117`, même si ses fondements écrits sont antérieurs.
- **B (complet) : 18 cellules × 3 graines = 54 runs.** Ce sont les 10 cellules de run 2 et les 8 de l'addendum. Donné pour comparaison de coût seulement.
- **Graines :** nouvel espace, qui exclut la graine 0 et toute graine regardée (`IA:111`). Un tronc préentraîné par nouvelle graine, coût faible (`A1025:205`).
- **Analyse :** appariée par graine (différence par graine, t à df = 2). Cela répond à la correction de covariance de `A1030:40-46`. Le texte actuel de `MS:139` (« no pairing by seed ») doit être amendé avant tout gel.

## 3. Coût

Taux mesurés : KXKM-AI, run de porte de 10 M = 7 434 s ≈ 2,07 h (`MS:348`). macM1, run de 50 M en run 2 = 18,2 h en moyenne sur les runs 1 à 7 (`DEV:20-26`), soit environ 3,64 h par 10 M en extrapolant. KXKM, run de 50 M ≈ 10,8 h (`A1023:332`). Évaluation : 6,7 min par modèle, KXKM seulement (`A1023:336`).
« Mixte » signifie que chaque graine reste en bloc sur un seul hôte (les bras d'une graine ne changent pas d'hôte) : 2 graines sur KXKM en parallèle d'1 graine sur macM1. **Déviation de substrat déclarée** vis-à-vis de G5 (KXKM seul, `MS:348`). Les builds Procgen ne sont pas identiques au bit près (`A1023:342`). macM1 met en pause le modèle de jour du second cerveau pendant toute la durée.

| Variante | Runs | KXKM seul | macM1 seul | Mixte (mur) | Heures PI (est.) | Fenêtre compatible avec `RM` |
|---|---:|---:|---:|---:|---:|---|
| A : porte, 3 graines × (`eng`, `nocomm`) | 6 | 12,4 h | 21,8 h | ≈ 8,3 h | ≈ 6 | S3-S5 sur KXKM (libre, `RM:155`), si le PI l'exempte du verrou (Q2) |
| **A+ : A + `nocomm_rep` (nul apparié)** | 9 | **18,6 h** | 32,8 h | ≈ 12,4 h | ≈ 8 | idem ; résultat disponible avant M1 (29 nov) |
| B' : `closed`, 6 cellules × 3 graines | 18 | 194 h ≈ 8,1 j | 328 h ≈ 13,7 j | ≈ 130 h ≈ 5,4 j | ≈ 30 (verrou ≈ 15-19, préreg et manifeste ≈ 7, suivi et analyse ≈ 6) | après verrou : gel au plus tôt S7, calcul S8-S9, fini avant M2 (3 jan) **seulement à la place de N10-D** (`RM:174`, une préreg ; `RM:164`, une campagne lourde) |
| B : 18 cellules × 3 graines | 54 | 583 h ≈ 24,3 j | 983 h ≈ 41 j | ≈ 389 h ≈ 16,2 j | ≈ 37 | **incompatible avec M2** : 24 j à partir de S8 mène après le 3 jan |

S'ajoutent les évaluations (B' ≈ 2 h, B ≈ 6 h sur KXKM) et, pour B, le cross-play (3 paires par cellule à 3 graines). Les heures PI sont des estimations de l'auteur. Le poste C de la roadmap (calcul, 15 h PI sur le trimestre, `RM:43`) ne couvre que A+.
**Le GB300 n'entre pas dans ce tableau** : l'annexe G7 le déclare indicatif, il ne décide rien (`g7-host-annex.md:26`, D2). Son pilote utilise les graines 9101-9103 (`MS:352`). Aucun PR de résultats G7 n'existe sur `master` au 06/11. Ce plan ne lit pas ses fichiers scellés (Q3).

## 4. Verrou avant gel, et lien avec N10-D

**Ce que le verrou exige pour geler B** (`RM:14`, `RM:18-22`) :
- une note de lecture propre à N10-C-bis : contrôles contenu ou présence, départ à chaud, VQ. Elle peut réutiliser les sources de `MS` et de la note N10-D, lues en entier et identifiants résolus ;
- un contradicteur à l'aveugle (Codex par défaut) qui ne reçoit ni la note ni ce plan ;
- au moins deux tours de veille datés, sur des sources différentes ;
- les trois tests : saturation, recoupement, couverture.

Corrections à porter dans `MS` avant tout tag :
- analyse appariée (§2.3) ;
- espace de graines neuf ;
- métrique d'efficacité ;
- comparateurs ré-entraînés (§2.1) ;
- déclaration de non-aveuglement (troisième rupture) ;
- statut des 6 décisions de `MS:224-233`, déjà déclarées non aveugles (`MS:288`).

**Objet A et verrou.** Lu à la lettre, le verrou s'applique à « toute campagne lourde (GPU…) » (`RM:14`). Or A est une suite de runs de développement, exploratoires, décidés le 30/10 dans `MS` G2-G6, qui n'entrent dans aucun jeu confirmatoire (`MS:346`). Son statut est une décision du PI (Q2). Pour lancer A, il faut :
- un court amendement daté de G2 : k = 3, graines, `nocomm_rep`, contrôles du §1.3, taux d'erreur écrits ;
- l'allocation `gate_dev` relevée par un commit séparé, selon G6 (`MS:350`). Le relèvement G7 (`eng` 19, `nocomm` 6) ne suffit pas pour 6 runs `nocomm` de plus si G7 en a consommé.

**Ordre recommandé si A, B et N10-D sont envisagés :**
1. **A+ d'abord** (S3-S5, KXKM, environ 19 h GPU). C'est le moins cher, et il tranche la condition G3 dans le harnais N10-C.
2. **Si A échoue**, application de G3 : N10-C-bis n'est pas gelé, N10-D est le seul candidat, N10-C est rapporté sous R3. Le verrou N10-D continue selon `RM:31`.
3. **Si A passe**, choix du PI à M1 entre B' et N10-D. Un seul des deux est gelé avant M2 (`RM:174`). Critère proposé : prendre N10-D si son verrou est passé à M1, car c'est la priorité de la roadmap et il est nécessaire par construction. Prendre B' si le verrou N10-D glisse au-delà de S7 et si A passe avec D̄ ≥ 0,20 et au moins 2 graines sur 3. L'autre étude part après M3.

## 5. Décisions PI (QCM)

**Q1. Que prépare-t-on sous le nom « N10-C-bis 3 graines » ?**
- (a) **Recommandé :** la porte A+ (G2 à 3 graines, `nocomm_rep` inclus), puis B' seulement si A passe et si le PI le choisit à M1.
- (b) Directement B' (sans porte), ce qui exige de lever G3 par une décision datée.
- (c) Rien : abandon de N10-C-bis au profit de N10-D corrigé (D7 b).

**Q2. La porte A est-elle soumise au verrou de lecture ?**
- (a) **Recommandé :** non. Ce sont des runs de développement exploratoires déjà prévus par G2-G6. Lancement en S3-S5 sur KXKM, après un amendement daté de G2.
- (b) Oui : A attend la fin du verrou N10-D (M1), ce qui décale toute décision B à S7.
- (c) Oui, et A passe son propre verrou.

**Q3. Quelles graines de développement pour A ?**
- (a) **Recommandé :** 9104, 9105 et une graine neuve (préentraînée sur KXKM), fixées avant tout run et distinctes des graines du pilote G7 (9101-9103), que les fichiers G7 soient scellés ou non.
- (b) 9101-9103, si le PI atteste qu'aucun résultat G7 n'a été produit ni lu.
- (c) 9101 (G2 par défaut), 9102, 9103.

**Q4. Quel critère pour la porte à k = 3 ?**
- (a) **Recommandé :** règle G2 inchangée (D̄ ≥ 0,20) avec porte d'apprentissage d'abord, contrôle `eng` et contrôle de bruit `nocomm_rep` (§1.3), et signes rapportés sans test. Sous σ_D = 0,142 supposé : type I ≈ 0,7 %, faux échec ≈ 11 % à D = 0,30.
- (b) Composite : D̄ ≥ 0,20 et au moins 2 graines sur 3 avec D_s ≥ 0,20 (≈ 0,5 % et ≈ 17 %). C'est un durcissement du seuil après lecture, donc à déclarer.
- (c) « Au moins 1 graine sur 3 au-dessus du P99 du nul apparié » (≈ 3 % et ≈ 20 %, permissif).

**Q5. Hôte de A ?**
- (a) **Recommandé :** KXKM seul (G5 tel quel, environ 19 h, un run à la fois).
- (b) Mixte, en bloc par graine, avec déviation de substrat déclarée (environ 12 h de mur, pause du second cerveau).
- (c) macM1 seul (environ 33 h, déviation de substrat déclarée, pause du second cerveau).

**Coût total estimé (recommandation) :** A+ = 9 runs ≈ 18,6 h GPU KXKM et ≈ 8 h PI. Si A passe et si B' est choisi : + 18 runs ≈ 194 h KXKM (≈ 8,1 j) et ≈ 30 h PI. B complet (54 runs, ≈ 583 h) ne tient pas avant M2.
