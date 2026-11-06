# Rapport du contradicteur aveugle — N10-D (2026-11-06) — passe Claude

## Provenance

- **Modèle :** Claude Opus 5.5 (agent Claude Code, `claude-opus-5-5`), passe aveugle exécutée en parallèle de la passe Codex.
- **Date :** 2026-11-06, ~07:03 → ~07:19 CET (≈ 16 min, instantané compris).
- **Instantané lu :** `hypneum-lab/nerve-wml` @ `6e682cc0bb67b14d0854d2eb10242a04e0607d12` (branche de la PR #100, HEAD détaché, autres refs et remote supprimés ; exclusions dans `snapshot-exclusions.md`).
- **Conclusions du labo parvenues par accident :** aucune conclusion sur des données N10-D (aucun run N10-D n'existe dans l'instantané). Deux éléments d'attente figurent **dans le protocole lui-même** et ont donc été vus : l'étiquette « NH-96 is labelled VARIANT (expected result, Lowe 2019) » (`protocol.md` §1) et la phrase source du catalogue citée en tête de `nh96-null.md`. Je ne les ai pas utilisées comme preuve ; la référence Lowe 2019 a été relue indépendamment. J'ai aussi vu, dans le doc N10-C d'origine, les résultats de la porte B3 de N10-C-bis (campagne précédente, pas N10-D).
- **Lu en entier (texte intégral extrait du PDF) :** `protocol.md`, `nh96-null.md`, `requirements-to-hash.md`, `experiments/n10d/*.py` (tous), noms des tests `tests/test_n10d_*.py`, section G2 de `docs/milestones/2026-10-25-n10c-bis-control-arms.md` ; Lowe et al. 2019 (arXiv:1903.05168) ; Eccles et al. 2019 (arXiv:1912.05676) ; Lazaridou et al. 2017 (arXiv:1612.07182) ; Rita et al. 2022 (arXiv:2209.15342), lu en entier pour §1-3, parcouru ensuite ; Lowe et al. 2017 MADDPG (arXiv:1706.02275), §5-6 lus ; JaxMARL (arXiv:2311.10090), annexes MPE parcourues. README `egg/zoo/basic_games` d'EGG au commit `54eb11c6` lu en entier.
- **[résumé moteur] seulement :** tout le reste (abstracts OpenAlex/Crossref/HAL, titres OpenReview).
- **Vérification des identifiants :** l'API arXiv `export.arxiv.org` a renvoyé **429** (« Rate exceeded ») dès le premier appel, et Semantic Scholar batch aussi (**429**). Les 19 identifiants arXiv ont donc été résolus via **DataCite** (`api.datacite.org/dois/10.48550/arXiv.<id>`, enregistrement officiel des DOI arXiv, 19/19 trouvés, titres et auteurs concordants), et les PDF lus via `arxiv.org/pdf/<id>` (4 s entre appels). Les DOI non-arXiv ont été résolus par Crossref (`api.crossref.org/works/<doi>`) ou OpenAlex. Le commit EGG a été vérifié par l'API GitHub.

## 1. Trouvailles qui rendent la question inutile ou tranchée

### 1.1 BLOQUANT (sur la moitié « nécessité ») — le contraste primaire est fixé par construction, donc calculable sans entraînement

Faits.
- Le bras primaire `eng` est un émetteur ingénié à code idéal fixe (`protocol.md` §3). Le bras `nocomm` met le canal à zéro. Dans `asym`, la docstring de `blind_ceiling()` (`spec.py`) démontre que **tout** auditeur sans message plafonne à 1/C, car la cible est uniforme parmi les C candidats et indépendante de ce qu'il voit.
- Avec un code idéal, l'auditeur optimal (au sens de Bayes) atteint 1. La valeur de population du contraste est donc **1 − 1/C = 0,75** dans `asym` et **0** dans `sym`, où le plafond aveugle vaut 1,0. `nh96-null.md` §5 l'admet pour `sym` (« 0 by construction of the task »).
- Eccles et al. 2019 (arXiv:1912.05676, §4.1, « Summing MNIST ») définissent exactement cette quantité : « The reward the listener can gain from using the communication channel is Pl(πs) = R(πlc, πs) − R(πlnc, πs) ». Ils la **calculent explicitement**, sans entraîner l'auditeur : « calculating the optimal policies explicitly ».
- Rita et al. 2022 (arXiv:2209.15342, §3, décomposition de la perte de Lewis) écrivent qu'avec un locuteur donné, le côté auditeur « merely corresponds to the optimization of its supervised task ». Dans leurs termes, avec un code idéal la perte d'information est nulle et il ne reste que la perte de co-adaptation. Cette perte est purement supervisée quand le locuteur est figé, ce qui est le cas de `eng`.

Inférence.
- La réponse à « le message est-il nécessaire quand l'auditeur ne peut réussir sans l'information du locuteur ? » est analytique (borne 1/C contre 1). Elle ne dépend pas des données.
- Le seul contenu empirique du contraste primaire est d'ordre technique : le tronc PPO de N10-C et la tuyauterie du canal apprennent-ils une table de correspondance supervisée à C = 4 voies sur K = 16 ? Le protocole l'annonce lui-même (« a test of the task and of the channel plumbing, not of emergent-language theory »). La formulation « necessity confirmed on this task » de l'issue A (§5) dit donc plus que ce que le dessin peut établir.
- Je classe ce point BLOQUANT pour la formulation scientifique (« nécessité »), pas pour l'utilité de la campagne comme test de tuyauterie.

### 1.2 MAJEUR — la moitié « utilisé » (bras `learned`, secondaire) a des réponses publiées dans les deux sens

Faits.
- Lazaridou et al. 2017 (arXiv:1612.07182, §4, Table 1) : jeu référentiel cible + 1 distracteur, locuteur et auditeur appris par REINFORCE, sans partage de poids. Résultat : « communication success » de 99 à 100 % pour les 8 configurations (« agents almost perfectly coordinate »).
- Eccles et al. 2019 (Table 1) : jeu en un coup à information privée (somme de chiffres MNIST), apprenants indépendants REINFORCE. Résultat : **0/50** « good runs » sans biais (IC 0,00-0,07) ; « the final reward of 0.1 is exactly what can be achieved in this environment with no communication ».
- Lowe et al. 2017 (arXiv:1706.02275, §6, « Cooperative communication ») : sur la tâche locuteur-auditeur MPE, DQN, Actor-Critic, TRPO et DDPG échouent tous (« the listener learns to ignore the speaker and simply moves to the middle of all observed landmarks »), alors que MADDPG réussit.

Inférence.
- Les résultats publiés sur le bras appris dépendent de l'algorithme et de la tâche. Le jeu référentiel discriminatif simple (le dessin N10-D, proche de Lazaridou 2017 et de la zoo EGG) est le cas où l'apprentissage conjoint réussit.
- La question n'est pas tranchée pour un tronc PPO donné, mais le résultat attendu est largement documenté. Le bras `learned` est secondaire et n'a pas de seuil pré-enregistré. Rien ne protège donc contre une lecture a posteriori.

### 1.3 MINEUR — des implémentations prêtes donnent le même test sans nouveau dessin

Faits.
- EGG @ `54eb11c6` (vérifié par l'API GitHub : « Merge pull request #267 », 2025-05-06) contient `egg/zoo/basic_games` avec un « discrimination game » (« Receiver gets multiple input items (the same target and a number of distractors, in random order) »), en modes `rf`/`gs`.
- JaxMARL (arXiv:2311.10090) contient `jaxmarl/environments/mpe/simple_speaker_listener.py` et `simple_reference.py` (vérifié par l'API GitHub). Ses annexes donnent des courbes IQL/VDN/QMIX sur « Simple Speaker Listener V4 ».

Inférence. Ces implémentations ne répondent pas à la question pour le tronc N10-C. Elles fournissent cependant des repères externes de faisabilité, et le protocole pourrait les citer comme témoins positifs.

### Requêtes documentées (absence d'antériorité « même question, même tronc »)

- OpenAlex (`api.openalex.org/works?search=…&filter=from_publication_date:2024-01-01`, 2026-11-06) : « emergent communication necessity ablation referential game » (508 résultats) ; « emergent communication causal influence listening measurement » ; « learning to communicate privileged information partner reinforcement learning ablation no communication » ; « Lewis signaling game reinforcement learning PPO » ; « emergent communication fails independent learners speaker listener ». Seuls les 10 premiers résultats par requête ont été examinés [résumé moteur]. Aucun ne pose la question N10-D avec un tronc PPO hérité d'une tâche Procgen.
- OpenReview (`api2.openreview.net/notes/search`, 2026-11-06) : « emergent communication ablation no communication baseline », « communication necessity referential game », « positive listening emergent communication », « Lewis game reinforcement learning listener ». Rien qui tranche ; retrouvé Eccles 2019 (NeurIPS 2019, forum `l0S7QgBMjM`).
- HAL (`api.archives-ouvertes.fr/search`, 2026-11-06) : « "communication émergente" » (5), « "emergent communication" » (19), « "jeux de signalisation" » (1, hors sujet), « "signaling games" apprentissage » (1). Thèses repérées [résumé moteur] : Chaabouni 2021 (tel-03536320), Toquebiau 2024 (tel-05076852), Karch 2023 (tel-04146927). Aucune ne contredit le point 1.1.
- arXiv API : non interrogeable (429), compensé par DataCite et le téléchargement direct des PDF.

## 2. Failles du dessin prévu

### 2.1 BLOQUANT (tel que rédigé, correction triviale) — le contrôle 3 rend `sym` impossible à interpréter, donc l'issue A inatteignable

Faits.
- Contrôle 3 (`protocol.md` §3) : « `eng` above `blind ceiling + P99 of the null` before D is interpreted ».
- En `sym`, le plafond aveugle vaut 1,0. `eng` ne peut pas dépasser 1,0 + P99.
- L'issue A exige « D_sym <= P99 of its calibrated null », donc une lecture de D_sym.

Inférence. Lu à la lettre, le contrôle 3 n'est jamais satisfait en `sym`. D_sym n'est jamais interprétable et l'issue A ne peut pas être atteinte. Il faut restreindre explicitement le contrôle 3 à `asym` et donner à `sym` sa propre porte, par exemple `nocomm_sym` ≥ 1 − P99.

### 2.2 MAJEUR — `sigma_seed` est mesuré sur un bras dont la variance inter-graines est nulle par construction

Faits.
- `sigma_seed.py` ne lit que `nocomm`/`asym`. D'après 1.1, toute politique aveugle a une espérance de succès exactement 1/C : la variance inter-graines de ce bras est **structurellement nulle**, il ne reste que le bruit binomial.
- J'ai simulé 2 000 pilotes (8 graines, n_eval = 500, p = 0,25, σ réel = 0) avec l'estimateur de `sigma_seed()`. L'estimateur renvoie **exactement 0 dans 57 %** des cas, et son P95 vaut 0,019.
- `required_seeds()` renvoie k = 5 (le plancher) pour tout σ ≤ 0,1 (n_eval = 500 comme 50), k = 8 à σ = 0,2 et k = 18 à σ = 0,3. Valeurs calculées avec le code de l'instantané.

Inférence.
- Le pilote ne peut à peu près rien apprendre. Il fixera k = 5 presque à coup sûr.
- La variance qui compte pour D est celle de `eng`, c'est-à-dire la fraction de graines où l'auditeur PPO n'apprend pas : distribution bimodale de D par graine, 0,75 ou environ 0. Ce n'est pas mesuré.
- Exemple : avec 20 % de graines en échec, l'écart-type de D par graine est d'environ 0,3. La formule donnerait k ≈ 18, mais le pilote donnera k = 5.
- Le nul de NH-96 et le calcul de puissance sont donc calibrés sur une quantité qui ne peut pas varier. C'est le cas d'un « nul non calibré » pour la source de variance pertinente. Mesurer le pilote aussi sur `eng`/`asym` reste compatible avec l'aveuglement sur D, si seul l'écart-type de `eng` est extrait.

### 2.3 MAJEUR — le nul de `sym` est dégénéré

Faits. `simulate_null(TaskSpec("sym",16,4), 8, 500, σ)` donne **P99 = 0,000** à σ = 0, 0,0145 à σ = 0,02 et 0,035 à σ = 0,05. Avec p0 = 1,0 et l'écrêtage `clip`, le bruit binomial disparaît.

Inférence.
- Compte tenu de 2.2, σ estimé vaudra 0 dans plus de la moitié des pilotes. Le seuil `D_sym ≤ P99` devient alors `D_sym ≤ 0`.
- Le moindre écart positif déclenchera l'issue D (« the symmetric control is not symmetric »), par exemple `nocomm_sym` à 0,99 sous l'entropie résiduelle de PPO contre `eng_sym` à 1,0.
- De plus, le σ utilisé pour `sym` vient d'un bras `asym`. Rien ne justifie de transporter cette variance.

### 2.4 MAJEUR — critères de décision incohérents entre `protocol.md` et `nh96-null.md`, et test de puissance mal ciblé

Faits.
- `protocol.md` §4 : test des signes unilatéral apparié de D > 0, plus bootstrap. La « non-rejection » y est « non concluant ».
- `protocol.md` §5 : issues A/B/C définies sur la **valeur ponctuelle** de D_asym (≥ 0,30 ; 0,10-0,30 ; < 0,10).
- `nh96-null.md` §4 : NH-96 est **rejetée** si la **borne supérieure** de l'IC bootstrap à 95 % est < 0,30.
- `required_seeds` dimensionne la puissance pour détecter D > 0 quand le vrai D vaut 0,30 (docstring : « One-sided paired test of D > 0 … detecting D = MARGIN »).

Inférence.
- Les mêmes données peuvent classer un résultat en issue B (« weaker than predicted ») et en même temps en NH-96 « rejetée ». Exemple : D = 0,20 avec un IC supérieur à 0,28.
- La puissance porte sur H0 : D = 0, alors que la décision porte sur le seuil 0,30. Si le vrai D vaut 0,30, la probabilité que D ponctuel soit ≥ 0,30 est d'environ 50 %.
- Le contraste attendu (0,75, cf. 1.1) rend ce problème probablement sans effet en pratique. Mais c'est une faille de principe dans une préinscription.

### 2.5 MAJEUR — k = 5 rend le bootstrap percentile et le test des signes fragiles

Faits.
- Avec k = 5, le test des signes unilatéral ne rejette à α = 0,05 que si les 5 différences sont positives (p = 1/32).
- Le bootstrap percentile de la moyenne de 5 valeurs n'a que 126 rééchantillons distincts. Sa couverture est connue pour être trop optimiste à n très petit (propriété générale, non citée ici).

Inférence. Si une graine sur cinq échoue (D ≈ 0, signe aléatoire), le test confirmatoire échoue avec une probabilité non négligeable. Avec 20 % d'échecs, P(5/5 positifs) vaut environ 0,9⁵ ≈ 0,59. Cela tient même si la moyenne passe largement 0,30.

### 2.6 MAJEUR — l'inférence de retour vers N10-C n'est pas soutenue par le dessin

Faits.
- §2 : l'échec de la porte N10-C « is read as "the Procgen task does not need the message", which N10-D tests on a task where it must ».
- §7 : N10-D tourne sur JaxMARL et EGG, pas sur le harnais Procgen. Le tronc est « adapted to each backend » et TBD-11 (adaptation exacte) est ouvert.

Inférence.
- Un succès N10-D montre que le tronc adapté apprend une table supervisée dans un autre harnais. Il ne montre pas que la tuyauterie du canal N10-C fonctionnait.
- Il ne sépare donc pas l'explication « la tâche Procgen n'exige pas le message » de l'explication « le canal ou le tronc N10-C ne transmettait pas ».
- Il faudrait au moins un bras N10-D exécuté dans le harnais N10-C, avec le même code de canal.

### 2.7 MINEUR — divergences entre la métrique et le modèle de bruit

Faits.
- La métrique primaire est la moyenne des 5 derniers logs de la courbe d'entraînement, chacun sur les 100 derniers épisodes (définition G2 de N10-C-bis). Le nul simule Binomial(n_eval = 500)/n_eval.

Inférence.
- Si les fenêtres de log se recouvrent, la taille effective est inférieure à 500 et le nul est trop étroit.
- La politique PPO d'entraînement est stochastique : avec un bonus d'entropie, `eng` plafonne sous 1, ce qui alimente 2.3.
- `n_eval` n'est pas défini pour une métrique d'entraînement. TBD-11 doit le fixer.

### 2.8 MINEUR — contrôles `msg_shuffle` et audit de fuite

Faits et inférences.
- `msg_shuffle` entraîné sur messages permutés équivaut à `nocomm`, ce qui est tautologique et reconnu (`nh96-null.md` §5).
- Évalué avec un auditeur entraîné sur `eng`, il donne en espérance 1/K + (1 − C/K)·q, où q est le succès de repli sur un message hors candidats. Cela fait 0,25 seulement si q = 1/C. Le protocole ne dit pas laquelle des deux variantes est utilisée.
- L'audit de fuite (contrôle 4) est utile. Le risque concret est la position de la cible : la zoo EGG lit cette position dans le fichier de données et le README exige un « random order ». Il faut vérifier le générateur et ne pas seulement constater la fuite a posteriori.
- Un critique centralisé (MAPPO) qui voit la cible ne fuit pas à l'exécution. Il faut le préciser dans le manifeste.

### 2.9 MINEUR — marge 0,30 non reliée à l'oracle

Fait. L'oracle vaut 0,75 et la marge 0,30. Inférence : un auditeur à 55 % de succès passerait l'issue A. Un seuil exprimé en fraction de l'oracle, par exemple ≥ 0,9 × (1 − 1/C), serait plus informatif sur la « tuyauterie ».

## 3. Antériorités manquantes (identifiants résolus)

Le protocole ne cite que Lowe 2019. Références à ajouter :

| réf. | identifiant (résolu) | apport pour N10-D |
|---|---|---|
| Lowe, Foerster, Boureau, Pineau, Dauphin 2019, « On the Pitfalls of Measuring Emergent Communication » | arXiv:1903.05168 (DataCite OK) | §6.2 : un gain de récompense avec canal est « a sufficient indicator » de communication. L'ablation à l'évaluation (« setting the messages to 0 ») est trompeuse ; il recommande CIC. `nocomm` est un bras entraîné, donc conforme, mais `msg_shuffle` à l'évaluation tombe sous cette mise en garde. Lu en entier. |
| Eccles et al. 2019, « Biases for Emergent Communication in MARL » | arXiv:1912.05676 (DataCite OK ; NeurIPS 2019, OpenReview `l0S7QgBMjM`) | Pl calculé explicitement (= contraste primaire) ; 0/50 sans biais pour apprenants indépendants. Lu en entier. |
| Lazaridou, Peysakhovich, Baroni 2017 | arXiv:1612.07182 (DataCite OK) | Jeu référentiel REINFORCE : 99-100 % de succès. Témoin positif externe du bras `learned`. Lu en entier. |
| Rita, Tallec, Michel et al. 2022, « Emergent Communication: Generalization and Overfitting in Lewis Games » | arXiv:2209.15342 (DataCite OK ; HAL hal-03831908) | Décomposition information/co-adaptation : avec un locuteur figé, l'auditeur résout une tâche supervisée. Lu en entier pour §1-3. |
| Lowe et al. 2017, MADDPG | arXiv:1706.02275 (DataCite OK) | Échec de DQN/AC/TRPO/DDPG sur locuteur-auditeur MPE (l'auditeur ignore le locuteur). §5-6 lus. |
| Foerster et al. 2016, RIAL/DIAL | arXiv:1605.06676 (DataCite OK) | Antériorité du problème « apprendre à communiquer » en deep MARL. [résumé moteur] |
| Havrylov & Titov 2017 | arXiv:1705.11192 (DataCite OK) | Jeu référentiel à séquences, REINFORCE contre Gumbel. [résumé moteur] |
| Kottur et al. 2017 | arXiv:1706.08502 (DataCite OK) | Protocoles appris non « naturels ». [résumé moteur] |
| Jaques et al. 2018/2019, « Social Influence as Intrinsic Motivation » | arXiv:1810.08647 (DataCite OK) | Origine de l'influence causale. [résumé moteur] |
| Kharitonov et al. 2019, EGG | arXiv:1907.00852 (DataCite OK) | Boîte à outils choisie (backend). [résumé moteur] |
| Rutherford et al. 2023, JaxMARL | arXiv:2311.10090 (DataCite OK) | Backend ; MPE speaker-listener déjà implémenté. Annexes parcourues. |
| Yu et al. 2021, MAPPO | arXiv:2103.01955 (DataCite OK) | Comportement de PPO en coopératif multi-agent (tronc). [résumé moteur] |
| Lazaridou & Baroni 2020 (survey) | arXiv:2006.02419 (DataCite OK) | Synthèse du domaine. [résumé moteur] |
| Boldt & Mortensen 2024 (review) ; Peters et al. 2024/2025 (survey, JAAMAS) | arXiv:2407.03302 ; arXiv:2409.02645 = doi:10.1007/s10458-025-09691-y (OpenAlex OK) | Synthèses récentes (période ≥ 2024). [résumé moteur] |
| **Hors ML** : Argiento, Pemantle, Skyrms, Volkov 2009, « Learning to signal: Analysis of a micro-level reinforcement model » | doi:10.1016/j.spa.2008.02.014 (Crossref OK ; Stoch. Proc. Appl.) | Probabilités : convergence d'un apprentissage par renforcement vers un système de signaux dans le jeu de Lewis. [résumé moteur] |
| **Hors ML** : Huttegger, Skyrms, Tarrès, Wagner 2014, « Some dynamics of signaling games » | doi:10.1073/pnas.1400838111 (Crossref OK ; PNAS) | Dynamiques d'apprentissage et équilibres de « pooling » partiel dans les jeux de Lewis à plus de 2 états, pertinent pour K = 16 et le bras `learned`. [résumé moteur] |
| **Hors ML** : Pawlowitsch 2008, « Why evolution does not always lead to an optimal signaling system » | doi:10.1016/j.geb.2007.08.009 (Crossref OK ; Games Econ. Behav.) | Équilibres sous-optimaux des jeux de signaux. [résumé moteur] |
| **Physique** : Baronchelli, Felici, Caglioti et al. (dépôt 2005, trois premiers auteurs selon DataCite), « Sharp transition towards shared vocabularies in multi-agent systems » | arXiv:physics/0509075 (DataCite OK) | Naming game en physique statistique. Contexte seulement, ne tranche pas N10-D. [résumé moteur] |

## 4. Ce que je n'ai pas pu vérifier

- L'API arXiv (`export.arxiv.org`) et Semantic Scholar batch ont renvoyé **429**. Les identifiants ont été résolus par DataCite, mais je n'ai pas fait de recherche plein texte arXiv par mots-clés. La veille arXiv ≥ 2024 repose sur OpenAlex (qui indexe arXiv) et OpenReview.
- Le PDF HAL de Rita 2022 a renvoyé une page HTML (anti-robot). Je l'ai lu via arXiv.
- Les thèses HAL (Chaabouni 2021, Toquebiau 2024, Karch 2023) n'ont pas été ouvertes [résumé moteur]. theses.fr n'a pas été interrogé séparément.
- bioRxiv, Zenodo, OSF et Hugging Face n'ont pas été interrogés (temps).
- Le comportement réel du tronc PPO N10-C sur un jeu en un coup (taux d'échec par graine, effet de l'entropie) est inconnu. Les chiffres de 2.2 et 2.5 sont des scénarios illustratifs, pas des mesures.
- Je n'ai pas lu le contenu des tests `tests/test_n10d_*.py`, seulement leurs noms. Je n'ai pas exécuté la suite de tests ; j'ai seulement exécuté `null.py` et `sigma_seed.py` en lecture.
- Hors vérification par design : la phrase du catalogue et le fichier `.github/.../priorites-hypotheses.md` cités par le protocole (interdits).

## 5. Verdict

**Partiellement.** La moitié « nécessité » de la question est tranchée par construction : borne aveugle exacte 1/C contre oracle 1, ce qui correspond au Pl d'Eccles 2019, calculable sans entraînement. Il ne reste d'ouvert et d'utile qu'un test de tuyauterie et d'optimisation : le tronc PPO adapté apprend-il cette table, et le bras `learned` émerge-t-il ? Ce test vaut la peine d'être fait, mais avec trois corrections :
- recalibrer `sigma_seed` et le nul `sym` sur des bras qui varient (2.2, 2.3) ;
- corriger le contrôle 3 pour `sym` (2.1) et unifier les critères de décision (2.4) ;
- ne pas en tirer de conclusion sur N10-C sans un bras exécuté dans le harnais N10-C (2.6).

Décompte : BLOQUANT 2 (1.1, 2.1) ; MAJEUR 6 (1.2, 2.2, 2.3, 2.4, 2.5, 2.6) ; MINEUR 4 (1.3, 2.7, 2.8, 2.9).
