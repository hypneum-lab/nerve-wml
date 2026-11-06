# Rapport du contradicteur — N10-D — 6 novembre 2026

**Modèle :** GPT-6, agent Codex ; identifiant précis du déploiement non exposé.  
**Date et durée :** début le 06-11-2026 à 07:09:10 CET ; rapport finalisé à 07:28:40 CET ; environ 19.5 minutes de travail écoulées.  
**Instantané :** `nerve-wml`, commit `6e682cc0bb67b14d0854d2eb10242a04e0607d12`, HEAD détaché.  
**Aveugle :** aucune conclusion de l’équipe sur N10-D/NH-96 ne m’est parvenue. Je n’ai ouvert ni note de lecture, ni journal de veille, ni brainstorm, ni rapport d’analyse du labo, ni catalogue, ni `docs/research/`, ni PR/issue interdite, ni autre branche. Le protocole autorisé contient l’étiquette « VARIANT (expected result, Lowe 2019) » et `nh96-null.md` une citation du catalogue ; ces mentions, annoncées dans `snapshot-exclusions.md`, sont des éléments du protocole examiné, pas des conclusions empiriques reçues.

**Lecture intégrale locale :** les trois documents `docs/prereg/n10d/`, les cinq modules centraux `spec.py`, `null.py`, `sigma_seed.py`, `interfaces.py`, `backends.py`, les deux fichiers de tests N10-D, `AGENTS.md`, le manifeste d’exclusions et `tests/conftest.py`. Le document N10-C-bis a été consulté pour son contexte initial et sa section « Necessity gate », lignes 320–350 ; je ne revendique pas une lecture exhaustive de ses autres annexes.  
**Littérature :** PDF/HTML intégraux ouverts et passages argumentaires examinés pour R1–R11 ci-dessous ; je distingue cette consultation des textes intégraux d’une vérification exhaustive de toutes les preuves et annexes. R12 repose uniquement sur le résumé français de la notice HAL : **[résumé moteur / notice primaire HAL]**. Les pistes OpenReview restées derrière un contrôle d’accès sont également **[résumé moteur, non vérifié]** et ne fondent aucun verdict.  
**Calculs :** uniquement tests existants du squelette, simulations du nul fourni et contre-exemples synthétiques en mémoire ; aucun entraînement, installation, téléchargement sur disque ou résultat empirique N10-D. Le seul fichier écrit est ce rapport. Le statut Git final contient exactement les six suppressions préexistantes annoncées dans le manifeste d’exclusions, sans modification supplémentaire.

## 1. Trouvailles qui rendent la question inutile ou déjà tranchée

### 1.1 MAJEUR — la nécessité informationnelle idéale est déjà déterminée

**Fait local.** `protocol.md`, §3, et `spec.py`, lignes 33–40, posent une cible uniforme parmi quatre candidats, sans autre renseignement pour l’auditeur asymétrique. L’émetteur ingénié fournit un code idéal et fixe, avec seize symboles pour seize cibles. La mesure primaire porte toutefois sur des politiques entraînées, pas sur des politiques optimales.

**Démonstration indépendante.** Notons \(X\) la cible, \(S\) les candidats et \(q(a\mid S)\) une politique sans information supplémentaire, choisissant toujours un candidat valide. Sous l’hypothèse annoncée \(P(X=x\mid S)=1/4\),

\[
P(a=X\mid S)=\sum_{x\in S}\tfrac14q(x\mid S)=\tfrac14.
\]

C’est une espérance exacte, pas un plafond sur une fréquence observée. Un message injectif `m = identité(X)`, suivi d’un décodeur qui choisit le candidat correspondant, atteint 1. Dans la jumelle symétrique, sélectionner directement la cible atteint déjà 1 sans message. Les valeurs optimales sont donc :

| Condition | Sans message | Code idéal et décodeur optimal | Contraste optimal |
|---|---:|---:|---:|
| Asymétrique | 0,25 | 1 | 0,75 |
| Symétrique | 1 | 1 | 0 |

**Inférence.** Aucun nouveau calcul RL n’est nécessaire pour décider si l’information privée peut être utile dans ce jeu. En revanche, cette démonstration ne fixe pas `eng − nocomm` après un budget fini de PPO : le décodeur ingénié n’est pas automatiquement un auditeur optimal. La question empirique restante concerne l’apprentissage et le raccordement du canal. Le protocole reconnaît déjà cette portée limitée, §1 ; il serait injustifié de présenter sa réussite comme une nouvelle théorie de la communication émergente.

**Précision de capacité.** Seize symboles permettent un code d’identité en un symbole. Cela ne signifie pas que quatre symboles suffiraient à une communication sans erreur : si le locuteur ne voit que la cible et que toutes les paires de cibles peuvent coexister parmi les candidats, deux cibles partageant un symbole deviennent indiscernables dans certains contextes. Le budget et la connaissance du contexte doivent être distingués.

### 1.2 MAJEUR — antériorités directes et contre-exemples à une lecture trop forte

**Fait bibliographique.** R1, Lazaridou, Peysakhovich et Baroni, décrit déjà le jeu à récompense commune où seul le locuteur connaît la désignation de la cible ; §4 et tableau 1 rapportent une coordination de 99–100 % dans leurs configurations. Ce n’est ni PPO, ni exactement K=16/C=4/A=16. [Texte intégral](https://arxiv.org/pdf/1612.07182).

**Inférence.** Le principe « l’information privée peut soutenir une communication apprise » a une antériorité directe ; ces résultats ne prédisent pas les scores du tronc N10-C.

**Fait bibliographique.** R2, Lowe et al., §6.2, p. 700, envisage explicitement le canal comme voie d’optimisation alternative. Il distingue influence causale, corrélation des messages et gain de récompense. [Texte intégral](https://ifaamas.org/Proceedings/aamas2019/pdfs/p693.pdf).

**Inférence.** Un gain avec message dans la condition symétrique n’établirait pas que la cible était cachée. Une information redondante peut être exploitée par une politique sous-optimale.

**Fait bibliographique.** R3, Chauhan et Bera, tableau 2, rapporte sur MuJoCo un IQM de 1,000 pour l’émetteur ingénié contre 0,482 pour l’émetteur appris en couplage partiel, à débit égal ; §5 précise que la condition avec canal sensoriel ouvert n’est pas mesurée. [HTML intégral, v2](https://arxiv.org/html/2609.34373v2).

**Inférence.** Cette antériorité est proche de la justification de campagne et sépare possibilité informationnelle et découverte du protocole. Elle ne tranche pas le contraste apparié asymétrique/symétrique de N10-D. Elle figure déjà dans le contexte N10-C-bis autorisé : je ne la présente pas comme une découverte absente de tout le labo.

### 1.3 Implémentation existante : base réutilisable, pas résultat N10-D déjà disponible

**Fait vérifié.** Le dépôt public EGG, au commit exact annoncé `54eb11c6b4ba3d3171811378f7ad7a117fa93a97`, contient déjà `egg/zoo/signal_game/`. Son README décrit un message à un symbole, des candidats mélangés et une récompense commune. `train.py:get_game` utilise REINFORCE ou Gumbel-Softmax ; `features.py:_BatchIterator.get_batch` fournit les candidats au locuteur et les permute pour l’auditeur. [README](https://github.com/facebookresearch/EGG/blob/54eb11c6b4ba3d3171811378f7ad7a117fa93a97/egg/zoo/signal_game/README.md), [code](https://github.com/facebookresearch/EGG/blob/54eb11c6b4ba3d3171811378f7ad7a117fa93a97/egg/zoo/signal_game/train.py), [échantillonnage](https://github.com/facebookresearch/EGG/blob/54eb11c6b4ba3d3171811378f7ad7a117fa93a97/egg/zoo/signal_game/features.py).

**Inférence.** Une grande partie du jeu existe déjà. Ce code n’est cependant pas le tronc PPO adapté, ni la paire de conditions et de contrôles prévue. Je n’ai trouvé aucun dépôt de résultats donnant directement les deux contrastes N10-D sans nouvelle exécution. L’absence constatée est limitée aux recherches documentées en §4, pas une preuve d’inexistence mondiale.

## 2. Failles du dessin prévu

### B1 — BLOQUANT : la porte d’apprentissage est impossible sur `sym`

**Fait.** `protocol.md`, lignes 64–69, annonce les contrôles à côté de chaque résultat et impose `eng > blind ceiling + P99`. `blind_ceiling(sym) = 1`. Avec `sigma_seed=0`, le code produit exactement `P99=0` : il faudrait donc `eng > 1`. Avec un P99 positif, l’exigence est encore plus forte. Aucune fréquence de succès ne peut franchir cette porte.

**Inférence.** La lecture littérale rend impossible le cas A confirmant simultanément les deux conditions. Le contrôle négatif symétrique n’a pas non plus à « tomber » à un niveau inférieur : son plafond sans message est déjà 1.

**Condition nécessaire avant gel.** Préciser la portée des portes par condition et réconcilier §3 avec §5. Si la porte était implicitement réservée à `asym`, cette intention doit être écrite ; je ne la décide pas pour le PI.

### B2 — BLOQUANT : le nul simulé n’est pas calibré pour l’hypothèse et la mesure revendiquées

**Fait.** `nh96-null.md`, lignes 12–25, pose les deux bras au plafond sans information et simule `p = clip(B0 + Normal(0, sigma_seed))`, puis une fréquence binomiale. `null.py`, lignes 27–43, tire les deux effets de bras indépendamment. L’écart-type provient uniquement de `nocomm/asym`.

**Démonstration indépendante.** Sous l’uniformité conditionnelle exacte du §1.1, toute politique asymétrique figée sans information, choisissant un candidat, a la même probabilité de succès 0,25. Un effet latent gaussien positif la place pourtant au-dessus de cette borne. Si ces écarts latents représentent une autre distribution des tâches ou une fuite, le modèle du jeu a changé. Ils ne sont pas justifiés par le seul hasard binomial déjà simulé. Cet ajout de variance peut être conservateur dans le modèle binomial idéal ; cette incohérence ne prouve pas à elle seule un excès de faux positifs en asymétrie.

**Démonstration indépendante.** Dans `sym`, voir la cible rend le succès 1 atteignable, mais n’oblige pas une politique entraînée à l’atteindre. Le nul « information inutilisable » autorise donc des performances inférieures à 1 et différentes entre bras. Il n’est pas le modèle « deux auditeurs presque parfaits ». De plus, le clipping à 1 transforme la gaussienne : à `sigma=0,05`, la moyenne latente vaut, à une correction négligeable près due à la borne 0, \(1-0,05/\sqrt{2\pi}=0,98005\).

**Contrôle numérique sur le code fourni.** Pour `k=8`, `n_eval=500`, `sigma=0,05`, `N=10 000`, graine du simulateur `20261101`, les quantiles exacts de cette exécution sont consignés en §4. La table de `nh96-null.md` est explicitement calculée en `asym` ; son P99 d’environ 0,06 ne peut pas être transféré à `sym` sans recalcul. Le recalcul donne P99=0,06250 en `asym` contre 0,03500 en `sym`. Cette différence de frontière est secondaire face au défaut d’hypothèse.

**Inférence.** Les p-valeurs de Monte-Carlo sont valides relativement au générateur choisi, pas automatiquement relativement aux entraînements. La réserve du document — il ne couvre pas les dynamiques d’apprentissage — est honnête, mais interdit précisément de l’utiliser comme justification suffisante d’un verdict sur ces dynamiques. Une résolution de p-valeur de `1/10001` n’est pas une garantie de contrôle de l’erreur de type I.

### M1 — MAJEUR : variance de `nocomm` et puissance du test des signes ne dimensionnent pas D

**Fait.** `sigma_seed.py`, lignes 28–49, estime une variance sur huit graines `nocomm/asym`, puis dimensionne un test normal de moyenne avec `sigma_D² = 2(sigma_seed² + 0,25/n_eval)`. Pour `n_eval=500`, le code retourne cinq graines pour `sigma_seed` égal à 0, 0,02, 0,05 ou 0,15 ; il en retourne 18 pour 0,30. L’analyse annoncée est pourtant un test des signes, accompagné d’un bootstrap de moyenne.

**Démonstration indépendante.** Avec cinq différences non nulles et un test des signes unilatéral exact à 5 %, seules cinq différences positives permettent de rejeter : `p=1/32`. Pour des graines indépendantes, si la probabilité d’un signe positif est q, sa puissance est donc q⁵ :

| q | Puissance à cinq graines |
|---:|---:|
| 0,80 | 32,8 % |
| 0,90 | 59,0 % |
| 0,95 | 77,4 % |

Atteindre 90 % exige ici q ≥ 0,97915, pas seulement une différence moyenne de 0,30.

**Contre-exemple synthétique, compatible avec les bornes du jeu.** Supposons `S_nocomm=0,25` pour chaque graine et `S_eng=0,6875` avec probabilité 0,8, sinon 0. Alors `E[D_i]=0,30`, l’écart-type de D_i vaut 0,275, et la variance latente de `nocomm` est nulle. La procédure retient cinq graines ; la puissance du test des signes, hors bruit d’évaluation, est seulement 0,32768. Il s’agit d’un modèle construit, pas de données N10-D.

**Fait mathématique.** `Var(D_i) = Var(S_eng,i) + Var(S_nocomm,i) − 2 Cov(S_eng,i, S_nocomm,i)`. Une estimation du seul second terme ne détermine pas cette variance. Huit graines et un budget réduit ajoutent une incertitude d’estimation que la formule ponctuelle ne propage pas.

**Inférence.** Le pilote aveugle choisi ne permet pas d’estimer la variabilité du décodeur entraîné ni celle de la différence. Augmenter `n_eval` ne résout pas cette absence. En outre, le test des signes cible la probabilité d’un signe positif ; sans hypothèses supplémentaires, il ne teste pas la moyenne primaire D. Le bootstrap de cinq graines peut également manquer des modes de convergence absents de l’échantillon.

### M2 — MAJEUR : « derniers cinq logs d’entraînement » n’est pas une binomiale d’évaluation

**Fait.** `protocol.md`, lignes 59–61, reprend la moyenne des cinq derniers logs de succès d’entraînement. Le contexte G2, ligne 337, définit chaque log sur les cent derniers épisodes terminés. `null.py` simule au contraire une seule fréquence binomiale par bras et graine ; `sigma_seed.py` lui soustrait une variance binomiale.

**Démonstration indépendante.** Cinq fenêtres indépendantes de cent épisodes d’une politique figée ont la variance d’une fréquence sur cinq cents épisodes. Cinq fenêtres identiques ont celle d’une fréquence sur cent épisodes. Le chevauchement introduit des covariances ; l’évolution de la politique ajoute de la non-stationnarité. Le nombre de logs ne détermine donc pas un `n_eval` effectif.

**Inférence.** La revendication « same pipeline » n’est pas établie. Le budget réduit et la fréquence de journalisation peuvent changer à la fois la variance et la quantité mesurée. TBD-11 doit fixer comment ces observations sont produites et comment leur dépendance est prise en compte, sans assimiler automatiquement `n_eval=500` à cinq logs de cent épisodes.

### M3 — MAJEUR : redondance, nullité du contraste et équivalence sont confondues

**Fait.** `nh96-null.md`, ligne 60, déduit `D_sym=0` de la seule possibilité d’atteindre 1 sans message. `protocol.md`, lignes 91–92, interprète un D_sym positif comme une absence de symétrie ; l’autre document admet aussi un canal utilisé quand il est redondant.

**Contre-exemple indépendant.** Dans un jeu parfaitement symétrique et sans fuite, un auditeur peut sélectionner la cible pour huit identités sur seize et choisir au hasard parmi quatre candidats pour les huit autres. Son succès est 0,625. Un auditeur lisant parfaitement le code ingénié atteint 1. Le contraste vaut 0,375, alors que les deux voient bien la cible. Ce sont des politiques valides ; je n’affirme pas que PPO les produira.

**Inférence.** La symétrie des observations n’entraîne pas l’égalité des performances à budget fini. Un D_sym positif peut être un effet de représentation ou d’optimisation, pas une fuite. Inversement, une même récompense ne prouve pas que le message n’influence pas l’action.

**Fait.** La règle `D_sym <= P99` accepte aussi des effets négatifs importants. Le seuil varie avec la variance pilote et le nombre de graines. Le cas A utilise un seuil ponctuel de D_asym ≥ 0,30, tandis que le test de puissance vise un rejet de zéro ; aucune règle complète ne précise comment l’intervalle intervient pour soutenir un effet d’au moins 0,30.

**Inférence.** Un quantile de bruit n’est pas une marge d’équivalence scientifique. Il faut préciser si la cible est « pas de gain positif », « effet pratiquement nul dans les deux sens » ou « deux politiques proches de leur optimum ». R8 fournit le cadre statistique pertinent ; je ne fixe aucune nouvelle marge à la place du PI.

### M4 — MAJEUR : permutation, audit de fuite et seuils portent sur des objets encore insuffisamment définis

**Fait.** `interfaces.py`, lignes 33–35, ne définit qu’une permutation fixe d’épisodes. Il n’existe pas encore de générateur réel, de transformateur de messages ni de procédure d’audit dans le squelette. `protocol.md` réutilise un P99 de différence moyenne entre bras pour des portes « au-dessus du plafond » sans définir partout l’unité d’agrégation.

**Inférence.** Il reste à garantir que le message permuté est indépendant de la cible actuelle conditionnellement aux entrées de l’auditeur, et que le mécanisme ne change ni la longueur, ni le timing, ni la distribution marginale des messages. Une permutation fixe sur une collection réutilisée peut créer une relation apprenable ; une permutation d’indices ne garantit pas, à elle seule, l’indépendance.

**Précision indépendante.** Deux épisodes distincts peuvent partager la même cible. Sous des cibles uniformes indépendantes, une coïncidence d’identité a probabilité 1/16 et n’est pas une fuite. Interdire les mêmes identités après permutation introduirait au contraire une information d’exclusion. Il faut distinguer dérangement des épisodes et exclusion des étiquettes.

**Inférence.** Le P99 d’une différence entre deux moyennes n’est pas le quantile d’une seule moyenne moins 0,25, encore moins celui d’un score individuel. Les contrôles de fuite doivent utiliser leur propre statistique et leur famille de tests. Une fréquence `nocomm` supérieure à 0,25 peut aussi être une fluctuation ; aucun audit fini ne démontre l’absence de tout canal latéral.

### Évaluation de TBD-10 et TBD-11, sans les trancher

**TBD-10 — MAJEUR pour la définition, proposition recevable sous conditions.** Un verdict par backend, sans mise en commun des graines, évite de sélectionner le backend favorable. Exiger deux preuves positives concordantes constitue une conjonction ; cela n’impose pas mécaniquement une correction de Bonferroni par deux pour cette seule conclusion globale. En revanche, deux non-rejets ne prouvent pas un nul commun. Il faut définir des statuts cohérents — effet, équivalence, non-mesurable, invalidité, incertitude — et la logique de leurs combinaisons. Deux implémentations partageant les mêmes erreurs de dessin ne constituent pas deux validations indépendantes du raisonnement.

**TBD-11 — MAJEUR pour la comparabilité.** EGG fournit des jeux et des estimateurs REINFORCE/GS, pas automatiquement le PPO demandé. Le fichier public JaxMARL `simple_reference.py`, au commit `31b0a2309d753f9315058c5572f2b210d5eb5db0` du 20 juillet 2026, impose trois landmarks, un canal de dimension dix et une récompense mêlant terme global et individuel avec `local_ratio=0,5` par défaut. Ce n’est pas le jeu K=16/C=4 à récompense identique pour les deux rôles. [Code vérifié](https://github.com/bold-lab-ai/JaxMARL/blob/31b0a2309d753f9315058c5572f2b210d5eb5db0/jaxmarl/environments/mpe/simple_reference.py).

**Inférence.** L’étiquette « simple_reference-like » ne suffit pas à fixer l’environnement. Cette inspection ne prétend pas auditer la future version épinglée JaxMARL 0.2.0. Avant comparaison, il faut rendre explicites : observations acteur/critique, ordre émission–réception–action, distribution et ordre des candidats, nombre de symboles par message, zéro/EOS, traitement des gradients, réinitialisation récurrente, récompense, mises à jour PPO, et unité de budget. `Sequence[int]` sans longueur maximale ne fixe pas un débit avec le seul alphabet A=16. Le budget du pilote et son `n_eval` restent à décider par le PI ; ni le budget ni le nombre de graines ne peuvent être certifiés tant que l’estimand et le test ne correspondent pas.

### Éléments OK et défaut secondaire d’implémentation

**OK.** Séparation développement/confirmation, namespaces de graines, absence de mélange des campagnes et backends, contrôles scriptés, refus de transformer après coup un échec en « architecture bloquée » : ces dispositions sont utiles. Les backends qui lèvent `NotImplementedError` et les dépendances non hashées sont annoncés comme limites L0 ; ce ne sont pas des découvertes cachées de cet audit.

**MINEUR.** Le CLI de `sigma_seed.py`, lignes 53–56, prend `n_eval` dans `recs[0]` plutôt que dans les lignes filtrées. Préfixer huit lignes valides et constantes `nocomm/asym` par une ligne d’un autre bras ayant `n_eval=1` laisse sigma à zéro, mais fait passer k de 5 à 48. Le test `test_other_arms_ignored` ne teste que l’extracteur, pas ce chemin CLI. Ce défaut peut être corrigé avant gel ; il ne résout pas M1.

## 3. Antériorités à intégrer, avec identifiants résolus

« Manquante » signifie ici absente de l’argumentation des documents N10-D consultés, pas absente d’un catalogue que je n’ai pas lu. Lowe est déjà nommé ; Chauhan–Bera figure déjà dans le contexte N10-C-bis. Les citations littérales ci-dessous sont courtes et accompagnées d’un emplacement exact. Les implications proposées sont mes inférences.

| Réf. et résolution au 06-11-2026 | Passage exact du texte consulté | Ce que cela change |
|---|---|---|
| **R1. Lazaridou, Peysakhovich & Baroni, Multi-Agent Cooperation and the Emergence of (Natural) Language**, ICLR 2017 ; [arXiv:1612.07182](https://arxiv.org/abs/1612.07182), GET neutre HTTP 200. PDF complet ouvert. | §4, p. 4 : « agents almost perfectly coordinate » ; tableau 1, p. 5. | **MAJEUR.** Antériorité directe du jeu coopératif à cible privée ; les configurations ne sont pas celles de N10-D. |
| **R2. Lowe et al., On the Pitfalls of Measuring Emergent Communication**, AAMAS 2019 ; [arXiv:1903.05168](https://arxiv.org/abs/1903.05168), GET neutre HTTP 200. PDF complet ouvert. | §6.2, p. 700 : « communication may act as an alternate pathway for optimization ». | **MAJEUR.** Empêche d’identifier redondance informationnelle et égalité des performances apprises. |
| **R3. Chauhan & Bera, Emergence, Not Bandwidth: Physical Coupling and the Limits of Learned Multi-Agent Communication**, prépublication 2026 ; [arXiv:2609.34373v2](https://arxiv.org/abs/2609.34373v2), identifiant de base résolu HTTP 200 ; HTML v2 complet ouvert. | §5 : « The open-channel condition is unmeasured. » ; résultats tableau 2, §4.1. | **MAJEUR.** Précédent proche sur couplage et émetteur ingénié ; sa limite interdit de lui attribuer le contraste exact des jumelles N10-D. Statut de publication évaluée non vérifié. |
| **R4. Zheyuan Zhang, A Combinatorial Approach to Neural Emergent Communication**, COLING 2025, pp. 1660–1666 ; [arXiv:2410.18806](https://arxiv.org/abs/2410.18806), HTTP 200. [PDF complet](https://aclanthology.org/2025.coling-main.112.pdf) ouvert. | §3, p. 1661 : « we only need one symbol for classification » ; §3.1 traite l’échantillonnage. | **MAJEUR pour la portée.** Un codebook d’identités peut suffire ; succès ne signifie pas structure linguistique. Je n’utilise pas ses approximations combinatoires pour prouver les bornes exactes de N10-D. |
| **R5. Kharitonov, Chaabouni, Bouchacourt & Baroni, EGG: a toolkit for research on Emergence of lanGuage in Games**, EMNLP-IJCNLP 2019, pp. 55–60 ; [DOI](https://doi.org/10.18653/v1/D19-3010), Crossref HTTP 200 ; [arXiv:1907.00852](https://arxiv.org/abs/1907.00852), HTTP 200. [PDF complet](https://aclanthology.org/D19-3010.pdf) ouvert. | §2.1, note 3, p. 56 : « single-symbol communication is typically simpler to train ». | **MAJEUR pour l’adaptation.** Jeux disponibles et choix d’estimateur à distinguer du PPO commun demandé. |
| **R6. Hofbauer & Huttegger, Feasibility of communication in binary signaling games**, Journal of Theoretical Biology 254 (2008), 843–849 ; [DOI](https://doi.org/10.1016/j.jtbi.2008.07.010), Crossref HTTP 200. [PDF complet](https://langev.com/pdf/hofbauer08signalGameJTB.pdf) ouvert. | §1, p. 843 : « the sender and the receiver may fail to communicate » ; p. 844 analyse les attracteurs partiellement informatifs. | **MAJEUR.** Contre-exemple fondateur hors ML : intérêt commun et possibilité de transmission n’assurent pas la convergence. Ce ne sont pas des théorèmes sur PPO. |
| **R7. Eccles, Bachrach, Lever, Lazaridou & Graepel, Biases for Emergent Communication in Multi-agent Reinforcement Learning**, NeurIPS 2019 ; [arXiv:1912.05676](https://arxiv.org/abs/1912.05676), HTTP 200. [PDF complet](https://proceedings.neurips.cc/paper/9470-biases-for-emergent-communication-in-multi-agent-reinforcement-learning.pdf) ouvert. | Résumé, p. 1 : « a difficult joint exploration problem » ; §4.1, p. 6, compare auditeurs optimaux avec et sans communication. | **MAJEUR.** Prévoit déjà la dissociation entre valeur disponible et apprentissage. Ne justifie pas un sauvetage post hoc des échecs N10-D. |
| **R8. Daniël Lakens, Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses**, Social Psychological and Personality Science 8 (2017), 355–362 ; [DOI](https://doi.org/10.1177/1948550617697177), Crossref HTTP 200. [PDF complet](https://journals.sagepub.com/doi/pdf/10.1177/1948550617697177) ouvert. | P. 355 : « the absence of an effect can be rejected, but not statistically supported » ; « Testing for Equivalence », p. 356. | **MAJEUR.** Un non-dépassement de P99 ne remplace pas une procédure d’équivalence et ses bornes explicites. |
| **R9. Bouchacourt & Baroni, How agents see things: On visual representations in an emergent language game**, EMNLP 2018, 981–985 ; [DOI](https://doi.org/10.18653/v1/D18-1119), Crossref HTTP 200 ; [arXiv:1808.10696](https://arxiv.org/abs/1808.10696), HTTP 200. [PDF complet](https://aclanthology.org/D18-1119.pdf) ouvert. | P. 981 et §3, p. 983 : « pseudo-images generated from random noise ». | **MINEUR pour l’objectif primaire.** Un succès référentiel ne démontre pas une sémantique naturelle ; le protocole a raison de limiter sa portée. |
| **R10. Główka, Zubek & Rączaszek-Leonardi, Context-dependent communication under environmental constraints**, Cognitive Systems Research 88 (2024), 101293 ; [DOI](https://doi.org/10.1016/j.cogsys.2024.101293), Crossref HTTP 200 ; [arXiv:2305.05821](https://arxiv.org/abs/2305.05821), HTTP 200. PDF complet de la version août 2024 ouvert. | Résumé, p. 1 : « signals may be ambiguous out of context » ; §3, expérience 2. | **MAJEUR pour la spécification.** Contraintes et distribution des candidats changent l’information exploitable ; il faut les figer, pas seulement K et C. |
| **R11. Yamamoto, Morita, Higashinaka & Takeuchi, Dissociating communicative success from representational alignment in common ground formation**, Frontiers in Computer Science 8 (2026), 1873726 ; [DOI](https://doi.org/10.3389/fcomp.2026.1873726), Crossref HTTP 200. [HTML complet](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1873726/full) ouvert. | §4.1 : « symbolic alignment and representational alignment do not necessarily improve in the same direction ». | **MINEUR.** Descendant récent de R1/R2, contre-exemple à des extrapolations sur l’alignement interne ; ne tranche pas la nécessité au sens de N10-D. |
| **R12. Maxime Toquebiau, Apprentissage par renforcement profond pour les systèmes multi-agents appliqué à la robotique mobile**, thèse Sorbonne Université, soutenue le 13-12-2024, NNT `2024SORUS385` ; [notice HAL](https://theses.hal.science/tel-05076852v1), API HAL HTTP 200 ; [DOI](https://doi.org/10.70675/3385d0bbzc66cz4fceza9e0z8696ff61ff4e), résolu par OpenAlex HTTP 200. **[résumé moteur / notice primaire HAL]**. | Résumé français, sans pagination : « À la place, nous proposons d'enseigner une langue prédéfinie aux agents. » | **MINEUR / piste.** La comparaison langage prédéfini/appris existe dans les thèses françaises. Le contenu détaillé n’a pas été vérifié ; aucun résultat numérique de cette thèse n’est utilisé. |

### Remontée et descente des citations

**R2, Lowe 2019.** Remontée vérifiée dans sa bibliographie, p. 701, référence [23], vers R1, dont le PDF a ensuite été ouvert ; prolongement ancien hors ML par R6. Descente par l’API OpenAlex, racine `W2922221969`, filtre `cites` sur 2024-01-01–2026-11-06 : 16 notices retournées. Parmi elles, R11 et la thèse R12 ont été examinés aux niveaux précisés ci-dessus. La bibliographie de R11 cite explicitement Lowe. Aucun ne contient le contraste PPO exact recherché.

**R1, Lazaridou 2017.** Remontée de sa formulation de jeu vers la tradition des jeux de signalisation, examinée ici à travers R6 ; je ne prétends pas avoir lu le livre de Lewis. Descente OpenAlex, racine `W2564324149`, mêmes bornes : 38 notices, dont R10 et R11, ouverts en texte intégral. R9, plus ancien, examine directement les jeux de R1 ; son PDF a été ouvert. Leurs différences de tâche et d’objectif sont conservées dans le tableau.

**Complément R4 et R3.** La bibliographie de R4 renvoie à R1 et R5, consultés. Les requêtes OpenAlex `cites:W4404307480` et `cites:W7214744394` avec les mêmes dates retournent zéro notice. Ce zéro décrit l’index interrogé, pas l’absence de citations réelles : la notice arXiv de R4 n’y expose même pas ses références. R3 cite R2 et R7, consultés. Les limites d’indexation empêchent de présenter cette descente comme exhaustive.

## 4. Vérifications, recherche négative et limites

### Résultats reproductibles sur le squelette

Les neuf tests ciblés passent, sous Python 3.14.8 / pytest 9.0.2, sans installation. Commande exécutée dans `nerve-wml/`, avec écriture des bytecodes et du cache pytest désactivée :

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -B -m pytest -q -p no:cacheprovider \
  tests/test_n10d_skeleton.py tests/test_n10d_sigma.py
```

Ces tests vérifient des constantes, des propriétés internes du simulateur, le filtrage de l’extracteur et la présence de stubs. Ils ne testent ni l’uniformité conditionnelle d’un environnement réel, ni le taux d’erreur du verdict, ni sa puissance.

Les simulations ont été lancées par imports avec `python3 -B`, sans fichiers intermédiaires :

```python
summarize(simulate_null(TaskSpec(condition), 8, 500, 0.05,
                        n_draws=10_000, rng_seed=20261101))
```

| Condition | Médiane | P95 | P99 |
|---|---:|---:|---:|
| Asymétrique | −0,00025 | 0,04450 | 0,06250 |
| Symétrique | 0,00000 | 0,02425 | 0,03500 |

Ces tirages sont ceux du simulateur L0, pas des scores PPO.

Le résultat `sym`, `sigma=0`, est exactement une masse en zéro, quelle que soit la graine du simulateur. Les valeurs de puissance q⁵ et le contre-exemple M1 sont des calculs analytiques indépendants des données. L’exécution supplémentaire de 2 000 tirages à `k=5` donne un P99 de 0,0764 en `asym` et 0,0468 en `sym` pour sigma=0,05 ; ce sont des illustrations du générateur, pas une calibration validée des entraînements.

### Canaux et requêtes exactes

Toutes les recherches suivantes ont été effectuées le **6 novembre 2026**, pendant cette revue. La fenêtre récente demandée est 2024-01-01–2026-11-06 ; les recherches fondatrices sont volontairement non bornées. Les recherches web ne garantissent pas à elles seules un filtrage chronologique exhaustif. Les documents de la campagne et leurs conclusions ont été exclus de la recherche externe.

**arXiv et découverte web associée :**

```text
emergent communication referential game asymmetric symmetric observations necessity Lowe pitfalls measuring communication 2019
site.arxiv.org 2024 2025 referential game communication redundancy shared observations
site.arxiv.org "referential game" "2025" "information"
"On the Pitfalls of Measuring Emergent Communication" Lowe
"referential game" "engineered" "2026"
"referential game" "symmetric" "asymmetric" communication
"Emergence, Not Bandwidth" Chauhan Bera
"communication" "necessity" "engineered" "referential"
"communication" "K=16" "C=4"
"Biases for Emergent Communication in Multi-agent Reinforcement Learning"
"Context-dependent communication under environmental constraints"
"Dissociating communicative success from representational alignment"
```

L’API `https://export.arxiv.org/api/query` a été interrogée avec :

```text
all:"referential game" AND (all:"asymmetric" OR all:"redundant" OR all:"information") AND submittedDate:[202401010000 TO 202611062359]
start=0, max_results=12, sortBy=relevance
```

Cette requête a expiré après 40 secondes : **non vérifié**, et non « zéro résultat ». Les identifiants arXiv cités ont tous été résolus séparément par `https://arxiv.org/abs/<id>`, User-Agent neutre `IndependentLiteratureReview/1.0`, HTTP 200, avec au moins trois secondes entre appels arXiv. Deux pistes d’identifiants retournant des titres sans rapport ont été écartées ; aucun identifiant erroné n’a été retenu.

**OpenReview, recherches web :**

```text
site.openreview.net communication necessity referential game information asymmetry
site.openreview.net "A Combinatorial Approach to Neural Emergent Communication"
site.openreview.net "communication" "redundant" "2024"
site.openreview.net "referential" "PPO" communication
site.openreview.net "communication" "necessity" "referential"
site.openreview.net "Multi-Agent Cooperation" "Language"
site.openreview.net "information asymmetry" "communication" "2025"
site.openreview.net "communication" "redundant" "2026"
```

La piste *Complexity-Limited Multi-Task Training for Compositional Emergent Communication* (`HTfNizaxFw`) a été interrogée via `api2.openreview.net/notes?content.title=...`, puis par l’ancienne API `api.openreview.net/notes?id=HTfNizaxFw` : HTTP 403. Le forum et le lien PDF redirigent vers une vérification du navigateur. La piste *AsymPuzl* (`SXcJh8hoLz`) reçoit également HTTP 403 via `api2.openreview.net/notes?id=SXcJh8hoLz`. **[résumé moteur, non vérifié]** pour ces pistes ; je ne leur attribue aucun résultat établi.

**Crossref, requêtes bibliographiques via API `/works` :**

```text
Lewis signaling games communication common interest
How do communication systems emerge
Some dynamics of signaling games
equivalence tests absence evidence Lakens
A Combinatorial Approach to Neural Emergent Communication
referential game information asymmetry communication redundancy
```

La dernière requête utilise `filter=from-pub-date:2024-01-01,until-pub-date:2026-11-06`, `rows=6` ; aucune des six réponses n’apporte l’étude exacte recherchée. Les recherches par mots-clés *Some dynamics...* et *equivalence tests...* ont reçu HTTP 429 : **non vérifié à ce stade**. Les références finalement retenues ont ensuite été résolues individuellement par les URLs Crossref `/works/<doi>` listées dans le tableau, avec temporisation de deux à trois secondes. La thèse R12 est résolue par OpenAlex plutôt que par Crossref.

**HAL, theses.fr et hors ML :**

```text
site.theses.fr communication émergente jeux référentiels information asymétrique
site.hal.science "jeux de signal" communication Lewis
site.hal.science "communication émergente"
site.hal.science "communication émergente" "jeu"
site.theses.fr "communication émergente" agents
"communication" "jeu de Lewis" thèse
Mathieu Rita thèse communication émergente 2023
"thèse" "jeux référentiels"
"thèse" "jeux de signalisation" "Lewis"
"communication émergente" "theses.hal.science"
"Lewis signaling" "no communication" "common interest" learning
site.biorxiv.org "signaling game" communication
```

API HAL : `https://api.archives-ouvertes.fr/search/`, `q="communication émergente"`, `rows=8`, `wt=json` : cinq notices, dont R12. Une requête `q=docid:5076852` a retourné titre, auteur, résumé français/anglais, NNT, DOI et URL du document. La biologie mathématique est effectivement couverte par le PDF de R6, et la psychologie/statistique par R8. La piste bioRxiv n’a pas fourni de résultat direct retenu ; elle ne constitue pas un dépouillement exhaustif de bioRxiv.

**Code, données et vérification d’antériorités :**

```text
"EGG: a toolkit for research on Emergence of lanGuage in Games"
"Multi-Agent Cooperation and the Emergence of (Natural) Language" code
"How agents see things" communication noise 2018
"Beyond latent codes" "Decodable reference interfaces"
"A Combinatorial Approach to Neural Emergent Communication" -site:aclanthology.org -site:arxiv.org -site:researchgate.net
"JaxMARL" "simple_reference" github
site.zenodo.org "referential game" communication
site.osf.io "signaling game" communication
site.huggingface.co/datasets "referential" "communication"
site.github.com "referential game" "engineered"
```

Le résultat Hugging Face `ZinengTang/PersReFex` décrit des dialogues spatiaux multi-perspectives ; sa fiche a été examinée au niveau du résultat moteur, sans audit du corpus. Aucun fichier de données de ce corpus n’a été téléchargé et aucun résultat N10-D ne lui est attribué. Les recherches Zenodo/OSF ne fournissent pas de réponse exacte retenue. Le connecteur GitHub a servi à lire les fichiers publics EGG et JaxMARL cités, jamais les issues ou PR du labo.

**OpenAlex et Semantic Scholar.** Une seule requête batch Semantic Scholar, pour les trois racines bibliographiques R1/R2/R4 avec références et citations, a reçu HTTP 429, sans `Retry-After` : **non vérifié** ; pas de boucle de nouvelles requêtes. Repli sur OpenAlex par titres et DOI, puis filtres `cites` des racines indiquées en §3. Requêtes par titres : `On the Pitfalls of Measuring Emergent Communication`, `A Combinatorial Approach to Neural Emergent Communication`, `Emergence Not Bandwidth Physical Coupling`, `Multi-Agent Cooperation and the Emergence of Natural Language`, `Some dynamics of signaling games`. Les résultats autres que les titres exacts n’ont pas été traités comme des correspondances.

### Ce que je n’ai pas pu vérifier

- **Résultats N10-D :** aucun n’est fourni ; les backends, l’émetteur ingénié, l’auditeur scripté et l’adaptation PPO ne sont pas implémentés dans cet instantané. Ni fuite effective, ni échec d’optimisation réel, ni performance future n’est donc affirmé.
- **Dépendances :** compatibilité et disponibilité des wheels, fermeture transitive et hashes non vérifiés. Le tableau de versions est une intention, pas un environnement exécutable validé.
- **OpenReview :** accès API et textes bloqués comme détaillé ci-dessus. Je ne déduis pas une absence de résultats d’un HTTP 403.
- **Thèse R12 :** PDF non obtenu par l’outil web, malgré une notice et un DOI résolus. La citation française provient du résumé de la notice, sans page inventée.
- **Autres pistes :** *How do communication systems emerge?*, DOI `10.1098/rspb.2011.2181`, et *Some dynamics of signaling games*, DOI `10.1073/pnas.1400838111`, sont résolus HTTP 200 par Crossref, mais les pages PMC ouvertes ont présenté un contrôle anti-robot. Leur contenu intégral n’est pas utilisé comme preuve. R6 fournit la source hors ML effectivement consultée.
- **Piste récente du graphe de citations :** *Beyond latent codes: Decodable reference interfaces for robust multi-agent communication*, DOI `10.1016/j.neucom.2026.135272`, résolu Crossref HTTP 200, mais texte ScienceDirect HTTP 403. OpenAlex indique une disponibilité en octobre 2026 et Crossref un numéro de janvier 2027 ; date exacte de première publication en ligne et contenu non vérifiés. Aucun résultat scientifique ne lui est attribué.
- **Espacement des accès arXiv :** les GET directs de résolution ont été espacés d’au moins trois secondes ; l’espacement interne des ouvertures PDF/HTML par l’outil web n’est pas auditable ici. La garantie temporelle porte donc sur les résolutions directes, pas sur toutes les consultations par l’outil web.
- **Exhaustivité :** index de citations incomplets, requête arXiv de recherche expirée, accès bloqués et dépouillement non systématique de toutes les thèses/bioRxiv. Le constat négatif porte sur l’étude exacte cherchée, pas sur toute la littérature possible.
- **Portée des calculs :** les preuves conditionnelles et contre-exemples supposent le générateur annoncé. Ils ne certifient pas un générateur absent. Les neuf tests passants ne remplacent pas une étude de calibration du verdict.

## 5. Verdict

**Partiellement : la nécessité informationnelle idéale est déjà tranchée, tandis que l’utilité comme validation du PPO et du canal reste ouverte ; les contradictions de porte d’apprentissage et de calibration empêchent toutefois le dessin actuel de fournir le verdict annoncé.**
