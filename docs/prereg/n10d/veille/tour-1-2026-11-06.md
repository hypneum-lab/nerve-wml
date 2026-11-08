# Verrou de lecture N10-D — tour de veille 1 (2026-11-06)

- **Date** : 2026-11-06, requêtes passées entre 16:50 et 17:40 CET.
- **Tour** : n° 1 du verrou après contradicteur (PR #113). Les deux tours de
  `veille-journal.md` (01/11) ont alimenté la note v0.3 ; ils ne sont pas
  recomptés ici. Dédoublonnage : 773 identifiants arXiv déjà vus (note,
  brainstorm, `snowball-tour1.tsv`, `snowball-tour2.tsv`).
- **Qui** : agent Claude (Opus 5.5) pour le PI, sans calcul local hors appels
  réseau et édition de texte. Ni la note de lecture ni le protocole n'ont été
  modifiés.
- **Canaux** (périmètre du brief) : arXiv, OpenReview, bioRxiv/PsyArXiv,
  GitHub, Hugging Face, Semantic Scholar/OpenAlex.
- **Fenêtre** : publications 2024-01-01 → 2026-11-06 pour les requêtes ; les
  références des contradicteurs sont résolues quelle que soit leur année.
- **Sources lues avant de chercher** : `master:docs/prereg/n10d/reading-note.md`
  (v0.3), `prereg/n10d-skeleton-amend-2026-11-06:docs/prereg/n10d/protocol.md`
  §8 (A1-A20), `docs/n10d-contradictor-2026-11-06:docs/n10d/contradictor-2026-11-06/SYNTHESE.md`.
- **Statut de lecture** : aucun PDF n'a été ouvert à ce tour. Tout ce qui suit
  est `[résumé moteur]` (résumé arXiv, OpenReview ou OpenAlex), ou `[titre
  seul]` quand l'API n'a renvoyé qu'un titre. Rien ici ne vaut lecture
  intégrale.

## 1. Requêtes et comptes brut → gardé

| canal | requête exacte | brut | gardé | pourquoi |
|---|---|---:|---:|---|
| arXiv (`export.arxiv.org`, HTTPS, 3 s entre appels ; HTTP 301 en HTTP clair, refait en HTTPS) | `abs:"emergent communication" AND abs:PPO` | 2 | 0 | télécoms UAV |
| arXiv | `abs:"emergent communication" AND abs:"referential game"` | 20 | 2 | 2410.18806, 2611.00006 |
| arXiv | `abs:"emergent communication" AND abs:"no-communication"` | 294 | 3 | requête trop large (racine « emergen- » : communications d'urgence) ; 2601.22041, 2605.11695, 2606.10582 |
| arXiv | `abs:"emergent communication" AND abs:"vocabulary size"` | 1 | 0 | hors sujet |
| arXiv | `abs:"emergent communication" AND abs:causal` | 6 | 0 | déjà vus ou hors sujet |
| arXiv | `abs:"emergent communication" AND abs:seeds` | 3 | 0 | 2609.34373 déjà dans la note |
| arXiv | `abs:"emergent communication" AND abs:speaker AND abs:listener` | 10 | 0 | rien de 2024+ pertinent |
| arXiv | `abs:"Lewis signaling game" OR abs:"Lewis signalling game"` | 11 | 0 | 2410.18806 en double |
| arXiv | `abs:"emergent communication" AND abs:ablation` | 5 | 0 | 2605.11695 en double |
| arXiv | `abs:"emergent language" AND abs:"reinforcement learning" AND abs:listener` | 3 | 0 | antérieurs à 2024 |
| arXiv | `abs:"emergent communication" AND abs:permutation` | 0 | 0 | — |
| arXiv | `abs:"signaling game" AND abs:babbling` | 0 | 0 | — |
| **arXiv, total** | 12 requêtes | **355** | **5** | |
| OpenReview (`api2.openreview.net/notes/search`, `source=forum`, limit 100) | `emergent communication PPO` | 100 | — | |
| OpenReview | `emergent communication referential game seeds` | 100 | — | |
| OpenReview | `speaker listener emergent communication ablation` | 100 | — | |
| OpenReview | `emergent communication no communication baseline` | 100 | — | |
| OpenReview | `Lewis signaling game reinforcement learning` | 100 | — | |
| OpenReview | les 5 ci-dessus, union | 500 → 306 forums → 160 de 2024+ | 4 | CQ5ivZEB4e, uyIn9jZ70C, 8L3khbpUJL, AiEfgKQh2P |
| OpenReview | `emergent communication`, `venueid=ICLR.cc/2027/Conference/Submission` | 100 | 1 | dF1DqBmdgX (OAmBy25HHg = Chauhan & Bera, déjà vu) |
| **OpenReview, total** | 6 requêtes | **600** | **5** | `notes?id=` : HTTP 403 `ChallengeRequiredError` (identifiants résolus par la réponse de `notes/search`) |
| bioRxiv/PsyArXiv via Europe PMC (`SRC:PPR`, 2024-2026) | `"emergent communication"` | 6 | 0 | Research Square (robotique) ; PsyArXiv kbfua (pragmatique, version de PLOS CB) ; km28h (abstraction) |
| Europe PMC | `("signaling game" OR "signalling game") AND "reinforcement learning"` | 0 | 0 | — |
| Europe PMC | `"referential game"` | 1 | 0 | PsyArXiv nwkag (alignement linguistique humain) |
| PsyArXiv (`api.osf.io/v2/preprints`, provider psyarxiv) | `filter[title]=emergent communication` | 1 | 0 | kbfua, en double |
| **bioRxiv/PsyArXiv, total** | 4 requêtes | **8** | **0** | aucun résultat côté cognitif ne porte sur l'usage causal d'un canal appris |
| GitHub (`gh search repos`, `--updated >=2025-01-01`) | `emergent communication` | 65 | 0 | projets d'étudiants, sans article |
| GitHub | `referential game` | 6 | 0 | idem |
| GitHub | `signaling game reinforcement` / `jaxmarl communication` / `EGG emergence language games` | 0 / 0 / 0 | 0 | — |
| GitHub | `lewis game` | 2 | 0 | hors sujet |
| GitHub | `gh api repos/facebookresearch/EGG`, `repos/FLAIROx/JaxMARL` (lecture directe des deux backends du protocole) | 2 | 2 | EGG archivé ; JaxMARL déplacé, v0.2.0 |
| **GitHub, total** | 8 appels | **75** | **2** | |
| Hugging Face (`huggingface.co/api`) | `models?search=emergent-communication`, `…emergent_communication`, `models?search=referential-game`, `datasets?search=emergent-communication`, `datasets?search=emergent language`, `datasets?search=referential-game`, `spaces?search=emergent-communication` | 0 × 7 | 0 | index vide sur ces termes |
| Hugging Face | `papers/search?q=emergent communication referential game` | 120 | 0 | 2024+ : surveys, LLM ; 2602.11583 déjà vu |
| Hugging Face | `datasets?search=elcc` (suite de 2407.04158) | 1 | 1 | `bboldt/elcc` |
| **Hugging Face, total** | 9 appels | **121** | **1** | |
| Semantic Scholar (`/paper/arXiv:<id>/citations`) | citations de 1912.05676 et 1903.05168 | — | — | **HTTP 429** aux deux appels : rien n'est compté |
| OpenAlex (`filter=cites:<W>,from_publication_date:2024-01-01`) | citants de Lowe 2019 (W2922221969) | 16 | — | |
| OpenAlex | citants d'Eccles 2019 (W2971094937) | 9 | — | |
| **OpenAlex, total** | union | **23** | **5** | Chernyavskiy, Yamamoto, Yao, Tucker, Skowrońska ; Bailey (JAIR 2026) déjà vu au tour 1 de la note |
| **Total** | | **1 182** | **18** | dont 1 en double inter-canal (2410.18806 = CQ5ivZEB4e), soit **17 entrées distinctes** |

Note sur OpenAlex : la couverture est faible (16 citants de Lowe depuis 2024,
contre 170 citants au total chez S2 au tour 1 de la note). Le snowball S2 en
lot reste à refaire (voir § 6).

## 2. Entrées gardées

Format : identifiant résolu — titre, année, premier auteur. Pourquoi gardée ;
phrase de la note touchée. Verdict Hypneum (confirme / infirme / dépasse /
nuance / outil / info) et impact.

### 2.1 Ce qui rendrait la question inutile ou déjà tranchée (a)

1. **arXiv:2410.18806** (= OpenReview CQ5ivZEB4e, COLING 2025) — « A
   Combinatorial Approach to Neural Emergent Communication », 2024, Zhang.
   `[résumé moteur]` Le jeu référentiel de Lewis ne demande en pratique qu'un
   ou deux symboles à cause d'un biais d'échantillonnage des distracteurs ;
   la complexité symbolique minimale se calcule (algorithme SMS).
   → Touche note § 1 et § 5 (K = 16, C = 4, A = 16) : discriminer parmi 4
   candidats demande au plus 2 bits, un symbole de A = 16 en porte 4. **Confirme**
   que la nécessité est tranchée par construction ; question nouvelle : le
   bras `learned` peut réussir avec un code partiel (discriminer sans
   identifier), ce qui pèse sur la lecture du message permuté. Impact : **Haut**.
   Cité par Codex (contradicteur), absent de la note.
2. **doi:10.3390/technologies14070432** — « In Pursuit of the Emergence
   Point: Extracting Phase Transitions in Multi-Agent Communication », 2026,
   Chernyavskiy (*Technologies*). `[résumé moteur]` Modèle EDO de substitution
   des croyances dans un jeu de Lewis ; le départ de la communication depuis
   l'équilibre de babillage uniforme est une bifurcation fourche
   supercritique, à un seuil fixé par la dissipation et la sensibilité.
   → Touche note § 4.4 (récepteur qui n'utilise pas un canal nécessaire) et la
   porte d'apprentissage du protocole (A1) : un seuil analytique sous lequel
   l'apprentissage reste au babillage. **Dépasse** sur le volet théorique (aucune
   source de la note ne donne un seuil). Impact : **Moyen** (modèle de
   substitution, pas PPO).
3. **OpenReview dF1DqBmdgX** — « Revisiting White-Box Communication in
   Multi-Agent Reinforcement Learning », soumission ICLR 2027, auteurs
   masqués. `[résumé moteur]` Des protocoles « boîte blanche » écrits par un LLM
   rivalisent avec la communication apprise sur quoi / quand / à qui
   communiquer. → Touche note § 1 (bras `eng` à code conçu contre `learned`) et
   § 5 (« canal inutilisé » contre « inutilisable ») : un émetteur conçu est
   une référence publiée. **Confirme** l'idée de garder `eng` comme borne.
   Impact : **Moyen**.

### 2.2 Antériorités sur « porte d'apprentissage » et nul apparié par graine (b)

4. **OpenReview AiEfgKQh2P** — « Interpretable Multi-Agent Communication via
   Information Gating », 2025 (atelier CFAgentic, ICML 2025, oral),
   Belogolovsky et al. `[résumé moteur]` La « porte » y est une politique qui
   filtre ce qui passe de l'observation au canal. → Homonymie : ce n'est pas la
   porte d'apprentissage d'A1 (seuil de réussite avant lecture du contraste),
   pas plus que la porte d'IC3Net (note § 2). **Ne touche pas** la question,
   mais signale le risque de confusion de vocabulaire dans la préreg.
   Impact : **Info**.
5. Nul apparié par graine dans les jeux référentiels : **aucune entrée** sur
   les six canaux de ce tour. Requêtes arXiv `abs:permutation` et
   `abs:seeds` : 0 et 3 résultats (dont un déjà vu) ; aucun résumé OpenReview
   ou OpenAlex ne décrit une permutation d'étiquettes de bras à l'intérieur
   d'une graine. → Ne change pas note § 4.1 (absence constatée, qui reste une
   inférence).

### 2.3 Communication émergente avec PPO et petits vocabulaires, 2025-2026 (c)

6. **OpenReview uyIn9jZ70C** — « Adversarially Robust Emergent Communication
   in Cooperative Multi-Agent Reinforcement Learning », 2026 (poster ADScAI),
   Himantha & Peiris. `[résumé moteur]` Locuteur-auditeur sous MAPPO, module
   de communication Transformer, messages **continus** attaqués par PGD
   (ε 0,05 à 0,50) ; entraînement adverse. → Touche note § 5 (interventions sur
   le message d'agents PPO entraînés) : premier cas trouvé de PPO +
   locuteur-auditeur + intervention sur le message en 2026, mais intervention
   adverse (pas permutée, pas de bras `nocomm`, pas de nul par graine).
   **Ne contredit pas** § 4.1. Impact : **Moyen**.
7. **arXiv:2601.22041** — « Learning to Communicate Across Modalities:
   Perceptual Heterogeneity in Multi-Agent Systems », 2026, Pitzer.
   `[résumé moteur]` Jeu de communication binaire en plusieurs étapes entre
   agents de modalités différentes ; perturbations de bits : le sens est
   distribué, pas compositionnel. → Touche note § 3 (intervention sur le
   message pour juger de l'usage) et § 4.2 (hétérogénéité, ici de modalité, pas
   de substrat MLP/LIF). **Nuance** § 4.2. Impact : **Moyen**.
8. **doi:10.1162/opmi_a_00188** — « Towards Human-Like Emergent Communication
   via Utility, Informativeness, and Complexity », 2025, Tucker (*Open Mind*).
   `[résumé moteur]` VQ-VIB : messages discrets sous contrainte d'information.
   → Touche note § 3 ligne « messages discrets appris » : **seconde source de
   nature différente** (revue à comité) à côté de 2107.02367 et 2609.34373.
   Impact : **Info**.

### 2.4 Cross-play et hétérogénéité (lignes de la note § 3 et § 4.2)

9. **arXiv:2606.10582** — « Drawing with Strangers: Population Scaling Drives
   Zero-Shot Mutual Intelligibility in Emergent Sketching », 2026, Kim.
   `[résumé moteur]` Des populations entraînées indépendamment deviennent
   mutuellement intelligibles sans contact quand la taille de population
   croît. → **Nuance** note § 3 « les protocoles indépendants ne sont pas
   mutuellement lisibles » : vrai pour des paires (Chauhan & Bera, Marincat),
   pas forcément pour de grandes populations. N10-D est en paires : ne change
   pas le dessin. Impact : **Moyen**.
10. **OpenReview 8L3khbpUJL** — « Referential communication in heterogeneous
    communities of pre-trained visual deep networks », TMLR (2024), Mahaut,
    Dessì, Franzon, Baroni. `[résumé moteur]` Réseaux visuels pré-entraînés
    d'architectures différentes : protocole partagé auto-supervisé ; un
    nouveau venu l'apprend. → Touche § 4.2 (hétérogénéité d'architecture,
    pas de substrat impulsionnel). **Ne touche pas** le cœur de 4.2. Impact :
    **Info**.
11. **arXiv:2605.11695** — « Emergent Communication between Heterogeneous
    Visual Agents through Decentralized Learning », 2026, Ochiai.
    `[résumé moteur]` Jeu MHCG, agents à représentations visuelles privées
    différentes, apprentissage décentralisé. → § 4.2, même portée que l'entrée
    10. Impact : **Info**.

### 2.5 Références des contradicteurs absentes de la note (d)

12. **doi:10.3389/fcomp.2026.1873726** — « Dissociating communicative success
    from representational alignment in common ground formation », 2026,
    Yamamoto (*Frontiers in Computer Science*). `[résumé moteur]` Le succès
    référentiel n'entraîne pas l'alignement des représentations internes. →
    Touche note § 3 ligne 2 (le succès ou les métriques côté émetteur ne
    prouvent pas l'usage) : source de nature différente (revue) qui va dans le
    même sens. Cité par Codex. Impact : **Moyen**.
13. **doi:10.1016/j.neucom.2026.135272** — « Beyond latent codes: Decodable
    reference interfaces for robust multi-agent communication », 2027 (en
    ligne 2026), Yao (*Neurocomputing*). `[titre seul]` (OpenAlex sans résumé,
    ScienceDirect 403 pour Codex). Cité par Codex. À lire ; **pas utilisé**
    comme preuve. Impact : non évalué.
14. **arXiv:2611.00006** — « Cultural Evolution Drives the Emergence of
    Conventions in Neural Emergent Communication », 2026, Toquebiau.
    `[résumé moteur]` Populations d'agents bidirectionnels, jeu référentiel,
    entraînement distribué. Même auteur que la thèse citée par Codex
    (doi:10.70675/3385d0bbzc66cz4fceza9e0z8696ff61ff4e, résolue par OpenAlex).
    → **Ne touche pas** la question N10-D. Impact : **Info**.
15. **doi:10.31234/osf.io/dse6n** — « Lost in Translation: Addressing the
    ecological validity of agent-based models of communication emergence
    with insights from an empirical study », 2024, Skowrońska (PsyArXiv,
    trouvé par OpenAlex comme citant de Lowe). `[résumé moteur]` Critique côté
    sciences du langage : le code statique est une vue trop étroite. → Seule
    entrée côté cognitif ; **ne touche pas** la question. Impact : **Info**.

Références des contradicteurs, résolues à ce tour, que la note ne cite pas
(détail dans `resolve-tour-1.tsv`) : Foerster 2016 (1605.06676), Lazaridou
2017 (1612.07182, ajoutée au protocole par A13), Havrylov & Titov 2017
(1705.11192), MADDPG (1706.02275), Kottur 2017 (1706.08502), EGG (1907.00852 ;
D19-3010), Lazaridou & Baroni 2020 (2006.02419), MAPPO (2103.01955), Rita 2022
(2209.15342), JaxMARL (2311.10090), Boldt 2024 (2407.03302), Peters 2024
(2409.02645 ; version revue 10.1007/s10458-025-09691-y), Bouchacourt & Baroni
2018 (1808.10696 ; D18-1119), Główka 2023 (2305.05821 ; version revue
10.1016/j.cogsys.2024.101293), Baronchelli 2005 (physics/0509075),
Pawlowitsch 2008 (10.1016/j.geb.2007.08.009), Argiento 2009
(10.1016/j.spa.2008.02.014), Hofbauer & Huttegger 2008
(10.1016/j.jtbi.2008.07.010), Huttegger 2014 (10.1073/pnas.1400838111),
Lakens 2017 (10.1177/1948550617697177), Scott-Phillips 2012
(10.1098/rspb.2011.2181). Titres concordants. **Aucune n'a été relue à ce
tour** : leur contenu n'est connu que par les rapports des contradicteurs.
Les quatre références de dynamique des jeux de signaux (Pawlowitsch,
Argiento, Hofbauer & Huttegger, Huttegger) forment le seul bloc théorique,
publié en revue, sur la non-convergence vers un système de signaux optimal ;
c'est l'antériorité la plus directe de la porte d'apprentissage (b) et de la
note § 4.4. À lire au tour 2.

### 2.6 Code et données

16. **github:facebookresearch/EGG** — dépôt **archivé** (`archived=true`,
    dernier push 2025-05-06, licence MIT). → Touche note § 5 (« JaxMARL ou
    EGG ») et la proposition TBD-10 (A17) : un backend archivé ne recevra plus
    de correctif ; à geler par empreinte si on le garde (politique HITL).
    Impact : **Moyen**.
17. **github:bold-lab-ai/JaxMARL** — `FLAIROx/JaxMARL` redirige ici ; release
    v0.2.0 du 2026-09-10 annoncée « several breaking changes » (JAX/CUDA,
    noms de variables d'état) ; `jaxmarl/environments/mpe/simple_reference.py`
    et `simple_speaker_listener.py` existent sur `main`. → Touche A11
    (environnement explicite) et TBD-10 : la version doit être épinglée, et
    l'écart `simple_reference` ≠ jeu K = 16 / C = 4 relevé par Codex reste.
    Impact : **Moyen**.
18. **hf:datasets/bboldt/elcc** — ELCC (arXiv:2407.04158, 2024, Boldt),
    corpus de langues émergentes issus d'implémentations publiées ; dernier
    changement 2024-12-11. → Outil, ne touche pas la question. Impact : **Info**.

## 3. Test de recoupement

Règle de la spec §4.2 et de la note : « recoupé » = au moins deux sources de
**nature différente** (article et code, article et thèse, preprint et version
revue, théorie et expérience). Les entrées de ce tour sont `[résumé moteur]`,
elles ne lèvent donc une marque que provisoirement.

| affirmation de la note | sources de nature différente (avant → après ce tour) | ≥ 2 ? |
|---|---|---|
| § 1 / § 3 l. 1 : un canal appris bat le silence quand l'information est privée par construction | actes (Lowe AAMAS, IC3Net ICLR) + preprints + démonstration (hukTsKYu8Y) ; + 2410.18806 / COLING 2025 (preprint et version revue) | **oui** |
| § 3 l. 2 : le succès ou les métriques côté émetteur ne prouvent pas l'usage | actes (Lowe, Eccles) + preprint (Cheng) ; + Yamamoto 2026 (revue) | **oui** |
| § 3 l. 3a : couper le canal en test n'est pas un test du contenu | actes (Lowe) + preprint (Cheng) | **oui** (inchangé) |
| § 3 l. 3b : bras entraînés de zéro avec et sans canal | recommandation (Lowe, actes) + pratique (Chauhan, preprint) | **oui en pratique** (inchangé) |
| § 3 l. 4 : dérangement / substitution / restauration comme protocole causal | articles + code (Gazzaev, dépôt) ; + Pitzer 2026 (perturbation de bits, preprint), uyIn9jZ70C (atelier, PGD) | **oui** |
| § 3 l. 5 : plan 2 × 2 nécessité / suffisance | hukTsKYu8Y seul, sans code | **non** — rien de neuf à ce tour |
| § 3 l. 6 : couplage rigide, canal sans valeur | Chauhan & Bera + Cheng : **deux preprints arXiv**, deux équipes | **non à la lettre** : même nature selon la convention de la note (« deux sources de même nature »). La note le marque « recoupé » ; à requalifier ou à appuyer sur la version revue (OAmBy25HHg, ICLR 2027, pas encore décidée) |
| § 3 l. 7 : le cross-play s'effondre entre entraînements indépendants | Chauhan & Bera + Marincat : **deux preprints** ; Kim 2026 (preprint) **nuance** (grandes populations) | **non à la lettre** (même remarque) ; plus une nuance de portée |
| § 3 l. 8 : messages discrets appris | actes (Liu NeurIPS 2021) + preprint (Chauhan) ; + Tucker 2025 (revue) | **oui** |
| § 3 l. 10 : positive listening partenaire contre étranger | preprint (Denamganaï) + actes (Lowe) + code | **oui** (inchangé) |
| § 4.1 : absence d'un nul apparié par graine sur bras entraînés avec et sans canal | absence constatée sur 6 canaux au tour 2, de nouveau sur 6 canaux ici | **absence recoupée**, qui reste une inférence |
| § 4.2 : même test sur substrats hétérogènes MLP/LIF | résumé OpenReview `spio7kWIWr` + lecture de hukTsKYu8Y ; ce tour : hétérogénéité de modalité (2601.22041), d'architecture (8L3khbpUJL, 2605.11695), **jamais de substrat impulsionnel** | **non** — à recouper |
| § 4.4 : un récepteur peut ne pas utiliser un canal nécessaire (existence) | actes (Eccles) + preprint (Chauhan) ; + théorie en revue (Chernyavskiy 2026 ; Pawlowitsch, Hofbauer & Huttegger, Huttegger, titres seuls) | **oui** pour l'existence |
| § 4.4 : le chiffre (appris 2 bits indistinguable du silence, p = 0,400) | Chauhan seul | **non** — à recouper |

Bilan : 5 marques « à recouper » à la lettre de la convention (§ 3 l. 5,
§ 4.2, chiffre de § 4.4, plus les deux lignes § 3 l. 6 et l. 7 qui ne tiennent
que par deux preprints), contre 3 dans la note v0.3. Ce tour n'en lève
aucune de façon définitive (pas de lecture intégrale) ; il en signale deux de
plus.

## 4. Constat de saturation

**Non saturé.** Ce tour apporte 17 entrées distinctes nouvelles, dont au moins
trois qui changeraient une phrase de la note si elles étaient lues en entier :
2410.18806 (code partiel suffisant, § 1 et § 5), Chernyavskiy 2026 (seuil
d'émergence, § 4.4), Kim 2026 (nuance du cross-play, § 3 l. 7). Le bloc
théorique des jeux de signaux (Pawlowitsch, Argiento, Hofbauer & Huttegger,
Huttegger), cité par les contradicteurs et absent de la note, est un trou de
lecture.

Ce que le tour 2 devrait viser :

1. **Snowball S2 en lot** (script `semantic-scholar`, clé, reprise sur 429)
   des citants depuis 2024 de Lowe 2019, Eccles 2019, Lazaridou 2017, et des
   références de hukTsKYu8Y (pp. 10-13 du PDF du PI). S2 a renvoyé 429 ici et
   OpenAlex ne couvre qu'une fraction.
2. **Lire en entier** 2410.18806 (COLING 2025), Chernyavskiy 2026, et au moins
   Pawlowitsch 2008 et Huttegger 2014, puis réécrire § 4.4 et la porte A1 avec
   ces sources.
3. **Version revue** de Chauhan & Bera (OpenReview OAmBy25HHg, décision ICLR
   2027) et une seconde source non-preprint pour § 3 l. 6 et l. 7.
4. **Substrats impulsionnels** (§ 4.2) : OpenReview `spiking` + `referential
   game` / `signaling`, et bioRxiv en balayage complet (le tour n'a utilisé
   que l'index Europe PMC des preprints).
5. **Thèses/HAL et theses.fr** (canal de la spec §4.2, non interrogé
   directement ici) : Toquebiau 2024 et thèses 2025-2026 sur la communication
   émergente.
6. **Pratique des backends** : changelog de JaxMARL v0.2.0 (ce qui casse
   `simple_reference`) et décision sur EGG archivé, avant de geler TBD-10.

## 5. Verdict provisoire

**La question telle qu'écrite en `P100` § 1 est tranchée par construction ; la
question réduite reste ouverte.**

- Tranché par construction : `eng − nocomm` sur `asym`. Preuves : contraste
  oracle 0,75 / 0 calculé sans entraînement (A2, convergence 1 de
  SYNTHESE.md) ; Lazaridou 2017 (1612.07182, résolu, cité par les deux
  contradicteurs) ; hukTsKYu8Y Prop. 1 (note § 1) ; et à ce tour 2410.18806,
  qui montre que le jeu référentiel se résout avec un ou deux symboles
  `[résumé moteur]`.
- Reste ouvert : l'usage **causal** du canal par le bras `learned` appris par
  PPO, contre un bras `nocomm` entraîné de zéro, avec un nul apparié par
  graine. Aucune des 17 entrées de ce tour ne combine les trois ; la plus
  proche (uyIn9jZ70C : PPO, locuteur-auditeur, intervention sur le message)
  fait une attaque adverse sur des messages continus, sans bras sans canal et
  sans nul par graine.
- Pas inutile : la littérature des dynamiques de jeux de signaux et
  Chernyavskiy 2026 disent que la sortie du babillage n'est pas garantie ;
  savoir si PPO la franchit sur K = 16 / C = 4 / A = 16 est donc une question
  empirique, même si elle est de portée modeste (« plumbing and PPO learning
  validated », A2).

Ce verdict tient sur des résumés ; il ne vaut pas passage du verrou.

## 6. Non résolus / inaccessibles

- Semantic Scholar : HTTP 429 sur `/paper/arXiv:1912.05676/citations` et
  `/paper/arXiv:1903.05168/citations` ; rien n'est compté.
- OpenReview `notes?id=` : HTTP 403 `ChallengeRequiredError` pour
  uyIn9jZ70C, 8L3khbpUJL, CQ5ivZEB4e, AiEfgKQh2P ; les titres et résumés
  viennent de `notes/search`, qui renvoie l'identifiant de forum.
- arXiv en HTTP clair : 301 ; tout a été refait en HTTPS (200).
- DataCite n'a pas de notice pour 10.48550/arXiv.2103.01955 (MAPPO) ; résolu
  par l'API arXiv `id_list`.
- doi:10.1016/j.neucom.2026.135272 : titre seul (pas de résumé OpenAlex).
- Aucun identifiant cité ici ne reste non résolu.
