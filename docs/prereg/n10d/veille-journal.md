# Journal de veille — N10-D

Un bloc par tour. Chaque identifiant gardé a été résolu par API (arXiv API ou
page abs arXiv, Semantic Scholar en lot, OpenReview `notes/search`) avec un
User-Agent neutre, 3 s entre deux appels arXiv. Aucun identifiant n'est écrit
de mémoire.

## Tour 1 — 2026-11-01

### Canaux interrogés

| canal | interrogé | remarque |
|---|---|---|
| arXiv API | oui | 5 requêtes |
| OpenReview API (`notes/search`) | oui | 4 termes × 25 ; `notes?id=` et PDF en 403 ChallengeRequired |
| Semantic Scholar (lot, références + citations) | oui | snowball sur les 7 sources primaires |
| bioRxiv | **non** | report au tour 2 |
| Crossref (revues) | **non** | report au tour 2 |
| Thèses / HAL | **non** | report au tour 2 |
| Code et données (GitHub, Hugging Face) | **non** | report au tour 2 |

### Requêtes exactes et comptes

| canal | requête | brut |
|---|---|---:|
| arXiv | `ti:"Biases for Emergent Communication"` | 3 |
| arXiv | `abs:"positive listening"` | 4 |
| arXiv | `abs:"emergent communication" AND abs:causal` | 5 |
| arXiv | `abs:"referential game" AND abs:asymmetr*` | 0 |
| arXiv | `abs:communication AND abs:multi-agent AND abs:"message shuffl*"` | 0 |
| OpenReview | `positive listening emergent communication` | 25 |
| OpenReview | `causal test communication channel use multi-agent` | 25 |
| OpenReview | `asymmetric information referential game communication necessary` | 25 |
| OpenReview | `message intervention listener ablation emergent language` | 25 |
| S2 snowball | références (amont) des 7 sources | 287 |
| S2 snowball | citations (aval) des 7 sources | 668 |

Comptes brut → dédoublonné → gardé :

- arXiv : 12 brut ; dédoublonnage non consigné à ce tour ; 6 entrées gardées
  (requêtes et pages abs), plus 2 résolues depuis les références (Eccles 2019,
  Jaques 2018).
- OpenReview : 100 notes → 80 uniques → 60 forums (beaucoup de reviews) →
  3 pertinents (AsymPuzl Uft46u0GRx ; me1MpmENpw ; Eccles 2019 l0S7QgBMjM) ;
  aucun n'est entré dans la note comme source primaire.
- Snowball S2 : 955 lignes → 865 identifiants distincts (17 sans identifiant)
  → 4 citants de Lowe gardés après filtre par titre (Vanneste 2020 et 2025,
  Denamganaï PoPLi, Bailey JAIR 2026). Fichier : `snowball-tour1.tsv`.

Sources primaires du snowball, résolues en lot S2 (titres conformes) :

| source | références | citations |
|---|---:|---:|
| arXiv:1903.05168 | 48 | 170 |
| arXiv:1812.09755 | 37 | 354 |
| arXiv:2107.02367 | 59 | 55 |
| arXiv:2405.08550 | 49 | 88 |
| arXiv:2609.27252 | 44 | 1 |
| arXiv:2610.08612 | 17 | 0 |
| arXiv:2609.32605 | 33 | 0 |

### Entrées gardées (identifiant résolu)

Toutes en `[résumé moteur]` à ce tour.

| id | auteurs, année | pourquoi |
|---|---|---|
| arXiv:2610.37849 | Denamganaï, 2026 | PoPLi : positive listening partenaire contre étranger, test d'usage du canal |
| arXiv:2609.34373 | Chauhan & Bera, 2026 | « Emergence, Not Bandwidth » : sous couplage rigide aucun canal ne bat le silence (p = 0,982, n = 25) ; démarrage à chaud 0,857 contre 0,562 ; cross-play 0,980 → 0,144 |
| arXiv:2607.26773 | Zhang & Emu, 2026 | audit causal de canaux latents |
| arXiv:2608.04893 | Cheng, Das & Ramnath, 2026 | audit causal de caches KV relayés (dérangés, mis à zéro, aléatoires) |
| arXiv:2610.15163 | Sanzaru, 2026 | asymétrie épistémique entre agents |
| arXiv:2601.17454 | Atif, 2026 | prédictions fixées d'avance (exemple de méthode), domaine tabulaire |
| arXiv:1912.05676 | Eccles et al., 2019 | biais de positive signalling et listening |
| arXiv:1810.08647 | Jaques et al., 2018 | influence sociale comme récompense, citée par Lowe [21] |

Hors API à ce tour : OpenReview hukTsKYu8Y (titre et résumé obtenus par
`notes/search`, texte intégral en 403) ; arXiv:2610.28905 (Gazzaev et al.,
cité dans le rapport du 01/11, `synthese/rapport.md:242`) **non résolu ici**,
« non vérifié ».

### Ce que le tour a changé dans la note

- § 1 : la question de `P100` (eng − nocomm) est reconnue comme attendue ; la
  question ouverte devient l'usage causal du bras `learned`.
- § 3 : l'antériorité du bras causal (dérangement / substitution /
  restauration) est établie en méthode par 4 sources, dont 3 en résumé.
- § 4 : ce qui reste ouvert se réduit à (a) le nul apparié par graine,
  (b) les substrats hétérogènes et le chemin γ/θ, (c) Claim B, tous
  « à recouper ».
- § 6 : saturation et couverture non passées.

## Tour 2 — 2026-11-01

User-Agent neutre « hypneum-lab reading-gate », 3 s entre deux appels arXiv,
aucune adresse e-mail transmise. Identifiants résolus par API avant d'être
écrits ; `resolve_ids.py` sur `reading-note.md` : voir fin du bloc.

### Canaux interrogés

| canal | interrogé | remarque |
|---|---|---|
| arXiv API | oui (complément) | 12 requêtes API, 8 PDF ouverts |
| OpenReview API (`notes/search`) | oui | 7 requêtes × 50 notes ; titre de hukTsKYu8Y retrouvé, PDF et `notes?id=` toujours en 403 |
| Semantic Scholar (snowball) | oui | 6 sources de plus (13 au total avec le tour 1) |
| bioRxiv | oui | API `details`, catégorie neurosciences seulement |
| Crossref (revues) | partiel | une requête de principe (Sourlas) et résolution d'identifiants ; pas de balayage par mots-clés |
| Thèses / HAL | oui | HAL (5 requêtes, 3 notices) et theses.fr (4 requêtes) |
| Code et données | oui | GitHub, Hugging Face, Zenodo, OSF |

### Requêtes exactes et comptes

| canal | requête | brut |
|---|---|---:|
| OpenReview | `emergent communication message shuffle ablation listener` | 50 (3 retenus 2026/27) |
| OpenReview | `learned communication versus no communication multi-agent seeds` | 50 (10) |
| OpenReview | `causal influence of communication permuted message` | 50 (4) |
| OpenReview | `referential game information asymmetry listener necessary` | 50 (7) |
| OpenReview | `communication channel use verified by intervention multi-agent reinforcement learning` | 50 (8) |
| OpenReview | `discrete messages heterogeneous modules spiking communication` | 50 (14) |
| OpenReview | `message derangement receiver audit` | 50 (7) |
| OpenReview | `Causal Protocol Verifying Global-Workspace Computation Language Model` (titre de hukTsKYu8Y) | 50 |
| arXiv | `ti:"Does the Workspace Do the Work"` | 0 |
| arXiv | `abs:"global workspace" AND abs:"causal protocol"` | 0 |
| arXiv | `abs:"global workspace" AND abs:"language model" AND abs:broadcast` | 3 (aucune n'est hukTsKYu8Y) |
| arXiv | `all:"Gazzaev"` ; `id_list=2610.28905` | 4 ; 1 (résolu) |
| arXiv | `ti:"Towards best practices of activation patching"` ; `ti:"inter-seed cross-play"` | 1 ; 1 |
| arXiv | `(abs:Hopfield OR abs:"associative memory") AND (abs:"error-correcting" OR abs:"channel coding" OR abs:"noisy channel") AND abs:decoding` | 4 |
| arXiv | `id_list` 1407.6029, 2511.10302, 2609.11365, 2206.11302 (notices) | 4 |
| bioRxiv | `details/biorxiv/2024-01-01/2026-11-01/<curseur>?category=neuroscience`, 1 271 pages de 30, filtre regex sur titre + résumé (inter-area/module communication, emergent communication, communication subspace + learn/causal/ablat/perturb, gamma/theta + communication + model, learned communication/message, multi-agent, message passing + spik/cortex/area, causal test/ablat/perturb + communication/message) | 38 126 notices ; 57 correspondances |
| HAL | `"emergent communication"` ; `"communication émergente"` ; `multi-agent AND "communication" AND "reinforcement"` ; `"jeu référentiel"` ; `"referential game"` avec `fq=docType_s:THESE` | 4 ; 4 ; 32 ; 0 ; 0 |
| theses.fr | `communication émergente` ; `communication multi-agents apprentissage` ; `emergent communication` ; `réseaux de neurones à impulsions communication modulaire` | 1 383 ; 208 ; 6 771 ; 2 (premières pages seulement) |
| GitHub | `emergent communication` ; `positive listening` ; `causal influence of communication` ; `referential game` ; `latent communication audit` ; `hidden messages diffusion` (plus 4 requêtes en phrases longues à 0 résultat) | 20 ; 2 ; 0 ; 12 ; 1 ; 1 |
| Hugging Face | `emergent communication` ; `positive listening` ; `message ablation` ; `emergent-communication` ; `referential` ; `multi-agent-communication` ; `marl` (modèles et jeux de données) | 0 pour les cinq premiers ; `referential` et `marl` : bruit sans rapport |
| Zenodo | `"emergent communication"` ; `"positive listening"` ; `"message ablation" AND multi-agent` ; `"referential game"` | 49 ; 8 ; 0 ; 3 |
| OSF | titre : `emergent communication` ; `positive listening` ; `multi-agent communication` | 0 ; 0 ; 0 |
| S2 snowball | références et citations des 6 sources du tour 2 | 291 lignes (177 amont, 114 aval) |

Comptes brut → dédoublonné → gardé :

- OpenReview : 400 notes brutes ; 53 forums 2026/27 retenus → 52 distincts →
  7 résumés lus → **3 gardés** (`spio7kWIWr`, `dz0yb4MlKd`, `OAmBy25HHg`,
  cette dernière étant la version soumise ICLR 2027 de arXiv:2609.34373, donc
  pas une source indépendante). Écartés après lecture du résumé : `dF1DqBmdgX`
  (protocoles de communication à boîte blanche), `gbgPFLw4oU` (sélection de
  messages, nul de densité), `98HYjoH78A` (MARL hors ligne), `uSOl5K0PJY`
  (désinformation et compositionnalité).
- arXiv : 12 requêtes ; **gardées** : 2610.28905 (résolue et lue), 2609.11365
  (lue), 1407.6029, 2608.03644, 2309.16042 (résumé), 2206.11302 (résumé,
  écartée de la note) ; 2511.10302 écartée (mémoire de Hopfield moderne pour
  la compression sémantique, pas la tolérance au bruit d'un code fixe).
- bioRxiv : 38 126 → 57 correspondances → 37 DOI distincts → 4 titres relus
  (communication subspace entre populations de neurones, théorie
  neuronale de l'inter-aire, géométrie de communication thalamo-corticale) →
  **0 gardé** : ce sont des études de neurosciences biologiques, aucune ne
  teste un message appris ni ne change une phrase de la note.
- HAL / theses.fr : 3 thèses relues sur résumé (Toquebiau 2024, Chaabouni
  2021, Pontes Mota 2024) → **0 gardée** (robotique MARL, universaux
  linguistiques, protocoles réseau). Titres vus sans résumé : Rita 2024
  (émergence du langage), Karch 2023, Marino 2025.
- Code et données : GitHub **3 gardés** (`Near32/PartnerOutsiderPositiveListening`
  MIT, `alanrbtx/hidden-messages-diffusion-agents` licence non déclarée,
  `SudarshanHegde326/emergent-comm-usage-marl` licence non déclarée) ;
  écarté : `ZYunYu/latent-communication-audit` (dépôt vide, 404 sur le
  contenu). Zenodo : 1 enregistrement relu (`10.5281/zenodo.23202925`,
  auteur signé comme un agent IA, monde jouet à 5 graines) → **écarté**.
  Hugging Face et OSF : 0.
- Snowball S2 : 291 lignes → 243 identifiants distincts → 176 hors
  `snowball-tour1.tsv` → 72 après filtre par titre (mots communication /
  message / listening / referential / signal / latent / workspace / hidden /
  spiking / intervention / causal / collusion / emergent) → titres lus :
  **1 pertinent qui change la note** (arXiv:2609.11365), 1 méthodologique
  écarté (arXiv:2206.11302). Fichier : `snowball-tour2.tsv`. Les 104 titres
  hors filtre ne sont pas triés.

### Entrées gardées (identifiant résolu)

| id | auteurs, année | statut de lecture | pourquoi |
|---|---|---|---|
| arXiv:2610.28905 | Gazzaev, Gavrilov, Muravyov, Makarov, 2026 | lue, 9 pages | dérangement, substitution ciblée, canal appris entre agents LM à diffusion ; bootstrap sur exemples ; code publié |
| arXiv:2609.11365 | Marincat, 2026 | lue, pp. 1-8 sur 17 | non-interopérabilité de protocoles appris entre entraînements indépendants (26/26) ; deuxième source pour le cross-play |
| arXiv:2608.04893 | Cheng, Das, Ramnath, 2026 | lue, pp. 1-10 sur 26 | tour 1 en résumé ; zéro contre dérangé (14,7 contre 0,4), régimes nécessaire / non nécessaire, graines de décodage |
| arXiv:2609.34373 | Chauhan & Bera, 2026 | lue, pp. 1-9 sur 17 | tour 1 en résumé ; tests appariés par graine (25), appris indistinguable du silence en couplage partiel |
| arXiv:2610.37849 | Denamganaï, 2026 | lue, pp. 1-6 sur 17 | tour 1 en résumé ; reprend Lowe §6.2 ; code publié |
| arXiv:1912.05676 | Eccles et al., 2019 | lue, pp. 1-5 sur 16 | tour 1 en résumé ; équilibre où le récepteur ignore le canal |
| arXiv:2607.26773 | Zhang & Emu, 2026 | lue, pp. 1-5 sur 9 | tour 1 en résumé ; quatre réglages de message ; préférence pour le remplacement dans la distribution |
| arXiv:1407.6029 | Fiete, Schwab, Tran, 2014 | `[résumé moteur]` | second groupe indépendant pour le code fixe décodé par attracteurs |
| arXiv:2608.03644 | Wolski, Hoernle, Forkel, Foerster, 2026 | `[résumé moteur]` | cross-play inter-graines : limite méthodologique, citée par Marincat ; non utilisée dans la note |
| arXiv:2309.16042 | Zhang & Nanda, 2024 | `[résumé moteur]` | bonnes pratiques du patching d'activations, citée par Zhang & Emu ; utilisée seulement comme renvoi |
| OpenReview spio7kWIWr | Gabayre et al., soumission ICLR 2026 | `[résumé moteur]` | messages impulsionnels dans un jeu référentiel ; précédent le plus proche pour § 4.2, sans intervention causale annoncée |
| OpenReview dz0yb4MlKd | auteurs masqués, soumission ICLR 2027 | `[résumé moteur]` | des entrées sans contenu spécifique, calibrées pour changer les réponses autant que les vrais messages, reproduisent 76 % du gain d'oracle ; renforce l'exigence de contrôles appariés |
| GitHub Near32/PartnerOutsiderPositiveListening | Denamganaï, 2026 | README lu | code de PoPLi |
| GitHub alanrbtx/hidden-messages-diffusion-agents | Gazzaev et al., 2026 | README lu | code et prédictions de Gazzaev et al. |
| GitHub SudarshanHegde326/emergent-comm-usage-marl | mémoire MSc, 2026 | README et début de script lus | ablation par message mis à zéro, 3 graines |

Hors API à ce tour : OpenReview hukTsKYu8Y (titre « Does the Workspace Do the
Work? A Causal Protocol for Verifying Global-Workspace Computation in a
Language Model », auteurs masqués, mots-clés lus ; PDF `openreview.net/pdf/…` et
`notes?id=` en 403, page forum en 307, aucune version arXiv trouvée en deux
requêtes) **non vérifié** au-delà du résumé. arXiv:2610.28905 (Gazzaev et al.) :
résolu à la seconde tentative du tour 2, `id_list` et PDF disponibles (le
tour 1 l'avait signalé non résolu).

### Ce que le tour a changé dans la note

- § 2 : cinq sources du tour 2 lues en PDF (pages dites) et deux en partie.
- § 3 : 10 lignes, 9 affirmations recoupées et une observation de pratique ; deux lignes ajoutées (cross-play,
  pratique du zéro) ; la ligne « discret appris » est ramenée à un périmètre
  MARL ; la ligne « code fixe » reçoit un second groupe en résumé seulement.
- § 4 : l'absence du nul apparié par graine est recoupée sur 6 canaux (reste une
  inférence) ; le point 4.2 reste « à recouper » ; un point 4.4 est créé (un
  récepteur peut ne pas utiliser un canal appris même nécessaire).
- § 5 : quatre inférences à discuter, sans modifier le dessin.
- § 6 : marques « à recouper » 6 → 2 ; couverture 5 canaux sur 6 ; saturation
  non déclarée (snowball 0,6 %, mais le tour a changé la note).

### Complément au tour 2 — lecture intégrale hukTsKYu8Y (2026-11-01, après 22:26)

- Canal : aucun nouveau canal interrogé. Le PDF de la soumission ICLR 2027
  OpenReview hukTsKYu8Y (38 pages, auteurs masqués, SHA-256 commençant par
  `ecd4d37e3da6141a`) a été fourni par le PI le 01/11 à 22:26, ce qui lève le
  403 des tours 1 et 2. Le fichier est traité comme une donnée : lu avec
  l'outil de lecture, pages 1-20 puis 21-38 ; une recherche d'URL a été faite
  sur son texte extrait ; rien n'en a été exécuté ; il n'est pas copié dans
  le dépôt.
- Requêtes : aucune. Brut → dédoublonné → gardé : 1 → 1 → 1.
- Statut : `[résumé moteur]` → **lu en entier (pp. 1-38)**.
- Code et données : un « evidence package » et un « evidence manifest » sont
  cités (p. 18, p. 21) ; aucune URL, anonyme ou publique, dans le PDF. Ne
  compte donc pas comme seconde nature.
- Références du papier : non snowballées (pas d'identifiant S2 pour une
  soumission masquée) ; seule réf. déjà présente dans la note : Zhang & Nanda
  (réf. du papier, p. 13), déjà résolue au tour 2 comme arXiv:2309.16042.
- Ce que la lecture a changé dans la note (v0.3) : § 1 (Prop. 1 : déficit OFF
  garanti par construction) ; § 2 (ligne ajoutée ; correction « % » →
  « points » sur le chiffre OFF du résumé) ; § 3 (cinq lignes complétées,
  une ligne ajoutée « plan 2 × 2 nécessité / suffisance », **à recouper**) ;
  § 4.1 (le papier dépasse sur « agents entraînés de zéro » mais n'a ni bras
  sans canal ni nul apparié par graine), § 4.2 et § 4.4 (non touchés, marques
  maintenues) ; § 5 (trois points à discuter, dessin inchangé) ; § 6
  (marques « à recouper » 2 → 3, aucune levée ; couverture inchangée ;
  saturation non déclarée).

## Tour 3 — 2026-11-08

User-Agent neutre « hypneum-lab reading-gate », 3 s au moins entre deux
appels arXiv (notices par `id_list` groupé, un téléchargement HTML par
tour d'outil), aucune adresse e-mail transmise. Identifiants résolus par API
avant d'être écrits ; aucun identifiant de mémoire. Les deux tours datés du
06 et du 07/11 (`veille/tour-1-2026-11-06.md`, `veille/tour-2-2026-11-07.md`,
après contradicteur) ont été lus avant ce tour ; leurs entrées ne sont pas
recomptées ici. Le rapport du contradicteur n'a pas été consulté.

Objet du tour : les manques listés en § 6 de la note v0.3 (tri des titres
hors filtre du tour 2 ; snowball de Marincat, de Chauhan & Bera et de
l'étude parent de Marincat ; bioRxiv hors neurosciences ; Crossref et
OpenAlex par mots-clés ; lecture intégrale de PoPLi, Zhang & Emu,
Marincat § 5-§ 7, Eccles ; second résultat pour 4.4 et 4.2).

### Canaux interrogés

| canal | interrogé | remarque |
|---|---|---|
| arXiv API | oui | 3 appels `id_list` (8, 7 et 1 notices) ; 6 textes HTML ouverts |
| OpenReview | non | rien de neuf à demander ; `OAmBy25HHg` (version ICLR 2027 de Chauhan & Bera) non redemandé |
| Semantic Scholar (snowball) | oui | 3 sources, 6 appels, 2 × 429 puis 200 |
| bioRxiv | oui | API `details` **par catégorie**, 10 catégories hors neurosciences, 2024-01-01 → 2026-11-08 ; Europe PMC (`SRC:PPR`) en complément |
| Crossref (revues) | oui | 7 requêtes `query.bibliographic`, `from-pub-date:2019` |
| OpenAlex | partiel | 6 requêtes `search`, **4 en HTTP 503** (3 essais chacune) ; résolution de 5 DOI |
| Thèses / HAL | non | couvert au tour 2 (01/11) et le 07/11 (13 requêtes, 0 gardé) |
| Code et données | oui | GitHub et Hugging Face : vérification du dépôt d'artefacts de Marincat |

### Requêtes exactes et comptes

| canal | requête | brut |
|---|---|---:|
| local | tri des titres du `snowball-tour2.tsv` absents de `snowball-tour1.tsv` et hors filtre par titre (recompte : 184 nouveaux, 77 dans le filtre, **107 hors filtre** ; le tour 2 écrivait 176 / 72 / 104, écart dû aux racines du filtre, `emergen`, `communicat`) | 107 |
| arXiv | `id_list=2610.22026,2606.03280,2608.20054,2610.33020,2003.02979,2506.06609,2604.03266,2610.15163` | 8 |
| S2 | `/paper/arXiv:2609.11365/references` ; `/citations` | 42 ; 1 |
| S2 | `/paper/arXiv:2609.34373/references` ; `/citations` | 18 ; 0 |
| S2 | `/paper/arXiv:2608.20054/references` ; `/citations` | 35 ; 1 |
| arXiv | `id_list=2609.17637,2608.16347,2608.13317,2608.11676,2607.11967,2502.07552,2204.12982` | 7 |
| arXiv | `id_list=2512.05654` | 1 |
| arXiv (HTML) | 2610.37849, 2607.26773, 2609.11365, 1912.05676, 2608.20054, 2609.17637 | 6 textes |
| Crossref | `emergent communication multi-agent reinforcement learning listener message ablation` | 40 |
| Crossref | `positive listening emergent communication causal` | 40 |
| Crossref | `referential game learned communication no communication baseline` | 40 |
| Crossref | `Lewis signaling game reinforcement learning receiver ignores signal` | 40 |
| Crossref | `communication channel use intervention multi-agent permuted message` | 40 |
| Crossref | `spiking neural network emergent communication heterogeneous agents` | 40 |
| Crossref | `learned discrete messages between modules spiking and artificial neurons` | 40 |
| OpenAlex | `spiking emergent communication heterogeneous` ; `signaling game reinforcement learning babbling receiver` (`from_publication_date:2019-01-01`) | 4 763 (40 pris) ; 54 (40 pris) |
| OpenAlex | `emergent communication listener message ablation causal` ; `positive listening emergent communication` ; `referential game no-communication baseline seeds` ; `learned communication channel intervention multi-agent reinforcement learning` | **503 × 4** |
| Europe PMC | `(SRC:PPR) AND ("emergent communication" OR "learned communication" OR "referential game" OR "signaling game") AND (spiking OR neuromorphic OR "integrate-and-fire")` | 0 |
| Europe PMC | `(SRC:PPR) AND "emergent communication" AND (ablation OR intervention OR causal OR shuffled OR permuted)` | 2 |
| Europe PMC | `(SRC:PPR) AND ("multi-agent" OR "multiagent") AND ("learned messages" OR "communication channel") AND ("reinforcement learning")` | 5 |
| Europe PMC | `(SRC:PPR) AND "positive listening"` | 1 |
| bioRxiv | `details/biorxiv/2024-01-01/2026-11-08/<curseur>?category=<c>` pour c ∈ {animal behavior and cognition, evolutionary biology, bioengineering, systems biology, biophysics, bioinformatics, ecology, synthetic biology, scientific communication and education, zoology} ; filtre regex sur titre + résumé (emergent/learned communication, learn to communicate, multi-agent, referential/signaling game, positive listening, message passing + agent/module, spiking + communicat + agent/module/learn, communicat + agent/module + ablat/perturb/intervention/causal) | 60 857 |
| GitHub / HF | `repos/tokenosopher/populus-evidence-partitioning` (+ README) ; `api/models/tokenosopher/populus-evidence-partitioning-checkpoints` | 1 ; 1 |

Comptes brut → dédoublonné → gardé :

- Titres hors filtre du tour 2 : 107 titres lus → 6 résumés ouverts
  (2610.22026, 2606.03280, 2608.20054, 2610.33020, 2003.02979, 2506.06609)
  → **3 gardés** (2608.20054, qui change la note ; 2606.03280 et
  2610.22026, information). Les 101 autres : infrastructure RL, LLM,
  caches KV, diffusion, stéganographie, jeux de données ; aucun ne porte sur
  un canal appris entre agents.
- Snowball S2 (tour 3) : 97 lignes → 85 identifiants distincts → **54
  hors `snowball-tour1.tsv` et `snowball-tour2.tsv`** → 22 après filtre par
  titre élargi (+ `interoperab`, `cross-play`, `protocol`) → 7 résumés lus
  (2609.17637, 2608.16347, 2608.13317, 2608.11676, 2607.11967, 2502.07552,
  2204.12982) → **4 gardés** (2609.17637, qui change la note ; 2608.16347,
  2607.11967, 2204.12982, information). Fichier : `snowball-tour3.tsv`.
  L'étude parent de Marincat est résolue : arXiv:2608.20054 (citée en
  amont par 2609.11365, titre conforme) ; sa confirmation préenregistrée
  est arXiv:2609.17637 (citant aval).
- Crossref : 280 notices brutes → 237 passent le filtre par titre (filtre
  trop lâche, « signal » et « message ») → 80 titres relus à la main → 8
  examinés → 5 DOI résolus par OpenAlex (3 avec résumé) → **1 gardé**
  (information) : doi:10.1109/wsai70435.2026.11708582. Les autres : réseaux
  de capteurs, télécoms, neurosciences matérielles, communication
  référentielle humaine.
- OpenAlex : 2 requêtes servies (80 notices prises) → 1 gardé
  (information) : arXiv:2512.05654 ; 4 requêtes en 503 persistant,
  **non vérifiées**.
- Europe PMC : 8 notices → 0 gardé (robotique, télécoms, COVID, audition).
- bioRxiv hors neurosciences : 60 857 notices parcourues dans 10
  catégories → 143 (98 DOI distincts) correspondances de mots-clés → 0
  gardée(s) (systèmes multi-agents LLM pour la bio-informatique, fourragement collectif appris, populations cellulaires, cognition sociale ; aucune étude d'un message appris entre agents ni d'une intervention sur un canal).
- Code et données : dépôt `tokenosopher/populus-evidence-partitioning`
  (MIT pour le code, CC BY 4.0 pour les données ; dernier push 2026-09-14 ;
  README lu : scripts, résultats JSON, `PREREGISTRATION.md`, `INCIDENTS.md`,
  dossier `companion/` pour 2609.11365) et dépôt Hugging Face
  `tokenosopher/populus-evidence-partitioning-checkpoints` (public, 102
  fichiers, dernière modification 2026-09-13) → **2 gardés** (code et
  données, non relancés).

### Entrées gardées (identifiant résolu)

| id résolu | auteurs, année | titre | pourquoi gardé | statut de lecture |
|---|---|---|---|---|
| arXiv:2608.20054 | Marincat, 2026 (v3) | What You Can't See Is What You Learn: Slot-Selective Evidence Masking Favors Compositional Generalization in Shared-Genome Language-Model Societies | étude parent de 2609.11365 ; 10 paires restreint/global appariées par initialisation et ordre de données ; nul par retournement de signe exact sur les 5 moyennes par initialisation ; coupure de tous les paquets en test = hasard 10/10 ; transplants même-valeur et contrefactuels, suppression, bruit, mélange inter-exemples ; préenregistrement anonymisé | **lue en partie** : § 3 (architecture) à § 7 (limites) ; § 1-2 en résumé, annexes A-F non lues |
| arXiv:2609.17637 | Marincat, 2026 (v2) | What You Can't See Is Still What You Learn: A Preregistered Sixty-Society Confirmation That Evidence Masking Drives Compositional Generalization | confirmation préenregistrée, 60 sociétés, 6 strates d'initialisation × 2 ordres, test de retournement de signe à six groupes, contrastes C1-C6, all-cut au hasard 12/12 ; batterie mécanisme non décisionnelle avec cellules intact / même-valeur / contrefactuel / contrefactuel apparié sur l'évidence / suppression (zéro) / **dérangé** (donneur aléatoire de valeur différente) / bruit | **lue en partie** : § 5 (résultats, Table 4) et début de § 6 (Table 5) ; reste non lu |
| arXiv:2606.03280 | Zhang & Xin, 2026 | A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting | alignement linéaire hors ligne (cosinus ≈ 0,97) insuffisant pour une communication causale utile dans le récepteur ; troisième preprint pour § 3 l. 7, LLM gelés | `[résumé moteur]` |
| arXiv:2608.16347 | Cardenas Piepereit, 2026 | Architecture-Dependent Causal Transfer of Activation States Across Large Language Models | effet causal préenregistré d'un transfert projeté entre architectures décodeurs ; **nuance** § 3 l. 7 (un alignement appris récupère quelque chose) | `[résumé moteur]` |
| arXiv:2610.22026 | Deng, Liu, Zhu, 2026 | Augment the Cache, Replace the Activation: A 2x2 Study of Internal-State Sharing | « 2 × 2 » = porteur × mécanisme de fusion, pas nécessité × suffisance ; ne recoupe **pas** § 3 l. 5 | `[résumé moteur]` |
| arXiv:2607.11967 | Truong & Phan, 2026 | Beyond Coordinate Gauge: An Audited Protocol for Detecting Donor-Specific Functional Fingerprints after Neural Collapse | protocole audité de transplantation entre réseaux indépendants (5 réseaux, 20 paires, p = 0,0083 par permutation exacte) ; distingue détectabilité, transplantabilité, persistance causale | `[résumé moteur]` |
| arXiv:2204.12982 | Rita, Strub, Grill, Pietquin, Dupoux, 2022 | On the role of population heterogeneity in emergent communication | la structure du langage émergent ne dépend que de l'écart **relatif** de vitesse d'apprentissage entre émetteur et auditeur ; source neuronale pour le lien « vitesse d'adaptation du récepteur » de la note § 4.4 (Hofbauer & Huttegger, Calvano), mais porte sur la structure, pas sur le non-usage | `[résumé moteur]` |
| doi:10.1109/wsai70435.2026.11708582 | WSAI 2026 (auteurs non lus) | Message Masking and Curriculum Adversarial Training for Robust Emergent Communication in Cooperative Multi Agent Systems | version en actes du travail `uyIn9jZ70C` (tour du 06/11) ; « message masking » = mettre à zéro la copie propre du message dans l'observation de l'auditeur pour forcer la dépendance au canal (raccourci caché) ; information pour § 3 l. 2 | `[résumé moteur]` (OpenAlex) |
| arXiv:2512.05654 | Jang, Song, Lee, Shim, 2025 | A Note on Emergent Behavior in Multi-agent Systems Enabled by Neuro-spike Communication | synchronisation d'agents hétérogènes par impulsions 1 bit (théorie du contrôle) ; aucun message appris, aucune intervention : **ne touche pas** § 4.2, consigné comme absence | `[résumé moteur]` |
| GitHub `tokenosopher/populus-evidence-partitioning` ; HF `tokenosopher/populus-evidence-partitioning-checkpoints` | Marincat, 2026 | artefacts de 2608.20054 et 2609.11365 | code MIT, données CC BY 4.0, préenregistrement, 20 checkpoints finaux + cartes d'alignement P1 ; seconde nature (code et données) pour les lignes tenues par Marincat | README lu ; rien relancé |

Lectures ciblées du plan (sources déjà dans la note, lues plus loin à ce
tour ; détail en § 7 de la note) : 2610.37849 (PoPLi, § 3.1 fin à § 6),
2607.26773 (Zhang & Emu, travaux liés à conclusion), 2609.11365 (Marincat,
§ 5-§ 8 et reproductibilité), 1912.05676 (Eccles, § 3 fin, § 4-§ 5).

Non gardés mais relus en résumé (trace) : 2003.02979 (Other-Play,
méthode de coordination zéro-coup, référence de Chauhan & Bera) ;
2506.06609 (recollage de modèles, SAE) ; 2610.33020 (confusion soi/autre
dans des caches partagés, LLM) ; 2608.13317 et 2608.11676 (ponts latents
sans entraînement ou entre familles de LLM, pas d'audit causal) ;
2502.07552 (traduction non supervisée de communication émergente) ;
doi:10.1109/icons62911.2024.00055 (SNN évolués homogènes, essaim) ;
doi:10.1109/icmi65310.2025.11141291 (méta-RL, protocoles émergents) ;
doi:10.1007/s10458-024-09665-6 et doi:10.5220/0013383400003905 (titre seul,
pas de résumé OpenAlex).

### Ce que le tour a changé dans la note (v0.4)

- § 2 : quatre sources passent de « lues en partie » à « corps lu en
  entier » (PoPLi, Zhang & Emu, Marincat, Eccles) ; deux sources ajoutées
  « lues en partie » (2608.20054, 2609.17637).
- § 3 : l. 2 et l. 3a reçoivent la décomposition présence / contenu de
  Zhang & Emu lue (GSM8K, Qwen3-4B : −1,00 pp = −6,17 pp autre-exemple
  + 5,17 pp contenu) ; l. 4 reçoit le message **constant** imposé d'Eccles
  (Table 3), les transplants et la cellule « dérangée » préenregistrée de
  Marincat (parent et confirmation), avec code et checkpoints publics ;
  l. 7 (cross-play) passe à **recoupé à la lettre** (article + code et
  données, dépôt vérifié par API, contenu non relu) ; l. 10 (positive
  listening partenaire contre étranger) : résultats de PoPLi lus
  (Δ_PL = 0,615 [0,453 ; 0,751] avec code, −0,027 [−0,112 ; 0,044] sans ;
  agents LM, émetteur scripté : « pas encore pour des codes appris »).
- § 4.1 : Marincat (parent et confirmation) combine plan apparié par
  initialisation × ordre, nul par retournement de signe dans les strates et
  intervention sur paquets d'agents dont l'interface est apprise ; mais
  aucun bras entraîné sans canal, backbone gelé, supervision par entropie
  croisée, paquets continus. L'absence du dessin complet est constatée sur
  9 sources lues et 7 canaux.
- § 4.2 : rien de neuf ne touche le cœur (2512.05654, ICONS 2024 : ne
  touchent pas) ; marque maintenue.
- § 4.4 : **la marque « chiffre » est levée dans sa formulation
  reformulée** : Eccles 2019 § 4.1 Table 1 (lu) donne un second résultat
  chiffré indépendant (0/50 runs sans biais apprennent à utiliser un canal
  nécessaire, IC de Wilson 0,00-0,07 ; récompense 0,1 = niveau sans
  communication ; Treasure Hunt : 28 % de runs, IC 0,18-0,42) ; la valeur
  p = 0,400 et la bimodalité restent propres à Chauhan & Bera.
- § 5 : deux points à discuter ajoutés (message constant d'Eccles comme
  précédent du « message gelé » ; retournement de signe par strate de
  Marincat comme gabarit du nul apparié).
- § 6 : marques « à recouper » 3 → 2 dans le décompte de la note (5 → 3
  dans le décompte des tours du 06-07/11) ; couverture 6 canaux sur 6 ;
  saturation non déclarée.

### Saturation

Entrées nouvelles pertinentes gardées ce tour : 12 (10 identifiants
résolus + 2 dépôts), sur 427 identifiants nouveaux examinés
(54 du snowball, 107 hors filtre du tour 2, 80 OpenAlex, 80 titres Crossref
relus, 8 Europe PMC, 143 (98 DOI distincts) bioRxiv). **Deux** entrées nouvelles
changent une phrase de la note (2608.20054, 2609.17637), soit
0,5 % ; le reste de ce que le tour change vient des lectures
intégrales de sources déjà gardées (Eccles, PoPLi, Zhang & Emu, Marincat).
Troisième tour consécutif sous 5 % pour les entrées nouvelles du snowball
(tour 2 : 0,6 % ; 07/11 : 0,8 % ; ici 1 sur 54, 1,9 %), mais chaque tour a
encore changé la note : **non saturé**.

### Non résolus / inaccessibles

- OpenAlex : 4 requêtes `search` sur 6 en HTTP 503 (3 essais à 5 s) ;
  l'API `works/<DOI>` a répondu.
- Semantic Scholar : 2 réponses 429 sur 6 appels, reprise à 5 s, tout a
  répondu.
- bioRxiv : l'API `details` sans filtre de catégorie (192 520 notices sur
  la fenêtre, 30 par page) était trop lente (deux semaines de 2024 en six
  minutes) ; abandonnée au profit du balayage par catégorie. Les catégories
  non listées ci-dessus (biologie cellulaire, microbiologie, génomique…)
  n'ont pas été balayées.
- Version en actes de Chauhan & Bera (`OAmBy25HHg`) : non redemandée.
- Annexes non lues : PoPLi A-E ; Eccles A-E ; 2608.20054 A-F ; 2609.17637
  § 1-4, § 7-9 et annexes.
