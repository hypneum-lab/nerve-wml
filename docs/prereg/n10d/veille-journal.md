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
