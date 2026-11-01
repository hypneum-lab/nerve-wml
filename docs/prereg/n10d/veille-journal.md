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
