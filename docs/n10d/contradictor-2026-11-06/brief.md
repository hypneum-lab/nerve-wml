# Brief du contradicteur — N10-D (2026-11-06)

Rôle : relecteur indépendant (statistiques, ML, apprentissage par renforcement multi-agent, communication émergente, neurosciences computationnelles, physique statistique selon la campagne). Tu ne connais pas les conclusions de l'équipe et tu ne dois pas les connaître.

## Interdictions explicites

- Tu ne lis **pas** la note de lecture de la campagne (`reading-note.md`, ni aucune version), ni le journal de veille, ni les brainstorms, ni le rapport d'analyse du labo, ni `docs/research/`, ni les issues/PR qui discutent des conclusions, ni le catalogue d'hypothèses.
- En particulier : tu n'ouvres pas la PR #102 de `hypneum-lab/nerve-wml` ni la branche `docs/n10d-reading-note`, ni le dépôt `hypneum-lab/.github` (`docs/research/`, catalogue, état des lieux), ni les issues GitHub de `hypneum-lab` qui parlent de N10-D/NH-96.
- Tu ne demandes pas à l'équipe ce qu'elle a conclu. Si une conclusion du labo te parvient par accident, tu le signales en tête de rapport et tu continues sans t'en servir.
- Tu ne modifies rien dans les dépôts fournis (lecture seule). Le seul fichier que tu écris est le livrable.

## Ce que tu reçois

- Les dépôts clonés de la campagne, dans le répertoire courant : `nerve-wml/` = instantané de `hypneum-lab/nerve-wml` au commit `6e682cc0bb67b14d0854d2eb10242a04e0607d12` (branche de la PR #100, squelette L0 du protocole N10-D, HEAD détaché ; autres branches et remote retirés). Pièces centrales : `nerve-wml/docs/prereg/n10d/protocol.md`, `nerve-wml/docs/prereg/n10d/nh96-null.md`, `nerve-wml/docs/prereg/n10d/requirements-to-hash.md`, `nerve-wml/experiments/n10d/` (`spec.py`, `null.py`, `sigma_seed.py`, `interfaces.py`, `backends.py`), tests `nerve-wml/tests/test_n10d_*.py`. Contexte d'origine (campagne précédente, porte de nécessité) : `nerve-wml/docs/milestones/2026-10-25-n10c-bis-control-arms.md`. La liste des fichiers retirés de l'instantané est dans `../codex/snapshot-exclusions.md`.
- Les canaux de recherche : arXiv, OpenReview, bioRxiv, Crossref/journaux, thèses (HAL, theses.fr), code et données publics (GitHub, Hugging Face, Zenodo, OSF).
- Période de départ : 2024-01-01 (pour la veille récente ; les antériorités plus anciennes et fondatrices restent pertinentes et doivent être citées si elles tranchent la question).

## Question de la campagne (telle que posée, sans réponse)

Dans un jeu référentiel locuteur-auditeur à récompense d'équipe partagée (K = 16 cibles, C = 4 candidats, alphabet A = 16, tronc PPO commun, backends JaxMARL et EGG), le contraste de succès d'équipe apparié par graine entre un canal ingénié et un canal coupé (`eng − nocomm`) est-il grand dans la condition asymétrique, où seul le locuteur voit la cible, et nul dans sa jumelle symétrique, où les deux agents la voient ?

Points ouverts du dessin (non tranchés par le PI à la date de l'instantané, à ne pas trancher toi-même mais à évaluer) : TBD-10, règle de divergence entre backends (proposition : verdict par backend, conclusion globale seulement si les deux concordent) ; TBD-11, budget de pas et `n_eval` du pilote réduit, et adaptation exacte du tronc PPO à chaque backend.

## Ta mission

Chercher activement le papier, la thèse ou le code qui **rend la question inutile ou déjà tranchée**, ou qui montre que le dessin prévu ne peut pas la trancher. Cherche aussi, dans cet ordre : (1) une antériorité directe (même question, autre vocabulaire, autre discipline) ; (2) un résultat négatif ou un contre-exemple publié ; (3) une implémentation ou un jeu de données qui donnent la réponse sans nouveau calcul ; (4) une erreur de principe dans le dessin (nul non calibré, contrôle tautologique, fuite, critère inversé, confusion de variables).

Utilise plusieurs vocabulaires par concept ; interroge au moins arXiv, OpenReview, Crossref et une source hors anglais/hors ML (thèses HAL, physique, neurosciences, statistiques). Remonte et descends les citations des deux ou trois papiers les plus proches.

## Règles de preuve

1. Aucun identifiant de mémoire : chaque DOI / arXiv id cité est résolu par API (arXiv `https://arxiv.org/abs/<id>` avec un User-Agent neutre, Crossref `https://api.crossref.org/works/<doi>`, OpenAlex, Semantic Scholar en lot). 3 s entre deux appels arXiv. Si une API renvoie 429/403, le noter : « non vérifié ».
2. Une source est « lue en entier » seulement si tu as ouvert le PDF/HTML complet ; sinon marque [résumé moteur].
3. Chaque trouvaille : id résolu, citation exacte (page ou section), et ce qu'elle change pour la question. Si tu ne trouves rien, écris les requêtes exactes, les canaux et les dates : une absence prouvée vaut résultat.
4. Faits et inférences dans des phrases séparées. Échelle : BLOQUANT / MAJEUR / MINEUR / OK. Si invérifiable, dis-le.

## Livrable

Rapport en français, sans modifier les dépôts, dans `../codex/contradictor-report-n10d-2026-11-06-codex.md` (en dehors de `repos/`) :

1. Trouvailles qui rendent la question inutile ou tranchée (ou « aucune », avec les requêtes).
2. Failles du dessin prévu (nul, critère, contrôles, puissance).
3. Antériorités manquantes (id résolus).
4. Ce que tu n'as pas pu vérifier.
5. Verdict : la question reste-t-elle ouverte et utile ? (oui / non / partiellement, une phrase de justification).

En tête du rapport : modèle, date, durée, ce qui a été lu en entier vs résumé, et toute conclusion du labo qui te serait parvenue par accident.

Le contradicteur « échoue » (au sens du verrou) s'il ne trouve rien de bloquant après une recherche documentée ; la note et les conclusions du labo ne lui sont communiquées qu'après remise du rapport.
