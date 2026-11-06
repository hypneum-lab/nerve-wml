# Synthèse — contradicteur aveugle N10-D (2026-11-06)

Instantané : `hypneum-lab/nerve-wml` @ `6e682cc0bb67b14d0854d2eb10242a04e0607d12` (branche PR #100, HEAD détaché, refs/remote purgés, 6 fichiers NH-01/brainstorm retirés de l'arbre ; cf. `snapshot-exclusions.md`). Dépôt non modifié par les deux passes : `git status` donne toujours les 6 suppressions de l'instantané et rien d'autre.

## État des passes

| passe | état | horaires | rapport |
|---|---|---|---|
| Codex `gpt-6.1-sol`, effort xhigh, PID 95099 | **FAIT**, sortie code 0, pas de limite d'usage | lancée 07:08:57, rapport à 07:28:40, processus terminé vers 07:29 CET (~20 min) | `contradictor-report-n10d-2026-11-06-codex.md` (5 517 mots) |
| Claude Opus 5.5 (agent) | **FAIT** | ~07:03 → 07:19 CET | `contradictor-report-n10d-2026-11-06-claude.md` |

Quota : rien n'a été atteint, donc pas de `QUOTA.md`. Le rapport Codex s'identifie comme « GPT-6, agent Codex ; identifiant précis du déploiement non exposé ». Le modèle demandé était `gpt-6.1-sol`.

Aveuglement : j'ai lu le log Codex par grep sur `reading-note`, `.github`, `pull/102`, `pr view` et `priorites-hypotheses`. On n'y trouve que la lecture du protocole et du manifeste d'exclusions, aucun accès interdit. Les deux passes signalent la même chose : l'étiquette « VARIANT (expected result, Lowe 2019) » et la phrase du catalogue font partie du protocole lui-même, ce ne sont pas des conclusions sur des données N10-D.

## Décompte par sévérité (findings numérotés, hors tableau de références)

| sévérité | Claude | Codex |
|---|---:|---:|
| BLOQUANT | 2 (1.1 nécessité tranchée par construction ; 2.1 contrôle 3 impossible sur `sym`) | 2 (B1 porte d'apprentissage impossible sur `sym` ; B2 nul non calibré pour des performances entraînées) |
| MAJEUR | 6 (1.2, 2.2, 2.3, 2.4, 2.5, 2.6) | 8 (1.1, 1.2, M1, M2, M3, M4, TBD-10, TBD-11) |
| MINEUR | 4 (1.3, 2.7, 2.8, 2.9) | 1 (bug CLI `sigma_seed.py` : `n_eval` lu dans `recs[0]` non filtré) + MAJEUR/MINEUR attribués à chaque référence du tableau §3 |
| OK | — | 1 bloc (séparation dev/confirmatoire, graines, pas de sauvetage post hoc, stubs annoncés) |

Verdicts :
- **Claude : partiellement.** La « nécessité » est analytique (1/C contre 1, soit le Pl d'Eccles 2019). Reste ouvert un test de tuyauterie et d'apprentissage PPO, à condition de recalibrer `sigma_seed` et le nul `sym`, de corriger le contrôle 3, d'unifier les critères, et de ne pas en tirer de conclusion sur N10-C sans un bras exécuté dans le harnais N10-C.
- **Codex : partiellement.** La nécessité informationnelle idéale est déjà tranchée ; l'utilité comme validation PPO + canal reste ouverte, mais la porte d'apprentissage et la calibration empêchent le dessin actuel de rendre le verdict annoncé.

## Convergences factuelles

1. **Contraste optimal calculable sans entraînement.** Les deux passes trouvent 0,75 en `asym` et 0 en `sym`, avec la même démonstration (cible uniforme parmi C = 4). La question restante porte sur l'apprentissage et le câblage. Claude le classe BLOQUANT, Codex MAJEUR.
2. **Le contrôle 3 (`eng > blind ceiling + P99`) est infranchissable en `sym`** (plafond 1,0), ce qui rend l'issue A inatteignable à la lettre. Les deux passes le classent BLOQUANT.
3. **`sigma_seed` mesuré sur `nocomm`/`asym` ne dimensionne pas D.** La variance de D vient de `eng`, avec un échec bimodal possible par graine. `required_seeds` donne k = 5 pour tout σ ≤ 0,1-0,15 et k = 18 à σ = 0,3 (même calcul dans les deux passes). Claude 2.2/2.5, Codex M1.
4. **Avec k = 5, le test des signes exige 5/5 positifs** (p = 1/32). La puissance vaut q⁵ ; les deux passes calculent ≈ 0,59 à q = 0,9.
5. **Le nul `sym` est dégénéré ou transposé de `asym`.** P99 `sym` = 0,035 à σ = 0,05, contre 0,0625 en `asym` (valeurs identiques dans les deux passes). Claude 2.3, Codex B2.
6. **La métrique « 5 derniers logs de 100 épisodes » n'est pas une Binomial(n_eval)** : recouvrement, non-stationnarité. Claude 2.7 (MINEUR), Codex M2 (MAJEUR).
7. **Incohérence entre seuil ponctuel (issues A/B/C), borne d'IC (`nh96-null` §4) et test de puissance visant D > 0.** Claude 2.4, Codex M3 (partie « Fait »).
8. **Le contrôle `msg_shuffle` et l'audit de fuite sont sous-spécifiés.** Claude 2.8, Codex M4.
9. **Antériorités communes** : Lazaridou 2017 (99-100 %), Lowe 2019 §6.2, Eccles 2019 (§4.1, Pl), EGG. Les deux passes ont lu en entier Lazaridou, Lowe 2019 et Eccles.

## Divergences factuelles (sans trancher le fond)

- **Sévérité de la « nécessité tranchée »** : BLOQUANT (Claude) contre MAJEUR (Codex). Codex insiste sur le fait que le décodeur entraîné par PPO n'est pas automatiquement optimal.
- **Nature du défaut du nul** : Claude dit « variance structurellement nulle sur `nocomm` ». Codex dit « un effet latent gaussien place une politique aveugle au-dessus de 1/C, ce qui change le modèle du jeu ; c'est peut-être conservateur en `asym` ». Les deux constats sont compatibles mais ne visent pas le même angle.
- **Constat propre à Claude** :
  - simulation de l'estimateur : σ̂ = 0 dans 57 % des pilotes, P95 0,019 ;
  - 2.6 : un succès N10-D ne remonte pas à N10-C sans bras exécuté dans le harnais N10-C ;
  - 2.9 : marge 0,30 non reliée à l'oracle.
- **Constat propre à Codex** :
  - bug CLI `sigma_seed.py` (`n_eval` pris dans `recs[0]` non filtré ; k peut passer de 5 à 48) ;
  - contre-exemple `sym` à 0,625 (symétrie des observations ≠ égalité des performances) ;
  - cadre d'équivalence (Lakens 2017) ;
  - `simple_reference.py` de JaxMARL ≠ jeu K = 16/C = 4 (3 landmarks, canal de dimension 10, `local_ratio` = 0,5) ;
  - EGG `egg/zoo/signal_game/` (Claude citait `basic_games`) ;
  - évaluation détaillée de TBD-10 et TBD-11 ;
  - les 9 tests N10-D passent (Claude ne les a pas exécutés).
- **Références propres à Codex** : Chauhan & Bera 2026 (arXiv:2609.34373, déjà dans le contexte N10-C-bis selon Codex), Zhang COLING 2025 (arXiv:2410.18806), Hofbauer & Huttegger 2008 (JTB), Lakens 2017, Bouchacourt & Baroni 2018, Główka et al. 2024, Yamamoto et al. 2026, thèse Toquebiau 2024 (HAL).
- **Références propres à Claude** : Rita et al. 2022, MADDPG 2017 (échec DQN/AC/TRPO/DDPG en locuteur-auditeur), Argiento et al. 2009, Huttegger et al. 2014 PNAS (cité par Codex comme non lu), Pawlowitsch 2008, Baronchelli et al. 2005, JaxMARL, MAPPO, et plusieurs surveys.
- **Accès arXiv** : l'API `export.arxiv.org` a renvoyé 429 à Claude, qui a résolu les identifiants par DataCite. Codex a obtenu HTTP 200 sur `arxiv.org/abs`, mais sa recherche par API a expiré.

## Identifiants cités non résolus

- J'ai recontrôlé par DataCite et Crossref, le 2026-11-06 vers 07:31, tous les identifiants cités par Codex : arXiv 2609.34373, 2410.18806, 1808.10696, 2305.05821 ; DOI 10.1016/j.jtbi.2008.07.010, 10.1177/1948550617697177, 10.18653/v1/D18-1119, 10.18653/v1/D19-3010, 10.3389/fcomp.2026.1873726, 10.1016/j.cogsys.2024.101293, 10.1016/j.neucom.2026.135272, 10.1098/rspb.2011.2181. **Tous se résolvent**, avec des titres concordants.
- Les identifiants de Claude ont été résolus pendant la passe (19 arXiv via DataCite, 3 DOI Crossref, 1 DOI OpenAlex).
- **Restent sans texte vérifié**, l'identifiant étant résolu :
  - Codex : forums OpenReview `HTfNizaxFw` et `SXcJh8hoLz` (HTTP 403), « Beyond latent codes » (ScienceDirect 403), PMC anti-robot pour rspb 2011 et PNAS 2014 ;
  - Claude : thèses HAL non ouvertes, PDF HAL de Rita (HTML anti-robot).
- DOI de thèse `10.70675/3385d0bbzc66cz4fceza9e0z8696ff61ff4e` (Toquebiau, résolu par OpenAlex selon Codex) : non recontrôlé par Claude.
