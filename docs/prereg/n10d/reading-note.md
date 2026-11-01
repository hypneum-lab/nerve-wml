# Note de lecture — N10-D (v0.3, 2026-11-01)

Statut : brouillon de lecture, verrou **non passé** (seul le PI le déclare,
après le contradicteur). Ce document ne modifie ni `protocol.md` (PR #100,
brouillon L0) ni aucune préreg. Il ne contient aucun verdict sur E1-small,
N10-C run 2 ou Me6, aucun tag de gel, et n'a demandé aucun calcul sur KXKM,
macM1 ou GrosMac.

Conventions : « Fait » = constat lu dans une source ou dans le dépôt ;
« Inférence » = raisonnement de l'auteur de la note, dans une phrase séparée.
`[résumé moteur]` = seul le résumé ou une notice d'API a été lu.
« à recouper » = une seule source, ou deux sources de même nature.

Références internes (chemins relatifs au dépôt indiqué) :
- `MS` = `nerve-wml:docs/milestones/2026-10-25-n10c-bis-control-arms.md`
- `P100` = PR #100, `docs/prereg/n10d/protocol.md` et `nh96-null.md` (non fusionnés)
- `RAP` = rapport d'analyse profonde du 01/11, `synthese/rapport.md`
  (dossier `.github`, PR #23)
- `VEI` = `veille/veille-2026-11-01.md` ; `RDF` = `veille/revue-de-fond-2026-11-01.md`

Historique : v0 (tour 1 : arXiv, OpenReview, Semantic Scholar) ; v0.2 (tour 2 :
OpenReview, bioRxiv, thèses/HAL, code et données, snowball sur 6 sources de
plus ; détail dans `veille-journal.md`) ; v0.3 (lecture intégrale d'OpenReview
hukTsKYu8Y, pp. 1-38, PDF fourni par le PI le 01/11 à 22:26, empreinte
SHA-256 commençant par `ecd4d37e3da6141a` ; le PDF n'est pas copié dans le
dépôt).

## 1. Question que la campagne prétend trancher (une phrase)

Un message devient-il nécessaire et utilisé quand, par construction, l'auditeur
ne peut pas réussir sans une information que seul l'émetteur détient, autrement
dit `eng − nocomm` est-il grand sur la tâche asymétrique et nul sur sa jumelle
symétrique (`P100` §1) ?

Fait : `P100` §1 qualifie lui-même NH-96 de « VARIANT (expected result, Lowe
2019) » et de « test of the task and of the channel plumbing ».
Inférence : avec un émetteur `eng` à code idéal figé, D_asym ≥ 0,30 est attendu
par construction ; la question scientifiquement ouverte n'est pas celle-là mais
l'usage **causal** du canal par le bras `learned`, qui n'est que secondaire dans
`P100` §3.

Fait (tour 2) : le même découpage « canal nécessaire par construction contre
canal non nécessaire » est la variable manipulée par Cheng et al. (régime
calibré où le récepteur ne peut pas résoudre seul : 100 % contre 23-25 % pour
des relais sans contenu pertinent ; régime naturel où il peut : effet d'appariement
borné à ±2,8 points). Inférence : « attendu par construction » est le
comportement publié d'un canal nécessaire, ce qui conforte l'écart entre la
question de `P100` §1 et la question ouverte.

Fait (v0.3) : hukTsKYu8Y démontre qu'avec un canal qui est la seule route
entre modules, couper le canal produit un déficit garanti dès que la tâche est
apprise (Proposition 1, p. 4) ; les auteurs en concluent que ce déficit
« certifie que l'information passe par le workspace, mais pas ce qui passe »
(p. 4) et ne s'appuient pas dessus (p. 9). Inférence : c'est le même statut que
`eng − nocomm` sur `asym` dans `P100` §1 (attendu par construction), formulé
cette fois comme un énoncé démontré ; la question d'usage causal reste celle
qui compte.

## 2. Sources primaires lues en entier

Tour 1 (PDF ouverts, pages indiquées) :

| id résolu | auteurs, année | pages lues | ce qu'elle établit |
|---|---|---|---|
| arXiv:1903.05168 | Lowe, Foerster, Boureau, Pineau, Dauphin, 2019 (AAMAS) | 1-13, annexes comprises | Distingue positive signaling et positive listening ; la cohérence de l'émetteur (SC) et l'entropie ne prouvent pas l'usage ; le CIC mesure l'influence causale ; Table 2 : SC avec messages brouillés ≈ SC par défaut (2×2 : 0,198 contre 0,202) ; §6.2 : couper le canal en test est peu fiable (décalage de distribution), il faut entraîner de zéro avec et sans canal. |
| arXiv:1812.09755 | Singh, Jain, Sukhbaatar, 2019 (ICLR) | 1-15 | IC3Net : porte de communication apprise ; traffic junction à vision nulle « non soluble sans communication » : IRIC 29,8 / 3,4 / 35,0 contre IC3Net 93,0 / 89,3 / 72,4 (5 runs). |
| arXiv:2107.02367 | Liu, Lamb et al., 2021 (NeurIPS) | 1-29 (corps, annexe E ; preuves de l'annexe D survolées) | Communication discrète (VQ) entre spécialistes ; VQ > Gumbel ; β = 0,25 ; limite §5 : le gain n'apparaît que si la communication entre spécialistes est importante pour la tâche. |
| arXiv:2405.08550 | Hu, Shen, Zhang, Tao, 2024 (ICLR) | 1-16 | CommFormer : graphe de communication appris ; sparsité S = 0,4 ≈ graphe complet ; Fig. 1 : grande variance des architectures prédéfinies sur 10 graines. |
| arXiv:2609.27252 | You, Jang, Mo, Jung, 2026 | corps + annexe B (33 p.) | Grille structure × géométrie ; calibration par permutation tenant compte de l'agrégation (K = 500, BH) ; structure relationnelle convergente, géométrie faible ; Spearman 0,968 entre mKNN et H0. |
| arXiv:2610.08612 | Agliari, Lepre, Roscani, 2026 | corps (33 p.), preuves survolées | RBM modulaire ≡ Hopfield couplés ; Thm 1 : point fixe CD-1 à w = η̄, dérive bornée ; désenchevêtrement numérique (L = 3, N = 5000). |
| arXiv:2609.32605 | Agliari, Alessandrelli, Barra, Fachechi, 2026 | corps (31 p.), annexe A en partie | Mémoires associatives modulaires denses (P = 4, D = 8, K = Θ(N)) appliquées à un canal bruité ; code hebbien **fixe** (rien d'appris) ; tolère 80 % d'effacement et ≈ 25-30 % de bits inversés ; taux de code ≈ 0,034. |

Tour 2 (PDF ouverts le 2026-11-01 ; pages lues dites exactement) :

| id résolu | auteurs, année | pages lues | ce qu'elle établit |
|---|---|---|---|
| arXiv:2608.04893 | Cheng, Das, Ramnath, 2026 (v2) | 1-10 (corps) sur 26 ; annexes et références survolées | Audit causal de relais de caches KV : bras vrai / dérangé / zéro / aléatoire à moments égalés ; dérangement = permutation sans point fixe qui conserve la marginale, donc ne peut pas produire d'effet par décalage de distribution (§3, identification) ; régime calibré 100 % contre 23-25 % (Qwen3-8B, 3 graines de décodage), régime naturel : équivalence à ±2,8 points (TOST, 5 graines, Holm) ; sur MedQA zéro coûte 14,7 points, dérangé 0,4 ; trois canaux livrés donnent plafond / partiel / aucun transfert (C2C). Les « graines » sont des graines de décodage du récepteur, pas des entraînements indépendants. |
| arXiv:2610.28905 | Gazzaev, Gavrilov, Muravyov, Makarov, 2026 (atelier DiffuLM, NeurIPS 2026) | 1-9 (texte et références ; pas d'annexes dans le PDF) | Canal appris supervisé entre agents LM à diffusion (backbones gelés) ; dérangement retire +63,6 points (Tiny-A2D) et +100,0 (Dream-7B) ; substitution ciblée : le récepteur suit le fait substitué (72,9 % ; 100 %) ; QASC : apparié 80,8 %, mauvais exemple 46,8 % (effet de contenu +34,0, IC [+30,7 ; +37,2]) ; inférence bootstrap apparié **sur les exemples**, pas sur des graines d'entraînement. Le dépôt de code `alanrbtx/hidden-messages-diffusion-agents` (licence non déclarée, prédictions par exemple publiées, aucune expérience relancée selon son README) existe. |
| arXiv:2609.34373 | Chauhan & Bera, 2026 (v2 ; soumission ICLR 2027, OpenReview OAmBy25HHg) | 1-9 (corps) sur 17 ; annexes survolées | Valeur d'un canal selon le couplage physique ; 25 graines par arène, tests appariés par graine, Holm, bornes unilatérales ; couplage rigide : aucun canal ne bat le silence (+0,001 ± 0,001, p = 0,982) ; couplage partiel : émetteur appris à 2 bits 0,482 (IQM) indistinguable du silence (p = 0,400) alors que l'émetteur conçu atteint 1,000 ; démarrage à chaud 0,857 contre 0,562 ; cross-play 0,980 → 0,144 sur 600 paires ordonnées. |
| arXiv:2610.37849 | Denamganaï, 2026 (PoPLi) | 1-6 (introduction, travaux liés, méthode, début du montage) sur 17 ; résultats et annexes non lus ici (le résumé donne les chiffres) | Test comportemental par paraphrase : partenaire contre étranger placés dans le même siège ; reprend la recommandation de Lowe §6.2 de ne pas mettre le message à zéro ; code publié `Near32/PartnerOutsiderPositiveListening` (MIT, README lu). |
| arXiv:2609.11365 | Marincat, 2026 | 1-8 (résumé, introduction, cadre, §4 P1) sur 17 ; §5-§7 et annexes survolés | Audit d'interopérabilité entre 6 sociétés de cellules LM entraînées indépendamment, paquets continus : 26 directions inter-initialisations sur 26 échouent à tous les échelons d'alignement ; deux sociétés de même initialisation sont interopérables à 1,000 ; contrôles par dérangement et par décalage ; protocole scellé et préenregistré. |

Complément v0.3 (PDF fourni par le PI, lu avec l'outil de lecture, rien
exécuté depuis le fichier) :

| id résolu | auteurs, année | pages lues | ce qu'elle établit |
|---|---|---|---|
| OpenReview hukTsKYu8Y | auteurs masqués, ICLR 2027 (soumission, relecture en double aveugle) | pp. 1-38 lues en entier (corps pp. 1-9, références pp. 10-13, annexes A-H pp. 14-38), PDF fourni par le PI 01/11 22:26 | Modèle « AFGW » de 6 891 784 paramètres **entraîné de zéro** (pas un LLM pré-entraîné) sur des mondes relationnels synthétiques, supervision par les seuls jetons de réponse (p. 3, p. 17-20) ; quatre modules récurrents de même architecture (style GRU), deux lecteurs de mémoire disjoints, un workspace compétitif dont le paquet continu de 640 dimensions est la **seule** route entre modules (p. 3-4, Table 2 p. 18). Protocole en échelons : présence, dépendance (OFF = broadcast mis à zéro), usage du contenu (transplant du paquet d'un donneur, cas choisis symboliquement pour que X, Y, Z soient distincts : ignoré / relais / calcul), mécanisme (restauration des requêtes natives du récepteur = nécessité ; transplant des seules requêtes = suffisance ; bruit de même norme), étendue (une source, trois récepteurs) (p. 2, p. 4-5). Contrôles : auto-remplacement, mêmes faits dans un autre ordre, source non pertinente, bruit de paquet de même norme, écriture jamais lue r = 3 (p. 5, p. 29-30). Confirmation **gelée et hachée localement, non préenregistrée publiquement** (p. 23) : 2 corpus × 2 initialisations = 4 modèles (A1, A2, B1, B2), panneaux de 512 cas (p. 6, p. 23-24). Résultats : OFF −41,41 à −51,56 **points** en composition, 0,39 à 1,37 point en local (p. 6, Table 3 p. 21) ; transplant → hybride Z dans 60,9-73,0 % (312-374/512) contre 4,9-12,7 % sans intervention (p. 7, Table 6 p. 24) ; restauration ramène Z au taux natif (réduction appariée 48,24-68,16 points) et rend la sortie native dans 2 047/2 048 cas, bruit de même norme garde 97,1-99,2 % de l'effet (p. 7, p. 22) ; transplant de requêtes seules reproduit la sortie du transplant complet dans 2 045/2 048 cas (supplémentaire, panneaux déjà vus, p. 7, p. 36) ; contraste positif dans les 64 combinaisons famille × initialisation, test du signe unilatéral p = 2^−32 en traitant les familles comme unités (p. 7, p. 23) ; variante trois récepteurs : 2 graines, familles de développement, hors étude gelée (p. 8, p. 31-33). Statistique : comptes et agrégats « descriptifs » (p. 22, p. 34, p. 36) ; **aucun bras entraîné sans canal, aucun nul par permutation d'étiquettes de bras**. Limites déclarées : suffisance au niveau de la sortie seulement, logits différents (p. 7, p. 37) ; la récupération 2 047/2 048 est un contrôle au niveau du décodage, pas une preuve de maintien entre jetons (p. 34-35) ; recombinaison au sein du même processus générateur (p. 6, p. 23) ; non établi : que la compétition ou la largeur du paquet soient nécessaires (Table 1, p. 17) ; le test ne sépare pas lectures contextuelles et lexicales, le bruit de même norme n'est qu'un contrôle d'amplitude (p. 29-30) ; restant ouvert selon les auteurs : insertion dans un Transformer pré-entraîné, ignition non linéaire, limites de capacité, maintien entre jetons, rapport (p. 9, p. 16) ; survie du mécanisme à un lecteur invariant par permutation après réentraînement (p. 38). Code : un « evidence package » et un « evidence manifest » sont mentionnés (p. 18, p. 21) mais **aucune URL, ni anonyme ni publique**, n'apparaît dans le PDF (recherche texte du PDF, 01/11). |

Correction d'une lecture `[résumé moteur]` de la v0.2 : le résumé de la page 1
écrit « lowers accuracy by 41–52% » ; le corps (p. 6) et la Table 3 (p. 21)
donnent des **points de pourcentage** (41,41-51,56 points). La v0.2 écrivait
« −41 à −52 % ».

Lues en partie seulement au tour 2 (PDF ouvert, sections dites) :

| id | pages / sections | usage dans la note |
|---|---|---|
| arXiv:1912.05676 Eccles et al., 2019 | 1-5 sur 16 (intro, cadre, §3 pertes de signalisation et d'écoute) | §3 : il existe un équilibre où le récepteur ignore le canal ; « on voit souvent les agents ne pas utiliser le canal du tout ». |
| arXiv:2607.26773 Zhang & Emu, 2026 | 1-5 sur 9 (résumé, introduction, travaux liés, début de méthode) ; résultats par recherche ciblée | §3 : audit à quatre réglages de message ; cite la préférence pour le remplacement dans la distribution (Zhang & Nanda, arXiv:2309.16042) et les risques hors distribution. |

Lue en résumé seulement :

| id | source | statut |
|---|---|---|
| OpenReview hukTsKYu8Y | « Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model », soumission ICLR 2027, auteurs masqués (relecture en double aveugle), mots-clés Global Workspace Theory, activation patching | `[résumé moteur]` : PDF (`openreview.net/pdf/c506e622…`) et `notes?id=` en 403 au tour 2 aussi ; titre et résumé via `notes/search` ; aucune version arXiv trouvée (requêtes `ti:"Does the Workspace Do the Work"` et `abs:"global workspace" AND abs:"causal protocol"`, 0 résultat chacune). Résumé : couper le broadcast −41 à −52 % sur les questions à deux modules (≤ 1,4 pt sur un module) ; broadcast substitué → combinaison dans 61-73 % des cas ; restauration 2 047/2 048 ; protocole fixé d'avance. **non vérifié** au-delà du résumé. **Remplacé en v0.3** par la lecture intégrale (pp. 1-38, tableau « Complément v0.3 » ci-dessus) ; ligne conservée pour l'historique. |

## 3. Déjà publié

| affirmation | sources (nature) | état |
|---|---|---|
| Sur une tâche où l'information est privée par construction, un canal appris bat nettement le silence. | 1812.09755 Tables (article, expérience) ; 1903.05168 §2 et §6 (article, analyse critique) ; 2608.04893 §4.1 (régime calibré, 100 % contre 23-25 %) ; 2610.28905 Table 1 (article) ; `MS` G1 et `P100` §1 qui citent l'attendu (document interne) ; v0.3 : hukTsKYu8Y Prop. 1 p. 4 et p. 6 (canal seule route entre modules : OFF −41,41 à −51,56 points en composition ; mais canal coupé en test, pas bras entraîné sans canal) | recoupé (5 articles + dépôt) ; hukTsKYu8Y **confirme** sous forme démontrée |
| Les métriques côté émetteur (SC, entropie) ne prouvent pas que l'auditeur utilise le message. | 1903.05168 Table 2 et §5 (article) ; 1912.05676 Eccles 2019 §3 (article, autre équipe : équilibre où le récepteur ignore le canal) ; 2608.04893 (« un gain de benchmark n'établit pas la transmission », article empirique) ; v0.3 : hukTsKYu8Y pp. 1-2 et p. 9 (présence ≠ usage ; une architecture rend l'usage possible sans le produire) | recoupé ; Eccles lu en partie (pp. 1-5) ; hukTsKYu8Y **confirme** (côté architecture plutôt que côté émetteur) |
| Couper le canal en test est un test biaisé ; il faut des bras entraînés avec et sans canal. | (a) couper/mettre à zéro n'est pas un test du contenu : 1903.05168 §6.2 (article, décalage de distribution) ; 2608.04893 §3 et MedQA (zéro −14,7 points contre dérangé −0,4 : effet de présence sans effet d'appariement) ; 2610.37849 §2 (reprend la recommandation de Lowe ; dépendant) ; 2607.26773 (cite la préférence pour le remplacement dans la distribution, pp. 1-5). (b) bras entraînés de zéro avec et sans canal : 1903.05168 §6.2 (recommandation) ; 2609.34373 (silence entraîné comme condition à part, 25 graines, pratique). v0.3, pour (a) : hukTsKYu8Y Prop. 1 p. 4 et preuve p. 29 (quand le canal est la seule route, la coupure renseigne sur la route, pas sur le calcul ; argument par construction, distinct du décalage de distribution) ; pour (b) : hukTsKYu8Y n'entraîne aucun modèle sans canal (ne touche pas) | (a) **recoupé**, hukTsKYu8Y **confirme** par un troisième argument ; (deux articles, deux équipes, expériences différentes ; nuance : Cheng parle d'effet de présence, Lowe de décalage) ; (b) recoupé en pratique, pas en énoncé |
| Le dérangement / la substitution / la restauration de messages comme protocole causal d'usage du canal. | 1903.05168 (CIC, messages brouillés) ; 2608.04893 (dérangé, zéro, aléatoire ; lu pp. 1-10) ; 2610.28905 (dérangé, substitution ciblée, zéro, aléatoire, auto-substitué ; lu pp. 1-9, code publié) ; 2607.26773 Zhang & Emu (4 réglages ; lu pp. 1-5) ; 2609.11365 (transplantation, dérangement, décalage ; lu pp. 1-8) ; v0.3 : hukTsKYu8Y lu en entier (OFF, transplant à trois valeurs, restauration des requêtes, transplant des requêtes seules, bruit de même norme, plan 2 × 2 et décomposition exacte Prop. 4 ; pp. 4-5, 7, 35-37 ; restauration 2 047/2 048, p. 7 et p. 22) | **recoupé en méthode** ; cinq sources lues, dont une avec code ; la restauration est désormais lue en entier dans hukTsKYu8Y (restauration des **requêtes du récepteur**, pas du message) ; hukTsKYu8Y **dépasse** les autres sources sur le volet mécanisme (voir ligne suivante) |
| Un plan 2 × 2 (broadcast × requêtes du récepteur) avec restauration (nécessité) et transplant des requêtes seules (suffisance) localise la voie par laquelle un contenu transmis agit. | hukTsKYu8Y pp. 5, 7, 35-37 (article seul ; aucun code public : « evidence package » cité p. 18 et p. 21 sans URL) | **à recouper** (une seule source ; nouvelle ligne v0.3) |
| Sous couplage rigide un canal appris peut ne rien apporter par rapport au silence. | 2609.34373 §4.1 (lu pp. 1-9 : +0,001 ± 0,001, p = 0,982, n = 25 ; couplage partiel : appris indistinguable du silence, p = 0,400) ; 2608.04893 §5 (régime naturel : l'appariement du cache vaut 0 à ±2,8 points quand le récepteur résout seul) ; `MS:326-328` (gtm_mis 0,81 contre gtm 0,54-0,73, une graine, non répliqué) | **recoupé** (MARL à contrôle continu et relais LLM, deux équipes, deux régimes) ; `MS` reste une observation à une graine |
| Les protocoles appris par des entraînements indépendants ne sont pas mutuellement lisibles (le cross-play s'effondre). | 2609.34373 §4.5 (0,980 → 0,144 sur 600 paires ordonnées, le meilleur alignement construit ne récupère que 17 % de l'écart) ; 2609.11365 §4 (26 directions inter-initialisations sur 26 échouent à tous les échelons ; seules les paires de même initialisation sont interopérables) | **recoupé** (deux articles, tâches et substrats différents ; Marincat lu pp. 1-8 ; initialisation et ordre des données distincts de la graine seule) |
| Messages discrets appris entre modules ou agents spécialisés. | 2107.02367 (article, modules spécialistes) ; 2609.34373 Table 2-3 (canal VQ de 2 bits, 25 graines, 0,983 contre silence 0,000 sur l'arène non couplée : un canal discret par VQ transmet quand il est nécessaire) | **recoupé, périmètre réduit** : agents MARL, pas modules spécialistes Transformer ; hukTsKYu8Y ne touche pas cette ligne (paquets **continus** de 640 dimensions, Table 2 p. 18) ; la veille interne `VEI:70-105` dérive de l'article et ne compte plus |
| Un code fixe décodé par attracteurs résiste à un fort bruit de canal sans aucun apprentissage. | 2609.32605 (article ; lu) ; 2610.08612 (article, même groupe) ; 1407.6029 Fiete, Schwab, Tran `[résumé moteur]` (autre groupe : réseaux de Hopfield binaires construits comme décodeurs de tout code neuronal bruité, taux d'erreur asymptotiquement nul à taux d'information 1/log n) | **recoupé** par un groupe indépendant, mais le second groupe en résumé seulement et sur un réseau **construit** (non appris) ; 10.1038/339693a0 Sourlas 1989 (notice Crossref seule) n'est qu'un précédent de principe, non utilisé |
| Positive listening partenaire contre étranger comme test d'usage. | 2610.37849 Denamganaï 2026 (lu pp. 1-6) ; 1903.05168 §3 (définition) ; dépôt `Near32/PartnerOutsiderPositiveListening` (code, README lu) | recoupé sur la notion et la méthode de remplacement ; résultats de PoPLi non lus ici |
| Une mesure d'usage fondée sur le zéro reste pratiquée. | `SudarshanHegde326/emergent-comm-usage-marl`, `scripts/analysis/spread_ablation.py` (mémoire MSc, 3 graines, ablation par message mis à zéro) `[README et début du script lus]` | observation de pratique, une source de code ; ne pas en tirer de conclusion |

v0.3, lignes non touchées par hukTsKYu8Y (fait, après lecture des pp. 1-38) :
couplage rigide / canal sans valeur (le papier n'a pas de condition où le canal
est inutile, la composition l'exige par construction, p. 3) ; cross-play (deux
initialisations par corpus, mais aucun appariement émetteur-récepteur croisé,
p. 23) ; code fixe décodé par attracteurs ; positive listening partenaire
contre étranger ; pratique du zéro (le papier utilise OFF comme échelon 1 en le
déclarant insuffisant, p. 4).

## 4. Reste ouvert

1. **Usage causal du canal `learned` sur une tâche asymétrique nécessaire par
   construction, avec un nul qui tient compte de la covariance de graine.**
   Preuve de l'absence (partielle) : `P100` §2 simule les deux bras
   indépendamment par graine (`nh96-null.md` §2, « Draw per seed and arm ») ;
   `RAP:94` (M9) relève l'absence de covariance. Dans la littérature lue :
   Lowe 2019 et IC3Net rapportent des moyennes sur graines sans nul apparié
   par graine (tour 1) ; au tour 2, Chauhan & Bera apparient les **tests**
   par graine (25 graines, Holm) mais ne permutent pas de messages sur des
   agents appris ; Cheng et al. et Gazzaev et al. permutent des messages mais
   sur des émetteurs et récepteurs **gelés** (les « graines » sont des graines
   de décodage ou l'inférence est un bootstrap sur les exemples) ; Marincat
   dérange des paquets sur des sociétés entraînées mais sur un point
   (checkpoint) par société, sans nul de permutation d'étiquettes de bras.
   Aucun résultat de bioRxiv, de HAL/theses.fr ni de HF/Zenodo/OSF ne contredit
   cette lecture. Fait : la méthode appariée (par graine) et l'intervention
   (message permuté) existent séparément, trois groupes différents.
   Inférence : leur combinaison, avec un nul par permutation des bras à
   l'intérieur de chaque graine sur des agents entraînés de zéro, n'a pas été
   trouvée. **recoupé en absence sur 6 canaux** (voir § 6 pour les limites),
   mais l'absence reste une inférence ; tout nouveau résultat sur ce point
   change la question.
   v0.3 — Fait : hukTsKYu8Y (lu pp. 1-38) applique OFF, transplant et
   restauration sur des modèles **entraînés de zéro** (4 initialisations en
   confirmation, p. 23), ce que Cheng et al. et Gazzaev et al. ne faisaient
   pas. Fait : il n'entraîne aucun bras sans canal, ne permute pas
   d'étiquettes de bras, et son inférence est un test du signe sur 64
   combinaisons famille × initialisation où les familles sont les unités
   (p. 7, p. 23) ; il qualifie lui-même ses regroupements de descriptifs et
   rappelle que deux initialisations d'une même famille ne sont pas
   indépendantes (p. 34, p. 36). Inférence : le papier **dépasse** la
   littérature lue sur « intervention de message sur des agents entraînés de
   zéro », mais **ne touche pas** le nul apparié par graine sur des bras
   entraînés avec et sans canal ; l'absence est désormais constatée sur 7
   sources lues et 6 canaux.
2. **Le même test sur des neurolettres discrètes entre substrats hétérogènes
   (MLP/LIF) et au chemin de réception γ/θ.** Preuve de l'absence : aucune des
   13 sources snowballées (7 au tour 1, 6 au tour 2 ; 668 + 114 citants en
   aval) n'a un titre portant sur ces substrats (filtre par titre
   seulement) ; OpenReview (7 requêtes dont « discrete messages heterogeneous
   modules spiking communication ») : seul `spio7kWIWr` (« Calibrated Spiking
   Messages for Emergent Multi-Agent Communication », soumission ICLR 2026,
   `[résumé moteur]`) échange des messages impulsionnels dans un jeu
   référentiel (Fashion-MNIST) avec encodeur partagé ; son résumé ne mentionne
   aucune intervention causale sur le message ni substrats hétérogènes.
   **à recouper** : seuls le filtre par titre et un résumé OpenReview
   étayent l'absence ; aucune lecture intégrale ne la confirme.
   v0.3 — Fait : le concurrent le plus proche, lu en entier (hukTsKYu8Y),
   utilise quatre modules de **même** architecture récurrente (style GRU ;
   seuls l'accès mémoire et le rôle diffèrent, p. 3, p. 17-19) et un paquet
   **continu** (640 dimensions, Table 2 p. 18) ; ni MLP contre LIF, ni
   impulsions, ni messages discrets, ni chemin de réception γ/θ. Inférence :
   ce papier **ne touche pas** 4.2 ; il ne contredit pas l'absence mais ne la
   prouve pas non plus (une lecture intégrale, même nature que le résumé
   OpenReview) ; la marque **à recouper** reste.
   Inférence sur la portée : AFGW n'est pas un grand LLM pré-entraîné mais un
   petit modèle (6,9 M paramètres) entraîné de zéro en 22 minutes environ par
   modèle (p. 20, Table 13 p. 28), ce qui le rapproche de N10-D plus que le
   titre ne le laisse penser ; les écarts qui restent sont la supervision
   (entropie croisée sur jetons de réponse, pas d'apprentissage par
   renforcement), le canal continu au lieu de neurolettres discrètes, et
   l'homogénéité des modules.
3. **Claim B (accord de classifieurs) contre usage du canal.** Fait : `RAP:56`
   (B3) indique que les codes « post-VQ » sont l'argmax de la tête π et que
   MI(code ; XOR) ≈ 0,001 nat selon un rejeu d'agent non reproductible depuis
   le dépôt. Inférence : la question « le canal porte-t-il l'information
   nécessaire » n'est pas tranchée par Claim B. Question interne, pas de
   littérature attendue.
4. **Nouveau au tour 2 : un récepteur peut ne pas utiliser un canal appris
   même quand il est nécessaire.** Fait : Chauhan & Bera, couplage partiel,
   émetteur appris à 2 bits indistinguable du silence (p = 0,400, n = 25)
   alors que l'émetteur conçu au même débit atteint 1,000 ; Eccles 2019 §3
   décrit l'équilibre « récepteur indépendant du canal ». Inférence : le bras
   `learned` de N10-D peut échouer à découvrir un protocole sans que la
   tâche ou le canal soit en défaut ; le critère de réfutation du § 5 doit
   distinguer « canal inutilisé » de « canal inutilisable ». Une source
   (Chauhan) en lecture partielle + un précédent théorique (Eccles) :
   **à recouper** pour le chiffre, recoupé pour l'existence du phénomène.
   v0.3 — Fait : dans hukTsKYu8Y le canal appris est utilisé dans les quatre
   modèles de confirmation (p. 7) ; les auteurs écrivent que l'architecture
   rend l'usage possible sans le produire et que le protocole aurait pu le
   réfuter (p. 9). Inférence : le papier **ne touche pas** le chiffre de 4.4
   (aucun cas de canal inutilisé n'y est rapporté) ; il énonce seulement la
   possibilité ; la marque reste.

Ce qui n'est **pas** ouvert : `eng − nocomm` sur asym avec émetteur conçu
(`P100` §1 le dit lui-même).

## 5. Dessin minimal qui tranche la question ouverte

Détail et alternatives : `brainstorm-2026-11-01.md`. Dessin minimal retenu
pour discussion (rien ne part avant le verrou) :

- **Bras.** Sur la tâche `asym` de `P100` (K = 16, C = 4, A = 16), bras
  `learned` et `nocomm` **entraînés de zéro**, appariés par graine (Lowe §6.2) ;
  sur chaque agent `learned` entraîné, interventions en test : message permuté
  entre épisodes, message gelé (constant), canal mis à zéro, plus restauration.
  `eng` et `msg_shuffle` restent des contrôles de chaîne.
- **Nul.** Permutation des étiquettes de bras **à l'intérieur de chaque graine**
  (nul apparié qui conserve la covariance de graine), plus le nul substitut
  binomial de `P100` §2 recalibré avec une composante de graine commune aux
  deux bras. `sym` sort de H1 (décision PI) et reste un contrôle de fuite.
- **Critère.** Usage causal soutenu si la chute permuté-contre-intact sur
  `learned` dépasse le P99 du nul apparié **et** si la restauration ramène la
  performance intacte ; réfuté si l'IC 95 % apparié de la chute contient 0
  alors que `learned − nocomm` dépasse le P99 (gain sans usage du message).
- **Coût.** Non mesuré. Inférence : un jeu référentiel K = 16 sous JaxMARL ou
  EGG est d'un ordre de grandeur bien inférieur à un run Procgen N10-C
  (≈ 10,8 h par run sur KXKM, `MS:179`). Hypothèse de travail à mesurer au
  pilote : ≤ 1 h GPU par run, ≈ 10-20 h GPU au total ; ≈ 15-25 h humaines
  (environnements réels manquants, `P100` §6).

Points à discuter issus du tour 2 (inférences, rien n'est modifié dans le
dessin) :
- Cheng et al. rapportent que « zéro » mêle présence et contenu (MedQA : zéro
  −14,7, dérangé −0,4). Inférence : la modalité « canal mis à zéro » du bras
  d'intervention ne peut pas, seule, soutenir un énoncé sur le contenu ; le
  dérangement conserve la marginale et reste le contraste de contenu.
- Cheng et al. ajoutent un bras « intact sans information privée » qui sépare
  le contenu pertinent du calcul générique de l'émetteur. Inférence : un
  analogue pour N10-D (émetteur `learned` fonctionnant sur une charge utile
  non pertinente) est une option de contrôle, à chiffrer au brainstorm.
- Chauhan & Bera : un `learned` indistinguable du silence est un résultat
  publié possible (voir § 4.4). Inférence : prévoir dès la préreg un seuil de
  « canal inutilisé » séparé de « canal inutilisable » (démarrage à chaud ou
  récepteur conçu comme référence).
- Gazzaev et al. et Cheng et al. inférent sur les exemples avec des
  backbones gelés ; Chauhan & Bera apparient par graine sur 25 graines.
  Inférence : 25 graines est un ordre de grandeur publié pour un test apparié
  d'agents appris de zéro, à comparer à la puissance visée par `P100`.

Points à discuter issus de la lecture de hukTsKYu8Y (v0.3 ; inférences, rien
n'est modifié dans le dessin) :
- Fait : le papier fait déjà « coupé » (OFF), « substitué depuis un autre
  input » (transplant, proche du message permuté) et une restauration (p. 4-7).
  Fait : il ne fait pas de message **gelé** (constant) et sa restauration porte
  sur les requêtes du récepteur, pas sur le retour au message intact.
  Inférence : la nouveauté de D1 ne tient plus à l'intervention elle-même mais
  à son cadre : agents appris par renforcement, messages discrets, bras
  `nocomm` entraîné de zéro, nul apparié par graine.
- Fait : le papier choisit symboliquement des cas où réponse du récepteur,
  réponse du donneur et réponse hybride sont distinctes deux à deux (p. 5).
  Inférence : un analogue pour la tâche `asym` (épisodes où la cible du
  donneur et la cible du récepteur diffèrent et où l'on sait d'avance quelle
  sortie trahit l'usage du contenu) séparerait « relais » et « usage » dans
  le permuté ; option à chiffrer.
- Fait : leur inférence traite les familles comme unités avec un test du signe
  (p. 7) ; 4 initialisations seulement. Inférence : ce n'est pas un modèle de
  nul pour N10-D ; le nul apparié par graine reste à écrire.

## 6. Trois tests du verrou

État v0.3 (après lecture intégrale de hukTsKYu8Y) :

| test | état | ce qui manque |
|---|---|---|
| Saturation | **non déclarée.** La lecture intégrale a changé la note (§ 1, § 2, § 3 cinq lignes et une ligne ajoutée, § 4.1, 4.2, 4.4, § 5) ; aucun snowball n'a été fait sur hukTsKYu8Y (OpenReview hors S2, auteurs masqués). | Snowball des références de hukTsKYu8Y (pp. 10-13) ; un tour sans entrée qui change la note. |
| Recoupement | **inchangé en nombre de marques levées, une marque ajoutée.** Avant v0.3 : 2 marques « à recouper » (4.2, 4.4). Après : **3** (4.2, 4.4, et la nouvelle ligne § 3 « plan 2 × 2 nécessité / suffisance », tenue par hukTsKYu8Y seul, sans code public). Aucune marque n'est levée : hukTsKYu8Y ne touche ni 4.2 ni 4.4. La ligne « dérangement / substitution / restauration » ne dépend plus d'une source non vérifiée. | Code public de hukTsKYu8Y (aucune URL dans le PDF) ou une seconde source de nature différente pour le volet mécanisme ; les manques de la v0.2 restent (PoPLi résultats, Zhang & Emu pp. 6-9, Marincat § 5-7, Eccles suite, second résultat pour 4.4). |
| Couverture | **inchangée** (5 canaux sur 6) : le PDF vient du PI, pas d'une nouvelle requête. | Comme en v0.2. |

État v0.2 (conservé) :

| test | état | ce qui manque |
|---|---|---|
| Saturation | **non déclarée.** Snowball du tour 2 sur 6 sources (291 lignes dont 177 amont et 114 aval, 243 identifiants distincts, 176 hors tour 1 ; filtre par titre : 72 ; titres lus : **1 nouveau pertinent qui change la note** (arXiv:2609.11365) soit 0,6 % de 176 ; 2 (1,1 %) si l'on compte une référence méthodologique, arXiv:2206.11302) : sous le seuil de 5 % pour le snowball. Mais le tour 2 a changé la note (§ 1, § 3 lignes 3-7, § 4.1 et 4.4, § 5), et les 104 lignes hors filtre par titre n'ont pas été triées. | Un tour sans entrée qui change la note ; tri des 104 titres hors filtre ; snowball de Marincat, de Chauhan & Bera et de l'étude parent de Marincat (non résolue ici). |
| Recoupement | **partiel, amélioré.** Au tour 1 : 3 affirmations de § 3 recoupées, 5 « à recouper » (recompte : 4 recoupées et 4 « à recouper » dans le tableau § 3 de la v0, plus 2 dans § 4, soit 6 marques, pas 5). Après le tour 2 : § 3 : 10 lignes, dont 9 affirmations **recoupées** (une en pratique seulement, une à périmètre réduit, une avec second groupe en résumé) et une observation de pratique à une source ; § 4 : **2 marques « à recouper »** restent (4.2 substrats hétérogènes ; 4.4 chiffre de Chauhan) ; 4.1 recoupé en absence sur 6 canaux. Marques « à recouper » : 6 → 2. | hukTsKYu8Y en texte intégral (403 persistant, auteurs masqués) ; lecture intégrale de PoPLi (résultats), de Zhang & Emu (pp. 6-9), de Marincat (§5-§7), d'Eccles (suite) ; un second résultat indépendant pour 4.4. |
| Couverture | **partiel, 5 canaux sur 6.** Interrogés : arXiv (12 requêtes API et 8 PDF), OpenReview (7 requêtes de plus, 350 notes brutes), bioRxiv (catégorie neurosciences, 2024-01-01 à 2026-11-01 : 38 126 notices parcourues, 57 correspondances de mots-clés, 37 DOI distincts, 0 gardé), Crossref (précédent Sourlas, résolution d'identifiants), thèses/HAL (5 requêtes, 3 notices) et theses.fr (4 requêtes), code et données (GitHub 10 requêtes, Hugging Face 14 appels, Zenodo 4, OSF 3), snowball S2 (13 sources). Voir `veille-journal.md`. | bioRxiv limité à la catégorie neurosciences (les autres catégories ne sont pas balayées) ; revues (Crossref) uniquement par résolution d'identifiants, pas de balayage par mots-clés ; OpenReview limité à `notes/search` ; Hugging Face sans résultat pertinent (index peu adapté). |

## 7. Journal de lecture

| date | source | pages / sections |
|---|---|---|
| 2026-11-01 | arXiv:1903.05168 (PDF) | 1-13, annexes comprises |
| 2026-11-01 | arXiv:1812.09755 (PDF) | 1-15 |
| 2026-11-01 | arXiv:2107.02367 (PDF) | 1-29 ; annexe D survolée |
| 2026-11-01 | arXiv:2405.08550 (PDF) | 1-16 |
| 2026-11-01 | arXiv:2609.27252 (PDF) | corps + annexe B |
| 2026-11-01 | arXiv:2610.08612 (PDF) | corps ; preuves survolées |
| 2026-11-01 | arXiv:2609.32605 (PDF) | corps ; annexe A en partie |
| 2026-11-01 | OpenReview hukTsKYu8Y | titre + résumé (`notes/search`) `[résumé moteur]`, PDF 403 (tour 1 et tour 2) |
| 2026-11-01 | 8 entrées du tour 1 (voir `veille-journal.md`) | résumés arXiv `[résumé moteur]`, sauf celles lues au tour 2 ci-dessous |
| 2026-11-01 | `MS` | l. 179-202 (plan de calcul), 300-352 (B3, porte G1-G7) |
| 2026-11-01 | `P100` | `protocol.md` et `nh96-null.md` en entier |
| 2026-11-01 | `RAP` | l. 56-57, 89-94, 148-151, 242, 285 |
| 2026-11-01 | `VEI` / `RDF` | `VEI:70-105` ; `RDF:113-168, 609-620, 685+` |
| 2026-11-01 (tour 2) | arXiv:2610.28905 Gazzaev et al. (PDF, résolu par API) | 1-9 (texte, tableaux, références) |
| 2026-11-01 (tour 2) | arXiv:2608.04893 Cheng et al. (PDF) | 1-10 de 26 ; annexes et références survolées |
| 2026-11-01 (tour 2) | arXiv:2609.34373 Chauhan & Bera (PDF) | 1-9 de 17 ; annexes survolées |
| 2026-11-01 (tour 2) | arXiv:2610.37849 Denamganaï (PDF) | 1-6 de 17 ; résultats et annexes non lus |
| 2026-11-01 (tour 2) | arXiv:2609.11365 Marincat (PDF) | 1-8 de 17 ; §5-§7 et annexes survolés |
| 2026-11-01 (tour 2) | arXiv:1912.05676 Eccles et al. (PDF) | 1-5 de 16 |
| 2026-11-01 (tour 2) | arXiv:2607.26773 Zhang & Emu (PDF) | 1-5 de 9 ; résultats par recherche ciblée |
| 2026-11-01 (tour 2) | arXiv:1407.6029 Fiete et al. ; arXiv:2608.03644 Wolski et al. ; arXiv:2309.16042 Zhang & Nanda ; arXiv:2206.11302 Boldt & Mortensen | `[résumé moteur]` (notices arXiv API) |
| 2026-11-01 (tour 2) | OpenReview `spio7kWIWr`, `dz0yb4MlKd`, `OAmBy25HHg` | titres et résumés via `notes/search` `[résumé moteur]` |
| 2026-11-01 (tour 2) | dépôts GitHub `Near32/PartnerOutsiderPositiveListening`, `alanrbtx/hidden-messages-diffusion-agents`, `SudarshanHegde326/emergent-comm-usage-marl` | README ; début de `spread_ablation.py` |
| 2026-11-01 (tour 2) | thèses HAL tel-05076852 (Toquebiau 2024), tel-03536320 (Chaabouni 2021), tel-04728859 (Pontes Mota 2024) | résumés `[résumé moteur]` ; non gardées (hors sujet ou fond) |
| 2026-11-01 (v0.3, après 22:26) | OpenReview hukTsKYu8Y, PDF fourni par le PI (38 pages, SHA-256 commençant par `ecd4d37e3da6141a`) | **pp. 1-38 en entier** : corps 1-9, déclarations et références 10-13, annexes A-H 14-38 (tableaux 1-23, propositions 1-7) ; recherche texte d'URL de code dans le PDF : aucune |
