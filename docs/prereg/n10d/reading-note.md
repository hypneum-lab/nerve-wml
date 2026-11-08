# Note de lecture — N10-D (v0.5, 2026-11-08)

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
dépôt) ; § 4.4 réécrit le 07/11 (PR #125) d'après les tours de veille
datés du 06 et du 07/11 (`veille/tour-1-2026-11-06.md`,
`veille/tour-2-2026-11-07.md`, après contradicteur) ; v0.4 (tour 3 du
08/11 : tri des 107 titres hors filtre du tour 2, snowball de Marincat, de
Chauhan & Bera et de l'étude parent de Marincat résolue en
arXiv:2608.20054, bioRxiv hors neurosciences par catégorie, Crossref et
OpenAlex par mots-clés ; lecture intégrale du corps de PoPLi, de Zhang &
Emu, de Marincat et d'Eccles ; détail dans `veille-journal.md`, tour 3).
v0.5 (08/11, après 16:45) : amendée en réponse au rapport du contradicteur
aveugle (Claude Opus à contexte vierge, Codex en quota ; brief sans la
note) et aux 12 décisions PI du 2026-11-08. Pièces archivées hors dépôt
(non copiées ici) : `contradictor-report-n10d-2026-11-08.md` (SHA-256
`d1e5f9c63efef3ac13306ec2388aa1bc479404294cc178ea7cb9872e54e7e696`),
`decisions-n10d-causal-2026-11-08.md` (QCM 16:30-16:45 CET),
`options-n10d-causal-2026-11-08.md`, `brief-n10d-2026-11-08.md`, dossier
`hypneum-reviews/2026-11-08-n10d-contradictor/`. Les trois identifiants
que le rapport ajoute ont été résolus de nouveau par API par l'auteur de
la note avant d'être cités (arXiv `id_list`, HAL `halId_s`, 08/11).

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

Fait (v0.4) : dans l'étude parent de Marincat (arXiv:2608.20054, § 5.1),
couper tous les paquets en test ramène les dix sociétés à visibilité
restreinte exactement au hasard (1/17), et l'auteur écrit (§ 6) que les
audits destructifs « établissent la nécessité du canal mais pas le
contenu ». Inférence : troisième formulation, par une troisième équipe, du
même statut pour `eng − nocomm` ; rien de neuf sur la question ouverte.

### v0.5 — Question tranchée, question retenue

Fait (rapport du contradicteur § 1.1 ; vérifié sur A21 § 2.3 de la
branche `prereg/n10d-skeleton`, e0c64a8) : sous un dérangement qui tire le
message d'un épisode indépendant, « E[S_msg_shuffle] = 1/C pour tout
auditeur », quoi que voie l'émetteur ; `spec.py:39-46` code ce plafond
aveugle. Fait : sur `asym`, la nécessité du message et son usage par tout
auditeur qui réussit sont donc des conséquences de la construction, non
des hypothèses à tester ; le même argument vaut pour le message figé et
le canal à zéro (options § 0). Inférence (reprise du rapport) : la seule
question empirique qui reste sur `asym` est « l'apprenant apprend-il à
décoder ? », qui est une question d'optimisation, pas d'usage causal.

Fait (arXiv:1905.13687, Kharitonov, Chaabouni, Bouchacourt et Baroni,
2020, ICML, PMLR 119 ; résolu par API le 08/11 ; § 3.3, § 4.1 et § 7 du
supplément lus par l'auteur de la note dans le texte intégral, le reste
non lu) : dans Guess Number (entier de 8 bits, le récepteur en voit k),
« a more nuanced protocol is only developed when it is necessary »
(§ 4.1) ; à k = 8, l'information que porte le message est « entirely
ignored by Receiver » : les messages brouillés entre entrées laissent la
performance inchangée (§ 4.1) ; le brouillage « preserves the marginal
distribution of Sender's messages, but destroys all the information
Sender transmits » (§ 7) ; seuls les runs à précision > 0,99 sont
analysés (§ 3.3) ; REINFORCE réussit dans ≈ 20 % des runs, Gumbel-Softmax
et SCG dans plus de 50 % (§ 4.1).

Fait (arXiv:2609.34373, Chauhan et Bera, 2026, v2 ; résolu par API le
08/11 ; corps lu au tour 2, pp. 1-9) : sur l'arène non couplée, le canal
conçu atteint 0,999 ± 0,001, le silence 0,000 (« The floor is a strategy,
not a failed learner », § 3.4), le plancher scripté sans canal 0,242 ±
0,037 et l'émetteur appris gagne +0,740 ± 0,038 contre ce plancher
(25 graines appariées). Fait : le texte intégral ne contient aucune
intervention sur le message en test (recherche du texte le 08/11 :
« ablation » et « intervention » 0 occurrence ; « shuffled » 3
occurrences, toutes sur le nul d'alignement du cross-play).

Fait (hal-03100681, Villecroze et Moulin-Frier, 2020, atelier SMILES,
ICDL 2020 ; résolu par l'API HAL le 08/11 ; **PDF de 7 pages lu en
entier** par l'auteur de la note, SHA-256 commençant par
`7e9763b6ab21872a`, non copié dans le dépôt) : grille 11 × 11, auditeur
A3C à entrée pixel ; § V-A : sans biais, aucune émergence après
3 × 10^8 pas ; § V-B, émetteur remplacé par un **oracle fixé** qui dit la
direction à prendre : « with any view size, the listener uses almost
exclusively the communication channel and neglects the visual input even
when it is able to see the target » (gradients, Fig. 3a) ; § V-C : CIC =
information mutuelle message-action calculée par contrefactuels,
« Without noise, CIC is maximal whatever the observability is », et
l'observabilité ne réduit le CIC que si le canal est bruité (Fig. 4,
10 graines) ; § V-D : avec un émetteur **appris** (MLP), l'apprentissage
conjoint ne converge pas (Fig. 5, 10 graines) ; § V-E : le biais de
signalisation positive d'Eccles le fait converger « most of the time ».

Inférence : la question de `P100` § 1 (NH-96, `eng − nocomm` grand sur
`asym`, nul sur `sym`) est tranchée par construction (A21 § 2.3) et
confirmée par trois résultats publiés de trois équipes (2020, 2020,
2026) ; la campagne ne l'adopte plus comme question primaire (décisions
PI Q8 et 1 du 08/11). Inférence : l'inférence de la v0 de cette note
(« la question scientifiquement ouverte n'est pas celle-là mais l'usage
causal du canal ») est confirmée de l'extérieur par un relecteur qui
n'avait pas accès à la note.

**Question retenue (décisions PI 1 et 2, formulation des options § A) :**

- **V1.** Sur la jumelle symétrique, où son observation propre suffit,
  l'auditeur PPO entraîné face à l'émetteur conçu fixé (`eng_sym`)
  choisit-il le candidat nommé par le message plutôt que celui que
  désigne son propre champ t, quand les deux sont mis en conflit,
  au-delà du taux placebo, avec une borne basse d'IC apparié ≥ δ_use ?
  Réfutée (« message ignoré ») si la borne haute est < δ_eq (TOST) ;
  sinon non concluant.
- **V2.** Cet indice de dépendance au message diffère-t-il entre
  émetteur fixé (`eng_sym`) et émetteur appris conjointement
  (`learned_sym`), à tronc, graines et budget égaux ?
- V1 et V2 sont co-primaires sous Holm, α = 0,01 ; l'ancienne H1 sur
  `asym` devient un contrôle positif de la machinerie, sans α
  (décision 3).

Inférence : Kharitonov (émetteur appris, k = 8 : message ignoré) et
Villecroze (émetteur oracle fixé : message utilisé même en voyant la
cible) sont les deux pôles de V2 ; aucune des deux sources ne compare
les deux émetteurs sur la même tâche, ni ne met message et observation
propre en conflit épisode par épisode. Le modérateur « émetteur fixé
contre appris » n'a été trouvé écrit nulle part : c'est une inférence
(voir § 4).

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
| arXiv:2610.37849 | Denamganaï, 2026 (PoPLi) | tour 2 : 1-6 (introduction, travaux liés, méthode, début du montage) sur 17 ; **v0.4 : corps lu en entier** (fin de § 3.1, § 3.2 méthode, § 3.3 montage, § 4 résultats, § 5 discussion et limites, § 6) ; annexes A-E non lues | Test comportemental par paraphrase : partenaire contre étranger placés dans le même siège ; reprend la recommandation de Lowe §6.2 de ne pas mettre le message à zéro ; code publié `Near32/PartnerOutsiderPositiveListening` (MIT, README lu). v0.4 : double enchère continue, agents DeepSeek v4-flash, **émetteur scripté** (code = première lettre S/L), 10 marchés par condition, T = 40 tours rejoués, bootstrap par sessions puis tours (pas de graines d'entraînement) ; Δ_PL = 0,615 [0,453 ; 0,751] avec code contre −0,027 [−0,112 ; 0,044] sans (joint reward, messages neutres) ; l'écart vient tout entier du partenaire (PL_Π(B) = 0,615, PL_Π(C₀) = 0,000) ; Table 2 : 0,590 / 0,480 / 0,525 dans les trois conditions avec code, intervalles contenant zéro dans les trois sans ; les moniteurs de messages ne séparent rien ; limites déclarées : « usable behavioural test for stimulus-level codes, but not yet for learned or emergent codes » ; aveugle aux codes portés par le sens (Prop. 5) ; un environnement, un modèle. |
| arXiv:2609.11365 | Marincat, 2026 | tour 2 : 1-8 (résumé, introduction, cadre, §4 P1) sur 17 ; **v0.4 : corps lu en entier** (§ 4 fin, § 5 P2, § 6 P3, § 7 P4, § 8 discussion et limites, reproductibilité) ; annexe A survolée | Audit d'interopérabilité entre 6 sociétés de cellules LM entraînées indépendamment, paquets continus : 26 directions inter-initialisations sur 26 échouent à tous les échelons d'alignement ; deux sociétés de même initialisation sont interopérables à 1,000 ; contrôles par dérangement et par décalage ; protocole scellé et préenregistré. v0.4 : P3 (§ 6) = factoriel apparié à trois bras sur le point de contrôle global (interface héritée gelée 0,065 ; héritée entraînable 0,169 ; interface fraîche 0,857 ; tous les bras à 0,0588 = hasard quand les paquets sont coupés) ; P4 (§ 7) : le bénéfice d'héritage préenregistré (+0,10) n'est jamais atteint, deux points de contrôle × deux flux ; limites : un seul modèle global, 17 états, familles de cartes testées seulement ; artefacts : dépôt GitHub `tokenosopher/populus-evidence-partitioning` (MIT, CC BY 4.0) et checkpoints Hugging Face (vérifiés par API le 08/11, README lu, rien relancé). |

Complément v0.3 (PDF fourni par le PI, lu avec l'outil de lecture, rien
exécuté depuis le fichier) :

| id résolu | auteurs, année | pages lues | ce qu'elle établit |
|---|---|---|---|
| OpenReview hukTsKYu8Y | auteurs masqués, ICLR 2027 (soumission, relecture en double aveugle) | pp. 1-38 lues en entier (corps pp. 1-9, références pp. 10-13, annexes A-H pp. 14-38), PDF fourni par le PI 01/11 22:26 | Modèle « AFGW » de 6 891 784 paramètres **entraîné de zéro** (pas un LLM pré-entraîné) sur des mondes relationnels synthétiques, supervision par les seuls jetons de réponse (p. 3, p. 17-20) ; quatre modules récurrents de même architecture (style GRU), deux lecteurs de mémoire disjoints, un workspace compétitif dont le paquet continu de 640 dimensions est la **seule** route entre modules (p. 3-4, Table 2 p. 18). Protocole en échelons : présence, dépendance (OFF = broadcast mis à zéro), usage du contenu (transplant du paquet d'un donneur, cas choisis symboliquement pour que X, Y, Z soient distincts : ignoré / relais / calcul), mécanisme (restauration des requêtes natives du récepteur = nécessité ; transplant des seules requêtes = suffisance ; bruit de même norme), étendue (une source, trois récepteurs) (p. 2, p. 4-5). Contrôles : auto-remplacement, mêmes faits dans un autre ordre, source non pertinente, bruit de paquet de même norme, écriture jamais lue r = 3 (p. 5, p. 29-30). Confirmation **gelée et hachée localement, non préenregistrée publiquement** (p. 23) : 2 corpus × 2 initialisations = 4 modèles (A1, A2, B1, B2), panneaux de 512 cas (p. 6, p. 23-24). Résultats : OFF −41,41 à −51,56 **points** en composition, 0,39 à 1,37 point en local (p. 6, Table 3 p. 21) ; transplant → hybride Z dans 60,9-73,0 % (312-374/512) contre 4,9-12,7 % sans intervention (p. 7, Table 6 p. 24) ; restauration ramène Z au taux natif (réduction appariée 48,24-68,16 points) et rend la sortie native dans 2 047/2 048 cas, bruit de même norme garde 97,1-99,2 % de l'effet (p. 7, p. 22) ; transplant de requêtes seules reproduit la sortie du transplant complet dans 2 045/2 048 cas (supplémentaire, panneaux déjà vus, p. 7, p. 36) ; contraste positif dans les 64 combinaisons famille × initialisation, test du signe unilatéral p = 2^−32 en traitant les familles comme unités (p. 7, p. 23) ; variante trois récepteurs : 2 graines, familles de développement, hors étude gelée (p. 8, p. 31-33). Statistique : comptes et agrégats « descriptifs » (p. 22, p. 34, p. 36) ; **aucun bras entraîné sans canal, aucun nul par permutation d'étiquettes de bras**. Limites déclarées : suffisance au niveau de la sortie seulement, logits différents (p. 7, p. 37) ; la récupération 2 047/2 048 est un contrôle au niveau du décodage, pas une preuve de maintien entre jetons (p. 34-35) ; recombinaison au sein du même processus générateur (p. 6, p. 23) ; non établi : que la compétition ou la largeur du paquet soient nécessaires (Table 1, p. 17) ; le test ne sépare pas lectures contextuelles et lexicales, le bruit de même norme n'est qu'un contrôle d'amplitude (p. 29-30) ; restant ouvert selon les auteurs : insertion dans un Transformer pré-entraîné, ignition non linéaire, limites de capacité, maintien entre jetons, rapport (p. 9, p. 16) ; survie du mécanisme à un lecteur invariant par permutation après réentraînement (p. 38). Code : un « evidence package » et un « evidence manifest » sont mentionnés (p. 18, p. 21) mais **aucune URL, ni anonyme ni publique**, n'apparaît dans le PDF (recherche texte du PDF, 01/11). |

Correction d'une lecture `[résumé moteur]` de la v0.2 : le résumé de la page 1
écrit « lowers accuracy by 41–52% » ; le corps (p. 6) et la Table 3 (p. 21)
donnent des **points de pourcentage** (41,41-51,56 points). La v0.2 écrivait
« −41 à −52 % ».

Lues en partie au tour 2, **corps lu en entier en v0.4** (texte HTML arXiv
ouvert le 08/11, sections dites) :

| id | pages / sections | usage dans la note |
|---|---|---|
| arXiv:1912.05676 Eccles et al., 2019 (NeurIPS 2019) | tour 2 : 1-5 sur 16 (intro, cadre, §3 pertes de signalisation et d'écoute) ; v0.4 : fin de § 3 (algorithme), § 4.1 somme MNIST, § 4.2 Treasure Hunt, § 5 ; annexes A-E non lues | §3 : il existe un équilibre où le récepteur ignore le canal ; « on voit souvent les agents ne pas utiliser le canal du tout ». v0.4 : § 4.1, Table 1 : somme de deux chiffres MNIST, canal **nécessaire** (l'auditeur ne voit pas le chiffre de l'émetteur), REINFORCE indépendant, 50 runs par réglage : **sans biais, 0,00 run « bon » (IC de Wilson 0,00-0,07), récompense finale 0,1 = « exactement ce que l'on obtient sans communication »** ; influence sociale : 0,00 aussi ; positive listening 0,88, positive signalling 0,98, les deux 1,00. § 4.2, Table 2 : Treasure Hunt (A2C + V-trace, 5 symboles), sans biais 0,28 de runs (IC 0,18-0,42) au-dessus du meilleur run sans communication (récompense 13), les deux biais 0,94 ; « avec ou sans biais, les agents trouvent souvent des optima locaux ». § 4.2, Table 3 : intervention par **message constant imposé** après 100 pas (symbole le plus associé à un tunnel) : temps d'arrivée 100,6 ± 14,7 → 36,1 ± 3,3 (run médian), 85,4 → 41,3 (meilleur run), 100 épisodes : « la communication constante du chercheur est bien utilisée ». |
| arXiv:2607.26773 Zhang & Emu, 2026 | tour 2 : 1-5 sur 9 (résumé, introduction, travaux liés, début de méthode) ; v0.4 : travaux liés (Table 1), cadre d'audit (frontière, quatre réglages, cinq métriques, attribution par composante, contrôles), expériences et résultats, conclusion → corps lu en entier (pas d'annexe dans le HTML) | §3 : audit à quatre réglages de message ; cite la préférence pour le remplacement dans la distribution (Zhang & Nanda, arXiv:2309.16042) et les risques hors distribution. v0.4 : LLM **gelés** (Qwen3-4B et 8B, LatentMAS), 60-100 exemples appariés, 1 à 3 **graines de décodage**, bootstrap apparié par exemple, test d'équivalence à ±1 pp ; identité OPE = OME + CAG (présence = effet d'un message d'un autre exemple + gain de contenu) ; GSM8K 4B : −1,00 pp = −6,17 pp (autre-exemple) + 5,17 pp (contenu), les deux intervalles excluant zéro ; MATH-500 4B : +15,00 = +8,33 + 6,67 (IC [0,42 ; 12,50]) ; SSG (vaut-il mieux qu'un message auto-généré à calcul égal) : GSM8K 4B −2,00 pp (IC contenant 0), MATH-500 8B +10,00 [2,50 ; 20,00] ; divergences au premier jeton (PL, CIC) sans rapport monotone avec la tâche ; aucun modèle entraîné, aucun bras sans canal entraîné. |

Lues en partie en v0.4 (texte HTML arXiv ouvert le 08/11 ; nouvelles
sources, snowball du tour 3) :

| id | pages / sections | usage dans la note |
|---|---|---|
| arXiv:2608.20054 Marincat, 2026 (v3 ; étude parent de 2609.11365) | § 3 (architecture, tâche, traitement), § 4 (plan apparié, unité statistique), § 5 (résultats, Tables 1-2, exception globale, stratification post hoc), § 6 (mécanisme : transplants), § 7 (limites) ; § 1-2 en résumé ; annexes A-F non lues | Quatre cellules sur un Qwen2.5-0.5B **gelé** + une LoRA partagée de rang 8 et trois projections (lecteur, écrivain, bouche), ≈ 4,32 M paramètres appris, supervision par entropie croisée sur le jeton de réponse, paquets **continus** 2 × 896 ; **dix paires jumelles** (5 initialisations × 2 ordres de données, mêmes octets d'initialisation et même flux, hachés), seuls les masques d'attention diffèrent ; seuils préenregistrés (préenregistrement anonymisé fourni) ; inférence groupée par initialisation : cinq moyennes par strate, bootstrap par grappes, **test de retournement de signe exact sur les cinq moyennes par initialisation**, p = 0,0625 = minimum atteignable (§ 4-§ 5.1) ; 9/10 paires > 0,20 aux deux profondeurs, médianes 0,7648 / 0,6050 ; **tous les paquets coupés en test → exactement 1/17 pour les dix sociétés restreintes** ; conjonction préenregistrée formellement **échouée** (médiane 0,6988 < 0,70) ; § 6 : sur six sociétés choisies après coup, transplant même-valeur 0,94-1,00, suppression ou bruit de même norme ≤ 0,11, mélange inter-exemples 0,07-0,11, contrefactuel 0,74-1,00 ; le seul modèle global performant : même-valeur 0,12-0,25 (code enchevêtré par épisode) ; limites : une architecture, 17 états, 10 trajectoires non indépendantes, confusion rôle / évidence non levée. |
| arXiv:2609.17637 Marincat, 2026 (v2 ; confirmation préenregistrée) | § 5 (résultats, Table 4, contrastes C1-C6, dépendance au canal) et début de § 6 (batterie mécanisme, Table 5) ; § 1-4, § 7-9 et annexes non lus | Soixante sociétés, cinq conditions, six strates d'initialisation × deux ordres ; **test de retournement de signe à six groupes**, intervalles bootstrap par grappes ; C1 (masqué + marqueurs contre global + marqueurs) : 12/12 trajectoires > 0,20, médianes +0,846 / +0,859, p = 0,03125 ; C6 (sans marqueurs) : 12/12, +0,884 / +0,917 ; toutes les trajectoires masquées à 0,0588 = hasard quand tous les paquets sont coupés ; C2 (marqueurs sous visibilité globale) échoue, aucun modèle G+ ne suit les marqueurs ; bras « remplissage neutre » non concluant. § 6 : batterie **non décisionnelle**, banque gelée (816 épisodes), cellules intact / même-valeur / contrefactuel (+3) / contrefactuel apparié sur l'évidence / suppression (zéro) / **« dérangé »** (donneur aléatoire de valeur différente) / bruit ; cas conditionnés au succès (« do not establish mediation »). |

Lues en v0.5 (après le rapport du contradicteur ; identifiants résolus
de nouveau par API le 08/11 avant citation) :

| id résolu | auteurs, année | pages lues | ce qu'elle établit |
|---|---|---|---|
| hal-03100681 (v1, déposé le 6 janvier 2021) | Villecroze, Moulin-Frier, 2020 (1st SMILES workshop, ICDL 2020) | **PDF HAL de 7 pages lu en entier** (résumé, § I-VI, Fig. 1-6 ; références non lues) ; SHA-256 commençant par `7e9763b6ab21872a` ; le PDF n'est pas copié dans le dépôt | Grille 11 × 11, auditeur A3C à entrée pixel, message discret one-hot du pas précédent ; § V-A : sans biais, aucune émergence en 3 × 10^8 pas ; § V-B : face à un oracle fixé (direction à prendre), l'auditeur « uses almost exclusively the communication channel and neglects the visual input even when it is able to see the target » ; § V-C : CIC par contrefactuels, maximal sans bruit quelle que soit l'observabilité, réduit par l'observabilité **ou** le bruit (Fig. 4, 10 graines) ; § V-D : émetteur appris simplifié, pas de convergence (Fig. 5, 10 graines) ; § V-E : biais de signalisation positive → convergence « most of the time ». Pas de conflit message / observation, pas de nul apparié, pas d'émetteurs appris et fixé comparés sur la même tâche. |
| arXiv:1905.13687 (v3) | Kharitonov, Chaabouni, Bouchacourt, Baroni, 2020 (ICML, PMLR 119) | texte intégral ouvert ; **lus : § 3.3, § 4.1 (Guess Number), § 7 du supplément** (recherche ciblée) ; § 1-2, § 4.2 (MNIST), § 5-6, § 8 non lus | Guess Number (8 bits, le récepteur voit k bits) : entropie du message proche du minimum nécessaire ; à k = 8 l'information du message est « entirely ignored by Receiver » (messages brouillés, performance inchangée) ; le brouillage conserve la marginale et détruit l'information (§ 7) ; seuls les runs à précision > 0,99 sont analysés (§ 3.3) ; REINFORCE ≈ 20 % de succès, GS et SCG > 50 %. Lu en entier par le contradicteur ; lecture partielle de l'auteur de la note. Fait : l'identifiant figurait déjà dans `snowball-tour1.tsv` (aval de Lowe, sous le titre « Information Minimization In Emergent Languages ») et n'avait pas passé le filtre par titre. |

Lue en résumé seulement :

| id | source | statut |
|---|---|---|
| OpenReview hukTsKYu8Y | « Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model », soumission ICLR 2027, auteurs masqués (relecture en double aveugle), mots-clés Global Workspace Theory, activation patching | `[résumé moteur]` : PDF (`openreview.net/pdf/c506e622…`) et `notes?id=` en 403 au tour 2 aussi ; titre et résumé via `notes/search` ; aucune version arXiv trouvée (requêtes `ti:"Does the Workspace Do the Work"` et `abs:"global workspace" AND abs:"causal protocol"`, 0 résultat chacune). Résumé : couper le broadcast −41 à −52 % sur les questions à deux modules (≤ 1,4 pt sur un module) ; broadcast substitué → combinaison dans 61-73 % des cas ; restauration 2 047/2 048 ; protocole fixé d'avance. **non vérifié** au-delà du résumé. **Remplacé en v0.3** par la lecture intégrale (pp. 1-38, tableau « Complément v0.3 » ci-dessus) ; ligne conservée pour l'historique. |

## 3. Déjà publié

| affirmation | sources (nature) | état |
|---|---|---|
| Sur une tâche où l'information est privée par construction, un canal appris bat nettement le silence. | 1812.09755 Tables (article, expérience) ; 1903.05168 §2 et §6 (article, analyse critique) ; 2608.04893 §4.1 (régime calibré, 100 % contre 23-25 %) ; 2610.28905 Table 1 (article) ; `MS` G1 et `P100` §1 qui citent l'attendu (document interne) ; v0.3 : hukTsKYu8Y Prop. 1 p. 4 et p. 6 (canal seule route entre modules : OFF −41,41 à −51,56 points en composition ; mais canal coupé en test, pas bras entraîné sans canal) | recoupé (5 articles + dépôt) ; hukTsKYu8Y **confirme** sous forme démontrée |
| Les métriques côté émetteur (SC, entropie) ne prouvent pas que l'auditeur utilise le message. | 1903.05168 Table 2 et §5 (article) ; 1912.05676 Eccles 2019 §3 (article, autre équipe : équilibre où le récepteur ignore le canal) ; 2608.04893 (« un gain de benchmark n'établit pas la transmission », article empirique) ; v0.3 : hukTsKYu8Y pp. 1-2 et p. 9 (présence ≠ usage ; une architecture rend l'usage possible sans le produire) ; v0.4 : Eccles § 4.1 lu (mesures P_l et P_s = valeur du canal pour chaque agent, calculées contre les politiques optimales avec et sans canal) ; Zhang & Emu lu en entier (PL et CIC au premier jeton « sans rapport monotone » avec le gain de tâche) | recoupé ; Eccles **corps lu en entier** (v0.4) ; hukTsKYu8Y **confirme** (côté architecture plutôt que côté émetteur) |
| Couper le canal en test est un test biaisé ; il faut des bras entraînés avec et sans canal. | (a) couper/mettre à zéro n'est pas un test du contenu : 1903.05168 §6.2 (article, décalage de distribution) ; 2608.04893 §3 et MedQA (zéro −14,7 points contre dérangé −0,4 : effet de présence sans effet d'appariement) ; 2610.37849 §2 (reprend la recommandation de Lowe ; dépendant) ; 2607.26773 (cite la préférence pour le remplacement dans la distribution, pp. 1-5). (b) bras entraînés de zéro avec et sans canal : 1903.05168 §6.2 (recommandation) ; 2609.34373 (silence entraîné comme condition à part, 25 graines, pratique). v0.3, pour (a) : hukTsKYu8Y Prop. 1 p. 4 et preuve p. 29 (quand le canal est la seule route, la coupure renseigne sur la route, pas sur le calcul ; argument par construction, distinct du décalage de distribution) ; pour (b) : hukTsKYu8Y n'entraîne aucun modèle sans canal (ne touche pas). v0.4, pour (a) : 2607.26773 lu en entier, identité OPE = OME + CAG et GSM8K 4B : −1,00 pp = −6,17 pp (message d'un autre exemple) + 5,17 pp (contenu), intervalles excluant zéro : la comparaison « avec contre sans message » mêle présence et contenu, mesuré sur LLM gelés ; pour (b) : 2608.20054 et 2609.17637 coupent les paquets **en test** seulement (ne touchent pas) | (a) **recoupé**, hukTsKYu8Y **confirme** par un troisième argument, Zhang & Emu par une quatrième mesure (deux articles, deux équipes, expériences différentes ; nuance : Cheng parle d'effet de présence, Lowe de décalage) ; (b) recoupé en pratique, pas en énoncé |
| Le dérangement / la substitution / la restauration de messages comme protocole causal d'usage du canal. | 1903.05168 (CIC, messages brouillés) ; 2608.04893 (dérangé, zéro, aléatoire ; lu pp. 1-10) ; 2610.28905 (dérangé, substitution ciblée, zéro, aléatoire, auto-substitué ; lu pp. 1-9, code publié) ; 2607.26773 Zhang & Emu (4 réglages ; lu pp. 1-5) ; 2609.11365 (transplantation, dérangement, décalage ; lu pp. 1-8) ; v0.3 : hukTsKYu8Y lu en entier (OFF, transplant à trois valeurs, restauration des requêtes, transplant des requêtes seules, bruit de même norme, plan 2 × 2 et décomposition exacte Prop. 4 ; pp. 4-5, 7, 35-37 ; restauration 2 047/2 048, p. 7 et p. 22) ; v0.4 : 1912.05676 Eccles § 4.2 Table 3 (message **constant** imposé en cours d'épisode sur des agents entraînés de zéro, A2C : temps d'arrivée 100,6 → 36,1) ; 2608.20054 § 6 (transplants même-valeur 0,94-1,00 et contrefactuels 0,74-1,00, suppression, bruit de même norme, mélange inter-exemples ; interface apprise, backbone gelé) et 2609.17637 § 6 (cellules intact / même-valeur / contrefactuel / **dérangé** / zéro / bruit, banque gelée, batterie préenregistrée non décisionnelle) avec code MIT et checkpoints publics (dépôt vérifié par API, README lu) | **recoupé en méthode** ; huit sources lues, dont deux avec code ; la restauration est désormais lue en entier dans hukTsKYu8Y (restauration des **requêtes du récepteur**, pas du message) ; hukTsKYu8Y **dépasse** les autres sources sur le volet mécanisme (voir ligne suivante) ; v0.4 : le **message constant** (Eccles) et la cellule **dérangée préenregistrée** (Marincat) ont désormais un précédent lu |
| Un plan 2 × 2 (broadcast × requêtes du récepteur) avec restauration (nécessité) et transplant des requêtes seules (suffisance) localise la voie par laquelle un contenu transmis agit. | hukTsKYu8Y pp. 5, 7, 35-37 (article seul ; aucun code public : « evidence package » cité p. 18 et p. 21 sans URL) ; v0.4 : arXiv:2610.22026 `[résumé moteur]` est un « 2 × 2 » porteur × mécanisme de fusion, pas nécessité × suffisance (ne touche pas) ; 2608.20054 § 6 et 2609.17637 § 6 font suppression, transplant et contrefactuel, mais ni restauration des requêtes du récepteur ni transplant des requêtes seules (ne touchent pas) | **à recouper** (une seule source ; nouvelle ligne v0.3 ; rien de neuf au tour 3) |
| Sous couplage rigide un canal appris peut ne rien apporter par rapport au silence. | 2609.34373 §4.1 (lu pp. 1-9 : +0,001 ± 0,001, p = 0,982, n = 25 ; couplage partiel : appris indistinguable du silence, p = 0,400) ; 2608.04893 §5 (régime naturel : l'appariement du cache vaut 0 à ±2,8 points quand le récepteur résout seul) ; `MS:326-328` (gtm_mis 0,81 contre gtm 0,54-0,73, une graine, non répliqué) | **recoupé** (MARL à contrôle continu et relais LLM, deux équipes, deux régimes) ; `MS` reste une observation à une graine |
| Les protocoles appris par des entraînements indépendants ne sont pas mutuellement lisibles (le cross-play s'effondre). | 2609.34373 §4.5 (0,980 → 0,144 sur 600 paires ordonnées, le meilleur alignement construit ne récupère que 17 % de l'écart) ; 2609.11365 §4 (26 directions inter-initialisations sur 26 échouent à tous les échelons ; seules les paires de même initialisation sont interopérables) ; v0.4 : 2609.11365 **corps lu en entier** (§ 8 : « dialectes de coordonnées ancrés par l'initialisation », sémantique portable mais coordonnées privées ; P3 : l'interface héritée du modèle global produit un transfert négatif sévère, 0,065-0,169 contre 0,857) + **code et données** publics (`tokenosopher/populus-evidence-partitioning`, MIT et CC BY 4.0, cartes d'alignement P1 `p1_maps.pt` sur Hugging Face ; dépôt vérifié par API, README lu, contenu non relancé) ; 2606.03280 `[résumé moteur]` (Pythia 160M → 410M : alignement linéaire hors ligne, cosinus ≈ 0,97, sans effet causal utile à l'injection) ; nuance 2608.16347 `[résumé moteur]` (projection apprise entre quatre architectures : effet causal préenregistré mais partiel, récupération 45-50 % contre 5 % de hasard) ; nuance Kim 2026 et Kondylidis 2025 (tours du 06-07/11 : grandes populations, agent naïf ajouté) | **recoupé à la lettre** (v0.4 : article + code et données pour Marincat, deux équipes ; les tours du 06-07/11 le tenaient « non à la lettre » faute de seconde nature) ; portée : paires entraînées séparément ; nuances consignées |
| Messages discrets appris entre modules ou agents spécialisés. | 2107.02367 (article, modules spécialistes) ; 2609.34373 Table 2-3 (canal VQ de 2 bits, 25 graines, 0,983 contre silence 0,000 sur l'arène non couplée : un canal discret par VQ transmet quand il est nécessaire) | **recoupé, périmètre réduit** : agents MARL, pas modules spécialistes Transformer ; hukTsKYu8Y ne touche pas cette ligne (paquets **continus** de 640 dimensions, Table 2 p. 18) ; la veille interne `VEI:70-105` dérive de l'article et ne compte plus |
| Un code fixe décodé par attracteurs résiste à un fort bruit de canal sans aucun apprentissage. | 2609.32605 (article ; lu) ; 2610.08612 (article, même groupe) ; 1407.6029 Fiete, Schwab, Tran `[résumé moteur]` (autre groupe : réseaux de Hopfield binaires construits comme décodeurs de tout code neuronal bruité, taux d'erreur asymptotiquement nul à taux d'information 1/log n) | **recoupé** par un groupe indépendant, mais le second groupe en résumé seulement et sur un réseau **construit** (non appris) ; 10.1038/339693a0 Sourlas 1989 (notice Crossref seule) n'est qu'un précédent de principe, non utilisé |
| Positive listening partenaire contre étranger comme test d'usage. | 2610.37849 Denamganaï 2026 (lu pp. 1-6 ; v0.4 : **corps lu en entier**, § 4 résultats : Δ_PL = 0,615 [0,453 ; 0,751] avec code, −0,027 [−0,112 ; 0,044] sans ; Table 2 : trois conditions avec code au-dessus de zéro, trois sans à zéro ; § 5 : agents LM gelés, émetteur **scripté**, « not yet for learned or emergent codes ») ; 1903.05168 §3 (définition) ; dépôt `Near32/PartnerOutsiderPositiveListening` (code, README lu) | recoupé sur la notion et la méthode de remplacement ; v0.4 : résultats lus, **portée** : codes instruits dans des LLM, pas codes appris ; le bootstrap porte sur sessions et tours, pas sur des graines d'entraînement |
| v0.5 — Un récepteur qui peut réussir seul ignore un message redondant ; l'émetteur appris continue pourtant d'encoder. | 1905.13687 § 4.1 et § 7 (lu, passages ; k = 8 : messages brouillés sans effet, émetteur appris) ; 1912.05676 Eccles § 3 (équilibre où le récepteur ignore le canal, lu) ; 2609.34373 § 4.1 (couplage rigide : aucun canal ne bat le silence, lu) ; hal-03100681 § V-C (l'observabilité réduit le CIC quand le canal est bruité, lu) | **recoupé** (quatre équipes, jeux et algorithmes différents) ; apporté par le contradicteur (Kharitonov), vérifié dans le texte par l'auteur de la note |
| v0.5 — Les analyses de protocole ne portent que sur les graines qui ont réussi. | 1905.13687 § 3.3 (« We select the runs that achieved a high level of performance », lu) ; pratique inverse : 2609.34373 (25 graines, toutes rapportées) et 2608.20054 (dix paires, toutes rapportées) | observation de pratique, une source pour la sélection ; la décision PI 11 (aucune exclusion par graine) prend le contre-pied |
| v0.5 — Face à un émetteur oracle **fixé**, l'auditeur utilise le message même quand il voit la cible. | hal-03100681 § V-B et § V-C (lu en entier par le contradicteur, puis par l'auteur de la note sur le PDF HAL ; grille 11 × 11, A3C, oracle directionnel, 10 graines) | **à recouper** : une seule source (deux lecteurs ne font pas deux sources) ; c'est le pôle « fixé » de V2 ; il manque une seconde équipe avec émetteur fixé et auditeur qui voit la cible |
| v0.5 — Chauhan & Bera n'interviennent pas sur le message en test. | 2609.34373 texte intégral (recherche du 08/11 : « ablation » 0, « intervention » 0, « shuffled » 3 occurrences, toutes sur le nul d'alignement § 3.4 et § 4.5) ; corps lu au tour 2 | **vérifié** (constat d'absence dans une source, pas une affirmation à recouper) ; conséquence : la ligne « couplage rigide » (§ 3 l. 6) mesure la valeur du canal, pas son usage causal |
| Une mesure d'usage fondée sur le zéro reste pratiquée. | `SudarshanHegde326/emergent-comm-usage-marl`, `scripts/analysis/spread_ablation.py` (mémoire MSc, 3 graines, ablation par message mis à zéro) `[README et début du script lus]` | observation de pratique, une source de code ; ne pas en tirer de conclusion |

v0.3, lignes non touchées par hukTsKYu8Y (fait, après lecture des pp. 1-38) :
couplage rigide / canal sans valeur (le papier n'a pas de condition où le canal
est inutile, la composition l'exige par construction, p. 3) ; cross-play (deux
initialisations par corpus, mais aucun appariement émetteur-récepteur croisé,
p. 23) ; code fixe décodé par attracteurs ; positive listening partenaire
contre étranger ; pratique du zéro (le papier utilise OFF comme échelon 1 en le
déclarant insuffisant, p. 4).

## 4. Reste ouvert

v0.5 — Ce qui reste ouvert après le rapport du contradicteur et les
décisions PI du 08/11 (constats ; le verrou n'est pas déclaré passé) :

- **V1** (usage du message sur `sym` face à l'émetteur fixé, conflit
  message / champ t contre placebo) et **V2** (différence de cet indice
  entre émetteur fixé et émetteur appris). Fait : aucune des 11 sources
  lues (9 du tour 3, plus Kharitonov et Villecroze) ne met le message et
  l'observation propre de l'auditeur en conflit épisode par épisode avec
  un nul apparié par graine ; Kharitonov brouille les messages entre
  entrées (k = 8, émetteur appris), Villecroze mesure un CIC par
  contrefactuels face à un oracle, Eccles impose un message constant
  (canal nécessaire), Cheng, Gazzaev, Zhang & Emu et Marincat
  interviennent sur des backbones gelés. Inférence : l'espace vide est
  étroit (options § A) mais **non recoupé** : la descente de citations de
  1905.13687 et de 2609.34373 n'a pas été faite (Semantic Scholar 429
  pour le contradicteur ; non refaite dans cette version) → tour 4.
- **Le modérateur « émetteur fixé contre appris »** (V2). Fait :
  Kharitonov (appris → ignoré) et Villecroze (fixé → utilisé) sont lus
  dans deux tâches, deux algorithmes, deux équipes. Inférence : qu'un
  auditeur dépende davantage d'un émetteur fixé que d'un émetteur appris
  conjointement, à tâche égale, n'a été trouvé **écrit nulle part** ; la
  direction attendue de V2 est une inférence de la note, pas une
  affirmation publiée.
- Le point 1 ci-dessous (usage causal du canal `learned` sur `asym` avec
  nul apparié) est **reclassé** : sur `asym`, l'usage est un corollaire
  du succès (§ 1, v0.5) ; ce qui en survit est le contrôle positif de la
  machinerie (décision 3 : `eng_asym` sous dérangement doit chuter de
  S − 1/C, plus un agent scripté « message seul ») et la méthode du nul
  apparié, transportée sur `sym` (décision 2). Le texte des points 1 à 4
  est conservé tel quel pour l'historique.

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
   v0.4 — Fait : l'étude parent de Marincat (arXiv:2608.20054, § 4-§ 5,
   lu) apparie les bras par initialisation **et** ordre de données (mêmes
   octets, mêmes flux hachés), groupe l'inférence par initialisation et
   applique un **test de retournement de signe exact sur les cinq moyennes
   par strate** (p = 0,0625, le minimum atteignable) ; sa confirmation
   (arXiv:2609.17637, § 5, lu) étend à six strates (p = 0,03125) et douze
   trajectoires appariées, avec seuils préenregistrés ; les deux
   interviennent sur les paquets (transplants, suppression, dérangé) des
   sociétés dont l'interface est apprise. Fait : le retournement de signe
   des différences appariées équivaut à une permutation des étiquettes de
   bras à l'intérieur de chaque strate. Fait : les deux bras ont le canal
   (la coupure est en test, « all-cut »), le backbone est gelé (seuls LoRA
   et projections sont appris), la supervision est une entropie croisée,
   les paquets sont continus, et l'auteur écrit lui-même que la conjonction
   préenregistrée du parent échoue formellement et que les audits
   conditionnés au succès « n'établissent pas la médiation ». Inférence :
   la méthode appariée par graine avec nul de permutation **et**
   l'intervention de message coexistent maintenant dans un même programme
   lu (à côté de hukTsKYu8Y pour le test du signe sur familles) ; ce qui
   manque encore, et que personne des 9 sources lues ne fait, est le
   troisième terme : un bras **entraîné sans canal**, apparié par graine,
   sur des agents appris par renforcement avec messages discrets. Absence
   constatée sur 9 sources lues et 7 canaux (Crossref par mots-clés,
   Europe PMC et bioRxiv hors neurosciences ajoutés au tour 3 ; OpenAlex 4
   requêtes sur 6 en 503, non vérifié) ; elle reste une inférence.
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
   v0.4 — Fait : Crossref (7 requêtes), OpenAlex (2 requêtes servies),
   Europe PMC (4 requêtes, préprints) et bioRxiv hors neurosciences (10
   catégories, voir journal) n'ont rendu que deux entrées impulsionnelles,
   toutes deux hors cible : arXiv:2512.05654 `[résumé moteur]`
   (synchronisation d'agents hétérogènes par impulsions de 1 bit, théorie
   du contrôle, aucun message appris, aucune intervention) et
   doi:10.1109/icons62911.2024.00055 `[résumé moteur]` (SNN évolués
   **homogènes** pilotant un essaim). Fait : les sociétés de Marincat sont
   homogènes (quatre cellules, un même backbone, une LoRA partagée).
   Inférence : l'absence est constatée sur un canal de plus (7), toujours
   sans lecture intégrale qui la confirme ; la marque **à recouper**
   reste.
3. **Claim B (accord de classifieurs) contre usage du canal.** Fait : `RAP:56`
   (B3) indique que les codes « post-VQ » sont l'argmax de la tête π et que
   MI(code ; XOR) ≈ 0,001 nat selon un rejeu d'agent non reproductible depuis
   le dépôt. Inférence : la question « le canal porte-t-il l'information
   nécessaire » n'est pas tranchée par Claim B. Question interne, pas de
   littérature attendue.
4. **Un récepteur peut ne pas utiliser un canal appris même quand il est
   nécessaire (tour 2 ; réécrit le 07/11 après lecture).**
   Faits, par source :
   - Chauhan & Bera, arXiv:2609.34373 v2 (lu en partie : pp. 1-9 sur 17 au
     tour 2 ; résumé, § 1 et § 4.1 relus le 07/11) : arène partiellement
     couplée, émetteur appris à 2 bits 0,562 ± 0,127 (IQM 0,482) contre
     silence 0,494 ± 0,124 (IQM 0,368), écart +0,068 ± 0,163, **p = 0,400**,
     n = 25, borne supérieure unilatérale à 95 % +0,202 ; l'émetteur conçu au
     même débit atteint une IQM de 1,000 ; les auteurs notent une
     distribution par graine bimodale.
   - Eccles et al. 2019 § 3 (lu en partie, pp. 1-5 sur 16) : équilibre où le
     récepteur ignore le canal.
   - Huttegger et al. 2014, *PNAS*, doi:10.1073/pnas.1400838111 (lu en
     entier : texte principal PMC4113927 ; annexe SI non lue) : au-delà de
     2 états, signaux et actes, un ouvert de conditions initiales du
     réplicateur converge vers des mises en commun partielles ; avec des
     états non équiprobables, des issues « with basically no communication »
     sont stables sous sélection-mutation (fig. 1), à condition que les
     récepteurs mutent assez peu devant les émetteurs (δ/ε < (2p − 1)² dans
     le jeu binaire tronqué, δ/ε < 2(1 − 2p)²/(1 − 2p + 2p²) dans le jeu
     complet ; Hofbauer & Huttegger 2008, *J. Theor. Biol.* 254:843-849,
     doi:10.1016/j.jtbi.2008.07.010, éq. 14 et 18, lu en entier, source
     primaire de cette phrase de Huttegger 2014, même auteur, donc pas une
     seconde source indépendante) ; attention : dans ces deux textes p est
     le poids de l'état le plus probable, pas une valeur p, sans rapport avec
     le p = 0,400 de Chauhan & Bera ; le renforcement de
     Roth-Erev n'atteint le signalement parfait avec probabilité 1 que dans
     le cas 2 × 2 équiprobable, et synonymes et goulots sont des limites de
     probabilité positive. Aucune probabilité chiffrée pour les jeux de
     Lewis.
   - Chernyavskiy et al. 2026, *Technologies*, doi:10.3390/
     technologies14070432 (lu en partie : § 3.1, § 5, § 6, annexes C.3-C.4 ;
     annexe E connue par § 5.8 et § 6) : jeu de Lewis 3 × 3, EDO de
     croyances, pas d'émergence sous le seuil β_c (§ 5.4) ; sans terme de
     renforcement, échec dans 33 % des scénarios (§ 5.6) ; les auteurs
     écrivent que l'annexe E ne valide pas leurs théorèmes pour le RL
     profond (§ 6).
   - Calvano, Possnig, Tolvanen 2026, arXiv:2602.12035 v1 (lu en entier :
     § 1-8, annexes A.1-A.4 et B ; preuves non vérifiées ligne à ligne) :
     émetteur Q-learning tabulaire, récepteur qui connaît la politique
     courante de l'émetteur et y répond au mieux (§ 3.1) ; avec préférences
     alignées le babillage se défait et toute limite stable a un bien-être
     normalisé ≥ 1 − O(1/K) (§ 5) ; K = 21, 1 000 runs : 0,980 à 0,997, la
     révélation complète n'est jamais atteinte (§ 5, fig. 2) ; l'extension
     ne couvre qu'un récepteur qui apprend plus vite que l'émetteur
     (annexe B) ; pour un récepteur plus lent, les auteurs conjecturent une
     issue moins informative, jusqu'au babillage (§ 8, non démontré).
   - De Silva & Sigmund 2024, *SIAM Review* 66(2) 368-387,
     doi:10.1137/23m156402x (lu en entier : § 1-11, pp. 368-387, PDF fourni
     par le PI le 07/11) : tutoriel, jeux 2 états × 2 signaux × 2 actes
     seulement, dynamiques adaptative, du réplicateur et de meilleure
     réponse. Jeu du messager (§ 4.1, fig. 4, p. 372-373) : outre les deux
     normes de signalement, des ensembles d'équilibres « babbling » où le
     joueur II ne réagit pas au signal attirent des orbites sous la dynamique
     adaptative ; leur bassin croît quand la probabilité a priori p d'un
     état s'éloigne de 1/2 et, près de p = 0 ou 1, il est à peu près aussi
     grand que celui de chaque norme ; à p = 1/2 ces équilibres sont des
     points fixes instables. Sous la dynamique de meilleure réponse, l'échec
     de communication dans le jeu du messager est « practically
     eliminated » (§ 10, fig. 15, p. 384-385), alors que dans le jeu de
     signal coûteux les joueurs à meilleure réponse finissent plus souvent
     sans aucun signal (p. 385). Bassins donnés seulement en figures
     (pourcentages de conditions initiales tirées au hasard, valeurs non
     tabulées) ; aucun test statistique, aucun chiffre comparable à
     p = 0,400.
   Inférences :
   - L'existence du phénomène est **recoupée** par quatre sources de nature
     différente (MARL continu, MARL discret, théorie des jeux évolutionnaires,
     EDO), mais aucune ne porte sur PPO avec émetteur et récepteur appris
     ensemble à K = 16 / C = 4.
   - La contre-nuance de Calvano ne contredit pas 4.4 : elle suppose un
     récepteur qui répond au mieux, c'est-à-dire qu'elle exclut par
     hypothèse l'équilibre d'Eccles ; elle déplace la question vers la vitesse
     d'adaptation du récepteur, sur laquelle sa propre conjecture (§ 8) va
     dans le sens de 4.4. Hofbauer & Huttegger 2008 (§ 4 : un récepteur qui
     explore beaucoup moins que l'émetteur « might end up not
     communicating ») en donnent un analogue démontré, dans une autre
     dynamique (sélection-mutation, jeu binaire) ; une source pour ce lien :
     **à recouper**. Avec Huttegger, Calvano prédit aussi des mises en
     commun partielles stables plutôt que l'optimum.
   - De Silva & Sigmund vont dans le même sens sur un point précis : dans le
     jeu du messager 2 × 2, le récepteur qui ignore le signal a un bassin
     non négligeable sous une adaptation par petits pas (gradient) quand les
     états ne sont pas équiprobables, et ce bassin disparaît presque sous la
     meilleure réponse. En 2 × 2, le mode d'adaptation du récepteur pèse donc
     sur l'issue autant que les gains. Une seule source sur ce contraste, et
     sans apprentissage par renforcement : **à recouper**. Sur `asym` la
     cible est uniforme parmi les C candidats (`P100`,
     `experiments/n10d/spec.py`), l'analogue des états équiprobables où ces
     équilibres sont instables en 2 × 2 ; rien n'en découle pour
     K = 16 / C = 4.
   - Le chiffre p = 0,400 est **celui de Chauhan & Bera, sans seconde
     source** : aucune source lue ne rapporte un test « canal appris contre
     silence » comparable (les 33 % de Chernyavskiy et les 0,980-0,997 de
     Calvano sont d'une autre nature). C'est un test apparié sur 25 graines
     à distribution bimodale, pas le constat d'un canal inutilisé dans
     chaque graine.
     **à recouper** pour le chiffre ; recoupé pour l'existence (état du
     07/11 ; **levée en v0.4** pour l'affirmation reformulée, voir le
     paragraphe v0.4 ci-dessous).
   - Conséquence pour § 5 (inchangée) : le critère de réfutation doit
     distinguer « canal inutilisé » de « canal inutilisable » ; s'y ajoute
     qu'un verdict sur la moyenne doit être lu avec la distribution par
     graine.
   v0.3 — Fait : dans hukTsKYu8Y le canal appris est utilisé dans les quatre
   modèles de confirmation (p. 7) ; les auteurs écrivent que l'architecture
   rend l'usage possible sans le produire et que le protocole aurait pu le
   réfuter (p. 9). Inférence : le papier **ne touche pas** le chiffre de 4.4
   (aucun cas de canal inutilisé n'y est rapporté) ; il énonce seulement la
   possibilité ; la marque reste.
   v0.4 — Fait : Eccles et al. 2019, § 4.1 et Table 1 (lu ; actes NeurIPS
   2019) : somme de deux chiffres MNIST, canal **nécessaire** par
   construction (l'auditeur ne voit pas le chiffre de l'émetteur, 20
   messages, 19 actions), REINFORCE indépendant, **50 runs** par réglage ;
   **sans biais, 0,00 run atteint une récompense > 0,2 (IC de Wilson
   0,00-0,07) et la récompense finale 0,1 est « exactement ce que l'on
   peut obtenir dans cet environnement sans communication »** ; le biais
   d'influence sociale de Jaques ne change rien (0,00) ; avec un biais de
   positive listening ou de positive signalling, 0,88 à 1,00 des runs
   communiquent. Fait : § 4.2, Table 2, Treasure Hunt (A2C) : sans biais
   0,28 des runs (IC 0,18-0,42) dépassent le meilleur des 50 runs sans
   communication. Fait : ce sont des taux de runs, pas un test apparié
   « canal appris contre silence entraîné » ; l'algorithme est REINFORCE ou
   A2C indépendant, pas PPO ; la tâche est à un pas.
   Inférence : la marque « à recouper pour le chiffre » visait l'absence de
   tout second résultat chiffré où un canal appris nécessaire reste au
   niveau du silence. Eccles 2019 en donne un, lu en entier, d'une autre
   nature (actes 2019, apprentissage indépendant, 50 runs, tâche à un pas)
   que Chauhan & Bera (preprint 2026, MARL continu, 25 graines appariées) :
   **la marque est levée pour l'affirmation reformulée** « un canal appris
   nécessaire peut rester au niveau du silence dans une fraction importante
   des graines (0/50 sous REINFORCE sans biais ; appris indistinguable du
   silence, p = 0,400 sur 25 graines en couplage partiel) ». La valeur
   p = 0,400 et la bimodalité par graine restent celles de Chauhan & Bera
   seul ; ce ne sont pas des affirmations à recouper mais des chiffres
   d'une étude, à citer comme tels. Les deux sources ont en commun des
   agents indépendants sans biais de communication ; rien n'en découle pour
   PPO à K = 16 / C = 4 (la conséquence pour § 5 est inchangée).
   v0.4 — Fait (lien « vitesse d'adaptation du récepteur ») : Rita et al.
   2022, arXiv:2204.12982 `[résumé moteur]` (jeu de Lewis neuronal) :
   « les propriétés du langage émergent ne sont modifiées que par l'écart
   **relatif** de vitesse d'apprentissage entre émetteur et auditeur ».
   Inférence : source neuronale, d'une autre nature que Hofbauer &
   Huttegger 2008, pour le rôle de l'asymétrie de vitesse ; mais elle porte
   sur la **structure** du langage, pas sur le non-usage du canal, et n'est
   lue qu'en résumé : le lien reste **à recouper**.
   v0.5 — Fait : Kharitonov et al. 2020 (§ 4.1, lu) rapportent un
   récepteur qui ignore entièrement un message qu'un émetteur appris
   continue d'encoder, quand sa propre entrée suffit (k = 8). Inférence :
   troisième source lue, d'une troisième équipe, pour l'existence du
   non-usage ; elle porte sur un canal **redondant**, pas sur un canal
   nécessaire inutilisé (Eccles Table 1, Chauhan couplage partiel) ; la
   sous-marque « vitesse d'adaptation » n'est pas touchée.

Ce qui n'est **pas** ouvert : `eng − nocomm` sur asym avec émetteur conçu
(`P100` §1 le dit lui-même) ; v0.5 : plus largement, sur `asym`, la
nécessité du message et son usage par tout auditeur qui réussit, pour
tout émetteur (A21 § 2.3 ; rapport du contradicteur § 1.1 ; options § 0).
La seule question qui y reste, « l'apprenant apprend-il à décoder ? »,
est une question d'optimisation (Eccles Table 1 : 0/50 sous REINFORCE
sans biais ; Kharitonov § 4.1 : REINFORCE ≈ 20 %, GS et SCG > 50 %), non
d'usage causal.

## 5. Dessin minimal qui tranche la question ouverte

v0.5 — Le dessin minimal des versions v0 à v0.4 (bras `learned` et
`nocomm` sur `asym`, interventions permuté / gelé / zéro / restauration,
nul par permutation des étiquettes de bras dans chaque graine) est
**remplacé** par les dessins D-2 restreint et D-3 des options du 08/11,
tels que fixés par les 12 décisions PI. Le protocole lui-même n'est pas
réécrit ici : il le sera dans `protocol.md` v2 (PR #100, brouillon L0,
par un autre agent dans une PR séparée) ; ce qui suit en est le résumé
de lecture, sans valeur normative. Rien ne part avant le verrou et avant
un nouveau contradicteur aveugle sur le protocole v2.

- **Question et bras.** V1 sur `eng_sym` (émetteur conçu fixé, auditeur
  PPO entraîné) ; V2 = Δρ entre `eng_sym` et `learned_sym` (bras
  confirmatoire, décision 4), co-primaires sous Holm α = 0,01
  (décision 1). `nocomm_sym` sert de porte de suffisance (l'observation
  propre suffit : `nocomm_sym` ≥ ε, TBD-15, sinon « non mesurable »,
  décision 7). `eng_asym` sous dérangement devient le **contrôle
  positif** de la machinerie (chute attendue de S − 1/C), sans α
  (décision 3).
- **Cinq conditions d'évaluation** sur les mêmes 500 épisodes par
  graine : intact ; `msg_shuffle` (dérangement) ; **conflit** (champ t
  intact, message conçu pour un t′ parmi les C − 1 autres candidats
  visibles) ; **champ t dérangé** (analogue de `intervention_input` de
  Kharitonov, restreint au champ t) ; **placebo** (message intact,
  candidat t″ ≠ t désigné d'avance).
- **Statistique.** ρ_i = P(action = t′ | conflit) − P(action = t″ |
  placebo) par graine i (décision 2) ; Δρ pour V2.
- **Nuls.** V1 : échange conflit / placebo par épisode **à l'intérieur de
  chaque graine**, B = 10 000 ; V2 : permutation des étiquettes de bras
  dans chaque indice de graine, B = 10 000 (options § B).
- **Ordre d'exécution (corrige F1 du rapport).** Audit 4a → contrôles
  scriptés qui peuvent échouer (« message seul » ρ ≈ 1, « champ t seul »
  ρ ≈ 0) → contrôle positif `eng_asym` → portes au niveau du bras
  (`eng_sym` ≥ 0,95 ; `nocomm_sym` ≥ ε ; décodabilité de `learned_sym`
  par information mutuelle ou sonde linéaire, décision 4) → primaires.
  Les portes sont au niveau du **bras**, jamais par graine ; toutes les
  graines sont rapportées (décision 11).
- **« Ignoré » = TOST** avec δ_eq fixée et justifiée avant gel (sd du
  pilote + bruit binomial ≤ 0,022 à 500 épisodes, décision 8) ;
  « dominant » si la borne basse de ρ ≥ δ_use.
- **k = 25 graines appariées par backend** ; puissance Monte-Carlo sur ρ
  et Δρ après pilote, montée par amendement avant gel si < 0,80
  (décision 10). **Deux backends** (EGG et JaxMARL), côte à côte,
  conjonctifs (décision 9, TBD-10).
- **Encodage** du message gelé au gel ; résultat déclaré spécifique à cet
  encodage (décision 5). **Restauration et message figé retirés**
  (triviaux à L = 1 sans état, décision 6) ; le zéro n'est plus une
  condition.
- **Hôtes et coût.** Tout sur macM1 (CPU/MPS) : pilote, entraînements,
  évaluations, nuls ; lockf, oMLX en pause, données T7, archives ES + T7
  (décision 12). Durée non mesurée : pilote de débit obligatoire (go PI)
  avant tout chiffre ; JaxMARL/Metal à valider au pilote. 0 entraînement
  supplémentaire pour V1 si les bras `sym` existent ; `learned_sym`
  coûte k × 2 runs (options § B, D-3).

Fait : les points F1 (porte 3-asym rendant NH-96 irréfutable), F2
(`nocomm` estime une constante), F3 (usage sur la jumelle non mesuré) et
F7 (randperm contre dérangement ; appariement des épisodes) du rapport
sont traités par cet ordre d'exécution, par le passage de la question
sur `sym`, par les conditions conflit / placebo et par le nul par
échange apparié ; F4 (étiquette « fuite » → « asymétrie
d'optimisation »), F5 (π = 0,8 sans source) et F6 (budget B par règle)
relèvent du protocole v2. Inférence : rien dans ce résumé n'est un
protocole ; TBD-10, TBD-15, δ_eq et δ_use restent à fixer avant gel.

Historique des points à discuter (v0.2 à v0.4, écrits pour le dessin
antérieur ; conservés tels quels ; plusieurs sont absorbés par les
décisions du 08/11 : 25 graines, porte au niveau du bras, retrait du
zéro, de la restauration et du message gelé, nul apparié dans chaque
graine) :

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

Points à discuter issus du tour 3 (v0.4 ; inférences, rien n'est modifié
dans le dessin) :
- Fait : Eccles 2019 § 4.2 impose un message **constant** en cours
  d'épisode et mesure l'effet sur l'auditeur (Table 3). Inférence : la
  modalité « message gelé » du bras d'intervention a un précédent publié en
  actes ; la v0.3 n'en connaissait aucun.
- Fait : Marincat (2608.20054 § 4 ; 2609.17637 § 5) apparie par
  initialisation × ordre de données et teste par retournement de signe
  exact des différences appariées dans les strates, avec p minimal
  = 2^−(nombre de strates). Inférence : c'est un gabarit publié du « nul
  par permutation des étiquettes de bras à l'intérieur de chaque graine »
  de § 5 ; avec 5 ou 6 strates il ne peut pas descendre sous 0,0625 ou
  0,03125, ce qui fixe un ordre de grandeur du nombre de graines à
  prévoir pour que le nul apparié ait un seuil P99 atteignable (≥ 7
  graines pour un test du signe exact à 1 %).
- Fait : Eccles 2019 § 4.1 rapporte que sans biais 0/50 runs utilisent un
  canal nécessaire sous REINFORCE. Inférence : la porte d'apprentissage
  (« canal inutilisé » contre « inutilisable ») doit prévoir le cas où la
  **majorité** des graines `learned` restent au niveau de `nocomm` ; le
  critère de § 5 doit dire ce qu'il déclare dans ce cas (réfutation de
  l'usage, ou tâche non apprise).

## 6. Trois tests du verrou

État v0.5 (2026-11-08, après le rapport du contradicteur et les 12
décisions PI ; constats seulement, le verrou n'est pas déclaré passé) :

| test | état | ce qui manque |
|---|---|---|
| Saturation | **non déclarée.** Le contradicteur, sans la note, a apporté **3 sources qui changent la note** (1905.13687 ; hal-03100681 ; la relecture analytique d'A21 § 2.3 ; 2609.34373 était déjà lue mais sa portée est révisée) : § 1 réécrit (question tranchée, question retenue), § 3 quatre lignes, § 4 reclassé, § 5 remplacé. Un tour mené par un lecteur indépendant a donc encore changé la note : saturation **non**. Fait : les deux sources nouvelles datent de 2020 et étaient dans le périmètre des tours 1 à 3 ; 1905.13687 figurait dans `snowball-tour1.tsv` (aval de Lowe) et n'a pas passé le filtre par titre ; hal-03100681 n'est pas sorti des requêtes HAL du tour 2, restreintes à `docType_s:THESE`. | Descente de citations de 1905.13687 et de 2609.34373 (S2 429 côté contradicteur, non refaite ici) ; snowball de hal-03100681 ; un tour sans entrée qui change la note **sur la question retenue V1/V2** (le périmètre a changé, le compteur repart) ; annexes listées en v0.4. |
| Recoupement | **une marque ajoutée, aucune levée, décompte 2 → 3.** Marques de la v0.4 : § 3 l. 5 (plan 2 × 2) et § 4.2 (substrats hétérogènes), maintenues, rien ne les touche. Ajoutée : § 3 « émetteur fixé → message utilisé même en voyant la cible » (hal-03100681 seul ; lu par le contradicteur puis par l'auteur de la note sur le PDF HAL, donc plus de seconde main, mais une seule source). Non marquées mais à une source : la sélection des graines réussies (pratique, Kharitonov) ; le modérateur fixé / appris de V2 (inférence, aucune source). Hors tableau : § 3 l. 6 (couplage rigide) toujours deux preprints ; sous-marque § 4.4 « vitesse d'adaptation » maintenue. | Pour la nouvelle marque : une seconde équipe avec émetteur fixé et auditeur qui voit la cible (à chercher au tour 4 dans la descente de 1905.13687 et dans les travaux liés de hal-03100681). Pour § 3 l. 5 et § 4.2 : comme en v0.4. Pour § 3 l. 6 : version en actes de Chauhan & Bera. |
| Couverture | **6 canaux sur 6, inchangée en nombre ; un canal révélé insuffisant.** HAL avait été interrogé (tour 2, 07/11) avec le filtre thèses : hal-03100681 (communication d'atelier, type COMM) n'est sorti qu'avec la requête du contradicteur (`"communication émergente" OR "emergent communication"`, 21 notices, sans filtre). Le contradicteur a aussi interrogé arXiv (4 requêtes), OpenReview (3), Crossref (Lewis, Barrett, Poffenberger), theses.fr et Exa ; il n'a pas cherché les neurosciences du workspace (§ 4 du rapport). | HAL hors thèses (articles et communications) sur toutes les formulations du tour 2 ; OpenReview `OAmBy25HHg` 403 ; Hill et al. 2025 arXiv:2508.02912, Rita 2022 (arXiv:2209.15342 / hal-03831908), thèses Chaabouni tel-03536320 et Polytechnique s275563 non lues ; annexes A-F de Chauhan & Bera ; neurosciences du workspace. |

Pour passer le verrou, d'après ces constats (inférence, liste non
décisionnelle) : un tour 4 limité à la question retenue (descente de
citations de 1905.13687 et 2609.34373, snowball de hal-03100681, HAL hors
thèses) sans entrée qui change la note ; une seconde source pour la
nouvelle marque § 3 ; le protocole v2 écrit (PR #100) ; le pilote de
débit macM1 (go PI) et la puissance Monte-Carlo sur ρ et Δρ ; puis un
**nouveau contradicteur aveugle sur le protocole v2** avant tout gel
(décisions du 08/11, « Suite »).

État v0.4 (tour 3, 2026-11-08 ; constats seulement, le verrou n'est pas
déclaré passé) :

| test | état | ce qui manque |
|---|---|---|
| Saturation | **non déclarée.** Entrées nouvelles examinées au tour 3 : 427 identifiants (54 du snowball tour 3, 107 titres hors filtre du tour 2, 80 OpenAlex, 80 titres Crossref relus, 8 Europe PMC, 143 (98 DOI distincts) bioRxiv) ; 12 gardées ; **2 changent une phrase de la note** (arXiv:2608.20054, arXiv:2609.17637), soit 0,5 % des entrées nouvelles, et 1 sur 54 (1,9 %) pour le snowball seul : troisième tour consécutif sous 5 % pour les entrées nouvelles (tour 2 : 0,6 % ; 07/11 : 0,8 %). Mais le tour **change la note** (§ 1, § 2 six lignes, § 3 cinq lignes, § 4.1, 4.2, 4.4, § 5 trois points), surtout par les lectures intégrales de sources déjà gardées (Eccles, PoPLi, Zhang & Emu, Marincat). | Un tour sans entrée ni lecture qui change la note. Annexes non lues : Chauhan & Bera (pp. 10-17), Cheng (pp. 11-26), Eccles A-E, PoPLi A-E, 2608.20054 A-F ; 2609.17637 § 1-4 et § 7-9. Version en actes de Chauhan & Bera (`OAmBy25HHg`). |
| Recoupement | **deux marques levées, deux maintenues, une maintenue hors tableau.** Décompte de la note v0.3 : 3 marques (§ 3 l. 5, § 4.2, § 4.4 chiffre). Tour 3 : **§ 4.4 chiffre levée** (Eccles 2019 Table 1, lu, nature différente ; affirmation reformulée, voir § 4.4) ; **§ 3 l. 7 cross-play** passe de « non à la lettre » (tours du 06-07/11) à **recoupé à la lettre** (article + code et données de Marincat, dépôt vérifié par API) ; **§ 3 l. 5** (plan 2 × 2 nécessité / suffisance) et **§ 4.2** (substrats hétérogènes) **maintenues**, rien ne les touche ; § 3 l. 6 (couplage rigide), signalée « non à la lettre » par les tours du 06-07/11 sans être marquée dans ce tableau, reste tenue par deux preprints plus une revue en résumé. Décompte de la note : 3 → **2** ; décompte des tours du 06-07/11 : 5 → **3**. Sous-marque § 4.4 « vitesse d'adaptation du récepteur » : maintenue (Rita 2022 en résumé, porte sur la structure). | Pour § 3 l. 5 : code public de hukTsKYu8Y ou une seconde équipe faisant restauration des requêtes du récepteur. Pour § 4.2 : une lecture intégrale qui confirme l'absence, ou une source impulsionnelle hétérogène avec intervention. Pour § 3 l. 6 : version en actes de Chauhan & Bera ou une source de nature différente sur le mécanisme du couplage. Lecture de Rita 2022 pour la sous-marque. |
| Couverture | **6 canaux sur 6 interrogés au moins une fois** sur l'ensemble des tours (arXiv ; OpenReview ; bioRxiv : neurosciences au tour 2, Europe PMC le 06/11, **10 catégories hors neurosciences au tour 3** ; Crossref : **7 requêtes par mots-clés au tour 3** ; thèses-HAL : tour 2 et 07/11 ; code-données : tours 2, 06/11 et 3). Sources primaires de § 2 : le corps de **toutes** les sources du tour 1 et du tour 2 est lu en entier (PoPLi, Zhang & Emu, Marincat, Eccles ajoutés en v0.4) ; hukTsKYu8Y lu pp. 1-38. | Catégories bioRxiv non balayées (biologie cellulaire, microbiologie, génomique…) ; OpenAlex 4 requêtes sur 6 en 503 (non vérifié) ; annexes listées en « Saturation » ; les deux sources v0.4 (2608.20054, 2609.17637) lues en partie. |

État v0.3 (conservé ; après lecture intégrale de hukTsKYu8Y) :

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
| 2026-11-07 (PR #125, § 4.4) | Huttegger et al. 2014 (PMC4113927) ; Hofbauer & Huttegger 2008 (copie d'auteur) ; Chernyavskiy et al. 2026 (HTML MDPI via Exa) ; Calvano et al. 2026, arXiv:2602.12035 ; De Silva & Sigmund 2024 (PDF fourni par le PI) ; Zhang 2024, arXiv:2410.18806 ; Kondylidis et al. 2025, arXiv:2509.24660 pp. 5-12 ; Xu & Chen 2026, arXiv:2611.01666 pp. 1-13, 19-21, 25-26 | détail dans `veille/tour-1-2026-11-06.md` et `veille/tour-2-2026-11-07.md` (sections dites) |
| 2026-11-08 (tour 3) | arXiv:2610.37849 Denamganaï (HTML arXiv) | fin de § 3.1, § 3.2, § 3.3, § 4, § 5, § 6 → **corps lu en entier** avec le tour 2 ; annexes A-E non lues |
| 2026-11-08 (tour 3) | arXiv:2607.26773 Zhang & Emu (HTML arXiv) | travaux liés (Table 1), cadre d'audit, expériences et résultats, conclusion → **corps lu en entier** |
| 2026-11-08 (tour 3) | arXiv:2609.11365 Marincat (HTML arXiv) | fin de § 4, § 5-§ 8, reproductibilité et artefacts → **corps lu en entier** ; annexe A survolée |
| 2026-11-08 (tour 3) | arXiv:1912.05676 Eccles et al. (HTML arXiv) | fin de § 3 (algorithme), § 4.1, § 4.2, § 5 → **corps lu en entier** ; annexes A-E non lues |
| 2026-11-08 (tour 3) | arXiv:2608.20054 Marincat, étude parent (HTML arXiv) | § 3-§ 7 lus ; § 1-2 en résumé ; annexes A-F non lues |
| 2026-11-08 (tour 3) | arXiv:2609.17637 Marincat, confirmation 60 sociétés (HTML arXiv) | § 5 et début de § 6 (Tables 4-5) ; reste non lu |
| 2026-11-08 (tour 3) | arXiv:2610.22026, 2606.03280, 2610.33020, 2003.02979, 2506.06609, 2604.03266, 2610.15163, 2608.16347, 2608.13317, 2608.11676, 2607.11967, 2502.07552, 2204.12982, 2512.05654 | `[résumé moteur]` (notices arXiv API, `id_list`) |
| 2026-11-08 (tour 3) | doi:10.1109/wsai70435.2026.11708582, doi:10.1109/icons62911.2024.00055, doi:10.1109/icmi65310.2025.11141291 | `[résumé moteur]` (OpenAlex) ; doi:10.1007/s10458-024-09665-6 et doi:10.5220/0013383400003905 : titre seul |
| 2026-11-08 (tour 3) | GitHub `tokenosopher/populus-evidence-partitioning` ; HF `tokenosopher/populus-evidence-partitioning-checkpoints` | README (60 premières lignes) ; métadonnées HF (102 fichiers, public) ; rien relancé |
| 2026-11-08 16:00-16:45 (v0.5) | `brief-n10d-2026-11-08.md` ; `contradictor-report-n10d-2026-11-08.md` (SHA-256 `d1e5f9c6…`, 395 lignes) ; `options-n10d-causal-2026-11-08.md` ; `decisions-n10d-causal-2026-11-08.md` (hors dépôt, dossier `hypneum-reviews/2026-11-08-n10d-contradictor/`) | brief lu en entier ; rapport lu en entier (§ 0 contamination, § 1.1-1.7, F1-F7, § 3, § 4 non vérifié, § 5 verdict) ; options § 0, § A, § B (D-2, D-3) ; les 12 décisions |
| 2026-11-08 (v0.5) | A21 (`prereg/n10d-skeleton`, e0c64a8) | § 2.3 relu (E[S_msg_shuffle] = 1/C pour tout auditeur) |
| 2026-11-08 (v0.5) | arXiv:1905.13687 Kharitonov et al. (texte intégral ; notice API relue) | § 3.3, § 4.1, § 7 du supplément (recherche ciblée) ; reste non lu |
| 2026-11-08 (v0.5) | hal-03100681 Villecroze & Moulin-Frier (PDF HAL ; notice API `halId_s` relue ; SHA-256 commençant par `7e9763b6ab21872a`) | **7 pages lues en entier** (résumé, § I-VI, Fig. 1-6) ; références non lues ; PDF non copié dans le dépôt |
| 2026-11-08 (v0.5) | arXiv:2609.34373 Chauhan & Bera (texte intégral ; notice API relue) | recherche du texte pour « ablation », « intervention », « shuffled », « floor is a strategy », « 0.740 » ; rien de nouveau lu au-delà du tour 2 |
