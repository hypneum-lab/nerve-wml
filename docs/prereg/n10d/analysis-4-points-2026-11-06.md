# N10-D — quatre divergences de dessin tranchées par la source et la littérature (2026-11-06)

**Date :** 2026-11-06, rédigée vers 18:30 CET.
**Versions comparées :**
- **A** = PR #100, `origin/prereg/n10d-skeleton` @ `cb7d66b` (décisions PI de 17:05). Abréviations : `A:P` = `docs/prereg/n10d/protocol.md`, `A:N` = `nh96-null.md`, `A:MC` = `experiments/n10d/required_seeds_mc.py`, `A:DEC` = `experiments/n10d/decision.py`, `A:SP` = `experiments/n10d/spec.py`, `A:TMC` = `tests/test_n10d_seeds_mc.py`.
- **B** = PR #115, `origin/prereg/n10d-skeleton-amend-2026-11-06` @ `aac4496` (décisions PI de 16:55, réconciliation de 17:12). Abréviations : `B:P`, `B:N`, `B:SS` = `experiments/n10d/sigma_seed.py`. Le protocole n'a pas changé entre `fb544bb` et `aac4496` (seul un commit de formatage ruff). Version antérieure à la réconciliation : `c82281f`.
- **Pièces vérifiées** (identiques sur `master` et sur la branche #113 @ `8ca3c20`) : `docs/n10d/contradictor-2026-11-06/{VERIFICATION,verification-2026-11-06,verification-claude-report,verification-codex-report,SYNTHESE}.md` et les deux rapports de contradicteur ; `master:docs/prereg/n10d/reading-note.md` ; tour de veille 1 (`docs/n10d-veille-tour-1-2026-11-06` @ `6c9607d`, `docs/prereg/n10d/veille/tour-1-2026-11-06.md`). J'ai lu, en lecture seule, les propositions de la session auteure de A (`…/43b1fc75-…/scratchpad/restes-0611/tbd-n10d-proposals.md`, noté `PROP`).

**Méthode.** Les deux protocoles et les fichiers de code cités ont été lus en entier (`git show`, aucune modification, aucun commit). Les références ont été résolues par API (Crossref, arXiv `export.arxiv.org`, PubMed, Europe PMC) le 2026-11-06. Les textes ouverts sont lus aux pages indiquées. Quand je n'ai eu que le résumé, je l'écris, et l'argument pèse moins (`references.tsv`). Les calculs sont de l'arithmétique binomiale en `python3 -I` (script hors dépôt : `…/scratchpad/n10d-calc/calc.py`, `calc2.py`, `calc3.py`). Aucun code du dépôt n'a été importé ni exécuté. Le Monte-Carlo de A n'a **pas** été rejoué. Je l'ai recoupé par une approximation exacte à deux points, décrite au point 3.

Convention : **[source]** = fait lu au chemin:ligne ; **[lit.]** = passage lu dans l'article ; **[résumé]** = résumé seulement ; **[calcul]** = arithmétique de cette note ; **[inférence]** = raisonnement de cette note.

---

## Point 1 — σ_D pour dimensionner k

### Ce que dit la source

- A : σ_D vient d'une **hypothèse de planification** bimodale. Une fraction π = 0,8 des graines apprend (D_i ~ N(0,75 ; 0,03)), le reste reste aveugle (D_i ~ N(0 ; 0,027)), d'où Δ = 0,60, σ_D = 0,30 et q = 0,90 [source `A:P:138-143`, `A:MC:7-11`, `A:MC:33-46`]. Le pilote ne lit que `nocomm` ; σ_seed(nocomm) sert de contrôle de cohérence et ne dimensionne pas k [source `A:P:56` (TBD-14), `A:P:164-167`]. Aucune brèche.
- B : σ_D est l'écart-type (ddof = 1) des différences appariées eng − nocomm sur les 8 graines pilotes [source `B:P:139-145`, `B:SS:100-102`]. Le pilote donne k = max(k du test des signes, k de la moyenne appariée), avec q = Φ(0,30/σ_D) [source `B:SS:146-159`, l. 153]. La brèche n° 1 est déclarée [source `B:P:171-184`].
- Les deux versions font tourner le pilote à **budget réduit** : « reduced budget » [source `A:P:49`, `B:P:35`], « pilot budget B/4 » [source `A:P:53`]. Le budget B n'est pas fixé (TBD-11 ouvert) [source `A:P:53`, `B:P:39`].
- Dans B, la marge (TBD-12) et la règle de décision (TBD-16) restent **OUVERTES** [source `B:P:40`, `B:P:44`].
- Variance de nocomm, constat n° 3 : « `sigma_seed` mesuré sur `nocomm`/`asym` ne dimensionne pas D » [source `verification-2026-11-06.md:19`]. σ̂ = 0 dans 57 % des pilotes, P(χ²₇ < 7) = 0,571 [source `VERIFICATION.md:14`, `verification-2026-11-06.md:30`], simulation 57,2 / 55,9 / 57,7 % [source `verification-claude-report.md:28`]. La variance qui compte est celle de `eng` (bimodale) [source `contradictor-report-…-claude.md:74`]. Le facteur 2 est faux si σ_D est mesuré directement : k = 9 et non 18 [source `VERIFICATION.md:34`, `verification-2026-11-06.md:32`]. Les vérifications proposaient de lire l'écart-type de `S_eng` seul, « compatible avec l'aveuglement » [source `verification-2026-11-06.md:63`, `verification-claude-report.md:58`]. A le refuse parce que l'écart-type de `eng` « révèle quasi la taille de l'effet » [source `PROP:124`, `PROP:133`].
- Coût d'une graine : **aucun chiffre d'heures dans les deux protocoles**. TBD-11 est ouvert dans les deux, et A le renvoie à « a throughput measurement without score » [source `A:P:53`, `A:P:198`, `B:P:39`]. La seule estimation est une inférence de la note de lecture : « ≤ 1 h GPU par run, ≈ 10-20 h GPU au total », non mesurée, à comparer à ≈ 10,8 h par run N10-C sur KXKM [source `reading-note.md:239-243`]. A compte en runs : 6 cellules entraînées par graine et par backend (asym `eng`, `nocomm`, `learned`, `msg_shuffle` ; sym `eng`, `nocomm`) [source `PROP:117`]. Dans B, `msg_shuffle` n'est qu'une évaluation [source `B:P:85`], donc 5 cellules.
- Le comportement réel du tronc PPO (taux d'échec par graine) est inconnu, et les chiffres des contradicteurs sont « des scénarios illustratifs, pas des mesures » [source `contradictor-report-…-claude.md:172`].

### Ce que dit la littérature

- **Bimodalité des graines et instabilité de π.** Eccles et al. 2019 rapportent la « proportion of good runs » sur 50 runs par réglage, avec des IC de Wilson. Selon les biais d'apprentissage, elle va de 0,00 (IC 0,00-0,07) à 1,00 (IC 0,93-1,0) [lit. arXiv:1912.05676, Table 1 p. 7 ; annexe E p. 16]. Chaabouni et al. 2019 comptent comme réussi un run à plus de 99 % de précision. Ils obtiennent « 25 out of 48 » runs réussis en moyenne [lit. arXiv:1905.12561, §4 p. 5]. La fraction de graines qui apprennent est donc bien une loi de mélange, et elle varie d'un facteur 1 à l'infini selon des détails de dessin. Aucune des sources lues ne donne π pour un tronc PPO sur un jeu de Lewis à C = 4.
- **Lowe et al. 2019 sur la variance entre graines.** Le texte rapporte des moyennes ± erreur standard sur 10 graines (5 pour le CIC), avec 1000 jeux de test par graine [lit. arXiv:1903.05168, annexe B p. 11]. Il ne modélise pas la variance entre graines et ne donne pas de taux d'échec. Son apport ici est autre (point 4). Je n'y trouve **rien** qui chiffre σ_D.
- **Pilote et taille d'échantillon.** Browne 1995 : prendre l'écart-type d'un petit pilote donne « quite likely » une puissance réelle inférieure à la puissance visée. Une limite de confiance supérieure à 100(1 − γ) % la garantit dans au moins 100(1 − γ) % des essais [résumé doi:10.1002/sim.4780141709]. Julious 2005 recommande 12 par groupe pour un pilote, au titre de la précision de la variance [résumé doi:10.1002/pst.185]. Le pilote N10-D en a 8. Kraemer et al. 2006 : se servir d'un pilote pour guider la puissance mène le plus souvent à des études sous-dimensionnées [résumé doi:10.1001/archpsyc.63.5.484]. Colas et al. 2018, en RL profond : des écarts-types estimés sur un petit pilote « lead to under-estimate the sample size » [lit. arXiv:1806.08295, §5.2 p. 16 ; recommandations p. 18].
- **Pilotes internes et réestimation aveugle.** Wittes & Brittain 1990 recalculent des paramètres « présélectionnés » (variances, taux du groupe contrôle) en cours d'essai [résumé doi:10.1002/sim.4780090113]. Kieser & Friede 2003 : des estimateurs de variance **aveugles** ne touchent pas l'erreur de type I et protègent la puissance [résumé doi:10.1002/sim.1585]. Friede & Kieser 2006 parlent de « nuisance parameters » [résumé doi:10.1002/bimj.200510238]. En 2013, ils recommandent la variance à un échantillon des données mises en commun, en aveugle, pour « maintaining the trial integrity » ; la réestimation non aveugle n'est qu'une référence de comparaison [résumé doi:10.1002/pst.1564]. Toute cette littérature suppose une variance qui est un **paramètre de nuisance**, distinct de l'effet.

### Calcul

[calcul] Sous la loi de mélange de A : moyenne = 0,75 π, écart-type = 0,75 √(π(1 − π)). L'écart-type est donc une fonction de l'effet, symétrique en π ↔ 1 − π (π = 0,8 et π = 0,2 donnent 0,30 tous les deux). P(χ²₇ < 7) = 0,5711 (recalculé).

[calcul] Formule de B (`B:SS:146-159`, réécrite à l'identique) appliquée à un pilote de 8 graines dont J apprennent :

| J appris sur 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| σ̂_D | 0,027 | 0,267 | 0,348 | 0,389 | 0,402 | 0,389 | 0,348 | 0,267 | 0,030 |
| k (B) | 8 | 13 | 21 | 26 | 28 | 26 | 21 | 13 | 8 |

- π_pilote = 0,8 : P(k = 8) = 0,168 (les 8 graines apprennent, σ̂_D ≈ 0,03) ; E[k] = 17,2.
- π_pilote = 0 (rien n'apprend à budget B/4) : k = 8 avec probabilité 1.
- Puissance confirmatoire réelle si π_confirmatoire = 0,8 et k = 8 : 0,503 sous la règle IC, 0,811 sous la table de B (point 3). Au global, 0,253 et 0,658.
- La limite supérieure de Browne (80 %, n = 8 : facteur √(7/χ²₀,₂₀;₇) = 1,353) ne corrige pas le cas dégénéré : σ̂ = 0,03 → 0,041 → k = 8.
- Coût en runs confirmatoires, deux backends : A à k = 18 → 18 × 6 × 2 = 216 ; k = 20 → 240 ; B à k = 8 → 8 × 5 × 2 = 80, à k = 16 → 160. Les heures ne sont pas estimables (TBD-11). Avec l'inférence ≤ 1 h GPU par run de la note, l'écart entre k = 8 et k = 20 coûterait au plus ≈ 160 h GPU [inférence].

### Conclusion

**A est défendable devant un relecteur. B, tel qu'écrit, ne l'est pas.**

1. **B n'estime pas la bonne quantité.** Le pilote tourne à budget réduit [source `A:P:53`, `B:P:35`], et π dépend du budget (Eccles Table 1, Chaabouni 2019 : la fraction de runs réussis est le paramètre qui bouge). Un σ_D pilote mesure donc la variance d'une autre loi que celle des runs confirmatoires.
2. **L'estimateur est instable, et dans le mauvais sens.** Sur 8 graines bimodales, le pilote tout-réussi ou tout-raté donne σ̂_D ≈ 0,03 et k = 8, alors que la puissance réelle à π = 0,8 est de 0,50 à 0,81 par backend. Browne, Kraemer et Colas décrivent exactement ce biais vers la sous-puissance.
3. **La réestimation « aveugle » de la littérature ne s'applique pas.** La variance de D est ici une fonction de l'effet. Aucun estimateur de variance ne peut être aveugle à π, puisque `nocomm` est constant à 1/C au bruit binomial près [source `verification-2026-11-06.md:31`]. Lire σ_D, c'est lire une partie de l'effet, comme l'écrit `PROP:124`. Les méthodes de Kieser & Friede supposent un paramètre de nuisance (résumés seulement, poids moyen).
4. **Le moment de la lecture aggrave le risque.** Dans B, la marge (TBD-12) et la règle (TBD-16) sont encore ouvertes [source `B:P:40`, `B:P:44`]. Une lecture qui renseigne sur π avant de fixer marge et règle ouvre un chemin de décision dépendant des données.

**Condition pour A :** l'hypothèse π = 0,8 n'est mesurée nulle part [source `contradictor-report-…-claude.md:172`]. Elle doit être écrite comme hypothèse, avec une table de sensibilité en π, et k doit être choisi pour rester acceptable si π est un peu plus bas (point 3 : k = 20 garde une puissance globale de 0,885 à π = 0,75 ; à k = 18 elle tombe à 0,737) [calcul]. B ne deviendrait défendable qu'à trois conditions cumulées : pilote au budget confirmatoire B, toutes les autres décisions closes avant la lecture, et k seulement révisé **vers le haut** à partir d'un plancher fixé a priori. La troisième ne règle toujours pas le cas dégénéré σ̂ ≈ 0,03.

### Ce qu'il faudra écrire dans §9

« σ_D (TBD-14) : hypothèse de planification bimodale π = 0,8 (Δ = 0,60, σ_D = 0,30), écrite comme hypothèse avec table de sensibilité π ∈ {0,70 ; 0,75 ; 0,80} ; pilote nocomm seul, aucune brèche ; la lecture appariée de PR #115 n'est pas retenue (pilote à budget réduit, σ̂ instable sur 8 graines bimodales, variance fonction de l'effet). »

---

## Point 2 — porte d'apprentissage `sym`

### Ce que dit la source

- A : porte sur `eng_sym` seul (moyenne sur k graines ≥ 1 − ε, ε = 0,05), `nocomm_sym` non gaté. Le cas D se déclenche si la borne basse de l'IC bootstrap 95 % de D_sym est ≥ δ_sym = 0,10 [source `A:P:57`, `A:P:119-120`, `A:P:127-130`, `A:P:186-191`, `A:SP:15-16`, `A:DEC:71-86`]. Un écart inférieur à 0,10 ne suspend pas A-C [source `A:P:191`].
- B : `eng_sym` et `nocomm_sym` doivent tous deux être ≥ 1 − ε_sym, avec ε_sym = max(P99 du nul sym, 1/n_eval) = 1/n_eval = 0,002, parce que le nul est dégénéré (P99 = 0). L'échec de la porte **est** le drapeau D [source `B:P:115-118`, `B:P:205-210`, `B:P:43`]. D_sym n'a aucun test [source `B:P:210`, `B:N:24-27`]. La règle d'action à l'évaluation (« sampled or argmax ») est encore TBD-11 [source `B:P:239-240`].
- Le nul sym est dégénéré : P99 = 0 à σ = 0, donc p = 1/10 001 pour tout D_sym > 0 [source `verification-2026-11-06.md:21`, `verification-claude-report.md:29`].
- Contre-exemple : `nocomm_sym` = 0,625 contre `eng_sym` = 1,0 donne D_sym = 0,375 sans fuite [source `verification-2026-11-06.md:27`]. Le contradicteur Claude note qu'avec l'entropie résiduelle de PPO, `nocomm_sym` à 0,99 déclencherait D [source `contradictor-report-…-claude.md:84`], et que « avec un bonus d'entropie, `eng` plafonne sous 1 » [source `…-claude.md:126`].
- Origine de la double porte : les vérifications proposaient en exemple « `eng_sym` et `nocomm_sym` ≥ 1 − quantile de niveau » [source `verification-2026-11-06.md:60`] ou « ≥ 1 − ε, ε fixé au manifeste » [source `verification-claude-report.md:56`]. La vérification Codex demandait seulement de « dire ce qui est rapporté (contrôle de fuite, sans seuil au-dessus du plafond) » [source `verification-codex-report.md:74-76`]. **Aucune** pièce vérifiée ne justifie ε = 1/n_eval, et aucune ne définit `nocomm_sym` comme contrôle du tronc. B écrit seulement « plumbing control, not a hypothesis » [source `B:P:115`].
- Argument de A (transmis par le coordinateur) : « si les deux bras passent ≥ 1 − ε, alors |D_sym| ≤ ε automatiquement ; […] le cas D n'est atteignable que pour des écarts négligeables » [source `PROP:140-143`].

### Ce que dit la littérature

- **Niveaux atteints.** Lazaridou et al. 2017 (REINFORCE, cible + **1** distracteur, test sur 10 000 jeux) obtiennent 100 % avec l'émetteur « informed » et 99 % avec l'émetteur « agnostic » [lit. arXiv:1612.07182, §3 p. 4 ; Table 1 p. 5]. Eccles et al. 2019 : récompense finale des bons runs 0,92 ± 0,03 ; 0,99 ± 0,00 ; 0,98 ± 0,01 selon les biais [lit. arXiv:1912.05676, Table 1 p. 7]. Havrylov & Titov 2017 (127 distracteurs) : « argmax encoding slightly but consistently outperforms the sampling strategy » [lit. arXiv:1705.11192, §3 p. 6]. Le niveau dépend donc de la règle d'action à l'évaluation. Les critères de réussite publiés sont « above 99 % » [lit. arXiv:1905.12561, §4 p. 5] ou « > 99,9 % training-set accuracy » [lit. arXiv:2004.09124, p. 4]. Aucun ne vaut « tous les épisodes sauf un sur 500 ». Je n'ai trouvé aucune source lue avec exactement C = 4 et PPO : la comparaison reste indicative.
- **Lowe et al. 2019** : la communication « may act as an alternate pathway for optimization » [lit. arXiv:1903.05168, §6.2 p. 8]. Un D_sym > 0 sans fuite est exactement ce cas : le canal aide l'optimisation alors que l'information est déjà dans l'observation. C'est pourquoi le cas D est libellé « fuite possible ou effet d'optimisation » dans les deux versions.

### Calcul

[calcul] P(500/500 | p = 0,99) = 0,0066 ; p = 0,995 : 0,0816 ; p = 0,998 : 0,3675.

[calcul] Porte de B : la moyenne ≥ 1 − 1/500 équivaut à « échecs ≤ k sur 500 k épisodes ». Probabilité qu'**un** bras passe, puis que **les deux** passent (bras indépendants), comparée à la porte de A (`eng_sym` ≥ 0,95) :

| p par épisode | B, un bras (k = 18) | B, deux bras | A, `eng_sym` (k = 18) |
|---|---|---|---|
| 0,95 | 0 | 0 | 0,51 |
| 0,97 | 0 | 0 | 1,00 |
| 0,99 (Lazaridou « agnostic », Eccles) | 0 | 0 | 1,00 |
| 0,995 | 0 | 0 | 1,00 |
| 0,998 | 0,56 | 0,32 | 1,00 |
| 0,999 | 0,998 | 0,995 | 1,00 |

Les valeurs sont les mêmes à k = 8 et 16, à 0,03 près. Aux niveaux publiés (0,92-0,99), on attend 5 à 40 échecs par graine sur 500 épisodes, contre au plus 1 en moyenne pour B.

[calcul] Arithmétique de A, exacte : si a ≥ 1 − ε et b ≥ 1 − ε avec a, b ≤ 1, alors |a − b| ≤ ε. Pour une moyenne de différences appariées, D_sym = moyenne(eng) − moyenne(nocomm), donc |D_sym| ≤ ε = 0,002 quand la porte de B passe.

### Conclusion

**A (porte sur `eng_sym`, cas D par la borne basse de l'IC de D_sym ≥ 0,10) est la porte qui teste le câblage sans punir le bruit d'échantillonnage. Hybride proposé : A, plus `nocomm_sym` rapporté avec son IC comme contrôle descriptif du tronc.**

**Réponse explicite à l'argument de A.** L'arithmétique est juste. Elle visait le texte de PR #100 à `e5fbe88`, où le cas D se déclenchait *après* une porte réussie, sur D_sym > P99 [source `PROP:140-143`]. Elle ne s'applique pas à B tel qu'écrit, où le drapeau D **est** l'échec de la porte [source `B:P:118`, `B:P:205`]. Dans B, le cas D n'est donc pas vide : l'exemple 0,625 le déclenche bien. La double porte n'est pas pour autant un bon dessin, pour trois raisons.

1. **Le seuil est en dessous du bruit binomial.** Avec ε = 1/n_eval, D se déclenche presque sûrement dès que la politique d'évaluation échantillonne ou plafonne à 99 % (P(les deux passent) ≈ 0 à p ≤ 0,995). La lecture A-C serait alors suspendue presque toujours [source `B:P:208-209`], et ce sur un bruit que la littérature tient pour normal.
2. **Le seuil dépend d'un choix encore ouvert.** ε = 1/n_eval n'est atteignable que si l'évaluation est en argmax et la politique exacte, alors que ce choix est encore TBD-11 [source `B:P:239-240`] (Havrylov p. 6).
3. **Elle mélange trois événements distincts sous un seul drapeau** : `eng_sym` en échec (défaut de câblage ou de tronc, qui relève de « non mesurable »), `nocomm_sym` à 0,99 (bruit d'optimisation) et `nocomm_sym` ≪ `eng_sym` (le vrai cas D de Lowe §6.2). A les sépare.

La porte ne doit pas garantir un écart nul. Elle doit garantir que le câblage marche quand l'information est présente (`eng_sym`), et le cas D doit mesurer la taille de l'écart, avec un seuil de magnitude parce que le nul est dégénéré. Le seuil δ_sym = 0,10 est cohérent avec le plancher du cas C. Le rôle de `nocomm_sym` comme contrôle du tronc (le tronc apprend la tâche sans canal) n'est revendiqué par aucune pièce vérifiée. Il est utile, mais il doit être **rapporté** (niveau et IC), pas gaté à 1 − 1/n_eval. Résidu [inférence] : un effet d'optimisation inférieur à 0,10 reste théoriquement capable de pousser D_asym au-dessus de 0,30 ; c'est à déclarer, ce n'est pas bloquant.

### Ce qu'il faudra écrire dans §9

« Porte sym (TBD-15) : `eng_sym` ≥ 0,95 seul ; cas D si la borne basse de l'IC 95 % de D_sym ≥ 0,10 ; `nocomm_sym` rapporté avec son IC (contrôle descriptif du tronc) ; ε = 1/n_eval sur les deux bras non retenu (seuil inférieur au bruit binomial, dépend de la règle d'évaluation TBD-11). »

---

## Point 3 — k et règle de décision

### Ce que dit la source

- Le constat n° 7 exige **une seule règle** qui relie test des signes, IC bootstrap et seuil 0,30 aux issues [source `verification-2026-11-06.md:23`, amendement `:66`].
- A, règle IC : C si D̄ < 0,10 ; A si L ≥ 0,30 **et** le test des signes rejette ; « B » si U < 0,30 ; sinon « non concluant » [source `A:P:58`, `A:P:171-174`, `A:DEC:58-68`]. k est donné par simulation de cette règle : K_CONFIRM = 18 pour p² ≥ 0,90, 16 pour un seul backend [source `A:P:145-158`, `A:MC:39`].
- B, table : C si U < 0,30 ou D̂ < 0,10 ; A si le test des signes rejette, D̂ ≥ 0,30 et L > 0 ; B (« non concluant ») sinon [source `B:P:192-198`]. k vient du test des signes seul, sous l'alternative D_i ~ N(0,30 ; σ_D), donc q = Φ(0,30/σ_D) [source `B:P:148-149`, `B:SS:149-153`], avec MIN_K = 8 [source `B:SS:45`].
- **Hypothèse génératrice du Monte-Carlo de A, vérifiée à la lecture** [source `A:MC:33-70`]. D_i est tiré indépendamment par graine : appris avec probabilité 0,8 → N(0,75 ; 0,03), sinon N(0 ; 0,027). Bootstrap percentile de 2 000 tirages, avec l'index `min(n−1, int(0,025 n))` [source `A:MC:56`], le même que `A:DEC:34-37`. Règle : D̄ ≥ 0,10 et L ≥ 0,30 et nombre de signes ≥ m_k [source `A:MC:66-67`]. Deux backends supposés indépendants [source `A:P:148`]. Le test `A:TMC` ne contrôle que la reproductibilité, P(A | π = 0) = 0 et p > 0,85 à k = 19, sur 300 simulations [source `A:TMC:4-17`]. Trois écarts à noter : (a) le bootstrap du MC (2 000 tirages) n'est pas celui de la décision (10 000, `A:DEC:30`), écart mineur ; (b) une graine apprise atteint l'oracle 1,0 (D = 0,75), alors que les bons runs publiés plafonnent à 0,92-0,99 (Eccles Table 1) ; (c) il n'y a **pas de troisième mode** de décodage partiel, que la littérature documente (point 4).

### Ce que dit la littérature

- **Bootstrap à petit n.** Hesterberg 2015 : « the common bootstrap percentile interval badly under-covers in small samples » [lit. arXiv:1411.5279, résumé p. 1]. « Both intervals are poor in small samples—they tend to be too narrow » ; « If n is small, the t interval has an advantage » [lit. p. 49]. Non-couverture unilatérale pour 2,5 % nominal (Table 5) : 0,077 (n = 5), 0,048 (n = 10), 0,036 (n = 20) [lit. p. 50]. « Bootstrapping does not overcome the weakness of small samples » [lit. p. 27]. Colas et al. 2018 : sous N < 10, type I ≈ 10 % pour 5 % nominal, sur des données bimodales ; « A bootstrap test should not be used with less than 20 samples » [lit. arXiv:1806.08295, §5.1 p. 16, recommandations p. 18]. Agarwal et al. 2021 : « Statistical concerns cannot be satisfactorily addressed with few runs » [lit. arXiv:2108.13264, p. 5] ; à 3 runs, les IC bootstrap sous-estiment [lit. Fig. 6 p. 6] (bootstrap stratifié sur plusieurs tâches, donc pas directement transposable). Efron & Tibshirani 1993 ch. 14 (BCa) : **non lu**.
- **Puissance faible.** Button et al. 2013 : une puissance faible réduit la probabilité qu'un résultat significatif soit vrai et gonfle les tailles d'effet [résumé doi:10.1038/nrn3475]. Ordre de grandeur publié pour un test apparié d'agents appris de zéro : 25 graines (Chauhan & Bera, cité par la note) [source `reading-note.md:259-262`].

### Calcul

[calcul] **Approximation à deux points** (bruits N(0 ; 0,03) négligés ; quantile bootstrap remplacé par le quantile exact de Bin(k, J/k) ; signes = J + Bin(k − J, ½)). Elle recoupe le MC de A à ± 0,02 et reproduit sa non-monotonie : k = 11..18 → 0,811 / 0,856 / 0,891 / 0,859 / 0,908 / 0,915 / 0,951 / 0,947, contre 0,812 / 0,838 / 0,878 / 0,875 / 0,891 / 0,919 / 0,947 / 0,952 pour le MC [source `A:P:153-154`]. L'hypothèse génératrice est bien celle décrite ci-dessus.

[calcul] Test des signes : m_k et puissance.

| k | m_k | taille | q = 0,8 | q = 0,9 | q = Φ(1) = 0,841 (alternative de B à σ_D = 0,30) |
|---|---|---|---|---|---|
| 8 | 7 | 0,035 | 0,503 | 0,813 | 0,630 |
| 16 | 12 | 0,038 | 0,798 | 0,983 | 0,904 |
| 18 | 13 | 0,048 | 0,867 | 0,994 | 0,947 |

[calcul] P(issue A) sous le plan (π = 0,8), par backend → global p² :

| k | règle IC (A) | règle table (B) |
|---|---|---|
| 8 | 0,503 → 0,253 | 0,811 → 0,658 |
| 16 | 0,915 → 0,836 (MC A : 0,919 → 0,845) | 0,983 → 0,966 |
| 18 | 0,947 → 0,898 (MC A : 0,952 → 0,906) | 0,994 → 0,987 |
| 20 | 0,982 → 0,964 (MC A : 0,966 → 0,933) | — |

[calcul] **Erreur contre la marge** (vraie moyenne D = 0,30, π = 0,4) : règle IC 0,0085 / 0,019 / 0,020 à k = 8 / 16 / 18 (MC A : 0,023 à 18). Table de B : **0,174 / 0,303 / 0,325**. La table de B déclare A une fois sur cinq à une fois sur trois quand l'effet vaut exactement la marge. Sous l'alternative même de B (N(0,30 ; σ_D)), P(D̂ ≥ 0,30) = 0,5, donc P(A_B) ≤ 0,5 : le k de B ne dimensionne pas la règle de B.

[calcul] Bootstrap percentile sous données **normales** (logique de Hesterberg, P(T_{n−1} ≥ 1,96 √((n−1)/n))) : non-couverture unilatérale 0,055 (k = 8), 0,039 (16), 0,037 (18), 0,036 (20), pour 0,025 nominal. Sous la loi bimodale, c'est l'inverse : le percentile est conservateur (0,020 à 18), par discrétisation. Le type I réel dépend donc d'une loi inconnue.

[calcul] Sensibilité de la règle IC à π (global p²) : k = 18 → 0,517 / 0,737 / 0,898 pour π = 0,70 / 0,75 / 0,80 ; k = 20 → 0,734 / 0,885 / 0,964.

### Conclusion

**Règle : la règle IC de A. k : 20 par backend (hybride : la règle de A, k relevé de 18 à 20).**

- Les issues A ou B portent une affirmation de **magnitude** (« le tronc apprend le code », marge 0,30). Seule la règle IC contrôle l'erreur à la marge (≈ 0,02 sous le plan, ≈ 0,04 au pire sous données normales). La table de B, qui exige seulement D̂ ≥ 0,30, se trompe dans 17 à 32 % des cas à la frontière. Elle correspond à la règle (ii), « ponctuelle », que `PROP:161` chiffrait à ≈ 50 % sans le test des signes.
- k = 8 est exclu sous la règle IC (0,50 par backend, 0,25 au global) et faible même sous la table de B (0,66 au global). C'est le cas de puissance faible décrit par Button et al. 2013 (résumé seulement).
- k = 18 atteint la cible globale 0,90 tout juste (0,898 à 0,906 selon la méthode) et tombe à 0,74 si π = 0,75. **k = 20** donne 0,93 (MC de A) à 0,96 (approximation) sous le plan, 0,885 à π = 0,75. Il respecte aussi le seuil de 20 de Colas et al. pour un bootstrap. Comme la puissance n'est pas monotone, k = 20 est un pic local ; 22-23 seraient moins bons.
- **Condition :** la méthode d'IC doit être fixée avant le gel. Le percentile sous-couvre aux petits n (Hesterberg) ; je recommande la borne basse d'un t apparié ou l'« expanded percentile » [lit. Hesterberg §5.3 p. 50]. Le MC de A doit ensuite être rejoué à k = 20 avec cette méthode, sur macM1 et non ici.

### Ce qu'il faudra écrire dans §9

« Règle unique (TBD-16) : règle IC, A ssi L ≥ 0,30 et test des signes ; k (TBD-13) = 20 par backend par simulation de cette règle (global ≈ 0,93-0,96 à π = 0,8 ; ≈ 0,885 à π = 0,75), méthode d'IC (t apparié ou percentile élargi) fixée au gel et MC rejoué avec elle ; MIN_K = 8 et la table point + signes de PR #115 non retenus (P(A | D = marge) = 0,17-0,32). »

---

## Point 4 — combinaison des backends, cas B

### Ce que dit la source

- **Le libellé « B » ne désigne pas la même chose dans les deux versions.**
  - A, §5 : B = « D_asym ≥ 0,10 et U < 0,30 : weaker than predicted ; NH-96 rejected » [source `A:P:180-182`, `A:N:91`, `A:DEC:11`]. Mais §7 classe ce même cas B dans le statut par backend `non concluant` [source `A:P:218`]. **A est incohérent en interne** : un rejet de NH-96 en §5 devient « non concluant » en §7.
  - B, table : la zone « U < 0,30 et D̂ ≥ 0,10 » est classée **C** (« failed prediction, NH-96 rejected », qui ouvre la piste « the architecture blocks ») [source `B:P:196`, `B:N:88-91`]. Le libellé B y est le reste, « non concluant » [source `B:P:198`].
- Combinaison :
  - A et B (tête) : deux A → A ; deux C → C ; tout autre couple → « non concluant ». « Two non-rejections do not establish a common null », et la conjonction n'est pas un Bonferroni [source `A:P:220-224`, `B:P:252-256`]. B l'a reprise de #100 à la réconciliation [source `B:P:328-332`].
  - Avant la réconciliation, B disait : « a global conclusion is drawn only if both statuses are the same » [source `c82281f:docs/prereg/n10d/protocol.md:209-210`]. Deux B (= deux « non concluant ») donnaient alors « non concluant » global, soit la même sortie.
  - **La vraie divergence** porte donc sur le traitement de la zone « effet présent mais sous la marge ». A la rabat sur « non concluant » au niveau global. B la rabat sur C, et deux zones donnent un C global.
- La vérification Codex confirme que « deux non-rejets ne prouvent pas un nul commun » [source `verification-codex-report.md:54`].

### Ce que dit la littérature

- **Ce que signifie un effet sous la marge.** Zhang 2024 : le jeu de Lewis se résout souvent avec « one or two symbols » [lit. arXiv:2410.18806, résumé p. 1, §3.1 p. 2-3]. Un code partiel suffit en partie, donc des succès intermédiaires stables sont plausibles. Eccles et al. 2019 : les runs trouvent souvent des « local optima », par exemple un protocole qui ne rapporte qu'un tunnel sur deux [lit. arXiv:1912.05676, Table 2 p. 8]. L'apprentissage partiel est donc un état documenté, distinct de l'absence d'apprentissage (C) comme de l'apprentissage du code (A). Lowe et al. 2019 : une hausse de récompense avec canal est un « sufficient indicator » que la communication émerge, mais le canal peut aussi être une voie d'optimisation [lit. arXiv:1903.05168, §6.2 p. 8]. Un B dit donc « le canal sert, partiellement », pas « le canal ne sert pas ». Eccles §4.1 : le gain disponible Pl est calculé « explicitly » à partir des politiques optimales [lit. p. 6]. B ne peut donc pas se lire « nécessité partielle », puisque la nécessité est fixée par construction [source `A:P:66-69`, `B:P:48-58`].
- **Agrégation de verdicts.** Cochrane Handbook §12.2.2.1 : le vote fondé sur la significativité « can lead to the wrong conclusion », parce que les études sous-puissantes sont comptées comme « pas d'effet » (Hedges & Vevea 1998) [lit. ch. 12]. Hedges & Olkin 1985 ch. 4 et 1980 : **non lus**, Cochrane sert de substitut lu. Klein et al. 2014 : la reproductibilité dépend davantage de l'effet que de l'échantillon ou du cadre [résumé doi:10.1027/1864-9335/a000178]. Klein et al. 2018 : hétérogénéité significative dans 39 % des effets, surtout parmi les plus grands [résumé doi:10.1177/2515245918810225]. Nosek & Errington 2020 : une réplication est « a study for which any outcome would be considered diagnostic evidence about a claim ». Elle suppose un « precommitment to taking all outcomes seriously », et décider après coup de ce qui compte « can leverage post hoc reasoning biases » [lit. PLoS Biol. e3000691, section « Replication redux »]. Berger & Hsu 1996 (tests intersection-union) : **résumé seulement**.

### Calcul

[calcul et inférence] Un B par backend n'est pas un non-rejet. C'est un **rejet** de « D ≥ 0,30 » au niveau unilatéral 0,025 (U < 0,30). Pour l'énoncé global « D < 0,30 sur les deux backends », le nul est « D ≥ 0,30 sur au moins un backend ». La règle « rejeter si les deux rejettent » est de niveau α : si le backend j satisfait son nul, P(les deux rejettent) ≤ P(j rejette) ≤ α. C'est la structure intersection-union, dérivée ici et non lue chez Berger & Hsu. C'est **exactement** la logique qui rend légitime « deux A → A ». Sous le plan, deux B indépendants sont rares, puisque A domine. La question est donc de libellé et de cohérence, pas de puissance.

### Conclusion

**Propager un B concordant, avec un libellé propre. Ni « non concluant » (A, §7), ni « C, échec de prédiction » (B).**

1. Le libellé « non concluant » de A (§7) jette une issue pré-déclarée et diagnostique (Nosek & Errington). Il contredit §5 et `A:N:91`, qui disent « NH-96 rejected ». L'objection du vote compté (Cochrane) ne s'applique pas ici : B n'est pas une absence de significativité, c'est un rejet de magnitude. Deux rejets concordants sont un résultat, au même titre que deux A.
2. Le classement C de B fusionne apprentissage partiel et absence d'apprentissage. Il envoie à tort vers « the architecture blocks » [source `B:P:196`], alors que la littérature documente des optima locaux partiels (Eccles Table 2, Zhang).
3. **Libellé proposé** : « B global : plus faible que prédit sur les deux backends. Le canal transmet et le tronc apprend en partie ; la prédiction de magnitude de NH-96 est rejetée (U < 0,30 sur chaque backend). Ce n'est ni une nécessité partielle (fixée par construction) ni un blocage d'architecture. »
4. **Conditions.** (a) Un B par backend exige aussi que le test des signes rejette, preuve que D > 0. Sinon le statut est « non concluant ». (b) La distribution des D_i par graine est rapportée, parce que D̂ ne distingue pas « une partie des graines apprend tout » (bimodal) de « chaque graine apprend en partie » (Zhang, Eccles). (c) Un couple A + B reste « non concluant », car c'est une divergence qui appelle un amendement. Un couple B + C donne « NH-96 rejeté sur les deux backends » sans mécanisme commun, seulement si U < 0,30 des deux côtés. (d) Il faut corriger l'incohérence de A entre `A:P:180-182` et `A:P:218`.

### Ce qu'il faudra écrire dans §9

« Combinaison (TBD-10) : deux A → A ; deux B → B global « plus faible que prédit sur les deux backends » (intersection-union, niveau α), B par backend exigeant U < 0,30, D̂ ≥ 0,10 et le rejet du test des signes ; deux C → C ; autre couple → « non concluant » ; la zone U < 0,30 n'est pas classée C ; §5 et §7 alignés. »

---

## Récapitulatif

| point | conclusion | preuve principale | chiffre clé |
|---|---|---|---|
| 1. σ_D | **A** (σ_D a priori, pilote nocomm seul), à condition d'écrire π = 0,8 comme hypothèse avec table de sensibilité | Pilote à budget réduit [source `A:P:53`, `B:P:35`] ; variance fonction de l'effet (loi de mélange, Eccles Table 1 p. 7, Chaabouni 2019 p. 5) ; Browne 1995 et Colas p. 18 (pilote petit → sous-dimensionnement) | P(k = 8) = 0,168 à π_pilote = 0,8 et 1 si rien n'apprend ; puissance réelle alors 0,50 (règle IC) |
| 2. porte sym | **A + hybride** : `eng_sym` ≥ 0,95, cas D par L(D_sym) ≥ 0,10, `nocomm_sym` rapporté avec son IC | L'argument de A ne vise pas B (dans B, D = échec de porte, non vide) ; mais 1/n_eval est sous le bruit binomial (Lazaridou 99-100 %, Eccles 0,92-0,99, Havrylov argmax > échantillonnage) | P(les deux bras passent) ≈ 0 à p ≤ 0,995 ; 0,32 à p = 0,998 ; P(500/500 \| 0,99) = 0,0066 |
| 3. k et règle | **Règle IC de A, k = 20** par backend, méthode d'IC fixée au gel | Seule la règle IC contrôle l'erreur à la marge ; percentile sous-couvrant (Hesterberg Table 5 p. 50) ; ≥ 20 pour un bootstrap (Colas p. 18) | Global 0,93 (MC A) à 0,96 ; 0,885 à π = 0,75 ; table de B : P(A \| D = marge) = 0,17-0,32 ; k = 8 : 0,25 global (IC) |
| 4. combinaison B | **Propager B** comme « plus faible que prédit sur les deux backends », avec le rejet du test des signes requis ; ni « non concluant » ni C | B est un rejet de magnitude (U < 0,30), donc intersection-union de niveau α comme deux A ; optima partiels documentés (Eccles Table 2 p. 8, Zhang 2024) ; précommitment (Nosek & Errington) | A : incohérence §5/§7 (`A:P:180-182` contre `:218`) ; B : la zone est classée C (`B:P:196`) |

## Ce qui reste non tranchable par la littérature (à remonter au PI)

1. **π, la fraction de graines où le tronc PPO apprend le code ingénié sur ce jeu.** Aucune source lue ne donne π pour PPO à C = 4. La littérature montre seulement que π varie de 0 à 1 selon le dessin (Eccles Table 1). Le choix π = 0,8 et la sensibilité acceptée relèvent du PI.
2. **La valeur de la marge 0,30.** C'est un choix du PI (TBD-12), fermé dans A et encore ouvert dans B. Aucune littérature ne la fixe.
3. **L'indépendance des deux backends**, supposée et non montrée [source `A:P:148`]. Si elle est fausse, la puissance globale change, dans un sens que la littérature ne permet pas de prédire.
4. **Le coût par graine (TBD-11).** Il n'y a aucun chiffre dans les protocoles. Le passage de k = 18 à 20 (24 runs de plus) ne peut être arbitré qu'après la mesure de débit sans score.
5. **La règle d'action à l'évaluation (argmax ou échantillonnage, TBD-11).** C'est un choix de dessin. Il fixe le niveau atteignable de `eng_sym` et `nocomm_sym`, donc la pertinence de tout ε.
6. **La méthode d'IC (t apparié, percentile élargi, BCa).** La littérature lue penche contre le percentile à petit n. Le choix final et la nouvelle simulation sont à faire (BCa : Efron & Tibshirani ch. 14 non lu).
7. **Accepter ou non une brèche déclarée comme politique de laboratoire** (hypneum-prereg). C'est une question de règle interne, pas de statistique.

## Références non lues en texte (poids réduit)

Résumé seulement : Wittes & Brittain 1990, Kieser & Friede 2003, Browne 1995, Julious 2005, Friede & Kieser 2006 et 2013, Kraemer et al. 2006, Button et al. 2013, Klein et al. 2014 et 2018, Lakens 2022, Berger & Hsu 1996. Non lus : Efron & Tibshirani 1993 (ch. 14), Hedges & Olkin 1985 (ch. 4) et 1980 (remplacés par Cochrane ch. 12, lu). Lus partiellement : Zhang 2024 (expériences §4 non détaillées), Agarwal et al. 2021. Détail et hachages : `references.tsv`.
