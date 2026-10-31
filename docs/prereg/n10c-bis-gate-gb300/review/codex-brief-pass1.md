Rôle : relecteur indépendant (statistiques, apprentissage par renforcement, reproductibilité, pré-enregistrement). Tu relis un brouillon de pré-enregistrement L0 ; tu n'en es pas l'auteur.

Entrée : le dépôt `repos/nerve-wml`, branche `prereg/n10c-bis-gate-gb300` (PR #99, brouillon), deux commits au-dessus de `origin/master` (`git diff origin/master...HEAD`) :
- `docs/prereg/n10c-bis-gate-gb300/g7-host-annex.md` (annexe d'hôte « G7 ») ;
- l'item G7 ajouté à la fin de `docs/milestones/2026-10-25-n10c-bis-control-arms.md` ;
- `experiments/procgen_multiagent_gtm_comm/harness/devledger.py` (relèvement de `GATE_DEV_ALLOWANCE`) et `tests/unit/n10c/test_trunk_fix.py`.
Contexte : section « Necessity gate (PI decision 2026-10-30) », G1–G6, du même addendum ; amendement `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md` (sections 4.2, 4.3) ; `experiments/procgen_multiagent_gtm_comm/manifest/stage2-manifest-fixc.md` ; `bench/` (patch Procgen, locks, `procgen-builds/README.md`) ; `harness/train.py`, `harness/throughput.py`, `harness/codes.py`, `campaign/guards.py`, `campaign/kxkm/gate_bis_lane.sh` ; records sous `docs/superpowers/research/2026-10-25-n10c-gate/`. Fiche matérielle sourcée de la machine cible : `../stack.md` (DGX Station GB300).

Objet : la porte de nécessité (`eng − nocomm ≥ 0,20`, condition `closed`, 10M pair-steps) doit tourner pendant 24 h sur une DGX Station GB300 (aarch64) au lieu de KXKM-AI. L'annexe doit déclarer ce changement de substrat et fixer hôte, règle « jamais en parallèle d'un autre job GPU à nous », graines, build Procgen ARM, MLX 0.31.1 + mlx-cuda-12 avec repli mlx-cuda-13 et son déclencheur, tests de fumée, critères d'arrêt, rapatriement, comparabilité intra-hôte, et le relèvement de l'allocation (G6).

Livrables (rapport en français, chemin:ligne pour chaque constat) :
1. Erreurs factuelles : chaque affirmation de l'annexe vérifiable dans le dépôt (hash de trunk, valeurs d'allocation, cadence de log, options CLI, flags CMake, versions) ; dis si elle est juste.
2. Les commandes de la section 8 et des tests S0–S5 marchent-elles avec le code (arguments existants, garde de liveness, plafonds `gate_dev`, ledger) ? Y a-t-il un garde-fou du code qui refuserait le run ?
3. Failles de protocole : arrêt optionnel, aveuglement, choix des graines, lecture de G6, déclencheur du repli, comparabilité, transfert du verdict vers KXKM, contradiction avec G2–G5 ou avec N10-C gelé.
4. Ce qui manque pour que le PI puisse taguer, et les « TBD by PI » mal posés ou manquants.
Échelle : BLOQUANT / MAJEUR / MINEUR / OK. Si c'est invérifiable, dis-le. Ne modifie rien dans `repos/`. Aucun calcul lourd, pas d'installation de paquets. Écris le rapport dans `../report-sol.md`.
