# RUNBOOK — porte de nécessité A+ de N10-C-bis (k = 3, hôte mixte)

**Statut :** PRÉPARÉ, NON LANCÉ. Rien ne tourne tant que la case « go PI daté » (§0) est vide, et tant que les prérequis du §1 ne sont pas fusionnés sur `master`.
**Références :** plan `three-seeds-plan-2026-11-06.md` §6 (abrégé `P6`) ; déviation `docs/milestones/n10c-bis-substrate-split-2026-11-06.md` (abrégée `DEV-S`) ; addendum `docs/milestones/2026-10-25-n10c-bis-control-arms.md` G2-G6 (abrégé `MS`).

## 0. Go PI

- [ ] **Go PI daté : ______** (date, heure CET). Ce go vaut aussi signature de `DEV-S` §4.
- Go **envisagé** : 2026-11-12, après la fin des contrôles E1-small sur KXKM-AI. Il reste **conditionnel** : la conséquence d'une catégorie « non mesurable » ou « non concluant » (§7, G8 point 5) doit être reconfirmée par le PI avec la prochaine veille scientifique, avant le go. Les trois autres points ouverts du §7 ont été tranchés le 2026-11-06 vers 17:00 CET (décisions ci-dessous).
- [ ] Q3 confirmée : graines 9104, 9105, 9106, et `nocomm_rep` sur 9114, 9115, 9116. Sinon, nouvelles graines : ______
- [ ] Q4 confirmée : G2 inchangé (D̄ ≥ 0,20) avec les contrôles du plan §1.3. Sinon : ______
- [ ] Règle « résultats non lus » de `P6` §6.5 confirmée (verdict catégoriel seul).

## 1. Prérequis (commits séparés et datés, avant le premier run)

Les prérequis 1 à 3 sont faits dans la PR #114 (commits du 06/11, non fusionnés). Ils ne valent qu'une fois fusionnés sur `master`.

| # | Quoi | Où | État au 06/11 |
|---|---|---|---|
| 1 | Crédits `gate_dev` : `eng` 25, `nocomm` 13 ; `pretrain_dev` 10 ; entrées GB300 de G7 fusionnées dans le registre | `harness/devledger.py` (`GATE_DEV_ALLOWANCE`, `GATE_ALLOWANCE`), `manifest/smoke_ledger.json` | **fait**, commit `293877a`. Détail et incertitudes au §1.1 |
| 2 | `gate_g2_decide.py` (verdict catégoriel k = 3, 10 M, `closed`) et ses tests | `harness/` et `tests/unit/n10c/test_gate_g2_decide.py` | **fait**, commit `caba588` |
| 3 | Lanes `gate_aplus_lane.sh` (KXKM-AI et macM1) et leurs tests | `campaign/kxkm/`, `campaign/macm1/`, `tests/unit/n10c/test_aplus_lanes.py` | **fait**, commit `34dd433`. Les commandes du §3 restent l'équivalent manuel |
| 4 | Amendement daté de G2 (k = 3, graines, `nocomm_rep`, contrôles, table hôtes, §6.5) | `MS`, ajout en fin de fichier | **manque** : au PI. Proposition de texte au §7, non signée |
| 5 | Tests du dépôt sur macM1 (`uv run python -m pytest tests/unit/n10c`) au commit de lancement | `ssh macm1` | à faire |
| 6 | Copie du tronc 9105 sur macM1, SHA-256 vérifié | macM1 | à faire |

### 1.1 Registre : source du recalage GB300

- **Faits.** Le pilote G7 (GB300, 01/11) a écrit dans le registre de sa copie de travail `repos/nerve-wml`. Ce répertoire était exclu du rapatriement (`hypneum-gb300-20261101/rsync-excludes.txt`, ligne `repos/`). Le registre final est donc perdu. Les lanes arrêtées par signal n'ont pas écrit de `smoke_ledger.final.json` (`gate_g7_lane.sh`, dernière copie après la boucle).
- **Sources rapatriées.** Copies `smoke_ledger.initial.json` des lanes (`g7/lane-*/`), `g7/smoke_ledger.after-smoke-attempts.json`, `driver.log` de chaque lane, `record/g7-launch.log`, `record/gb300-shared.log`.
- **Fusionnées par id (7 entrées).** Prises dans `g7/lane-expl-s9102-20261101T102953Z/smoke_ledger.initial.json` : 4 mesures S5 (`measurement`, ne comptent pas dans les crédits) et 3 runs `gate_dev` `eng` graine 9101 (10:18:49, 10:26:10, 10:29:54 UTC). Leur résultat vaut `started` dans la source : l'entraîneur a été tué par SIGTERM, `finish()` n'a pas tourné. Il est noté `operator_stop` ici.
- **Reconstruites (2 entrées, id `gb300-reconstructed-*`, drapeau `reconstructed`).** `nocomm` 9102 et `eng` 9103, lancées à 10:29:55 et 10:29:58 UTC. Aucune copie ne les contient. Preuves : la ligne « start » de `driver.log` et l'existence du répertoire de run. `train_run` ne crée ce répertoire qu'après `reserve` et `claim` (`harness/train.py`, `train_run`). Id et heure de réservation exactes inconnus.
- **Non enregistrée : tentative RTX PRO 2000.** Vers 17:00 UTC, une copie de la lane sur la carte RTX a fini en OOM (mémoire de session, « G7 sur RTX … OOM 16 Go, lane arrêtée 17:15Z »). Aucun fichier rapatrié ne la décrit : le dernier `gb300-shared.log` rapatrié s'arrête à 15:12 UTC. Inférence : au plus 2 réservations `eng` 9101 (tentative et relance), si elle a utilisé le même clone. Elles sont couvertes par une **marge de 2** dans le crédit `eng`, pas par des entrées.
- **Calcul.** `eng` : 13 (`master`) + 4 (GB300) + 2 (marge RTX) + 6 (A+) = **25**. `nocomm` : 1 (GB300) + 12 (A+) = **13**, au lieu des 18 proposés avant recalage (18 supposait 6 crédits G7 consommés). `pretrain_dev` : 8 + 2 = **10**. Le test `test_gb300_g7_entries_are_merged_and_aplus_slots_remain` vérifie qu'il reste 8 `eng` (6 + 2), 12 `nocomm` et 2 `pretrain_dev`.
- **Incertitudes.** (a) Nombre réel de réservations RTX (0, 1 ou 2). (b) Identité exacte des 2 entrées reconstruites. (c) Des registres de copies KXKM-AI ou macM1 non fusionnés n'ont pas été cherchés ici.

**Pourquoi pas le runner gelé de N10-C run 2.** `campaign/run_campaign.py` ne lance que des runs enregistrés : 50 M pas, graines enregistrées sur leur hôte fixe, tag de gel et SWHID. `harness/train.py` refuse tout budget qui n'est ni ≤ au plafond de développement ni exactement 50 M (`train.py`, `main`). Les runs de la porte sont des runs de développement de 10 M (`gate_dev`, plafond `GATE_MAX_STEPS`). On appelle donc directement `harness.train` en mode développement.

**Pourquoi pas de nouveau preset.** G2 et B1 fixent `fix-c-coin-pretrain` (fix C, sans correction de biais Adam). Ce preset existe sur `master` (`harness/options.py:65`).

## 2. Paramètres fixes

| Graine | Hôte | Tronc (SHA-256 complet) | Ordre des cellules |
|---|---|---|---|
| 9104 | KXKM-AI | `894688cd4a62b868ea1f18d7021be57a64acb52699a8742048c9e2508da4f16f` | `eng` (9104), `nocomm` (9104), `nocomm_rep` (seed 9114, tronc 9104) |
| 9106 | KXKM-AI | à produire (§3.1) ; SHA consigné ici avant le premier run de porte : ______ | `nocomm` (9106), `nocomm_rep` (seed 9116, tronc 9106), `eng` (9106) |
| 9105 | macM1 | `d4916633ae7203f5e6ba5353451abf32fd280fa70c1b2404d1fb5803e2538a7d` | `eng` (9105), `nocomm` (9105), `nocomm_rep` (seed 9115, tronc 9105) |

Les SHA de 9104 et 9105 viennent de `init_trunk_sha256` dans `docs/superpowers/research/2026-10-25-n10c-gate/kxkm/adam-study/seeds-fix-c-coin-pretrain-eng-s910{4,5}/run.json`.

Paramètres communs :
- `--condition closed --steps 10000000 --preset fix-c-coin-pretrain --dev-kind gate_dev` ;
- `--log-every 200000 --ckpt-every 1000000` : les 5 derniers logs tombent à 9,2 / 9,4 / 9,6 / 9,8 / 10 M, comme G7 ;
- garde de vivacité `abort` tous les 100 k pas, fixée par le preset.

Réglages par hôte :
- KXKM-AI : `--micro-batch-pairs 512 --clear-cache --n-workers 8` ;
- macM1 : aucun de ces drapeaux (`HOST_SETTINGS["macm1"]`).

Pannes :
- un run qui plante ou s'arrête sur la garde de vivacité est relancé **une fois**, sur le même hôte, dans un nouveau répertoire (`-a2`) ;
- une deuxième panne arrête **la lane** (pas seulement la graine) : G2 dit « the PI decides before any further run ». La graine devient « non mesurable » si le PI ne décide pas autrement ;
- une panne = code de sortie ≠ 0 ou `FAIL:<conditions>` du contrôle de cellule (§4), y compris la porte d'apprentissage par run (sonde signalée après le pas 0, taux de pièce qui ne monte pas).

## 3. Lanes (prérequis 3)

Les deux lanes appliquent le §2 : ordre fixe, troncs vérifiés par SHA-256 avant le premier run, une relance par cellule, arrêt à la deuxième panne, sortie de l'entraîneur dans `*.result.sealed.json`, contrôle de cellule par `gate_g2_decide check-cell` (`OK` / `FAIL:<conditions>`), `STOP`, `driver.log`, copies du registre au début et à la fin. Relancer une lane avec le même `OUT` reprend : une cellule réussie est sautée, une tentative ratée ou tuée compte comme utilisée.

```bash
# KXKM-AI (no tmux; flock refuses a second lane on the same OUT)
B=/ai/saisail/n10c-aplus
REPO=$B/repo OUT=$B/runs TRUNK_DIR=$B/trunks TRUNK_SHA_9106=<SHA of §2> \
  nohup bash $B/repo/experiments/procgen_multiagent_gtm_comm/campaign/kxkm/gate_aplus_lane.sh \
  --min-free-gb 9.3 </dev/null >/dev/null 2>&1 &
```

```zsh
# macM1 (detaches itself under HEAVY_LOCK, day model paused for the block)
B=~/scratch/n10c-aplus
REPO=$B/repo OUT=$B/runs TRUNK_DIR=$B/trunks \
  zsh $B/repo/experiments/procgen_multiagent_gtm_comm/campaign/macm1/gate_aplus_lane.sh --min-free-gb 12
```

`--min-free-gb` : KXKM-AI, mémoire GPU libre avant chaque run (passée à `gpu_wait.sh` en MiB) ; macM1, mémoire unifiée libre (pourcentage de `memory_pressure` × mémoire physique), attendue avant chaque run. Le préentraînement du tronc 9106 n'est pas dans la lane (§3.1).

## 3 bis. Commandes exactes (équivalent manuel des lanes)

### 3.1 KXKM-AI (`ssh kxkm-ai`, utilisateur `electron`, sous `/ai/saisail`)

```bash
# once: clean clone at the launch commit (prerequisites merged)
B=/ai/saisail/n10c-aplus; mkdir -p $B
git clone https://github.com/hypneum-lab/nerve-wml.git $B/repo
git -C $B/repo checkout <LAUNCH_SHA>            # write the SHA here: ______
export CUDA_HOME=/ai/saisail/n10c-bench/cuda_home PYTHONPATH=.
PY=/ai/saisail/n10c-run2/.venv/bin/python      # run-2 venv (verify: pip freeze == n10c-bench lock)
OUT=$B/runs; mkdir -p $OUT $B/trunks
# trunk 9104: copy from the Adam study (presumed path, CHECK by SHA)
cp /ai/saisail/n10c-dev/adam-study/pretrain-s9104/trunk.safetensors $B/trunks/trunk-9104.safetensors
sha256sum $B/trunks/trunk-9104.safetensors   # must equal 894688cd…f16f

# trunk 9106 (pretrain_dev, 1M, KXKM settings; about 6 min)
source $B/repo/experiments/procgen_multiagent_gtm_comm/campaign/kxkm/gpu_wait.sh
log() { echo "$(date '+%F %T') $*" >> $OUT/driver.log; }
wait_free
(cd $B/repo && nice -n 10 $PY -m experiments.procgen_multiagent_gtm_comm.harness.pretrain \
  --seed 9106 --steps 1000000 --out $B/trunks/pretrain-9106 \
  --micro-batch-pairs 512 --clear-cache > $B/trunks/pretrain-9106.log 2>&1)
cp $B/trunks/pretrain-9106/trunk.safetensors $B/trunks/trunk-9106.safetensors
sha256sum $B/trunks/trunk-9106.safetensors   # record in §2 BEFORE any gate run

# one cell (repeat in the order of §2; never in parallel)
cell() {  # seed trunk_seed arm attempt
  local s=$1 t=$2 arm=$3 a=$4 d=$OUT/closed-$1-$3-t$2-a$4
  [ -e $OUT/STOP ] && { log "STOP file: not starting $d"; return 2; }
  [ ! -e $d ] || { log "refused: $d exists"; return 1; }
  wait_free; nvidia-smi -q > $d.nvsmi-before.txt 2>&1
  log "start seed $s trunk $t arm $arm attempt $a"
  (cd $B/repo && nice -n 10 $PY -m experiments.procgen_multiagent_gtm_comm.harness.train \
    --arm $arm --seed $s --condition closed --steps 10000000 --out $d \
    --log-every 200000 --ckpt-every 1000000 --preset fix-c-coin-pretrain \
    --dev-kind gate_dev --micro-batch-pairs 512 --clear-cache --n-workers 8 \
    --init-trunk $B/trunks/trunk-$t.safetensors \
    > $d.result.sealed.json 2> $d.stderr.log); local rc=$?
  nvidia-smi -q > $d.nvsmi-after.txt 2>&1; log "end $d rc=$rc"; return $rc
}
# order (each line: attempt 1, else attempt 2, else stop the seed)
cell 9104 9104 eng 1    || cell 9104 9104 eng 2
cell 9104 9104 nocomm 1 || cell 9104 9104 nocomm 2
cell 9114 9104 nocomm 1 || cell 9114 9104 nocomm 2      # nocomm_rep of 9104
cell 9106 9106 nocomm 1 || cell 9106 9106 nocomm 2
cell 9116 9106 nocomm 1 || cell 9116 9106 nocomm 2      # nocomm_rep of 9106
cell 9106 9106 eng 1    || cell 9106 9106 eng 2
```

Notes :
- Lancer avec `nohup bash <script> </dev/null >/dev/null 2>&1 &`, **sans session tmux** : `gpu_wait.sh` attend les sessions `n10c-dev-*`, et une session à nous se bloquerait elle-même.
- `gpu_wait.sh` sérialise avec E1-small (venv `/ai/saisail/e1-small/venv/`, sessions `e1s-*`) et avec l'évaluation N10-C graine 0 (session `n10c`) (G5).
- Un « rc ≠ 0 » ou un `FAIL` du contrôle d'état (§4) déclenche la relance.
- Vérification des pics VRAM : `nvsmi-after` de la première cellule. Le pic `nvidia-smi` y est consigné. Pic MLX attendu : ≈ 6,7 Gio (`P6` §6.4).

### 3.2 macM1 (`ssh macm1`)

```zsh
B=~/scratch/n10c-aplus; mkdir -p $B/runs $B/trunks
git clone https://github.com/hypneum-lab/nerve-wml.git $B/repo && git -C $B/repo checkout <LAUNCH_SHA>
PY=~/scratch/n10c-run2/.venv/bin/python        # run-2 venv (arm64 procgen + patch; verify)
# trunk 9105 copied from KXKM-AI (scp via GrosMac), then:
shasum -a 256 $B/trunks/trunk-9105.safetensors # must equal d4916633…8a7d
cat > $B/block-9105.sh <<'EOS'
#!/bin/zsh
B=~/scratch/n10c-aplus; PY=~/scratch/n10c-run2/.venv/bin/python; OUT=$B/runs
cd $B/repo; export PYTHONPATH=.
cell() { local s=$1 t=$2 arm=$3 a=$4 d=$OUT/closed-$1-$3-t$2-a$4
  [ -e $OUT/STOP ] && return 2; [ ! -e $d ] || return 1
  print "$(date '+%F %T') start $d" >> $OUT/driver.log
  nice -n 10 $PY -m experiments.procgen_multiagent_gtm_comm.harness.train \
    --arm $arm --seed $s --condition closed --steps 10000000 --out $d \
    --log-every 200000 --ckpt-every 1000000 --preset fix-c-coin-pretrain \
    --dev-kind gate_dev --init-trunk $B/trunks/trunk-$t.safetensors \
    > $d.result.sealed.json 2> $d.stderr.log; local rc=$?
  print "$(date '+%F %T') end $d rc=$rc" >> $OUT/driver.log; return $rc }
cell 9105 9105 eng 1    || cell 9105 9105 eng 2
cell 9105 9105 nocomm 1 || cell 9105 9105 nocomm 2
cell 9115 9105 nocomm 1 || cell 9115 9105 nocomm 2      # nocomm_rep of 9105
EOS
# one heavy job at a time (shared lock), day model paused for the whole block
HEAVY_LOCK=~/scratch/.heavy.lock zsh $B/repo/experiments/procgen_multiagent_gtm_comm/campaign/macm1/day_model_paused.sh \
  --detach $B/launcher.state $B/launcher.log -- caffeinate -is zsh $B/block-9105.sh
```

Si le lanceur meurt : `rm ~/.mtplx/paused`, puis vérifier que :8080 répond (runbook N10-C). Fenêtre visée : vendredi 13 nov 20:00 → samedi 14 nov vers 07:00 (`P6` §6.3).

## 4. Suivi pendant les runs (aucune valeur lue)

- État : `tail $OUT/driver.log` ; présence de `ckpt_<n>.safetensors` ; `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` (KXKM-AI).
- **Interdit avant le gel :** `cat` / `jq` / `less` sur `*.result.sealed.json`, sur `run.json` (champs `success_curve`, `coin_rate_curve`, `losses`) et sur les stdout. On ne lit pas non plus les courbes avec un outil de résumé.
- Contrôle d'état par cellule : `gate_g2_decide check-cell`. Il n'imprime que `OK` ou `FAIL:<conditions>` : identité (bras, graine, `closed`, preset), stub, `training_failure`, `liveness_abort`, pas faits = 10 M, SHA du tronc, mode `abort` tous les 100 k, réglages d'hôte, couverture des sondes, présence et finitude des 5 derniers logs, fenêtre de 100, puis porte d'apprentissage (`liveness_flagged`, `coin_rate_rise`). Les lanes l'appellent. À la main :
  `PYTHONPATH=. $PY -m experiments.procgen_multiagent_gtm_comm.harness.gate_g2_decide check-cell <cell dir> --arm nocomm --seed 9114 --trunk-seed 9104 --host kxkm` (ajouter `--trunk-sha <hex>` pour 9106).
- **STOP :** `touch $OUT/STOP`. Aucune nouvelle cellule ne démarre ; la cellule en cours finit. Un arrêt en cours de run (`kill`) rend la cellule « panne d'infrastructure », relancée une fois.
- Journal : `driver.log` ; registre `manifest/smoke_ledger.json` de chaque copie de travail, avec une note par entrée : hôte, preset, code, et appariement « `nocomm_rep` de S » pour 9114, 9115 et 9116. Les entrées sont fusionnées par id de run dans le dépôt, comme pour les portes précédentes.

## 5. Décision (S5)

```bash
# after the 9 cells (plus reruns) exist on both hosts; macM1 records copied to KXKM-AI or GrosMac
PYTHONPATH=. $PY -m experiments.procgen_multiagent_gtm_comm.harness.gate_g2_decide decide \
  --runs-root <KXKM-AI runs> --runs-root <macM1 runs> \
  --trunk-sha 9106=<SHA of §2> --sealed-out <new file>/g2-aplus.sealed.json
```

Ce script est le prérequis 2 (commit `caba588`, pas encore sur `master`). Il n'imprime **qu'une ligne**, la catégorie. Toutes les valeurs (succès par run, D_s, D0_s, D̄, contrôle `eng`, tentatives retenues, SHA-256 des `run.json`) vont dans le fichier scellé. Ce fichier ne doit pas exister ; il est créé en lecture seule (0400) et jamais écrasé. Règles, dans cet ordre (plan §1.3, `P6` §6.5) :
1. porte d'apprentissage par run : pour chaque cellule, la tentative 1 si `check-cell` dit `OK`, sinon la tentative 2. Une graine sans `eng` ou sans `nocomm` utilisable → **`NOT_MEASURABLE_LEARNING`** (« NON MESURABLE (apprentissage) ») ;
2. contrôle `eng` (succès final ≥ 0,05 et au-dessus du premier log) en échec sur au moins 2 graines → **`NOT_MEASURABLE_ENG_CONTROL`** (« NON MESURABLE (contrôle eng) ») ;
3. contrôle de bruit : |D0_s| ≥ 0,10, ou `nocomm_rep` inutilisable, sur au moins 2 graines → **`INCONCLUSIVE_NOISE`** (« NON CONCLUANT (bruit) ») ;
4. D̄ ≥ 0,20 → **`PASS`**, sinon **`FAIL`**.

Succès d'équipe d'un run = moyenne des 5 derniers logs (G2). Les seuils sont inclusifs, avec une tolérance de 1e-9 sur l'arithmétique flottante. Un nom de cellule présent sous deux racines, ou une tentative 3, est refusé (code 2, message sans valeur).

La catégorie est ajoutée à G2 avec la date. Les valeurs restent scellées (§0, quatrième case).

## 6. Après

- KXKM-AI : aucun service n'est arrêté pour A+ ; rien à restaurer.
- macM1 : le lanceur lève la pause du modèle de jour.
- Archives : `runs/` des deux hôtes vers electron-server, comme N10-C, sans lecture.

## 7. Proposition de texte pour l'amendement daté de G2 (prérequis 4, NON SIGNÉE)

À ajouter par le PI en fin de `MS`, daté, avant le premier run A+. Texte proposé, en anglais comme `MS` ; les crochets sont à remplir ou à trancher par le PI :

> **G8. Necessity gate at k = 3, mixed hosts (A+) ([date, time CET], PI decision, before any A+ run; non-blind with respect to the seed-0 results read on 2026-11-05).** G2 is run again as a development contrast on three development seeds, **9104, 9105 and 9106**, distinct from the G7 pilot seeds (9101–9103). Each seed stays on one host with its own pre-trained trunk: 9104 and 9106 on KXKM-AI, 9105 on macM1 (declared substrate deviation `docs/milestones/n10c-bis-substrate-split-2026-11-06.md`; this changes G5 for A+ only). Per seed, three cells, condition `closed`, preset `fix-c-coin-pretrain`, 10,000,000 pair-steps, ledger kind `gate_dev`: `eng` (seed S), `nocomm` (seed S) and **`nocomm_rep`** (`nocomm` on seed S + 10 with the trunk of S: 9114, 9115, 9116), in the order of `RUNBOOK-aplus.md` §2. Each cell gets one rerun; a second failure of any cell stops the whole lane of that host and the PI decides before any further run (PI decision 2026-11-06). The measure is that of G2. The decision is computed by `harness/gate_g2_decide.py` at commit `caba588` and applies, in order: (1) per-run learnability gate (no liveness flag after step 0, no abort, own-coin rate rising); a seed without a usable `eng` and `nocomm` run makes the gate **not measurable**; (2) positive control: `eng` final team success ≥ 0.05 and above its first log; failure on ≥ 2 seeds makes the gate **not measurable**; (3) noise control: D0_s = `nocomm_rep` − `nocomm`; |D0_s| ≥ 0.10 (the N10-C SESOI), or no usable `nocomm_rep` (a missing rerun counts as a failed noise control; PI decision 2026-11-06), on ≥ 2 seeds makes the gate **inconclusive**; (4) **pass** if D̄ = mean over seeds of (`eng` − `nocomm`) ≥ 0.20, else **fail**. Under the planning assumption σ_D = 0.142 (an inference, not a pilot estimate), the type I error of rule (4) is about 0.7 % at D = 0 and its power about 0.89 at D = 0.30 (plan §1.1). Only the category is printed and reported; every value stays sealed until the freeze of B′ or of N10-D, whichever comes first, or until a dated PI declaration (PI decision 2026-11-06). The consequences of G3 apply to the category: pass → B′ may be prepared; fail → N10-D; not measurable or inconclusive → [TO BE CONFIRMED by the PI with the next scientific watch, before the go: one rerun on three new development seeds (9107–9109) then N10-D if the category repeats, or N10-D directly]. The gate stays exploratory (G4). Ledger: `eng` 25, `nocomm` 13, `pretrain_dev` 10 (commit `293877a`, GB300 G7 entries merged or reconstructed as stated there).

Décisions PI du 2026-11-06 (vers 17:00 CET, QCM en session) sur les points laissés ouverts :

| Point | Décision | État |
|---|---|---|
| Arrêt à la deuxième panne d'une cellule | arrêt de **toute la lane de l'hôte**, le PI décide avant tout run suivant | tranché, porté dans le texte |
| `nocomm_rep` manquant | compte comme un **échec du contrôle de bruit** (porte « non concluante » si ≥ 2 graines) | tranché, porté dans le texte |
| Durée du scellement des valeurs | jusqu'au **gel de B′ ou de N10-D**, ou déclaration PI datée | tranché, porté dans le texte |
| Conséquence d'une catégorie « non mesurable » / « non concluant » | **à reconfirmer avec la prochaine veille scientifique**, avant le go ; options en lice : une relance sur 9107–9109 puis N10-D, ou N10-D directement | **ouvert** |

Le texte reste NON SIGNÉ tant que le dernier point est ouvert ; la signature (ajout de G8 en fin de `MS`, daté) et le go du §0 se font ensemble.

