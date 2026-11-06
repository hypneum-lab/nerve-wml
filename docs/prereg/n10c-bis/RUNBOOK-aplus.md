# RUNBOOK — porte de nécessité A+ de N10-C-bis (k = 3, hôte mixte)

**Statut :** PRÉPARÉ, NON LANCÉ. Rien ne tourne tant que la case « go PI daté » (§0) est vide, et tant que les prérequis du §1 ne sont pas fusionnés sur `master`.
**Références :** plan `three-seeds-plan-2026-11-06.md` §6 (abrégé `P6`) ; déviation `docs/milestones/n10c-bis-substrate-split-2026-11-06.md` (abrégée `DEV-S`) ; addendum `docs/milestones/2026-10-25-n10c-bis-control-arms.md` G2-G6 (abrégé `MS`).

## 0. Go PI

- [ ] **Go PI daté : ______** (date, heure CET). Ce go vaut aussi signature de `DEV-S` §4.
- [ ] Q3 confirmée : graines 9104, 9105, 9106, et `nocomm_rep` sur 9114, 9115, 9116. Sinon, nouvelles graines : ______
- [ ] Q4 confirmée : G2 inchangé (D̄ ≥ 0,20) avec les contrôles du plan §1.3. Sinon : ______
- [ ] Règle « résultats non lus » de `P6` §6.5 confirmée (verdict catégoriel seul).

## 1. Prérequis (commits séparés et datés, avant le premier run ; aucun n'est fait dans la PR #114)

| # | Quoi | Où | État au 06/11 |
|---|---|---|---|
| 1 | Crédits `gate_dev` : `eng` 25, `nocomm` 18 ; `pretrain_dev` 10 | `harness/devledger.py` (`GATE_DEV_ALLOWANCE`, `GATE_ALLOWANCE`) | **manque**. `eng` 13/19 et `nocomm` 0/6 sur `master` ; `pretrain_dev` 8/8. À recaler sur le registre GB300 de G7 s'il a consommé des crédits |
| 2 | `gate_g2_decide.py` (verdict catégoriel k = 3, 10 M, `closed`) et ses tests | `harness/` et `tests/unit/n10c/` | **manque**. `gate_report.verdict` ne juge que des runs de 1 M |
| 3 | Lanes `gate_aplus_lane.sh` (KXKM-AI et macM1) | `campaign/kxkm/`, `campaign/macm1/` | **manque**. Les commandes du §3 sont l'équivalent manuel |
| 4 | Amendement daté de G2 (k = 3, graines, `nocomm_rep`, contrôles, table hôtes, §6.5) | `MS`, ajout en fin de fichier | **manque** |
| 5 | Tests du dépôt sur macM1 (`uv run python -m pytest tests/unit/n10c`) au commit de lancement | `ssh macm1` | à faire |
| 6 | Copie du tronc 9105 sur macM1, SHA-256 vérifié | macM1 | à faire |

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
- une deuxième panne arrête la graine (« non mesurable ») et le PI décide (G2).

## 3. Commandes exactes (équivalent manuel des lanes)

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
- Contrôle d'état par cellule. Il n'imprime que `OK` ou `FAIL:<condition>`, sur le modèle de `gate_g7_lane.sh`, `status_ok` : stub, `training_failure`, `liveness_abort`, pas faits = 10 M, SHA du tronc, mode `abort`, réglages d'hôte, présence et finitude des 5 derniers logs, fenêtre de 100. La lane du prérequis 3 l'intègre. À la main, copier la fonction `status_ok` de `gate_g7_lane.sh:75-97`, avec le tuple de réglages de l'hôte (macM1 : `(None, False, 1)`).
- **STOP :** `touch $OUT/STOP`. Aucune nouvelle cellule ne démarre ; la cellule en cours finit. Un arrêt en cours de run (`kill`) rend la cellule « panne d'infrastructure », relancée une fois.
- Journal : `driver.log` ; registre `manifest/smoke_ledger.json` de chaque copie de travail, avec une note par entrée : hôte, preset, code, et appariement « `nocomm_rep` de S » pour 9114, 9115 et 9116. Les entrées sont fusionnées par id de run dans le dépôt, comme pour les portes précédentes.

## 5. Décision (S5)

```bash
# after the 9 cells (plus reruns) exist on both hosts; macM1 records copied to KXKM-AI or GrosMac
PYTHONPATH=. $PY -m experiments.procgen_multiagent_gtm_comm.harness.gate_g2_decide \
  --pairs 9104:9104,9114 9105:9105,9115 9106:9106,9116 \
  --runs-root <dir holding closed-* of both hosts> --categorical-only
```

Ce script est le prérequis 2. **Il n'existe pas sur `master` au 06/11.** Il n'imprime qu'une catégorie, dans cet ordre de règles (plan §1.3, `P6` §6.5) :
1. porte d'apprentissage par run (adaptée à 10 M) → une graine sans `eng` et `nocomm` appris est « non mesurable » ;
2. contrôle `eng` : échec sur au moins 2 graines → **NON MESURABLE** ;
3. contrôle de bruit : |D0_s| ≥ 0,10 sur au moins 2 graines → **NON CONCLUANT** ;
4. D̄ ≥ 0,20 → **PASS**, sinon **FAIL**.

La catégorie est ajoutée à G2 avec la date. Les valeurs restent scellées (§0, quatrième case).

## 6. Après

- KXKM-AI : aucun service n'est arrêté pour A+ ; rien à restaurer.
- macM1 : le lanceur lève la pause du modèle de jour.
- Archives : `runs/` des deux hôtes vers electron-server, comme N10-C, sans lecture.
