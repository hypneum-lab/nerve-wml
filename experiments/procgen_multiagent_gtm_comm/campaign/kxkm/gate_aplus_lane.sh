#!/usr/bin/env bash
# N10-C-bis necessity gate A+ on KXKM-AI (plan docs/prereg/n10c-bis/
# three-seeds-plan-2026-11-06.md section 6; RUNBOOK-aplus.md sections 2-4):
# seeds 9104 and 9106, each with its own trunk; per seed the cells eng,
# nocomm and nocomm_rep (= nocomm on seed S+10 with the trunk of S), in the
# fixed order of the runbook; condition closed, 10,000,000 pair-steps,
# preset fix-c-coin-pretrain, ledger kind gate_dev, KXKM-AI settings.
#
# One run at a time, never next to another GPU job of ours (gpu_wait.sh:
# E1-small, other N10-C lanes; foreign jobs count only through the free
# memory). Each cell gets at most one rerun (G2); a second failure stops
# the lane (the PI decides before any further run). Re-running the lane
# with the same OUT resumes: a cell whose attempt passed is skipped, a
# failed or killed attempt counts as used. Team success is never printed:
# the trainer's stdout goes to <cell>.result.sealed.json, and the cell
# check (harness/gate_g2_decide.py check-cell) prints only OK or
# FAIL:<conditions>. STOP: touch $OUT/STOP (no new cell starts; the
# running one finishes). No tmux session: gpu_wait.sh waits on n10c-dev-*
# sessions and would deadlock against the lane's own wait.
#
#   REPO=/ai/saisail/n10c-aplus/repo OUT=/ai/saisail/n10c-aplus/runs \
#   TRUNK_DIR=/ai/saisail/n10c-aplus/trunks TRUNK_SHA_9106=<64 hex> \
#   nohup bash gate_aplus_lane.sh [--min-free-gb 9.3] </dev/null >/dev/null 2>&1 &
#
# TRUNK_DIR holds trunk-9104.safetensors and trunk-9106.safetensors; both
# are checked by SHA-256 before the first run. --min-free-gb: free GPU
# memory needed before each run (default 9.3 GiB = the 9,500 MiB of
# gpu_wait.sh).
set -u
usage() { echo "usage: $0 [--min-free-gb GB]" >&2; exit 64; }
MIN_FREE_GB=9.3
while [ $# -gt 0 ]; do
  case $1 in
    --min-free-gb) [ $# -ge 2 ] || usage; MIN_FREE_GB=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$MIN_FREE_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || usage
MIN_FREE_MIB=$(awk -v g="$MIN_FREE_GB" 'BEGIN { printf "%d", g * 1024 }')
export MIN_FREE_MIB
REPO="${REPO:?REPO = clean clone at the launch commit}"
OUT="${OUT:?OUT = run directory of this host}"
TRUNK_DIR="${TRUNK_DIR:?TRUNK_DIR = holds trunk-<seed>.safetensors}"
PY="${PY:-/ai/saisail/n10c-run2/.venv/bin/python}"
HOST=kxkm
STEPS=10000000
declare -A TRUNK_SHA=(
  [9104]=894688cd4a62b868ea1f18d7021be57a64acb52699a8742048c9e2508da4f16f
  [9106]="${TRUNK_SHA_9106:-}"
)
# seed:cell, fixed before any run (RUNBOOK-aplus.md section 2)
ORDER=(9104:eng 9104:nocomm 9104:nocomm_rep 9106:nocomm 9106:nocomm_rep 9106:eng)
export CUDA_HOME="${CUDA_HOME:-/ai/saisail/n10c-bench/cuda_home}" PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
LOG="$OUT/driver.log"
log() { echo "$(date '+%F %T%z') $*" >> "$LOG"; }
stop() { log "LANE STOP: $*"; echo "$*" > "$OUT/LANE_STOP"; exit "${2:-0}"; }

exec 9> "$OUT/.lane.lock"
flock -n 9 || { echo "another A+ lane holds $OUT/.lane.lock" >&2; exit 2; }
rm -f "$OUT/LANE_STOP"
[[ "${TRUNK_SHA[9106]}" =~ ^[0-9a-f]{64}$ ]] \
  || stop "TRUNK_SHA_9106 must be the 64-hex SHA-256 recorded in the runbook" 1
[ -x "$PY" ] || stop "no python $PY" 1
# shared wait rules (MIN_FREE_MIB from --min-free-gb)
source "$(dirname "${BASH_SOURCE[0]}")/gpu_wait.sh"

TRAINER=""
trap '[ -n "$TRAINER" ] && kill "$TRAINER" 2>/dev/null; stop "operator stop (signal)"' TERM INT HUP
log "lane start host $HOST code $(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l) min_free_mib $MIN_FREE_MIB"
cp "$REPO/experiments/procgen_multiagent_gtm_comm/manifest/smoke_ledger.json" \
   "$OUT/smoke_ledger.initial.$(date +%Y%m%dT%H%M%S).json"
for s in 9104 9106; do  # every trunk BEFORE the first run
  f="$TRUNK_DIR/trunk-$s.safetensors"
  [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "${TRUNK_SHA[$s]}" ] \
    || stop "trunk of seed $s missing or SHA-256 mismatch (before any run)" 1
done

check() {  # dir seed arm trunk -> OK / FAIL:<conditions> (never a value)
  (cd "$REPO" && "$PY" -m experiments.procgen_multiagent_gtm_comm.harness.gate_g2_decide \
    check-cell "$1" --arm "$3" --seed "$2" --trunk-seed "$4" --host $HOST \
    --trunk-sha "${TRUNK_SHA[$4]}" 2>> "$OUT/check.err")
}

run_one() {  # seed arm trunk attempt -> 0 usable, 1 failed
  local s=$1 arm=$2 t=$3 att=$4 d
  d="$OUT/closed-$1-$2-t$3-a$4"
  [ ! -e "$OUT/STOP" ] || stop "STOP file: $d not started"
  wait_free
  [ ! -e "$OUT/STOP" ] || stop "STOP file: $d not started"
  nvidia-smi -q > "$d.nvsmi-before.txt" 2>&1 || true
  log "start seed $s arm $arm trunk $t attempt $att"
  ( cd "$REPO" && exec nice -n 10 "$PY" -m \
      experiments.procgen_multiagent_gtm_comm.harness.train \
      --arm "$arm" --seed "$s" --condition closed --steps $STEPS --out "$d" \
      --log-every 200000 --ckpt-every 1000000 --preset fix-c-coin-pretrain \
      --dev-kind gate_dev --micro-batch-pairs 512 --clear-cache --n-workers 8 \
      --init-trunk "$TRUNK_DIR/trunk-$t.safetensors" \
      > "$d.result.sealed.json" 2> "$d.stderr.log" ) &
  TRAINER=$!
  local rc=0; wait "$TRAINER" || rc=$?; TRAINER=""
  nvidia-smi -q > "$d.nvsmi-after.txt" 2>&1 || true
  local st; st=$(check "$d" "$s" "$arm" "$t")
  log "end seed $s arm $arm trunk $t attempt $att rc=$rc status=$st"
  [ "$rc" = 0 ] && [ "$st" = OK ]
}

for item in "${ORDER[@]}"; do
  S=${item%%:*}; cell=${item#*:}
  case $cell in
    eng) s=$S arm=eng ;;
    nocomm) s=$S arm=nocomm ;;
    nocomm_rep) s=$((S + 10)) arm=nocomm ;;
  esac
  done_cell=0
  for att in 1 2; do
    d="$OUT/closed-$s-$arm-t$S-a$att"
    if [ -e "$d" ] || [ -e "$d.result.sealed.json" ]; then  # earlier session
      st=$(check "$d" "$s" "$arm" "$S")
      log "found $d status=$st"
      [ "$st" = OK ] && { done_cell=1; break; }
      continue  # a failed or killed attempt uses its slot
    fi
    if run_one "$s" "$arm" "$S" "$att"; then done_cell=1; break; fi
  done
  [ $done_cell = 1 ] || stop "second failure: seed $s arm $arm trunk $S (G2: the PI decides)"
  log "cell done: seed $S $cell"
done
cp "$REPO/experiments/procgen_multiagent_gtm_comm/manifest/smoke_ledger.json" \
   "$OUT/smoke_ledger.final.json"
log "lane done"
