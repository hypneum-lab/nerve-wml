#!/usr/bin/env bash
# Freeze plan, phase T on KXKM-AI (PI decisions 2026-10-28): the pre-freeze
# registered pre-trainings of the KXKM seeds 73, 101, 131 (T2, shared
# trunks) and of the 10 control units of seed 101 (T3, own trunks),
# `harness.pretrain --pre-freeze` (exact registered procedure; KXKM settings
# forced; ledger kind pretrain_registered). One job at a time, each after
# gpu_wait.sh says the card is free (foreign GPU jobs are never touched). An
# unusable trunk gets ONE restart (the ledger refuses a third attempt); then
# the lane stops and the PI decides. Needs the PI's go; never run otherwise.
#   REPO=/ai/saisail/n10c-freeze/repo OUT=/ai/saisail/n10c-freeze/trunks \
#   [PARTS="T2 T3"] nohup bash pretrain_lane.sh </dev/null >/dev/null 2>&1 &
set -u
REPO="${REPO:?REPO = clean clone at the pushed freeze-preparation commit}"
OUT="${OUT:?OUT = output root (one directory per trunk)}"
PY="${PY:-/ai/saisail/n10c-run/.venv/bin/python}"
PARTS="${PARTS:-T2 T3}"
export CUDA_HOME="${CUDA_HOME:-/ai/saisail/n10c-bench/cuda_home}" PYTHONPATH=.
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$OUT/driver.log"; }
source "$(dirname "${BASH_SOURCE[0]}")/gpu_wait.sh"

usable() {  # dir -> 0 if its pretrain.json reads usable: true
  [ -f "$1/pretrain.json" ] && "$PY" -c 'import json,sys
sys.exit(0 if json.load(open(sys.argv[1]))["usable"] else 1)' "$1/pretrain.json"
}

one() {  # name seed [--per-run arm cond]
  local name=$1 seed=$2; shift 2
  local d="$OUT/$name"
  for attempt in 1 2; do
    if [ -e "$d" ]; then
      if usable "$d"; then log "skip $name: usable trunk present"; return 0; fi
      mv "$d" "$d.unusable-$attempt"  # kept, reported
    fi
    wait_free
    log "start $name seed $seed $* (attempt $attempt)"
    (cd "$REPO" && nice -n 10 "$PY" -m experiments.procgen_multiagent_gtm_comm.harness.pretrain \
      --seed "$seed" --steps 1000000 --out "$d" --pre-freeze "$@" \
      > "$OUT/$name.log" 2>&1)
    log "end $name rc=$?"
    if usable "$d"; then return 0; fi
  done
  log "STOP: $name has no usable trunk after one restart; the PI decides"
  exit 2
}

log "lane start, code $(git -C "$REPO" rev-parse --short HEAD), parts $PARTS"
case " $PARTS " in *" T2 "*)
  for s in 73 101 131; do one "seed-$s" "$s"; done ;;
esac
case " $PARTS " in *" T3 "*)
  units=$(cd "$REPO" && "$PY" -c 'from experiments.procgen_multiagent_gtm_comm.campaign.queue import control_queue
for r in control_queue(): print(r.arm, r.condition)')
  while read -r arm cond; do
    [ -n "$arm" ] || continue
    one "seed-101-$cond-$arm" 101 --per-run "$arm" "$cond"
  done <<< "$units" ;;
esac
log "lane done"
