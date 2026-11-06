#!/bin/zsh
# N10-C-bis necessity gate A+ on macM1 (plan docs/prereg/n10c-bis/
# three-seeds-plan-2026-11-06.md section 6; RUNBOOK-aplus.md sections 2-4):
# seed 9105 with its trunk; cells eng, nocomm and nocomm_rep (= nocomm on
# seed 9115 with the trunk of 9105), in the fixed order of the runbook;
# condition closed, 10,000,000 pair-steps, preset fix-c-coin-pretrain,
# ledger kind gate_dev, macM1 settings (no micro-batch / cache / worker
# flag, HOST_SETTINGS["macm1"]).
#
# The lane detaches itself under the shared heavy-job lock and with the
# second-brain day model paused for the whole block (day_model_paused.sh
# --detach, HEAVY_LOCK -> /usr/bin/lockf), then runs one cell at a time.
# Each cell gets at most one rerun (G2); a second failure stops the lane
# (the PI decides before any further run). Re-running with the same OUT
# resumes: a cell whose attempt passed is skipped, a failed or killed
# attempt counts as used. Team success is never printed: the trainer's
# stdout goes to <cell>.result.sealed.json and the cell check
# (harness/gate_g2_decide.py check-cell) prints only OK or
# FAIL:<conditions>. STOP: touch $OUT/STOP.
#
#   REPO=~/scratch/n10c-aplus/repo OUT=~/scratch/n10c-aplus/runs \
#   TRUNK_DIR=~/scratch/n10c-aplus/trunks \
#   zsh gate_aplus_lane.sh [--min-free-gb 12]
#
# --min-free-gb: free unified memory (memory_pressure free percentage x
# physical memory) needed before each run, waited for (default 12 GiB; the
# 1M macM1 gate run peaked at 10.27 GiB). HEAVY_LOCK defaults to
# ~/scratch/.heavy.lock. Launcher state / log: $OUT/launcher.{state,log}.
set -u
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:$PATH
HERE="${0:A:h}"
usage() { print -r -- "usage: $0 [--min-free-gb GB]" >&2; exit 64; }
MIN_FREE_GB=12
args=("$@")
while [ $# -gt 0 ]; do
  case $1 in
    --min-free-gb) [ $# -ge 2 ] || usage; MIN_FREE_GB=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$MIN_FREE_GB" =~ '^[0-9]+([.][0-9]+)?$' ]] || usage
REPO="${REPO:?REPO = clean clone at the launch commit}"
OUT="${OUT:?OUT = run directory of this host}"
TRUNK_DIR="${TRUNK_DIR:?TRUNK_DIR = holds trunk-9105.safetensors}"
PY="${PY:-$HOME/scratch/n10c-run2/.venv/bin/python}"
REPO=${REPO:A} OUT=${OUT:A} TRUNK_DIR=${TRUNK_DIR:A}
export REPO OUT TRUNK_DIR PY
mkdir -p "$OUT"

if [ -z "${GATE_APLUS_INNER:-}" ]; then  # outer call: detach under the lock
  export HEAVY_LOCK="${HEAVY_LOCK:-$HOME/scratch/.heavy.lock}"
  exec zsh "$HERE/day_model_paused.sh" --detach "$OUT/launcher.state" \
    "$OUT/launcher.log" -- env GATE_APLUS_INNER=1 caffeinate -is \
    zsh "${0:A}" "${args[@]}"
fi

HOST=macm1
STEPS=10000000
typeset -A TRUNK_SHA
TRUNK_SHA[9105]=d4916633ae7203f5e6ba5353451abf32fd280fa70c1b2404d1fb5803e2538a7d
ORDER=(9105:eng 9105:nocomm 9105:nocomm_rep)  # RUNBOOK-aplus.md section 2
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
LOG="$OUT/driver.log"
log() { print -r -- "$(date '+%F %T%z') $*" >> "$LOG"; }
stop() { log "LANE STOP: $1"; print -r -- "$1" > "$OUT/LANE_STOP"; exit "${2:-0}"; }
rm -f "$OUT/LANE_STOP"
[ -x "$PY" ] || stop "no python $PY" 1

free_gb() {  # free unified memory in GiB (integer), or empty on failure
  local pct mem
  pct=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  mem=$(sysctl -n hw.memsize 2>/dev/null)
  [[ "$pct" == <-> && "$mem" == <-> ]] || return 0
  print -r -- $(( mem * pct / 100 / 1073741824 ))
}
wait_mem() {
  local said=0 f
  while :; do
    f=$(free_gb)
    if [[ "$f" == <-> ]] && (( f >= MIN_FREE_GB )); then return 0; fi
    (( said )) || { log "waiting: free memory ${f:-?} GiB < $MIN_FREE_GB GiB"; said=1; }
    [ ! -e "$OUT/STOP" ] || stop "STOP file while waiting for memory"
    sleep 60
  done
}

TRAINER=""
trap '[ -n "$TRAINER" ] && kill "$TRAINER" 2>/dev/null; stop "operator stop (signal)"' TERM INT HUP
log "lane start host $HOST code $(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l | tr -d ' ') min_free_gb $MIN_FREE_GB"
cp "$REPO/experiments/procgen_multiagent_gtm_comm/manifest/smoke_ledger.json" \
   "$OUT/smoke_ledger.initial.$(date +%Y%m%dT%H%M%S).json"
f="$TRUNK_DIR/trunk-9105.safetensors"  # BEFORE the first run
[ -f "$f" ] && [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" = "${TRUNK_SHA[9105]}" ] \
  || stop "trunk of seed 9105 missing or SHA-256 mismatch (before any run)" 1

check() {  # dir seed arm trunk -> OK / FAIL:<conditions> (never a value)
  (cd "$REPO" && "$PY" -m experiments.procgen_multiagent_gtm_comm.harness.gate_g2_decide \
    check-cell "$1" --arm "$3" --seed "$2" --trunk-seed "$4" --host $HOST \
    --trunk-sha "${TRUNK_SHA[$4]}" 2>> "$OUT/check.err")
}

run_one() {  # seed arm trunk attempt -> 0 usable, 1 failed
  local s=$1 arm=$2 t=$3 att=$4 d="$OUT/closed-$1-$2-t$3-a$4" rc=0 st
  [ ! -e "$OUT/STOP" ] || stop "STOP file: $d not started"
  wait_mem
  log "start seed $s arm $arm trunk $t attempt $att"
  ( cd "$REPO" && exec nice -n 10 "$PY" -m \
      experiments.procgen_multiagent_gtm_comm.harness.train \
      --arm "$arm" --seed "$s" --condition closed --steps $STEPS --out "$d" \
      --log-every 200000 --ckpt-every 1000000 --preset fix-c-coin-pretrain \
      --dev-kind gate_dev --init-trunk "$TRUNK_DIR/trunk-$t.safetensors" \
      > "$d.result.sealed.json" 2> "$d.stderr.log" ) &
  TRAINER=$!
  wait "$TRAINER" || rc=$?
  TRAINER=""
  st=$(check "$d" "$s" "$arm" "$t")
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
  (( done_cell )) || stop "second failure: seed $s arm $arm trunk $S (G2: the PI decides)"
  log "cell done: seed $S $cell"
done
cp "$REPO/experiments/procgen_multiagent_gtm_comm/manifest/smoke_ledger.json" \
   "$OUT/smoke_ledger.final.json"
log "lane done"
