#!/usr/bin/env bash
# N10-C-bis necessity gate on the DGX Station GB300 (G7 host annex,
# docs/prereg/n10c-bis-gate-gb300/g7-host-annex.md, sections 3, 8, 9).
# One run at a time; fixed seed and arm order; one ledgered rerun per
# failed run (G2); the session stops on a second failure, a GPU process of
# ours other than the run, a trunk mismatch or the time rule. Team success
# is never printed: the trainer's stdout (which carries the success curve)
# goes to a sealed file that is not read before the session has ended.
#
# Required environment (absolute paths):
#   REPO       clean clone at the merge commit of the annex
#   PY         python of the reviewed run venv
#   OUT_ROOT   new session directory (must not exist)
#   TRUNK_DIR  holds trunk-<seed>.safetensors for the seeds of SEEDS
#   END_EPOCH  end of access, Unix seconds
#   RATE_ENG / RATE_NOCOMM  S5 pair_steps_per_s_incl_warmup per arm
#   GPU_UUID   UUID of the GB300 GPU (nvidia-smi -L)
#   CORES      CPU list for taskset (disjoint from any other job)
set -euo pipefail
: "${REPO:?}" "${PY:?}" "${OUT_ROOT:?}" "${TRUNK_DIR:?}" "${END_EPOCH:?}"
: "${RATE_ENG:?}" "${RATE_NOCOMM:?}" "${GPU_UUID:?}" "${CORES:?}"
STEPS=10000000
MARGIN_S=7200     # repatriation margin (annex section 9)
SAFETY=1.5        # factor on the S5 estimate (annex section 9)
# seed:first-arm order, fixed before the session (annex section 3)
ORDER=("9101:eng nocomm" "9102:nocomm eng" "9103:eng nocomm" "9104:nocomm eng")
declare -A TRUNK_SHA=(
  [9101]=0c27f2daa0a976e6407a85d6dfa6eafe20a6285c6324795b2194366b1a9aa704
  [9102]=b6805dee76c2f62be352383899d3b3b199458babbac1d260acdfffb78aa62bf1
  [9103]=8c0b72ee4f362e133771f6e0de93e6bac96ced0edb077b8b63f2c974cfd24a7a
  [9104]=894688cd4a62b868ea1f18d7021be57a64acb52699a8742048c9e2508da4f16f
)
[ ! -e "$OUT_ROOT" ] || { echo "OUT_ROOT exists: refuse" >&2; exit 64; }
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/driver.log"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
stop() { log "SESSION STOP: $*"; echo "$*" > "$OUT_ROOT/SESSION_STOP"; exit 0; }
export CUDA_VISIBLE_DEVICES="$GPU_UUID" PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
"$PY" -I -c "import math,sys; r=[float(x) for x in sys.argv[1:]]; sys.exit(0 if all(math.isfinite(x) and x>0 for x in r) else 1)" \
  "$RATE_ENG" "$RATE_NOCOMM" || stop "RATE_ENG / RATE_NOCOMM must be finite and > 0"
[ "$(nvidia-smi --id="$GPU_UUID" --query-gpu=compute_cap --format=csv,noheader | tr -d ' ')" = 10.3 ] \
  || stop "GPU $GPU_UUID does not report compute capability 10.3"
TRAINER=""
trap '[ -n "$TRAINER" ] && kill "$TRAINER" 2>/dev/null; stop "operator stop (signal)"' TERM INT
log "lane start code $(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
cp "$REPO/experiments/procgen_multiagent_gtm_comm/manifest/smoke_ledger.json" \
   "$OUT_ROOT/smoke_ledger.initial.json"

# all trunks are checked BEFORE the first gate run; the order is fixed after
for item in "${ORDER[@]}"; do
  s=${item%%:*}; f="$TRUNK_DIR/trunk-$s.safetensors"
  [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "${TRUNK_SHA[$s]}" ] \
    || stop "trunk of seed $s missing or SHA-256 mismatch (before any gate run)"
done

tree_of() {  # pid and all its descendants (spawned env workers open CUDA too)
  local p=$1 c; echo "$p"
  for c in $(pgrep -P "$p" || true); do tree_of "$c"; done
}

others_on_gpu() {  # count of GPU compute processes outside the tree of $1; fail closed
  local apps mine
  apps=$(nvidia-smi --id="$GPU_UUID" --query-compute-apps=pid --format=csv,noheader) \
    || { echo query-failed; return; }
  mine=$( [ -n "${1:-}" ] && tree_of "$1" || true )
  echo "$apps" | tr -d ' ' | grep -v "^$" | grep -vxF -f <(echo "${mine:-none}") | grep -c . || true
}

status_ok() {  # prints OK or the failed condition; never prints success values
  "$PY" -I - "$1/run.json" "$2" "${TRUNK_SHA[$3]}" <<'EOF'
import json, math, sys
r = json.load(open(sys.argv[1]))
c = r.get("success_curve") or []
n = r.get("success_curve_n")
last = c[-5:]
checks = {
    "using_stub": r.get("using_stub") is False,
    "training_failure": r.get("training_failure") is False,
    "liveness_abort": r.get("liveness_abort") is False,
    "steps": r.get("pair_steps_done") == int(sys.argv[2]),
    "trunk": r.get("init_trunk_sha256") == sys.argv[3],
    "liveness_mode": (r.get("train_options") or {}).get("liveness") == "abort",
    "settings": (r.get("micro_batch_pairs"), r.get("clear_cache"), r.get("n_workers")) == (512, True, 8),
    "last5_present": len(last) == 5 and [p[0] for p in last] == [9200000, 9400000, 9600000, 9800000, 10000000],
    "last5_finite": all(isinstance(p[1], (int, float)) and math.isfinite(p[1]) for p in last),
    "window_100": (n[-5:] == [100] * 5) if isinstance(n, list) else n == 100,
}
bad = [k for k, v in checks.items() if not v]
print("OK" if not bad else "FAIL:" + ",".join(bad))
EOF
}

run_one() {  # seed arm attempt -> 0 ok, 1 failed, exits on stop conditions
  local s=$1 arm=$2 att=$3 d="$OUT_ROOT/gb300-closed-$1-$2-a$3"
  [ ! -e "$d" ] || stop "run directory $d exists"
  [ "$(others_on_gpu)" = 0 ] || stop "another GPU process before $s $arm"
  log "start seed $s arm $arm attempt $att"
  ( cd "$REPO" && exec taskset -c "$CORES" nice -n 10 "$PY" -m \
      experiments.procgen_multiagent_gtm_comm.harness.train \
      --arm "$arm" --seed "$s" --condition closed --steps $STEPS --out "$d" \
      --log-every 200000 --ckpt-every 1000000 --preset fix-c-coin-pretrain \
      --dev-kind gate_dev --micro-batch-pairs 512 --clear-cache --n-workers 8 \
      --init-trunk "$TRUNK_DIR/trunk-$s.safetensors" \
      > "$d.result.sealed.json" 2> "$d.stderr.log" ) &
  local pid=$! cont=0; TRAINER=$pid
  nvidia-smi -q > "$d.nvsmi-before.txt" 2>&1 || true
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ "$(date +%s)" -gt $((END_EPOCH - 3600)) ]; then
      kill "$pid"; sleep 60; kill -9 "$pid" 2>/dev/null || true; wait "$pid" || true
      stop "deadline: $s $arm attempt $att killed (pair truncated)"
    fi
    if [ "$(others_on_gpu "$pid")" != 0 ]; then
      kill "$pid"; sleep 60; kill -9 "$pid" 2>/dev/null || true; cont=1; break
    fi
  done
  local rc=0; wait "$pid" || rc=$?; TRAINER=""
  nvidia-smi -q > "$d.nvsmi-after.txt" 2>&1 || true
  [ "$cont" = 0 ] || stop "contamination: another GPU process during $s $arm attempt $att"
  local st="FAIL:no-run-json"; [ -f "$d/run.json" ] && st=$(status_ok "$d" $STEPS "$s")
  log "end seed $s arm $arm attempt $att rc=$rc status=$st"
  [ "$rc" = 0 ] && [ "$st" = OK ]
}

for item in "${ORDER[@]}"; do
  s=${item%%:*}; arms=${item#*:}
  need=$("$PY" -c "print(int($SAFETY*($STEPS/$RATE_ENG+$STEPS/$RATE_NOCOMM)))")
  if [ $(( $(date +%s) + need )) -gt $((END_EPOCH - MARGIN_S)) ]; then
    log "time rule: seed $s not started (needs ${need}s)"; break
  fi
  for arm in $arms; do
    run_one "$s" "$arm" 1 || run_one "$s" "$arm" 2 \
      || stop "second failure: seed $s arm $arm (G2: PI decides)"
  done
  log "pair complete: seed $s"
done
cp "$REPO/experiments/procgen_multiagent_gtm_comm/manifest/smoke_ledger.json" \
   "$OUT_ROOT/smoke_ledger.final.json"
log "lane done"
