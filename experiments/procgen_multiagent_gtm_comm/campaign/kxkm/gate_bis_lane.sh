#!/usr/bin/env bash
# Learnability gate of the N10-C-bis arms under fix C (PI decision
# 2026-10-26 22:02): eng_gtmrx, gtm_warm (warm-started from that eng_gtmrx
# gate run), gtm_mis, vq; preset fix-c-coin-pretrain, development seed 9101,
# registered condition, 1,000,000 pair-steps, the seed-9101 trunk, KXKM-AI
# per-host settings, ledger kind gate_dev. One run at a time; it never runs
# next to another GPU job of ours (E1-small, other N10-C lanes): before
# every run it waits until no such job is alive and the card is free
# (ComfyUI, pid 712018 by default, is tolerated and never touched).
#   REPO=/ai/saisail/n10c-dev/repo-bis OUT=/ai/saisail/n10c-dev/gate-bis \
#   TRUNK=/ai/saisail/n10c-dev/gate/pretrain/trunk.safetensors \
#   [ARMS="vq"] nohup bash gate_bis_lane.sh &   (no tmux session: the E1-small
#   GPU wait helper waits on gatebis / n10c-dev-* sessions, which would
#   deadlock against this lane's own wait)
set -u
REPO="${REPO:?REPO = working copy of the PR head}"
OUT="${OUT:?OUT = output directory}"
TRUNK="${TRUNK:?TRUNK = seed-9101 pre-trained trunk}"
PY="${PY:-/ai/saisail/n10c-run/.venv/bin/python}"
TOLERATED_PIDS="${TOLERATED_PIDS:-712018}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9500}"
SEED=9101
ARMS="${ARMS:-eng_gtmrx gtm_warm gtm_mis vq}"
want() { case " $ARMS " in *" $1 "*) return 0 ;; esac; return 1; }
PRESET=fix-c-coin-pretrain
export CUDA_HOME="${CUDA_HOME:-/ai/saisail/n10c-bench/cuda_home}" PYTHONPATH=.
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$OUT/driver.log"; }

busy() {  # another GPU job of ours alive, or a non-tolerated GPU process
  if tmux ls 2>/dev/null | grep -Eq '^(e1s-|n10c-dev-|n10c:)'; then return 0; fi
  # E1-small jobs (its venv python); not its own GPU-wait helper, which
  # waits for us in turn
  if pgrep -f '/ai/saisail/e1-small/venv/' >/dev/null; then return 0; fi
  if pgrep -f 'procgen_multiagent_gtm_comm\.(harness|campaign)\.' \
      | grep -vx "$$" >/dev/null; then return 0; fi
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) \
    || return 0
  for p in $pids; do
    case " $TOLERATED_PIDS " in *" $p "*) ;; *) return 0 ;; esac
  done
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null) \
    || return 0  # query failed: treat as busy
  [[ "$free" =~ ^[0-9]+$ ]] || return 0  # N/A or garbage: busy
  [ "$free" -lt "$MIN_FREE_MIB" ]
}

wait_free() {
  local said=0
  while busy; do
    [ $said = 1 ] || { log "waiting: GPU or another job busy"; said=1; }
    sleep 60
  done
}

run() {  # arm [extra args...]
  local arm=$1; shift
  wait_free
  log "start $PRESET $arm seed $SEED $*"
  (cd "$REPO" && nice -n 10 "$PY" -m experiments.procgen_multiagent_gtm_comm.harness.train \
    --arm "$arm" --seed $SEED --steps 1000000 --out "$OUT/$PRESET-$arm" \
    --log-every 40000 --ckpt-every 100000 --preset $PRESET --dev-kind gate_dev \
    --micro-batch-pairs 512 --clear-cache --n-workers 8 --init-trunk "$TRUNK" "$@" \
    > "$OUT/$PRESET-$arm.log" 2>&1)
  log "end $PRESET $arm rc=$?"
}

[ -f "$TRUNK" ] || { log "no trunk $TRUNK: stop"; exit 1; }
log "lane start, code $(git -C "$REPO" rev-parse --short HEAD), trunk $(sha256sum "$TRUNK" | cut -c1-12)"
want eng_gtmrx && run eng_gtmrx
# warm source: the last checkpoint of the eng_gtmrx gate run (addendum:
# the last finite checkpoint of a failed source, else initialisation)
SRC=$(ls -v "$OUT/$PRESET-eng_gtmrx" 2>/dev/null \
      | grep -E '^ckpt_[0-9]+\.safetensors$' | tail -1)  # never a .tmp file
[ -n "$SRC" ] && SRC="$OUT/$PRESET-eng_gtmrx/$SRC"
if ! want gtm_warm; then :
elif [ -n "$SRC" ] && [ -f "$SRC" ]; then
  run gtm_warm --warm-start-from "$SRC"
else  # rule of the addendum: no finite source -> initialisation, flagged
  log "eng_gtmrx gate run left no final checkpoint: gtm_warm starts untrained"
  run gtm_warm
fi
want gtm_mis && run gtm_mis
want vq && run vq
log "lane done"
