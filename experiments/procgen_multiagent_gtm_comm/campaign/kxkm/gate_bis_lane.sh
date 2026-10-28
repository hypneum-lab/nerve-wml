#!/usr/bin/env bash
# Learnability gate of the N10-C-bis arms under fix C (PI decision
# 2026-10-26 22:02): eng_gtmrx, gtm_warm (warm-started from that eng_gtmrx
# gate run), gtm_mis, vq; preset fix-c-coin-pretrain, development seed 9101,
# registered condition, 1,000,000 pair-steps, the seed-9101 trunk, KXKM-AI
# per-host settings, ledger kind gate_dev. One run at a time; it never runs
# next to another GPU job of ours (E1-small, other N10-C lanes): before
# every run it waits until no such job is alive and the card has enough
# free memory (gpu_wait.sh: foreign GPU jobs such as ComfyUI or other
# studies' processes are never touched and count only through memory).
#   REPO=/ai/saisail/n10c-dev/repo-bis OUT=/ai/saisail/n10c-dev/gate-bis \
#   TRUNK=/ai/saisail/n10c-dev/gate/pretrain/trunk.safetensors \
#   ARMS=vq nohup bash gate_bis_lane.sh </dev/null >/dev/null 2>&1 &
#   (ARMS optional, default all four; a rerun needs a NEW OUT: the lane
#   refuses to reuse a run directory. No tmux session: the E1-small
#   GPU wait helper waits on gatebis / n10c-dev-* sessions, which would
#   deadlock against this lane's own wait)
set -u
REPO="${REPO:?REPO = working copy of the PR head}"
OUT="${OUT:?OUT = output directory}"
TRUNK="${TRUNK:?TRUNK = seed-9101 pre-trained trunk}"
PY="${PY:-/ai/saisail/n10c-run/.venv/bin/python}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9500}"
SEED=9101
ARMS="${ARMS:-eng_gtmrx gtm_warm gtm_mis vq}"
want() { case " $ARMS " in *" $1 "*) return 0 ;; esac; return 1; }
for a in $ARMS; do
  case $a in eng_gtmrx|gtm_warm|gtm_mis|vq) ;; *) echo "unknown arm $a in ARMS" >&2; exit 64 ;; esac
done
PRESET=fix-c-coin-pretrain
export CUDA_HOME="${CUDA_HOME:-/ai/saisail/n10c-bench/cuda_home}" PYTHONPATH=.
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$OUT/driver.log"; }

# shared wait rules (foreign GPU jobs count only through free memory)
source "$(dirname "${BASH_SOURCE[0]}")/gpu_wait.sh"

run() {  # arm [extra args...]
  local arm=$1; shift
  if [ -e "$OUT/$PRESET-$arm" ]; then  # never overwrite an earlier attempt
    log "refused $arm: $OUT/$PRESET-$arm exists (use a new OUT)"; return 1
  fi
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
