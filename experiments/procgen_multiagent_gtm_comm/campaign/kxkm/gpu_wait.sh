# Sourced by the KXKM-AI lanes (gate_bis_lane.sh, pretrain_lane.sh): wait
# until the card is free for one N10-C job. Never signals any process.
#
# Busy = another GPU job of OURS is alive (an N10-C harness / campaign
# process, an E1-small job, or one of our tmux lanes), or the card has less
# than MIN_FREE_MIB free. Foreign GPU processes (ComfyUI, other studies'
# jobs such as nh81's measure.py, even when they run from our venv) are not
# ours and only count through the free-memory threshold.
# Needs: log() defined by the caller; MIN_FREE_MIB (default 9500).

busy() {
  if tmux ls 2>/dev/null | grep -Eq '^(e1s-|n10c-dev-|n10c:)'; then return 0; fi
  # E1-small jobs (its venv python); not its own GPU-wait helper, which
  # waits for us in turn
  if pgrep -f '/ai/saisail/e1-small/venv/' >/dev/null; then return 0; fi
  if pgrep -f 'procgen_multiagent_gtm_comm\.(harness|campaign)\.' \
      | grep -vx "$$" >/dev/null; then return 0; fi
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null) \
    || return 0  # query failed: treat as busy
  [[ "$free" =~ ^[0-9]+$ ]] || return 0  # N/A or garbage: busy
  [ "$free" -lt "${MIN_FREE_MIB:-9500}" ]
}

wait_free() {
  local said=0
  while busy; do
    [ $said = 1 ] || { log "waiting: GPU or another job of ours busy"; said=1; }
    sleep 60
  done
}
