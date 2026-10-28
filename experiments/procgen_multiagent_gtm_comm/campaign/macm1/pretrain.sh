#!/bin/zsh
# Trunk pre-training on macM1 (harness.pretrain), detached, with the
# second-brain day model paused for the whole run (amendment 2026-10-25,
# section 6: every macM1 run of the new configuration, pre-training
# included). Same pause lifecycle as start.sh (day_model_paused.sh).
#   [HEAVY_LOCK=~/scratch/.heavy.lock] BASE=~/scratch/n10c-freeze \
#     zsh pretrain.sh OUT_DIR --seed N --pre-freeze   (freeze plan, phase T)
#   or ... --seed N [--registered --freeze-tag ... --swhid ...]
# HEAVY_LOCK: wait for the shared heavy-job lock before pausing the day
# model (day_model_paused.sh).
# Log and launcher state: OUT_DIR/console.log, OUT_DIR/launcher.state.
set -eu
HERE="${0:A:h}"
BASE="${BASE:-$HOME/scratch/n10c-run}"
PY="${PY:-$BASE/.venv/bin/python}"
[ $# -ge 1 ] || { echo "usage: $0 OUT_DIR --seed N [pretrain args]" >&2; exit 64; }
OUT="${1:A}"; shift
export PATH=/opt/homebrew/bin:$PATH
cd "$BASE/repo"
mkdir -p "$OUT"
CMD=(env PYTHONPATH=. "$PY" -m experiments.procgen_multiagent_gtm_comm.harness.pretrain
     --out "$OUT" "$@")
exec zsh "$HERE/day_model_paused.sh" --detach "$OUT/launcher.state" \
  "$OUT/console.log" -- "${CMD[@]}"
