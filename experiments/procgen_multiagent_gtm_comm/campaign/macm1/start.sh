#!/bin/zsh
# Start (or resume) the N10-C campaign on macM1, detached (nohup + caffeinate).
# Mirrors kxkm/start.sh: synchronous guard dry-run first, refuse on WOULD REFUSE.
# PI decision 2026-10-25 11:54: the second-brain day model is paused for the
# whole campaign process (pause flag + oMLX backend on :8081 stopped) and the
# pause is lifted when the process exits, whatever its exit code.
#   BASE=~/scratch/n10c-run zsh start.sh --freeze-tag n10c-freeze-YYYYMMDD \
#        --swhid swh:1:rev:<40 hex> [--archive-to host:/path]
set -eu
BASE="${BASE:-$HOME/scratch/n10c-run}"
PY="${PY:-$BASE/.venv/bin/python}"
CAMP="$BASE/campaign"
PAUSE="$HOME/.mtplx/paused"
MIN_FREE_PCT="${MIN_FREE_PCT:-40}"
export PATH=/opt/homebrew/bin:$PATH
cd "$BASE/repo"
mkdir -p "$CAMP"
CMD=(env PYTHONPATH=. "$PY" -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign
     --campaign-dir "$CAMP" --host macm1 "$@")
if pgrep -f "run_campaign --campaign-dir $CAMP" >/dev/null; then
  echo "a campaign runner is already alive" >&2; exit 2
fi
PRE="$("${CMD[@]}" --dry-run)"; print -r -- "$PRE" | tail -3
if print -r -- "$PRE" | grep -q "WOULD REFUSE"; then
  echo "guards would refuse: not starting" >&2; exit 2
fi

# Pause the day model, then stop its backend (the :8080 front honours the flag).
touch "$PAUSE"
for pid in $(lsof -ti tcp:8081 -sTCP:LISTEN 2>/dev/null); do kill "$pid"; done
sleep 5
free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
if [ "${free:-0}" -lt "$MIN_FREE_PCT" ]; then
  rm -f "$PAUSE"
  echo "only ${free}% memory free (need $MIN_FREE_PCT): not starting, pause lifted" >&2
  exit 2
fi

LOG="$CAMP/console.log"
nohup zsh -c '
  caffeinate -is "$@" >>"'"$LOG"'" 2>&1; rc=$?
  rm -f "'"$PAUSE"'"
  echo "{\"at\": \"$(date +%FT%T%z)\", \"phase\": \"launcher\", \"state\": \"day_model_resumed\", \"rc\": $rc}" >>"'"$LOG"'"
' _ "${CMD[@]}" >/dev/null 2>&1 &
echo "started: wrapper pid $! (log $LOG); day model paused until the runner exits (${free}% free)"
