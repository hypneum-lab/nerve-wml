#!/usr/bin/env bash
# Start (or resume) the N10-C campaign on the host, detached (tmux, else nohup).
# All hard guards live in guards.py / run_campaign.py and refuse without the
# annotated freeze tag n10c-freeze-YYYYMMDD on origin (mandatory), a
# Software Heritage id, the tagged manifest hash matching the tag message and
# a clean tree. An OSF DOI is optional metadata only. Needs the PI's go.
#   BASE=/ai/saisail/n10c-run bash start.sh --freeze-tag n10c-freeze-YYYYMMDD \
#        --swhid swh:1:rev:<40 hex> [--manifest-doi 10.17605/OSF.IO/ID]
set -euo pipefail
BASE="${BASE:-/ai/saisail/n10c-run}"
cd "$BASE/repo"
PY="$BASE/.venv/bin/python"
CAMP="$BASE/campaign"
mkdir -p "$CAMP"
CMD=(env PYTHONPATH=. "$PY" -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign
     --campaign-dir "$CAMP" --host kxkm "$@")
# Pre-check synchronously so a refusal is visible (guards re-run at start).
PRE="$("${CMD[@]}" --dry-run)"; echo "$PRE"
if grep -q "WOULD REFUSE" <<<"$PRE"; then echo "guards would refuse: not starting" >&2; exit 2; fi
if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t n10c 2>/dev/null && { echo "tmux session n10c exists" >&2; exit 2; }
  tmux new-session -d -s n10c "${CMD[*]@Q} 2>&1 | tee -a $CAMP/console.log"
  echo "started in tmux session n10c (attach: tmux attach -t n10c)"
else
  nohup "${CMD[@]}" >>"$CAMP/console.log" 2>&1 &
  echo "started with nohup, pid $! (log $CAMP/console.log)"
fi
