#!/bin/zsh
# Start (or resume) the N10-C campaign on macM1, detached (nohup + caffeinate).
# Mirrors kxkm/start.sh: synchronous guard dry-run first, refuse on WOULD REFUSE.
# PI decision 2026-10-25 11:54: the second-brain day model is paused for the
# whole campaign process and resumed when it exits, whatever its exit code.
# The pause lifecycle (flag ownership, memory check, cleanup on every exit
# path) lives in day_model_paused.sh; this script never touches the flag.
#   BASE=~/scratch/n10c-run zsh start.sh --freeze-tag n10c-freeze-YYYYMMDD \
#        --swhid swh:1:rev:<40 hex> [--archive-to host:/path]
set -eu
HERE="${0:A:h}"
BASE="${BASE:-$HOME/scratch/n10c-run}"
PY="${PY:-$BASE/.venv/bin/python}"
CAMP="$BASE/campaign"
export PATH=/opt/homebrew/bin:$PATH
cd "$BASE/repo"
mkdir -p "$CAMP"
CMD=(env PYTHONPATH=. "$PY" -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign
     --campaign-dir "$CAMP" --host macm1 "$@")
if pgrep -f -- "run_campaign --campaign-dir $CAMP" >/dev/null; then
  echo "a campaign runner is already alive" >&2; exit 2
fi
PRE="$("${CMD[@]}" --dry-run)"; print -r -- "$PRE" | tail -3
if print -r -- "$PRE" | grep -q "WOULD REFUSE"; then
  echo "guards would refuse: not starting" >&2; exit 2
fi
exec zsh "$HERE/day_model_paused.sh" --detach "$CAMP/launcher.state" \
  "$CAMP/console.log" -- "${CMD[@]}"
