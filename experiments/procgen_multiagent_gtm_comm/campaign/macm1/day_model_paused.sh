#!/bin/zsh
# Run ONE command on macM1 with the second-brain day model paused
# (amendment 2026-10-25, section 6, PI decision 2026-10-25 11:54): every
# macM1 run of the new configuration, campaign runner or pre-training.
#
#   zsh day_model_paused.sh [--detach] STATE LOG -- CMD [ARG...]
#
# The process that creates the pause flag is the only one that ever removes
# it: the EXIT trap is installed BEFORE the flag is created, the flag is
# created exclusively (an existing flag belongs to someone else: refuse and
# leave it), and it is removed on every exit path after creation (refusal on
# free memory, command exit, HUP / INT / TERM: on any of these signals the
# command is sent SIGTERM and waited for first; the original signal is not
# forwarded as such). With --detach the script re-runs itself under nohup and
# waits until the detached copy reports "running" or "refused" in STATE;
# the caller never touches the flag, so there is no ownership to hand over.
# STATE, LOG and CMD are positional arguments, never interpolated into shell
# source. The command's stdout / stderr are appended to LOG, as are the
# launcher's JSON events.
#
# Environment (tests): PAUSE_FLAG (default ~/.mtplx/paused), DAY_MODEL_PORT
# (default 8081, the oMLX backend; the :8080 front honours the flag),
# MIN_FREE_PCT (default 40), SETTLE_S (default 5), DETACH_WAIT_S (default 60).
set -u
PAUSE="${PAUSE_FLAG:-$HOME/.mtplx/paused}"
PORT="${DAY_MODEL_PORT:-8081}"
MIN_FREE_PCT="${MIN_FREE_PCT:-40}"
export PATH=/opt/homebrew/bin:/usr/sbin:/usr/bin:/bin:$PATH

usage() { print -r -- "usage: $0 [--detach] STATE LOG -- CMD [ARG...]" >&2; exit 64; }
detach=0
if [ "${1:-}" = "--detach" ]; then detach=1; shift; fi
[ $# -ge 4 ] && [ "$3" = "--" ] || usage
STATE=$1 LOG=$2
shift 3

if (( detach )); then
  rm -f -- "$STATE"
  nohup zsh "${0:A}" "$STATE" "$LOG" -- "$@" >/dev/null 2>&1 &
  wpid=$!
  for _ in $(seq 1 "${DETACH_WAIT_S:-60}"); do
    if [ -s "$STATE" ]; then
      st=$(<"$STATE")
      case $st in
        running*) print -r -- "started: wrapper pid $wpid, $st (log $LOG)"
                  exit 0 ;;
        refused*) print -r -- "$st" >&2; exit 2 ;;
      esac
    fi
    kill -0 "$wpid" 2>/dev/null || break
    sleep 1
  done
  print -r -- "wrapper pid $wpid did not report (state: $(cat -- "$STATE" 2>/dev/null)); see $LOG" >&2
  exit 3
fi

owned=0
child=""
event() {  # $1 = state, $2 = extra JSON members (no leading comma) or ""
  local extra="${2:+, $2}"
  print -r -- "{\"at\": \"$(date +%FT%T%z)\", \"phase\": \"launcher\", \"state\": \"$1\"$extra}" >>"$LOG"
}
cleanup() {
  local rc=$?
  trap - EXIT HUP INT TERM
  if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" 2>/dev/null
    while kill -0 "$child" 2>/dev/null; do wait "$child" 2>/dev/null; done
  fi
  if (( owned )); then
    rm -f -- "$PAUSE"
    event day_model_resumed "\"rc\": $rc"
  fi
  exit $rc
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

refuse() { print -r -- "refused: $1" >"$STATE"; event refused; exit 2; }

mkdir -p -- "${PAUSE:h}"
if ! ( setopt NO_CLOBBER; : >"$PAUSE" ) 2>/dev/null; then
  refuse "pause flag $PAUSE already exists (another run owns it, or a stale flag: check and remove it by hand)"
fi
owned=1
event day_model_paused
# Stop the backend; the front does not restart it while the flag exists.
for pid in $(lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep "${SETTLE_S:-5}"
free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
if ! [[ "$free" == <-> ]] || (( free < MIN_FREE_PCT )); then
  refuse "only ${free:-?}% memory free (need $MIN_FREE_PCT): not starting, pause lifted"
fi

caffeinate -is -w $$ >/dev/null 2>&1 &
"$@" >>"$LOG" 2>&1 &
child=$!
print -r -- "running: command pid $child, day model paused, ${free}% free" >"$STATE"
event running "\"pid\": $child, \"free_pct\": $free"
wait "$child"
rc=$?
child=""
exit $rc
