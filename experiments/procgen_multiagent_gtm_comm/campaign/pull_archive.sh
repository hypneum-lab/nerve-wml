#!/usr/bin/env bash
# Pull completed N10-C runs from KXKM-AI to electron-server (KXKM cannot reach
# electron-server; electron-server -> KXKM works). Run ON electron-server:
#   bash pull_archive.sh            # e.g. from cron or by hand, repeatable
# Env: KXKM_SRC (default electron@10.2.0.237:/ai/saisail/n10c-run/campaign),
#      DEST (default /home/electron/archives/n10c), LOG (default DEST/pull.log).
# Completed run = status file with eval_done true. Copies run.json, eval.json
# and the last checkpoint only; also cross/ and cross_status/. Idempotent
# (rsync), never deletes, never touches the KXKM side.
set -euo pipefail
SRC="${KXKM_SRC:-electron@10.2.0.237:/ai/saisail/n10c-run/campaign}"
DEST="${DEST:-/home/electron/archives/n10c}"
LOG="${LOG:-$DEST/pull.log}"
SSH="ssh -o BatchMode=yes -o ControlPath=none"
mkdir -p "$DEST" "$DEST/.status"
log() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $*" | tee -a "$LOG"; }

log "pull start from $SRC"
rsync -a -e "$SSH" "$SRC/status/" "$DEST/.status/"
n=0
for f in "$DEST"/.status/*.json; do
  [ -e "$f" ] || continue
  read -r rid ck < <(python3 - "$f" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
if s.get("eval_done"):
    print(s["run_id"], s.get("last_checkpoint") or "-")
PY
  ) || true
  [ -n "${rid:-}" ] || continue
  files=(--include=run.json --include=eval.json)
  if [ "$ck" != "-" ]; then files+=(--include="$(basename "$ck")"); fi
  rsync -a -e "$SSH" "${files[@]}" --exclude='*' "$SRC/runs/$rid/" "$DEST/$rid/"
  log "pulled $rid ($ck)"
  n=$((n + 1))
  unset rid ck
done
for d in cross cross_status; do
  rsync -a -e "$SSH" "$SRC/$d/" "$DEST/$d/" 2>/dev/null || true
done
cp -f "$DEST"/.status/*.json "$DEST/.status/" 2>/dev/null || true
log "pull done: $n completed run(s)"
