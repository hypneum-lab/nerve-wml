#!/usr/bin/env bash
# Pull completed N10-C runs from KXKM-AI to electron-server (KXKM cannot reach
# electron-server; electron-server -> KXKM works). Run ON electron-server:
#   bash pull_archive.sh            # e.g. from cron or by hand, repeatable
# Env: KXKM_SRC (default electron@10.2.0.237:/ai/saisail/n10c-run/campaign),
#      DEST (default /home/electron/archives/n10c), LOG (default DEST/pull.log).
# Completed run = status file with eval_done true. Copies run.json, eval.json
# and the STATUS-SELECTED checkpoint (for an exhausted run that is
# ../<run>.failed-N/ckpt_*.safetensors, resolved explicitly and stored as
# failed-N__ckpt_*.safetensors), verifies its SHA-256 against the digest the
# campaign recorded at training time and writes PROVENANCE.json. Also copies
# campaign.json, attempts.jsonl (to _campaign/), cross/ and cross_status/.
# Idempotent (rsync), never deletes, never touches the KXKM side. Exit 1 if
# any run failed its digest check or its checkpoint could not be fetched.
set -euo pipefail
SRC="${KXKM_SRC:-electron@10.2.0.237:/ai/saisail/n10c-run/campaign}"
DEST="${DEST:-/home/electron/archives/n10c}"
LOG="${LOG:-$DEST/pull.log}"
SSH="ssh -o BatchMode=yes -o ControlPath=none"
mkdir -p "$DEST" "$DEST/.status" "$DEST/_campaign"
log() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $*" | tee -a "$LOG"; }

log "pull start from $SRC"
rsync -a -e "$SSH" "$SRC/status/" "$DEST/.status/"
for f in campaign.json attempts.jsonl; do
  rsync -a -e "$SSH" "$SRC/$f" "$DEST/_campaign/" 2>/dev/null || true
done
n=0
bad=0
for f in "$DEST"/.status/*.json; do
  [ -e "$f" ] || continue
  read -r rid ck sha < <(python3 - "$f" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
if s.get("eval_done"):
    print(s["run_id"], s.get("last_checkpoint") or "-",
          s.get("last_checkpoint_sha256") or "-")
PY
  ) || true
  [ -n "${rid:-}" ] || continue
  rsync -a -e "$SSH" --include=run.json --include=eval.json --exclude='*' \
    "$SRC/runs/$rid/" "$DEST/$rid/"
  name="-"
  if [ "$ck" != "-" ]; then
    # resolve the status-selected checkpoint relative to runs/<rid>/
    case "$ck" in
      ../*) rel="${ck#../}"; dir="$(dirname "$rel")"; base="$(basename "$rel")"
            name="${dir##*.}__$base" ;;
      *)    rel="$rid/$ck"; base="$(basename "$ck")"; name="$base" ;;
    esac
    tmp="$DEST/$rid/.fetch-$base"
    if rsync -a -e "$SSH" "$SRC/runs/$rel" "$tmp"; then
      got="$(sha256sum "$tmp" | cut -d' ' -f1)"
      if [ "$sha" != "-" ] && [ "$sha" != "$got" ]; then
        log "ERROR $rid: checkpoint $ck SHA-256 $got != recorded $sha (not stored)"
        rm -f "$tmp"; bad=$((bad + 1)); unset rid ck sha; continue
      fi
      mv -f "$tmp" "$DEST/$rid/$name"
      printf '{"run_id": "%s", "status_last_checkpoint": "%s", "archived_as": "%s", "sha256": "%s", "recorded_sha256": "%s", "source": "%s/runs/%s"}\n' \
        "$rid" "$ck" "$name" "$got" "$sha" "$SRC" "$rel" > "$DEST/$rid/PROVENANCE.json"
    else
      log "ERROR $rid: could not fetch checkpoint $ck"
      bad=$((bad + 1)); unset rid ck sha; continue
    fi
  fi
  log "pulled $rid ($ck -> $name)"
  n=$((n + 1))
  unset rid ck sha
done
for d in cross cross_status; do
  rsync -a -e "$SSH" "$SRC/$d/" "$DEST/$d/" 2>/dev/null || true
done
log "pull done: $n completed run(s), $bad failed"
[ "$bad" -eq 0 ]
