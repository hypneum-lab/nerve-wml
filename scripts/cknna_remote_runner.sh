#!/usr/bin/env bash
set -euo pipefail
# Drive CKNNA bench on KXKM-AI 4090 via SSH jump.

JUMP=${JUMP:-electron-server}
HOST=${KXKM_HOST:-kxkm@10.2.0.237}
REMOTE_DIR=${REMOTE_DIR:-/tmp/nerve-wml-bench}
LOCAL_OUT=${LOCAL_OUT:-docs/superpowers/research/2026-05-20-gpu-backend-bench-cuda.json}

echo "[remote-bench] preparing $HOST via $JUMP"
ssh -J "$JUMP" "$HOST" "mkdir -p $REMOTE_DIR/scripts $REMOTE_DIR/track_p"

echo "[remote-bench] rsyncing minimal source"
rsync -e "ssh -J $JUMP" -a \
  scripts/cknna_backend_bench.py \
  "$HOST:$REMOTE_DIR/scripts/"
rsync -e "ssh -J $JUMP" -a \
  track_p/transducer.py track_p/__init__.py \
  "$HOST:$REMOTE_DIR/track_p/" || true

# Provide a minimal track_p/__init__.py shim in case multiplexer deps are missing.
ssh -J "$JUMP" "$HOST" "cat > $REMOTE_DIR/track_p/__init__.py" <<'PYEOF'
# minimal shim for remote bench
PYEOF

echo "[remote-bench] running on $HOST (this may take a few minutes)"
REMOTE_CMD="cd $REMOTE_DIR && (source ~/venv/bin/activate 2>/dev/null || true) && python -m scripts.cknna_backend_bench --device cuda --out cuda.json --sizes 256 1024 4096 8192 16384"
ssh -J "$JUMP" "$HOST" "$REMOTE_CMD" || {
  echo "[remote-bench] remote run failed; will document"
  exit 1
}

echo "[remote-bench] scp result back"
rsync -e "ssh -J $JUMP" -a "$HOST:$REMOTE_DIR/cuda.json" "$LOCAL_OUT"
echo "[remote-bench] wrote $LOCAL_OUT"
