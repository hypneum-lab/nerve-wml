#!/usr/bin/env bash
# Drive CKNNA bench on macm1 via SSH (Tailscale).
set -euo pipefail

HOST=${MACM1_HOST:-macm1}
REMOTE_DIR=${REMOTE_DIR:-/tmp/nerve-wml-bench-macm1}
LOCAL_DIR="docs/superpowers/research"
# macm1 has no torch in system python3.9 — use a uv-managed venv (python 3.12).
REMOTE_PY=${REMOTE_PY:-/tmp/bench-venv/bin/python}

echo "[macm1-bench] preparing $HOST"
ssh "$HOST" "mkdir -p $REMOTE_DIR"

echo "[macm1-bench] copying standalone bench"
scp /tmp/cknna_standalone_bench.py "$HOST:$REMOTE_DIR/bench.py"

echo "[macm1-bench] checking python+torch"
ssh "$HOST" "cd $REMOTE_DIR && $REMOTE_PY -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'"

# CPU
echo "[macm1-bench] CPU run"
ssh "$HOST" "cd $REMOTE_DIR && $REMOTE_PY bench.py --device cpu --out cpu.json --sizes 256 1024 4096 8192"
scp "$HOST:$REMOTE_DIR/cpu.json" "$LOCAL_DIR/2026-05-20-gpu-backend-bench-macm1-cpu.json"

# MPS
echo "[macm1-bench] MPS run"
ssh "$HOST" "cd $REMOTE_DIR && $REMOTE_PY bench.py --device mps --out mps.json --sizes 256 1024 4096 8192" \
  && scp "$HOST:$REMOTE_DIR/mps.json" "$LOCAL_DIR/2026-05-20-gpu-backend-bench-macm1-mps.json" \
  || echo "[macm1-bench] MPS run failed; missing or unavailable"

echo "[macm1-bench] done"
