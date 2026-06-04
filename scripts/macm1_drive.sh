#!/usr/bin/env bash
set -euo pipefail

HOST=${MACM1_HOST:-macm1}
REMOTE_DIR=${REMOTE_DIR:-/tmp/nerve-wml-macm1-sci}
REMOTE_PY=${REMOTE_PY:-/tmp/bench-venv/bin/python}
LOCAL_OUT="docs/superpowers/research/2026-05-20-macm1-scientific-eval.json"
LOG="docs/superpowers/research/2026-05-20-macm1-scientific-eval.log"
BUDGET_S=${BUDGET_S:-5100}

mkdir -p "$(dirname "$LOCAL_OUT")"

echo "[drive] preparing $HOST"
ssh "$HOST" "mkdir -p $REMOTE_DIR"

echo "[drive] rsync script"
rsync -a scripts/macm1_scientific_eval.py "$HOST:$REMOTE_DIR/eval.py"

echo "[drive] sanity check torch"
ssh "$HOST" "$REMOTE_PY -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'"

echo "[drive] running eval (this may take 30-90 minutes)"
ssh "$HOST" "cd $REMOTE_DIR && $REMOTE_PY eval.py --device mps --out result.json --seeds 50 --budget-s $BUDGET_S 2>&1" | tee "$LOG"

echo "[drive] copying result back"
scp "$HOST:$REMOTE_DIR/result.json" "$LOCAL_OUT"

echo "[drive] done: $LOCAL_OUT"
