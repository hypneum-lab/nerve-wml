#!/usr/bin/env bash
# Deploy the pinned N10-C clone + venv recipe to KXKM-AI (/ai/saisail only).
# Reaches KXKM through the electron-server jump; never runs `tailscale switch`.
# Run from a clone checked out at the commit to pin:
#   bash kxkm/deploy.sh [--dry-run]
# NOT to be run without the PI's go (see ../README.md).
set -euo pipefail

JUMP="${N10C_JUMP:-electron@100.78.191.52}"
HOST="${N10C_HOST:-electron@10.2.0.237}"
BASE="${N10C_BASE:-/ai/saisail/n10c-run}"
case "$BASE" in /ai/saisail/*) ;; *) echo "BASE must be under /ai/saisail" >&2; exit 2;; esac
SSH=(ssh -o BatchMode=yes -o ControlPath=none -J "$JUMP" "$HOST")
RSYNC_E="ssh -o BatchMode=yes -o ControlPath=none -J $JUMP"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then echo "git tree dirty: refuse" >&2; exit 2; fi
COMMIT="$(git rev-parse HEAD)"
BUNDLE="$(mktemp -d)/n10c.bundle"
git bundle create "$BUNDLE" HEAD >/dev/null 2>&1

# Pinned venv recipe (supply-chain HITL: exact pins; mlx CUDA build must
# be the same version as the Mac harness lock, 0.31.1).
PINS='mlx==0.31.1 mlx-cuda-12==0.31.1 numpy==1.26.4 gym3==0.3.3 gym==0.26.2 procgen==0.10.7 pytest==9.0.3'

echo "commit   : $COMMIT"
echo "target   : $HOST via $JUMP : $BASE"
echo "venv pins: $PINS (uv standalone python 3.10)"
if [ "$DRY" = 1 ]; then echo "dry-run: nothing copied"; exit 0; fi

"${SSH[@]}" "mkdir -p '$BASE' && command -v uv >/dev/null || { echo 'uv missing on KXKM: stop (no new install without HITL)' >&2; exit 3; }"
rsync -a --partial -e "$RSYNC_E" "$BUNDLE" "$HOST:$BASE/n10c.bundle"
"${SSH[@]}" bash -s -- "$BASE" "$COMMIT" "$PINS" <<'REMOTE'
set -euo pipefail
BASE="$1"; COMMIT="$2"; PINS="$3"
cd "$BASE"
if [ ! -d repo/.git ]; then git clone -q n10c.bundle repo; else git -C repo fetch -q ../n10c.bundle HEAD; fi
git -C repo checkout -q --detach "$COMMIT"
[ "$(git -C repo rev-parse HEAD)" = "$COMMIT" ]
uv venv --python 3.10 --managed-python .venv
# shellcheck disable=SC2086
uv pip install --python .venv/bin/python $PINS
.venv/bin/python - <<'PY'
import mlx.core as mx, procgen, numpy
print("mlx", mx.__version__ if hasattr(mx, "__version__") else "?", "default device", mx.default_device())
print("numpy", numpy.__version__)
PY
echo "deployed $COMMIT to $BASE/repo ; venv $BASE/.venv"
REMOTE
