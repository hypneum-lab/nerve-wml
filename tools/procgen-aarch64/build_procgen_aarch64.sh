#!/usr/bin/env bash
# Build a Procgen 0.10.7 wheel for Linux aarch64 + CPython 3.12.
# Idempotent: re-running reuses WORK and rebuilds from a clean source tree.
# Target: Ubuntu 24.04 arm64 (DGX Station GB300). Needs sudo for apt only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HOME/procgen-aarch64-build}"
PROCGEN_URL="https://github.com/openai/procgen.git"
PROCGEN_COMMIT="5e1dbf341d291eff40d1f9e0c0a0d5003643aebf"   # tag 0.10.7
NICE="${NICE:-nice -n 19}"
JOBS="${JOBS:-4}"

[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: aarch64 required (got $(uname -m))" >&2; exit 1; }

# Exact apt versions of Ubuntu 24.04 (noble) as resolved 2026-10-31 on the
# amd64 archive; the arm64 archive carries the same version strings. If a
# version is gone (security update), set APT_LOOSE=1 to take the candidate
# and record `dpkg -l` in the README.
APT_PKGS=(
  build-essential=12.10ubuntu1
  g++=4:13.2.0-7ubuntu1
  cmake=3.28.3-1build7
  ninja-build=1.11.1-2
  pkg-config=1.8.1-2build1
  git=1:2.43.0-1ubuntu7.4
  qtbase5-dev=5.15.13+dfsg-1ubuntu1
  libgl-dev=1.7.0-1build1
  python3.12-dev=3.12.3-1ubuntu0.17
  python3.12-venv=3.12.3-1ubuntu0.17
)
if [ "${SKIP_APT:-0}" != 1 ]; then
  if [ "${APT_LOOSE:-0}" = 1 ]; then
    sudo apt-get install -y --no-install-recommends "${APT_PKGS[@]%%=*}"
  else
    sudo apt-get install -y --no-install-recommends "${APT_PKGS[@]}"
  fi
fi

mkdir -p "$WORK"; cd "$WORK"

# 1. source, pinned to a commit
[ -d procgen/.git ] || git clone -q "$PROCGEN_URL" procgen
git -C procgen fetch -q --tags origin
git -C procgen reset -q --hard "$PROCGEN_COMMIT"
git -C procgen clean -q -fdx
[ "$(git -C procgen rev-parse HEAD)" = "$PROCGEN_COMMIT" ] || { echo "ERROR: commit mismatch" >&2; exit 1; }

# 2. patches (sha256 checked against the committed list)
( cd "$HERE/patches" && sha256sum -c ../patches.sha256 )
for p in "$HERE"/patches/*.patch; do git -C procgen apply --index "$p"; done

# 3. hashed python deps (pip --require-hashes, aarch64 cp312 lockfile)
[ -x venv/bin/python ] || python3.12 -m venv venv
venv/bin/python -m pip install -q --require-hashes -r "$HERE/requirements-aarch64-cp312.lock"

# 4. wheel (git is present, so version becomes 0.10.7+5e1dbf3)
export PROCGEN_CMAKE_PREFIX_PATH=/usr/lib/aarch64-linux-gnu/cmake
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
rm -rf dist; mkdir dist
$NICE venv/bin/python -m pip wheel --no-build-isolation --no-deps -w dist ./procgen
WHEEL="$(ls dist/procgen-*.whl)"

# 5. smoke test in a clean venv, outside the clone (namespace-package trap)
rm -rf smoke; python3.12 -m venv smoke
smoke/bin/python -m pip install -q --require-hashes -r "$HERE/requirements-aarch64-cp312.lock"
smoke/bin/python -m pip install -q --no-deps "$WORK/$WHEEL"
( cd / && QT_QPA_PLATFORM=offscreen "$WORK/smoke/bin/python" -I -c "
import numpy as np, hashlib
from procgen import ProcgenGym3Env
e = ProcgenGym3Env(num=2, env_name='coinrun', start_level=0, num_levels=1, rand_seed=0)
h = hashlib.sha256()
for _ in range(50):
    e.act(np.zeros(2, dtype=np.int32)); r, o, f = e.observe(); h.update(o['rgb'].tobytes())
print('smoke ok, obs-hash', h.hexdigest())
" )

echo "WHEEL $WORK/$WHEEL"
sha256sum "$WORK/$WHEEL"
