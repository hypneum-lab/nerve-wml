#!/bin/bash
# Confirmatory launcher for P1bis-ext and P8c-nocap on kxkm-ai (CPU only).
# Pre-registrations: docs/prereg/hopfield-p1bis-ext/prereg.md,
#                    docs/prereg/hopfield-p8c-nocap/prereg.md
#
# Refuses to start unless: the repo clone is clean and HEAD is the commit of
# the freeze tag given as $1, the tag is annotated, the venv freeze matches the
# hash-pinned requirements, and no STOP file exists.
#
# Usage (on kxkm-ai, user electron, AFTER the freeze tag and the PI go):
#   nohup scripts/hopfield_conf/kxkm_launch.sh prereg-hopfield-p1bis-p8c-YYYYMMDD \
#       > /ai/saisail/p1bis-p8c-20261104/logs/launch.nohup 2>&1 &
# Stop: touch /ai/saisail/p1bis-p8c-20261104/STOP   (all lanes; checked between cells)
#
# Lanes (<= 24 worker processes in total, nice 19, single-threaded BLAS/numba):
#   A  P8c-nocap N = 16384, alpha = 0.5 LANE_A_WORKERS (default 8; memory bound)
#   B  P1bis-ext (all N), then P8c-nocap N = 4096   LANE_B_WORKERS (default 12)
set -u
TAG=${1:?usage: kxkm_launch.sh <freeze-tag>}
W=${W:-/ai/saisail/p1bis-p8c-20261104}
REPO=${REPO:-$W/repo}
PY=$W/venv/bin/python
LANE_A_WORKERS=${LANE_A_WORKERS:-8}
LANE_B_WORKERS=${LANE_B_WORKERS:-12}
P8C_MAX_STOCH_SECS=${P8C_MAX_STOCH_SECS:-64800}
WALL_CAP_P1=${WALL_CAP_P1:-21600}
WALL_CAP_P8=${WALL_CAP_P8:-259200}
STOP=$W/STOP
LOG=$W/logs/campaign.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "$(ts) ABORT $*" | tee -a "$LOG"; exit 1; }

mkdir -p "$W/logs" "$W/results"
[ -e "$STOP" ] && die "STOP file present: $STOP"
[ "$(hostname)" = "kxkm-ai" ] || die "host is $(hostname), pre-registered host is kxkm-ai"
[ "$(id -un)" = "electron" ] || die "user must be electron"
cd "$REPO" || die "no repo at $REPO"
[ "$(git cat-file -t "$TAG" 2>/dev/null)" = "tag" ] || die "$TAG is not an annotated tag"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$TAG^{commit}")" ] || die "HEAD is not $TAG"
[ -z "$(git status --porcelain)" ] || die "working tree not clean"
REQ=experiments/hopfield_dream_conf/requirements-hopfield-conf.txt
# Name and version only: wheel hashes were enforced when the venv was built
# with `uv pip sync --require-hashes --only-binary :all:` from this file.
want=$(grep -E '^[a-z0-9_.-]+==' "$REQ" | sed 's/ .*//' | tr 'A-Z_' 'a-z-' | sort)
have=$("$PY" -I -c 'import importlib.metadata as m
for d in m.distributions(): print(d.metadata["Name"] + "==" + d.version)' | tr 'A-Z_' 'a-z-' | sort)
[ "$want" = "$have" ] || die "venv freeze differs from $REQ"

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMBA_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
{
  echo "$(ts) START tag=$TAG commit=$(git rev-parse HEAD) host=$(hostname) user=$(id -un)"
  echo "$(ts) python=$("$PY" -V 2>&1) lanes A=$LANE_A_WORKERS B=$LANE_B_WORKERS"
  echo "$(ts) code sha256:"; sha256sum experiments/hopfield_dream_conf/*.py \
      experiments/hopfield_dream_conf/vendor/hopfield.py "$REQ" scripts/hopfield_conf/kxkm_launch.sh
} >> "$LOG"

# wall-time caps: create the STOP file when exceeded
DONE=$W/logs/.campaign-done
rm -f "$DONE"
( end=$((SECONDS + WALL_CAP_P8))
  while [ $SECONDS -lt $end ]; do sleep 30; [ -e "$DONE" ] && exit 0; done
  [ -e "$STOP" ] || { echo "$(ts) WALL_CAP P8c reached" >> "$LOG"; touch "$STOP"; }
) > /dev/null 2>&1 &
WATCH=$!

laneA() {
  echo "$(ts) START laneA p8c N=16384" >> "$LOG"
  nice -n 19 "$PY" -m experiments.hopfield_dream_conf.p8c_nocap --confirmatory \
      --out "$W/results/hopfield_p8c_nocap/n16384" --ns 16384 --alphas 0.5 --workers "$LANE_A_WORKERS" \
      --max-stoch-secs "$P8C_MAX_STOCH_SECS" --stop-file "$STOP" \
      >> "$W/logs/p8c_n16384.log" 2>&1
  echo "$(ts) STOP laneA rc=$?" >> "$LOG"
}
laneB() {
  echo "$(ts) START laneB p1bis" >> "$LOG"
  timeout "$WALL_CAP_P1" nice -n 19 "$PY" -m experiments.hopfield_dream_conf.p1bis_ext \
      --confirmatory --out "$W/results/hopfield_p1bis_ext" --workers "$LANE_B_WORKERS" \
      --stop-file "$STOP" >> "$W/logs/p1bis.log" 2>&1
  echo "$(ts) STOP laneB-p1bis rc=$?" >> "$LOG"
  [ -e "$STOP" ] && return
  echo "$(ts) START laneB p8c N=4096" >> "$LOG"
  nice -n 19 "$PY" -m experiments.hopfield_dream_conf.p8c_nocap --confirmatory \
      --out "$W/results/hopfield_p8c_nocap/n4096" --ns 4096 --workers "$LANE_B_WORKERS" \
      --max-stoch-secs "$P8C_MAX_STOCH_SECS" --stop-file "$STOP" >> "$W/logs/p8c_n4096.log" 2>&1
  echo "$(ts) STOP laneB-p8c rc=$?" >> "$LOG"
}
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "$(ts) DRY_RUN: all checks passed, no lane started" | tee -a "$LOG"; touch "$DONE"; exit 0
fi
laneA & PA=$!
laneB & PB=$!
wait $PA $PB
touch "$DONE"
for d in hopfield_p1bis_ext hopfield_p8c_nocap/n4096 hopfield_p8c_nocap/n16384; do
  [ -d "$W/results/$d" ] && (cd "$W/results/$d" && sha256sum ./*.jsonl ./run_info.json > SHA256SUMS)
done
echo "$(ts) END campaign" >> "$LOG"
