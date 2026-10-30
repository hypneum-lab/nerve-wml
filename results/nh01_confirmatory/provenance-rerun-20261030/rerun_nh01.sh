#!/usr/bin/env bash
# Provenance re-run of an NH-01 run from a clean clone at the prereg tag.
# Usage: rerun_nh01.sh <clone_dir> <tag> <module> <workers> <out_dir>
set -euo pipefail
CLONE=$1; TAG=$2; MOD=$3; WORKERS=$4; OUT=$5
PY=$HOME/scratch/nh01/venv/bin/python
mkdir -p "$OUT/provenance"
P=$OUT/provenance
cd "$CLONE"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

{
  echo "tag: $TAG"
  echo "head: $(git rev-parse HEAD)"
  echo "tag_object_local: $(git rev-parse "refs/tags/$TAG")"
  echo "tag_target_local: $(git rev-parse "$TAG^{commit}")"
  echo "describe: $(git describe --tags --exact-match)"
  echo "origin: $(git remote get-url origin)"
  echo "hostname: $(hostname)"
  echo "nproc_all: $(nproc --all)"
  echo "uptime_at_launch: $(uptime)"
  echo "module: $MOD"
  echo "workers: $WORKERS"
  echo "nice: 19"
  echo "env: OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
  echo "python: $($PY -V 2>&1) ($PY)"
} > "$P/launch.txt"

# GitHub-side tag record, fetched before launch.
gh api "repos/hypneum-lab/nerve-wml/git/ref/tags/$TAG" > "$P/github_tag_ref.json"
TAGOBJ=$(python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["object"]["sha"])' "$P/github_tag_ref.json")
gh api "repos/hypneum-lab/nerve-wml/git/tags/$TAGOBJ" > "$P/github_tag_object.json"
echo "github_fetch_utc: $(date -u +%FT%TZ)" >> "$P/launch.txt"

git status --porcelain > "$P/git_status_before.txt"
if [ -s "$P/git_status_before.txt" ]; then
  echo "dirty tree, abort" >&2; exit 2
fi
git ls-files -s > "$P/git_ls_files.txt"
sha256sum experiments/nh01_hopfield_dream/*.py > "$P/code_sha256.txt"
git show "$TAG:experiments/nh01_hopfield_dream/" >/dev/null
for f in experiments/nh01_hopfield_dream/*.py; do
  echo "$(git show "$TAG:$f" | sha256sum | cut -d" " -f1)  $f"
done > "$P/code_sha256_at_tag.txt"
cmp <(sed 's/ .*//' "$P/code_sha256.txt") \
    <(sed 's/ .*//' "$P/code_sha256_at_tag.txt")
"$HOME/.local/bin/uv" pip freeze --python "$PY" > "$P/uv_pip_freeze.txt"

echo "start_utc: $(date -u +%FT%TZ)" >> "$P/launch.txt"
set +e
nice -n 19 "$PY" -m "experiments.nh01_hopfield_dream.$MOD" \
  --workers "$WORKERS" --out "$OUT" > "$P/stdout.log" 2>&1
RC=$?
set -e
echo "end_utc: $(date -u +%FT%TZ)" >> "$P/launch.txt"
echo "exit_code: $RC" >> "$P/launch.txt"
git status --porcelain > "$P/git_status_after.txt"
exit $RC
