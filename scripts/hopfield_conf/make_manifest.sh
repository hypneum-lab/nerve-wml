#!/bin/bash
# Print the freeze-manifest file table (path + SHA-256) for P1bis-ext / P8c-nocap.
# Run from the repo root; paste the output into
# docs/prereg/hopfield-p1bis-ext/freeze-manifest.md, then record that file's own
# SHA-256 in the annotated tag message.
set -eu
FILES=(
  docs/prereg/hopfield-p1bis-ext/prereg.md
  docs/prereg/hopfield-p8c-nocap/prereg.md
  experiments/hopfield_dream_conf/__init__.py
  experiments/hopfield_dream_conf/p1bis_ext.py
  experiments/hopfield_dream_conf/p8c_nocap.py
  experiments/hopfield_dream_conf/stats.py
  experiments/hopfield_dream_conf/analyse.py
  experiments/hopfield_dream_conf/power.py
  experiments/hopfield_dream_conf/requirements-hopfield-conf.txt
  experiments/hopfield_dream_conf/vendor/__init__.py
  experiments/hopfield_dream_conf/vendor/hopfield.py
  experiments/hopfield_dream_conf/vendor/LICENSE
  tests/unit/test_hopfield_dream_conf_stats.py
  tests/unit/test_hopfield_dream_conf_drivers.py
  scripts/hopfield_conf/kxkm_launch.sh
  scripts/hopfield_conf/make_manifest.sh
  .ruff.toml
)
echo "| File | SHA-256 |"
echo "|---|---|"
for f in "${FILES[@]}"; do
  h=$( (sha256sum "$f" 2>/dev/null || shasum -a 256 "$f") | cut -d' ' -f1)
  echo "| \`$f\` | \`$h\` |"
done
echo
echo "Tree \`experiments/hopfield_dream_conf\` at HEAD: \`$(git rev-parse HEAD:experiments/hopfield_dream_conf)\`"
