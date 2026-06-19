# Scripts

Build, dev, and automation scripts. 63 Python + 3 Bash.

## Conventions

- Executable: `chmod +x script.sh`
- Header comment with usage, e.g. `# Usage: ./script.sh --flag value`
- Bash: start with `set -euo pipefail`
- Exit codes: 0 success, non-zero with a clear message

## Adding new scripts

- Live in this directory
- Add an alias in the project's standard task runner if used often
- Keep them single-purpose; chain instead of mixing concerns
- **Before adding numerical claims** to docs/superpowers/research/ or papers/: ensure the corresponding JSON output is in `docs/superpowers/research/` so `scripts/factcheck_audit.py --ci` covers it.

## Anti-patterns

- Don't hardcode absolute paths — use `$HOME`, env vars, or compute relative
- Don't skip error handling — `set -e` + explicit checks at boundaries
- Don't assume tools exist — `command -v <tool>` first
- Don't write non-idempotent operations without `--dry-run` support

## Script inventory

Run `ls scripts/*.py` or `ls scripts/*.sh` for the current inventory — not enumerated here to avoid drift.
