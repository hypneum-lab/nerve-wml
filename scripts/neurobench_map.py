"""CLI: map one nerve-wml validation outcome to a NeuroBench row.

Usage:
    uv run python -m scripts.neurobench_map \
        --substrate BioWML --correct 82 --total 100 \
        --synops 11000 --params 4096 --latency-ms 12.5 \
        --out reports/neurobench_biowml.json

Emits a flat JSON row (see neuromorphic.neurobench_mapping) that can
be dropped into a NeuroBench comparison table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neuromorphic.neurobench_mapping import (
    ValidationOutcome,
    map_to_neurobench,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", required=True)
    parser.add_argument("--correct", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--synops", type=int, required=True)
    parser.add_argument("--params", type=int, required=True)
    parser.add_argument("--latency-ms", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    outcome = ValidationOutcome(
        substrate=args.substrate,
        n_correct=args.correct,
        n_total=args.total,
        total_synops=args.synops,
        param_count=args.params,
        latency_ms=args.latency_ms,
    )
    try:
        result = map_to_neurobench(outcome)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result.as_row(), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
