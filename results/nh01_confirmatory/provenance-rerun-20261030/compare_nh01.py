"""Compare a provenance re-run with the committed NH-01 artefacts.

Usage: python -I compare_nh01.py <committed_dir> <rerun_dir> <out_json>
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

TIMING = {"task_seconds"}


def load_cells(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            c = json.loads(line)
            key = tuple((k, c[k]) for k in sorted(c) if k not in TIMING
                        and k in ("N", "seed", "alpha", "P", "arm", "t",
                                  "temp", "flips", "ancestor_m"))
            out[key] = c
    return out


def strip_meta(o):
    if isinstance(o, dict):
        return {k: strip_meta(v) for k, v in o.items() if k != "run_meta"}
    if isinstance(o, list):
        return [strip_meta(v) for v in o]
    return o


def main():
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rep = {}
    ca, cb = load_cells(a / "cells.jsonl.gz"), load_cells(b / "cells.jsonl.gz")
    rep["n_cells"] = [len(ca), len(cb)]
    rep["keys_equal"] = set(ca) == set(cb)
    diff_fields = {}
    max_abs = {}
    n_cells_diff = 0
    for k in set(ca) & set(cb):
        x, y = ca[k], cb[k]
        d = False
        for f in set(x) | set(y):
            if f in TIMING:
                continue
            if json.dumps(x.get(f)) != json.dumps(y.get(f)):  # NaN == NaN
                d = True
                diff_fields[f] = diff_fields.get(f, 0) + 1
                try:
                    m = abs(float(x[f]) - float(y[f]))
                    max_abs[f] = max(max_abs.get(f, 0.0), m)
                except (TypeError, ValueError, KeyError):
                    pass
        n_cells_diff += d
    rep["cells_differing_non_timing"] = n_cells_diff
    rep["fields_differing"] = diff_fields
    rep["max_abs_diff"] = max_abs
    rep["timing_fields_ignored"] = sorted(TIMING)
    rep["line_order_equal"] = list(ca) == list(cb)
    rep["note"] = ("cells compared by key; NaN compared as NaN; cells file bytes "
                   "differ by line order (imap_unordered) and task_seconds")
    sa = json.loads((a / "summary.json").read_text())
    sb = json.loads((b / "summary.json").read_text())
    rep["summary_equal_excl_run_meta"] = json.dumps(
        strip_meta(sa), sort_keys=True) == json.dumps(strip_meta(sb), sort_keys=True)
    rep["summary_bytes_equal"] = (a / "summary.json").read_bytes() == (
        b / "summary.json").read_bytes()
    for n in ("cells.jsonl.gz", "summary.json", "figure.png"):
        ha = hashlib.sha256((a / n).read_bytes()).hexdigest()
        hb = hashlib.sha256((b / n).read_bytes()).hexdigest()
        rep[f"sha256_{n}"] = {"committed": ha, "rerun": hb, "equal": ha == hb}
    out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
