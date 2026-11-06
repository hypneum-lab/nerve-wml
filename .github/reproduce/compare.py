"""Compare a Gate W pilot report against the committed expected values.

Usage: python .github/reproduce/compare.py ACTUAL.json EXPECTED.json

Every metric of the Gate W report is a ratio of integer counts
(accuracies on a fixed evaluation batch, and relative gaps between
them), so a genuine reproduction matches exactly; the absolute
tolerance below only absorbs float formatting. A single flipped
prediction moves a metric by far more than the tolerance.
"""

import json
import math
import sys

ABS_TOL = 1e-9


def main(actual_path: str, expected_path: str) -> int:
    with open(actual_path) as f:
        actual = json.load(f)
    with open(expected_path) as f:
        expected = json.load(f)

    failures = []
    for key, want in expected.items():
        if key not in actual:
            failures.append(f"{key}: missing from report")
            continue
        got = actual[key]
        if isinstance(want, bool):
            ok = got is want
        else:
            ok = math.isclose(got, want, rel_tol=0.0, abs_tol=ABS_TOL)
        status = "ok" if ok else "MISMATCH"
        print(f"{status:8} {key}: got {got!r}, expected {want!r}")
        if not ok:
            failures.append(key)

    if failures:
        print(f"FAIL: {len(failures)} metric(s) differ (abs_tol={ABS_TOL})")
        return 1
    print(f"PASS: all {len(expected)} metrics match (abs_tol={ABS_TOL})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
