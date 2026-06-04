import json
import subprocess
import sys


def test_script_emits_a_neurobench_row(tmp_path):
    out = tmp_path / "row.json"
    cmd = [
        sys.executable, "-m", "scripts.neurobench_map",
        "--substrate", "BioWML",
        "--correct", "82", "--total", "100",
        "--synops", "11000", "--params", "4096",
        "--latency-ms", "12.5",
        "--out", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    row = json.loads(out.read_text())
    assert row["task"] == "streaming_classification"
    assert row["substrate"] == "BioWML"
    assert abs(row["accuracy"] - 0.82) < 1e-9
    assert row["synaptic_ops"] == 11000


def test_script_rejects_zero_total(tmp_path):
    cmd = [
        sys.executable, "-m", "scripts.neurobench_map",
        "--substrate", "MlpWML",
        "--correct", "0", "--total", "0",
        "--synops", "0", "--params", "1",
        "--latency-ms", "1.0",
        "--out", str(tmp_path / "x.json"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
