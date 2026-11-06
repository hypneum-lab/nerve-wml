"""N10-C-bis gate A+ lanes (campaign/{kxkm,macm1}/gate_aplus_lane.sh):
syntax, fixed order, trunk SHAs and trainer flags, without running them."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.procgen_multiagent_gtm_comm.harness import gate_g2_decide as g

CAMP = (Path(__file__).resolve().parents[3] / "experiments"
        / "procgen_multiagent_gtm_comm" / "campaign")
KXKM = CAMP / "kxkm" / "gate_aplus_lane.sh"
MACM1 = CAMP / "macm1" / "gate_aplus_lane.sh"


def test_kxkm_lane_parses():
    subprocess.run(["bash", "-n", str(KXKM)], check=True)


@pytest.mark.skipif(not shutil.which("zsh"), reason="zsh absent")
def test_macm1_lane_parses():
    subprocess.run(["zsh", "-n", str(MACM1)], check=True)


def _order(text):
    m = re.search(r"^ORDER=\(([^)]*)\)", text, re.M)
    return m.group(1).split()


def test_order_and_hosts_follow_the_runbook():
    assert _order(KXKM.read_text()) == [
        "9104:eng", "9104:nocomm", "9104:nocomm_rep",
        "9106:nocomm", "9106:nocomm_rep", "9106:eng"]
    assert _order(MACM1.read_text()) == [
        "9105:eng", "9105:nocomm", "9105:nocomm_rep"]
    for lane, host in ((KXKM, "kxkm"), (MACM1, "macm1")):
        seeds = {int(c.split(":")[0]) for c in _order(lane.read_text())}
        assert {g.SEED_HOST[s] for s in seeds} == {host}
        assert f"HOST={host}" in lane.read_text()


def test_trunk_shas_match_the_decision_script():
    assert g.TRUNK_SHA[9104] in KXKM.read_text()
    assert g.TRUNK_SHA[9105] in MACM1.read_text()


def test_trainer_flags():
    common = ("--condition closed --steps $STEPS", "--log-every 200000",
              "--ckpt-every 1000000", "--preset fix-c-coin-pretrain",
              "--dev-kind gate_dev", "STEPS=10000000",
              "> \"$d.result.sealed.json\"", "nocomm_rep) s=$((S + 10)) arm=nocomm")
    for lane in (KXKM, MACM1):
        t = lane.read_text()
        for flag in common:
            assert flag in t, (lane.name, flag)
    assert "--micro-batch-pairs 512 --clear-cache --n-workers 8" in KXKM.read_text()
    m1 = MACM1.read_text()
    assert "--micro-batch-pairs" not in m1.split("run_one()")[1]
    assert "--clear-cache" not in m1.split("run_one()")[1]
    assert "gpu_wait.sh" in KXKM.read_text()
    assert "day_model_paused.sh" in m1 and "HEAVY_LOCK" in m1
    for lane in (KXKM, MACM1):
        assert "--min-free-gb" in lane.read_text()
        assert "gate_g2_decide" in lane.read_text()  # OK / FAIL only
