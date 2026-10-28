"""macM1 day-model pause lifecycle (campaign/macm1/day_model_paused.sh).

macOS only (zsh, memory_pressure, caffeinate); skipped elsewhere. Uses a
temporary pause flag and an unused port, never the real day model."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[3] / "experiments"
          / "procgen_multiagent_gtm_comm" / "campaign" / "macm1"
          / "day_model_paused.sh")

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not shutil.which("zsh")
    or not shutil.which("memory_pressure"),
    reason="macOS launcher (zsh, memory_pressure)")


def _env(tmp_path, **kw):
    env = dict(os.environ, PAUSE_FLAG=str(tmp_path / "mtplx" / "paused"),
               DAY_MODEL_PORT="1", SETTLE_S="0", MIN_FREE_PCT="0",
               DETACH_WAIT_S="20")
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _events(log: Path) -> list[str]:
    out = []
    for line in log.read_text().splitlines():
        if line.startswith("{"):
            out.append(json.loads(line)["state"])
    return out


def _wait_gone(path: Path, timeout=20.0) -> bool:
    t = time.time() + timeout
    while time.time() < t:
        if not path.exists():
            return True
        time.sleep(0.1)
    return False


# Paths with quotes, dollar signs, backticks and spaces must stay data.
WEIRD = "we'ird $HOME `x` dir"


def test_flag_held_during_command_and_lifted_after(tmp_path):
    d = tmp_path / WEIRD
    d.mkdir()
    flag = tmp_path / "mtplx" / "paused"
    seen = d / "seen"
    cmd = ["/bin/sh", "-c", 'test -e "$1" && echo yes >"$2"; exit 7', "_",
           str(flag), str(seen)]
    p = subprocess.run(["zsh", str(SCRIPT), str(d / "st"), str(d / "log"),
                        "--", *cmd], env=_env(tmp_path))
    assert p.returncode == 7
    assert seen.read_text().strip() == "yes"
    assert not flag.exists()
    assert _events(d / "log") == ["day_model_paused", "running",
                                  "day_model_resumed"]
    assert json.loads((d / "log").read_text().splitlines()[-1])["rc"] == 7


def test_existing_flag_is_not_taken_over(tmp_path):
    flag = tmp_path / "mtplx" / "paused"
    flag.parent.mkdir()
    flag.write_text("someone else")
    p = subprocess.run(["zsh", str(SCRIPT), str(tmp_path / "st"),
                        str(tmp_path / "log"), "--", "/usr/bin/true"],
                       env=_env(tmp_path))
    assert p.returncode == 2
    assert flag.read_text() == "someone else"
    assert (tmp_path / "st").read_text().startswith("refused")


def test_memory_refusal_lifts_the_pause(tmp_path):
    flag = tmp_path / "mtplx" / "paused"
    p = subprocess.run(["zsh", str(SCRIPT), "--detach", str(tmp_path / "st"),
                        str(tmp_path / "log"), "--", "/usr/bin/true"],
                       env=_env(tmp_path, MIN_FREE_PCT=101),
                       capture_output=True, text=True)
    assert p.returncode == 2 and "memory free" in p.stderr
    assert _wait_gone(flag)
    assert _events(tmp_path / "log")[-1] == "day_model_resumed"


def test_detached_wrapper_owns_cleanup_on_term(tmp_path):
    flag = tmp_path / "mtplx" / "paused"
    p = subprocess.run(["zsh", str(SCRIPT), "--detach", str(tmp_path / "st"),
                        str(tmp_path / "log"), "--", "/bin/sleep", "300"],
                       env=_env(tmp_path), capture_output=True, text=True)
    assert p.returncode == 0 and "started: wrapper pid" in p.stdout
    assert flag.exists()
    wpid = int(p.stdout.split("wrapper pid ")[1].split(",")[0])
    child = int((tmp_path / "st").read_text().split("command pid ")[1]
                .split(",")[0])
    os.kill(wpid, signal.SIGTERM)
    assert _wait_gone(flag)
    t = time.time() + 10
    while time.time() < t:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("command survived the wrapper's TERM")
    assert _events(tmp_path / "log")[-1] == "day_model_resumed"


def test_heavy_lock_is_taken_before_the_pause(tmp_path):
    """HEAVY_LOCK (freeze plan C6): while another job holds the lock the
    detached wrapper is queued and the day model is NOT paused yet."""
    lock = tmp_path / "heavy.lock"
    flag = tmp_path / "mtplx" / "paused"
    holder = subprocess.Popen(["/usr/bin/lockf", "-k", str(lock), "/bin/sleep", "4"])
    try:
        time.sleep(0.5)
        p = subprocess.run(
            ["zsh", str(SCRIPT), "--detach", str(tmp_path / "st"),
             str(tmp_path / "log"), "--", "/bin/sleep", "1"],
            env=_env(tmp_path, HEAVY_LOCK=lock), capture_output=True, text=True)
        assert p.returncode == 0 and "queued" in p.stdout, p.stdout + p.stderr
        assert not flag.exists()  # not paused while waiting for the lock
    finally:
        holder.wait(timeout=10)
    t = time.time() + 20
    while time.time() < t and "day_model_resumed" not in (
            (tmp_path / "log").read_text() if (tmp_path / "log").exists() else ""):
        time.sleep(0.2)
    assert _events(tmp_path / "log") == ["day_model_paused", "running",
                                         "day_model_resumed"]
    assert not flag.exists()
