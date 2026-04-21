"""Pin the [axioms] extras group presence and target."""
from __future__ import annotations

import tomllib
from pathlib import Path


def _load_pyproject() -> dict:
    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    return tomllib.loads(path.read_text())


def test_version_is_1_8_0():
    assert _load_pyproject()["project"]["version"] == "1.8.0"


def test_axioms_extras_group_exists():
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert "axioms" in extras


def test_axioms_extras_references_dreamofkiki_v0_9_0():
    extras = _load_pyproject()["project"]["optional-dependencies"]
    axioms = extras["axioms"]
    assert len(axioms) == 1
    spec = axioms[0]
    assert "dreamofkiki" in spec
    assert "hypneum-lab/dream-of-kiki" in spec
    assert "v0.9.0" in spec
