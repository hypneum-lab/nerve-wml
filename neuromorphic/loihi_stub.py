"""Loihi 2 compiler stub -- historical interface, no supported path.

This module documents the API surface that nerve-wml once reserved for an
Intel Loihi 2 compilation step through the ``lava-nc`` SDK. It has never
compiled anything and no longer promises a Loihi 2 deployment path:

* Intel's Lava repositories (``lava-nc/lava``, ``lava-nc/lava-dl``) are
  archived on GitHub (``archived: true`` via the GitHub API, checked
  2026-10-25); their last push dates from 2026-05-13. Nothing is
  maintained upstream to wire in.
* nerve-wml never had Loihi 2 hardware or Intel NRC access.

The class is kept so that code importing it keeps working and so that CI
still detects that no hardware integration exists. Any future neuromorphic
target is a new, documented decision; see
``docs/neuromorphic/deployment-guide.md`` section "Loihi 2".

Plan 6 Task 9 (2026-04-18); requalified 2026-10-25 (phase 0, issue #64).
"""
from __future__ import annotations


class LoihiCompiler:
    """Placeholder kept for import compatibility; Loihi 2 is not a target."""

    @staticmethod
    def compile(artefact: dict) -> None:
        """Always raise: there is no supported Loihi 2 compilation path.

        The ``lava-nc`` SDK this stub was written against is archived
        upstream (last push 2026-05-13), so the historical recipe
        (``pip install lava-nc`` then ``lava.lib.dl.netx.hdf5``) is not a
        supported integration and is not promised by nerve-wml.
        """
        raise NotImplementedError(
            "No supported Loihi 2 compilation path: the `lava-nc` SDK is "
            "archived upstream (GitHub lava-nc/lava, last push 2026-05-13) "
            "and nerve-wml does not target Loihi 2. The INT8 artefact and "
            "MockNeuromorphicRunner remain the supported path. See "
            "docs/neuromorphic/deployment-guide.md §Loihi 2."
        )
