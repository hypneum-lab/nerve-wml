"""Frozen N10-C reference of the six arm implementations (test fixture).

Verbatim copies of ``harness/comm.py``, ``harness/policy.py`` and
``harness/gtm_mlx.py`` at the N10-C freeze tag ``n10c-freeze-20261024``
(commit ``2ce4214b9e587328c3c4eab64dce4650409edd7d``), with their relative
imports intact. They are never edited: the N10-C-bis invariance test
(``test_bis_invariance.py``) builds every N10-C arm from this package and
from the live harness with the same key and requires bit-identical
initial parameters, identical parameter counts (the N10-C stage-2
manifest table) and an identical forward pass.
"""
