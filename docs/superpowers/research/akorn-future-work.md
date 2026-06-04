# AKOrN comparison — scoped future work

The GTM ablation (Task 9) compares `GammaThetaMultiplexer` against a plain
learned gating control (`SimpleGatingMultiplexer`) and measures synchrony
collapse. A third, stronger comparison point is **AKOrN** — Artificial
Kuramoto Oscillatory Neurons (Miyato et al., ICLR 2025, arXiv:2410.13821) —
which replaces threshold units with Kuramoto oscillators whose phase
dynamics give them an explicit synchrony mechanism.

## Why it is out of scope for this plan

- AKOrN needs a Kuramoto ODE integrator and a phase-aware read-out; that is
  a self-contained module larger than the rest of this suite combined.
- nerve-wml's substrates (`MlpWML`, `LifWML`, `TransformerWML`) are
  feed-forward; an AKOrN substrate would be a fourth `track_w` substrate,
  not a drop-in baseline.

## What lands instead

The synchrony-collapse analysis in Task 9 (`_synchrony_index`) already
captures the failure mode AKOrN is designed to avoid. If GTM's synchrony
index stays well below 1.0, the band-multiplexing claim holds without an
AKOrN head-to-head.

## Follow-up

Tracked as a future `track_w/akorn_wml.py` substrate: implement the Kuramoto
update, then add an `akorn` arm to `scripts/gtm_ablation_pilot.py`. Estimated
1-2 tasks of their own; deferred to a later validation plan.
