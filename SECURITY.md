# Security policy — nerve-wml

nerve-wml is a **research code base** providing a substrate-agnostic
Nerve Protocol reference implementation for the dreamOfkiki research
program. Correctness and reproducibility of the Gate W / Gate M
measurements cited in Paper 1 are our top priorities ; security is
handled on a best-effort basis.

## Scope

This policy applies to :

- The code in this repository (`genial-lab/nerve-wml`)
- The `MlpWML` and `LifWML` substrate implementations and their shared
  `Nerve` Protocol
- The measurement scripts (`scripts/run_w2_hard`, `scripts/merge_pilot`)
  whose outputs are cited in the parent paper
- The documentation in `docs/`

It does *not* apply to dependencies (NumPy, PyTorch, surrogate-gradient
libraries).

## Reporting a vulnerability

If you discover a security issue that could compromise :

- the determinism of the FlowProxyTask / HardFlowProxyTask runs under
  fixed seeds
- the integrity of the polymorphism-gap measurements cited in
  dreamOfkiki Paper 1 v0.2 §7.4
- silent numerical errors in `LifWML` spike-path integration
- silent numerical errors in `MlpWML` gradient flow

please report it **privately** via one of these channels :

1. Email : `clement@saillant.cc` — subject starting with `[SECURITY]`
2. GitHub Private Vulnerability Reporting :
   https://github.com/genial-lab/nerve-wml/security/advisories/new

Please include :

- a description of the issue
- reproduction steps (seed, commit SHA, substrate variant)
- the measured value of the affected metric if applicable
- suggested mitigation if available

We aim to acknowledge reports within **5 business days** and publish
a fix within **30 days** for critical issues. Reporters will be
credited unless they prefer otherwise.

## Out of scope

- General questions about the polymorphism gap on harder tasks — open
  a regular GitHub issue.
- Dependency vulnerabilities — report upstream.
- Licensing or byline questions — see `LICENSE` and `CONTRIBUTORS.md`.

## Threat model

Expected threats : *inadvertent reproducibility regressions* and
*silent numerical invalidation* of the measurements cited in the
parent paper. Write-access abuse is outside the model ; review
happens via PR discipline.
