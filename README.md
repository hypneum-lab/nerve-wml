# nerve-wml

Substrate-agnostic nerve protocol for inter-WML (World Model Language) communication. Research engine — see `docs/superpowers/specs/2026-04-18-nerve-wml-design.md` for the full design.

## Install

```bash
uv sync --all-extras
```

## Run tests

```bash
uv run pytest
```

## Cited in

- **dreamOfkiki — Paper 1 v0.2 (2026-04-19), §7.4 cross-substrate portability** — [github.com/genial-lab/dream-of-kiki](https://github.com/genial-lab/dream-of-kiki). The Gate W and Gate M measurements reported in this repository (MlpWML / LifWML polymorphism on FlowProxyTask and HardFlowProxyTask) provide the empirical corroboration cited in Paper 1 of the parent research program as independent evidence of the substrate-agnosticism principle (DR-3 Conformance Criterion). OSF pre-registration : [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN).

## Program context

This repository is part of the **genial-lab** research organization, which develops executable formal frameworks for cognitive AI. The programmatic parent is the `dreamOfkiki` project (paper 1 formal framework, paper 2 empirical). nerve-wml serves as a reference implementation for the **substrate-agnostic communication principle** the parent framework relies on.

Sibling repositories:

- [dream-of-kiki](https://github.com/genial-lab/dream-of-kiki) — formal framework (axioms DR-0..DR-4, Conformance Criterion, Paper 1)
- [kiki-flow-research](https://github.com/genial-lab/kiki-flow-research) — Wasserstein-gradient-flow engine (upstream)
- [micro-kiki](https://github.com/genial-lab/micro-kiki) — 35 domain-expert MoE-LoRA deployable instance
- nerve-wml (this repo) — substrate-agnostic Nerve Protocol + cross-substrate polymorphism measurement

## License

MIT (code) + CC-BY-4.0 (docs).
