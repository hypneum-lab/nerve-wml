# nerve-wml

**Substrate-agnostic nerve protocol for inter-module communication in hybrid neural systems.**

Research engine that validates a discrete-code communication layer between heterogeneous neural modules (World Model Languages, or WMLs). Modules exchange **neuroletters** over a sparse learned topology, multiplexed on gamma/theta rhythms, and converted between local codebooks by per-edge transducers. The paper draft is at [`papers/paper1/main.tex`](papers/paper1/main.tex); the full spec is at [`docs/superpowers/specs/2026-04-18-nerve-wml-design.md`](docs/superpowers/specs/2026-04-18-nerve-wml-design.md).

## Status — 11 gates

| Tag | What it proves |
|---|---|
| [`gate-p-passed`](../../releases/tag/gate-p-passed) | Track-P protocol simulator correct on toy signals |
| [`gate-w-passed`](../../releases/tag/gate-w-passed) | `MlpWML` and `LifWML` interoperate with < 5 % gap through the same nerve (N=4) |
| [`gate-m-passed`](../../releases/tag/gate-m-passed) | Merge fine-tunes only transducers; retains ≥ 95 % of mock-baseline accuracy |
| [`gate-m2-passed`](../../releases/tag/gate-m2-passed) | Four scientific shortcuts from §13.1 resolved with honest measurements |
| [`gate-scale-passed`](../../releases/tag/gate-scale-passed) | Polymorphie + continual learning hold at N=16 pools; router stays connected to N=32 |
| [`gate-interp-passed`](../../releases/tag/gate-interp-passed) | Per-WML `code → concept` semantics table rendered as HTML |
| [`gate-neuro-passed`](../../releases/tag/gate-neuro-passed) | LifWML → INT8 artefact → pure-numpy mock runner (Loihi / Akida stubs documented) |
| [`gate-dream-passed`](../../releases/tag/gate-dream-passed) | ε-trace consolidation bridge to dream-of-kiki (schema v0; partial — awaits kiki_oniric v0.5+) |
| [`gate-adaptive-passed`](../../releases/tag/gate-adaptive-passed) | Per-WML alphabet shrinks/grows via `active_mask` + transducer resize |
| [`gate-llm-advisor-passed`](../../releases/tag/gate-llm-advisor-passed) | Env-gated, never-raising `NerveWmlAdvisor` for micro-kiki, < 50 ms warm latency |

Plus two paper drafts: [`paper-v0.2-draft`](../../releases/tag/paper-v0.2-draft) and [`paper-v0.3-draft`](../../releases/tag/paper-v0.3-draft).

## Install

```bash
uv sync --all-extras
```

Python 3.12+, macOS arm64 (MLX-friendly) or Linux x86_64. No vendor SDK deps are pulled by default (Loihi, Akida, dream-of-kiki, sentence-transformers are all optional integrations).

## Run the suite

```bash
uv run pytest -m "not slow"    # 220+ tests under 80 s on commodity M-series
uv run pytest                  # full suite incl. paper figure rendering
uv run pytest --cov=nerve_core --cov=track_p --cov=track_w --cov=bridge --cov=harness --cov=interpret --cov=neuromorphic
```

## Reproduce the gate numbers

```bash
uv run python scripts/track_p_pilot.py       # Gate P (+ Task 6 ablation)
uv run python scripts/track_w_pilot.py       # Gate W
uv run python scripts/track_w_pilot.py scale # Gate Scale (N=16, N=32)
uv run python scripts/merge_pilot.py         # Gate M
uv run python scripts/interpret_pilot.py     # Gate Interp (emits reports/interp/*.html)
uv run python scripts/adaptive_pilot.py      # Gate Adaptive
```

## Build the paper

```bash
uv run python scripts/render_paper_figures.py   # regenerate figures from frozen golden NPZs
cd papers/paper1 && tectonic main.tex           # or pdflatex, bibtex, pdflatex, pdflatex
```

## Integrations (env-gated, default off)

- **Dream consolidation**: `DREAM_CONSOLIDATION_ENABLED=1` + install `dream-of-kiki` locally → `bridge.dream_bridge.DreamBridge`.
- **LLM advisor (micro-kiki)**: `NERVE_WML_ENABLED=1` + `NERVE_WML_CHECKPOINT_PATH=/path/to/checkpoint` → `bridge.kiki_nerve_advisor.NerveWmlAdvisor`. Wiring recipe: [`docs/integration/micro-kiki-wiring.md`](docs/integration/micro-kiki-wiring.md).
- **Neuromorphic hardware**: install `lava-nc` or `akida` → wire in `neuromorphic.loihi_stub` / `neuromorphic.akida_stub`. Schema v0: [`docs/neuromorphic/deployment-guide.md`](docs/neuromorphic/deployment-guide.md).

## Cited in

- **dreamOfkiki — Paper 1 v0.2 (2026-04-19), §7.4 cross-substrate portability** — [github.com/genial-lab/dream-of-kiki](https://github.com/genial-lab/dream-of-kiki). The Gate W and Gate M measurements reported here (MlpWML / LifWML polymorphism on FlowProxyTask and HardFlowProxyTask) provide the empirical corroboration cited in Paper 1 as independent evidence of the substrate-agnosticism principle (DR-3 Conformance Criterion). OSF pre-registration: [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN).

## Program context

This repository is part of **genial-lab**, which develops executable formal frameworks for cognitive AI. The programmatic parent is `dreamOfkiki` (paper 1 formal framework, paper 2 empirical); `nerve-wml` is the reference implementation for the substrate-agnostic communication principle.

Sibling repositories:

- [dream-of-kiki](https://github.com/genial-lab/dream-of-kiki) — formal framework (axioms DR-0..DR-4, Conformance Criterion, Paper 1)
- [kiki-flow-research](https://github.com/genial-lab/kiki-flow-research) — Wasserstein-gradient-flow engine (upstream)
- [micro-kiki](https://github.com/genial-lab/micro-kiki) — 35 domain-expert MoE-LoRA deployable instance (advisor consumer)
- **nerve-wml** (this repo) — substrate-agnostic nerve protocol + cross-substrate polymorphism

## Repository layout

```
nerve_core/        Neuroletter, Nerve + WML Protocols, invariants (N-1..N-5, W-1..W-4)
track_p/           Track-P — SimNerve, VQCodebook, Transducer, SparseRouter, AdaptiveCodebook
track_w/           Track-W — MockNerve, MlpWML, LifWML, toy tasks, training loop, pool factory
bridge/            Merge, dream, LLM advisor — SimNerveAdapter, MergeTrainer, DreamBridge, NerveWmlAdvisor
harness/           R1 reproducibility — run_registry
interpret/         Gate Interp — code_semantics, clustering, HTML renderer
neuromorphic/      Gate Neuro — spike_encoder, INT8 export, mock_runner, vendor stubs
scripts/           All gate pilots + figure renderers + freeze_golden
tests/             Unit + integration + golden NPZ regressions
docs/              specs/, integration/, neuromorphic/, dream/, interpret/
papers/paper1/     LaTeX source + bib + Makefile (figures regenerated deterministically)
```

## License

MIT (code) + CC-BY-4.0 (docs).
