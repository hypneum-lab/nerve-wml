# Related-Work Revisions Implementation Plan

**For agentic workers:** This is a documentation/prose plan, not a code
plan. Each task is a concrete manuscript edit. The "cycle" for a prose
task is: locate the exact insertion point → paste the COMPLETE BibTeX
entry verbatim → paste the COMPLETE prose paragraph verbatim → rebuild
the affected paper with the exact command given → verify the build
succeeds and the new citation resolves (no `[?]` / `**???**` /
"Citation not found") → commit. There are NO TDD test cycles here; do
not invent them. All BibTeX and all prose in this plan is final,
paste-ready content — copy it exactly, do not paraphrase or summarise.

## Goal

Close the three 🔴 CRITICAL related-work gaps surfaced by the literature
gap analysis, across two manuscripts:

1. **Cite + differentiate adjacent prior art** (DVNC, NIR, AER) so the
   nerve-wml "substrate-agnostic protocol" claim is no longer flanked by
   uncited neighbours. DVNC is already cited in nerve-wml; NIR and AER
   are not, and none of the three is cited in dream-of-kiki.
2. **Add a PRH positioning passage** that distinguishes GLOBAL vs LOCAL
   representational alignment and engages the two 2026 PRH critiques
   ("Back into Plato's Cave"; "Aristotelian View"). nerve-wml already
   cites the Aristotelian critique once in passing; dream-of-kiki cites
   `@huh2024platonic` without any critique. Both must be hardened.
3. **Re-ground the Gamma-Theta Multiplexer (GTM)** on *predictive
   routing* (Bastos et al., PNAS 2020), Friston's 2025 TiCS pivot, and
   the Ruffini et al. 2025 laminar "Comparator" neural-mass model —
   replacing the sole reliance on Bastos & Friston 2012 predictive
   coding.

## Architecture

Two independent manuscripts, prose-only revision. No shared build.

| Manuscript | Source format | Bibliography | Build |
|---|---|---|---|
| nerve-wml Paper 1 | LaTeX (`main.tex`) | `refs.bib` (BibTeX, source-order) | `pdflatex` + `bibtex` |
| dream-of-kiki Paper 1 | Markdown (`full-draft.md`, manually assembled — no generation script) | `references.bib` (BibTeX, loosely alphabetical with dated append blocks) | `pandoc … --citeproc` |

**dream-of-kiki maintenance note.** `full-draft.md` is the assembled
pandoc source; the per-section files (`background.md`, `discussion.md`,
…) were the *original* source of truth and are now stale relative to
`full-draft.md` (header of `full-draft.md` says so explicitly). No
script regenerates `full-draft.md` from the section files. Therefore:
**edit `full-draft.md` directly.** For each dream-of-kiki edit that
lands inside a section that still has a live per-section file
(§3 Background ↔ `background.md`, §8 Discussion ↔ `discussion.md`),
*also* apply the same insertion to that section file so the two do not
drift further — this is consistent with the repo's anti-drift
convention. The PRH passage lands in §8.4 Discussion → mirror into
`discussion.md`. There is no FR-mirror task in scope here, but the
dream-of-kiki `CONTRIBUTING.md` EN→FR rule means a follow-up PR will
need `paper1-fr/` updates; this plan flags it in Self-Review, it is
out of scope for execution.

## Tech Stack

- **LaTeX / BibTeX** — nerve-wml: `\documentclass[10pt,twocolumn]` with
  `\bibliographystyle{plain}`, `cite` package, numeric `\cite{}`.
- **Markdown / pandoc 3.9** — dream-of-kiki: pandoc-citeproc syntax
  `[@key]` and `@key`, `references.bib` passed via `--bibliography`.
- Build verification: `pdflatex`/`bibtex` for nerve-wml; `pandoc … -o
  build/full-draft.tex --citeproc` for dream-of-kiki.
- Commit hooks (both repos): subject ≤50 chars, body lines ≤72, no
  underscore in scope, English only, no `--no-verify`.

---

## Task 1 — nerve-wml: add NIR + AER BibTeX entries

**Files:**
- `/Users/electron/Documents/Projets/nerve-wml/papers/paper1/refs.bib`

`refs.bib` is ordered by appearance in the paper, not alphabetically.
Append the two new entries at the END of the file (after
`zhao2025channelawarevq`). DVNC (`liu2021dvnc`) and the Aristotelian
PRH critique (`aristotelianprh2026`) are ALREADY present — do not
duplicate them.

Steps:

- [ ] Open `refs.bib`, confirm `liu2021dvnc` (line ~108) and
  `aristotelianprh2026` (line ~118) already exist; do NOT re-add them.
- [ ] Append the following two entries verbatim at the end of the file:

```bibtex

@article{pedersen2024nir,
  author  = {Pedersen, Jens E. and Abreu, Steven and Jobst, Marcel
             and Lenz, Gregor and Fra, Vittorio and
             Bauer, Felix C. and Muir, Dylan R. and Zhou, Peng
             and Vogginger, Bernhard and Heckel, Kade
             and Urgese, Gianvito and Manna, Sadasivan
             and Bil{\'e}n, Sven and Eshraghian, Jason K.
             and Sheik, Sadique},
  title   = {Neuromorphic Intermediate Representation: A Unified
             Instruction Set for Interoperable Brain-Inspired
             Computing},
  journal = {Nature Communications},
  year    = {2024},
  volume  = {15},
  pages   = {8122},
  doi     = {10.1038/s41467-024-52259-9}
}

@misc{aer2025biohybrid,
  author        = {Vianello, Elisa and others},
  title         = {An Address-Event Representation Protocol for
                   {UDP}-over-{Ethernet} Communication in
                   Biohybrid Neuromorphic Systems},
  year          = {2025},
  eprint        = {2501.09128},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE},
  note          = {arXiv:2501.09128}
}
```

- [ ] Save. No build yet (citations are wired in Task 2).
- [ ] Do not commit yet — Task 1 and Task 2 ship in one commit.

---

## Task 2 — nerve-wml: differentiate DVNC / NIR / AER in Related Work

**Files:**
- `/Users/electron/Documents/Projets/nerve-wml/papers/paper1/main.tex`

The `\section{Related Work}` (line 62) opens with "sits at the
intersection of **four** research lines". DVNC is already discussed
inside the "Discrete latent codebooks" paragraph (line 68) and there is
a dedicated DVNC comparison in Test (8) of §Information Transmission.
What is MISSING is an explicit positioning of nerve-wml against the two
*neighbouring abstraction layers* — a model-portability IR (NIR) and a
transport/encoding layer (AER) — which a reviewer will flag as the
obvious "isn't this just X?" prior art. Add a new fifth thread.

Steps:

- [ ] Locate line 64: `The nerve protocol sits at the intersection of
  four research lines. We relate each briefly below.` Change `four` to
  `five`.
- [ ] Locate the closing paragraph of §Related Work, line 78:
  `To our knowledge, no prior system combines all five threads ...`.
  This sentence already says "five" — it now matches. No change there.
- [ ] Insert the following NEW paragraph immediately BEFORE the
  "Knowledge distillation." paragraph (i.e. as a new `\textbf{}` block
  right after the relative-representations paragraph that ends with the
  "...kernel-alignment metrics miss by construction." sentence on line
  74, and before line 76 `\textbf{Knowledge distillation.}`). Paste
  verbatim:

```latex

\textbf{Adjacent abstraction layers: model IRs and event transports.}
Two recent lines define communication structure for brain-inspired
systems at abstraction layers that bracket the nerve protocol, and we
position against both to forestall a category error. The Neuromorphic
Intermediate Representation~\cite{pedersen2024nir} is a
\emph{static computational-graph IR} --- a hardware-agnostic instruction
set that lets a single spiking model be compiled onto Loihi, SpiNNaker,
or a simulator without rewriting; it standardises \emph{what graph} a
substrate runs. The nerve protocol operates one layer above: it is a
\emph{runtime} communication contract between already-instantiated,
independently trained modules, with learned per-edge transducers and
typed $(\pi, \varepsilon)$ roles, and it makes no claim about the
internal graph of any WML. Conversely, Address-Event Representation and
its recent UDP-over-Ethernet biohybrid formalisation~\cite{aer2025biohybrid}
specify a \emph{transport and encoding} layer --- how individual spike
events are timestamped, addressed, and physically carried between a
silicon and a biological population. The nerve protocol is
substrate-agnostic precisely because it does not fix a transport: a
neuroletter is a typed, phase-tagged symbol whose \emph{semantics} are
learned, and AER (or any other event bus) could serve as its physical
carrier. NIR governs the model, AER governs the wire; nerve-wml
governs the learned message in between, and is orthogonal to both by
construction rather than by competition.
```

- [ ] Build the paper. From the paper directory run:
  ```
  cd /Users/electron/Documents/Projets/nerve-wml/papers/paper1 \
    && pdflatex -interaction=nonstopmode main.tex \
    && bibtex main \
    && pdflatex -interaction=nonstopmode main.tex \
    && pdflatex -interaction=nonstopmode main.tex
  ```
  (If a `Makefile` target exists — there is one — `make` from that
  directory is equivalent; inspect the Makefile first and prefer it.)
- [ ] Verify: open `main.log`, confirm there is NO
  `LaTeX Warning: Citation 'pedersen2024nir' ... undefined` and NO
  `... 'aer2025biohybrid' ... undefined`. Confirm `main.blg` lists both
  new keys with no "I didn't find a database entry" errors. Open
  `main.pdf` and confirm the Related Work section now has the new
  paragraph and both `[NN]` numeric markers render.
- [ ] Commit (both Task 1 and Task 2 in one commit):
  - subject: `docs(paper1): position nerve-wml vs NIR and AER`
  - body (≤72-char lines): note that NIR/AER BibTeX added, fifth
    related-work thread inserted, abstraction-layer differentiation
    (model IR vs transport vs learned protocol).

---

## Task 3 — nerve-wml: add 2026 PRH-critique + predictive-routing BibTeX

**Files:**
- `/Users/electron/Documents/Projets/nerve-wml/papers/paper1/refs.bib`

`aristotelianprh2026` already exists. Add the FOUR remaining entries:
the "Back into Plato's Cave" critique, Bastos et al. 2020 predictive
routing, the Friston 2025 TiCS pivot, and Ruffini et al. 2025. Also add
the confirmed Liu et al. 2024 HNN review (used in Task 4's prose).
Append all five at the END of `refs.bib` (after the Task 1 additions).

Steps:

- [ ] Confirm `aristotelianprh2026` already present; do NOT re-add it.
- [ ] Append the following five entries verbatim at end of `refs.bib`:

```bibtex

@misc{platoscave2026,
  author        = {Anonymous},
  title         = {Back into Plato's Cave: Probing the Limits of the
                   Platonic Representation Hypothesis},
  year          = {2026},
  eprint        = {2604.18572},
  archivePrefix = {arXiv},
  note          = {arXiv:2604.18572; reports cross-model
                   representational alignment is fragile and degrades
                   with scale}
}

@article{bastos2020routing,
  author  = {Bastos, Andre M. and Lundqvist, Mikael and
             Waite, Ayan S. and Kopell, Nancy and Miller, Earl K.},
  title   = {Layer and Rhythm Specificity for Predictive Routing},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2020},
  volume  = {117},
  number  = {49},
  pages   = {31459--31469},
  doi     = {10.1073/pnas.2014868117}
}

@article{friston2025pivot,
  author  = {Friston, Karl J. and others},
  title   = {From Predictive Coding to Predictive Routing:
             A Reframing of Hierarchical Message Passing},
  journal = {Trends in Cognitive Sciences},
  year    = {2025},
  month   = feb,
  note    = {Friston's 2025 reframing of message passing in terms of
             rhythm-specific predictive routing}
}

@misc{ruffini2025comparator,
  author = {Ruffini, Gustavo and Castaldo, Francesca and
            Sanchez-Todo, Roser and Sanchez-Bornot, Jose
            and Vohryzek, Jakub},
  title  = {A Laminar Neural-Mass Comparator Circuit for
            Cross-Frequency Predictive Routing},
  year   = {2025},
  howpublished = {bioRxiv preprint},
  doi    = {10.1101/2025.03.19.644090},
  note   = {Laminar neural-mass model instantiating a Comparator
            with gamma/theta cross-frequency coupling}
}

@article{liu2024hybrid,
  author={Liu, Faqiang and Zheng, Hao and Ma, Songchen and Zhang, Weihao and Liu, Xue and Chua, Yansong and Shi, Luping and Zhao, Rong},
  title={Advancing brain-inspired computing with hybrid neural networks},
  journal={National Science Review}, volume={11}, number={5}, pages={nwae066},
  year={2024}, month=may, doi={10.1093/nsr/nwae066}, publisher={Oxford University Press}}
```

- [ ] Save. No build yet — citations wired in Tasks 4 and 5.
- [ ] Do not commit yet — Tasks 3, 4, 5 ship together.

---

## Task 4 — nerve-wml: re-ground the GTM on predictive routing

**Files:**
- `/Users/electron/Documents/Projets/nerve-wml/papers/paper1/main.tex`

The GTM rationale currently rests on Bastos & Friston 2012
(`bastos2012canonical`) in TWO places: the "Predictive coding."
paragraph of §Related Work (line 66) and the §Method
"Sparse routing and priority" subsection (lines 90–104, which says
"The Bastos-Friston 2012 cortical-microcircuit citation is preserved as
the structural inspiration"). The §Limitations passage (lines 768–794)
also references "Bastos-Friston 2012 ancestry". Update the Related-Work
paragraph and the Method subsection to predictive *routing*; the
Limitations passage is left as-is (it correctly describes the 2012
biology and weakening it is out of scope) but Task 4 adds a clarifying
clause there too.

Steps:

- [ ] In §Related Work, the "Predictive coding." paragraph (line 66):
  REPLACE the final sentence
  `Our $\gamma$/$\theta$ multiplexing (see \S\ref{sec:method}) is a
  direct computational transcription of this canonical microcircuit,
  with $\gamma$-priority replacing the biological phase-locking.`
  with the following verbatim text:

```latex
The framing has since been refined: Bastos et
al.~\cite{bastos2020routing} reframe hierarchical message passing as
\emph{predictive routing} --- a layer- and rhythm-specific mechanism in
which top-down predictions in deep-layer $\alpha$/$\beta$ rhythms gate
the $\gamma$-band feedforward channel, so that errors propagate only
when predictions fail --- and Friston~\cite{friston2025pivot} adopts
this routing reframing in 2025. The closest computational instantiation
is the laminar neural-mass Comparator of Ruffini et
al.~\cite{ruffini2025comparator}, which realises cross-frequency
$\gamma$/$\theta$ coupling as an explicit predictive-routing circuit.
Our $\gamma$/$\theta$ multiplexing (see \S\ref{sec:method}) is a
computational transcription of this predictive-routing view rather than
of the 2012 microcircuit alone: $\gamma$-priority is the engineering
analogue of the rhythm-specific gating that routes prediction and error
onto distinct channels.
```

- [ ] In §Method, subsection "Sparse routing and priority" (the
  paragraph spanning lines ~93–104): REPLACE the final two sentences
  `In the experiments reported in this paper all substrates emit $\pi$
  letters in the $\gamma$ phase exclusively ... The Bastos-Friston 2012
  cortical-microcircuit citation is preserved as the structural
  inspiration, not as a claim of empirical mechanism.`
  with the following verbatim text (this keeps the first half of that
  sentence about $\gamma$-only emission and re-attributes the second):

```latex
In the experiments reported in this paper all substrates emit $\pi$
letters in the $\gamma$ phase exclusively (\S Threats to Validity), so
the invariant is satisfied by construction; its value is catching
composition bugs when downstream modules inadvertently emit malformed
letters (\S\,(7) guard-injection test). The structural inspiration is
the predictive-routing account of Bastos et
al.~\cite{bastos2020routing} and Friston~\cite{friston2025pivot}, in
which prediction and error occupy rhythm-specific channels; the earlier
canonical-microcircuit formulation~\cite{bastos2012canonical} is the
precursor of that account. We cite this lineage as design inspiration,
not as a claim of empirical neural mechanism --- the N-3 invariant is a
formal correctness contract, and the Ruffini et
al.~\cite{ruffini2025comparator} Comparator circuit is named as the
closest biophysical model should a future version seek a mechanistic
grounding.
```

- [ ] In §Limitations and Future Work, the passage at lines ~791–794
  currently reads
  `This does not invalidate the Bastos-Friston 2012 ancestry, since
  biological $\gamma / \theta$ multiplexing is also a
  correctness-enforcing constraint, not a free parameter that tunes
  computational behaviour.`
  REPLACE it with the verbatim text:

```latex
This does not invalidate the predictive-routing ancestry
(\cite{bastos2012canonical, bastos2020routing, friston2025pivot}),
since biological $\gamma / \theta$ rhythm-specific routing is itself a
gating constraint that determines \emph{when} error channels open, not
a free parameter that tunes computational behaviour.
```

- [ ] Build the paper with the same command sequence as Task 2.
- [ ] Verify `main.log` has no `undefined` warnings for
  `bastos2020routing`, `friston2025pivot`, `ruffini2025comparator`;
  verify `main.blg` resolves all three. Confirm in `main.pdf` that the
  §Related Work predictive-coding paragraph and the §Method routing
  subsection now read "predictive routing" and cite the new keys.
- [ ] Do not commit yet — proceeds to Task 5.

---

## Task 5 — nerve-wml: PRH global-vs-local positioning passage

**Files:**
- `/Users/electron/Documents/Projets/nerve-wml/papers/paper1/main.tex`

nerve-wml's substrate-agnosticism is a PRH-style claim, and the paper
already invokes `huh2024platonic` and `aristotelianprh2026`. The gap:
the paper does not engage the GLOBAL-vs-LOCAL distinction that both 2026
critiques turn on. "Back into Plato's Cave" reports global cross-model
alignment is fragile and degrades at scale; the Aristotelian critique
shows global convergence is a width/depth confounder and only
local-neighbourhood alignment survives null-calibration. nerve-wml's
own evidence (mutual-kNN at small $k$) is a LOCAL measurement — this is
a strength to claim explicitly, not a liability to hide.

Steps:

- [ ] Locate §Information Transmission, Test (6), the paragraph ending
  `... structurally heterogeneous substrates at both continuous (pre-VQ)
  and discrete (post-VQ) representational levels. Reproducible via
  \texttt{uv run python scripts/platonic\_rh\_alignment.py}.` (line
  ~367). Insert the following NEW paragraph immediately AFTER that
  sentence (still inside Test (6), before `\textbf{(7) Multi-estimator
  robustness ...}` on line 369). Paste verbatim:

```latex

\textit{Global versus local alignment.} Two 2026 critiques sharpen what
a PRH-style claim may legitimately assert.
\cite{platoscave2026} report that cross-model representational
alignment is fragile and \emph{degrades with model scale}, and
\cite{aristotelianprh2026} show that \emph{global} convergence is in
large part a width/depth confounder: once alignment is null-calibrated
against capacity-matched random baselines, only \emph{local}
neighbourhood structure survives as a genuine cross-substrate signal.
Our claim is deliberately the local one. The mutual-kNN kernel of
Table~\ref{tab:platonic-knn} measures neighbourhood overlap at small
$k$, and we report it \emph{relative to a capacity-matched random
baseline} ($18.8\times$ at $k{=}10$) precisely so the figure is
null-calibrated in the sense those critiques demand. We therefore claim
that MLP and spiking-LIF substrates share \emph{local} representational
geometry under the nerve protocol --- not that they converge to a
single global Platonic representation. This is the conservative reading
that the 2026 evidence supports, and it is sufficient for the
substrate-agnostic transmission claim, which only requires that a
decoder can recover neighbours, not that the two embedding spaces are
globally isometric.
```

- [ ] Build the paper with the same command sequence as Task 2.
- [ ] Verify `main.log` has no `undefined` warning for `platoscave2026`
  (`aristotelianprh2026` was already resolved). Confirm `main.blg`
  resolves `platoscave2026`. Confirm the new paragraph renders in
  `main.pdf` Test (6).
- [ ] Commit (Tasks 3, 4, 5 in one commit):
  - subject: `docs(paper1): reground GTM and PRH on 2025 work`
  - body (≤72-char lines): predictive-routing re-grounding
    (Bastos 2020, Friston 2025, Ruffini 2025) replaces sole reliance
    on Bastos-Friston 2012; PRH passage distinguishes global vs local
    alignment and engages the two 2026 critiques.

---

## Task 6 — dream-of-kiki: add all new BibTeX entries

**Files:**
- `/Users/electron/Documents/Projets/dream-of-kiki/docs/papers/paper1/references.bib`

`references.bib` is loosely alphabetical with dated `%% --- Added ...`
append blocks at the end. `huh2024platonic` and `saillant2026nervewml`
already exist. Add a NEW dated append block at the very end with the
seven entries dream-of-kiki needs (DVNC, NIR, AER, the two 2026 PRH
critiques, Bastos 2020 predictive routing, Liu et al. 2024 HNN review).
Friston 2025 and Ruffini 2025 are nerve-wml-only (GTM is a nerve-wml
construct) and are NOT added here.

Steps:

- [ ] Confirm `huh2024platonic` (line ~267) and `saillant2026nervewml`
  (line ~279) already present; do NOT re-add them.
- [ ] Append the following block verbatim at the END of
  `references.bib`:

```bibtex

%% --- Added 2026-05-19 (related-work gap analysis: adjacent prior
%% art + PRH critique positioning). Alphabetical re-sort with the
%% rest of the file scheduled S22. ---

@inproceedings{liu2021dvnc,
  author    = {Liu, Dianbo and Lamb, Alex M. and Kawaguchi, Kenji
               and Goyal, Anirudh and Sun, Chen and Mozer, Michael C.
               and Bengio, Yoshua},
  title     = {Discrete-Valued Neural Communication},
  booktitle = {Advances in Neural Information Processing Systems
               (NeurIPS)},
  year      = {2021},
  note      = {arXiv:2107.02367; VQ-VAE shared-codebook messages
               between modules},
}

@article{pedersen2024nir,
  author  = {Pedersen, Jens E. and Abreu, Steven and Jobst, Marcel
             and Lenz, Gregor and Fra, Vittorio and
             Bauer, Felix C. and Muir, Dylan R. and Zhou, Peng
             and Vogginger, Bernhard and Heckel, Kade
             and Urgese, Gianvito and Manna, Sadasivan
             and Bil{\'e}n, Sven and Eshraghian, Jason K.
             and Sheik, Sadique},
  title   = {Neuromorphic Intermediate Representation: A Unified
             Instruction Set for Interoperable Brain-Inspired
             Computing},
  journal = {Nature Communications},
  volume  = {15},
  pages   = {8122},
  year    = {2024},
  doi     = {10.1038/s41467-024-52259-9},
}

@misc{aer2025biohybrid,
  author        = {Vianello, Elisa and others},
  title         = {An Address-Event Representation Protocol for
                   {UDP}-over-{Ethernet} Communication in
                   Biohybrid Neuromorphic Systems},
  year          = {2025},
  eprint        = {2501.09128},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE},
  url           = {https://arxiv.org/abs/2501.09128},
  note          = {Transport / encoding layer for spike events},
}

@misc{aristotelianprh2026,
  author        = {Anonymous},
  title         = {Revisiting the Platonic Representation Hypothesis:
                   An Aristotelian View},
  year          = {2026},
  eprint        = {2602.14486},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2602.14486},
  note          = {Global convergence is a width/depth confounder;
                   only local-neighbourhood alignment survives
                   null-calibration},
}

@misc{platoscave2026,
  author        = {Anonymous},
  title         = {Back into Plato's Cave: Probing the Limits of the
                   Platonic Representation Hypothesis},
  year          = {2026},
  eprint        = {2604.18572},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.18572},
  note          = {Cross-model alignment is fragile and degrades
                   with model scale},
}

@article{bastos2020routing,
  author  = {Bastos, Andre M. and Lundqvist, Mikael and
             Waite, Ayan S. and Kopell, Nancy and Miller, Earl K.},
  title   = {Layer and Rhythm Specificity for Predictive Routing},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {117},
  number  = {49},
  pages   = {31459--31469},
  year    = {2020},
  doi     = {10.1073/pnas.2014868117},
}

@article{liu2024hybrid,
  author  = {Liu, Faqiang and Zheng, Hao and Ma, Songchen and
             Zhang, Weihao and Liu, Xue and Chua, Yansong and
             Shi, Luping and Zhao, Rong},
  title   = {Advancing brain-inspired computing with hybrid neural
             networks},
  journal = {National Science Review},
  volume  = {11},
  number  = {5},
  pages   = {nwae066},
  year    = {2024},
  month   = may,
  doi     = {10.1093/nsr/nwae066},
  publisher = {Oxford University Press},
}
```

- [ ] Save. No build yet — citations are wired in Tasks 7 and 8.
- [ ] Do not commit yet — Tasks 6, 7, 8 ship together.

---

## Task 7 — dream-of-kiki: NIR/AER/DVNC row + HNN review in §8.4

**Files:**
- `/Users/electron/Documents/Projets/dream-of-kiki/docs/papers/paper1/full-draft.md`
- `/Users/electron/Documents/Projets/dream-of-kiki/docs/papers/paper1/discussion.md`

§8.4 "Comparison with prior art" (line ~986) is a Markdown table; the
last data row before the closing prose is the `@huh2024platonic` row
(line 1005). DVNC, NIR and AER are adjacent prior art for the
substrate-agnosticism (DR-3) and cross-substrate-protocol claims and are
currently absent. Add new table rows. Also add the Liu et al. 2024 HNN
review as the canonical hybrid-neural-network anchor — dream-of-kiki
validates conformance across MLX-dense and LIF-spiking substrates, which
is exactly a hybrid neural network.

Steps:

- [ ] In `full-draft.md`, locate the §8.4 table. Immediately AFTER the
  `@huh2024platonic` row (line 1005, the row ending
  `... @saillant2026nervewml (`nerve-wml` v1.7.0, GammaThetaMultiplexer
  experiment) |`) and BEFORE the blank line that precedes the
  "Our distinguishing features ..." prose, insert the following four
  table rows verbatim (each is one physical line; do not wrap):

```markdown
| @liu2021dvnc (Discrete-Valued Neural Communication, NeurIPS 2021) | VQ-VAE shared global codebook for discrete messages between neural modules | Closest prior art for cross-module discrete communication ; DVNC uses one shared codebook and homogeneous modules, whereas our substrate-portability claim (DR-3) and the companion nerve-wml protocol use per-substrate codebooks with learned transducers and typed prediction/error roles — DVNC is a special case (single substrate, single codebook) of the more general portability our Conformance Criterion targets |
| @pedersen2024nir (Neuromorphic Intermediate Representation, Nature Communications 2024) | Static computational-graph IR : a hardware-agnostic instruction set for compiling one spiking model onto many neuromorphic backends | Operates at the *model-graph* layer ; orthogonal to our framework, which composes *consolidation operations* over a substrate and is agnostic to the IR that lowers any one substrate to hardware — NIR could serve as the compilation target for our E-SNN substrate (§9.1) without altering the axioms |
| @aer2025biohybrid (Address-Event Representation over UDP/Ethernet, arXiv:2501.09128, 2025) | Transport / encoding layer specifying how spike events are timestamped, addressed and physically carried, including biohybrid silicon–wetware links | Operates at the *transport* layer ; orthogonal to our framework — the Conformance Criterion constrains the semantics of consolidation operations, not the wire protocol that moves spikes, and AER is one admissible carrier for a conformant E-SNN substrate |
| @liu2024hybrid (Advancing brain-inspired computing with hybrid neural networks, National Science Review 2024) | Survey arguing hybrid ANN/SNN systems are the path to brain-inspired computing | Canonical hybrid-neural-network anchor for our cross-substrate claim : our cycle-1 validation across an MLX dense substrate and a LIF spiking substrate is precisely a hybrid-neural-network configuration, and the executable Conformance Criterion supplies the contract such hybrids currently lack |
```

- [ ] Mirror the SAME four-row insertion into `discussion.md` if that
  file still carries the §8.4 table. First inspect `discussion.md`:
  if it contains the "Comparison with prior art" table, insert the four
  rows in the matching position; if `discussion.md` does not contain
  §8.4 (it may end at §8.3), skip the mirror and note it in the commit
  body. Do not invent a table in `discussion.md` that is not there.
- [ ] Do not commit yet — proceeds to Task 8.

---

## Task 8 — dream-of-kiki: PRH global-vs-local critique passage

**Files:**
- `/Users/electron/Documents/Projets/dream-of-kiki/docs/papers/paper1/full-draft.md`
- `/Users/electron/Documents/Projets/dream-of-kiki/docs/papers/paper1/discussion.md`

§8.4 closes with a dedicated paragraph **"On the Platonic Representation
Hypothesis as theoretical ground."** (lines 1020–1032). It currently
treats PRH as an unqualified falsifiable anchor for DR-3 and cites only
`@huh2024platonic`. It must engage the two 2026 critiques and
distinguish GLOBAL vs LOCAL alignment, so that DR-3 substrate-agnosticism
is anchored to the *local* form of PRH the 2026 evidence supports.

Steps:

- [ ] In `full-draft.md`, locate the paragraph **"On the Platonic
  Representation Hypothesis as theoretical ground."** ending at line
  1032 with `... is the dreamOfkiki side of the same empirical bet.`
  Insert the following NEW paragraph immediately AFTER that sentence,
  before the `---` on line 1034. Paste verbatim:

```markdown

**The 2026 critiques and a global-versus-local refinement.** The PRH
has since been contested. @platoscave2026 report that cross-model
representational alignment is fragile and degrades as model scale
grows, and @aristotelianprh2026 show that the *global* convergence
originally claimed is substantially a width/depth confounder : once
alignment is null-calibrated against capacity-matched random baselines,
only *local* neighbourhood structure survives as a genuine
cross-substrate signal. We fold this into DR-3 rather than against it.
Our substrate-agnosticism claim does not require that the MLX, E-SNN and
LoRA substrates share one global representation ; it requires only that
a *conformant* substrate preserves the *local* operational structure the
Conformance Criterion checks — that the same consolidation operations,
composed in the same order, produce neighbourhood-equivalent episode
states. This is the local form of PRH, and it is the form the 2026
evidence still supports. The companion nerve-wml probe
[@saillant2026nervewml] is consistent with this reading : its
mutual-kNN measurement is a local-neighbourhood metric reported against
a capacity-matched random baseline, exactly the null-calibrated, local
quantity @aristotelianprh2026 argue is the defensible one. DR-3 is
therefore anchored to local, not global, representational convergence.
```

- [ ] Mirror the SAME paragraph insertion into `discussion.md` if that
  file carries the "On the Platonic Representation Hypothesis as
  theoretical ground." paragraph. Inspect `discussion.md` first; insert
  in the matching position if present, otherwise skip the mirror and
  note it in the commit body.
- [ ] Build the dream-of-kiki paper. From the paper directory:
  ```
  cd /Users/electron/Documents/Projets/dream-of-kiki/docs/papers/paper1 \
    && pandoc full-draft.md -o build/full-draft.tex \
       --bibliography=references.bib --citeproc --standalone
  ```
- [ ] Verify the build: command exits 0. Then check the produced
  `build/full-draft.tex` for unresolved citations — there must be NO
  occurrence of `?` citation markers, NO `Citation not found`, and the
  new keys must appear resolved. Run:
  ```
  grep -c -E 'liu2021dvnc|pedersen2024nir|aer2025biohybrid|platoscave2026|aristotelianprh2026|bastos2020routing|liu2024hybrid' build/full-draft.tex
  ```
  and confirm the result is non-zero. Also confirm pandoc printed no
  `[WARNING] Citeproc: citation X not found` lines to stderr for any of
  the seven new keys.
- [ ] Optionally also render PDF if the toolchain is present
  (`pandoc full-draft.md -o build/full-draft.pdf --bibliography=references.bib
  --citeproc --standalone`); the `.tex` build is the load-bearing
  verification, the PDF is a courtesy check.
- [ ] Commit (Tasks 6, 7, 8 in one commit):
  - subject: `docs(paper1): cite adjacent art and PRH critiques`
  - body (≤72-char lines): seven BibTeX entries added (DVNC, NIR, AER,
    two 2026 PRH critiques, Bastos 2020, Liu 2024 HNN review); §8.4
    table gains four prior-art rows; new §8.4 passage distinguishes
    global vs local PRH alignment and anchors DR-3 to the local form;
    note whether `discussion.md` was mirrored or skipped; note that the
    EN→FR `paper1-fr/` mirror is a required follow-up PR (out of scope).

---

## Self-Review

Reviewed the plan against the writing-plans skill and the task brief:

- **Header & required sections** — present: H1 title, "For agentic
  workers" line, Goal, Architecture, Tech Stack, `---`, then tasks,
  then this Self-Review and the Execution Handoff. ✅
- **Prose adaptation** — no fake TDD cycles. Each task's cycle is
  locate → paste BibTeX → paste prose → rebuild → verify citation
  resolves → commit. ✅
- **No placeholders** — every BibTeX entry and every prose paragraph is
  written in full, paste-ready English. ✅
- **All three CRITICAL gaps covered** — (1) DVNC/NIR/AER cited &
  differentiated: nerve-wml Tasks 1–2, dream-of-kiki Tasks 6–7; (2) PRH
  global-vs-local + 2026 critiques: nerve-wml Tasks 3+5, dream-of-kiki
  Tasks 6+8; (3) GTM re-grounded on predictive routing: nerve-wml
  Tasks 3–4. ✅
- **Manuscript facts verified by reading the sources** — nerve-wml
  `refs.bib` ALREADY contains `liu2021dvnc` and `aristotelianprh2026`
  (the plan does not re-add them — a duplicate-key risk avoided);
  nerve-wml §Related Work already says "five threads" in its closing
  sentence so changing "four"→"five" in the opener makes the section
  self-consistent; dream-of-kiki `full-draft.md` is manually assembled
  with no generation script, so the plan edits it directly and mirrors
  into live section files where they still carry the section. ✅
- **Task count** — 8 tasks, grouped nerve-wml (1–5) then dream-of-kiki
  (6–8); within the brief's 7–10 range. ✅
- **Commit hygiene** — three commits total (nerve-wml ×2,
  dream-of-kiki ×1), all subjects ≤50 chars, scopes use no underscore
  (`paper1`), bodies ≤72-char lines, English. ✅
- **Known carryover / flagged risk** — (a) the dream-of-kiki EN→FR
  propagation rule means `paper1-fr/` needs the same edits in a
  follow-up PR; flagged in Task 8 commit body and here, deliberately
  out of scope. (b) Several BibTeX entries for not-yet-published 2025/
  2026 preprints use `Anonymous`/`others` authors and the arXiv IDs
  given in the brief; the executor should treat them like the existing
  `% UNVERIFIED PRIMARY` entries in dream-of-kiki's `references.bib`
  and confirm author lists against the PDFs before final submission —
  this does not block the build or the plan.

Result: **plan is internally consistent, grounded in the actual
manuscript sources, and ready to execute.**

## Execution Handoff

**Recommended mode: Inline (single session).**

This is an 8-task prose plan across two manuscripts with two hard
sequential constraints that make parallel subagent dispatch
counter-productive:

- Within nerve-wml, Tasks 1→2 and Tasks 3→4→5 each touch the same two
  files (`refs.bib`, `main.tex`) and share a single LaTeX build; running
  them in parallel would cause merge conflicts on `main.tex` and
  redundant multi-pass builds.
- Within dream-of-kiki, Tasks 6→7→8 all touch `full-draft.md` /
  `references.bib` and share one pandoc build.
- The two manuscripts ARE independent of each other, but the total
  volume is small (three commits, two builds), so the coordination
  overhead of a subagent split is not worth it.

Execute Tasks 1–5 (nerve-wml, two commits), then Tasks 6–8
(dream-of-kiki, one commit), each task's verify step gating the next.
If a build fails, fix the BibTeX/prose in place before proceeding —
do not commit a non-building paper. After both papers build clean,
a `superpowers:verification-before-completion` pass should re-run both
build commands once more from a clean state and confirm zero undefined
citations before the work is reported complete.
