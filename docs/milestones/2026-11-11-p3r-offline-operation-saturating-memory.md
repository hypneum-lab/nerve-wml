# Pre-registration (L0, draft) — P3r: which offline operation suits a saturating one-way memory

**Status:** L0 draft written on 2026-11-11 at the request of the PI,
as the reformulation of candidate P3; revised the same evening on the
PI's decisions: **P3r-b is the only hypothesis; P3r-a becomes the
positive control of the instrument; P3r-c is dropped.** The reading
gate applies (Parisi 1986 and Tononi-Cirelli 2003 are paywalled and
wait for a PI decision on access; the 2014 review is read); a CPU run
may happen before M3, results read after the method note. Nothing is
frozen, nothing has been run, no catalogue number is assigned.
Drafted by an agent.

## 1. Question, in plain words

P3 said: in a one-way memory whose synapses saturate, a global
rescaling between learning sessions restores capacity, and noise of
the same size does not. The readings of 2026-11-11 leave it without
biological support (library notes on Bidel 2023 and Pophale 2023),
and half of it is true by construction: shrinking full synapses makes
room.

What is left to ask is what happens to **old memories**, and it
depends on two properties of the circuit:

- whether the read-out adjusts to the total synaptic weight (a
  threshold that follows, or an inhibition proportional to total
  input, the role Bidel et al. suggest for the complex amacrine
  cells);
- whether a synapse is graded or a two-state switch (the octopus
  plasticity is described as a maintained switch). A switch cannot be
  multiplied by 0.8; the only equivalent is to turn a random fraction
  off.

## 2. What is known (reading note, to complete before any freeze)

- Expected by a signal-to-noise argument, before any simulation: a
  proportional rescaling multiplies signal and interference by the
  same factor. With a read-out that follows, old memories are
  untouched; with a fixed threshold, the weakest ones fall below it.
  The first statement below is close to an analytical fact; the
  simulation checks it, it does not discover it.
- Prior art. **Read**: Tononi and Cirelli 2014 (library note of
  2026-11-04, `references/notes/doi-10.1016_j.neuron.2013.12.025.md`):
  SHY states renormalisation by **down-selection**, with three
  computational variants (proportional reduction with a threshold;
  depression-biased STDP sparing strong synapses; protection of
  co-activated synapses); the « rescale » arm below is the first
  variant, the « random reset » arm is none of them. **Not read,
  paywalled** (listed in `library/references/to-fetch/`): Parisi 1986,
  « A memory which forgets » (J. Phys. A 19, L617), the palimpsest with
  bounded synapses; Tononi and Cirelli 2003, the first statement of
  SHY. Not resolved: Amit and Fusi on bounded synapses.
- **Read in full on 2026-11-11 (evening)**: Lepre 2026, « Thermodynamics
  of Forgetting in a Bounded Hopfield Network », arXiv:2611.08258
  (library note). Facts: the Parisi palimpsest (hard clipping at +/- A,
  increments 1/sqrt(N)) has its memory horizon maximised at A = 0.351
  with x_c = 0.0405 N patterns at T = 0 (pp. 20-21; asymptotic, the
  N = 1 000 value is lower), falling by a factor ten between T = 0 and
  T = 0.6; the optimum bound 0.041 at A = 0.354 is due to van Hemmen,
  Keller and Kühn 1988, not to Parisi 1986 (p. 21). The paper contains
  **no offline operation**, no dilution, no rescaling: it does not
  decide P3r-b. Its signal-to-noise decomposition (remark 4, p. 18:
  age enters the signal only, the noise is the same for every memory)
  gives the reader's expectation for the Hopfield analogue: a
  proportional reduction J -> (1 - eps) J is exactly neutral at T = 0
  and a heating T -> T / (1 - eps) at T > 0 (the counter-evidence of
  dream-of-kiki PR #100 again); a random reset of a fraction eps adds
  a Sherrington-Kirkpatrick noise term of variance eps (1 - eps) A^2 / 3,
  about 0.053 A^2 at eps = 0.2, of the same order as the clipping noise
  0.052 A^2. **Reader's inference:** the sign of P3r-b is expected in
  the recurrent analogue, not its size; the lab's model (non-negative
  sparse weights in [0, 1], zero as a floor) is not the paper's, so
  the analytic estimate required by section 4 still has to be done on
  the lab's model. The lab already
  vendors a model with bounded synapses and dreaming
  (`experiments/hopfield_dream_conf/vendor/hopfield.py`, Marinari,
  Rossi, Zamponi), which is a recurrent relative of this question.
  The novelty of P3r is therefore expected to be low.
- Counter-evidence inside the lab: in the Hopfield control of
  dream-of-kiki PR #100 (section 6 ter), a uniform reduction at
  finite temperature acts as a heating and destroys retrieval, while
  noise of the same norm does not.

## 3. Design

- **Memory.** One layer, feedforward. 1 000 binary inputs with 5 %
  active per pattern; 50 output units; non-negative weights bounded
  in [0, 1]. An association (pattern, output unit) is learnt by
  potentiating the synapses from the active inputs to that unit.
- **Two synapse types.** Graded: `w += q (1 - w)`. Binary: `w` set to
  1 with probability q.
- **Two read-outs.** Fixed threshold, set once on the first session.
  Proportional inhibition: the unit fires when its input exceeds
  `g * mean(w) * (number of active inputs)`, g set once.
- **Sessions.** Eight sessions of K associations, K chosen so that
  the no-operation arm saturates by session 4 [to calibrate on the
  graded, fixed-threshold arm only, and to write down before the
  run].
- **Three operations between sessions**, matched on the total weight
  removed (fraction eps = 0.2): none; **rescale** (`w *= 1 - eps`,
  graded only); **random reset** (each synapse set to 0 with
  probability eps).
- **Measures after the last session.** `new`: recognition accuracy
  for the associations of the last session against novel patterns;
  `old`: the same for the first session; both as balanced accuracy,
  16 seeds.

## 4. Statements and decision rule (to fix before any run; revised 2026-11-11 evening)

Equivalence and difference are read on 99 % bootstrap intervals over
seeds, margin 0.05 of balanced accuracy.

- **Gate 0 (saturation).** In the no-operation, graded, fixed-threshold
  arm, `new` at the last session is below `new` at the first by more
  than the margin. Otherwise the memory did not saturate and the
  verdict is `non_test`.
- **Gate 1 (positive control of the instrument, former P3r-a), graded
  synapses.** Expected and required: with proportional inhibition,
  `old` under rescale is equivalent to `old` under no operation while
  `new` is higher by more than the margin; with a fixed threshold,
  `old` under rescale is lower than under no operation by more than
  the margin. Both halves follow from the signal-to-noise argument of
  section 2; a simulator that does not show them is not trusted, and
  the verdict is `non_test`. No hypothesis is attached.
- **P3r-b (the only hypothesis), graded synapses with proportional
  inhibition.** At equal total weight removed, `old` is higher under
  rescale than under random reset by more than the margin, with `new`
  equivalent. Two-sided: supported, refuted (reset at least as good on
  `old`, or `new` not equivalent), or inconclusive.
- Dropped (PI, 2026-11-11): the descriptive P3r-c on switch synapses.
  The binary-synapse code path is not written.

**Tautology check (2026-11-11).** P3r-b is closer to a calculation
than the draft of this morning said. With proportional inhibition a
rescale leaves every relative weight unchanged, so `old` is untouched
by construction; a random reset of a fraction eps of the synapses
multiplies each memory's signal and interference by (1 - eps) **in
expectation** and the inhibition follows, so the only loss on `old`
comes from the **binomial variance** of the deletion on the weakest
memories. The size of that term is computable before any run (number
of active inputs, K, eps, read-out margin). **Required before a
freeze:** that analytic estimate, written in this section; if it falls
under the 0.05 margin, P3r-b is a null by construction and goes back
to a note (PI decision). The `new` half is the open part: rescale frees
room in every synapse, reset frees full room in a fraction eps of them
and none elsewhere.

## 5. What each verdict changes

- **Gate 1 passes:** the instrument reproduces what the signal-to-noise
  argument says (free where the read-out is normalised, harmful where
  it is not); this is reported as a control, not as a result.
- **P3r-b supported:** one constructed case where a rescaling beats a
  random reset; refuted: the motif has no advantage over forgetting
  even in the case built for it, and should leave the lab's
  biological analogies.

## 6. Known weaknesses of this draft

- P3r-a was nearly a calculation and is now the positive control;
  P3r-b may be one too at second order (tautology check, section 4).
- The model is a toy; it claims nothing about the octopus. The
  mapping of "proportional inhibition" to the complex amacrine cells
  is the authors' own hypothesis in Bidel et al., not a finding.
- The two read-outs differ by one free constant each; they are set on
  session 1 and never retuned, which favours neither by design but
  is a choice.
- Matching operations on total weight removed is one convention;
  matching on the norm of the change is another and can reverse a
  close result.
- The prior art is unread. This draft may restate a known result.

## 7. Cost, and what is not done

CPU only, numpy, minutes. Not done: the code; the calibration of K;
the analytic estimate of the P3r-b effect size (section 4); the
reading of Parisi 1986 and Tononi-Cirelli 2003 (paywalled, PI
decision on access). Decided by the PI on 2026-11-11 (evening): P3r-b
only, P3r-a as control, P3r-c dropped; reading gate applies; CPU run
allowed before M3, results read after the method note.
