# Pre-registration (L0, draft) — P3r: which offline operation suits a saturating one-way memory

**Status:** L0 draft written on 2026-11-11 at the request of the PI,
as the reformulation of candidate P3. Nothing is frozen, nothing has
been run, no catalogue number is assigned. Drafted by an agent.

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
- Prior art, **not read**: memories with bounded synapses that forget
  (Parisi 1986, listed in the library as to fetch; Amit and Fusi);
  synaptic-scaling models; the synaptic homeostasis hypothesis of
  sleep, cited from memory in the working document. The lab already
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

## 4. Statements and decision rule (to fix before any run)

Equivalence and difference are read on 99 % bootstrap intervals over
seeds, margin 0.05 of balanced accuracy.

- **P3r-a (interaction with the read-out), graded synapses.** With
  proportional inhibition, `old` under rescale is equivalent to `old`
  under no operation while `new` is higher by more than the margin.
  With a fixed threshold, `old` under rescale is lower than under no
  operation by more than the margin. Supported iff both hold;
  refuted iff rescale preserves `old` with a fixed threshold or loses
  it with inhibition.
- **P3r-b (the claim of the sleep hypothesis), graded synapses with
  inhibition.** At equal weight removed, `old` is higher under
  rescale than under random reset by more than the margin, with `new`
  equivalent. Two-sided: supported, refuted (reset at least as good),
  or inconclusive.
- **P3r-c (switch synapses), descriptive.** With binary synapses only
  the random reset is defined. Reported: whether `old` decays with
  the number of sessions since learning (a palimpsest), with and
  without inhibition. No verdict.
- **Gate.** In the no-operation, graded, fixed-threshold arm, `new`
  at the last session is below `new` at the first by more than the
  margin. Otherwise the memory did not saturate and every verdict is
  `non_test`.

## 5. What each verdict changes

- **P3r-a supported:** the downscaling motif of the dream axis gets a
  statement of scope: free where the read-out is normalised, harmful
  where it is not. It is the associative-memory counterpart of C1.
- **P3r-b supported:** one constructed case where a rescaling beats a
  random reset; refuted: the motif has no advantage over forgetting
  even in the case built for it, and should leave the lab's
  biological analogies.
- **P3r-c:** tells whether "sleep as rescaling" can be said at all of
  a switch-like plasticity, or only "sleep as random forgetting".

## 6. Known weaknesses of this draft

- P3r-a is nearly a calculation. If the PI wants only results that
  could surprise, P3r-b is the one test worth running.
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
the reading of the references of section 2; a decision of the PI on
whether this is worth a run or stays a note.
