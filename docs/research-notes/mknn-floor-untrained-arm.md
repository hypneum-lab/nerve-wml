# Test (6) mutual-kNN: analytic floor and untrained arm (issue #110)

Date: 2026-11-06. Status: code and method note only. No published
number of Paper 1 is recomputed here, and `papers/paper1/main.tex`
is not edited.

## 1. Status of Test (6)

Test (6) is labelled *post-hoc* in Paper 1 (`papers/paper1/main.tex:423`,
"Post-hoc validation via mutual-kNN alignment (v1.5.2)"). No
pre-registration defines it (under `docs/prereg/`, mKNN appears only in
the N10-D reading notes `docs/prereg/n10d/reading-note.md` and
`snowball-tour1.tsv`) and no `prereg-*` tag covers it. Its decision rule lives in code only:
`alignment > max(3 x random_baseline, 0.05)`
(`scripts/platonic_rh_alignment.py`, docstring and `main`). This note
leaves that rule unchanged.

## 2. Kernel as coded

`mutual_knn` (`scripts/platonic_rh_alignment.py:48-66`): rows are
L2-normalised, self-similarity is masked, the k nearest neighbours of
each row are taken in A and in B under cosine similarity, and the score
is the mean over rows of `|N_A(i) ∩ N_B(i)| / k`. It is a per-row
(not symmetrised) overlap, normalised by k, in [0, 1].

## 3. Analytic floor

If A and B are independent and the rows of B are exchangeable, then
for each i the set N_B(i) is a uniform k-subset of the n-1 other
points, independent of N_A(i). The overlap is hypergeometric with mean
k·k/(n-1). Divided by k, the expected score is

    E[mKNN] = k / (n - 1)

(Groger, Wen, Brbic 2026, arXiv:2602.14486, Prop. 4.2, Eq. (4), p. 4).
Ties have probability zero for continuous embeddings.

Checks run on 2026-11-06:
- exact hypergeometric mean via `math.comb` at n = 1024 equals k/1023
  for k ∈ {5, 10, 20, 50} (0.004888, 0.009775, 0.019550, 0.048876);
  the table's "chance" k/N differs only in the fourth significant digit;
- pure-Python Monte Carlo, 20 i.i.d. Gaussian pairs, n = 256, k = 16,
  d = 16: mean 0.06215 vs floor 0.06275 (relative error 0.95 %);
- unit test `tests/unit/test_mknn_floor.py` (same size, torch kernel).

`mutual_knn_floor` (`scripts/platonic_rh_alignment.py:69`) is now
reported in the JSON next to `chance`, with a `mlp_lif_x_floor` ratio.
It is a reference, not a threshold.

What the floor can and cannot do. It can fail as a check of the
kernel: an independent-Gaussian score far from k/(n-1) would reveal an
implementation error. As a comparison for the trained MLP–LIF score it
is **non informatif**: any two representations computed from the same
inputs share some neighbourhood structure and are expected to exceed
the independence floor, so exceeding it says nothing about learned
convergence.

## 4. Untrained-network arm

`untrained_arm` (`scripts/platonic_rh_alignment.py:127`), enabled by
`--untrained-arm`: MLP and LIF are built by the same `_train_mlp` /
`_train_lif` functions with `steps=0`, so weights equal the seeded
initial weights of the trained arm (asserted for the MLP in
`tests/unit/test_mknn_untrained_arm.py`). The held-out batch comes
from a fresh `HardFlowProxyTask` with the same seed, whose sampler uses
its own generator, so the batch is identical to the trained arm's.
Embeddings go through the shared `_embed` helper (line 124).

Fact: the LIF embedding used by Test (6) is
`lif.input_proj(lif_encoder(x))` (line 124), two `Linear` layers with
no spiking step, i.e. an affine function of x. The MLP embedding is
`mlp.core(x)`, four `Linear` layers with ReLU
(`track_w/mlp_wml.py:60-74`).

Inference: both arms inherit neighbourhood structure from the shared
input x, and the LIF side remains affine in x after training.

What this control can make fail. If the trained MLP–LIF score is not
clearly above the untrained score at the same seeds, N, d and k, the
reading "alignment produced by training on the task" fails, and the
18.8× ratio against independent Gaussians would be attributable to
shared input geometry. The control is therefore informative.

What it cannot test. A trained score above the untrained one does not
show substrate-independent or Platonic convergence: it is also what a
shared task signal pulled through two different function classes
would give. It does not involve the spiking dynamics (see the fact
above), it has no pre-registered threshold, and with three seeds it is
descriptive. A same-N, same-d permutation null and an input-space
baseline (mKNN of x against each embedding) remain separate controls.

## 5. Not done here

No run of the script on its default configuration, no new number for
Table `tab:platonic-knn`, no change to Paper 1 text. Reporting both
references in the Results, and any wording change, is a PI decision.
