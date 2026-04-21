# Sleep-EDF EEG pipeline protocol (Voie 3 Semaines 3-5)

**Status:** in-progress (2026-04-21 evening, download launched).
**Pivot rationale:** CHB-MIT requires PhysioNet registration +
usage agreement before download (humans-in-loop, ~5-10 min).
Sleep-EDF is open access via `mne.datasets.sleep_physionet`,
~8 GB total, canonical EEG ML benchmark, supported natively by
MNE-Python. Smaller and more accessible -> faster iteration.

## Dataset

Sleep-EDF Database Expanded (PhysioNet), age subset:
- 78 subjects, 153 nightly recordings
- 2 EEG channels (Fpz-Cz, Pz-Oz), EOG, EMG, event marker
- Recordings labelled with 5 sleep stages (W, N1, N2, N3, REM)
  + Movement
- Standard ML task: 5-class sleep stage classification on
  30-second epochs.

Download (current run on kxkm-ai background):
```python
mne.datasets.sleep_physionet.age.fetch_data(
    subjects=list(range(10)),  # 10 subjects, ~1 GB
    recording=[1],             # one night per subject
)
```

Returns list of (psg_edf_path, hypnogram_edf_path) tuples.

## Pipeline

### Stage 1 -- preprocess (Sleep-EDF specific)

`scripts/eeg_preprocess_sleep_edf.py`:

1. Load each (psg, hypnogram) EDF pair via `mne.io.read_raw_edf`.
2. Pick 2 EEG channels (Fpz-Cz, Pz-Oz) -- drop EOG/EMG/marker
   for first iteration. Bandpass 0.5-30 Hz, downsample to 100 Hz.
3. Read hypnogram annotations, map sleep stages
   (W=0, N1=1, N2=2, N3=3, REM=4), drop Movement.
4. Segment into 30-second non-overlapping epochs (3000 samples
   per epoch at 100 Hz). Each epoch keeps shape `[n_channels=2,
   n_samples=3000]`. Label is the dominant stage in the epoch.
5. Patient-wise train/val/test split (70/15/15) -- no temporal
   leakage between sets within a patient.
6. Save NPZ artefact: `tests/golden/sleep_edf_epochs.npz`
   - `x_train: float32[N_train, 2, 3000]`
   - `y_train: int64[N_train]`  (5-class)
   - `x_val, y_val, x_test, y_test` similarly.

Estimated output size: ~10 MB per subject -> 100 MB for 10
subjects. Fits under git's blob limit; can commit as
reproducibility artefact.

### Stage 2 -- save codes (analogue to save_codes_for_checks.py)

`scripts/save_codes_eeg.py`:

1. Load `sleep_edf_epochs.npz`.
2. Flatten epochs to `[N, 2*3000=6000]` or use
   `MlpWML.from_spectrogram(...)` factory (v1.5.0 feature, perfect
   reuse here -- the factory is exactly designed for this).
3. Train MLP and LIF substrates analogously to
   scripts/save_codes_for_checks.py but on Sleep-EDF instead of
   HardFlowProxyTask. Match all hyperparameters: d_hidden=16,
   alphabet=64, lr=1e-2, batch=64.
4. Eval-time: extract argmax codes + pre-VQ continuous embeddings
   for both substrates on x_test.
5. Save NPZ: `tests/golden/codes_mlp_lif_eeg.npz` with same
   schema as `codes_mlp_lif.npz`.

Estimated training: ~5-10 min per substrate per seed on GPU
(5000 epochs of 6000-dim input -> 16-dim hidden -> 64-code
alphabet -> 5-class). On kxkm-ai 4090 trivially fast.

### Stage 3 -- measurements (reuse existing scripts)

The existing `scripts/measure_*.py` consume the NPZ schema:

```bash
uv run python scripts/measure_mi_null_model.py \
    --codes tests/golden/codes_mlp_lif_eeg.npz \
    --shuffles 1000

uv run python scripts/measure_mi_bootstrap_ci.py \
    --codes tests/golden/codes_mlp_lif_eeg.npz \
    --resamples 1000

uv run python scripts/measure_mi_multi_estimator.py \
    --codes tests/golden/codes_mlp_lif_eeg.npz

uv run python scripts/measure_mi_mine.py \
    --codes tests/golden/codes_mlp_lif_eeg.npz
```

Output JSONs to `papers/paper1/figures/mi_*_eeg.json`.

### Stage 4 -- paper integration (Test (9) ?)

New paper section in §Information Transmission: Test (9) "Real
neural data validation (Sleep-EDF EEG)". Reports the same 4
estimators on EEG and compares to HardFlowProxyTask numbers.

Key narrative questions:
- Does the substrate-equivalent task competence (Claim A)
  generalize from synthetic XOR-on-noise to real EEG?
- Does the substrate-agnostic information transmission
  (Claim B) hold on neural recordings?
- Does the VQ near-lossless compression observation (Test 7)
  transfer?

Failure modes worth flagging upfront:
- Both substrates may fail to converge on EEG due to
  capacity/architecture mismatch -> document honestly,
  Claim A would need refinement.
- Plug-in MI/H may collapse due to alphabet under-utilisation
  on a 5-class task -> consider alphabet=8 instead of 64.

## Compute distribution

- **Local laptop (GrosMac)**: scripts editing only. NO compute
  on EEG data per the user's "rien sur grosmac" rule.
- **kxkm-ai**: download (current background), preprocess pipeline
  CPU, MLP/LIF training (CPU OK, GPU bonus).
- **Tower**: alternative for redundancy if kxkm-ai busy.

## Schedule (revised vs original CHB-MIT plan)

| Day | Action | Compute |
|---|---|---|
| S3 d1 | Download finishes (background, started 2026-04-21 evening) | kxkm-ai I/O |
| S3 d2 | Preprocess pipeline + go/no-go (MLP baseline acc > 0.6 ?) | kxkm-ai CPU |
| S3 d3 | save_codes_eeg.py 3 seeds | kxkm-ai CPU |
| S4 d1 | Run 4 measurement scripts on EEG NPZ | kxkm-ai CPU |
| S4 d2 | Fetch JSONs back, write paper Test (9) draft | local |
| S4 d3 | Compile paper, commit, propose v1.6.0 | local |

Total: ~1 week (vs 3 weeks original CHB-MIT estimate).

## Sister bouba_sens reference

bouba_sens v0.5.0 already imports `nerve_wml.methodology` for
its B-1/B-2/B-3 invariants. The Sleep-EDF pipeline naturally
becomes a candidate "real-data" check for ADR-0007 paired-run
eval -- mention in the paper integration step that the same
methodology is now exercised on synthetic, MIT-BIH ECG (via
v1.5.0 SpectrogramEncoder), AND EEG. Three real datasets,
one shared methodology.

## Cross-refs

- `scripts/eeg_preprocess_sleep_edf.py` -- skeleton prepared 2026-04-21
- `scripts/save_codes_eeg.py` -- skeleton prepared 2026-04-21
- `tests/golden/sleep_edf_epochs.npz` -- target artefact (Stage 1 output)
- `tests/golden/codes_mlp_lif_eeg.npz` -- target artefact (Stage 2 output)
- `papers/paper1/figures/mi_*_eeg.json` -- target measurement JSONs
- `docs/research-notes/dvnc-baseline-protocol.md` -- sibling 3-day protocol
