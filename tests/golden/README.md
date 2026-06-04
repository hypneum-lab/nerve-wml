# Golden test fixtures

Large fixtures (>100 MB) are not committed to git — GitHub rejects them.

## Sleep-EDF epochs (`sleep_edf_epochs.npz`, 312 MB)

Pre-extracted Sleep-EDF Database epochs used by the reproducibility
test suite. Local-only; regenerate via:

```bash
python scripts/build_sleep_edf_golden.py --out tests/golden/sleep_edf_epochs.npz
```

Or pull from the project Zenodo archive (DOI 10.5281/zenodo.19656342)
when the fixture is published as a versioned release artefact.
