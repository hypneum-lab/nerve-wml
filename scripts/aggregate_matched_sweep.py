"""Aggregate matched-capacity scale sweep on Sleep-EDF EEG."""
import json
from pathlib import Path
import numpy as np
from nerve_wml.methodology import (
    mi_plugin_discrete, null_model_mi, entropy_discrete,
)

results = []
for d in [16, 32, 64, 128, 256]:
    npz_path = Path(f"tests/golden/codes_mlp_lif_eeg_matched_d{d}.npz")
    if not npz_path.exists():
        print(f"MISSING: {npz_path}")
        continue
    data = np.load(npz_path)
    cm = data["mlp_codes"]
    cl = data["lif_codes"]
    per_seed = []
    for i in range(3):
        a = cm[i].astype(np.int64)
        b = cl[i].astype(np.int64)
        mi = mi_plugin_discrete(a, b)
        nm = null_model_mi(a, b, n_shuffles=500, seed=i)
        ha = entropy_discrete(a)
        per_seed.append({
            "seed":       i,
            "mi_plugin":  mi,
            "null_z":     nm.z_score,
            "h_a":        ha,
        })
    mi_mean = float(np.mean([r["mi_plugin"] for r in per_seed]))
    z_mean = float(np.mean([r["null_z"] for r in per_seed]))

    acc_mlp = float(np.mean(data["acc_mlp"])) if "acc_mlp" in data else None
    acc_lif = float(np.mean(data["acc_lif"])) if "acc_lif" in data else None
    gap = abs(acc_mlp - acc_lif) if acc_mlp is not None else None
    alph_mlp = int(np.mean([len(np.unique(cm[i])) for i in range(3)]))
    alph_lif = int(np.mean([len(np.unique(cl[i])) for i in range(3)]))

    results.append({
        "d_hidden":  d,
        "mi_plugin_mean": mi_mean,
        "null_z_mean":    z_mean,
        "acc_mlp_mean":   acc_mlp,
        "acc_lif_mean":   acc_lif,
        "gap":            gap,
        "alphabet_mlp":   alph_mlp,
        "alphabet_lif":   alph_lif,
    })

    print(
        f"d={d:4d}: MI/H={mi_mean:.4f}  z={z_mean:7.1f}  "
        f"MLP={acc_mlp:.4f}  LIF={acc_lif:.4f}  gap={gap:.4f}  "
        f"alph={alph_mlp}/{alph_lif}"
    )

out_path = Path("papers/paper1/figures/eeg_matched_scale_sweep.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({"per_d_hidden": results}, indent=2))
print()
print(f"saved {out_path}")
