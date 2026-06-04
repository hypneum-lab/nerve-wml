"""Sensitivity sweeps across four hyperparameter axes.

Each axis varies one parameter while holding others at their default;
multi-seed mean and std are reported per parameter value via
``scripts.multi_seed.run_multi_seed``.

Axes:
    k_cknna       in {5, 10, 20}          on scale_robustness substrate pair
    lambda_cycle  in {1.0, 10.0, 100.0}   on transducer Vec2Vec arm
    n_anchors     in {8, 16, 32, 64}      on transducer RelativeRep arm
    steps         in {500, 2000, 8000}    on transducer learned arm
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nerve_wml.methodology.hsic_cknna import cknna
from scripts.multi_seed import run_multi_seed
from scripts.scale_robustness_pilot import _substrate_embeddings
from scripts.transducer_baselines_pilot import (
    _build_task,
    _mi_entropy_bits,
    _train_learned,
)
from track_p.transducer_baselines import (
    RelativeRepTransducer,
    Vec2VecTransducer,
)


def _cknna_runner(*, seed: int, k: int) -> dict[str, dict[str, float]]:
    mlp, lif = _substrate_embeddings(seed, 256)
    return {"cknna_k": {"value": float(
        cknna(mlp.cpu().numpy(), lif.cpu().numpy(), k=k)
    )}}


def _vec2vec_runner(
    *, seed: int, lambda_cycle: float
) -> dict[str, dict[str, float]]:
    src_codes, dst_codes, src_wml, dst_wml = _build_task(seed)
    src_cb = src_wml.codebook.detach()
    dst_cb = dst_wml.codebook.detach()
    v2v = Vec2VecTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb,
        lambda_cycle=lambda_cycle, seed=seed,
    )
    v2v.fit(steps=500)
    pred = v2v.forward(src_codes)
    row = _mi_entropy_bits(pred, dst_codes)
    return {"vec2vec": {"mi_bits": float(row["mi_bits"])}}


def _relrep_runner(
    *, seed: int, n_anchors: int
) -> dict[str, dict[str, float]]:
    src_codes, dst_codes, src_wml, dst_wml = _build_task(seed)
    src_cb = src_wml.codebook.detach()
    dst_cb = dst_wml.codebook.detach()
    rel = RelativeRepTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb,
        n_anchors=n_anchors, seed=seed,
    )
    pred = rel.forward(src_codes)
    row = _mi_entropy_bits(pred, dst_codes)
    return {"relrep": {"mi_bits": float(row["mi_bits"])}}


def _steps_runner(*, seed: int, steps: int) -> dict[str, dict[str, float]]:
    src_codes, dst_codes, *_ = _build_task(seed)
    learned = _train_learned(src_codes, dst_codes, steps)
    pred = learned.forward(src_codes, hard=True)
    row = _mi_entropy_bits(pred, dst_codes)
    return {"learned": {"mi_bits": float(row["mi_bits"])}}


def _sweep_axis(
    runner: Any,
    *,
    seeds: Sequence[int],
    param: str,
    values: Sequence[Any],
    method: str,
    metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for v in values:
        agg = run_multi_seed(runner, seeds=seeds, **{param: v})
        leaf = agg[method][metric]
        rows.append({
            "param": param,
            "value": v,
            "mean": leaf["mean"],
            "std": leaf["std"],
            "values": leaf["values"],
        })
    return rows


def run_sensitivity_sweeps(
    *, seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, list[dict[str, Any]]]:
    """Vary one hyperparameter at a time, return multi-seed stats."""
    return {
        "k_cknna": _sweep_axis(
            _cknna_runner, seeds=seeds, param="k",
            values=(5, 10, 20),
            method="cknna_k", metric="value",
        ),
        "lambda_cycle": _sweep_axis(
            _vec2vec_runner, seeds=seeds, param="lambda_cycle",
            values=(1.0, 10.0, 100.0),
            method="vec2vec", metric="mi_bits",
        ),
        "n_anchors": _sweep_axis(
            _relrep_runner, seeds=seeds, param="n_anchors",
            values=(8, 16, 32, 64),
            method="relrep", metric="mi_bits",
        ),
        "steps": _sweep_axis(
            _steps_runner, seeds=seeds, param="steps",
            values=(500, 2000, 8000),
            method="learned", metric="mi_bits",
        ),
    }


def main() -> None:  # pragma: no cover
    out = run_sensitivity_sweeps()
    for axis, rows in out.items():
        print(f"\n=== axis: {axis} ===")
        for row in rows:
            print(
                f"  {row['param']}={row['value']!r:<8}  "
                f"mean={row['mean']:.4f}  std={row['std']:.4f}"
            )


if __name__ == "__main__":  # pragma: no cover
    main()
