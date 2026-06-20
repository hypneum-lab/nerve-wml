"""Culminating cross-substrate transmission experiment: does the paper's
transmission claim survive the FULL real protocol (trained per-edge
``track_p.transducer.Transducer`` + GTM γ/θ channel)?

The paper's transmission numbers (``scripts/measure_info_transmission.py``,
Test 2 round-trip / Test 3 cross-merge) are measured on an IDEALISED path:
two task-supervised ``nn.Linear`` transducers operating on CONTINUOUS code
logit vectors ``[B, alphabet=64]``, with NO discretisation to an integer
code and NO pass through the γ/θ GTM channel. The deployable Nerve Protocol
(``track_p.protocol_nerve.ProtocolNerve``) instead transports a SINGLE
discrete integer code per symbol through the learnable code→code
``track_p.transducer.Transducer`` and then the GTM PAC channel.

This script measures the same round-trip task-fidelity metric the paper
reports, through three paths:

  (a) idealised  — the paper's exact ``run_test_2_round_trip_fidelity`` path
                   (continuous logit transducers, no GTM). Reproduced
                   verbatim from ``measure_info_transmission`` for sanity.
  (b) untrained  — discrete code → UNTRAINED ``Transducer`` → GTM → LIF.
  (c) trained    — discrete code → TRAINED ``Transducer`` → GTM → LIF.

Metric (identical to the paper, Test 2):
    fidelity_ratio = acc_roundtrip / acc_direct
where ``acc_direct`` is the frozen MLP's own task accuracy and
``acc_roundtrip`` is the task accuracy recovered at the LIF end after the
cross-substrate pass. A ratio near 1.0 means the substrate-agnostic
transmission claim is load-bearing through that path.

Run on Tower (CPU is fine — models are tiny):
    python -m scripts.transmission_full_protocol --seeds 20 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: N812, E402

from scripts.measure_info_transmission import _train_pair  # noqa: E402
from track_p.multiplexer import GammaThetaMultiplexer  # noqa: E402
from track_p.transducer import Transducer, TransducerGating  # noqa: E402
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask  # noqa: E402

# ---------------------------------------------------------------------------
# Path (a): the paper's idealised round-trip (verbatim metric).
# ---------------------------------------------------------------------------


def path_a_idealised(
    mlp, lif, input_encoder, seed: int, *, steps_transducer: int, batch: int
) -> dict[str, float]:
    """Reproduce ``run_test_2_round_trip_fidelity`` for one frozen pair.

    Two task-supervised ``nn.Linear`` transducers on continuous logits,
    NO discretisation, NO GTM. Substrates frozen. This is the path the
    paper's transmission number was measured on.
    """
    alphabet_size = mlp.emit_head_pi.out_features
    n_neurons = lif.n_neurons
    torch.manual_seed(seed + 100)
    transducer_ml = torch.nn.Linear(alphabet_size, n_neurons)
    transducer_lm = torch.nn.Linear(alphabet_size, alphabet_size)
    opt = torch.optim.Adam(
        list(transducer_ml.parameters()) + list(transducer_lm.parameters()), lr=1e-2
    )
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for _ in range(steps_transducer):
        x, y = task.sample(batch=64)
        pi_mlp = mlp.emit_head_pi(mlp.core(x))
        lif_features = transducer_ml(pi_mlp)
        pi_lif_out = lif.emit_head_pi(lif_features)
        pi_back_at_mlp = transducer_lm(pi_lif_out)
        loss = F.cross_entropy(pi_back_at_mlp[:, : task.n_classes], y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    x, y = task.sample(batch=batch)
    with torch.no_grad():
        pi_mlp = mlp.emit_head_pi(mlp.core(x))
        lif_features = transducer_ml(pi_mlp)
        pi_lif_out = lif.emit_head_pi(lif_features)
        pi_back_at_mlp = transducer_lm(pi_lif_out)
        pred_rt = pi_back_at_mlp[:, : task.n_classes].argmax(-1)
        acc_rt = (pred_rt == y).float().mean().item()
        pred_direct = mlp.emit_head_pi(mlp.core(x))[:, : task.n_classes].argmax(-1)
        acc_direct = (pred_direct == y).float().mean().item()
    return {
        "acc_direct": acc_direct,
        "acc_roundtrip": acc_rt,
        "fidelity_ratio": acc_rt / max(acc_direct, 1e-6),
    }


# ---------------------------------------------------------------------------
# Discrete-code real-protocol channel: Transducer (code→code) + GTM.
# ---------------------------------------------------------------------------


def _gtm_roundtrip_codes(gtm: GammaThetaMultiplexer, codes: torch.Tensor) -> torch.Tensor:
    """Pass a [B] long code vector through one GTM demod(forward()) round-trip.

    Mirrors ``ProtocolNerve.listen`` exactly: codes are chunked into batches
    of ``symbols_per_theta`` (K=7), each chunk traverses one PAC round-trip,
    recovered hard. Returns a [B] long tensor of recovered codes.
    """
    k_cap = gtm.cfg.symbols_per_theta
    out = torch.empty_like(codes)
    with torch.no_grad():
        for start in range(0, codes.shape[0], k_cap):
            chunk = codes[start : start + k_cap].unsqueeze(0)  # [1, k]
            n = chunk.shape[1]
            carrier = gtm.forward(chunk)
            # demodulate always reconstructs all k_cap slots; keep only the
            # first n that were actually transmitted in this (possibly short)
            # chunk. This mirrors ProtocolNerve.listen, which zips recovered
            # back onto the original chunk letters by position.
            recovered = gtm.demodulate(carrier, hard=True)  # [1, k_cap]
            out[start : start + n] = recovered[0, :n]
    return out


def _real_protocol_label(
    mlp,
    lif,
    transducer: Transducer,
    gtm: GammaThetaMultiplexer,
    t_lm: torch.nn.Linear,
    x: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Run the discrete real-protocol forward pass, return predicted labels.

    Pipeline (matches ProtocolNerve send()/listen()):
      x → MLP.emit_head_pi → argmax           [B] integer src code
        → Transducer (code→code, hard)        [B] integer dst code
        → GTM demod(forward())                [B] integer recovered code
        → LIF.codebook lookup (embed)         [B, n_neurons] feature
        → LIF.emit_head_pi                    [B, alphabet]
        → T_LM (learned linear back to MLP code space, then argmax label)

    The src→dst code mapping (Transducer) and the GTM are the REAL protocol
    components. The final ``t_lm`` read-out is the LIF-side decoder, the
    direct analogue of the paper's ``transducer_lm`` (it must exist on BOTH
    paths or the comparison is unfair to the real protocol; we keep it
    linear and identical in capacity to path (a)).
    """
    with torch.no_grad():
        pi_mlp = mlp.emit_head_pi(mlp.core(x))  # [B, alphabet]
        src_codes = pi_mlp.argmax(-1)  # [B] discrete src code
        dst_codes = transducer.forward(src_codes, hard=True)  # [B] discrete dst code
        recovered = _gtm_roundtrip_codes(gtm, dst_codes)  # [B] after GTM
        # LIF consumes the recovered code by looking it up in its codebook
        # (exactly embed_inbound for a single inbound letter) → input_proj.
        feat = lif.codebook[recovered]  # [B, n_neurons]
        pi_lif = lif.emit_head_pi(feat)  # [B, alphabet]
        pi_back = t_lm(pi_lif)  # [B, alphabet]
        return pi_back[:, :n_classes].argmax(-1)


def path_bc_real_protocol(
    mlp,
    lif,
    seed: int,
    *,
    train_transducer: bool,
    steps_transducer: int,
    batch: int,
    gtm_seed: int = 42,
) -> dict[str, float]:
    """Discrete real-protocol round-trip: Transducer(code→code) + GTM channel.

    ``train_transducer=False`` → path (b): the ``Transducer`` keeps its
    near-uniform random init (≈ untrained code map). ``train_transducer=True``
    → path (c): the ``Transducer`` is trained in GUMBEL_SOFTMAX mode under
    task supervision, with the GTM in the loss path, to align MLP→LIF
    codebooks. The LIF-side linear read-out ``t_lm`` is trained on BOTH paths
    (it is the analogue of the paper's ``transducer_lm``); only the
    code→code ``Transducer`` differs between (b) and (c).
    """
    alphabet_size = mlp.emit_head_pi.out_features
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)

    # Freeze substrates (as the paper does for round-trip).
    for p in list(mlp.parameters()) + list(lif.parameters()):
        p.requires_grad_(False)

    torch.manual_seed(seed + 300)
    transducer = Transducer(
        alphabet_size=alphabet_size, gating=TransducerGating.GUMBEL_SOFTMAX
    )
    gtm = GammaThetaMultiplexer(seed=gtm_seed)
    t_lm = torch.nn.Linear(alphabet_size, alphabet_size)

    if train_transducer:
        params = list(transducer.parameters()) + list(t_lm.parameters())
    else:
        # Path (b): transducer is NOT trained; only t_lm (the LIF-side decoder,
        # present on path (a) too) gets to fit. This isolates the effect of an
        # untrained code→code map + GTM versus the trained one.
        params = list(t_lm.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)

    for _ in range(steps_transducer):
        x, y = task.sample(batch=64)
        with torch.no_grad():
            pi_mlp = mlp.emit_head_pi(mlp.core(x))
            src_codes = pi_mlp.argmax(-1)  # [B] discrete
        # Soft code distribution from the transducer (differentiable).
        soft_dst = transducer.forward(src_codes, hard=False, tau=1.0)  # [B, alphabet]
        # Soft GTM round-trip: encode the soft-code expectation, demodulate
        # to a soft code distribution (gradient-flowing through the channel).
        # We take the argmax code for the carrier (hard, no grad through GTM
        # index) but keep the transducer gradient via straight-through on the
        # code distribution into the LIF feature lookup.
        dst_codes_hard = soft_dst.argmax(-1)
        recovered = _gtm_roundtrip_codes(gtm, dst_codes_hard)  # [B] after GTM
        # Straight-through estimator: forward uses the discrete recovered code's
        # LIF feature; backward flows the transducer's soft distribution into a
        # soft LIF feature so the code→code map gets task gradient through GTM.
        feat_hard = lif.codebook[recovered]  # [B, n_neurons]
        feat_soft = soft_dst @ lif.codebook  # [B, n_neurons] differentiable proxy
        feat = feat_hard + (feat_soft - feat_soft.detach())
        pi_lif = lif.emit_head_pi(feat)
        pi_back = t_lm(pi_lif)
        loss = F.cross_entropy(pi_back[:, : task.n_classes], y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    x, y = task.sample(batch=batch)
    with torch.no_grad():
        pred_rt = _real_protocol_label(
            mlp, lif, transducer, gtm, t_lm, x, task.n_classes
        )
        acc_rt = (pred_rt == y).float().mean().item()
        acc_direct = (
            (mlp.emit_head_pi(mlp.core(x))[:, : task.n_classes].argmax(-1) == y)
            .float()
            .mean()
            .item()
        )
        # GTM code-survival diagnostic on the (post-transducer) dst codes.
        pi_mlp = mlp.emit_head_pi(mlp.core(x))
        src_codes = pi_mlp.argmax(-1)
        dst_codes = transducer.forward(src_codes, hard=True)
        recovered = _gtm_roundtrip_codes(gtm, dst_codes)
        gtm_code_survival = (recovered == dst_codes).float().mean().item()
    return {
        "acc_direct": acc_direct,
        "acc_roundtrip": acc_rt,
        "fidelity_ratio": acc_rt / max(acc_direct, 1e-6),
        "gtm_code_survival": gtm_code_survival,
    }


# ---------------------------------------------------------------------------
# Multi-seed driver + bootstrap CIs.
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: list[float], *, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = arr.shape[0]
    for i in range(n_boot):
        boot[i] = arr[rng.integers(0, n, n)].mean()
    return (
        float(arr.mean()),
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def run(
    *,
    seeds: list[int],
    train_steps: int,
    transducer_steps: int,
    batch: int,
) -> dict:
    rows: list[dict] = []
    for seed in seeds:
        torch.manual_seed(seed)
        # One frozen MLP/LIF pair, reused across all three paths.
        mlp, lif, input_encoder = _train_pair(seed, steps=train_steps)

        a = path_a_idealised(
            mlp, lif, input_encoder, seed,
            steps_transducer=transducer_steps, batch=batch,
        )
        b = path_bc_real_protocol(
            mlp, lif, seed, train_transducer=False,
            steps_transducer=transducer_steps, batch=batch,
        )
        c = path_bc_real_protocol(
            mlp, lif, seed, train_transducer=True,
            steps_transducer=transducer_steps, batch=batch,
        )
        rows.append({"seed": seed, "a": a, "b": b, "c": c})
        print(
            f"seed={seed:2d} | (a) ideal ratio={a['fidelity_ratio']:.3f}  "
            f"(b) untrained+GTM ratio={b['fidelity_ratio']:.3f} "
            f"[gtm_surv={b['gtm_code_survival']:.2f}]  "
            f"(c) trained+GTM ratio={c['fidelity_ratio']:.3f} "
            f"[gtm_surv={c['gtm_code_survival']:.2f}]"
        )

    summary: dict[str, dict] = {}
    for path in ("a", "b", "c"):
        ratios = [r[path]["fidelity_ratio"] for r in rows]
        mean, lo, hi = _bootstrap_ci(ratios, seed=12345)
        entry = {
            "fidelity_ratio_mean": mean,
            "fidelity_ratio_ci95": [lo, hi],
            "fidelity_ratio_std": float(np.std(ratios, ddof=1)),
            "acc_roundtrip_mean": float(
                np.mean([r[path]["acc_roundtrip"] for r in rows])
            ),
            "acc_direct_mean": float(np.mean([r[path]["acc_direct"] for r in rows])),
        }
        if path in ("b", "c"):
            entry["gtm_code_survival_mean"] = float(
                np.mean([r[path]["gtm_code_survival"] for r in rows])
            )
        summary[path] = entry

    return {
        "experiment": "transmission_full_protocol",
        "metric": "fidelity_ratio = acc_roundtrip / acc_direct (paper Test 2)",
        "paths": {
            "a": "idealised: continuous-logit nn.Linear transducers, NO GTM "
            "(paper's measured path, run_test_2_round_trip_fidelity)",
            "b": "untrained track_p.transducer.Transducer (code->code) + GTM channel, "
            "discrete codes; only LIF-side linear decoder trained",
            "c": "TRAINED track_p.transducer.Transducer (gumbel-softmax) + GTM channel, "
            "discrete codes; transducer + LIF-side decoder trained",
        },
        "config": {
            "task": "HardFlowProxyTask(dim=16, n_classes=12)",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "train_steps": train_steps,
            "transducer_steps": transducer_steps,
            "batch": batch,
            "alphabet_size": 64,
            "gtm_modulation": "psk",
            "gtm_seed": 42,
        },
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--transducer-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.perf_counter()
    result = run(
        seeds=list(range(args.seeds)),
        train_steps=args.train_steps,
        transducer_steps=args.transducer_steps,
        batch=args.batch,
    )
    result["wall_clock_s"] = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("VERDICT — fidelity_ratio (acc_roundtrip / acc_direct), mean [95% CI]")
    print("=" * 72)
    for path, label in (
        ("a", "idealised (paper path, no GTM)"),
        ("b", "untrained transducer + GTM"),
        ("c", "TRAINED transducer + GTM (real protocol)"),
    ):
        s = result["summary"][path]
        lo, hi = s["fidelity_ratio_ci95"]
        extra = (
            f"  gtm_surv={s.get('gtm_code_survival_mean', float('nan')):.2f}"
            if path in ("b", "c")
            else ""
        )
        print(
            f"  ({path}) {label:<42} {s['fidelity_ratio_mean']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]{extra}"
        )
    print(f"\nwall_clock = {result['wall_clock_s']:.1f}s")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
