"""N10-A smoke runner: CIFAR-10 + ResNet18 + GTMBottleneck.

Stand-in for full ImageNet-100 sweep. Validates pipeline end-to-end:
- Load CIFAR-10 (auto-downloaded by torchvision)
- ResNet18 frozen feature extraction
- GTMBottleneck classification head
- Train 5 epochs, report top-1 accuracy

Plan: HYPNEUM-PLANS/2026-05-11-niveau10-scaling-up.md (N10-A smoke).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
import torchvision.transforms as T  # noqa: N812
from torch.utils.data import DataLoader

from experiments.imagenet100_gtm_bottleneck.architectures.gtm_bottleneck import (
    GTMBottleneck,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/tmp/cifar10_n10a")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "experiments/imagenet100_gtm_bottleneck/smoke_results.json"
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # CIFAR-10 dataset
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    args.data_dir.mkdir(parents=True, exist_ok=True)
    train = torchvision.datasets.CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    val = torchvision.datasets.CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train, batch_size=args.batch, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val, batch_size=args.batch, shuffle=False, num_workers=2
    )

    # Frozen ResNet18 backbone
    backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    backbone.fc = nn.Identity()  # keep features [B, 512]
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Bottleneck head
    head = GTMBottleneck(
        feat_dim=512,
        code_dim=args.code_dim,
        n_classes=10,
        seed=args.seed,
    ).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    # Limit smoke to first N batches per epoch to keep wallclock < 10 min
    train_steps = min(50, len(train_loader))
    val_steps = min(20, len(val_loader))

    t0 = time.perf_counter()
    epoch_metrics = []
    for ep in range(args.epochs):
        # Train
        head.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        for i, (x, y) in enumerate(train_loader):
            if i >= train_steps:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = backbone(x)
            logits, _ = head(feats)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            ep_correct += int((logits.argmax(-1) == y).sum().item())
            ep_total += y.size(0)
        train_acc = ep_correct / ep_total
        train_loss = ep_loss / train_steps

        # Val
        head.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= val_steps:
                    break
                x, y = x.to(device), y.to(device)
                feats = backbone(x)
                logits, _ = head(feats)
                v_correct += int((logits.argmax(-1) == y).sum().item())
                v_total += y.size(0)
        val_acc = v_correct / v_total
        epoch_metrics.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })
        print(
            f"  ep={ep} train_loss={train_loss:.3f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

    wallclock = time.perf_counter() - t0
    print(
        f"\n=== Smoke complete : {wallclock:.1f}s for {args.epochs} epochs "
        f"(limited to {train_steps} train + {val_steps} val batches/ep) ==="
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "device": device,
        "stand_in_dataset": "CIFAR-10",
        "stand_in_backbone": "ResNet18 (frozen, ImageNet pretrained)",
        "code_dim": args.code_dim,
        "n_classes": 10,
        "epochs": args.epochs,
        "train_steps_per_ep": train_steps,
        "val_steps_per_ep": val_steps,
        "epoch_metrics": epoch_metrics,
        "wallclock_s": wallclock,
    }, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
