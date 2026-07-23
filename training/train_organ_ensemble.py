from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from training.common import ORGANS, atomic_torch_save, center_of, verify_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    verify_sha256(args.features, args.features_sha256)
    if args.skip_existing and args.out.exists():
        print(f"SKIP {args.out}")
        return

    import torch
    import torch.nn as nn

    with np.load(args.features, allow_pickle=False) as data:
        features = data["feats"].astype(np.float32)
        organs = data["organ"].astype(str)
        wsis = data["wsi"].astype(str)
    unknown = sorted(set(organs) - set(ORGANS))
    if unknown:
        raise ValueError(f"unknown organs: {unknown}")
    labels = np.asarray([ORGANS.index(organ) for organ in organs], dtype=np.int64)
    mean = features.mean(axis=0)
    standard_deviation = features.std(axis=0) + 1e-6
    normalized = torch.from_numpy((features - mean) / standard_deviation)
    targets = torch.from_numpy(labels)

    cells = np.asarray(
        [f"{center_of(wsi)}|{organ}" for wsi, organ in zip(wsis, organs)]
    )
    wsi_counts = {cell: len(set(wsis[cells == cell])) for cell in sorted(set(cells))}
    tile_counts = Counter(cells.tolist())
    sample_weights = np.asarray(
        [np.sqrt(wsi_counts[cell]) / tile_counts[cell] for cell in cells],
        dtype=np.float64,
    )
    sample_weights /= sample_weights.sum()
    class_counts = np.bincount(labels, minlength=len(ORGANS)).astype(float)
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=args.device)
    )

    states = []
    for offset in range(args.heads):
        seed = args.seed + offset
        torch.manual_seed(seed)
        generator = np.random.RandomState(seed)
        model = nn.Linear(features.shape[1], len(ORGANS)).to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        for _ in range(args.epochs):
            indices = generator.choice(
                len(features), size=len(features), replace=True, p=sample_weights
            )
            for start in range(0, len(indices), args.batch_size):
                batch = indices[start : start + args.batch_size]
                optimizer.zero_grad()
                loss = criterion(
                    model(normalized[batch].to(args.device)),
                    targets[batch].to(args.device),
                )
                loss.backward()
                optimizer.step()
        states.append(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
        )
    bundle = {
        "states": states,
        "mu": mean,
        "sd": standard_deviation,
        "organs": list(ORGANS),
        "n": args.heads,
        "dim": features.shape[1],
        "release": "0.6.0",
        "seeds": list(range(args.seed, args.seed + args.heads)),
    }
    atomic_torch_save(args.out, bundle)
    print(f"WROTE {args.out}: {args.heads} heads, dimension={features.shape[1]}")


if __name__ == "__main__":
    main()
