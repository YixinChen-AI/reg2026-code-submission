from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from training.common import (
    atomic_torch_save,
    center_of,
    diagnosis_of,
    load_cases,
    verify_sha256,
    wsi_stem,
)


def head_from_config(config: Path, head_id: str) -> tuple[dict, dict]:
    document = json.loads(config.read_text(encoding="utf-8"))
    matches = [head for head in document["heads"] if head["id"] == head_id]
    if len(matches) != 1:
        raise ValueError(f"expected one head named {head_id}, found {len(matches)}")
    return document["model"], matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256")
    parser.add_argument("--head-id", required=True)
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--cot-sha256")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-tiles", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    verify_sha256(args.config, args.config_sha256)
    verify_sha256(args.cot, args.cot_sha256)
    model_config, head = head_from_config(args.config, args.head_id)
    output = args.work_root / str(head["output"])
    if args.skip_existing and output.exists():
        print(f"SKIP {output}")
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class GatedAttention(nn.Module):
        def __init__(self, branches: int):
            super().__init__()
            self.V = nn.Linear(768, 128)
            self.U = nn.Linear(768, 128)
            self.W = nn.Linear(128, branches)

        def forward(self, features):
            return self.W(
                torch.tanh(self.V(features)) * torch.sigmoid(self.U(features))
            ).T

    class ACMIL(nn.Module):
        def __init__(self, branches: int, classes: int):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(1536, 768), nn.ReLU())
            self.attn = GatedAttention(branches)
            self.branch_cls = nn.ModuleList(
                [nn.Linear(768, classes) for _ in range(branches)]
            )
            self.bag_cls = nn.Linear(768, classes)
            self.branches = branches

        def forward(self, bag, training: bool, n_masked: int, mask_drop: float):
            hidden = self.fc(bag)
            attention = self.attn(hidden)
            if training and self.branches > 1 and n_masked > 0 and mask_drop > 0:
                masked = attention.clone()
                for branch in range(self.branches):
                    count = min(n_masked, attention.shape[1])
                    top = torch.topk(attention[branch], count).indices
                    drop_count = int(round(count * mask_drop))
                    if drop_count:
                        masked[
                            branch,
                            top[torch.randperm(count, device=bag.device)[:drop_count]],
                        ] = -1e9
                attention = masked
            weights = torch.softmax(attention, dim=1)
            branch_features = weights @ hidden
            branch_logits = torch.stack(
                [
                    classifier(branch_features[index])
                    for index, classifier in enumerate(self.branch_cls)
                ]
            )
            bag_logits = self.bag_cls(weights.mean(0, keepdim=True) @ hidden)
            return bag_logits, branch_logits, weights

    cases = load_cases(args.cot)
    metadata = {}
    for case in cases:
        diagnosis = diagnosis_of(case)
        if diagnosis:
            metadata[wsi_stem(case.get("id"))] = (
                str(case.get("organ", "")).lower(),
                diagnosis,
            )
    feature_directory = args.work_root / str(head["features"])
    pool = sorted(
        path.stem
        for path in feature_directory.glob("*.npy")
        if metadata.get(path.stem, (None,))[0] == head["organ"]
    )
    if len(pool) < 3:
        raise RuntimeError(f"insufficient bags for {args.head_id}: {len(pool)}")
    vocab = sorted({metadata[stem][1] for stem in pool})
    label_to_index = {label: index for index, label in enumerate(vocab)}
    seed = int(head["seed"])
    split_rng = np.random.RandomState(seed)
    training_rng = np.random.RandomState(seed)
    shuffled = list(np.asarray(pool)[split_rng.permutation(len(pool))])
    train_count = min(
        len(pool), max(2, int(round(len(pool) * float(model_config["subsample"]))))
    )
    train_stems = shuffled[:train_count]
    if train_count < len(pool):
        selection_stems = shuffled[train_count:]
    else:
        selection_stems = shuffled[: max(1, len(pool) // 10)]
    epochs = args.epochs if args.epochs is not None else int(model_config["epochs"])

    def load_bag(stem: str):
        values = np.load(feature_directory / f"{stem}.npy", allow_pickle=False).astype(
            np.float32
        )
        if args.max_tiles and len(values) > args.max_tiles:
            indices = np.linspace(0, len(values) - 1, args.max_tiles).astype(int)
            values = values[indices]
        if not len(values):
            raise ValueError(f"empty feature bag: {stem}")
        return torch.from_numpy(values)

    torch.manual_seed(seed)
    train_bags = {stem: load_bag(stem) for stem in train_stems}
    selection_bags = {stem: load_bag(stem).to(args.device) for stem in selection_stems}
    counts = np.bincount(
        [label_to_index[metadata[stem][1]] for stem in train_stems],
        minlength=len(vocab),
    ).astype(float)
    class_weights = counts.sum() / (len(vocab) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=args.device)
    )
    branches = int(model_config["branches"])
    n_masked = int(model_config["n_masked"])
    mask_drop = float(model_config["mask_drop"])
    model = ACMIL(branches, len(vocab)).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    sampling = None
    if model_config.get("center_balance"):
        center_counts = Counter(center_of(stem) for stem in train_stems)
        sampling = np.asarray(
            [1.0 / center_counts[center_of(stem)] for stem in train_stems],
            dtype=np.float64,
        )
        sampling /= sampling.sum()
    best_accuracy = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        order = (
            training_rng.choice(
                len(train_stems), len(train_stems), replace=True, p=sampling
            )
            if sampling is not None
            else training_rng.permutation(len(train_stems))
        )
        for index in order:
            stem = train_stems[int(index)]
            target = torch.tensor(
                [label_to_index[metadata[stem][1]]], device=args.device
            )
            logits, branch_logits, attention = model(
                train_bags[stem].to(args.device), True, n_masked, mask_drop
            )
            loss = criterion(logits, target)
            if branches > 1:
                loss += criterion(branch_logits, target.expand(branches))
                normalized = functional.normalize(attention, dim=1)
                similarity = normalized @ normalized.T
                loss += (similarity.sum() - branches) / (branches * (branches - 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.inference_mode():
            for stem, bag in selection_bags.items():
                logits, _, _ = model(bag, False, n_masked, mask_drop)
                correct += int(vocab[int(logits.argmax())] == metadata[stem][1])
        accuracy = correct / len(selection_stems)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    checkpoint = {
        "state": best_state,
        "vocab": vocab,
        "K": branches,
        "n_masked": n_masked,
        "mask_drop": mask_drop,
        "organ": head["organ"],
        "scale": int(head["scale"]),
        "seed": seed,
        "release": "0.6.0",
    }
    atomic_torch_save(output, checkpoint)
    print(
        f"WROTE {output}: train={len(train_stems)} selection={len(selection_stems)} "
        f"selection_overlap={len(set(train_stems) & set(selection_stems))} "
        f"classes={len(vocab)} best_selection_accuracy={best_accuracy:.6f}"
    )


if __name__ == "__main__":
    main()
