"""Fine-tune shared MoE heads on Train; select checkpoint on Val only."""

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset import RefELDataset
from shared_moe import SharedMoE


ROOT = Path(__file__).resolve().parent


def router():
    path = ROOT / "43_final_testB_gate_v2_and_m2_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_router", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loss_fn(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    probs = torch.sigmoid(logits)
    intersection = (probs * mask).sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1) /
                (probs.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1)).mean()
    return bce + dice


def sample_indices(dataset, count, seed):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return indices[:min(count, len(dataset))]


@torch.no_grad()
def evaluate(model, dataset, indices, route, device):
    model.eval()
    intersection = union = 0
    for index in indices:
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device) > 0.5
        general, context = model.forward_general(image, texts=[item["text"]])
        ratio = float((torch.sigmoid(general) >= 0.5).float().mean())
        branch = route.route_sample(item["text"], ratio)[0]
        pred = torch.sigmoid(model.forward_expert(context, branch)) >= 0.5
        intersection += int((pred & mask).sum())
        union += int((pred | mask).sum())
    return intersection / max(union, 1)


def main(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    route = router()
    model = SharedMoE.from_checkpoints(ROOT / "checkpoints", device)
    if args.resume:
        model.load_heads(args.resume)
    for param in model.general.parameters():
        param.requires_grad_(False)
    train = RefELDataset(ROOT, split="train", image_size=384, random_text=True)
    val = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    val_ids = sample_indices(val, args.val_samples, args.seed)
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=args.lr)
    output = ROOT / "results" / "shared_moe_training"
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.general.eval()
        train_ids = sample_indices(train, args.train_samples, args.seed + epoch)
        total_loss = 0.0
        for step, index in enumerate(train_ids, 1):
            item = train[index]
            image = item["image"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).to(device).float()
            with torch.no_grad():
                general, context = model.forward_general(image, texts=[item["text"]])
                ratio = float((torch.sigmoid(general) >= 0.5).float().mean())
                branch = route.route_sample(item["text"], ratio)[0]
            logits = model.forward_expert(context, branch)
            loss = loss_fn(logits, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{len(train_ids)} loss {total_loss/step:.4f}", flush=True)
        score = evaluate(model, val, val_ids, route, device)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_ids),
               "val_cIoU": score, "val_samples": len(val_ids)}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        model.save_heads(output / "latest_heads.pth")
        if score > best:
            best = score
            model.save_heads(output / "best_heads.pth")
        print(row, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--val-samples", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    options = parser.parse_args()
    if min(options.epochs, options.train_samples, options.val_samples) < 1:
        parser.error("epochs and sample counts must be positive")
    main(options)
