"""Controlled e2, e1+e2, e2+e3 expert-selection ablation."""

import argparse
import hashlib
import json
import os
import random
from datetime import datetime

import torch

from dataset import RefELDataset
from feature_bridge_experiment import BASE_CHECKPOINT
from multilevel_selection import build_model
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, evaluate, load_or_create_split,
    load_router, pooled_for, route_indices, segmentation_loss, stack_items,
)
from train_text import FormalTextCache


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(metrics):
    return {key: value for key, value in metrics.items() if key != "rows"}


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for an explicit CPU run")
    levels = tuple(args.levels.split("+"))
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "levels": levels, "seed": args.seed, "device": str(device),
        "epochs": args.epochs, "train_samples_per_epoch": args.train_samples,
        "batch_size": args.batch_size, "lr": args.lr,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_sha256": hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest(),
        "web_untouched": True, "testB_used": False, "holdout_used": False,
        "evaluation_note": "Historical expert weights may have seen Val images.",
    }
    write_json(output / "manifest.json", manifest)
    train = RefELDataset(ROOT, "train", 384, False)
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    dev, holdout = load_or_create_split(val, args.seed)
    pool, excluded, audit = build_disjoint_train_pool(train, val, dev, holdout)
    write_json(output / "split_audit.json", {
        "dev_indices": dev, "holdout_indices": holdout,
        "excluded_image_ids": sorted(excluded), "audit": audit,
    })
    model = build_model(device, levels)
    router = load_router()
    initial = evaluate(model, val, dev, router, device, cache, "val", args.batch_size)
    write_json(output / "initial_metrics.json", summarize(initial))
    model.save_checkpoint(output / "initial.pth", manifest)
    parameters = [parameter for parameter in model.selections.parameters()
                  if parameter.requires_grad]
    manifest["new_parameters"] = sum(parameter.numel() for parameter in parameters)
    write_json(output / "manifest.json", manifest)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    best_c_iou = initial["cIoU"]
    history = [{"epoch": 0, "metrics": summarize(initial)}]
    for epoch in range(1, args.epochs + 1):
        indices = list(pool)
        random.Random(args.seed + epoch).shuffle(indices)
        indices = indices[:min(args.train_samples, len(indices))]
        groups = route_indices(model, train, indices, router, device,
                               cache, "train", args.batch_size)
        model.train()
        losses = []
        for branch, branch_indices in groups.items():
            for start in range(0, len(branch_indices), args.batch_size):
                ids = branch_indices[start:start + args.batch_size]
                items, images, masks, _ = stack_items(train, ids, device)
                pooled = pooled_for(cache, items, "train", device)
                # The shared encoder and old decoder stay frozen throughout.
                with torch.no_grad():
                    _, context = model.forward_general(images, pooled_text=pooled)
                logits = model.forward_expert(context, branch)
                loss = segmentation_loss(logits, masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5)
                optimizer.step()
                losses.append(float(loss.detach()))
        metrics = evaluate(model, val, dev, router, device, cache, "val", args.batch_size)
        row = {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)),
               "routes": {branch: len(ids) for branch, ids in groups.items()},
               "metrics": summarize(metrics)}
        history.append(row)
        write_json(output / "history.json", history)
        model.save_checkpoint(output / "latest.pth", manifest)
        if metrics["cIoU"] > best_c_iou:
            best_c_iou = metrics["cIoU"]
            model.save_checkpoint(output / "best.pth", manifest)
        print({"levels": levels, "epoch": epoch,
               "mean_IoU": metrics["mean_IoU"], "cIoU": metrics["cIoU"],
               "ultra_tiny": metrics["by_bucket"]["ultra_tiny"]["mean_IoU"],
               "tiny": metrics["by_bucket"]["tiny"]["mean_IoU"]}, flush=True)
    model.save_checkpoint(output / "completed.pth", manifest)
    write_json(output / "selection.json", {
        "selected_checkpoint": "best.pth" if (output / "best.pth").exists()
                               else "initial.pth",
        "best_dev_cIoU": best_c_iou,
        "holdout_not_examined": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", choices=("e2", "e1+e2", "e2+e3", "e3+e4"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
