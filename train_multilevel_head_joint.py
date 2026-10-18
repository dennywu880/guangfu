"""Isolated head-plus-adapter stage after the frozen e3/e4 ablation.

The shared image/text encoder and frozen routing never change.  This stage
tests whether an expert head can adapt to the shared feature distribution.
It never reads the historical holdout or modifies the production application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import torch

from dataset import RefELDataset
from multilevel_selection import build_model
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, evaluate, load_or_create_split,
    load_router, pooled_for, route_indices, segmentation_loss, stack_items,
    size_loss_weights, teacher_for, decode_teacher,
)
from train_text import FormalTextCache


SOURCE = ROOT / "results" / "multilevel_e3_e4_20260925" / "best.pth"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compact(metrics):
    return {key: value for key, value in metrics.items() if key != "rows"}


def passes_promotion_gate(metrics, baseline):
    return (
        metrics["cIoU"] > baseline["cIoU"]
        and metrics["mean_IoU"] >= baseline["mean_IoU"]
        and metrics["by_bucket"]["ultra_tiny"]["mean_IoU"]
            >= baseline["by_bucket"]["ultra_tiny"]["mean_IoU"]
        and metrics["by_bucket"]["tiny"]["mean_IoU"]
            >= baseline["by_bucket"]["tiny"]["mean_IoU"]
        and metrics["false_activation"] <= baseline["false_activation"]
    )


def save(path: Path, model, metadata):
    torch.save({
        "format": "multilevel_head_joint_v1",
        "metadata": metadata,
        "source_sha256": metadata["source_sha256"],
        "heads": {branch: head.state_dict()
                  for branch, head in model.base.heads.items()},
        "selections": {branch: {level: module.state_dict()
                                for level, module in levels.items()}
                       for branch, levels in model.selections.items()},
    }, path)


def load(path: Path, model):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "multilevel_head_joint_v1":
        raise ValueError("Unsupported head-joint checkpoint")
    if state.get("source_sha256") != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        raise ValueError("Source checkpoint changed; refusing incompatible restore")
    if set(state["heads"]) != set(model.base.heads) or set(state["selections"]) != set(model.selections):
        raise ValueError("Checkpoint expert branches are incomplete")
    for branch, head in model.base.heads.items():
        head.load_state_dict(state["heads"][branch], strict=True)
    for branch, levels in model.selections.items():
        if set(state["selections"][branch]) != set(levels):
            raise ValueError(f"Checkpoint levels are incomplete for {branch}")
        for level, module in levels.items():
            module.load_state_dict(state["selections"][branch][level], strict=True)
    return state["metadata"]


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for an explicit CPU run")
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "architecture": "one frozen shared encoder, three e3/e4 adapters and heads",
        "trainable": "e3/e4 adapters and expert heads; head BatchNorm running stats frozen",
        "router": "fixed historical rule",
        "seed": args.seed,
        "epochs": args.epochs,
        "train_samples_per_epoch": args.train_samples,
        "batch_size": args.batch_size,
        "adapter_lr": args.adapter_lr,
        "head_lr": args.head_lr,
        "teacher_distill_weight": args.teacher_distill_weight,
        "tiny_loss_weight": args.tiny_loss_weight,
        "ultra_loss_weight": args.ultra_loss_weight,
        "positive_only_teacher": True,
        "promotion_gate": "cIoU improves; mean, tiny and ultra-tiny IoU do not decline; false activations do not increase",
        "testB_used": False,
        "holdout_used": False,
        "web_untouched": True,
        "limitation": "Historical expert weights may have seen Val images.",
    }
    write_json(output / "manifest.json", metadata)
    train = RefELDataset(ROOT, "train", 384, False)
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    dev, holdout = load_or_create_split(val, args.seed)
    pool, excluded, audit = build_disjoint_train_pool(train, val, dev, holdout)
    write_json(output / "split_audit.json", {
        "dev_indices": dev, "holdout_indices": holdout,
        "excluded_image_ids": sorted(excluded), "audit": audit,
    })
    model = build_model(device, ("e3", "e4"))
    model.load_checkpoint(SOURCE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.selections.parameters():
        parameter.requires_grad_(True)
    for parameter in model.base.heads.parameters():
        parameter.requires_grad_(True)
    adapter_parameters = list(model.selections.parameters())
    head_parameters = list(model.base.heads.parameters())
    metadata["new_parameters"] = sum(p.numel() for p in adapter_parameters)
    metadata["joint_trainable_parameters"] = sum(
        p.numel() for p in adapter_parameters + head_parameters
    )
    write_json(output / "manifest.json", metadata)
    optimizer = torch.optim.AdamW([
        {"params": adapter_parameters, "lr": args.adapter_lr},
        {"params": head_parameters, "lr": args.head_lr},
    ], weight_decay=1e-4)
    router = load_router()
    initial = compact(evaluate(model, val, dev, router, device, cache, "val", args.batch_size))
    write_json(output / "initial_metrics.json", initial)
    save(output / "initial.pth", model, metadata)
    history = [{"epoch": 0, "metrics": initial}]
    best = initial["cIoU"]
    best_eligible = initial["cIoU"]
    promoted_epoch = None
    for epoch in range(1, args.epochs + 1):
        indices = list(pool)
        random.Random(args.seed + epoch).shuffle(indices)
        indices = indices[:min(args.train_samples, len(indices))]
        groups = route_indices(model, train, indices, router, device,
                               cache, "train", args.batch_size)
        model.train()
        model.base.eval()
        model.base.heads.eval()
        model.selections.train()
        losses = []
        for branch, branch_indices in groups.items():
            teacher = teacher_for(branch, device) if args.teacher_distill_weight else None
            for start in range(0, len(branch_indices), args.batch_size):
                ids = branch_indices[start:start + args.batch_size]
                items, images, masks, _ = stack_items(train, ids, device)
                pooled = pooled_for(cache, items, "train", device)
                with torch.no_grad():
                    _, context = model.forward_general(images, pooled_text=pooled)
                logits = model.forward_expert(context, branch)
                weights = size_loss_weights(
                    masks, args.tiny_loss_weight, args.ultra_loss_weight
                ) if args.tiny_loss_weight != 1 or args.ultra_loss_weight != 1 else None
                loss = segmentation_loss(logits, masks, weights)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits, _ = decode_teacher(
                            teacher, images, context.pooled_text
                        )
                    # Teachers are only trusted on images with a real target;
                    # no-target samples keep the ground-truth negative signal.
                    positive = (masks.flatten(1).sum(dim=1) > 0).float()
                    imitation = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, teacher_logits.sigmoid(), reduction="none"
                    ).flatten(1).mean(dim=1)
                    if bool(positive.any().item()):
                        loss = loss + args.teacher_distill_weight * (
                            imitation * positive
                        ).sum() / positive.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_parameters + head_parameters, 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            if teacher is not None:
                del teacher
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        metrics = compact(evaluate(model, val, dev, router, device,
                                   cache, "val", args.batch_size))
        row = {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)),
               "routes": {branch: len(ids) for branch, ids in groups.items()},
               "metrics": metrics}
        history.append(row)
        write_json(output / "history.json", history)
        save(output / "latest.pth", model, metadata)
        if metrics["cIoU"] > best:
            best = metrics["cIoU"]
            save(output / "best.pth", model, metadata)
        if passes_promotion_gate(metrics, initial) and metrics["cIoU"] > best_eligible:
            best_eligible = metrics["cIoU"]
            promoted_epoch = epoch
            save(output / "promoted.pth", model, metadata)
        print({"epoch": epoch, "mean_IoU": metrics["mean_IoU"],
               "cIoU": metrics["cIoU"],
               "ultra_tiny": metrics["by_bucket"]["ultra_tiny"]["mean_IoU"],
               "tiny": metrics["by_bucket"]["tiny"]["mean_IoU"]}, flush=True)
    save(output / "completed.pth", model, metadata)
    write_json(output / "selection.json", {
        "selected_checkpoint": "promoted.pth" if promoted_epoch is not None
                               else "initial.pth",
        "promoted_epoch": promoted_epoch,
        "highest_cIoU_checkpoint": "best.pth" if (output / "best.pth").exists()
                                   else "initial.pth",
        "highest_dev_cIoU": best,
        "promotion_gate": metadata["promotion_gate"],
        "holdout_not_examined": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--adapter-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--teacher-distill-weight", type=float, default=0.05)
    parser.add_argument("--tiny-loss-weight", type=float, default=1.25)
    parser.add_argument("--ultra-loss-weight", type=float, default=1.75)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--cpu", action="store_true")
    options = parser.parse_args()
    if min(options.epochs, options.train_samples, options.batch_size) < 1:
        parser.error("epochs, train-samples and batch-size must be positive")
    if min(options.teacher_distill_weight, options.tiny_loss_weight,
           options.ultra_loss_weight) < 0:
        parser.error("loss weights must be non-negative")
    run(options)
