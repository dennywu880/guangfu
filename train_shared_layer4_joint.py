"""Isolated fine-tuning of the shared layer4 and three specialist paths.

The existing router, presence gate, deployed model, and source checkpoints are
untouched. Evaluation image IDs are excluded from this stage's training pool.
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
from shared_moe import SharedFeatures
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, evaluate, load_or_create_split,
    load_router, pooled_for, route_indices, segmentation_loss, stack_items,
    size_loss_weights,
)
from train_multilevel_head_joint import compact, load as load_source
from train_text import FormalTextCache


SOURCE = ROOT / "results" / "head_joint_e3_e4_20260925" / "best.pth"
DEVELOPMENT_INDICES = (
    ROOT / "results" / "expanded_routing_audit_20260926" / "sample_indices.json"
)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save(path: Path, model, metadata):
    torch.save({
        "format": "shared_layer4_joint_v1",
        "metadata": metadata,
        "source_sha256": metadata["source_sha256"],
        "layer4": model.base.general.layer4.state_dict(),
        "heads": {name: head.state_dict() for name, head in model.base.heads.items()},
        "selections": {
            name: {level: module.state_dict() for level, module in levels.items()}
            for name, levels in model.selections.items()
        },
    }, path)


def load(path: Path, model):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "shared_layer4_joint_v1":
        raise ValueError("Unsupported layer4-joint checkpoint")
    if state["source_sha256"] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        raise ValueError("Source checkpoint changed; refusing incompatible restore")
    model.base.general.layer4.load_state_dict(state["layer4"], strict=True)
    if set(state["heads"]) != set(model.base.heads):
        raise ValueError("Expert heads are incomplete")
    for name, head in model.base.heads.items():
        head.load_state_dict(state["heads"][name], strict=True)
    if set(state["selections"]) != set(model.selections):
        raise ValueError("Expert adapters are incomplete")
    for name, levels in model.selections.items():
        if set(state["selections"][name]) != set(levels):
            raise ValueError(f"Missing adapter level for {name}")
        for level, module in levels.items():
            module.load_state_dict(state["selections"][name][level], strict=True)
    return state["metadata"]


def expert_context(model, images, pooled):
    encoder = model.base.general
    with torch.no_grad():
        e0 = encoder.stem(images)
        e1 = encoder.layer1(encoder.maxpool(e0))
        e2 = encoder.layer2(e1)
        e3 = encoder.layer3(e2)
    e4 = encoder.layer4(e3)
    return SharedFeatures((e0, e1, e2, e3, e4), pooled, images.shape[-2:], images)


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for an explicit CPU run")
    if not SOURCE.is_file() or not DEVELOPMENT_INDICES.is_file():
        raise FileNotFoundError("Required source or development index file is missing")
    output = ROOT / "results" / args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train = RefELDataset(ROOT, "train", 384, False)
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    dev, holdout = load_or_create_split(val, args.seed)
    pool, excluded, split_audit = build_disjoint_train_pool(train, val, dev, holdout)
    expanded = json.loads(DEVELOPMENT_INDICES.read_text(encoding="utf-8"))
    expanded_images = {val.refs[index]["image_id"] for index in expanded}
    excluded |= expanded_images
    pool = [index for index in pool if train.refs[index]["image_id"] not in excluded]
    if not pool:
        raise RuntimeError("No training references after image exclusion")
    model = build_model(device, ("e3", "e4"))
    load_source(SOURCE, model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.base.general.layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in model.selections.parameters():
        parameter.requires_grad_(True)
    for parameter in model.base.heads.parameters():
        parameter.requires_grad_(True)
    layer4_params = list(model.base.general.layer4.parameters())
    adapter_params = list(model.selections.parameters())
    head_params = list(model.base.heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": layer4_params, "lr": args.layer4_lr},
        {"params": adapter_params, "lr": args.adapter_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "architecture": "one shared ResNet34; train layer4 plus e3/e4 expert adapters and heads",
        "router": "frozen size-word and general-predicted-area rule",
        "presence_gate": "frozen, used in later full-system audit only",
        "epochs": args.epochs, "train_samples_per_epoch": args.train_samples,
        "batch_size": args.batch_size, "seed": args.seed,
        "layer4_lr": args.layer4_lr, "adapter_lr": args.adapter_lr,
        "head_lr": args.head_lr,
        "tiny_loss_weight": args.tiny_loss_weight,
        "ultra_loss_weight": args.ultra_loss_weight,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "eligible_train_references": len(pool),
        "excluded_image_count": len(excluded),
        "expanded_development_images_excluded": len(expanded_images),
        "split_audit": split_audit,
        "historical_image_exposure_not_excluded": True,
        "testB_used": False, "web_untouched": True,
    }
    write_json(output / "manifest.json", metadata)
    router = load_router()
    initial = compact(evaluate(model, val, dev, router, device, cache, "val", args.batch_size))
    save(output / "initial.pth", model, metadata)
    history = [{"epoch": 0, "metrics": initial, "train_loss": None}]
    write_json(output / "history.json", history)
    print({"epoch": 0, "metrics": initial}, flush=True)
    for epoch in range(1, args.epochs + 1):
        indices = list(pool)
        random.Random(args.seed + epoch).shuffle(indices)
        indices = indices[:min(args.train_samples, len(indices))]
        groups = route_indices(model, train, indices, router, device, cache,
                               "train", args.batch_size)
        model.train()
        model.base.eval()
        model.selections.train()
        losses = []
        for branch, branch_indices in groups.items():
            for start in range(0, len(branch_indices), args.batch_size):
                ids = branch_indices[start:start + args.batch_size]
                items, images, masks, _ = stack_items(train, ids, device)
                pooled = pooled_for(cache, items, "train", device)
                context = expert_context(model, images, pooled)
                logits = model.forward_expert(context, branch)
                weights = size_loss_weights(
                    masks, args.tiny_loss_weight, args.ultra_loss_weight
                )
                loss = segmentation_loss(logits, masks, weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(layer4_params + adapter_params + head_params, 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            print({"epoch": epoch, "branch": branch,
                   "samples": len(branch_indices), "steps_done": len(losses)}, flush=True)
        metrics = compact(evaluate(model, val, dev, router, device,
                                   cache, "val", args.batch_size))
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses),
               "routes": {name: len(ids) for name, ids in groups.items()},
               "metrics": metrics}
        history.append(row)
        write_json(output / "history.json", history)
        save(output / f"epoch_{epoch:02d}.pth", model, metadata)
        print({"epoch": epoch, "train_loss": row["train_loss"],
               "metrics": metrics}, flush=True)
    write_json(output / "completion.json", {
        "completed_epochs": args.epochs,
        "initial_checkpoint": "initial.pth",
        "epoch_checkpoints": [f"epoch_{i:02d}.pth" for i in range(1, args.epochs + 1)],
        "testB_not_evaluated": True,
        "web_not_switched": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layer4-lr", type=float, default=5e-6)
    parser.add_argument("--adapter-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--tiny-loss-weight", type=float, default=1.25)
    parser.add_argument("--ultra-loss-weight", type=float, default=1.75)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--cpu", action="store_true")
    options = parser.parse_args()
    if min(options.epochs, options.train_samples, options.batch_size) < 1:
        parser.error("epochs, train-samples and batch-size must be positive")
    run(options)
