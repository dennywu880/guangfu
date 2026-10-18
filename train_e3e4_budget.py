"""Increase the training budget of the historical e3/e4 adapter experiment.

This is a development experiment on historical weights. It does not alter the
deployed model or use the locked historical holdout or testB for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime

import torch

from dataset import RefELDataset
from multilevel_selection import build_model
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, evaluate, load_or_create_split,
    load_router, pooled_for, route_indices, segmentation_loss, stack_items,
)
from train_multilevel_head_joint import load as load_head_joint
from train_text import FormalTextCache


SOURCE = ROOT / "results" / "head_joint_e3_e4_20260925" / "initial.pth"
DEV_INDICES = ROOT / "results" / "expanded_routing_audit_20260926" / "sample_indices.json"
STAGES = (3000, 6000, 12000)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def checkpoint_state(model, optimizer, manifest, stage):
    return {
        "format": "e3e4_budget_v1",
        "source_sha256": manifest["source_sha256"],
        "stage": stage,
        "heads": {name: head.state_dict() for name, head in model.base.heads.items()},
        "selections": {
            name: {level: module.state_dict() for level, module in levels.items()}
            for name, levels in model.selections.items()
        },
        "optimizer": optimizer.state_dict(),
    }


def restore(path, model, optimizer, source_sha256, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != "e3e4_budget_v1" or state.get("source_sha256") != source_sha256:
        raise ValueError("Incompatible budget checkpoint")
    if set(state["heads"]) != set(model.base.heads) or set(state["selections"]) != set(model.selections):
        raise ValueError("Checkpoint is missing expert branches")
    for name, head in model.base.heads.items():
        head.load_state_dict(state["heads"][name], strict=True)
    for name, levels in model.selections.items():
        if set(state["selections"][name]) != set(levels):
            raise ValueError(f"Checkpoint is missing {name} adapter levels")
        for level, module in levels.items():
            module.load_state_dict(state["selections"][name][level], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state["stage"])


def metric_summary(metrics):
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
    if not SOURCE.is_file() or not DEV_INDICES.is_file():
        raise FileNotFoundError("Historical source checkpoint or development indices missing")

    output = ROOT / "results" / args.output
    if output.exists() != args.resume:
        raise FileExistsError(output) if output.exists() else FileNotFoundError(output)
    if not args.resume:
        output.mkdir(parents=True)

    train = RefELDataset(ROOT, "train", 384, False)
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    old_dev, holdout = load_or_create_split(val, args.seed)
    pool, excluded_images, split_audit = build_disjoint_train_pool(train, val, old_dev, holdout)
    dev = json.loads(DEV_INDICES.read_text(encoding="utf-8"))
    if len(dev) != 1200 or len({val.refs[i]["image_id"] for i in dev}) != len(dev):
        raise ValueError("Expanded development indices must contain 1200 distinct source images")
    excluded_images |= {val.refs[i]["image_id"] for i in dev}
    pool = [i for i in pool if train.refs[i]["image_id"] not in excluded_images]
    random.Random(args.seed).shuffle(pool)
    if len(pool) < STAGES[-1]:
        raise ValueError(f"Only {len(pool)} eligible training references")
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(SOURCE), "source_sha256": source_sha256,
        "architecture": "frozen shared encoder; train e3/e4 adapters and expert heads",
        "loss": "BCE plus Dice; no size weighting or teacher distillation",
        "route": "fixed historical text and General predicted area",
        "image_size": 384, "mask_threshold": 0.5,
        "stages_cumulative_references": STAGES,
        "batch_size": args.batch_size, "adapter_lr": args.adapter_lr,
        "head_lr": args.head_lr, "seed": args.seed,
        "eligible_train_references": len(pool),
        "train_pool_sha256": hashlib.sha256(json.dumps(pool).encode()).hexdigest(),
        "dev_indices_sha256": hashlib.sha256(DEV_INDICES.read_bytes()).hexdigest(),
        "dev_references": len(dev), "excluded_source_images": len(excluded_images),
        "split_audit": split_audit,
        "historical_checkpoint_exposure": "possible; development-only mechanism test",
        "holdout_used": False, "testB_used": False, "web_untouched": True,
    }
    if args.resume:
        saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for key in ("source_sha256", "train_pool_sha256", "dev_indices_sha256",
                    "batch_size", "adapter_lr", "head_lr", "seed"):
            if saved_manifest[key] != manifest[key]:
                raise ValueError(f"Resume protocol differs: {key}")
        manifest = saved_manifest
    else:
        write_json(output / "manifest.json", manifest)

    model = build_model(device, ("e3", "e4"))
    load_head_joint(SOURCE, model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = list(model.selections.parameters())
    heads = list(model.base.heads.parameters())
    for parameter in adapters + heads:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": adapters, "lr": args.adapter_lr},
        {"params": heads, "lr": args.head_lr},
    ], weight_decay=1e-4)
    router = load_router()
    history_path = output / "history.json"
    if args.resume:
        history = json.loads(history_path.read_text(encoding="utf-8"))
        completed = restore(output / f"stage_{history[-1]['stage']:05d}.pth",
                            model, optimizer, source_sha256, device)
        if completed != history[-1]["stage"]:
            raise ValueError("Resume stage and history disagree")
    else:
        initial = metric_summary(evaluate(model, val, dev, router, device, cache, "val", args.batch_size))
        history = [{"stage": 0, "metrics": initial}]
        write_json(history_path, history)
        torch.save(checkpoint_state(model, optimizer, manifest, 0), output / "stage_00000.pth")
        completed = 0
        print({"stage": 0, "metrics": initial}, flush=True)

    for stage in STAGES:
        if stage > args.max_stage:
            break
        if stage <= completed:
            continue
        ids = pool[completed:stage]
        groups = route_indices(model, train, ids, router, device, cache, "train", args.batch_size)
        model.train()
        model.base.eval()
        model.base.heads.eval()
        model.selections.train()
        losses = []
        start_time = time.monotonic()
        processed = 0
        for branch, branch_ids in groups.items():
            for start in range(0, len(branch_ids), args.batch_size):
                items, images, masks, _ = stack_items(
                    train, branch_ids[start:start + args.batch_size], device
                )
                pooled = pooled_for(cache, items, "train", device)
                with torch.no_grad():
                    _, context = model.forward_general(images, pooled_text=pooled)
                logits = model.forward_expert(context, branch)
                loss = segmentation_loss(logits, masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapters + heads, 5)
                optimizer.step()
                losses.append(float(loss.detach()))
                processed += len(items)
                if processed % 500 < len(items):
                    print({"stage": stage, "processed": processed,
                           "of": len(ids), "branch": branch}, flush=True)
        metrics = metric_summary(evaluate(model, val, dev, router, device, cache, "val", args.batch_size))
        row = {"stage": stage, "additional_references": len(ids),
               "route_counts": {name: len(group) for name, group in groups.items()},
               "mean_train_loss": sum(losses) / len(losses),
               "seconds": round(time.monotonic() - start_time, 1),
               "metrics": metrics}
        torch.save(checkpoint_state(model, optimizer, manifest, stage),
                   output / f"stage_{stage:05d}.pth")
        history.append(row)
        write_json(history_path, history)
        print({"stage": stage, "gIoU": metrics["mean_IoU"],
               "cIoU": metrics["cIoU"],
               "ultra_tiny": metrics["by_bucket"]["ultra_tiny"]["mean_IoU"],
               "tiny": metrics["by_bucket"]["tiny"]["mean_IoU"],
               "false_activation": metrics["false_activation"]}, flush=True)
        completed = stage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="e3e4_budget_20260928")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--adapter-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--max-stage", type=int, choices=STAGES, default=STAGES[-1])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    options = parser.parse_args()
    if options.batch_size < 1:
        parser.error("batch-size must be positive")
    run(options)
