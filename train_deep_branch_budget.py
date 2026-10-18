"""Matched-budget training of expert-specific e3/e4 blocks on historical weights."""

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
from deep_branch_selection import build_deep_branch
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, evaluate, load_or_create_split,
    load_router, pooled_for, route_indices, segmentation_loss, stack_items,
)
from train_e3e4_budget import DEV_INDICES, SOURCE, STAGES, metric_summary, write_json
from train_text import FormalTextCache


def checkpoint_state(model, optimizer, source_sha256, stage):
    return {
        "format": "deep_branch_budget_v1",
        "source_sha256": source_sha256,
        "stage": stage,
        "branches": model.branches.state_dict(),
        "heads": model.shared.base.heads.state_dict(),
        "selections": model.shared.selections.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def restore(path, model, optimizer, source_sha256, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != "deep_branch_budget_v1" or state.get("source_sha256") != source_sha256:
        raise ValueError("Incompatible deep-branch checkpoint")
    model.branches.load_state_dict(state["branches"], strict=True)
    model.shared.base.heads.load_state_dict(state["heads"], strict=True)
    model.shared.selections.load_state_dict(state["selections"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state["stage"])


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for an explicit CPU run")
    output = ROOT / "results" / args.output
    if output.exists() != args.resume:
        raise FileExistsError(output) if output.exists() else FileNotFoundError(output)
    if not args.resume:
        output.mkdir(parents=True)

    train = RefELDataset(ROOT, "train", 384, False)
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    old_dev, holdout = load_or_create_split(val, args.seed)
    pool, excluded, split_audit = build_disjoint_train_pool(train, val, old_dev, holdout)
    dev = json.loads(DEV_INDICES.read_text(encoding="utf-8"))
    excluded |= {val.refs[i]["image_id"] for i in dev}
    pool = [i for i in pool if train.refs[i]["image_id"] not in excluded]
    random.Random(args.seed).shuffle(pool)
    reference_manifest = json.loads((ROOT / "results" / "e3e4_budget_20260928" / "manifest.json")
                                    .read_text(encoding="utf-8"))
    pool_sha256 = hashlib.sha256(json.dumps(pool).encode()).hexdigest()
    if pool_sha256 != reference_manifest["train_pool_sha256"]:
        raise ValueError("Deep-branch train order differs from adapter experiment")
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(SOURCE), "source_sha256": source_sha256,
        "architecture": "shared General encoder e0-e2; per-expert trainable e3/e4 plus adapters and heads",
        "initialization": "deep expert blocks copied from frozen General e3/e4",
        "loss": "BCE plus Dice; no size weighting or teacher distillation",
        "route": "fixed historical text and General predicted area",
        "image_size": 384, "mask_threshold": 0.5,
        "stages_cumulative_references": STAGES, "batch_size": args.batch_size,
        "branch_lr": args.branch_lr, "adapter_lr": args.adapter_lr,
        "head_lr": args.head_lr, "seed": args.seed,
        "branch_only": args.branch_only,
        "eligible_train_references": len(pool),
        "train_pool_sha256": pool_sha256,
        "dev_indices_sha256": hashlib.sha256(DEV_INDICES.read_bytes()).hexdigest(),
        "dev_references": len(dev), "excluded_source_images": len(excluded),
        "split_audit": split_audit,
        "historical_checkpoint_exposure": "possible; development-only mechanism test",
        "holdout_used": False, "testB_used": False, "web_untouched": True,
    }
    if args.resume:
        saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for key in ("source_sha256", "train_pool_sha256", "dev_indices_sha256",
                    "batch_size", "branch_lr", "adapter_lr", "head_lr", "seed",
                    "branch_only"):
            if saved[key] != manifest[key]:
                raise ValueError(f"Resume protocol differs: {key}")
        manifest = saved
    else:
        write_json(output / "manifest.json", manifest)

    model = build_deep_branch(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    branches = list(model.branches.parameters())
    adapters = list(model.shared.selections.parameters())
    heads = list(model.shared.base.heads.parameters())
    trainable = branches if args.branch_only else branches + adapters + heads
    for parameter in trainable:
        parameter.requires_grad_(True)
    groups = [{"params": branches, "lr": args.branch_lr}]
    if not args.branch_only:
        groups.extend([
            {"params": adapters, "lr": args.adapter_lr},
            {"params": heads, "lr": args.head_lr},
        ])
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    manifest["trainable_parameters"] = sum(p.numel() for p in trainable)
    manifest["total_parameters_excluding_bert"] = sum(p.numel() for p in model.parameters())
    manifest["active_parameters_excluding_bert"] = (
        sum(p.numel() for p in model.shared.base.general.parameters())
        + sum(p.numel() for p in model.shared.base.detail_adapters.parameters()) // 3
        + sum(p.numel() for p in model.shared.base.heads.parameters()) // 3
        + sum(p.numel() for p in model.shared.selections.parameters()) // 3
        + sum(p.numel() for p in model.branches.parameters()) // 3
    )
    if not args.resume:
        write_json(output / "manifest.json", manifest)
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
        reference_initial = json.loads((ROOT / "results" / "e3e4_budget_20260928" / "history.json")
                                       .read_text(encoding="utf-8"))[0]["metrics"]
        if abs(initial["mean_IoU"] - reference_initial["mean_IoU"]) > 1e-6:
            raise RuntimeError("Deep branch initialization did not reproduce adapter baseline")
        history = [{"stage": 0, "metrics": initial}]
        write_json(history_path, history)
        torch.save(checkpoint_state(model, optimizer, source_sha256, 0),
                   output / "stage_00000.pth")
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
        model.shared.eval()
        model.branches.eval()
        model.shared.selections.train()
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
                torch.nn.utils.clip_grad_norm_(trainable, 5)
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
        torch.save(checkpoint_state(model, optimizer, source_sha256, stage),
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
    parser.add_argument("--output", default="deep_branch_budget_20260928")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--branch-lr", type=float, default=2e-5)
    parser.add_argument("--adapter-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--max-stage", type=int, choices=STAGES, default=STAGES[-1])
    parser.add_argument("--branch-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    options = parser.parse_args()
    if options.batch_size < 1:
        parser.error("batch-size must be positive")
    run(options)
