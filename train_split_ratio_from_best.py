"""Compare encoder sharing boundaries from the existing best shared checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from dataset import RefELDataset
from multilevel_selection import build_model
from shared_moe import BRANCH_FILES
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, load_or_create_split, load_router,
    pooled_for, route_indices, segmentation_loss, size_bucket, stack_items,
)
from train_shared_layer4_joint import DEVELOPMENT_INDICES, SOURCE as LAYER4_SOURCE
from train_shared_layer4_joint import load as load_layer4_checkpoint
from train_text import FormalTextCache


SOURCE = ROOT / "results" / "layer4_joint_20260926" / "epoch_02.pth"
BASE_CHECKPOINT = ROOT / "results" / "head_joint_e3_e4_20260925" / "best.pth"
SPLITS = ("shared", "e4", "e3e4")


class SplitCandidate(nn.Module):
    """Duplicate only the selected encoder tail for each expert."""

    def __init__(self, shared, split):
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.shared = shared
        self.split = split
        levels = {"shared": (), "e4": ("e4",), "e3e4": ("e3", "e4")}[split]
        source = shared.base.general
        self.branches = nn.ModuleDict({
            name: nn.ModuleDict({level: deepcopy(getattr(source, "layer" + level[1:]))
                                 for level in levels})
            for name in BRANCH_FILES
        })

    def forward_general(self, *args, **kwargs):
        return self.shared.forward_general(*args, **kwargs)

    def forward_expert(self, context, branch):
        features = list(self.shared.base.detail_features(context, branch))
        if self.split == "shared":
            for level in ("e3", "e4"):
                features[int(level[1:])] = self.shared.selections[branch][level](
                    features[int(level[1:])])
        elif self.split == "e4":
            raw_e3 = features[3]
            features[4] = self.branches[branch]["e4"](raw_e3)
            features[3] = raw_e3
            features[3] = self.shared.selections[branch]["e3"](features[3])
            features[4] = self.shared.selections[branch]["e4"](features[4])
        else:
            raw_e3 = self.branches[branch]["e3"](features[2])
            raw_e4 = self.branches[branch]["e4"](raw_e3)
            features[3] = raw_e3
            features[4] = raw_e4
            features[3] = self.shared.selections[branch]["e3"](features[3])
            features[4] = self.shared.selections[branch]["e4"](features[4])
        return self.shared.base._decode(
            self.shared.base.heads[branch], context, features
        )


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def state_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_parameters(model):
    general = sum(p.numel() for n, p in model.shared.base.general.named_parameters()
                  if "text_encoder" not in n)
    general_text = sum(p.numel() for n, p in model.shared.base.general.named_parameters()
                       if "text_encoder" in n)
    all_heads = sum(p.numel() for p in model.shared.base.heads.parameters())
    all_selectors = sum(p.numel() for p in model.shared.selections.parameters())
    branch_total = sum(p.numel() for p in model.branches.parameters())
    expert_names = tuple(BRANCH_FILES)
    selected_head = sum(p.numel() for p in model.shared.base.heads[expert_names[0]].parameters())
    selected_selector = sum(p.numel() for p in model.shared.selections[expert_names[0]].parameters())
    selected_branch = sum(p.numel() for p in model.branches[expert_names[0]].parameters())
    return {
        "total_model_parameters_with_cached_text_features": general + all_heads + all_selectors + branch_total,
        "active_one_route_parameters_with_cached_text_features": general + selected_head + selected_selector + selected_branch,
        "text_encoder_parameters_stored_in_model": general_text,
        "branch_specific_total": branch_total,
        "all_expert_heads_and_selectors": all_heads + all_selectors,
    }


@torch.inference_mode()
def assert_split_initialization_matches(model, images, pooled):
    """All split candidates must reproduce the source path before training."""
    model.eval()
    _, context = model.forward_general(images, pooled_text=pooled)
    for branch in BRANCH_FILES:
        reference = model.shared.forward_expert(context, branch)
        candidate = model.forward_expert(context, branch)
        if not torch.allclose(reference, candidate, rtol=1e-5, atol=1e-6):
            error = float((reference - candidate).abs().max())
            raise RuntimeError(
                f"Split {model.split} changed initial {branch} logits; max_abs_error={error}"
            )


def metrics_for_rows(rows):
    def summarize(group):
        if not group:
            return {"n": 0, "mean_IoU": None, "cIoU": None, "false_activation": 0}
        return {
            "n": len(group),
            "mean_IoU": sum(row["iou"] for row in group) / len(group),
            "cIoU": sum(row["intersection"] for row in group) /
                    max(1, sum(row["union"] for row in group)),
            "false_activation": sum(row["gt_empty"] and not row["pred_empty"]
                                    for row in group),
        }
    result = summarize(rows)
    result["by_bucket"] = {
        name: summarize([row for row in rows if row["bucket"] == name])
        for name in ("no_target", "ultra_tiny", "tiny", "larger")
    }
    result["by_branch"] = {
        name: summarize([row for row in rows if row["route"] == name])
        for name in BRANCH_FILES
    }
    positive = [row for row in rows if not row["gt_empty"]]
    if positive and "expert_ious" in positive[0]:
        result["positive_mean_IoU"] = sum(row["iou"] for row in positive) / len(positive)
        result["oracle_positive_mean_IoU"] = sum(
            max(row["expert_ious"].values()) for row in positive
        ) / len(positive)
        result["per_expert_positive_mean_IoU"] = {
            name: sum(row["expert_ious"][name] for row in positive) / len(positive)
            for name in BRANCH_FILES
        }
        result["oracle_best_expert_counts"] = {
            name: sum(max(row["expert_ious"], key=row["expert_ious"].get) == name
                      for row in positive)
            for name in BRANCH_FILES
        }
    return result


@torch.inference_mode()
def evaluate_all(model, dataset, indices, router, device, cache, split_name, batch_size):
    model.eval()
    rows = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        items, images, masks, texts = stack_items(dataset, ids, device)
        pooled = pooled_for(cache, items, split_name, device)
        general, context = model.forward_general(images, pooled_text=pooled)
        predicted_area = (general.sigmoid() >= 0.5).float().flatten(1).mean(1).tolist()
        routes = [router.route_sample(text, float(area))[0]
                  for text, area in zip(texts, predicted_area)]
        logits_by_expert = {
            name: model.forward_expert(context, name) for name in BRANCH_FILES
        }
        predictions = {name: logits.sigmoid() >= 0.5
                       for name, logits in logits_by_expert.items()}
        for offset, (item, gt, route) in enumerate(zip(items, masks, routes)):
            gt = gt > 0.5
            expert_ious = {}
            for name in BRANCH_FILES:
                pred = predictions[name][offset]
                inter = int((pred & gt).sum().item())
                union = int((pred | gt).sum().item())
                expert_ious[name] = inter / union if union else 1.0
            pred = predictions[route][offset]
            inter = int((pred & gt).sum().item())
            union = int((pred | gt).sum().item())
            rows.append({
                "ref_id": int(item["ref_id"]),
                "image_id": str(item["image_id"]),
                "route": route,
                "bucket": size_bucket(gt),
                "intersection": inter,
                "union": union,
                "iou": inter / union if union else 1.0,
                "gt_empty": not bool(gt.any().item()),
                "pred_empty": not bool(pred.any().item()),
                "expert_ious": expert_ious,
            })
    return metrics_for_rows(rows), rows


def save_checkpoint(path, model, metadata, epoch):
    shared = model.shared
    payload = {
        "format": "split_ratio_from_existing_v1",
        "source_sha256": metadata["source_sha256"],
        "split": model.split,
        "epoch": epoch,
        "layer4": shared.base.general.layer4.state_dict(),
        "heads": {name: mod.state_dict() for name, mod in shared.base.heads.items()},
        "selections": {name: {lev: mod.state_dict() for lev, mod in levels.items()}
                       for name, levels in shared.selections.items()},
        "branches": model.branches.state_dict(),
    }
    torch.save(payload, path)


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for explicit CPU execution")
    device = torch.device("cpu" if args.cpu else "cuda")
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    if not SOURCE.is_file() or not BASE_CHECKPOINT.is_file() or not DEVELOPMENT_INDICES.is_file():
        raise FileNotFoundError("Required source, base, or fixed development indices are missing")

    train = RefELDataset(ROOT, "train", 384, False)
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    dev = json.loads(DEVELOPMENT_INDICES.read_text(encoding="utf-8"))
    old_dev, holdout = load_or_create_split(val, args.seed)
    pool, excluded, split_audit = build_disjoint_train_pool(train, val, old_dev, holdout)
    dev_images = {val.refs[i]["image_id"] for i in dev}
    excluded |= dev_images
    pool = [i for i in pool if train.refs[i]["image_id"] not in excluded]
    if len(dev) != 1200 or len(dev_images) != 1200:
        raise ValueError("The fixed development set must be 1,200 image-distinct references")
    if args.smoke_samples:
        dev = dev[:args.smoke_samples]
    if len(pool) < args.train_samples:
        raise ValueError(f"Eligible training pool too small: {len(pool)}")
    router = load_router()

    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(SOURCE),
        "source_sha256": state_hash(SOURCE),
        "source_description": "best existing lightweight checkpoint: shared encoder plus e3/e4 selectors and heads, after Layer4 joint epoch 2",
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_sha256": state_hash(BASE_CHECKPOINT),
        "architectures": list(SPLITS),
        "split_definition": {"shared": "encoder e0-e4 shared; expert selectors/heads", "e4": "e0-e3 shared; e4 branch-specific", "e3e4": "e0-e2 shared; e3/e4 branch-specific"},
        "input_size": 384,
        "route": "fixed historical text and General predicted-area rule",
        "mask_threshold": 0.5,
        "presence_gate": "not included in training evaluator; fixed General path retained",
        "loss": "BCE plus Dice, unweighted",
        "epochs": args.epochs,
        "train_samples_per_epoch": args.train_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "branch_lr": args.branch_lr,
        "selector_lr": args.selector_lr,
        "head_lr": args.head_lr,
        "dev_indices_sha256": hashlib.sha256(DEVELOPMENT_INDICES.read_bytes()).hexdigest(),
        "dev_references": len(dev),
        "smoke_test": bool(args.smoke_samples),
        "train_pool_references": len(pool),
        "split_audit": split_audit,
        "historical_checkpoint_exposure": "possible; this is development evidence, not a blind generalization test",
        "test_sets_read": False,
        "deployment_changed": False,
    }
    write_json(output / "manifest.json", metadata)

    history = {}
    for split in SPLITS:
        print(f"\n=== Initialize {split} from the same best checkpoint ===", flush=True)
        shared = build_model(device, ("e3", "e4"))
        load_layer4_checkpoint(SOURCE, shared)
        model = SplitCandidate(shared, split).to(device)
        counts = count_parameters(model)
        model.train()
        for p in model.parameters():
            p.requires_grad_(False)
        # Keep the General prediction/router path identical in every candidate.
        for module in model.shared.selections.parameters():
            module.requires_grad_(True)
        for module in model.shared.base.heads.parameters():
            module.requires_grad_(True)
        for module in model.branches.parameters():
            module.requires_grad_(True)
        selection_params = list(model.shared.selections.parameters())
        head_params = list(model.shared.base.heads.parameters())
        branch_params = list(model.branches.parameters())
        optimizer_groups = []
        if branch_params:
            optimizer_groups.append({"params": branch_params, "lr": args.branch_lr})
        optimizer_groups.extend([
            {"params": selection_params, "lr": args.selector_lr},
            {"params": head_params, "lr": args.head_lr},
        ])
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)
        split_dir = output / split
        split_dir.mkdir()
        initial_metrics, initial_rows = evaluate_all(
            model, val, dev, router, device, cache, "val", args.eval_batch_size
        )
        sample_items, sample_images, _, _ = stack_items(val, dev[:min(2, len(dev))], device)
        sample_pooled = pooled_for(cache, sample_items, "val", device)
        assert_split_initialization_matches(model, sample_images, sample_pooled)
        history[split] = [{"epoch": 0, "metrics": initial_metrics,
                           "parameter_counts": counts}]
        write_json(split_dir / "epoch_00_metrics.json", {
            "metrics": initial_metrics, "rows": initial_rows
        })
        save_checkpoint(split_dir / "epoch_00.pth", model, metadata, 0)
        print({"split": split, "epoch": 0, "metrics": initial_metrics,
               "parameters": counts}, flush=True)

        for epoch in range(1, args.epochs + 1):
            indices = list(pool)
            random.Random(args.seed + epoch).shuffle(indices)
            indices = indices[:args.train_samples]
            groups = route_indices(model, train, indices, router, device, cache,
                                   "train", args.batch_size)
            model.train()
            model.shared.base.eval()
            model.shared.base.general.eval()
            model.shared.base.heads.eval()
            model.shared.selections.train()
            if model.branches:
                model.branches.eval()  # freeze BN running statistics; gradients remain enabled
            losses = []
            for branch, branch_indices in groups.items():
                for start in range(0, len(branch_indices), args.batch_size):
                    ids = branch_indices[start:start + args.batch_size]
                    items, images, masks, _ = stack_items(train, ids, device)
                    pooled = pooled_for(cache, items, "train", device)
                    with torch.no_grad():
                        _, context = model.forward_general(images, pooled_text=pooled)
                    logits = model.forward_expert(context, branch)
                    loss = segmentation_loss(logits, masks)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        branch_params + selection_params + head_params, 5
                    )
                    optimizer.step()
                    losses.append(float(loss.detach()))
                print({"split": split, "epoch": epoch, "branch": branch,
                       "samples": len(branch_indices)}, flush=True)
            metrics, rows = evaluate_all(
                model, val, dev, router, device, cache, "val", args.eval_batch_size
            )
            row = {"epoch": epoch, "metrics": metrics,
                   "parameter_counts": counts,
                   "mean_train_loss": sum(losses) / max(1, len(losses)),
                   "route_counts": {name: len(ids) for name, ids in groups.items()}}
            history[split].append(row)
            write_json(split_dir / f"epoch_{epoch:02d}_metrics.json", {
                "metrics": metrics, "rows": rows
            })
            save_checkpoint(split_dir / f"epoch_{epoch:02d}.pth", model, metadata, epoch)
            print({"split": split, "epoch": epoch, "gIoU": metrics["mean_IoU"],
                   "cIoU": metrics["cIoU"],
                   "ultra_tiny": metrics["by_bucket"]["ultra_tiny"]["mean_IoU"],
                   "tiny": metrics["by_bucket"]["tiny"]["mean_IoU"],
                   "oracle": metrics.get("oracle_positive_mean_IoU"),
                   "false_activation": metrics["false_activation"]}, flush=True)
        write_json(output / "history.json", history)
        del model, shared, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(output / "completion.json", {
        "completed": True,
        "source_checkpoint_unchanged": state_hash(SOURCE) == metadata["source_sha256"],
        "deployment_changed": False,
        "test_sets_read": False,
        "next": "Review development tradeoffs; no candidate is promoted without an independent matched evaluation.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="split_ratio_existing_best_20261001")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--branch-lr", type=float, default=1e-5)
    parser.add_argument("--selector-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--smoke-samples", type=int, default=0,
                        help="Run a tiny development/train smoke test; 0 uses all 1,200 dev references")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.train_samples, args.batch_size, args.eval_batch_size) < 1:
        parser.error("epochs, sample counts, and batch sizes must be positive")
    if args.smoke_samples < 0:
        parser.error("smoke-samples cannot be negative")
    run(args)
