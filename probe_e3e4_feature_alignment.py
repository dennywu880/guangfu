"""Frozen-head e3/e4 alignment probe for the shared TG-UNet encoder.

This diagnostic contrasts the same independent expert head with its own
features, unmodified shared features, and two trained mappings of shared
e3/e4 features. It does not modify any deployed model or source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from dataset import RefELDataset
from detail_adapter_model import DepthwiseDetailAdapter, build_textless_unet
from shared_moe import BRANCH_FILES, SharedMoE, _state
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, load_or_create_split, load_router,
    pooled_for, route_indices, stack_items, teacher_for,
)
from train_text import FormalTextCache


GENERAL = ROOT / "checkpoints" / "text_gate_epoch13_profile_fixed.pth"
PREVIOUS_DEV = ROOT / "results" / "expanded_routing_audit_20260926" / "sample_indices.json"
LEVEL_CHANNELS = {"e3": 256, "e4": 512}


class ChannelMap(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.delta = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, feature):
        return feature + self.delta(feature)


class AlignmentMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = nn.ModuleDict({
            branch: nn.ModuleDict({
                level: ChannelMap(channels)
                for level, channels in LEVEL_CHANNELS.items()
            }) for branch in BRANCH_FILES
        })
        self.spatial = nn.ModuleDict({
            branch: nn.ModuleDict({
                level: DepthwiseDetailAdapter(channels)
                for level, channels in LEVEL_CHANNELS.items()
            }) for branch in BRANCH_FILES
        })

    def adapted(self, features, branch, kind):
        mapped = list(features)
        for level, index in (("e3", 3), ("e4", 4)):
            mapped[index] = getattr(self, kind)[branch][level](mapped[index])
        return tuple(mapped)


def encode(model, image):
    e0 = model.stem(image)
    e1 = model.layer1(model.maxpool(e0))
    e2 = model.layer2(e1)
    e3 = model.layer3(e2)
    e4 = model.layer4(e3)
    return e0, e1, e2, e3, e4


def normalized_error(candidate, target):
    return (candidate - target).square().mean() / target.detach().var().clamp_min(0.1)


def choose_eval_images(val, dev, holdout, count, seed):
    previous = json.loads(PREVIOUS_DEV.read_text(encoding="utf-8"))
    excluded = {
        val.refs[index]["image_id"] for index in set(dev + holdout + previous)
    }
    indices = list(range(len(val)))
    random.Random(seed).shuffle(indices)
    chosen = []
    seen = set(excluded)
    for index in indices:
        image_id = val.refs[index]["image_id"]
        if image_id in seen:
            continue
        chosen.append(index)
        seen.add(image_id)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError("Insufficient distinct evaluation source images")
    return chosen, seen


def bucket(gt):
    ratio = float(gt.float().mean().item())
    if ratio == 0:
        return "no_target"
    if ratio < 0.001:
        return "ultra_tiny"
    if ratio < 0.006:
        return "tiny"
    return "larger"


def binary_iou(logits, gt):
    pred = logits.sigmoid() >= 0.5
    intersection = int((pred & gt).sum().item())
    union = int((pred | gt).sum().item())
    return intersection / union if union else 1.0, intersection, union, not bool(pred.any().item())


def summarize(rows):
    result = {}
    for variant in ("shared", "channel", "spatial", "own_deep", "own_all"):
        entries = [row for row in rows if row["variant"] == variant]
        positives = [row for row in entries if row["bucket"] != "no_target"]
        result[variant] = {
            "all_N": len(entries), "positive_N": len(positives),
            "positive_mean_IoU": sum(row["iou"] for row in positives) / len(positives),
            "cIoU": sum(row["intersection"] for row in entries)
            / max(1, sum(row["union"] for row in entries)),
            "no_target_false_activations": sum(
                not row["pred_empty"] for row in entries if row["bucket"] == "no_target"
            ),
            "by_bucket": {
                name: {
                    "N": len(group),
                    "positive_mean_IoU": sum(row["iou"] for row in group) / len(group)
                    if group else None,
                }
                for name in ("ultra_tiny", "tiny", "larger")
                for group in [[row for row in positives if row["bucket"] == name]]
            },
            "by_branch": {
                name: {
                    "N": len(group),
                    "positive_mean_IoU": sum(row["iou"] for row in group) / len(group)
                    if group else None,
                }
                for name in BRANCH_FILES
                for group in [[row for row in positives if row["branch"] == name]]
            },
        }
    return result


@torch.inference_mode()
def evaluate(shared, teachers, maps, val, indices, cache, router, device):
    shared.eval()
    maps.eval()
    rows = []
    feature_errors = defaultdict(list)
    for position, index in enumerate(indices, 1):
        item = val[index]
        image = item["image"].unsqueeze(0).to(device)
        gt = item["mask"].unsqueeze(0).to(device).bool()
        pooled = pooled_for(cache, [item], "val", device)
        general_logits, context = shared.forward_general(image, pooled_text=pooled)
        pred_ratio = float((general_logits.sigmoid() >= 0.5).float().mean().item())
        branch = router.route_sample(item["text"], pred_ratio)[0]
        own = encode(teachers[branch], image)
        mixed = context.image[:3] + own[3:]
        variants = {
            "shared": context.image,
            "channel": maps.adapted(context.image, branch, "channel"),
            "spatial": maps.adapted(context.image, branch, "spatial"),
            "own_deep": mixed,
            "own_all": own,
        }
        for variant, features in variants.items():
            logits = shared._decode(shared.heads[branch], context, features)
            iou, intersection, union, empty = binary_iou(logits, gt)
            rows.append({
                "ref_id": int(item["ref_id"]), "image_id": int(item["image_id"]),
                "branch": branch, "bucket": bucket(gt), "variant": variant,
                "iou": iou, "intersection": intersection, "union": union,
                "pred_empty": empty,
            })
        for variant in ("shared", "channel", "spatial"):
            feature_errors[variant].append(sum(
                float(normalized_error(variants[variant][level], own[level]).item())
                for level in (3, 4)
            ) / 2)
        if position % 100 == 0 or position == len(indices):
            print(f"Evaluated {position}/{len(indices)}", flush=True)
    return {
        "metrics": summarize(rows),
        "normalized_e3e4_feature_error": {
            name: sum(values) / len(values) for name, values in feature_errors.items()
        },
    }, rows


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This probe requires CUDA")
    output = ROOT / "results" / args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    val = RefELDataset(ROOT, "val", 384, False)
    train = RefELDataset(ROOT, "train", 384, False)
    dev, holdout = load_or_create_split(val, args.seed)
    eval_indices, excluded_images = choose_eval_images(
        val, dev, holdout, args.eval_count, args.seed + 1
    )
    train_pool, _, _ = build_disjoint_train_pool(train, val, dev, holdout)
    train_pool = [
        index for index in train_pool
        if train.refs[index]["image_id"] not in excluded_images
    ]
    cache = FormalTextCache(ROOT)
    general = build_textless_unet(_state(GENERAL), device)
    shared = SharedMoE.from_checkpoints(ROOT / "checkpoints", device, general=general).eval()
    teachers = {branch: teacher_for(branch, device) for branch in BRANCH_FILES}
    maps = AlignmentMaps().to(device)
    for parameter in shared.parameters():
        parameter.requires_grad_(False)
    for teacher in teachers.values():
        teacher.eval()
    router = load_router()
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "general_sha256": hashlib.sha256(GENERAL.read_bytes()).hexdigest(),
        "teacher_sha256": {
            branch: hashlib.sha256((ROOT / "checkpoints" / name).read_bytes()).hexdigest()
            for branch, name in BRANCH_FILES.items()
        },
        "train_samples_per_epoch": args.train_samples,
        "eval_images": len(eval_indices), "seed": args.seed,
        "epochs": args.epochs, "lr": args.lr,
        "eval_images_excluded_from_new_training": True,
        "historical_checkpoint_exposure_unknown": True,
        "testB_used": False, "web_untouched": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "eval_indices.json").write_text(
        json.dumps(eval_indices, indent=2), encoding="utf-8"
    )
    torch.save({"channel": maps.channel.state_dict(),
                "spatial": maps.spatial.state_dict(),
                "metadata": metadata}, output / "initial.pth")
    initial, rows = evaluate(shared, teachers, maps, val, eval_indices,
                             cache, router, device)
    (output / "initial.json").write_text(
        json.dumps(initial, indent=2), encoding="utf-8"
    )
    print({"epoch": 0, "metrics": initial}, flush=True)
    optimizer = torch.optim.AdamW(maps.parameters(), lr=args.lr, weight_decay=1e-4)
    history = [{"epoch": 0, "metrics": initial}]
    for epoch in range(1, args.epochs + 1):
        sampled = list(train_pool)
        random.Random(args.seed + epoch).shuffle(sampled)
        sampled = sampled[:min(args.train_samples, len(sampled))]
        groups = route_indices(shared, train, sampled, router, device,
                               cache, "train", args.batch_size)
        maps.train()
        losses = []
        for branch, indices in groups.items():
            for start in range(0, len(indices), args.batch_size):
                ids = indices[start:start + args.batch_size]
                items, images, _, _ = stack_items(train, ids, device)
                pooled = pooled_for(cache, items, "train", device)
                with torch.no_grad():
                    _, context = shared.forward_general(images, pooled_text=pooled)
                    own = encode(teachers[branch], images)
                loss = sum(
                    normalized_error(maps.adapted(context.image, branch, kind)[level], own[level])
                    for kind in ("channel", "spatial") for level in (3, 4)
                ) / 4
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            print({"epoch": epoch, "branch": branch, "samples": len(indices)}, flush=True)
        result, rows = evaluate(shared, teachers, maps, val, eval_indices,
                                cache, router, device)
        entry = {
            "epoch": epoch, "mean_alignment_loss": sum(losses) / len(losses),
            "route_counts": {name: len(indices) for name, indices in groups.items()},
            "metrics": result,
        }
        history.append(entry)
        torch.save({"channel": maps.channel.state_dict(),
                    "spatial": maps.spatial.state_dict(),
                    "metadata": metadata}, output / f"epoch_{epoch:02d}.pth")
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print({"epoch": epoch, "metrics": result}, flush=True)
    (output / "final_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-count", type=int, default=600)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260926)
    run(parser.parse_args())
