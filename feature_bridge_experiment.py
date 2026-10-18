"""Isolated e2 bridge / edge distillation experiments for shared TG-UNet.

The production application does not import this module. All candidates start
from the same selected detail-adapter checkpoint and retain one image encoder.
"""

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F
from torch import nn

from dataset import RefELDataset
from detail_adapter_model import BRANCH_FILES, DepthwiseDetailAdapter, build_detail_model
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, decode_teacher, evaluate,
    load_or_create_split, load_router, pooled_for, route_indices,
    segmentation_loss, stack_items, teacher_for,
)
from train_text import FormalTextCache


BASE_CHECKPOINT = (ROOT / "results" / "detail_adapter_small_weight_20260924"
                   / "head_detail" / "best.pth")


class BridgeCandidate(nn.Module):
    def __init__(self, base, use_bridge):
        super().__init__()
        self.base = base
        self.use_bridge = use_bridge
        self.e2_bridges = nn.ModuleDict()
        if use_bridge:
            device = next(base.parameters()).device
            self.e2_bridges = nn.ModuleDict({
                name: DepthwiseDetailAdapter(128).to(device)
                for name in BRANCH_FILES
            })

    def forward_general(self, *args, **kwargs):
        return self.base.forward_general(*args, **kwargs)

    def forward_expert(self, context, branch):
        features = self.expert_image_features(context, branch)
        return self.base._decode(self.base.heads[branch], context, features)

    def expert_image_features(self, context, branch):
        features = list(self.base.detail_features(context, branch))
        if self.use_bridge:
            features[2] = self.e2_bridges[branch](features[2])
        return features

    def trainable(self, scope="head_detail", freeze_head_bn=False):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if scope == "head_detail":
            for parameter in self.base.heads.parameters():
                parameter.requires_grad_(True)
            for parameter in self.base.detail_adapters.parameters():
                parameter.requires_grad_(True)
        for parameter in self.e2_bridges.parameters():
            parameter.requires_grad_(True)
        self.base.general.eval()
        if scope == "bridge_only":
            self.base.heads.eval()
            self.base.detail_adapters.eval()
        elif freeze_head_bn:
            for module in self.base.heads.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def save(self, path, metadata):
        state = {
            "format": "feature_bridge_candidate_v1",
            "metadata": metadata,
            "use_bridge": self.use_bridge,
            "heads": {name: head.state_dict() for name, head in self.base.heads.items()},
            "detail_adapters": {
                name: adapter.state_dict()
                for name, adapter in self.base.detail_adapters.items()
            },
            "e2_bridges": {
                name: bridge.state_dict() for name, bridge in self.e2_bridges.items()
            },
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "feature_bridge_candidate_v1":
            raise ValueError("Not a feature-bridge experiment checkpoint")
        if state["use_bridge"] != self.use_bridge:
            raise ValueError("Bridge architecture does not match checkpoint")
        for name in BRANCH_FILES:
            self.base.heads[name].load_state_dict(state["heads"][name])
            self.base.detail_adapters[name].load_state_dict(
                state["detail_adapters"][name])
            if self.use_bridge:
                self.e2_bridges[name].load_state_dict(state["e2_bridges"][name])
        return state["metadata"]


def build_candidate(device, use_bridge):
    base = build_detail_model(ROOT / "checkpoints", device)
    base.load_checkpoint(BASE_CHECKPOINT)
    return BridgeCandidate(base, use_bridge).to(device)


def edge_map(probability):
    horizontal = F.pad((probability[..., 1:] - probability[..., :-1]).abs(),
                       (0, 1, 0, 0))
    vertical = F.pad((probability[..., 1:, :] - probability[..., :-1, :]).abs(),
                     (0, 0, 0, 1))
    return horizontal + vertical


def edge_distillation(student_logits, teacher_logits, ground_truth):
    """Match teacher boundaries only near annotated defects, not every false positive."""
    student = student_logits.sigmoid()
    teacher = teacher_logits.detach().sigmoid()
    roi = F.max_pool2d(ground_truth, kernel_size=15, stride=1, padding=7)
    student_edge = edge_map(student)
    teacher_edge = edge_map(teacher)
    return ((student_edge - teacher_edge).abs() * roi).sum() / roi.sum().clamp_min(1)


def concise(metrics):
    return {key: value for key, value in metrics.items() if key != "rows"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu explicitly for a CPU run")
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "variant": args.variant, "train_scope": args.train_scope,
        "freeze_head_bn": args.freeze_head_bn,
        "device": str(device), "seed": args.seed,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_sha256": hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest(),
        "web_untouched": True, "testB_used": False,
        "note": "Prior expert weights may have seen Val images; diagnostic only.",
    }
    write_json(output / "manifest.json", metadata)

    # The router and training batch must see the same expression for each ref.
    train = RefELDataset(ROOT, split="train", image_size=384, random_text=False)
    val = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    cache = FormalTextCache(ROOT)
    dev_indices, holdout_indices = load_or_create_split(val, args.seed)
    train_pool, excluded, audit = build_disjoint_train_pool(
        train, val, dev_indices, holdout_indices)
    write_json(output / "split_audit.json", {
        "source": str(ROOT / "results" / "single_encoder_joint_20260924" / "split.json"),
        "audit": audit, "excluded_image_ids": sorted(excluded),
        "dev_indices": dev_indices, "holdout_indices": holdout_indices,
    })
    router = load_router()
    model = build_candidate(device, args.variant in ("bridge", "combined", "feature_distill"))
    initial = evaluate(model, val, dev_indices, router, device, cache, "val", args.batch_size)
    write_json(output / "initial_metrics.json", concise(initial))
    model.save(output / "initial.pth", metadata)
    best = initial["cIoU"]
    history = [{"epoch": 0, "metrics": concise(initial)}]
    parameters = model.trainable(args.train_scope, args.freeze_head_bn)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        ids = list(train_pool)
        random.Random(args.seed + epoch).shuffle(ids)
        ids = ids[:min(args.train_samples, len(ids))]
        groups = route_indices(model, train, ids, router, device, cache,
                               "train", args.batch_size)
        model.train()
        model.trainable(args.train_scope, args.freeze_head_bn)
        losses = []
        for branch, branch_ids in groups.items():
            teacher = (teacher_for(branch, device)
                       if args.variant in ("distill", "combined", "feature_distill") else None)
            for start in range(0, len(branch_ids), args.batch_size):
                batch = branch_ids[start:start + args.batch_size]
                items, images, masks, _ = stack_items(train, batch, device)
                pooled = pooled_for(cache, items, "train", device)
                _, context = model.forward_general(images, pooled_text=pooled)
                logits = model.forward_expert(context, branch)
                loss = segmentation_loss(logits, masks)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits, teacher_features = decode_teacher(
                            teacher, images, pooled,
                            return_e2=args.variant == "feature_distill")
                    if args.variant == "feature_distill":
                        student_e2 = model.expert_image_features(context, branch)[2]
                        teacher_e2 = teacher_features[2]
                        alignment = F.smooth_l1_loss(student_e2, teacher_e2)
                        loss = loss + args.distill_weight * alignment / (
                            teacher_e2.std().clamp_min(0.1))
                    else:
                        loss = loss + args.distill_weight * edge_distillation(
                            logits, teacher_logits, masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            del teacher
            if device.type == "cuda":
                torch.cuda.empty_cache()
        metrics = evaluate(model, val, dev_indices, router, device, cache,
                           "val", args.batch_size)
        row = {"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)),
               "routes": {name: len(group) for name, group in groups.items()},
               "metrics": concise(metrics)}
        history.append(row)
        write_json(output / "history.json", history)
        model.save(output / "latest.pth", metadata)
        if metrics["cIoU"] > best:
            best = metrics["cIoU"]
            model.save(output / "best.pth", metadata)
        print({"variant": args.variant, "epoch": epoch,
               "cIoU": metrics["cIoU"], "mean_IoU": metrics["mean_IoU"],
               "ultra_tiny": metrics["by_bucket"]["ultra_tiny"]["mean_IoU"],
               "tiny": metrics["by_bucket"]["tiny"]["mean_IoU"]}, flush=True)
    model.save(output / "completed.pth", metadata)
    write_json(output / "selection.json", {
        "baseline": concise(initial), "best_dev_cIoU": best,
        "best_checkpoint": "best.pth" if (output / "best.pth").exists() else "initial.pth",
        "holdout_not_examined": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("control", "bridge", "distill", "combined", "feature_distill"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--train-scope", choices=("head_detail", "bridge_only"),
                        default="head_detail")
    parser.add_argument("--freeze-head-bn", action="store_true")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--cpu", action="store_true")
    options = parser.parse_args()
    if options.train_scope == "bridge_only" and options.variant not in ("bridge", "combined", "feature_distill"):
        parser.error("bridge_only requires a bridge variant")
    run(options)
