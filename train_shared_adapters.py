"""Train a small feature bridge for each expert on Train, selecting on Val."""

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dataset import RefELDataset
from shared_moe import BRANCH_FILES, SharedFeatures, SharedMoE


ROOT = Path(__file__).resolve().parent


def route_module():
    path = ROOT / "43_final_testB_gate_v2_and_m2_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_router_repair", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_indices(dataset, size, seed):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return indices[:min(size, len(dataset))]


def mask_loss(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    probs = logits.sigmoid()
    intersection = (probs * mask).sum()
    dice = 1 - (2 * intersection + 1) / (probs.sum() + mask.sum() + 1)
    return bce + dice


@torch.no_grad()
def route_indices(model, dataset, indices, route, device):
    model.eval()
    groups = {name: [] for name in BRANCH_FILES}
    for index in indices:
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(device)
        general, _ = model.forward_general(image, texts=[item["text"]])
        ratio = float((general.sigmoid() >= 0.5).float().mean())
        branch = route.route_sample(item["text"], ratio)[0]
        groups[branch].append(index)
    return groups


@torch.no_grad()
def evaluate(model, dataset, groups, device):
    model.eval()
    summary = {}
    total_intersection = total_union = 0
    for branch, indices in groups.items():
        intersection = union = 0
        for index in indices:
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            gt = item["mask"].unsqueeze(0).to(device) > 0.5
            _, context = model.forward_general(image, texts=[item["text"]])
            pred = model.forward_expert(context, branch).sigmoid() >= 0.5
            intersection += int((pred & gt).sum().item())
            union += int((pred | gt).sum().item())
        summary[branch] = {"n": len(indices), "cIoU": intersection / max(1, union)}
        total_intersection += intersection
        total_union += union
    summary["all"] = {"n": sum(len(indices) for indices in groups.values()),
                      "cIoU": total_intersection / max(1, total_union)}
    return summary


@torch.no_grad()
def teacher_context(teacher, image, pooled_text):
    e0 = teacher.stem(image)
    e1 = teacher.layer1(teacher.maxpool(e0))
    e2 = teacher.layer2(e1)
    e3 = teacher.layer3(e2)
    e4 = teacher.layer4(e3)
    return SharedFeatures((e0, e1, e2, e3, e4), pooled_text, image.shape[-2:])


class PooledTextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Identity()

    def forward(self, *args, **kwargs):
        raise RuntimeError("Teacher requires pooled_text")


def main(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    route = route_module()
    model = SharedMoE.from_checkpoints(
        ROOT / "checkpoints", device,
        expert_backbone=args.expert_backbone,
    )
    if not args.from_original_heads:
        model.load_heads(args.start_heads)
    model.enable_adapters()
    for parameter in model.general.parameters():
        parameter.requires_grad_(False)
    if model.expert_encoder is not None:
        for parameter in model.expert_encoder.parameters():
            parameter.requires_grad_(False)
    for parameter in model.heads.parameters():
        parameter.requires_grad_(False)
    active_branches = args.adapt_branches.split(",") if args.adapt_branches else list(BRANCH_FILES)
    if not active_branches or any(branch not in BRANCH_FILES for branch in active_branches):
        raise ValueError("adapt-branches must list known expert branches")
    if args.train_heads:
        for branch in active_branches:
            for parameter in model.heads[branch].parameters():
                parameter.requires_grad_(True)
    train = RefELDataset(ROOT, split="train", image_size=384, random_text=True)
    val = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    val_groups = route_indices(model, val, choose_indices(val, args.val_samples, args.seed), route, device)
    if args.expert_backbone:
        folder = f"shared_moe_{args.expert_backbone}_repair"
    else:
        folder = "shared_moe_repair_joint" if args.train_heads else "shared_moe_repair"
    output = ROOT / "results" / folder
    output.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(model, val, val_groups, device)
    print(f"Initial validation: {baseline}", flush=True)
    best = baseline["all"]["cIoU"]
    history = [{"epoch": 0, "val": baseline, "train_loss": None}]
    groups_to_train = [{"params": [parameter for branch in active_branches
                                     for parameter in model.adapters[branch].parameters()],
                        "lr": args.lr}]
    if args.train_heads:
        groups_to_train.append({"params": [parameter for branch in active_branches
                                           for parameter in model.heads[branch].parameters()],
                                "lr": args.head_lr})
    optimizer = torch.optim.AdamW(groups_to_train, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        candidate_ids = choose_indices(train, args.train_samples, args.seed + epoch)
        groups = route_indices(model, train, candidate_ids, route, device)
        model.train()
        model.general.eval()
        model.heads.eval()
        if model.expert_encoder is not None:
            model.expert_encoder.eval()
        total_loss = total_steps = 0
        for branch, indices in groups.items():
            if not indices or branch not in active_branches:
                continue
            teacher = route.load_seg_model(ROOT / "checkpoints" / BRANCH_FILES[branch], device)
            teacher.text_encoder = PooledTextOnly()
            teacher.eval()
            for index in indices:
                item = train[index]
                image = item["image"].unsqueeze(0).to(device)
                mask = item["mask"].unsqueeze(0).to(device).float()
                with torch.no_grad():
                    _, context = model.forward_general(image, texts=[item["text"]])
                    expert_context = teacher_context(teacher, image, context.pooled_text)
                    teacher_logits = SharedMoE._decode(teacher, expert_context)
                adapted = model.expert_features(context, branch)
                student_logits = SharedMoE._decode(model.heads[branch], context, adapted)
                supervised = mask_loss(student_logits, mask)
                imitation = F.binary_cross_entropy_with_logits(
                    student_logits, teacher_logits.sigmoid()
                )
                feature_loss = sum(
                    F.smooth_l1_loss(student, target) /
                    target.detach().std().clamp_min(0.1)
                    for student, target in zip(adapted, expert_context.image)
                ) / len(adapted)
                loss = supervised + args.distill_weight * imitation + args.feature_weight * feature_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_steps += 1
                total_loss += float(loss.detach())
                if total_steps % 50 == 0:
                    print(f"Epoch {epoch} {total_steps}/{len(candidate_ids)} "
                          f"loss={total_loss/total_steps:.4f}", flush=True)
            del teacher
            if device.type == "cuda":
                torch.cuda.empty_cache()
        val_score = evaluate(model, val, val_groups, device)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_steps),
               "train_routes": {branch: len(indices) for branch, indices in groups.items()},
               "val": val_score}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        model.save_repair(output / "latest_repair.pth")
        if val_score["all"]["cIoU"] > best:
            best = val_score["all"]["cIoU"]
            model.save_repair(output / "best_repair.pth")
        print(row, flush=True)
    print(f"Best validation cIoU: {best:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-heads", type=Path, default=ROOT / "results" / "shared_moe_training" / "best_heads.pth")
    parser.add_argument("--from-original-heads", action="store_true")
    parser.add_argument("--expert-backbone", choices=tuple(BRANCH_FILES))
    parser.add_argument("--adapt-branches", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--val-samples", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--train-heads", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--feature-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    if min(options.epochs, options.train_samples, options.val_samples) < 1:
        parser.error("epochs and sample counts must be positive")
    main(options)
