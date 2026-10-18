"""Train the shallow-detail shared TG-UNet experiment in recoverable stages.

The web application never imports this file.  Each stage writes an immutable
checkpoint plus a latest checkpoint and JSON metrics.  The default run uses
the fixed development/holdout split created for the shared-encoder diagnosis;
the holdout is reported once at the end and is never used to choose a stage.
"""

import argparse
import importlib.util
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dataset import RefELDataset
from detail_adapter_model import BRANCH_FILES, build_detail_model, build_textless_unet
from train_text import FormalTextCache
from shared_moe import SharedFeatures, _state


ROOT = Path(__file__).resolve().parent
SPLIT_FILE = ROOT / "results" / "single_encoder_joint_20260924" / "split.json"


def load_router():
    path = ROOT / "43_final_testB_gate_v2_and_m2_metrics.py"
    spec = importlib.util.spec_from_file_location("detail_frozen_router", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def size_bucket(mask):
    pixels = int(mask.sum().item())
    if pixels == 0:
        return "no_target"
    ratio = pixels / mask.numel()
    if ratio < 0.001:
        return "ultra_tiny"
    if ratio < 0.006:
        return "tiny"
    return "larger"


def dice_loss(logits, mask):
    probs = logits.sigmoid()
    dims = tuple(range(1, logits.ndim))
    intersection = (probs * mask).sum(dims)
    return 1 - ((2 * intersection + 1) /
                (probs.sum(dims) + mask.sum(dims) + 1)).mean()


def segmentation_loss(logits, mask, weight=None):
    if weight is None:
        bce = F.binary_cross_entropy_with_logits(logits, mask)
        return bce + dice_loss(logits, mask)
    bce = F.binary_cross_entropy_with_logits(
        logits, mask, reduction="none"
    ).flatten(1).mean(1)
    probs = logits.sigmoid().flatten(1)
    targets = mask.flatten(1)
    dice = 1 - (2 * (probs * targets).sum(1) + 1) / (
        probs.sum(1) + targets.sum(1) + 1
    )
    return ((bce + dice) * weight).sum() / weight.sum()


def size_loss_weights(masks, tiny_weight, ultra_weight):
    ratios = (masks > 0.5).float().flatten(1).mean(1)
    weights = torch.ones_like(ratios)
    weights[(ratios > 0) & (ratios < 0.006)] = tiny_weight
    weights[(ratios > 0) & (ratios < 0.001)] = ultra_weight
    return weights


def stack_items(dataset, indices, device):
    items = [dataset[i] for i in indices]
    images = torch.stack([item["image"] for item in items]).to(device)
    masks = torch.stack([item["mask"] for item in items]).to(device)
    texts = [item["text"] for item in items]
    return items, images, masks, texts


def pooled_for(cache, items, split, device):
    batch = {
        "sent_id": torch.tensor([item["sent_id"] for item in items], dtype=torch.long),
        "ref_id": torch.tensor([item["ref_id"] for item in items], dtype=torch.long),
        "text": [item["text"] for item in items],
    }
    return cache.get_batch(batch, expected_split=split, device=device)


@torch.no_grad()
def route_indices(model, dataset, indices, router, device, cache, split, batch_size=4):
    model.eval()
    groups = {name: [] for name in BRANCH_FILES}
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start:start + batch_size]
        items, images, _, texts = stack_items(dataset, batch_ids, device)
        pooled = pooled_for(cache, items, split, device)
        general, _ = model.forward_general(images, pooled_text=pooled)
        ratios = (general.sigmoid() >= 0.5).float().flatten(1).mean(1).tolist()
        for index, text, ratio in zip(batch_ids, texts, ratios):
            branch = router.route_sample(text, float(ratio))[0]
            groups[branch].append(index)
    return groups


def binary_row(pred, gt):
    pred = pred.bool()
    gt = gt.bool()
    inter = int((pred & gt).sum().item())
    union = int((pred | gt).sum().item())
    pred_n = int(pred.sum().item())
    gt_n = int(gt.sum().item())
    return inter, union, (inter / union if union else (1.0 if pred_n == gt_n == 0 else 0.0))


@torch.no_grad()
def evaluate(model, dataset, indices, router, device, cache, split, batch_size=4):
    model.eval()
    groups = route_indices(model, dataset, indices, router, device, cache, split, batch_size)
    rows = []
    for branch, branch_ids in groups.items():
        for start in range(0, len(branch_ids), batch_size):
            ids = branch_ids[start:start + batch_size]
            items, images, masks, texts = stack_items(dataset, ids, device)
            pooled = pooled_for(cache, items, split, device)
            _, context = model.forward_general(images, pooled_text=pooled)
            logits = model.forward_expert(context, branch)
            preds = logits.sigmoid() >= 0.5
            for item, pred, gt in zip(items, preds, masks):
                inter, union, iou = binary_row(pred, gt)
                rows.append({
                    "index": int(item["ref_id"]), "branch": branch,
                    "bucket": size_bucket(gt), "intersection": inter,
                    "union": union, "iou": iou,
                    "gt_empty": not bool(gt.any().item()),
                    "pred_empty": not bool(pred.any().item()),
                })
    def summarize(subset):
        if not subset:
            return {"n": 0, "mean_IoU": None, "cIoU": None}
        return {
            "n": len(subset),
            "mean_IoU": sum(row["iou"] for row in subset) / len(subset),
            "cIoU": sum(row["intersection"] for row in subset) /
                    max(1, sum(row["union"] for row in subset)),
            "false_activation": sum(row["gt_empty"] and not row["pred_empty"]
                                     for row in subset),
        }
    result = summarize(rows)
    result["by_bucket"] = {
        bucket: summarize([row for row in rows if row["bucket"] == bucket])
        for bucket in ("no_target", "ultra_tiny", "tiny", "larger")
    }
    result["by_branch"] = {
        branch: summarize([row for row in rows if row["branch"] == branch])
        for branch in BRANCH_FILES
    }
    result["rows"] = rows
    return result


def set_phase(model, phase):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for branch in BRANCH_FILES:
        for parameter in model.detail_adapters[branch].parameters():
            parameter.requires_grad_(True)
    if phase in ("head_detail", "shallow_joint"):
        for parameter in model.heads.parameters():
            parameter.requires_grad_(True)
    if phase == "shallow_joint":
        for name, parameter in model.general.named_parameters():
            if name.startswith(("stem.", "layer1.", "layer2.")):
                parameter.requires_grad_(True)


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def phase_defaults(name):
    return {
        "detail_only": (2, 900, 4e-4, 0.20, 0.05),
        "head_detail": (2, 1200, 8e-5, 0.20, 0.05),
        "shallow_joint": (2, 1500, 2e-5, 0.25, 0.05),
    }[name]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_or_create_split(val, seed):
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Required fixed evaluation split is missing: {SPLIT_FILE}")
    data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    dev, holdout = data["dev_indices"], data["holdout_indices"]
    if not dev or not holdout or len(set(dev + holdout)) != len(dev) + len(holdout):
        raise ValueError("Evaluation reference indices must be nonempty and disjoint")
    if any(not isinstance(index, int) or index < 0 or index >= len(val)
           for index in dev + holdout):
        raise ValueError("Evaluation split contains an out-of-range reference index")
    return dev, holdout


def image_ids(dataset, indices):
    return {dataset.refs[int(index)]["image_id"] for index in indices}


def build_disjoint_train_pool(train, val, dev_indices, holdout_indices):
    """Return train references whose source images never enter evaluation.

    Ref-EL has multiple referring expressions per source image.  Filtering by
    reference index is insufficient: every reference belonging to an eval
    image must be removed from the new training pool.
    """
    dev_images = image_ids(val, dev_indices)
    holdout_images = image_ids(val, holdout_indices)
    overlap_eval = dev_images & holdout_images
    if overlap_eval:
        raise RuntimeError(
            "Development and holdout split share source images: "
            f"{len(overlap_eval)}"
        )
    excluded = dev_images | holdout_images
    eligible = [
        index for index, ref in enumerate(train.refs)
        if ref["image_id"] not in excluded
    ]
    leaked = {
        train.refs[index]["image_id"] for index in eligible
    } & excluded
    if leaked:
        raise RuntimeError(
            "Training pool still overlaps evaluation images: "
            f"{len(leaked)}"
        )
    return eligible, excluded, {
        "dev_images": len(dev_images),
        "holdout_images": len(holdout_images),
        "excluded_images": len(excluded),
        "train_refs_before": len(train),
        "train_refs_after": len(eligible),
        "train_val_source_image_overlap_total": len(
            {ref["image_id"] for ref in train.refs} &
            {ref["image_id"] for ref in val.refs}
        ),
    }


def teacher_for(branch, device):
    teacher = build_textless_unet(device=device)
    path = ROOT / "checkpoints" / BRANCH_FILES[branch]
    teacher.load_state_dict(
        {key: value for key, value in _state(path).items()
         if not key.startswith("text_encoder.")},
        strict=False,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def teacher_logits(teacher, images, pooled_text):
    e0 = teacher.stem(images)
    e1 = teacher.layer1(teacher.maxpool(e0))
    e2 = teacher.layer2(e1)
    e3 = teacher.layer3(e2)
    e4 = teacher.layer4(e3)
    context = SharedFeatures((e0, e1, e2, e3, e4), pooled_text,
                             images.shape[-2:])
    return teacher._decode(context) if hasattr(teacher, "_decode") else None


def decode_teacher(teacher, images, pooled_text, return_e2=False):
    e0 = teacher.stem(images)
    e1 = teacher.layer1(teacher.maxpool(e0))
    e2 = teacher.layer2(e1)
    e3 = teacher.layer3(e2)
    e4 = teacher.layer4(e3)
    pooled = pooled_text
    x = teacher.decoder4(teacher.text_gate_e4(e4, pooled),
                         teacher.text_gate_e3(e3, pooled))
    x = teacher.decoder3(x, teacher.text_gate_e2(e2, pooled))
    x = teacher.decoder2(x, e1)
    x = teacher.decoder1(x, e0)
    x = F.interpolate(x, size=images.shape[-2:], mode="bilinear", align_corners=False)
    features = (e0, e1, e2) if return_e2 else (e0, e1)
    return teacher.segmentation_head(teacher.final_conv(x)), features


def train_phase(model, phase, train, train_pool, val, dev_indices, args, router, cache, output):
    epochs, samples, lr, distill_weight, feature_weight = phase_defaults(phase)
    epochs = args.epochs or epochs
    samples = args.train_samples or samples
    model.train()
    set_phase(model, phase)
    # Keep frozen BatchNorm/text encoder statistics fixed, especially at batch 1.
    model.general.eval()
    model.general.text_encoder.eval()
    model.heads.train()
    model.detail_adapters.train()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=lr,
                                  weight_decay=1e-4)
    stage_dir = output / phase
    stage_dir.mkdir(parents=True, exist_ok=True)
    initial = evaluate(model, val, dev_indices, router, args.device, cache, "val", args.batch_size)
    save_json(stage_dir / "initial.json", initial)
    history = [{"epoch": 0, "metrics": initial, "train_loss": None}]
    best = initial["cIoU"] or 0.0
    model.save_checkpoint(stage_dir / "initial.pth", phase,
                          {"base_checkpoint_dir": str(ROOT / "checkpoints")})
    phase_seeds = {"detail_only": 1000, "head_detail": 2000, "shallow_joint": 3000}
    for epoch in range(1, epochs + 1):
        ids = list(train_pool)
        random.Random(args.seed + phase_seeds[phase] + epoch).shuffle(ids)
        ids = ids[:min(samples, len(ids))]
        groups = route_indices(model, train, ids, router, args.device, cache, "train", args.batch_size)
        total = 0.0
        steps = 0
        for branch, branch_ids in groups.items():
            if not branch_ids:
                continue
            teacher = teacher_for(branch, args.device)
            for start in range(0, len(branch_ids), args.batch_size):
                batch_ids = branch_ids[start:start + args.batch_size]
                items, images, masks, texts = stack_items(train, batch_ids, args.device)
                # Student shared features and text embeddings.
                pooled = pooled_for(cache, items, "train", args.device)
                _, context = model.forward_general(images, pooled_text=pooled)
                student_logits = model.forward_expert(context, branch)
                with torch.no_grad():
                    teacher_logits_batch, teacher_low = decode_teacher(
                        teacher, images, context.pooled_text)
                weights = size_loss_weights(
                    masks, args.tiny_loss_weight, args.ultra_loss_weight
                ) if args.tiny_loss_weight != 1 or args.ultra_loss_weight != 1 else None
                sup = segmentation_loss(student_logits, masks, weights)
                imitation = F.binary_cross_entropy_with_logits(
                    student_logits, teacher_logits_batch.sigmoid()
                )
                adapted = model.detail_features(context, branch)
                detail = sum(
                    F.smooth_l1_loss(adapted[i], teacher_low[i]) /
                    teacher_low[i].detach().std().clamp_min(0.1)
                    for i in (0, 1)
                ) / 2
                loss = sup + distill_weight * imitation + feature_weight * detail
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 5.0)
                optimizer.step()
                total += float(loss.detach())
                steps += 1
            del teacher
            if args.device.type == "cuda":
                torch.cuda.empty_cache()
        metrics = evaluate(model, val, dev_indices, router, args.device, cache, "val", args.batch_size)
        row = {"epoch": epoch, "train_loss": total / max(1, steps),
               "metrics": metrics,
               "sampled_train_indices": ids,
               "routes": {name: len(group) for name, group in groups.items()}}
        history.append(row)
        save_json(stage_dir / "history.json", history)
        model.save_checkpoint(stage_dir / "latest.pth", phase,
                              {"base_checkpoint_dir": str(ROOT / "checkpoints")})
        if (metrics["cIoU"] or 0.0) > best:
            best = metrics["cIoU"] or 0.0
            model.save_checkpoint(stage_dir / "best.pth", phase,
                                  {"base_checkpoint_dir": str(ROOT / "checkpoints")})
        print({"phase": phase, "epoch": epoch, "train_loss": row["train_loss"],
               "n_train": len(ids), "cIoU": metrics["cIoU"],
               "mean_IoU": metrics["mean_IoU"],
               "by_bucket": {name: value["mean_IoU"]
                             for name, value in metrics["by_bucket"].items()}}, flush=True)
    # Always retain an explicit completion snapshot, even if validation did not improve.
    model.save_checkpoint(stage_dir / "completed.pth", phase,
                          {"base_checkpoint_dir": str(ROOT / "checkpoints")})
    return history[-1]["metrics"]


def main(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    # Write a manifest first so an interrupted run still identifies its base.
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "base_checkpoint_dir": str(ROOT / "checkpoints"),
        "web_untouched": True,
        "device": str(args.device),
        "seed": args.seed,
        "tiny_loss_weight": args.tiny_loss_weight,
        "ultra_loss_weight": args.ultra_loss_weight,
        "stages": ["detail_only", "head_detail", "shallow_joint"],
    }
    save_json(output / "manifest.json", manifest)
    train = RefELDataset(ROOT, split="train", image_size=384, random_text=True)
    val = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    cache = FormalTextCache(ROOT)
    dev_indices, holdout_indices = load_or_create_split(val, args.seed)
    train_pool, excluded_images, split_audit = build_disjoint_train_pool(
        train, val, dev_indices, holdout_indices
    )
    save_json(output / "split_used.json", {
        "dev_indices": dev_indices, "holdout_indices": holdout_indices,
        "source": str(SPLIT_FILE),
        "excluded_image_ids_from_training": sorted(excluded_images),
        "train_pool_indices": train_pool,
        "audit": split_audit,
    })
    router = load_router()
    model = build_detail_model(ROOT / "checkpoints", args.device)
    # Save a zero-adapter baseline before any update.
    base_metrics = evaluate(model, val, dev_indices, router, args.device, cache, "val", args.batch_size)
    save_json(output / "dev_baseline.json", base_metrics)
    model.save_checkpoint(output / "baseline_initial.pth", "baseline",
                          {"base_checkpoint_dir": str(ROOT / "checkpoints")})
    current = base_metrics
    for phase in ("detail_only", "head_detail", "shallow_joint"):
        current = train_phase(model, phase, train, train_pool, val, dev_indices, args,
                              router, cache, output)
        save_json(output / f"after_{phase}.json", current)
    model.save_checkpoint(output / "final.pth", "final",
                          {"base_checkpoint_dir": str(ROOT / "checkpoints")})
    summary = {"output": str(output), "dev_final": {
        "mean_IoU": current["mean_IoU"], "cIoU": current["cIoU"]}}
    if not args.skip_holdout:
        # Holdout is evaluated only after the experiment is selected on dev.
        holdout = evaluate(model, val, holdout_indices, router, args.device, cache,
                           "val", args.batch_size)
        save_json(output / "holdout_final.json", holdout)
        summary["holdout_final"] = {
            "mean_IoU": holdout["mean_IoU"], "cIoU": holdout["cIoU"]
        }
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="detail_adapter_20260924")
    parser.add_argument("--epochs", type=int, default=0,
                        help="0 uses conservative stage defaults")
    parser.add_argument("--train-samples", type=int, default=0,
                        help="0 uses stage-specific samples")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--tiny-loss-weight", type=float, default=1.0)
    parser.add_argument("--ultra-loss-weight", type=float, default=1.0)
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    options = parser.parse_args()
    if options.tiny_loss_weight < 1 or options.ultra_loss_weight < 1:
        parser.error("Small-defect loss weights must be at least 1")
    main(options)
