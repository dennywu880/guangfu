"""Paired teacher/student audit on the fixed development references only.

No training or checkpoint selection occurs here. Both models see the same
image, cached text embedding, fixed route, mask threshold, and ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from dataset import RefELDataset
from multilevel_selection import build_model
from train_detail_adapter import (
    ROOT, binary_row, decode_teacher, load_or_create_split, load_router,
    pooled_for, route_indices, size_bucket, stack_items, teacher_for,
)
from train_text import FormalTextCache


SOURCE = ROOT / "results" / "multilevel_e3_e4_20260925" / "best.pth"


def summary(rows, prefix):
    if not rows:
        return {"n": 0, "mean_IoU": None, "cIoU": None, "false_activation": 0}
    return {
        "n": len(rows),
        "mean_IoU": sum(row[f"{prefix}_iou"] for row in rows) / len(rows),
        "cIoU": sum(row[f"{prefix}_intersection"] for row in rows)
        / max(1, sum(row[f"{prefix}_union"] for row in rows)),
        "false_activation": sum(
            row["gt_empty"] and not row[f"{prefix}_pred_empty"] for row in rows
        ),
    }


@torch.inference_mode()
def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for an explicit CPU audit")
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    dev, _ = load_or_create_split(val, args.seed)
    model = build_model(device, ("e3", "e4"))
    model.load_checkpoint(SOURCE)
    model.eval()
    router = load_router()
    groups = route_indices(model, val, dev, router, device, cache, "val", args.batch_size)
    rows = []
    for branch, indices in groups.items():
        teacher = teacher_for(branch, device)
        for start in range(0, len(indices), args.batch_size):
            ids = indices[start:start + args.batch_size]
            items, images, masks, _ = stack_items(val, ids, device)
            pooled = pooled_for(cache, items, "val", device)
            _, context = model.forward_general(images, pooled_text=pooled)
            student = model.forward_expert(context, branch).sigmoid() >= 0.5
            teacher_logits, _ = decode_teacher(teacher, images, pooled)
            teacher_pred = teacher_logits.sigmoid() >= 0.5
            for index, item, gt, student_pred, teacher_mask in zip(
                ids, items, masks, student, teacher_pred
            ):
                student_inter, student_union, student_iou = binary_row(student_pred, gt)
                teacher_inter, teacher_union, teacher_iou = binary_row(teacher_mask, gt)
                rows.append({
                    "index": index,
                    "image_id": item["image_id"],
                    "branch": branch,
                    "bucket": size_bucket(gt),
                    "gt_empty": not bool(gt.any().item()),
                    "student_intersection": student_inter,
                    "student_union": student_union,
                    "student_iou": student_iou,
                    "student_pred_empty": not bool(student_pred.any().item()),
                    "teacher_intersection": teacher_inter,
                    "teacher_union": teacher_union,
                    "teacher_iou": teacher_iou,
                    "teacher_pred_empty": not bool(teacher_mask.any().item()),
                })
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "source": str(SOURCE),
        "split": "historical_val_development_only",
        "same_conditions": "same image/text, fixed route, 384 resolution and 0.5 mask threshold",
        "n": len(rows),
        "overall": {name: summary(rows, name) for name in ("student", "teacher")},
        "by_bucket": {
            bucket: {name: summary([row for row in rows if row["bucket"] == bucket], name)
                     for name in ("student", "teacher")}
            for bucket in ("no_target", "ultra_tiny", "tiny", "larger")
        },
        "by_branch": {
            branch: {name: summary([row for row in rows if row["branch"] == branch], name)
                     for name in ("student", "teacher")}
            for branch in groups
        },
        "paired_positive_wins": {
            "teacher": sum(row["teacher_iou"] > row["student_iou"]
                           for row in rows if not row["gt_empty"]),
            "student": sum(row["teacher_iou"] < row["student_iou"]
                           for row in rows if not row["gt_empty"]),
            "ties": sum(row["teacher_iou"] == row["student_iou"]
                        for row in rows if not row["gt_empty"]),
        },
        "rows": rows,
        "limitation": "Not an independent blind test; historical weights may have seen Val images.",
    }
    (output / "teacher_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--cpu", action="store_true")
    options = parser.parse_args()
    if options.batch_size < 1:
        parser.error("batch-size must be positive")
    run(options)
