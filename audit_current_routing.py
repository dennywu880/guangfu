"""Read-only routing and expert-capacity diagnosis for the current shared MoE.

The oracle uses ground truth and is strictly a diagnostic upper bound, not a
deployable predictor. Historical checkpoints may have seen validation images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from dataset import RefELDataset
from detail_adapter_model import BRANCH_FILES
from multilevel_selection import build_model
from train_multilevel_head_joint import load as load_joint
from train_detail_adapter import (
    ROOT, binary_row, decode_teacher, load_or_create_split, load_router,
    pooled_for, size_bucket, teacher_for,
)
from train_text import FormalTextCache


SOURCE = ROOT / "results" / "multilevel_e3_e4_20260925" / "best.pth"
HEAD_JOINT = ROOT / "results" / "head_joint_e3_e4_20260925" / "best.pth"


def summarize(rows, prefix):
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


def summarize_routing(rows, family):
    positives = [row for row in rows if not row["gt_empty"]]

    def group_summary(group):
        gains = [row[family]["oracle_gain"] for row in group]
        return {
            "n": len(group),
            "rule_mean_IoU": sum(row[family]["rule_iou"] for row in group)
            / len(group) if group else None,
            "oracle_mean_IoU": sum(row[family]["oracle_iou"] for row in group)
            / len(group) if group else None,
            "mean_oracle_gain": sum(gains) / len(gains) if gains else None,
            "gain_over_0_05": sum(gain > 0.05 for gain in gains),
            "gain_over_0_10": sum(gain > 0.10 for gain in gains),
        }

    return {
        "positive_only": group_summary(positives),
        "by_bucket": {
            bucket: group_summary([row for row in positives if row["bucket"] == bucket])
            for bucket in ("ultra_tiny", "tiny", "larger")
        },
    }


def family_result(scores, routed):
    best = max(scores, key=scores.get)
    return {
        "per_expert_IoU": scores,
        "rule_expert": routed,
        "best_expert": best,
        "rule_iou": scores[routed],
        "oracle_iou": scores[best],
        "oracle_gain": scores[best] - scores[routed],
    }


@torch.inference_mode()
def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu to explicitly run on CPU")
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    val = RefELDataset(ROOT, "val", 384, False)
    dev, holdout = load_or_create_split(val, 20260924)
    if {val.refs[i]["image_id"] for i in dev} & {
        val.refs[i]["image_id"] for i in holdout
    }:
        raise RuntimeError("Development and holdout source images overlap")
    cache = FormalTextCache(ROOT)
    router = load_router()
    source = HEAD_JOINT if args.candidate == "head_joint" else SOURCE
    shared = build_model(device, ("e3", "e4"))
    if args.candidate == "head_joint":
        load_joint(source, shared)
    else:
        shared.load_checkpoint(source)
    shared.eval()
    rows = []
    for name in BRANCH_FILES:
        if not (ROOT / "checkpoints" / BRANCH_FILES[name]).is_file():
            raise FileNotFoundError(BRANCH_FILES[name])
    teachers = {name: teacher_for(name, device) for name in BRANCH_FILES}
    for position, index in enumerate(dev, 1):
        item = val[index]
        image = item["image"].unsqueeze(0).to(device)
        gt = item["mask"].unsqueeze(0).to(device)
        pooled = pooled_for(cache, [item], "val", device)
        general, context = shared.forward_general(image, pooled_text=pooled)
        ratio = float((general.sigmoid() >= 0.5).float().mean().item())
        routed = router.route_sample(item["text"], ratio)[0]
        shared_rows = {}
        teacher_rows = {}
        for name in BRANCH_FILES:
            shared_pred = shared.forward_expert(context, name).sigmoid() >= 0.5
            teacher_logits, _ = decode_teacher(teachers[name], image, pooled)
            teacher_pred = teacher_logits.sigmoid() >= 0.5
            shared_rows[name] = binary_row(shared_pred, gt)
            teacher_rows[name] = binary_row(teacher_pred, gt)
        shared_scores = {name: value[2] for name, value in shared_rows.items()}
        teacher_scores = {name: value[2] for name, value in teacher_rows.items()}
        si, su, score = shared_rows[routed]
        ti, tu, tscore = teacher_rows[routed]
        rows.append({
            "val_index": index,
            "ref_id": int(item["ref_id"]),
            "image_id": int(item["image_id"]),
            "bucket": size_bucket(gt),
            "gt_empty": not bool(gt.any().item()),
            "route": routed,
            "shared": family_result(shared_scores, routed),
            "teacher": family_result(teacher_scores, routed),
            "shared_intersection": si, "shared_union": su, "shared_iou": score,
            "shared_pred_empty": su == 0 if not bool(gt.any().item()) else False,
            "teacher_intersection": ti, "teacher_union": tu, "teacher_iou": tscore,
            "teacher_pred_empty": tu == 0 if not bool(gt.any().item()) else False,
        })
        if position % 40 == 0:
            print(f"Evaluated {position}/{len(dev)}", flush=True)
    positives = [row for row in rows if not row["gt_empty"]]
    result = {
        "protocol": "Historical 160-reference val development set; cached text; 384px; threshold 0.5; fixed route; no presence Gate",
        "limitations": [
            "Oracle uses ground truth and is not a deployable prediction.",
            "Historical checkpoints may have seen validation images; not a blind test.",
        ],
        "candidate": args.candidate,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n": len(rows),
        "same_route": {
            family: {
                "overall": summarize(rows, family),
                "positive": summarize(positives, family),
                "by_bucket": {
                    bucket: summarize([r for r in rows if r["bucket"] == bucket], family)
                    for bucket in ("no_target", "ultra_tiny", "tiny", "larger")
                },
            }
            for family in ("shared", "teacher")
        },
        "routing_headroom": {
            family: summarize_routing(rows, family)
            for family in ("shared", "teacher")
        },
        "positive_cases_teacher_better_same_route": sum(
            row["teacher_iou"] > row["shared_iou"] for row in positives
        ),
        "positive_cases_shared_better_same_route": sum(
            row["shared_iou"] > row["teacher_iou"] for row in positives
        ),
        "rows": rows,
    }
    (output / "diagnosis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", choices=("multilevel", "head_joint"),
                        default="multilevel")
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
