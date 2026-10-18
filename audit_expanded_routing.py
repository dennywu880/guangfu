"""Read-only fixed-router versus per-reference oracle audit on val images.

The oracle reads ground-truth masks and is not deployable. This development
split is not blind because historical checkpoints may have seen val images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import RefELDataset
from evaluate_shared_testB import load_candidate, original_protocol
from multilevel_selection import build_model
from train_detail_adapter import ROOT, load_or_create_split
from train_shared_layer4_joint import load as load_layer4
from train_text import FormalTextCache


def selected_indices(dataset, count, seed):
    _, holdout = load_or_create_split(dataset, seed)
    reserved_images = {dataset.refs[index]["image_id"] for index in holdout}
    eligible = [
        index for index, ref in enumerate(dataset.refs)
        if ref["image_id"] not in reserved_images
    ]
    random.Random(seed).shuffle(eligible)
    chosen = []
    images = set()
    for index in eligible:
        image_id = dataset.refs[index]["image_id"]
        if image_id in images:
            continue
        chosen.append(index)
        images.add(image_id)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"Requested {count} unique images, found {len(chosen)}")
    return chosen, reserved_images


def put_stats(record, stats):
    record.update({
        "final_tp": stats["tp"],
        "final_fp": stats["fp"],
        "final_fn": stats["fn"],
        "final_intersection": stats["intersection"],
        "final_union": stats["union"],
        "final_dice": stats["dice"],
        "final_iou": stats["iou_regular"],
        "final_pred_empty": stats["pred_empty"],
    })


def summary(protocol, records):
    rows = list(records.values())
    positive = [row for row in rows if row["is_positive"]]
    metrics = protocol.compute_m2segnet_style_metrics(records)
    return {
        "N": len(rows),
        "positive_N": len(positive),
        "no_target_N": len(rows) - len(positive),
        "positive_mean_IoU": sum(row["final_iou"] for row in positive) / len(positive),
        "gIoU": metrics["gIoU"],
        "cIoU": metrics["cIoU"],
        "no_target_accuracy": sum(
            row["final_pred_empty"] for row in rows if not row["is_positive"]
        ) / (len(rows) - len(positive)),
        "by_true_size": {
            name: {
                "N": len(group),
                "mean_IoU": sum(row["final_iou"] for row in group) / len(group)
                if group else None,
            }
            for name in protocol.GROUPS
            for group in [[row for row in positive if row["gt_3way"] == name]]
        },
        "pixel_counts": metrics["pixel_counts"],
    }


@torch.inference_mode()
def main(args):
    protocol = original_protocol()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for an explicit CPU run")
    output = ROOT / "results" / args.output
    if output.exists():
        raise FileExistsError(output)
    dataset = RefELDataset(ROOT, "val", protocol.IMAGE_SIZE, False)
    indices, reserved = selected_indices(dataset, args.count, args.seed)
    cache = FormalTextCache(ROOT)
    if args.checkpoint:
        model = build_model(device, ("e3", "e4"))
        load_layer4(args.checkpoint, model)
        model.eval()
        model_path = args.checkpoint
    else:
        model = load_candidate("head_joint_best", device)
        model_path = ROOT / "results" / "head_joint_e3_e4_20260925" / "best.pth"
    gate, mean, std, hook, threshold = protocol.load_presence_gate(device)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=protocol.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
    )
    base, _, _ = protocol.phase1_general_gate_router(
        model.base.general, gate, mean, std, hook, threshold,
        loader, device, cache, expected_split="val",
    )
    fixed = {key: row.copy() for key, row in base.items()}
    oracle = {key: row.copy() for key, row in base.items()}
    evaluated = 0
    gains = []
    for position, index in enumerate(indices, 1):
        item = dataset[index]
        ref_id = int(item["ref_id"])
        record = base[ref_id]
        if not record["gate_rejects_as_no_target"]:
            image = item["image"].unsqueeze(0).to(device)
            gt = item["mask"].bool()
            pooled = cache.get_batch({
                "ref_id": torch.tensor([ref_id]),
                "sent_id": torch.tensor([item["sent_id"]]),
                "text": [item["text"]],
            }, expected_split="val", device=device)
            _, context = model.forward_general(image, pooled_text=pooled)
            scores = {}
            for branch in protocol.GROUPS:
                pred = (model.forward_expert(context, branch).sigmoid()
                        >= protocol.EXPERT_MASK_THRESHOLD).squeeze(0).cpu()
                scores[branch] = protocol.binary_stats(pred, gt)
            routed = record["route_branch"]
            best = routed
            if record["is_positive"]:
                best = max(protocol.GROUPS, key=lambda name: scores[name]["iou_regular"])
            put_stats(fixed[ref_id], scores[routed])
            put_stats(oracle[ref_id], scores[best])
            oracle[ref_id]["oracle_expert"] = best
            gain = scores[best]["iou_regular"] - scores[routed]["iou_regular"]
            gains.append({
                "ref_id": ref_id, "gt_3way": record["gt_3way"],
                "fixed_expert": routed, "oracle_expert": best,
                "fixed_IoU": scores[routed]["iou_regular"],
                "oracle_IoU": scores[best]["iou_regular"],
                "gain": gain,
            })
            evaluated += 1
        if position % 100 == 0 or position == len(indices):
            print(f"Evaluated {position}/{len(indices)}; gate survivors={evaluated}", flush=True)
    if any(row["final_iou"] is None for row in fixed.values()):
        raise RuntimeError("Fixed-route evaluation is incomplete")
    if any(row["final_iou"] is None for row in oracle.values()):
        raise RuntimeError("Oracle evaluation is incomplete")
    fixed_summary = summary(protocol, fixed)
    oracle_summary = summary(protocol, oracle)
    positive_gains = [row for row in gains if base[row["ref_id"]]["is_positive"]]
    result = {
        "protocol": {
            "split": "val development; never testB",
            "sampling": "one seeded random reference per source image, holdout images excluded",
            "seed": args.seed, "image_size": protocol.IMAGE_SIZE,
            "mask_threshold": protocol.EXPERT_MASK_THRESHOLD,
            "presence_gate_threshold": threshold,
            "fixed_gate_and_router": True,
            "oracle": "GT-selected expert for gate-surviving positive references only",
            "reserved_holdout_images": len(reserved),
            "checkpoint": str(model_path),
            "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "limitation": "Historical checkpoints may have seen val source images; diagnostic, not blind test",
        },
        "fixed": fixed_summary,
        "oracle": oracle_summary,
        "delta": {
            key: oracle_summary[key] - fixed_summary[key]
            for key in ("positive_mean_IoU", "gIoU", "cIoU", "no_target_accuracy")
        },
        "gain_over_0_05": sum(row["gain"] > 0.05 for row in positive_gains),
        "gain_over_0_10": sum(row["gain"] > 0.10 for row in positive_gains),
        "positive_gate_survivors": len(positive_gains),
        "positive_gate_rejected": sum(
            row["is_positive"] and row["gate_rejects_as_no_target"]
            for row in base.values()
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "positive_gains.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(positive_gains[0]))
        writer.writeheader()
        writer.writerows(positive_gains)
    (output / "sample_indices.json").write_text(
        json.dumps(indices, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", default="expanded_routing_audit_20260926")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
