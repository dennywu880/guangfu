"""Measure frozen-rule routing headroom without training or changing the UI.

Oracle selection uses ground-truth masks and is diagnostic only.  It is never
a deployable prediction or a valid model score.  The comparison deliberately
excludes the presence Gate and reports target-present references separately:
otherwise the oracle can trivially choose an empty mask for no-target text.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from dataset import RefELDataset
from detail_adapter_model import BRANCH_FILES, build_detail_model
from train_detail_adapter import (load_router, pooled_for, size_bucket,
                                  teacher_for)
from train_text import FormalTextCache


ROOT = Path(__file__).resolve().parent


def summarize(rows, prefix):
    selected = [r for r in rows if not r["gt_empty"]]
    def mean(key, subset):
        return sum(r[prefix][key] for r in subset) / len(subset) if subset else None
    return {
        "n_target": len(selected),
        "rule_mean_IoU": mean("rule_iou", selected),
        "oracle_mean_IoU": mean("oracle_iou", selected),
        "potential_mean_IoU_gain": mean("gain", selected),
        "different_best_expert": sum(r[prefix]["different_best"] for r in selected),
        "meaningful_gain_over_0_05": sum(r[prefix]["gain"] > 0.05 for r in selected),
        "by_size": {
            bucket: {
                "n": len(group := [r for r in selected if r["size"] == bucket]),
                "rule_mean_IoU": mean("rule_iou", group),
                "oracle_mean_IoU": mean("oracle_iou", group),
                "meaningful_gain_over_0_05": sum(
                    r[prefix]["gain"] > 0.05 for r in group),
            }
            for bucket in ("ultra_tiny", "tiny", "larger")
        },
    }


def iou(pred, gt):
    intersection = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return intersection / union if union else 1.0


def branch_result(scores, routed):
    best = max(BRANCH_FILES, key=lambda name: scores[name])
    return {
        "per_expert_IoU": scores,
        "rule_expert": routed,
        "best_expert": best,
        "rule_iou": scores[routed],
        "oracle_iou": scores[best],
        "gain": scores[best] - scores[routed],
        "different_best": scores[best] > scores[routed] + 1e-9,
    }


@torch.no_grad()
def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    result_dir = ROOT / "results" / args.experiment
    split = json.loads((result_dir / "split_used.json").read_text(encoding="utf-8"))
    indices = split["dev_indices"]
    data = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    cache = FormalTextCache(ROOT)
    router = load_router()
    shared = build_detail_model(ROOT / "checkpoints", device)
    shared.load_checkpoint(result_dir / args.checkpoint)
    shared.eval()
    teachers = {name: teacher_for(name, device) for name in BRANCH_FILES}
    rows = []
    for number, index in enumerate(indices, 1):
        item = data[index]
        image = item["image"].unsqueeze(0).to(device)
        gt = item["mask"].unsqueeze(0).to(device) >= 0.5
        pooled = pooled_for(cache, [item], "val", device)
        general, context = shared.forward_general(image, pooled_text=pooled)
        ratio = float((general.sigmoid() >= 0.5).float().mean().item())
        routed = router.route_sample(item["text"], ratio)[0]
        shared_scores = {}
        original_scores = {}
        for name in BRANCH_FILES:
            shared_scores[name] = iou(shared.forward_expert(context, name).sigmoid() >= 0.5, gt)
            original_scores[name] = iou(
                teachers[name](image, pooled_text=pooled).sigmoid() >= 0.5, gt
            )
        rows.append({
            "ref_id": int(item["ref_id"]), "val_index": int(index),
            "image_id": int(item["image_id"]),
            "gt_empty": not bool(gt.any().item()), "size": size_bucket(gt),
            "shared": branch_result(shared_scores, routed),
            "original": branch_result(original_scores, routed),
        })
        if number % 40 == 0:
            print(f"Compared {number}/{len(indices)} references", flush=True)
    result = {
        "protocol": "val_development_only; target-present oracle; no presence Gate; 384px; threshold 0.5",
        "warning": "Ground-truth oracle is diagnostic only and cannot be deployed or compared as a model result.",
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "shared": summarize(rows, "shared"),
        "original": summarize(rows, "original"),
        "rows": rows,
    }
    suffix = Path(args.checkpoint).stem
    destination = result_dir / f"expert_routing_diagnosis_dev_{suffix}.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination),
                      "shared": result["shared"], "original": result["original"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="detail_adapter_disjoint_20260924")
    parser.add_argument("--checkpoint", default="final.pth")
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
