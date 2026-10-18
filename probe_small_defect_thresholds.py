"""Calibrate small-branch mask thresholds on disjoint Val subsets."""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch

from dataset import RefELDataset
from inference_engine import BASELINE, DefectInferenceEngine
from shared_moe import BRANCH_FILES, SharedMoE


ROOT = Path(__file__).resolve().parent
THRESHOLDS = (0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75)
BRANCHES = ("ultra_tiny", "tiny", "larger")


def sampled_indices(length, count, seed):
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def record(pred, gt):
    stats = BASELINE.binary_stats(pred, gt)
    return {"intersection": stats["intersection"], "union": stats["union"],
            "iou": stats["iou_regular"], "pred_empty": stats["pred_empty"]}


def summary(rows):
    return {
        "n": len(rows),
        "mean_IoU": sum(row["iou"] for row in rows) / max(1, len(rows)),
        "cIoU": sum(row["intersection"] for row in rows) /
                max(1, sum(row["union"] for row in rows)),
        "no_target_empty_rate": sum(row["pred_empty"] for row in rows if row["gt_empty"]) /
                                max(1, sum(row["gt_empty"] for row in rows)),
    }


def evaluate_indices(indices, dataset, engine, shared, label):
    rows = {}
    routes = defaultdict(list)
    device = engine.device
    for step, index in enumerate(indices, 1):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(device)
        gt = item["mask"].unsqueeze(0).to(device) > 0.5
        engine.hook.value = None
        general, context = shared.forward_general(image, texts=[item["text"]])
        if engine.hook.value is None:
            raise RuntimeError("Presence Gate feature hook did not fire")
        image_feature = engine._image_feature_from_hook(engine.hook.value).float()
        stats, values = engine._general_stats(general.sigmoid())
        branch, _, text_size = BASELINE.route_sample(item["text"], values["area050"])
        features = torch.cat([
            image_feature, context.pooled_text.float(), stats.float(),
            torch.tensor([BASELINE.route_onehot(branch)], device=device),
            torch.tensor([BASELINE.text_onehot(text_size)], device=device),
        ], dim=1)
        reject = bool((engine.gate((features - engine.gate_mean) / engine.gate_std)
                       .sigmoid() >= engine.gate_threshold).item())
        rows[index] = {"branch": branch, "gt_empty": not bool(gt.any().item()),
                       "gate_reject": reject}
        if reject:
            empty = record(torch.zeros_like(gt), gt)
            rows[index]["old"] = empty
            rows[index]["shared"] = {str(t): empty for t in THRESHOLDS}
        else:
            routes[branch].append((index, context.pooled_text.detach().cpu()))
        if step % 100 == 0:
            print(f"{label} General/Gate {step}/{len(indices)}", flush=True)

    for branch, items in routes.items():
        teacher = engine._activate_expert(branch)
        for step, (index, pooled_cpu) in enumerate(items, 1):
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            gt = item["mask"].unsqueeze(0).to(device) > 0.5
            pooled = pooled_cpu.to(device)
            _, context = shared.forward_general(image, pooled_text=pooled)
            old = teacher(image, pooled_text=pooled).sigmoid()
            probabilities = shared.forward_expert(context, branch).sigmoid()
            rows[index]["old"] = record(old >= 0.5, gt)
            rows[index]["shared"] = {
                str(t): record(probabilities >= t, gt) for t in THRESHOLDS
            }
            if step % 100 == 0:
                print(f"{label} {branch} {step}/{len(items)}", flush=True)
    return [rows[index] for index in indices]


def combined(rows, thresholds):
    old = summary([{**row["old"], "gt_empty": row["gt_empty"]} for row in rows])
    shared = summary([{**row["shared"][str(thresholds[row["branch"]])],
                       "gt_empty": row["gt_empty"]} for row in rows])
    return {"old": old, "shared": shared}


def choose(dev):
    selected = {"larger": 0.5}
    for branch in ("ultra_tiny", "tiny"):
        rows = [row for row in dev if row["branch"] == branch]
        baseline = summary([{**row["shared"]["0.5"], "gt_empty": row["gt_empty"]}
                            for row in rows])
        candidates = []
        for threshold in THRESHOLDS:
            metrics = summary([{**row["shared"][str(threshold)],
                                "gt_empty": row["gt_empty"]} for row in rows])
            if (metrics["cIoU"] >= baseline["cIoU"] - 0.005
                    and metrics["no_target_empty_rate"] >=
                    baseline["no_target_empty_rate"] - 0.01):
                candidates.append((metrics["mean_IoU"], -abs(threshold - 0.5), threshold))
        selected[branch] = max(candidates)[2] if candidates else 0.5
    return selected


@torch.inference_mode()
def run(dev_count, holdout_count):
    os.environ["TG_SHARED_MOE"] = "0"
    torch.set_num_threads(4)
    engine = DefectInferenceEngine()
    shared = SharedMoE.from_checkpoints(
        ROOT / "checkpoints", engine.device,
        general=engine.general_model, expert_backbone="larger")
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    previous_120 = set(sampled_indices(len(dataset), 120, 42))
    previous_500 = [index for index in sampled_indices(len(dataset), len(dataset), 314159)
                    if index not in previous_120][:500]
    blocked = previous_120 | set(previous_500)
    blocked_images = {dataset.refs[index]["image_id"] for index in blocked}
    available = []
    selected_images = set()
    for index in sampled_indices(len(dataset), len(dataset), 161803):
        image_id = dataset.refs[index]["image_id"]
        if image_id not in blocked_images and image_id not in selected_images:
            available.append(index)
            selected_images.add(image_id)
    dev_indices = available[:dev_count]
    holdout_indices = available[dev_count:dev_count + holdout_count]
    if len(holdout_indices) != holdout_count:
        raise ValueError("Not enough unused validation examples")

    dev = evaluate_indices(dev_indices, dataset, engine, shared, "Dev")
    selected = choose(dev)
    holdout = evaluate_indices(holdout_indices, dataset, engine, shared, "Holdout")
    fixed = dict.fromkeys(BRANCHES, 0.5)
    report = {
        "split": "val", "seed": 161803, "image_disjoint": True,
        "excluded_prior_examples": len(blocked),
        "excluded_prior_images": len(blocked_images),
        "dev_count": len(dev), "holdout_count": len(holdout),
        "tested_thresholds": THRESHOLDS, "selected_on_dev": selected,
        "dev_fixed": combined(dev, fixed), "dev_selected": combined(dev, selected),
        "holdout_fixed": combined(holdout, fixed),
        "holdout_selected": combined(holdout, selected),
        "holdout_by_branch": {
            branch: {"fixed": combined([r for r in holdout if r["branch"] == branch], fixed),
                     "selected": combined([r for r in holdout if r["branch"] == branch], selected)}
            for branch in BRANCHES
        },
        "holdout_gate_rejections": sum(row["gate_reject"] for row in holdout),
    }
    output = ROOT / "results" / "small_defect_threshold_probe.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=int, default=160)
    parser.add_argument("--holdout", type=int, default=160)
    args = parser.parse_args()
    run(args.dev, args.holdout)
