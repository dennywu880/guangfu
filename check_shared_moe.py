"""Smoke test and same-sample expert-only comparison on the validation split."""

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import torch

from dataset import RefELDataset
from shared_moe import BRANCH_FILES, SharedMoE


ROOT = Path(__file__).resolve().parent


def baseline_module():
    path = ROOT / "43_final_testB_gate_v2_and_m2_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def run(samples, seed, exclude_seed=None, exclude_samples=0):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = baseline_module()
    backbone = os.environ.get("TG_EXPERT_BACKBONE") or None
    moe = SharedMoE.from_checkpoints(ROOT / "checkpoints", device,
                                     expert_backbone=backbone)
    if os.environ.get("TG_SHARED_HEADS"):
        moe.load_heads(os.environ["TG_SHARED_HEADS"])
    if os.environ.get("TG_SHARED_REPAIR"):
        moe.load_repair(os.environ["TG_SHARED_REPAIR"])
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    if not len(dataset):
        raise ValueError("Empty validation split")

    ids = list(range(len(dataset)))
    random.Random(seed).shuffle(ids)
    if exclude_seed is not None and exclude_samples:
        excluded = list(range(len(dataset)))
        random.Random(exclude_seed).shuffle(excluded)
        blocked = set(excluded[:exclude_samples])
        ids = [index for index in ids if index not in blocked]
    ids = ids[:min(samples, len(ids))]
    groups = {name: [] for name in BRANCH_FILES}
    for index in ids:
        row = dataset[index]
        image = row["image"].unsqueeze(0).to(device)
        text = row["text"]
        general_logits, context = moe.forward_general(image, texts=[text])
        reference_logits = moe.general(image, pooled_text=context.pooled_text)
        if not torch.allclose(general_logits, reference_logits, atol=1e-5, rtol=1e-5):
            raise AssertionError("Shared general path differs from frozen baseline")
        ratio = float((torch.sigmoid(general_logits) >= 0.5).float().mean())
        branch, _, _ = baseline.route_sample(text, ratio)
        groups[branch].append(index)

    results = []
    original_param_count = 0
    for branch, indices in groups.items():
        if not indices:
            continue
        old = baseline.load_seg_model(
                ROOT / "checkpoints" / BRANCH_FILES[branch], device,
        )
        if not original_param_count:
            expert_params = sum(p.numel() for p in old.parameters())
            bert_params = sum(p.numel() for p in old.text_encoder.parameters())
            original_param_count = (sum(p.numel() for p in moe.general.parameters())
                                    + 3 * (expert_params - bert_params))
        for index in indices:
            row = dataset[index]
            image = row["image"].unsqueeze(0).to(device)
            gt = row["mask"].unsqueeze(0).to(device) > 0.5
            _, context = moe.forward_general(image, texts=[row["text"]])
            original_logits = old(image, pooled_text=context.pooled_text)
            shared_logits = moe.forward_expert(context, branch)

            def stats(logits):
                pred = torch.sigmoid(logits) >= 0.5
                intersection = int((pred & gt).sum().item())
                union = int((pred | gt).sum().item())
                return (1.0 if union == 0 else intersection / union,
                        intersection, union)

            old_iou, old_intersection, old_union = stats(original_logits)
            shared_iou, shared_intersection, shared_union = stats(shared_logits)

            results.append({
                "dataset_index": index,
                "branch": branch,
                "gt_empty": not bool(gt.any().item()),
                "old_pred_empty": not bool((original_logits.sigmoid() >= 0.5).any().item()),
                "shared_pred_empty": not bool((shared_logits.sigmoid() >= 0.5).any().item()),
                "old_iou": old_iou,
                "shared_iou": shared_iou,
                "old_intersection": old_intersection,
                "old_union": old_union,
                "shared_intersection": shared_intersection,
                "shared_union": shared_union,
            })
            print(f"{len(results)}/{len(ids)} {branch}: "
                  f"old={results[-1]['old_iou']:.3f} shared={results[-1]['shared_iou']:.3f}", flush=True)
        del old
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "split": "val", "presence_gate": "not applied",
        "num_samples": len(results),
        "seed": seed,
        "excluded_selection": {"seed": exclude_seed, "samples": exclude_samples}
        if exclude_seed is not None else None,
        "expert_backbone": backbone or "general",
        "old_mean_iou": sum(x["old_iou"] for x in results) / len(results),
        "shared_mean_iou": sum(x["shared_iou"] for x in results) / len(results),
        "old_cIoU": sum(x["old_intersection"] for x in results) / max(sum(x["old_union"] for x in results), 1),
        "shared_cIoU": sum(x["shared_intersection"] for x in results) / max(sum(x["shared_union"] for x in results), 1),
        "shared_total_parameters": sum(p.numel() for p in moe.parameters()),
        "original_runtime_parameters_excluding_duplicate_bert": original_param_count,
        "shared_trainable_parameters": sum(p.numel() for p in moe.parameters() if p.requires_grad),
        "samples": results,
    }
    report["by_branch"] = {}
    for branch in BRANCH_FILES:
        branch_rows = [row for row in results if row["branch"] == branch]
        report["by_branch"][branch] = {
            "n": len(branch_rows),
            "old_mean_iou": sum(row["old_iou"] for row in branch_rows) / max(1, len(branch_rows)),
            "shared_mean_iou": sum(row["shared_iou"] for row in branch_rows) / max(1, len(branch_rows)),
            "old_cIoU": sum(row["old_intersection"] for row in branch_rows) /
                        max(1, sum(row["old_union"] for row in branch_rows)),
            "shared_cIoU": sum(row["shared_intersection"] for row in branch_rows) /
                           max(1, sum(row["shared_union"] for row in branch_rows)),
        }
    empty_rows = [row for row in results if row["gt_empty"]]
    report["no_target"] = {
        "n": len(empty_rows),
        "old_empty_prediction_rate": sum(row["old_pred_empty"] for row in empty_rows) / max(1, len(empty_rows)),
        "shared_empty_prediction_rate": sum(row["shared_pred_empty"] for row in empty_rows) / max(1, len(empty_rows)),
    }
    suffix = f"_holdout_{seed}" if exclude_seed is not None else ""
    out = ROOT / "results" / (
        f"shared_moe_{backbone}{suffix}_val_probe.json" if backbone
        else f"shared_moe{suffix}_val_probe.json"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-seed", type=int)
    parser.add_argument("--exclude-samples", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    run(args.samples, args.seed, args.exclude_seed, args.exclude_samples)
