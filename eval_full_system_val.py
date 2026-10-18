"""Paired full-system Val comparison using the same General, Gate and Router."""

import argparse
import json
import os
import random
from pathlib import Path

import torch

from dataset import RefELDataset
from inference_engine import BASELINE, DefectInferenceEngine
from shared_moe import BRANCH_FILES, SharedMoE


ROOT = Path(__file__).resolve().parent


def select_indices(length, count, seed):
    values = list(range(length))
    random.Random(seed).shuffle(values)
    return values[:count]


def metric(rows, key):
    return {
        "n": len(rows),
        "cIoU": sum(row[f"{key}_intersection"] for row in rows) /
                max(1, sum(row[f"{key}_union"] for row in rows)),
        "mean_IoU": sum(row[f"{key}_iou"] for row in rows) / max(1, len(rows)),
        "no_target_empty_rate": sum(row[f"{key}_empty"] for row in rows if row["gt_empty"]) /
                                max(1, sum(row["gt_empty"] for row in rows)),
    }


@torch.inference_mode()
def run(count, seed):
    torch.set_num_threads(4)
    os.environ["TG_SHARED_MOE"] = "0"
    engine = DefectInferenceEngine()
    shared = SharedMoE.from_checkpoints(
        ROOT / "checkpoints", engine.device,
        general=engine.general_model, expert_backbone="larger",
    )
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)

    selected_120 = set(select_indices(len(dataset), 120, 42))
    candidate_500 = select_indices(len(dataset), len(dataset), 314159)
    selected_500 = set(index for index in candidate_500 if index not in selected_120)
    selected_500 = set(list(index for index in candidate_500 if index not in selected_120)[:500])
    blocked = selected_120 | selected_500
    indices = [index for index in select_indices(len(dataset), len(dataset), seed)
               if index not in blocked][:count]

    records = {}
    routes = {name: [] for name in BRANCH_FILES}
    for step, index in enumerate(indices, 1):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(engine.device)
        gt = item["mask"].unsqueeze(0).to(engine.device) > 0.5
        engine.hook.value = None
        general, context = shared.forward_general(image, texts=[item["text"]])
        if engine.hook.value is None:
            raise RuntimeError("Presence Gate image feature hook did not fire")
        image_feature = engine._image_feature_from_hook(engine.hook.value).float()
        stats, values = engine._general_stats(general.sigmoid())
        branch, _, text_size = BASELINE.route_sample(item["text"], values["area050"])
        features = torch.cat([
            image_feature, context.pooled_text.float(), stats.float(),
            torch.tensor([BASELINE.route_onehot(branch)], device=engine.device),
            torch.tensor([BASELINE.text_onehot(text_size)], device=engine.device),
        ], dim=1)
        reject = bool((engine.gate((features-engine.gate_mean)/engine.gate_std)
                       .sigmoid() >= engine.gate_threshold).item())
        records[index] = {"dataset_index": index, "branch": branch,
                          "gate_reject": reject, "gt_empty": not bool(gt.any().item()),
                          "pooled_text": context.pooled_text.detach().cpu()}
        if reject:
            stats_zero = BASELINE.binary_stats(torch.zeros_like(gt), gt)
            for name in ("old", "shared"):
                records[index].update({
                    f"{name}_intersection": stats_zero["intersection"],
                    f"{name}_union": stats_zero["union"],
                    f"{name}_iou": stats_zero["iou_regular"],
                    f"{name}_empty": True,
                })
        else:
            routes[branch].append(index)
        if step % 100 == 0:
            print(f"General/Gate {step}/{len(indices)}", flush=True)

    for branch, branch_indices in routes.items():
        if not branch_indices:
            continue
        teacher = engine._activate_expert(branch)
        for step, index in enumerate(branch_indices, 1):
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(engine.device)
            gt = item["mask"].unsqueeze(0).to(engine.device) > 0.5
            pooled_text = records[index]["pooled_text"].to(engine.device)
            _, context = shared.forward_general(image, pooled_text=pooled_text)
            predictions = {
                "old": teacher(image, pooled_text=pooled_text),
                "shared": shared.forward_expert(context, branch),
            }
            for name, logits in predictions.items():
                pred = logits.sigmoid() >= BASELINE.EXPERT_MASK_THRESHOLD
                stats = BASELINE.binary_stats(pred, gt)
                records[index].update({
                    f"{name}_intersection": stats["intersection"],
                    f"{name}_union": stats["union"],
                    f"{name}_iou": stats["iou_regular"],
                    f"{name}_empty": stats["pred_empty"],
                })
            if step % 100 == 0:
                print(f"Expert {branch} {step}/{len(branch_indices)}", flush=True)
    rows = [records[index] for index in indices]
    for row in rows:
        del row["pooled_text"]
    report = {
        "split": "val", "samples": len(rows), "seed": seed,
        "excluded_previous_validation": len(blocked),
        "gate_rejections": sum(row["gate_reject"] for row in rows),
        "old": metric(rows, "old"), "shared_larger": metric(rows, "shared"),
        "by_branch": {
            branch: {name: metric([row for row in rows if row["branch"] == branch], name)
                     for name in ("old", "shared")}
            for branch in BRANCH_FILES
        },
        "rows": rows,
    }
    out = ROOT / "results" / "full_system_larger_val_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=271828)
    options = parser.parse_args()
    if options.samples < 1:
        parser.error("samples must be positive")
    run(options.samples, options.seed)
