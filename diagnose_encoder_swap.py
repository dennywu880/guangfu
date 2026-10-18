"""Hold expert heads fixed while swapping image encoders on unseen Val images."""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch

from dataset import RefELDataset
from inference_engine import BASELINE, DefectInferenceEngine
from shared_moe import BRANCH_FILES, SharedFeatures, SharedMoE


ROOT = Path(__file__).resolve().parent
CANDIDATES = ("general", "ultra_tiny", "tiny", "larger")


def shuffled(length, seed):
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices


def features_from(model, image):
    e0 = model.stem(image)
    e1 = model.layer1(model.maxpool(e0))
    e2 = model.layer2(e1)
    e3 = model.layer3(e2)
    e4 = model.layer4(e3)
    return e0, e1, e2, e3, e4


def stats(pred, gt):
    result = BASELINE.binary_stats(pred, gt)
    return {"intersection": result["intersection"],
            "union": result["union"], "iou": result["iou_regular"],
            "pred_empty": result["pred_empty"]}


def summarize(rows):
    if not rows:
        return {"n": 0}
    positives = [row for row in rows if not row["gt_empty"]]
    no_target = [row for row in rows if row["gt_empty"]]
    return {
        "n": len(rows), "positive_n": len(positives), "no_target_n": len(no_target),
        "mean_IoU": sum(row["iou"] for row in rows) / len(rows),
        "positive_mean_IoU": sum(row["iou"] for row in positives) / max(1, len(positives)),
        "cIoU": sum(row["intersection"] for row in rows) /
                max(1, sum(row["union"] for row in rows)),
        "positive_empty_count": sum(row["pred_empty"] for row in positives),
        "no_target_false_activation_count": sum(not row["pred_empty"] for row in no_target),
        "mean_probability_on_target": sum(row["mean_gt_probability"] for row in positives) /
                                      max(1, len(positives)),
    }


@torch.inference_mode()
def run(count, seed):
    os.environ["TG_SHARED_MOE"] = "0"
    torch.set_num_threads(4)
    engine = DefectInferenceEngine()
    shared = SharedMoE.from_checkpoints(ROOT / "checkpoints", engine.device,
                                         general=engine.general_model)
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    # The earlier 500-reference audit excluded the first 120 references.
    first_120 = set(shuffled(len(dataset), 42)[:120])
    previous_500 = [index for index in shuffled(len(dataset), 314159)
                    if index not in first_120][:500]
    excluded_images = {dataset.refs[index]["image_id"]
                       for index in first_120 | set(previous_500)}
    selected = []
    seen_images = set()
    for index in shuffled(len(dataset), seed):
        image_id = dataset.refs[index]["image_id"]
        if image_id not in excluded_images and image_id not in seen_images:
            selected.append(index)
            seen_images.add(image_id)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError("Not enough image-disjoint Val cases")

    routes = defaultdict(list)
    case_info = {}
    for step, index in enumerate(selected, 1):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(engine.device)
        engine.hook.value = None
        general, context = shared.forward_general(image, texts=[item["text"]])
        image_feature = engine._image_feature_from_hook(engine.hook.value).float()
        numbers, values = engine._general_stats(general.sigmoid())
        branch, _, size = BASELINE.route_sample(item["text"], values["area050"])
        gate_input = torch.cat([
            image_feature, context.pooled_text.float(), numbers.float(),
            torch.tensor([BASELINE.route_onehot(branch)], device=engine.device),
            torch.tensor([BASELINE.text_onehot(size)], device=engine.device),
        ], dim=1)
        reject = bool((engine.gate((gate_input - engine.gate_mean) / engine.gate_std)
                       .sigmoid() >= engine.gate_threshold).item())
        gt_empty = not bool((item["mask"] > 0.5).any().item())
        case_info[index] = {"branch": branch, "gt_empty": gt_empty,
                            "gate_rejected": reject}
        if not reject:
            routes[branch].append((index, context.pooled_text.detach().cpu()))
        if step % 100 == 0:
            print(f"General/Gate {step}/{len(selected)}", flush=True)

    by_head = {}
    own_encoder_max_abs_difference = {}
    for branch, items in routes.items():
        by_head[branch] = {}
        for encoder_name in CANDIDATES:
            encoder = (engine.general_model if encoder_name == "general"
                       else engine.experts[encoder_name].to(engine.device))
            entries = []
            for index, pooled_cpu in items:
                item = dataset[index]
                image = item["image"].unsqueeze(0).to(engine.device)
                gt = item["mask"].unsqueeze(0).to(engine.device) > 0.5
                pooled = pooled_cpu.to(engine.device)
                context = SharedFeatures(features_from(encoder, image), pooled,
                                         image.shape[-2:])
                prob = SharedMoE._decode(shared.heads[branch], context).sigmoid()
                if encoder_name == branch and index == items[0][0]:
                    direct = encoder(image, pooled_text=pooled).sigmoid()
                    difference = float((prob - direct).abs().max().item())
                    own_encoder_max_abs_difference[branch] = difference
                    if difference > 1e-5:
                        raise RuntimeError(
                            f"Reassembled {branch} expert differs from checkpoint: {difference}")
                mask = prob >= 0.5
                row = {"index": index, "gt_empty": case_info[index]["gt_empty"],
                       **stats(mask, gt),
                       "mean_gt_probability": float(prob[gt].mean().item())
                       if bool(gt.any().item()) else None}
                entries.append(row)
            by_head[branch][encoder_name] = summarize(entries)
            print(f"{branch} head + {encoder_name} encoder: "
                  f"{by_head[branch][encoder_name]}", flush=True)
            if encoder_name != "general":
                encoder.to("cpu")
                torch.cuda.empty_cache()

    report = {
        "split": "val", "seed": seed, "count": len(selected),
        "image_disjoint": True, "excluded_previous_images": len(excluded_images),
        "gate_rejections": sum(info["gate_rejected"] for info in case_info.values()),
        "own_encoder_max_abs_difference": own_encoder_max_abs_difference,
        "fixed": "same text, general model, gate, route, expert head and 0.5 mask threshold",
        "intervention": "only the image encoder changes",
        "results_by_head": by_head,
    }
    output = ROOT / "results" / "encoder_swap_diagnosis.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=98765)
    args = parser.parse_args()
    run(args.samples, args.seed)
