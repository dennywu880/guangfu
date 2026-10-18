"""Layer-wise image-encoder swap diagnosis for the shared TG-UNet MoE.

This is a frozen diagnostic, not a training or deployment script.  The text,
general model, gate, route, expert head, threshold and image split are fixed.
For each routed case, only one encoder feature level (e0..e4) is replaced by
the corresponding feature from another encoder.  Results are written to a new
directory and the script refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import RefELDataset
from inference_engine import BASELINE, DefectInferenceEngine
from shared_moe import SharedFeatures, SharedMoE


ROOT = Path(__file__).resolve().parent
SOURCES = ("general", "ultra_tiny", "tiny", "larger")
LEVELS = ("e0", "e1", "e2", "e3", "e4")


def shuffled(length: int, seed: int):
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices


def features_from(model, image):
    e0 = model.stem(image)
    e1 = model.layer1(model.maxpool(e0))
    e2 = model.layer2(e1)
    e3 = model.layer3(e2)
    e4 = model.layer4(e3)
    return (e0, e1, e2, e3, e4)


def binary_row(probability, gt, case_info, variant):
    prediction = probability >= BASELINE.EXPERT_MASK_THRESHOLD
    result = BASELINE.binary_stats(prediction, gt)
    gt_empty = not bool(gt.any().item())
    gt_ratio = float(gt.float().mean().item())
    if gt_ratio < 0.001:
        area_group = "ultra_tiny"
    elif gt_ratio < 0.006:
        area_group = "tiny"
    else:
        area_group = "larger"
    return {
        "index": case_info["index"],
        "image_id": case_info["image_id"],
        "branch": case_info["branch"],
        "gt_empty": gt_empty,
        "gt_area_ratio": gt_ratio,
        "area_group": area_group,
        "variant": variant,
        "intersection": result["intersection"],
        "union": result["union"],
        "iou": result["iou_regular"],
        "pred_empty": result["pred_empty"],
        "mean_gt_probability": (
            float(probability[gt].mean().item()) if bool(gt.any().item()) else None
        ),
    }


def summarize(rows):
    if not rows:
        return {"n": 0}
    positives = [row for row in rows if not row["gt_empty"]]
    no_target = [row for row in rows if row["gt_empty"]]

    def summary(part):
        if not part:
            return {"n": 0}
        return {
            "n": len(part),
            "mean_IoU": sum(row["iou"] for row in part) / len(part),
            "cIoU": sum(row["intersection"] for row in part)
            / max(1, sum(row["union"] for row in part)),
            "pred_empty_count": sum(row["pred_empty"] for row in part),
        }

    groups = {}
    for group in ("ultra_tiny", "tiny", "larger"):
        groups[group] = summary([row for row in positives if row["area_group"] == group])

    target_probabilities = [
        row["mean_gt_probability"] for row in positives
        if row["mean_gt_probability"] is not None
    ]
    return {
        "n": len(rows),
        "positive_n": len(positives),
        "no_target_n": len(no_target),
        "mean_IoU": sum(row["iou"] for row in rows) / len(rows),
        "positive_mean_IoU": (
            sum(row["iou"] for row in positives) / max(1, len(positives))
        ),
        "cIoU": sum(row["intersection"] for row in rows)
        / max(1, sum(row["union"] for row in rows)),
        "positive_empty_count": sum(row["pred_empty"] for row in positives),
        "no_target_false_activation_count": sum(
            not row["pred_empty"] for row in no_target
        ),
        "mean_probability_on_target": (
            sum(target_probabilities) / len(target_probabilities)
            if target_probabilities else None
        ),
        "by_area_group": groups,
    }


def make_selection(dataset, count, seed):
    # Keep the exact image-disjoint exclusions used by the earlier diagnosis.
    first_120 = set(shuffled(len(dataset), 42)[:120])
    previous_500 = [
        index for index in shuffled(len(dataset), 314159)
        if index not in first_120
    ][:500]
    excluded_images = {
        dataset.refs[index]["image_id"]
        for index in first_120 | set(previous_500)
    }
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
    return selected, excluded_images


@torch.inference_mode()
def run(count: int, seed: int, output: Path, smoke: bool = False):
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic directory: {output}"
        )
    output.mkdir(parents=True)

    os.environ["TG_SHARED_MOE"] = "0"
    torch.set_num_threads(4)
    engine = DefectInferenceEngine()
    shared = SharedMoE.from_checkpoints(
        ROOT / "checkpoints", engine.device, general=engine.general_model
    ).eval()
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    selected_indices, excluded_images = make_selection(dataset, count, seed)

    # Expert encoders are the already loaded frozen expert checkpoints.  They
    # are moved one at a time to the inference device to keep memory bounded.
    expert_models = engine.experts
    records = defaultdict(lambda: defaultdict(list))
    feature_alignment = defaultdict(lambda: defaultdict(list))
    case_info = []
    gate_rejections = 0
    reconstruction_differences = {}

    for step, index in enumerate(selected_indices, 1):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(engine.device)
        gt = (item["mask"].unsqueeze(0).to(engine.device) > 0.5)

        engine.hook.value = None
        general_logits, general_context = shared.forward_general(
            image, texts=[item["text"]]
        )
        pooled = general_context.pooled_text
        general_features = general_context.image
        image_feature = engine._image_feature_from_hook(engine.hook.value).float()
        numbers, values = engine._general_stats(general_logits.sigmoid())
        branch, _, size = BASELINE.route_sample(item["text"], values["area050"])
        gate_input = torch.cat(
            [
                image_feature,
                pooled.float(),
                numbers.float(),
                torch.tensor(
                    [BASELINE.route_onehot(branch)], device=engine.device
                ),
                torch.tensor([BASELINE.text_onehot(size)], device=engine.device),
            ],
            dim=1,
        )
        rejected = bool(
            (
                engine.gate(
                    (gate_input - engine.gate_mean) / engine.gate_std
                ).sigmoid()
                >= engine.gate_threshold
            ).item()
        )
        info = {
            "index": index,
            "image_id": item["image_id"],
            "branch": branch,
            "gt_empty": not bool(gt.any().item()),
            "gate_rejected": rejected,
        }
        case_info.append(info)
        if rejected:
            gate_rejections += 1
            continue

        # Only compare the routed expert's own encoder to General.  Other
        # experts would introduce an additional, hard-to-interpret source.
        source_features = {"general": general_features}
        expert_model = expert_models[branch].to(engine.device).eval()
        source_features[branch] = features_from(expert_model, image)
        for level, general_feature, expert_feature in zip(
            LEVELS, general_features, source_features[branch]
        ):
            general_flat = general_feature.float().flatten(1)
            expert_flat = expert_feature.float().flatten(1)
            general_norm = general_flat.norm(dim=1).clamp_min(1e-8)
            expert_norm = expert_flat.norm(dim=1).clamp_min(1e-8)
            feature_alignment[branch][level].append({
                "index": index,
                "image_id": item["image_id"],
                "cosine": float(F.cosine_similarity(
                    general_flat, expert_flat, dim=1
                ).item()),
                "expert_to_general_norm_ratio": float(
                    (expert_norm / general_norm).item()
                ),
                "relative_rmse": float(
                    ((expert_flat - general_flat).square().mean(dim=1).sqrt()
                     / general_flat.square().mean(dim=1).sqrt().clamp_min(1e-8)).item()
                ),
            })

        # Validate full reassembly once per branch: the shared head plus the
        # branch's complete original encoder must reproduce the expert output.
        own_features = source_features[branch]
        own_context = SharedFeatures(own_features, pooled, image.shape[-2:])
        reassembled = SharedMoE._decode(
            shared.heads[branch], own_context
        ).sigmoid()
        direct = expert_model(image, pooled_text=pooled).sigmoid()
        diff = float((reassembled - direct).abs().max().item())
        if diff > 1e-5:
            raise RuntimeError(
                f"Reassembled {branch} expert differs from checkpoint: {diff}"
            )
        reconstruction_differences[branch] = max(
            diff, reconstruction_differences.get(branch, 0.0)
        )
        expert_model.to("cpu")
        torch.cuda.empty_cache()

        case = {"index": index, "image_id": item["image_id"], "branch": branch}
        records[branch]["all_own"].append(
            binary_row(reassembled, gt, case, "all_own")
        )
        variants = [
            ("all_general", general_features),
            *[
                (
                    f"swap_{level}_from_own",
                    tuple(
                        source_features[branch][i] if i == level_index else general_features[i]
                        for i in range(len(LEVELS))
                    ),
                )
                for level_index, level in enumerate(LEVELS)
            ],
        ]
        # Cumulative replacement isolates whether the damage accumulates
        # monotonically as more shared/general features enter the expert head.
        # prefix_ek keeps the routed expert's features through ek and uses the
        # General features for all deeper levels.  This complements the
        # single-level intervention above; it does not prove causality by
        # itself because upstream features are not independent.
        for last_index, level in enumerate(LEVELS):
            variants.append(
                (
                    f"prefix_{level}_from_own",
                    tuple(
                        source_features[branch][i]
                        if i <= last_index else general_features[i]
                        for i in range(len(LEVELS))
                    ),
                )
            )
        for variant, selected_features in variants:
            context = SharedFeatures(selected_features, pooled, image.shape[-2:])
            probability = SharedMoE._decode(shared.heads[branch], context).sigmoid()
            records[branch][variant].append(
                binary_row(probability, gt, case, variant)
            )

        if step % 5 == 0 or step == len(selected_indices):
            progress = {
                "processed": step,
                "requested": len(selected_indices),
                "gate_rejections": gate_rejections,
                "routed_cases": len(case_info) - gate_rejections,
            }
            (output / "progress.json").write_text(
                json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"Layerwise diagnosis {step}/{len(selected_indices)}; "
                f"routed={len(case_info)-gate_rejections}, rejected={gate_rejections}",
                flush=True,
            )

    results = {
        branch: {
            variant: summarize(rows)
            for variant, rows in variants.items()
        }
        for branch, variants in records.items()
    }
    alignment_summary = {
        branch: {
            level: {
                "n": len(rows),
                "mean_cosine": sum(row["cosine"] for row in rows) / len(rows),
                "mean_norm_ratio": sum(
                    row["expert_to_general_norm_ratio"] for row in rows
                ) / len(rows),
                "mean_relative_rmse": sum(
                    row["relative_rmse"] for row in rows
                ) / len(rows),
            }
            for level, rows in levels.items()
        }
        for branch, levels in feature_alignment.items()
    }
    report = {
        "split": "val",
        "seed": seed,
        "requested_count": len(selected_indices),
        "routed_count": len(case_info) - gate_rejections,
        "gate_rejections": gate_rejections,
        "image_disjoint": True,
        "excluded_previous_images": len(excluded_images),
        "fixed": (
            "same image/text, General model, Presence Gate, frozen route, "
            "expert head and 0.5 mask threshold"
        ),
        "intervention": (
            "for each routed sample, only one e0/e1/e2/e3/e4 feature is "
            "replaced, plus cumulative expert-prefix replacements; "
            "gate-rejected samples are reported separately"
        ),
        "levels": list(LEVELS),
        "sources": "General and each routed expert's own encoder",
        "own_encoder_reassembly_max_abs_difference": reconstruction_differences,
        "results_by_head": results,
        "feature_alignment_summary": alignment_summary,
        "feature_alignment_rows": {
            branch: dict(levels) for branch, levels in feature_alignment.items()
        },
        "case_info": case_info,
        "raw_rows": {branch: dict(variants) for branch, variants in records.items()},
        "note": (
            "This is a diagnostic on a historical Val split. It is not an "
            "independent blind test and does not authorize deployment."
        ),
    }
    (output / "layerwise_encoder_diagnosis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {output / 'layerwise_encoder_diagnosis.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=98765)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "layerwise_encoder_diagnosis_20260925",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.samples, args.seed, args.output, args.smoke)
