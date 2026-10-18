"""Compare initial and selected checkpoints on the diagnosis-disjoint holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import RefELDataset
from multilevel_selection import build_model
from train_detail_adapter import (
    ROOT,
    evaluate,
    image_ids,
    load_or_create_split,
    load_router,
)
from train_text import FormalTextCache


def load_selection_model(checkpoint: Path, device: torch.device):
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    levels = tuple(data["levels"])
    model = build_model(device, levels)
    model.load_checkpoint(checkpoint)
    return model, levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument(
        "--diagnosis",
        type=Path,
        default=ROOT / "results" / "layerwise_encoder_diagnosis_20260925"
        / "layerwise_encoder_diagnosis.json",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    experiment = ROOT / "results" / args.experiment
    output = experiment / "holdout_evaluation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    val = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    cache = FormalTextCache(ROOT)
    dev_indices, holdout_indices = load_or_create_split(val, 20260924)
    if image_ids(val, dev_indices) & image_ids(val, holdout_indices):
        raise RuntimeError("Development and holdout images overlap")
    diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    diagnosis_images = {case["image_id"] for case in diagnosis["case_info"]}
    excluded = [
        index for index in holdout_indices
        if val.refs[index]["image_id"] in diagnosis_images
    ]
    clean_holdout = [
        index for index in holdout_indices
        if val.refs[index]["image_id"] not in diagnosis_images
    ]
    if not clean_holdout or image_ids(val, clean_holdout) & diagnosis_images:
        raise RuntimeError("No diagnosis-disjoint holdout remains")
    router = load_router()
    metrics_by_checkpoint = {}
    levels = None
    for checkpoint_name in ("initial.pth", args.checkpoint):
        model, levels = load_selection_model(experiment / checkpoint_name, device)
        metrics = evaluate(
            model, val, clean_holdout, router, device, cache, "val", args.batch_size
        )
        metrics_by_checkpoint[checkpoint_name] = {
            key: value for key, value in metrics.items() if key != "rows"
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "levels": levels,
        "device": str(device),
        "original_holdout_n": len(holdout_indices),
        "diagnosis_overlap_excluded_n": len(excluded),
        "diagnosis_overlap_excluded_indices": excluded,
        "evaluated_holdout_n": len(clean_holdout),
        "holdout_image_disjoint_from_development_and_diagnosis": True,
        "metrics_by_checkpoint": metrics_by_checkpoint,
        "note": (
            "Compared only after development checkpoint selection. Historical "
            "expert weights may have seen Val images; this is not a fresh blind test. "
            "No model or web deployment changed."
        ),
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
