"""Evaluate frozen independent and shared experts on the locked image split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_clean_general import CleanRefDataset, ROOT, load_split, sha256
from train_clean_experts import (
    GENERAL, build_candidate, evaluate, make_base, restore_candidate,
)
from train_detail_adapter import load_router
from train_text import FormalTextCache


def load_frozen_candidate(name: str, mode: str, device: torch.device):
    folder = ROOT / "results" / name
    checkpoint = torch.load(folder / "best.pth", map_location="cpu", weights_only=False)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if checkpoint["format"] != "clean_experts_v1" or checkpoint["metadata"] != manifest:
        raise ValueError(f"Checkpoint metadata mismatch: {name}")
    if manifest["mode"] != mode or manifest["general_sha256"] != sha256(GENERAL):
        raise ValueError(f"Wrong mode or general checkpoint: {name}")
    general = make_base(device).eval()
    candidate, _ = build_candidate(mode, general, device)
    restore_candidate(candidate, mode, checkpoint["candidate"])
    return candidate.eval(), general, manifest, checkpoint["epoch"]


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Locked test result already exists: {args.output}")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched test evaluation")
    device = torch.device("cuda")
    split = load_split()
    loader = DataLoader(
        CleanRefDataset("test", split["indices"]["test"]),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    cache = FormalTextCache(ROOT)
    router = load_router()
    results = {
        "scope": "locked source-image-disjoint test; no Presence Gate",
        "split_sha256": sha256(ROOT / "results" / "source_image_split_20260925" / "split.json"),
        "general_sha256": sha256(GENERAL),
        "test_references": len(loader.dataset),
        "candidates": {},
    }
    for mode, name in (("independent", args.independent), ("shared", args.shared)):
        candidate, general, manifest, best_epoch = load_frozen_candidate(name, mode, device)
        if manifest["split_sha256"] != results["split_sha256"]:
            raise ValueError(f"Wrong data split: {name}")
        metrics = evaluate(candidate, mode, general, router, loader, cache, device)
        results["candidates"][mode] = {
            "name": name, "checkpoint_sha256": sha256(ROOT / "results" / name / "best.pth"),
            "best_epoch": best_epoch, "metrics": metrics,
            "parameter_counts": manifest["parameter_counts"],
        }
    left = results["candidates"]["independent"]["metrics"]["rows"]
    right = results["candidates"]["shared"]["metrics"]["rows"]
    if {row["ref_id"] for row in left} != {row["ref_id"] for row in right}:
        raise ValueError("Candidates were not evaluated on the same test references")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for mode, candidate in results["candidates"].items():
        metrics = candidate["metrics"]
        print(json.dumps({"mode": mode, "best_epoch": candidate["best_epoch"],
                          "gIoU": metrics["mean_iou"], "cIoU": metrics["ciou"],
                          "positive_mean_iou": metrics["positive_mean_iou"],
                          "no_target_accuracy": metrics["no_target_accuracy"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent", default="clean_experts_independent_source_disjoint")
    parser.add_argument("--shared", default="clean_experts_shared_source_disjoint")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results" /
                        "clean_experts_locked_test.json")
    options = parser.parse_args()
    if options.batch_size < 1:
        parser.error("Batch size must be positive")
    run(options)
