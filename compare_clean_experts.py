"""Paired development comparison for source-image-disjoint expert baselines."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read_candidate(name):
    folder = RESULTS / name
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = folder / "best.pth"
    rows = json.loads((folder / "best_dev_rows.json").read_text(encoding="utf-8"))
    history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    if not checkpoint.is_file() or not rows or not history:
        raise ValueError(f"Candidate is incomplete: {folder}")
    best_row = max(history, key=lambda row: row["dev"]["mean_iou"])
    return manifest, best_row, {row["ref_id"]: row for row in rows}


def summarize(rows):
    if not rows:
        return {"n": 0, "mean_iou": None, "ciou": None}
    union = sum(row["union"] for row in rows)
    return {"n": len(rows),
            "mean_iou": sum(row["iou"] for row in rows) / len(rows),
            "ciou": sum(row["intersection"] for row in rows) / union if union else 1.0}


def grouped_bootstrap(left, right, rounds, seed):
    images = defaultdict(list)
    for ref_id, row in left.items():
        images[row["image_id"]].append(ref_id)
    image_ids = list(images)
    rng = random.Random(seed)
    mean_deltas = []
    ciou_deltas = []
    for _ in range(rounds):
        sampled = [rng.choice(image_ids) for _ in image_ids]
        refs = [ref_id for image_id in sampled for ref_id in images[image_id]]
        left_rows = [left[ref_id] for ref_id in refs]
        right_rows = [right[ref_id] for ref_id in refs]
        mean_deltas.append(summarize(left_rows)["mean_iou"] -
                           summarize(right_rows)["mean_iou"])
        ciou_deltas.append(summarize(left_rows)["ciou"] -
                           summarize(right_rows)["ciou"])
    def interval(values):
        ordered = sorted(values)
        return [ordered[int(0.025 * (len(ordered) - 1))],
                ordered[int(0.975 * (len(ordered) - 1))]]
    return {"mean_iou_delta_95pct": interval(mean_deltas),
            "ciou_delta_95pct": interval(ciou_deltas),
            "resampling_unit": "source image", "rounds": rounds}


def run(args):
    left_meta, left_best, left = read_candidate(args.left)
    right_meta, right_best, right = read_candidate(args.right)
    for key in ("split_sha256", "general_sha256", "cache_sha256", "seed",
                "batch_size", "train_references", "dev_references"):
        if left_meta[key] != right_meta[key]:
            raise ValueError(f"Candidates differ on {key}")
    if set(left) != set(right):
        raise ValueError("Development reference IDs do not match")
    for ref_id, row in left.items():
        other = right[ref_id]
        for key in ("image_id", "bucket", "branch"):
            if row[key] != other[key]:
                raise ValueError(f"Development mismatch at {ref_id}: {key}")
    left_trace = {row["epoch"]: row["train_trace_sha256"]
                  for row in json.loads((RESULTS / args.left / "history.json").read_text())}
    right_trace = {row["epoch"]: row["train_trace_sha256"]
                   for row in json.loads((RESULTS / args.right / "history.json").read_text())}
    common_epochs = sorted(set(left_trace) & set(right_trace))
    if not common_epochs:
        raise ValueError("Candidates have no common trained epoch")
    mismatched_epochs = [epoch for epoch in common_epochs
                         if left_trace[epoch] != right_trace[epoch]]
    buckets = ("no_target", "ultra_tiny", "tiny", "larger")
    groups = {}
    for name in ("all", *buckets):
        ids = [ref_id for ref_id, row in left.items()
               if name == "all" or row["bucket"] == name]
        a = summarize([left[ref_id] for ref_id in ids])
        b = summarize([right[ref_id] for ref_id in ids])
        groups[name] = {"left": a, "right": b,
                        "mean_iou_delta": (a["mean_iou"] - b["mean_iou"]
                                           if ids else None),
                        "ciou_delta": (a["ciou"] - b["ciou"] if ids else None)}
    result = {
        "scope": "development only; locked test not read",
        "left": {"name": args.left, "mode": left_meta["mode"],
                 "best_epoch": left_best["epoch"],
                 "parameter_counts": left_meta["parameter_counts"]},
        "right": {"name": args.right, "mode": right_meta["mode"],
                  "best_epoch": right_best["epoch"],
                  "parameter_counts": right_meta["parameter_counts"]},
        "matched_training_epochs": common_epochs,
        "mismatched_training_trace_epochs": mismatched_epochs,
        "training_trace_note": "A mismatch precludes an exact matched-training claim; "
                               "development reference IDs, image IDs and routes "
                               "are checked separately above.",
        "trained_epochs": {"left": sorted(left_trace), "right": sorted(right_trace)},
        "matched_development_refs": len(left),
        "matched_development_images": len({row["image_id"] for row in left.values()}),
        "groups": groups,
        "paired_bootstrap": grouped_bootstrap(left, right, args.rounds, args.seed),
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Comparison already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="clean_experts_shared_source_disjoint")
    parser.add_argument("--right", default="clean_experts_independent_source_disjoint")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.rounds < 100:
        parser.error("At least 100 bootstrap rounds are required")
    run(options)
