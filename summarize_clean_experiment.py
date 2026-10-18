"""Create paired, source-image-grouped uncertainty summaries for clean runs."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def paired_bootstrap(left_rows, right_rows, rounds=2000, seed=20260929):
    left = {row["ref_id"]: row for row in left_rows}
    right = {row["ref_id"]: row for row in right_rows}
    if left.keys() != right.keys():
        raise ValueError("Candidate reference IDs differ")
    for ref_id in left:
        if (left[ref_id]["image_id"] != right[ref_id]["image_id"]
                or left[ref_id]["bucket"] != right[ref_id]["bucket"]
                or left[ref_id]["branch"] != right[ref_id]["branch"]):
            raise ValueError(f"Image, bucket, or route mismatch at ref_id={ref_id}")
    images = defaultdict(list)
    for ref_id, row in left.items():
        images[row["image_id"]].append(ref_id)
    image_ids = list(images)
    rng = random.Random(seed)
    mean_deltas, ciou_deltas = [], []
    for _ in range(rounds):
        sampled = [rng.choice(image_ids) for _ in image_ids]
        ids = [ref_id for image_id in sampled for ref_id in images[image_id]]
        mean_deltas.append(
            sum(left[i]["iou"] - right[i]["iou"] for i in ids) / len(ids)
        )
        left_intersection = sum(left[i]["intersection"] for i in ids)
        left_union = sum(left[i]["union"] for i in ids)
        right_intersection = sum(right[i]["intersection"] for i in ids)
        right_union = sum(right[i]["union"] for i in ids)
        ciou_deltas.append(
            left_intersection / max(1, left_union)
            - right_intersection / max(1, right_union)
        )

    def interval(values):
        ordered = sorted(values)
        return [ordered[int(0.025 * (len(ordered) - 1))],
                ordered[int(0.975 * (len(ordered) - 1))]]

    return {
        "left_minus_right": "shared minus independent",
        "mean_iou_delta_95pct": interval(mean_deltas),
        "cIoU_delta_95pct": interval(ciou_deltas),
        "resampling_unit": "source image",
        "rounds": rounds,
        "unique_images": len(image_ids),
        "paired_references": len(left),
    }


def run(args):
    result_path = ROOT / "results" / args.test_result
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = result["candidates"]
    ci = paired_bootstrap(candidates["shared"]["metrics"]["rows"],
                          candidates["independent"]["metrics"]["rows"],
                          args.rounds, args.seed)
    summary = {
        "dataset": result["scope"],
        "split_sha256": result["split_sha256"],
        "general_sha256": result["general_sha256"],
        "reference_count": result["test_references"],
        "paired_bootstrap": ci,
        "candidates": {
            name: {key: value for key, value in candidate.items()
                   if key != "metrics"}
            | {"metrics": {key: value for key, value in candidate["metrics"].items()
                           if key != "rows"}}
            for name, candidate in candidates.items()
        },
    }
    destination = ROOT / "results" / args.output
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-result", default="clean_experts_locked_test_20260929.json")
    parser.add_argument("--output", default="clean_experts_locked_test_summary_20260929.json")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260929)
    run(parser.parse_args())
