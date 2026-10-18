"""Paired, source-image-clustered bootstrap for frozen testB comparisons."""

import csv
import json
from pathlib import Path

import numpy as np

from dataset import RefELDataset


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "shared_testB_comparison_20260925"
FILES = {
    "independent": ROOT / "results/final_testB_gate_v2/final_testB_gate_v2_samples.csv",
    "multilevel_best": OUTPUT / "multilevel_best_samples.csv",
    "head_joint_best": OUTPUT / "head_joint_best_samples.csv",
    "distill_dev_ciou_best": OUTPUT / "distill_dev_ciou_best_samples.csv",
}
PAIRS = [
    ("distill_dev_ciou_best", "head_joint_best"),
    ("distill_dev_ciou_best", "multilevel_best"),
    ("head_joint_best", "multilevel_best"),
    ("distill_dev_ciou_best", "independent"),
]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {int(row["ref_id"]): row for row in csv.DictReader(stream)}


def main():
    dataset = RefELDataset(ROOT, split="testB", image_size=384, random_text=False)
    image_id = {int(ref["ref_id"]): int(ref["image_id"]) for ref in dataset.refs}
    all_rows = {name: read_rows(path) for name, path in FILES.items()}
    refs = sorted(all_rows["independent"])
    if len(refs) != 7000 or any(set(rows) != set(refs) for rows in all_rows.values()):
        raise RuntimeError("Candidate reference IDs differ")
    images = sorted(set(image_id.values()))
    index = {key: i for i, key in enumerate(images)}
    cluster = np.array([index[image_id[ref]] for ref in refs])
    data = {}
    for name, rows in all_rows.items():
        ordered = [rows[ref] for ref in refs]
        intersection = np.array([int(row["final_intersection"]) for row in ordered])
        union = np.array([int(row["final_union"]) for row in ordered])
        giou = np.array([float(row["final_iou"]) if row["is_positive"] == "True"
                         else float(row["final_pred_empty"] == "True") for row in ordered])
        data[name] = np.stack([
            np.bincount(cluster, weights=intersection, minlength=len(images)),
            np.bincount(cluster, weights=union, minlength=len(images)),
            np.bincount(cluster, weights=giou, minlength=len(images)),
            np.bincount(cluster, minlength=len(images)),
        ], axis=1)
    rng = np.random.default_rng(20260925)
    draws = rng.integers(0, len(images), size=(1000, len(images)))

    def metrics(name):
        summed = data[name][draws].sum(axis=1)
        return {
            "cIoU": summed[:, 0] / summed[:, 1],
            "gIoU": summed[:, 2] / summed[:, 3],
        }

    bootstrap = {name: metrics(name) for name in data}
    comparisons = {}
    for left, right in PAIRS:
        comparisons[f"{left}_minus_{right}"] = {
            key: {
                "mean_delta": float(np.mean(bootstrap[left][key] - bootstrap[right][key])),
                "cluster_bootstrap_95_percent_interval": np.quantile(
                    bootstrap[left][key] - bootstrap[right][key], [0.025, 0.975]
                ).tolist(),
            }
            for key in ("cIoU", "gIoU")
        }
    result = {
        "protocol": "1000 paired bootstrap draws clustered by original image_id; not independent test proof",
        "source_images": len(images),
        "references": len(refs),
        "seed": 20260925,
        "comparisons": comparisons,
    }
    path = OUTPUT / "paired_bootstrap.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
