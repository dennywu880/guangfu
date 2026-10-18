"""Describe which annotated Val cases improve or regress in the full system."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataset import RefELDataset


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "full_system_larger_val_comparison.json"


def summarize(rows):
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "old_mean_iou": sum(row["old_iou"] for row in rows) / len(rows),
        "shared_mean_iou": sum(row["shared_iou"] for row in rows) / len(rows),
        "old_cIoU": sum(row["old_intersection"] for row in rows) /
                    max(1, sum(row["old_union"] for row in rows)),
        "shared_cIoU": sum(row["shared_intersection"] for row in rows) /
                       max(1, sum(row["shared_union"] for row in rows)),
        "gt_pixels_median": sorted(row["gt_pixels"] for row in rows)[len(rows) // 2],
        "old_to_empty": sum(not row["old_empty"] and row["shared_empty"] for row in rows),
        "empty_to_detected": sum(row["old_empty"] and not row["shared_empty"] for row in rows),
    }


def run(source):
    report = json.loads(source.read_text(encoding="utf-8"))
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    instances = json.loads(dataset.instance_json.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): cat["name"] for cat in instances["categories"]}

    groups = defaultdict(list)
    enriched = []
    for row in report["rows"]:
        sample = dataset[row["dataset_index"]]
        ref = dataset.refs[row["dataset_index"]]
        gt_pixels = int(sample["mask"].sum().item())
        ann_ids = ref.get("ann_id", [])
        if not isinstance(ann_ids, list):
            ann_ids = [ann_ids]
        target_count = sum(int(ann_id) >= 0 for ann_id in ann_ids)
        cat_ids = ref.get("category_id", [])
        if not isinstance(cat_ids, list):
            cat_ids = [cat_ids]
        category = ", ".join(sorted({categories.get(int(cid), str(cid)) for cid in cat_ids})) or "none"
        delta = row["shared_iou"] - row["old_iou"]
        outcome = "better" if delta > 1e-6 else "worse" if delta < -1e-6 else "same"
        area = gt_pixels / (384 * 384)
        size = "no_target" if gt_pixels == 0 else (
            "ultra_tiny" if area < 0.001 else "tiny" if area < 0.006 else "larger"
        )
        item = {**row, "gt_pixels": gt_pixels, "gt_area_ratio": area,
                "target_count": target_count, "category": category,
                "expression": sample["text"], "file_name": sample["file_name"],
                "size": size, "delta_iou": delta, "outcome": outcome}
        enriched.append(item)
        for name, value in (("outcome", outcome), ("size", size),
                            ("route", row["branch"]), ("category", category),
                            ("target_count", str(target_count)),
                            ("gate", "rejected" if row["gate_reject"] else "passed")):
            groups[name, value].append(item)

    summary = {
        "source": str(source), "n": len(enriched),
        "outcomes": dict(Counter(item["outcome"] for item in enriched)),
        "groups": {name: {key: summarize(value) for (group, key), value in groups.items()
                          if group == name}
                   for name in ("outcome", "size", "route", "category", "target_count", "gate")},
        "largest_improvements": sorted(enriched, key=lambda item: item["delta_iou"], reverse=True)[:15],
        "largest_regressions": sorted(enriched, key=lambda item: item["delta_iou"])[:15],
        "cases": enriched,
    }
    output = ROOT / "results" / "shared_failure_analysis.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    concise = {name: {key: {metric: value[metric] for metric in
                            ("n", "old_mean_iou", "shared_mean_iou", "old_to_empty", "empty_to_detected")
                            if metric in value}
                      for key, value in group.items()}
               for name, group in summary["groups"].items()}
    print(json.dumps({"outcomes": summary["outcomes"], "groups": concise}, indent=2,
                     ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    run(args.source)
