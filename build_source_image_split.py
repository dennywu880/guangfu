"""Build a deterministic source-image-disjoint Ref-EL protocol.

This only writes an index manifest. It does not move images, annotations or
model weights. Existing checkpoints are not valid blind-test models for this
protocol because their original training split had image overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def bucket(image_id: int, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "dev"
    return "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output", default="source_image_split_20260925")
    args = parser.parse_args()
    refs_path = ROOT / "datasets" / "grefcoco" / "grefs(unc).json"
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for index, ref in enumerate(refs):
        by_image[int(ref["image_id"])].append(index)
    indices = {"train": [], "dev": [], "test": []}
    image_ids = {name: [] for name in indices}
    for image_id, ref_indices in sorted(by_image.items()):
        name = bucket(image_id, args.seed)
        image_ids[name].append(image_id)
        indices[name].extend(ref_indices)
    for name in indices:
        indices[name].sort()
    overlap = {
        f"{left}_{right}": sorted(set(image_ids[left]) & set(image_ids[right]))
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(overlap.values()):
        raise RuntimeError("Source-image split overlap detected")
    summary = {}
    for name in indices:
        selected = [refs[index] for index in indices[name]]
        summary[name] = {
            "references": len(selected),
            "images": len(image_ids[name]),
            "no_target": sum(bool(ref.get("no_target", False)) for ref in selected),
            "positive": sum(not bool(ref.get("no_target", False)) for ref in selected),
            "original_split_counts": {
                split: sum(ref.get("split") == split for ref in selected)
                for split in ("train", "val", "testB")
            },
        }
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": "source_image_disjoint_refel_v1",
        "seed": args.seed,
        "source": str(refs_path),
        "source_sha256": hashlib.sha256(refs_path.read_bytes()).hexdigest(),
        "assignment": "sha256(seed:image_id), 70% train / 15% dev / 15% test",
        "summary": summary,
        "image_ids": image_ids,
        "indices": indices,
        "overlap_check": {key: len(value) for key, value in overlap.items()},
        "warning": "Existing checkpoints were trained under the original reference-level split and are not blind-test models for this protocol.",
    }
    (output / "split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output / "split.json"), "summary": summary,
                      "overlap_check": manifest["overlap_check"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
