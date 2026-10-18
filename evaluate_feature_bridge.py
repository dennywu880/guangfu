"""One-shot diagnostic comparison of the frozen base and selected e2 bridge."""

import argparse
import json
import os
import random
from pathlib import Path

import torch

from dataset import RefELDataset
from feature_bridge_experiment import BASE_CHECKPOINT, build_candidate
from train_detail_adapter import ROOT, evaluate, load_router
from train_text import FormalTextCache


def paired_interval(before, after, bucket=None, seed=20260924):
    old = {row["index"]: row for row in before["rows"]}
    new = {row["index"]: row for row in after["rows"]}
    if old.keys() != new.keys():
        raise ValueError("Reference IDs differ between paired evaluations")
    ids = sorted(i for i in old if bucket is None or old[i]["bucket"] == bucket)
    if not ids:
        return {"n": 0}
    rng = random.Random(seed)
    differences = []
    for _ in range(2000):
        draw = rng.choices(ids, k=len(ids))
        differences.append(sum(new[i]["iou"] - old[i]["iou"] for i in draw)
                           / len(draw))
    differences.sort()
    return {"n": len(ids),
            "mean_IoU_difference": sum(new[i]["iou"] - old[i]["iou"] for i in ids) / len(ids),
            "bootstrap_95pct": [differences[50], differences[1949]]}


@torch.no_grad()
def benchmark(model, device):
    model.eval()
    image = torch.randn(1, 3, 384, 384, device=device)
    pooled = torch.randn(1, 768, device=device, dtype=torch.float32)
    for _ in range(5):
        _, context = model.forward_general(image, pooled_text=pooled)
        model.forward_expert(context, "tiny")
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(30):
        _, context = model.forward_general(image, pooled_text=pooled)
        model.forward_expert(context, "tiny")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"image_plus_tiny_head_ms": (time.perf_counter() - start) * 1000 / 30,
            "note": "Synthetic tensor, cached pooled text; excludes disk I/O, text encoder and router."}


def summary(metrics):
    return {key: value for key, value in metrics.items() if key != "rows"}


def main(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = ROOT / "results" / args.experiment
    output = directory / "holdout_diagnostic.json"
    if output.exists():
        raise FileExistsError("Holdout diagnostic already exists; do not repeatedly tune against it")
    split = json.loads((directory / "split_audit.json").read_text(encoding="utf-8"))
    data = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    cache = FormalTextCache(ROOT)
    router = load_router()
    results = {}
    for label, use_bridge, checkpoint in (
        ("original_shared_checkpoint", False, None),
        ("bridge_only_epoch2", True, directory / "completed.pth"),
    ):
        model = build_candidate(device, use_bridge)
        if checkpoint is not None:
            model.load(checkpoint)
        metrics = evaluate(model, data, split["holdout_indices"], router,
                           device, cache, "val", args.batch_size)
        results[label] = {"summary": summary(metrics),
                          "rows": metrics["rows"],
                          "model_parameters_excluding_BERT": sum(
                              parameter.numel() for parameter in model.parameters()),
                          "benchmark": benchmark(model, device)}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    before = results["original_shared_checkpoint"]
    after = results["bridge_only_epoch2"]
    analysis = {bucket or "all": paired_interval(before, after, bucket)
                for bucket in (None, "ultra_tiny", "tiny", "larger", "no_target")}
    output.write_text(json.dumps({
        "protocol": "Same 160 image-distinct Val references, fixed rule router, no presence Gate",
        "base_checkpoint": str(BASE_CHECKPOINT),
        "candidate_checkpoint": str(directory / "completed.pth"),
        "warning": "Historical weights may have seen Val images; not a blind publication test.",
        "results": {name: {key: value for key, value in result.items() if key != "rows"}
                    for name, result in results.items()},
        "paired_difference": analysis,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "paired_difference": analysis,
                      "results": {name: result["summary"] for name, result in results.items()}},
                     indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="feature_bridge_only_20260924")
    parser.add_argument("--batch-size", type=int, default=2)
    main(parser.parse_args())
