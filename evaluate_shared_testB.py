"""Read-only, frozen testB comparison of three shared-encoder checkpoints."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import RefELDataset
from multilevel_selection import build_model
from train_multilevel_head_joint import load as load_joint
from train_text import FormalTextCache


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CANDIDATES = {
    "multilevel_best": RESULTS / "multilevel_e3_e4_20260925" / "best.pth",
    "head_joint_best": RESULTS / "head_joint_e3_e4_20260925" / "best.pth",
    "distill_dev_ciou_best": RESULTS / "selective_distill_e3_e4_20260925" / "best.pth",
}


def original_protocol():
    path = ROOT / "43_final_testB_gate_v2_and_m2_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_testb_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_candidate(name, device):
    path = CANDIDATES[name]
    model = build_model(device, ("e3", "e4"))
    if name == "multilevel_best":
        model.load_checkpoint(path)
    else:
        load_joint(path, model)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_experts(protocol, model, dataset, records, cache, device):
    mapping = {
        "tp": "final_tp", "fp": "final_fp", "fn": "final_fn",
        "intersection": "final_intersection", "union": "final_union",
        "dice": "final_dice", "iou_regular": "final_iou",
        "pred_empty": "final_pred_empty",
    }
    for branch in protocol.GROUPS:
        indices = [row["dataset_index"] for row in records.values()
                   if row["route_branch"] == branch
                   and not row["gate_rejects_as_no_target"]]
        print(f"  {branch}: {len(indices)} surviving references", flush=True)
        loader = DataLoader(Subset(dataset, indices), batch_size=protocol.BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
        for batch_number, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            pooled = cache.get_batch(batch=batch, expected_split="testB", device=device)
            _, context = model.forward_general(images, pooled_text=pooled)
            pred = (torch.sigmoid(model.forward_expert(context, branch))
                    >= protocol.EXPERT_MASK_THRESHOLD).cpu()
            gt = (batch["mask"] > 0.5)
            for i, ref_id in enumerate(batch["ref_id"].tolist()):
                stats = protocol.binary_stats(pred[i], gt[i])
                records[int(ref_id)].update({dst: stats[src] for src, dst in mapping.items()})
            if batch_number % 250 == 0:
                print(f"    {branch}: {batch_number}/{len(loader)} batches", flush=True)


def save_samples(path, records):
    fields = ["ref_id", "is_positive", "gt_3way", "gt_area_ratio",
              "route_branch", "route_source", "p_no_target",
              "gate_rejects_as_no_target", "final_dice", "final_iou",
              "final_pred_empty", "final_tp", "final_fp", "final_fn",
              "final_intersection", "final_union"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(records.values(), key=lambda item: item["ref_id"]):
            writer.writerow({field: row[field] for field in fields})


def summarize(protocol, records):
    if len(records) != 7000 or any(row["final_iou"] is None for row in records.values()):
        raise RuntimeError("Incomplete testB predictions")
    positive = [row for row in records.values() if row["is_positive"]]
    by_size = protocol.summarize_by_true_size(records)
    for group in protocol.GROUPS:
        rows = [row for row in positive if row["gt_3way"] == group]
        by_size[group]["mean_iou"] = protocol.mean_or_zero([r["final_iou"] for r in rows])
    metrics = protocol.compute_m2segnet_style_metrics(records)
    return {
        "system": protocol.summarize_system(records),
        "gate": protocol.summarize_gate(records),
        "by_true_size": by_size,
        "by_route": protocol.summarize_by_route(records),
        "positive_mean_iou": protocol.mean_or_zero([r["final_iou"] for r in positive]),
        "all_mean_iou": metrics["gIoU"],
        "m2segnet_style_metrics": {k: v for k, v in metrics.items() if k != "pixel_counts"},
        "pixel_counts": metrics["pixel_counts"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="shared_testB_comparison_20260925")
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES,
                        default=list(CANDIDATES))
    args = parser.parse_args()
    protocol = original_protocol()
    for path in (protocol.PRESENCE_GATE_CHECKPOINT, *CANDIDATES.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = RESULTS / args.output
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; candidates: {args.candidates}", flush=True)
    cache = FormalTextCache(ROOT)
    dataset = RefELDataset(root_dir=ROOT, split="testB", image_size=protocol.IMAGE_SIZE,
                           random_text=False)
    if len(dataset) != 7000:
        raise RuntimeError(f"Expected 7000 testB references, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=protocol.BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
    gate, mean, std, hook, threshold = protocol.load_presence_gate(device)
    phase_model = load_candidate("multilevel_best", device)
    records, _, _ = protocol.phase1_general_gate_router(
        phase_model.base.general, gate, mean, std, hook, threshold,
        loader, device, cache)
    index = protocol.build_ref_to_index(dataset)
    for ref_id, row in records.items():
        row["dataset_index"] = index[ref_id]
    common = {
        "dataset": "Ref-EL-Defect testB", "reference_count": len(dataset),
        "image_size": protocol.IMAGE_SIZE, "mask_threshold": protocol.EXPERT_MASK_THRESHOLD,
        "presence_gate_threshold": threshold, "router_ultra_tiny": protocol.ROUTER_ULTRA_TINY,
        "router_tiny_larger": protocol.ROUTER_TINY_LARGER,
        "gate_checkpoint": str(protocol.PRESENCE_GATE_CHECKPOINT),
        "gate_checkpoint_sha256": digest(protocol.PRESENCE_GATE_CHECKPOINT),
        "protocol": "Original step-43 metrics, fixed gate and router; shared model's general encoder",
        "testB_used_for_training_or_threshold_tuning": False,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    }
    write_json(output / "protocol.json", common)
    del phase_model, gate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary = {}
    for name in args.candidates:
        print(f"Evaluating {name}", flush=True)
        model = load_candidate(name, device)
        current = {ref_id: row.copy() for ref_id, row in records.items()}
        evaluate_experts(protocol, model, dataset, current, cache, device)
        result = summarize(protocol, current)
        result["checkpoint"] = str(CANDIDATES[name])
        result["checkpoint_sha256"] = digest(CANDIDATES[name])
        result["shared_model_parameters_without_text_encoder"] = sum(
            parameter.numel() for parameter in model.parameters())
        write_json(output / f"{name}.json", result)
        save_samples(output / f"{name}_samples.csv", current)
        summary[name] = result
        print(f"  cIoU={result['m2segnet_style_metrics']['cIoU']:.6f} "
              f"gIoU={result['m2segnet_style_metrics']['gIoU']:.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(output / "comparison.json", {"protocol": common, "candidates": summary})
    print(f"Saved: {output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
