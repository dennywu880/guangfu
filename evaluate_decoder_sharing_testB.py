"""Frozen TestB evaluation for the 2026-10-08 decoder-sharing candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Subset

from dataset import RefELDataset
from train_decoder_sharing import SHARE_CANDIDATES, make_candidate
from train_text import FormalTextCache


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCE_DIR = RESULTS / "decoder_sharing_from_shared_encoder_20261008"
NAMES = ("private", "decoder4", "deep3", "full_decoder")
BERT_BASE_UNCASED_PARAMETERS = 109_482_240


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol():
    path = ROOT / "43_final_testB_gate_v2_and_m2_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_testb_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_candidate(name, device):
    path = SOURCE_DIR / f"{name}_epoch_02.pth"
    model = make_candidate(device, name)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "decoder_sharing_joint_v1":
        raise ValueError(f"Unsupported checkpoint: {path}")
    if state.get("candidate") != name or tuple(state.get("shared_names", ())) != SHARE_CANDIDATES[name]:
        raise ValueError(f"Candidate metadata mismatch: {path}")
    if set(state["heads"]) != set(model.base.heads):
        raise ValueError(f"Incomplete expert heads: {path}")
    for branch, head in model.base.heads.items():
        head.load_state_dict(state["heads"][branch], strict=True)
    for branch, levels in model.selections.items():
        for level, selector in levels.items():
            selector.load_state_dict(state["selections"][branch][level], strict=True)
    if set(state["shared_modules"]) != set(model.shared_modules):
        raise ValueError(f"Shared module mismatch: {path}")
    for key, module in model.shared_modules.items():
        module.load_state_dict(state["shared_modules"][key], strict=True)
    model.eval()
    model.base.eval()
    return model, path


def add_resources(protocol, result, candidate, checkpoint):
    counts = candidate.parameter_counts()
    result["resources"] = {
        "image_model_total_parameters_excluding_external_text_encoder": counts[
            "total_image_model_parameters_cached_text_excluded"
        ],
        "text_encoder_parameters_counted_once": BERT_BASE_UNCASED_PARAMETERS,
        "combined_image_plus_text_encoder_parameters": counts[
            "total_image_model_parameters_cached_text_excluded"
        ] + BERT_BASE_UNCASED_PARAMETERS,
        "active_one_route_image_parameters_excluding_external_text_encoder": counts[
            "active_one_route_image_model_parameters_cached_text_excluded"
        ],
        "active_one_route_plus_text_encoder_parameters": counts[
            "active_one_route_image_model_parameters_cached_text_excluded"
        ] + BERT_BASE_UNCASED_PARAMETERS,
        "external_text_encoder_counted_once": True,
        "checkpoint_file_bytes_incremental_not_self_contained": checkpoint.stat().st_size,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "peak_cuda_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
    }


@torch.inference_mode()
def profile_forward_flops(candidate, dataset, records, cache, device):
    row = next(row for row in records.values()
               if not row["gate_rejects_as_no_target"])
    item = dataset[row["dataset_index"]]
    images = item["image"].unsqueeze(0).to(device)
    batch = {"ref_id": torch.tensor([item["ref_id"]]),
             "sent_id": torch.tensor([item["sent_id"]]),
             "text": [item["text"]]}
    pooled = cache.get_batch(batch=batch, expected_split="testB", device=device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with profile(activities=activities, with_flops=True, profile_memory=True) as prof:
        _, context = candidate.forward_general(images, pooled_text=pooled)
        candidate.forward_expert(context, row["route_branch"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    flops = sum(event.flops for event in prof.key_averages())
    return {
        "general_plus_one_routed_expert_flops_per_image": int(flops),
        "measurement": "PyTorch profiler supported-operator FLOP estimate for one 384x384 image; excludes BERT transformer, Presence Gate, GUI and unsupported operators",
    }


@torch.inference_mode()
def evaluate_candidate(protocol, candidate, dataset, records, cache, device, batch_size):
    fields = {
        "tp": "final_tp", "fp": "final_fp", "fn": "final_fn",
        "intersection": "final_intersection", "union": "final_union",
        "dice": "final_dice", "iou_regular": "final_iou",
        "pred_empty": "final_pred_empty",
    }
    expert_time = 0.0
    expert_samples = 0
    for branch in protocol.GROUPS:
        selected = [
            row["dataset_index"] for row in records.values()
            if row["route_branch"] == branch and not row["gate_rejects_as_no_target"]
        ]
        loader = DataLoader(
            Subset(dataset, selected), batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
        )
        print(f"  {branch}: {len(selected)} surviving references", flush=True)
        for batch in loader:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            images = batch["image"].to(device, non_blocking=True)
            pooled = cache.get_batch(batch=batch, expected_split="testB", device=device)
            _, context = candidate.forward_general(images, pooled_text=pooled)
            pred = (candidate.forward_expert(context, branch).sigmoid()
                    >= protocol.EXPERT_MASK_THRESHOLD).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            expert_time += time.perf_counter() - started
            gt_batch = batch["mask"] > 0.5
            for i, ref_id in enumerate(batch["ref_id"].tolist()):
                stats = protocol.binary_stats(pred[i], gt_batch[i])
                records[int(ref_id)].update({dst: stats[src] for src, dst in fields.items()})
            expert_samples += len(batch["ref_id"])
    if any(row["final_iou"] is None for row in records.values()):
        raise RuntimeError("Some TestB rows lack final predictions")
    metrics = protocol.compute_m2segnet_style_metrics(records)
    return metrics, protocol.summarize_system(records), protocol.summarize_gate(records), \
        protocol.summarize_by_true_size(records), protocol.summarize_by_route(records), \
        expert_time, expert_samples


def write_samples(path, records):
    fields = ["ref_id", "is_positive", "gt_3way", "gt_area_ratio", "route_branch",
              "route_source", "p_no_target", "gate_rejects_as_no_target", "final_dice",
              "final_iou", "final_pred_empty", "final_tp", "final_fp", "final_fn",
              "final_intersection", "final_union"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(records.values(), key=lambda item: item["ref_id"]):
            writer.writerow({field: row[field] for field in fields})


def size_metrics(protocol, records, existing):
    for group, summary in existing.items():
        rows = [row for row in records.values()
                if row["is_positive"] and row["gt_3way"] == group]
        tp = sum(int(row["final_tp"]) for row in rows)
        fp = sum(int(row["final_fp"]) for row in rows)
        fn = sum(int(row["final_fn"]) for row in rows)
        summary["positive_mean_iou"] = protocol.mean_or_zero(
            [row["final_iou"] for row in rows]
        )
        summary["positive_cIoU"] = sum(
            int(row["final_intersection"]) for row in rows
        ) / max(1, sum(int(row["final_union"]) for row in rows))
        summary["Recall"] = tp / max(1, tp + fn)
        summary["Precision"] = tp / max(1, tp + fp)
        p, r = summary["Precision"], summary["Recall"]
        summary["F1"] = 2 * p * r / max(1e-12, p + r)
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="decoder_sharing_testB_frozen_20261009")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    output = RESULTS / args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_protocol()
    gate, mean, std, hook, gate_threshold = protocol.load_presence_gate(device)
    cache = FormalTextCache(ROOT)
    dataset = RefELDataset(ROOT, "testB", protocol.IMAGE_SIZE, False)
    if len(dataset) != 7000:
        raise RuntimeError(f"Expected 7000 TestB references; found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    reference = load_candidate("private", device)[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    records, _, _ = protocol.phase1_general_gate_router(
        reference.base.general, gate, mean, std, hook, gate_threshold,
        loader, device, cache, expected_split="testB",
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    common_phase_seconds = time.perf_counter() - started
    ref_to_index = protocol.build_ref_to_index(dataset)
    for ref_id, row in records.items():
        row["dataset_index"] = ref_to_index[ref_id]
    if len(records) != 7000:
        raise RuntimeError(f"Incomplete TestB routing: {len(records)} rows")
    gate_hash = digest(protocol.PRESENCE_GATE_CHECKPOINT)
    protocol_info = {
        "dataset": "Ref-EL-Defect TestB",
        "references": len(records),
        "input_size": protocol.IMAGE_SIZE,
        "general_mask_threshold": protocol.GENERAL_MASK_THRESHOLD,
        "expert_mask_threshold": protocol.EXPERT_MASK_THRESHOLD,
        "presence_gate_threshold": gate_threshold,
        "router_ultra_tiny": protocol.ROUTER_ULTRA_TINY,
        "router_tiny_larger": protocol.ROUTER_TINY_LARGER,
        "gate_checkpoint": str(protocol.PRESENCE_GATE_CHECKPOINT),
        "gate_checkpoint_sha256": gate_hash,
        "protocol": "Frozen step-43 General + Presence Gate v2 + text/area routing; decoder candidate only varies expert path",
        "historical_testB_exposure": "TestB has prior project use; this is a frozen re-evaluation, not an untouched blind test",
        "threshold_or_model_tuning_on_testB": False,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "shared_general_gate_router_pass_seconds": common_phase_seconds,
    }
    (output / "protocol.json").write_text(json.dumps(protocol_info, indent=2), encoding="utf-8")
    summary = {"protocol": protocol_info, "candidates": {}}
    del reference, gate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for name in NAMES:
        print(f"Evaluating TestB candidate: {name}", flush=True)
        model, checkpoint = load_candidate(name, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        current = {ref_id: row.copy() for ref_id, row in records.items()}
        result, system, gate_metrics, by_size, by_route, expert_seconds, expert_samples = \
            evaluate_candidate(protocol, model, dataset, current, cache, device, args.batch_size)
        by_size = size_metrics(protocol, current, by_size)
        item = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "m2segnet_style_metrics": {key: value for key, value in result.items()
                                       if key != "pixel_counts"},
            "pixel_counts": result["pixel_counts"],
            "system": system,
            "presence_gate": gate_metrics,
            "by_true_size": by_size,
            "by_route": by_route,
            "compute": profile_forward_flops(model, dataset, current, cache, device),
            "runtime": {
                "shared_general_gate_router_pass_seconds": common_phase_seconds,
                "expert_path_pass_seconds": expert_seconds,
                "combined_evaluation_pipeline_seconds": common_phase_seconds + expert_seconds,
                "combined_seconds_per_reference": (common_phase_seconds + expert_seconds) / len(records),
                "references_reaching_expert_path": expert_samples,
                "timing_scope": "Dataset loading, cached text-feature lookup, General/Gate/Router pass, routed expert pass; excludes GUI rendering and BERT transformer execution because formal cached text features were used",
            },
        }
        add_resources(protocol, item, model, checkpoint)
        (output / f"{name}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        write_samples(output / f"{name}_samples.csv", current)
        summary["candidates"][name] = item
        print(f"  gIoU={result['gIoU']:.6f} cIoU={result['cIoU']:.6f} "
              f"Recall={result['Recall']:.6f} Precision={result['Precision']:.6f} "
              f"F1={result['F1']:.6f} mPr={result['mPr']:.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved TestB report: {output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
