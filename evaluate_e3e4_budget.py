"""Paired development audit of budget checkpoints with the fixed Gate and router."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch
from torch.utils.data import DataLoader, Subset

from audit_expanded_routing import put_stats, summary
from dataset import RefELDataset
from evaluate_shared_testB import original_protocol
from multilevel_selection import build_model
from train_detail_adapter import ROOT, decode_teacher, teacher_for
from train_e3e4_budget import DEV_INDICES, SOURCE, restore
from train_multilevel_head_joint import load as load_head_joint
from train_text import FormalTextCache


def evaluate_shared(protocol, model, dataset, base, cache, device):
    records = {ref_id: row.copy() for ref_id, row in base.items()}
    for branch in protocol.GROUPS:
        indices = [row["dataset_index"] for row in records.values()
                   if row["route_branch"] == branch
                   and not row["gate_rejects_as_no_target"]]
        loader = DataLoader(Subset(dataset, indices), batch_size=protocol.BATCH_SIZE,
                            shuffle=False, num_workers=0)
        for batch in loader:
            images = batch["image"].to(device)
            pooled = cache.get_batch(batch, expected_split="val", device=device)
            _, context = model.forward_general(images, pooled_text=pooled)
            predictions = (model.forward_expert(context, branch).sigmoid()
                           >= protocol.EXPERT_MASK_THRESHOLD).cpu()
            truth = batch["mask"] > 0.5
            for i, ref_id in enumerate(batch["ref_id"].tolist()):
                put_stats(records[int(ref_id)], protocol.binary_stats(predictions[i], truth[i]))
    return records


def evaluate_independent(protocol, dataset, base, cache, device):
    records = {ref_id: row.copy() for ref_id, row in base.items()}
    for branch in protocol.GROUPS:
        teacher = teacher_for(branch, device)
        indices = [row["dataset_index"] for row in records.values()
                   if row["route_branch"] == branch
                   and not row["gate_rejects_as_no_target"]]
        loader = DataLoader(Subset(dataset, indices), batch_size=protocol.BATCH_SIZE,
                            shuffle=False, num_workers=0)
        for batch in loader:
            images = batch["image"].to(device)
            pooled = cache.get_batch(batch, expected_split="val", device=device)
            logits, _ = decode_teacher(teacher, images, pooled)
            predictions = (logits.sigmoid() >= protocol.EXPERT_MASK_THRESHOLD).cpu()
            truth = batch["mask"] > 0.5
            for i, ref_id in enumerate(batch["ref_id"].tolist()):
                put_stats(records[int(ref_id)], protocol.binary_stats(predictions[i], truth[i]))
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for an explicit CPU run")
    protocol = original_protocol()
    output = ROOT / "results" / args.output
    if not output.is_dir():
        raise FileNotFoundError(output)
    indices = json.loads(DEV_INDICES.read_text(encoding="utf-8"))
    dataset = RefELDataset(ROOT, "val", protocol.IMAGE_SIZE, False)
    cache = FormalTextCache(ROOT)
    model = build_model(device, ("e3", "e4"))
    load_head_joint(SOURCE, model)
    model.eval()
    gate, mean, std, hook, threshold = protocol.load_presence_gate(device)
    loader = DataLoader(Subset(dataset, indices), batch_size=protocol.BATCH_SIZE,
                        shuffle=False, num_workers=0)
    base, _, _ = protocol.phase1_general_gate_router(
        model.base.general, gate, mean, std, hook, threshold,
        loader, device, cache, expected_split="val",
    )
    for index in indices:
        ref_id = int(dataset.refs[index]["ref_id"])
        base[ref_id]["dataset_index"] = index
    if len(base) != len(indices):
        raise RuntimeError("Gate/route audit did not cover every reference")
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    results = {}
    rows = {}
    for stage in (0, 3000, 6000, 12000):
        path = output / f"stage_{stage:05d}.pth"
        if not path.is_file():
            continue
        restore(path, model, None, source_sha256, device)
        model.eval()
        records = evaluate_shared(protocol, model, dataset, base, cache, device)
        name = f"shared_{stage:05d}"
        results[name] = summary(protocol, records)
        results[name]["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows[name] = {str(ref_id): {"iou": row["final_iou"],
                                    "intersection": row["final_intersection"],
                                    "union": row["final_union"]}
                      for ref_id, row in records.items()}
        print({"candidate": name, "gIoU": results[name]["gIoU"],
               "cIoU": results[name]["cIoU"]}, flush=True)
    independent = evaluate_independent(protocol, dataset, base, cache, device)
    results["independent"] = summary(protocol, independent)
    rows["independent"] = {str(ref_id): {"iou": row["final_iou"],
                                     "intersection": row["final_intersection"],
                                     "union": row["final_union"]}
                           for ref_id, row in independent.items()}
    print({"candidate": "independent", "gIoU": results["independent"]["gIoU"],
           "cIoU": results["independent"]["cIoU"]}, flush=True)
    report = {
        "protocol": {
            "split": "historical val development", "reference_count": len(indices),
            "indices_sha256": hashlib.sha256(DEV_INDICES.read_bytes()).hexdigest(),
            "source_sha256": source_sha256,
            "fixed_gate_router_and_threshold": True,
            "gate_threshold": threshold,
            "historical_image_exposure": "possible; not a blind generalization test",
            "testB_used": False,
        },
        "candidates": results,
    }
    (output / "full_system_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "full_system_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="e3e4_budget_20260928")
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
