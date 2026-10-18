"""Render existing paired failure cases without repeating routing or training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_current_routing import HEAD_JOINT
from dataset import RefELDataset
from evaluate_shared_testB import digest, load_candidate
from train_detail_adapter import ROOT, decode_teacher, pooled_for, teacher_for
from train_text import FormalTextCache


SOURCE = ROOT / "results" / "head_joint_routing_audit_20260926" / "diagnosis.json"
OUTPUT = ROOT / "results" / "head_joint_failure_review_20260927"


def iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union else 1.0


def panel(image, title, pred=None, gt=None):
    visual = image.copy()
    if gt is not None and pred is None:
        visual[gt] = (visual[gt] * 0.35 + np.array([0, 220, 90]) * 0.65).astype(np.uint8)
    elif pred is not None:
        tp = pred & gt
        fp = pred & ~gt
        fn = gt & ~pred
        for mask, color in ((tp, (0, 210, 90)), (fp, (245, 55, 45)),
                            (fn, (255, 220, 0))):
            visual[mask] = (visual[mask] * 0.35 + np.array(color) * 0.65).astype(np.uint8)
    band = np.full((38, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(band, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                (35, 35, 35), 1, cv2.LINE_AA)
    return np.concatenate((band, visual), axis=0)


def save_png(path, rgb):
    ok, data = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    data.tofile(str(path))


@torch.inference_mode()
def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    audit = json.loads(SOURCE.read_text(encoding="utf-8"))
    if audit["candidate"] != "head_joint" or audit["source_sha256"] != digest(HEAD_JOINT):
        raise RuntimeError("Paired audit does not match the current head-joint checkpoint")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    selected = []
    for row in audit["rows"]:
        if row["gt_empty"] or row["shared_iou"] >= 0.5:
            continue
        group = "independent_rescues" if row["teacher_iou"] >= 0.5 else "both_below_05"
        selected.append((group, row))
    if len(selected) != 51:
        raise RuntimeError(f"Expected 51 pre-audited failure cases, found {len(selected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    shared = load_candidate("head_joint_best", device).eval()
    teachers = {branch: teacher_for(branch, device) for branch in ("ultra_tiny", "tiny", "larger")}
    output_rows = []
    gallery = {"independent_rescues": [], "both_below_05": []}
    for position, (group, row) in enumerate(selected, 1):
        item = dataset[row["val_index"]]
        if int(item["ref_id"]) != row["ref_id"] or int(item["image_id"]) != row["image_id"]:
            raise RuntimeError("Dataset reference does not match saved paired audit")
        image = item["image"].unsqueeze(0).to(device)
        pooled = pooled_for(cache, [item], "val", device)
        _, context = shared.forward_general(image, pooled_text=pooled)
        shared_pred = (shared.forward_expert(context, row["route"]).sigmoid() >= 0.5)[0, 0].cpu().numpy()
        teacher_logits, _ = decode_teacher(teachers[row["route"]], image, pooled)
        teacher_pred = (teacher_logits.sigmoid() >= 0.5)[0, 0].cpu().numpy()
        gt = item["mask"].squeeze().cpu().numpy() > 0.5
        actual_shared, actual_teacher = iou(shared_pred, gt), iou(teacher_pred, gt)
        if abs(actual_shared - row["shared_iou"]) > 1e-6 or abs(actual_teacher - row["teacher_iou"]) > 1e-6:
            raise RuntimeError(f"Prediction mismatch on ref_id={row['ref_id']}")
        file_name = dataset.images_by_id[row["image_id"]]["file_name"]
        source = cv2.imread(str(dataset.image_dir / file_name), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(file_name)
        source = cv2.cvtColor(cv2.resize(source, (384, 384)), cv2.COLOR_BGR2RGB)
        figure = np.concatenate([
            panel(source, f"Ref {row['ref_id']}  {row['bucket']}"),
            panel(source, f"Ground truth {int(gt.sum())} px", gt=gt),
            panel(source, f"Independent IoU {actual_teacher:.3f}", teacher_pred, gt),
            panel(source, f"Shared IoU {actual_shared:.3f}", shared_pred, gt),
        ], axis=1)
        path = OUTPUT / group / f"ref_{row['ref_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_png(path, figure)
        reference = dataset.refs[row["val_index"]]
        ann_ids = reference.get("ann_id", [])
        if not isinstance(ann_ids, list):
            ann_ids = [ann_ids]
        output_rows.append({
            "group": group, "ref_id": row["ref_id"], "image_id": row["image_id"],
            "val_index": row["val_index"], "file_name": file_name,
            "text": item["text"], "branch": row["route"], "gt_size": row["bucket"],
            "annotation_ids": ann_ids, "shared_iou": actual_shared,
            "independent_iou": actual_teacher, "figure": str(path),
        })
        gallery[group].append(figure)
        if position % 10 == 0 or position == len(selected):
            print(f"Rendered {position}/{len(selected)}", flush=True)
    for group, figures in gallery.items():
        for start in range(0, len(figures), 4):
            sheet = np.concatenate(figures[start:start + 4], axis=0)
            save_png(OUTPUT / group / f"sheet_{start // 4 + 1:02d}.png", sheet)
    (OUTPUT / "cases.json").write_text(json.dumps({
        "source_audit": str(SOURCE), "checkpoint_sha256": digest(HEAD_JOINT),
        "criteria": "positive, shared IoU < 0.5; independent same-route IoU >= 0.5 or < 0.5",
        "gate": "not applied in the source 160-reference audit",
        "legend": "green=TP, red=FP, yellow=FN; GT panel is green",
        "cases": output_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
