"""Save paired prediction examples for visual failure analysis."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import RefELDataset
from inference_engine import BASELINE
from shared_moe import BRANCH_FILES, SharedMoE


ROOT = Path(__file__).resolve().parent
CASE_REPORT = ROOT / "results" / "shared_failure_analysis.json"


def overlay(image, mask, color):
    rgb = image.copy()
    shade = np.zeros_like(rgb)
    shade[:, :] = color
    fg = mask.astype(bool)
    rgb[fg] = (0.4 * rgb[fg] + 0.6 * shade[fg]).astype(np.uint8)
    return rgb


def pane(image, label, accent):
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_NEAREST)
    band = np.full((42, 384, 3), 255, dtype=np.uint8)
    cv2.putText(band, label, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                accent, 2, cv2.LINE_AA)
    return np.concatenate((band, image), axis=0)


def build(case, dataset, model, teacher, device):
    sample = dataset[case["dataset_index"]]
    image = sample["image"].unsqueeze(0).to(device)
    _, context = model.forward_general(image, texts=[sample["text"]])
    with torch.inference_mode():
        old = (teacher(image, pooled_text=context.pooled_text).sigmoid() >= 0.5)[0, 0].cpu().numpy()
        new = (model.forward_expert(context, case["branch"]).sigmoid() >= 0.5)[0, 0].cpu().numpy()
    gt = sample["mask"].squeeze(0).cpu().numpy() > 0.5
    image_path = dataset.image_dir / sample["file_name"]
    source = cv2.imread(str(image_path))
    source = cv2.cvtColor(cv2.resize(source, (384, 384)), cv2.COLOR_BGR2RGB)
    panels = [
        pane(source, f"#{case['dataset_index']} {case['branch']}", (40, 40, 40)),
        pane(overlay(source, gt, (0, 200, 80)), "Annotation", (10, 110, 40)),
        pane(overlay(source, old, (30, 100, 255)), f"Original IoU {case['old_iou']:.3f}", (20, 80, 180)),
        pane(overlay(source, new, (255, 60, 50)), f"Shared IoU {case['shared_iou']:.3f}", (180, 45, 40)),
    ]
    return np.concatenate(panels, axis=1)


def run():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    report = json.loads(CASE_REPORT.read_text(encoding="utf-8"))
    selected = report["largest_regressions"][:4] + report["largest_improvements"][:4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    model = SharedMoE.from_checkpoints(ROOT / "checkpoints", device, expert_backbone="larger")
    gallery = []
    teacher = None
    current_branch = None
    for case in selected:
        branch = case["branch"]
        if branch != current_branch:
            del teacher
            teacher = BASELINE.load_seg_model(ROOT / "checkpoints" / BRANCH_FILES[branch], device)
            current_branch = branch
        with torch.inference_mode():
            gallery.append(build(case, dataset, model, teacher, device))
    combined = np.concatenate(gallery, axis=0)
    output = ROOT / "results" / "shared_failure_gallery.png"
    cv2.imencode(".png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))[1].tofile(str(output))
    print(output)


if __name__ == "__main__":
    run()
