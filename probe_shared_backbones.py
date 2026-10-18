"""Compare frozen backbone choices with identical heads and routed Val samples."""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch
from torch import nn

from dataset import RefELDataset
from shared_moe import BRANCH_FILES, SharedFeatures, SharedMoE


ROOT = Path(__file__).resolve().parent


class PooledTextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Identity()


def baseline_module():
    path = ROOT / "43_final_testB_gate_v2_and_m2_metrics.py"
    spec = importlib.util.spec_from_file_location("frozen_backbone_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(model, image, pooled_text):
    e0 = model.stem(image)
    e1 = model.layer1(model.maxpool(e0))
    e2 = model.layer2(e1)
    e3 = model.layer3(e2)
    e4 = model.layer4(e3)
    return SharedFeatures((e0, e1, e2, e3, e4), pooled_text, image.shape[-2:])


@torch.inference_mode()
def run(samples, seed):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = baseline_module()
    moe = SharedMoE.from_checkpoints(ROOT / "checkpoints", device)
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    ids = list(range(len(dataset)))
    random.Random(seed).shuffle(ids)
    ids = ids[:min(samples, len(dataset))]

    routes = {}
    for index in ids:
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(device)
        general, _ = moe.forward_general(image, texts=[item["text"]])
        ratio = float((general.sigmoid() >= 0.5).float().mean())
        routes[index] = baseline.route_sample(item["text"], ratio)[0]

    report = {}
    for name in ("general", *BRANCH_FILES):
        candidate = moe.general if name == "general" else baseline.load_seg_model(
            ROOT / "checkpoints" / BRANCH_FILES[name], device
        )
        if name != "general":
            candidate.text_encoder = PooledTextOnly()
            candidate.eval()
        counts = {branch: {"n": 0, "intersection": 0, "union": 0}
                  for branch in BRANCH_FILES}
        for index in ids:
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            gt = item["mask"].unsqueeze(0).to(device) > 0.5
            _, pooled_text, _ = moe.general.text_encoder([item["text"]], device)
            features = encode(candidate, image, pooled_text)
            branch = routes[index]
            pred = SharedMoE._decode(moe.heads[branch], features).sigmoid() >= 0.5
            count = counts[branch]
            count["n"] += 1
            count["intersection"] += int((pred & gt).sum().item())
            count["union"] += int((pred | gt).sum().item())
        total_intersection = sum(x["intersection"] for x in counts.values())
        total_union = sum(x["union"] for x in counts.values())
        result = {"cIoU": total_intersection / max(1, total_union),
                  "branches": {key: {"n": value["n"],
                                     "cIoU": value["intersection"] / max(1, value["union"])}
                               for key, value in counts.items()}}
        report[name] = result
        print(f"{name}: {result}", flush=True)
        if name != "general":
            del candidate
            if device.type == "cuda":
                torch.cuda.empty_cache()
    output = ROOT / "results" / "shared_backbone_probe.json"
    output.write_text(json.dumps({"split": "val", "samples": len(ids),
                                  "seed": seed, "presence_gate": "not applied",
                                  "backbones": report}, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    if options.samples < 1:
        parser.error("samples must be positive")
    run(options.samples, options.seed)
