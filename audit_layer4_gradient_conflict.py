"""Measure whether routed expert batches disagree on shared layer4 updates."""

from __future__ import annotations

import argparse
import json
import random

import torch
from torch.nn import functional as F

from dataset import RefELDataset
from multilevel_selection import build_model
from train_detail_adapter import (
    ROOT, load_router, pooled_for, route_indices, segmentation_loss, stack_items,
)
from train_multilevel_head_joint import load as load_source
from train_shared_layer4_joint import SOURCE, expert_context, load as load_layer4
from train_text import FormalTextCache


def run(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, ("e3", "e4"))
    load_source(SOURCE, model)
    if args.checkpoint:
        load_layer4(ROOT / "results" / args.checkpoint, model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.base.general.layer4.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    train = RefELDataset(ROOT, "train", 384, False)
    cache = FormalTextCache(ROOT)
    indices = list(range(len(train)))
    random.Random(args.seed).shuffle(indices)
    groups = route_indices(model, train, indices[:args.pool], load_router(),
                           device, cache, "train", 4)
    if any(len(ids) < args.steps for ids in groups.values()):
        raise RuntimeError("Not enough samples in one routed expert group")
    names = list(groups)
    cosines = {f"{a}:{b}": [] for i, a in enumerate(names) for b in names[i + 1:]}
    for step in range(args.steps):
        gradients = {}
        for name in names:
            item, images, masks, _ = stack_items(train, [groups[name][step]], device)
            pooled = pooled_for(cache, item, "train", device)
            logits = model.forward_expert(expert_context(model, images, pooled), name)
            loss = segmentation_loss(logits, masks)
            grad = torch.autograd.grad(loss, parameters)
            gradients[name] = torch.cat([part.detach().flatten() for part in grad])
        for key in cosines:
            first, second = key.split(":")
            cosines[key].append(float(F.cosine_similarity(
                gradients[first], gradients[second], dim=0).item()))
    result = {
        "checkpoint": args.checkpoint or str(SOURCE),
        "steps": args.steps,
        "by_pair": {
            key: {
                "mean_cosine": sum(values) / len(values),
                "negative_fraction": sum(value < 0 for value in values) / len(values),
            }
            for key, values in cosines.items()
        },
        "note": "Different samples per expert; negative gradient cosine is suggestive, not proof of harmful task interference.",
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--pool", type=int, default=240)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260926)
    run(parser.parse_args())
