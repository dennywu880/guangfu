"""One-epoch PCGrad ablation for the shared layer4, with reversible checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime

import torch

from dataset import RefELDataset
from multilevel_selection import build_model
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, evaluate, load_or_create_split,
    load_router, pooled_for, route_indices, segmentation_loss, stack_items,
    size_loss_weights,
)
from train_multilevel_head_joint import compact, load as load_source
from train_shared_layer4_joint import DEVELOPMENT_INDICES, SOURCE, expert_context, save
from train_text import FormalTextCache


def project_conflicts(gradients, rng):
    projected = []
    negative_pairs = 0
    for index, original in enumerate(gradients):
        current = original.clone()
        others = [i for i in range(len(gradients)) if i != index]
        rng.shuffle(others)
        for other in others:
            reference = gradients[other]
            dot = torch.dot(current, reference)
            if dot < 0:
                current = current - dot / reference.square().sum().clamp_min(1e-12) * reference
                negative_pairs += 1
        projected.append(current)
    return torch.stack(projected).sum(dim=0), negative_pairs


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This ablation requires CUDA")
    output = ROOT / "results" / args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train = RefELDataset(ROOT, "train", 384, False)
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    dev, holdout = load_or_create_split(val, args.seed)
    pool, excluded, split_audit = build_disjoint_train_pool(train, val, dev, holdout)
    expanded = json.loads(DEVELOPMENT_INDICES.read_text(encoding="utf-8"))
    excluded |= {val.refs[index]["image_id"] for index in expanded}
    pool = [index for index in pool if train.refs[index]["image_id"] not in excluded]
    model = build_model(device, ("e3", "e4"))
    load_source(SOURCE, model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.base.general.layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in model.selections.parameters():
        parameter.requires_grad_(True)
    for parameter in model.base.heads.parameters():
        parameter.requires_grad_(True)
    layer4 = list(model.base.general.layer4.parameters())
    adapters = list(model.selections.parameters())
    heads = list(model.base.heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": layer4, "lr": args.layer4_lr},
        {"params": adapters, "lr": args.adapter_lr},
        {"params": heads, "lr": args.head_lr},
    ], weight_decay=1e-4)
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "architecture": "one shared encoder, expert e3/e4 adapters and heads",
        "method": "PCGrad on shared layer4 only; branch head gradients unchanged",
        "train_samples": args.train_samples, "batch_size": args.batch_size,
        "seed": args.seed, "layer4_lr": args.layer4_lr,
        "adapter_lr": args.adapter_lr, "head_lr": args.head_lr,
        "excluded_images": len(excluded), "eligible_train_refs": len(pool),
        "split_audit": split_audit, "testB_used": False,
        "web_untouched": True, "historical_image_exposure_not_excluded": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    router = load_router()
    initial = compact(evaluate(model, val, dev, router, device, cache, "val", args.batch_size))
    save(output / "initial.pth", model, metadata)
    indices = list(pool)
    random.Random(args.seed + 1).shuffle(indices)
    indices = indices[:min(args.train_samples, len(indices))]
    groups = route_indices(model, train, indices, router, device, cache,
                           "train", args.batch_size)
    batches = {
        name: [ids[start:start + args.batch_size]
               for start in range(0, len(ids), args.batch_size)]
        for name, ids in groups.items()
    }
    n_triplets = min(len(value) for value in batches.values())
    model.train()
    model.base.eval()
    model.selections.train()
    rng = random.Random(args.seed)
    losses = []
    conflicts = 0

    def backward_branch(name, ids):
        items, images, masks, _ = stack_items(train, ids, device)
        pooled = pooled_for(cache, items, "train", device)
        logits = model.forward_expert(expert_context(model, images, pooled), name)
        weights = size_loss_weights(masks, 1.25, 1.75)
        loss = segmentation_loss(logits, masks, weights)
        loss.backward()
        losses.append(float(loss.detach()))

    names = list(batches)
    for step in range(n_triplets):
        optimizer.zero_grad(set_to_none=True)
        gradients = []
        for name in names:
            backward_branch(name, batches[name][step])
            gradients.append(torch.cat([p.grad.detach().flatten() for p in layer4]))
            for parameter in layer4:
                parameter.grad = None
        shared, count = project_conflicts(gradients, rng)
        conflicts += count
        offset = 0
        for parameter in layer4:
            size = parameter.numel()
            parameter.grad = shared[offset:offset + size].view_as(parameter)
            offset += size
        torch.nn.utils.clip_grad_norm_(layer4 + adapters + heads, 5)
        optimizer.step()
        if (step + 1) % 50 == 0:
            print({"triplets_done": step + 1, "total": n_triplets}, flush=True)
    for name, branch_batches in batches.items():
        for ids in branch_batches[n_triplets:]:
            optimizer.zero_grad(set_to_none=True)
            backward_branch(name, ids)
            torch.nn.utils.clip_grad_norm_(layer4 + adapters + heads, 5)
            optimizer.step()
        print({"remaining_branch": name, "steps": len(branch_batches) - n_triplets}, flush=True)
    metrics = compact(evaluate(model, val, dev, router, device, cache, "val", args.batch_size))
    save(output / "epoch_01.pth", model, metadata)
    result = {
        "initial": initial, "epoch_01": metrics,
        "mean_train_loss": sum(losses) / len(losses),
        "triplet_steps": n_triplets,
        "negative_projections": conflicts,
        "remaining_steps": sum(len(value) - n_triplets for value in batches.values()),
        "route_counts": {name: len(ids) for name, ids in groups.items()},
    }
    (output / "history.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layer4-lr", type=float, default=5e-6)
    parser.add_argument("--adapter-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260924)
    run(parser.parse_args())
