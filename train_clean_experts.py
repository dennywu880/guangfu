"""Paired independent/shared expert baselines on the image-disjoint split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_clean_general import (
    ROOT, CleanRefDataset, load_split, load_vision_state,
    pooled_for_mixed_batch, restore_rng, rng_state, save, sha256, summarize,
    vision_state,
)
from detail_adapter_model import build_textless_unet
from multilevel_selection import GatedExpertFeature
from shared_moe import ExpertHead, SharedMoE
from train_detail_adapter import load_router, size_bucket
from train_text import FormalTextCache, calculate_loss


GENERAL = ROOT / "results" / "clean_general_source_disjoint" / "best.pth"
BRANCHES = ("ultra_tiny", "tiny", "larger")


def make_base(device):
    state = torch.load(GENERAL, map_location="cpu", weights_only=False)
    if state["format"] != "clean_general_v1" or state["metadata"]["split_sha256"] != sha256(
            ROOT / "results" / "source_image_split_20260925" / "split.json"):
        raise ValueError("General checkpoint is not from the fixed clean split")
    model = build_textless_unet(device=device)
    load_vision_state(model, state["model"])
    return model


def head_from_general(general):
    head = ExpertHead()
    state = general.state_dict()
    head.load_state_dict({key: state[key] for key in head.state_dict()}, strict=True)
    return head


class SharedExperts(SharedMoE):
    def __init__(self, general):
        heads = {branch: head_from_general(general) for branch in BRANCHES}
        super().__init__(general, heads)
        self.selections = nn.ModuleDict({
            branch: nn.ModuleDict({"e3": GatedExpertFeature(256),
                                   "e4": GatedExpertFeature(512)})
            for branch in BRANCHES
        })

    def forward_expert(self, context, branch):
        features = list(context.image)
        features[3] = self.selections[branch]["e3"](features[3])
        features[4] = self.selections[branch]["e4"](features[4])
        return self._decode(self.heads[branch], context, features)


def build_candidate(mode, general, device):
    if mode == "shared":
        candidate = SharedExperts(general).to(device)
        for parameter in candidate.general.parameters():
            parameter.requires_grad_(False)
        groups = [
            {"params": candidate.heads.parameters(), "lr": 1e-4},
            {"params": candidate.selections.parameters(), "lr": 5e-5},
        ]
    else:
        candidate = nn.ModuleDict({branch: make_base(device) for branch in BRANCHES})
        groups = []
        for model in candidate.values():
            encoder = list(model.stem.parameters()) + list(model.layer1.parameters())
            encoder += list(model.layer2.parameters()) + list(model.layer3.parameters())
            encoder += list(model.layer4.parameters())
            encoder_ids = {id(parameter) for parameter in encoder}
            head = [parameter for parameter in model.parameters()
                    if parameter.requires_grad and id(parameter) not in encoder_ids]
            groups.extend(({"params": encoder, "lr": 3e-5},
                           {"params": head, "lr": 1e-4}))
    return candidate, torch.optim.AdamW(groups, weight_decay=1e-4)


def parameter_counts(candidate, mode, general):
    general_count = sum(parameter.numel() for parameter in general.parameters())
    if mode == "shared":
        branch_counts = {
            branch: sum(parameter.numel() for parameter in candidate.heads[branch].parameters())
                    + sum(parameter.numel() for parameter in candidate.selections[branch].parameters())
            for branch in BRANCHES
        }
    else:
        branch_counts = {branch: sum(parameter.numel() for parameter in candidate[branch].parameters())
                         for branch in BRANCHES}
    return {"scope": "image path, text gates and decoder; excludes common frozen BERT",
            "candidate_total": general_count + sum(branch_counts.values()),
            "general": general_count, "branches": branch_counts,
            "active_by_route": {branch: general_count + count
                                for branch, count in branch_counts.items()}}


def predict(candidate, mode, images, pooled, branch, context=None):
    if mode == "shared":
        return candidate.forward_expert(context, branch)
    return candidate[branch](images, pooled_text=pooled)


def training_mode(candidate, mode):
    candidate.train()
    if mode == "shared":
        candidate.general.eval()
    else:
        for model in candidate.values():
            model.text_encoder.eval()
    for module in candidate.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def routed_batch(general, router, images, pooled, texts, mode, candidate):
    with torch.no_grad():
        if mode == "shared":
            logits, context = candidate.forward_general(images, pooled_text=pooled)
        else:
            # Match the shared candidate's General forward path exactly so a
            # prediction at an area threshold cannot change the routed expert.
            logits, _ = SharedMoE(general, {}).forward_general(images, pooled_text=pooled)
            context = None
        ratios = (logits.sigmoid() >= 0.5).float().flatten(1).mean(1).tolist()
    branches = [router.route_sample(text, float(ratio))[0]
                for text, ratio in zip(texts, ratios)]
    return branches, context


@torch.inference_mode()
def evaluate(candidate, mode, general, router, loader, cache, device, limit=None):
    candidate.eval()
    general.eval()
    rows = []
    route_counts = {branch: 0 for branch in BRANCHES}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True) > 0.5
        pooled = pooled_for_mixed_batch(cache, batch, device)
        branches, context = routed_batch(general, router, images, pooled,
                                         batch["text"], mode, candidate)
        for branch in BRANCHES:
            ids = [index for index, selected in enumerate(branches) if selected == branch]
            if not ids:
                continue
            route_counts[branch] += len(ids)
            selected_context = None
            if context is not None:
                selected_context = type(context)(
                    tuple(feature[ids] for feature in context.image),
                    context.pooled_text[ids], context.input_size)
            prediction = predict(candidate, mode, images[ids], pooled[ids], branch,
                                 selected_context).sigmoid() >= 0.5
            for local_index, pred in zip(ids, prediction):
                gt = masks[local_index]
                intersection = int((pred & gt).sum().item())
                union = int((pred | gt).sum().item())
                rows.append({"ref_id": int(batch["ref_id"][local_index]),
                             "image_id": int(batch["image_id"][local_index]),
                             "bucket": size_bucket(gt), "branch": branch,
                             "intersection": intersection, "union": union,
                             "iou": intersection / union if union else 1.0})
        if limit is not None and len(rows) >= limit:
            break
    result = summarize(rows)
    result["routes"] = route_counts
    result["rows"] = rows
    return result


def candidate_state(candidate, mode):
    if mode == "shared":
        return {"heads": {branch: {key: value.detach().cpu() for key, value in head.state_dict().items()}
                          for branch, head in candidate.heads.items()},
                "selections": {branch: {key: value.detach().cpu() for key, value in module.state_dict().items()}
                               for branch, module in candidate.selections.items()}}
    return {branch: vision_state(model) for branch, model in candidate.items()}


def restore_candidate(candidate, mode, state):
    if mode == "shared":
        for branch in BRANCHES:
            candidate.heads[branch].load_state_dict(state["heads"][branch], strict=True)
            candidate.selections[branch].load_state_dict(state["selections"][branch], strict=True)
    else:
        for branch in BRANCHES:
            load_vision_state(candidate[branch], state[branch])


def run(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for expert training")
    device = torch.device("cuda")
    split = load_split()
    output_name = args.output or f"clean_experts_{args.mode}_source_disjoint"
    if Path(output_name).name != output_name:
        raise ValueError("Output must be a single directory name")
    output = ROOT / "results" / output_name
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Experiment already exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = FormalTextCache(ROOT)
    train = CleanRefDataset("train", split["indices"]["train"])
    dev = CleanRefDataset("dev", split["indices"]["dev"])
    train_indices = list(range(len(train)))
    if args.smoke:
        train_indices = train_indices[:args.smoke_train_refs]
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    general = make_base(device).eval()
    for parameter in general.parameters():
        parameter.requires_grad_(False)
    candidate, optimizer = build_candidate(args.mode, general, device)
    router = load_router()
    scaler = torch.amp.GradScaler("cuda")
    metadata = {"format": "clean_experts_v1", "mode": args.mode,
                "general_sha256": sha256(GENERAL),
                "split_sha256": sha256(ROOT / "results" / "source_image_split_20260925" / "split.json"),
                "cache_sha256": cache.manifest["feature_file_sha256"],
                "train_references": len(train_indices), "dev_references": len(dev),
                "test_used": False, "presence_gate_used": False,
                "router": "fixed historical text/General-area rule",
                "training_loss": "TG-UNet BCE plus Dice",
                "batch_norm": "running statistics frozen for both candidates",
                "expert_initialization": "same clean general checkpoint for all three branches",
                "seed": args.seed, "batch_size": args.batch_size,
                "epochs_requested": args.epochs,
                "trainable": ("heads and e3/e4 selectors" if args.mode == "shared"
                              else "each expert image encoder and head")}
    metadata["parameter_counts"] = parameter_counts(candidate, args.mode, general)
    if args.resume:
        checkpoint = torch.load(output / "latest.pth", map_location=device,
                                weights_only=False)
        previous = checkpoint["metadata"]
        for key in ("format", "mode", "general_sha256", "split_sha256",
                    "seed", "batch_size", "train_references"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}")
        metadata = previous
        restore_candidate(candidate, args.mode, checkpoint["candidate"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        best, stale = checkpoint["best"], checkpoint["stale"]
        start_epoch = checkpoint["epoch"] + 1
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    else:
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        save(output / "initial.pth", {"format": "clean_experts_v1", "epoch": 0,
                                      "candidate": candidate_state(candidate, args.mode),
                                      "metadata": metadata})
        best, stale, start_epoch, history = float("-inf"), 0, 1, []
    for epoch in range(start_epoch, args.epochs + 1):
        order = list(train_indices)
        random.Random(args.seed + epoch).shuffle(order)
        train_loader = DataLoader(torch.utils.data.Subset(train, order),
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=True)
        random.seed(args.seed + epoch * 100003)
        training_mode(candidate, args.mode)
        total_loss = 0.0
        seen = 0
        routes = {branch: 0 for branch in BRANCHES}
        trace = hashlib.sha256()
        for step, batch in enumerate(train_loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            presence = batch["presence"].to(device, non_blocking=True)
            pooled = pooled_for_mixed_batch(cache, batch, device)
            branches, context = routed_batch(general, router, images, pooled,
                                             batch["text"], args.mode, candidate)
            for ref_id, sent_id, branch in zip(batch["ref_id"].tolist(),
                                               batch["sent_id"].tolist(), branches):
                trace.update(f"{ref_id}:{sent_id}:{branch}\n".encode("ascii"))
            optimizer.zero_grad(set_to_none=True)
            weighted_losses = []
            for branch in BRANCHES:
                ids = [index for index, selected in enumerate(branches) if selected == branch]
                if not ids:
                    continue
                routes[branch] += len(ids)
                selected_context = None
                if context is not None:
                    selected_context = type(context)(
                        tuple(feature[ids] for feature in context.image),
                        context.pooled_text[ids], context.input_size)
                with torch.autocast("cuda"):
                    logits = predict(candidate, args.mode, images[ids], pooled[ids],
                                     branch, selected_context)
                    loss, _, _ = calculate_loss(logits, masks[ids], presence[ids])
                weighted_losses.append(loss * len(ids))
            loss = sum(weighted_losses) / len(images)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]], 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
            seen += len(images)
            if step % args.log_every == 0:
                print(json.dumps({"mode": args.mode, "epoch": epoch, "step": step,
                                  "steps": len(train_loader), "loss": total_loss / seen,
                                  "routes": routes}), flush=True)
        metrics = evaluate(candidate, args.mode, general, router, dev_loader,
                           cache, device, args.smoke_dev_refs if args.smoke else None)
        rows = metrics.pop("rows")
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "train_routes": routes, "train_trace_sha256": trace.hexdigest(),
                        "dev": metrics})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if metrics["mean_iou"] > best:
            best, stale = metrics["mean_iou"], 0
            (output / "best_dev_rows.json").write_text(
                json.dumps(rows), encoding="utf-8")
            save(output / "best.pth", {"format": "clean_experts_v1", "epoch": epoch,
                                        "candidate": candidate_state(candidate, args.mode),
                                        "metadata": metadata, "dev": metrics})
        else:
            stale += 1
        save(output / "latest.pth", {"format": "clean_experts_v1", "epoch": epoch,
                                      "candidate": candidate_state(candidate, args.mode),
                                      "metadata": metadata, "optimizer": optimizer.state_dict(),
                                      "scaler": scaler.state_dict(), "rng": rng_state(),
                                      "best": best, "stale": stale})
        epoch_snapshot = output / f"epoch_{epoch:02d}.pth"
        if epoch_snapshot.exists():
            raise FileExistsError(f"Refusing to overwrite an epoch snapshot: {epoch_snapshot}")
        save(epoch_snapshot, {"format": "clean_experts_v1", "epoch": epoch,
                              "candidate": candidate_state(candidate, args.mode),
                              "metadata": metadata, "dev": metrics})
        print(json.dumps({"mode": args.mode, "epoch": epoch,
                          "dev_giou": metrics["mean_iou"],
                          "dev_ciou": metrics["ciou"],
                          "positive_mean_iou": metrics["positive_mean_iou"],
                          "by_bucket": metrics["by_bucket"]}), flush=True)
        if not args.smoke and stale >= args.patience:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shared", "independent"), required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-refs", type=int, default=16)
    parser.add_argument("--smoke-dev-refs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    options = parser.parse_args()
    if min(options.epochs, options.batch_size, options.patience,
           options.smoke_train_refs, options.smoke_dev_refs) < 1:
        parser.error("All counts must be positive")
    run(options)
