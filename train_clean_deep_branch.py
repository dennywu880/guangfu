"""Train clean e0-e2 shared / e3-e4 expert branches on the image split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from train_clean_experts import (
    BRANCHES, GENERAL, ROOT, CleanRefDataset, SharedExperts, evaluate,
    load_split, make_base, pooled_for_mixed_batch, restore_rng, rng_state,
    routed_batch, save, sha256, summarize,
)
from train_detail_adapter import load_router
from train_text import FormalTextCache, calculate_loss


SPLIT = ROOT / "results" / "source_image_split_20260925" / "split.json"


class CleanDeepExperts(SharedExperts):
    """Keep e0-e2 shared; copy and train e3/e4 separately per expert."""

    def __init__(self, general):
        super().__init__(general)
        self.deep_blocks = nn.ModuleDict({
            branch: nn.ModuleDict({
                "e3": copy.deepcopy(general.layer3),
                "e4": copy.deepcopy(general.layer4),
            })
            for branch in BRANCHES
        })

    def forward_expert(self, context, branch):
        e0, e1, e2 = context.image[:3]
        e3 = self.deep_blocks[branch]["e3"](e2)
        e4 = self.deep_blocks[branch]["e4"](e3)
        e3 = self.selections[branch]["e3"](e3)
        e4 = self.selections[branch]["e4"](e4)
        return self._decode(self.heads[branch], context, (e0, e1, e2, e3, e4))


def candidate_state(candidate):
    return {
        "deep_blocks": {
            branch: {key: value.detach().cpu()
                     for key, value in blocks.state_dict().items()}
            for branch, blocks in candidate.deep_blocks.items()
        },
        "heads": {
            branch: {key: value.detach().cpu()
                     for key, value in head.state_dict().items()}
            for branch, head in candidate.heads.items()
        },
        "selections": {
            branch: {key: {name: tensor.detach().cpu()
                           for name, tensor in module.state_dict().items()}
                     for key, module in levels.items()}
            for branch, levels in candidate.selections.items()
        },
    }


def restore_candidate(candidate, state):
    for branch in BRANCHES:
        candidate.deep_blocks[branch].load_state_dict(state["deep_blocks"][branch], strict=True)
        candidate.heads[branch].load_state_dict(state["heads"][branch], strict=True)
        for level in ("e3", "e4"):
            candidate.selections[branch][level].load_state_dict(
                state["selections"][branch][level], strict=True)


def save_checkpoint(path, candidate, optimizer, scaler, metadata, epoch,
                    best, stale, include_optimizer):
    payload = {
        "format": "clean_deep_experts_v1",
        "epoch": epoch,
        "candidate": candidate_state(candidate),
        "metadata": metadata,
        "best": best,
        "stale": stale,
    }
    if include_optimizer:
        payload.update({"optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "rng": rng_state()})
    save(path, payload)


def metrics_without_rows(metrics):
    return {key: value for key, value in metrics.items() if key != "rows"}


def run(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for clean deep-branch training")
    device = torch.device("cuda")
    split = load_split()
    output_name = args.output or ("clean_deep_branch_smoke_20260929" if args.smoke
                                  else "clean_deep_branch_matched_20260929")
    if Path(output_name).name != output_name:
        raise ValueError("Output must be a single directory name")
    output = ROOT / "results" / output_name
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Experiment already exists; refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cache = FormalTextCache(ROOT)
    train = CleanRefDataset("train", split["indices"]["train"])
    dev = CleanRefDataset("dev", split["indices"]["dev"])
    train_indices = list(range(len(train)))
    if args.smoke:
        train_indices = train_indices[:args.smoke_train_refs]
    dev_subset = Subset(dev, list(range(min(args.smoke_dev_refs, len(dev))))) if args.smoke else dev
    dev_loader = DataLoader(dev_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    general = make_base(device).eval()
    for parameter in general.parameters():
        parameter.requires_grad_(False)
    candidate = CleanDeepExperts(general).to(device)
    for parameter in candidate.general.parameters():
        parameter.requires_grad_(False)
    deep_parameters = list(candidate.deep_blocks.parameters())
    selection_parameters = list(candidate.selections.parameters())
    head_parameters = list(candidate.heads.parameters())
    trainable = deep_parameters + selection_parameters + head_parameters
    optimizer = torch.optim.AdamW([
        {"params": deep_parameters, "lr": 3e-5},
        {"params": selection_parameters, "lr": 5e-5},
        {"params": head_parameters, "lr": 1e-4},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    router = load_router()
    metadata = {
        "format": "clean_deep_experts_v1",
        "architecture": "one frozen General path for e0-e2 and routing; per-expert copied/trainable e3/e4, selectors, and heads",
        "initialization": "all expert blocks and heads copied from clean General checkpoint",
        "general_sha256": sha256(GENERAL),
        "split_sha256": sha256(SPLIT),
        "cache_sha256": cache.manifest["feature_file_sha256"],
        "train_references": len(train_indices),
        "dev_references": len(dev_subset),
        "test_used": False,
        "presence_gate_used": False,
        "router": "fixed historical text and General-area rule",
        "loss": "same BCE plus Dice/presence loss used by clean expert controls",
        "input_size": 384,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "trainable": "expert-specific e3/e4 blocks, e3/e4 selectors, and heads",
        "learning_rates": {"deep_blocks": 3e-5, "selectors": 5e-5, "heads": 1e-4},
        "parameter_counts": {
            "scope": "image path, text gates and decoder; excludes common frozen BERT",
            "candidate_total": sum(p.numel() for p in candidate.parameters()),
            "general": sum(p.numel() for p in candidate.general.parameters()),
            "branches": {
                branch: sum(p.numel() for p in candidate.deep_blocks[branch].parameters())
                        + sum(p.numel() for p in candidate.selections[branch].parameters())
                        + sum(p.numel() for p in candidate.heads[branch].parameters())
                for branch in BRANCHES
            },
            "active_by_route": {
                branch: sum(p.numel() for p in candidate.general.parameters())
                        + sum(p.numel() for p in candidate.deep_blocks[branch].parameters())
                        + sum(p.numel() for p in candidate.selections[branch].parameters())
                        + sum(p.numel() for p in candidate.heads[branch].parameters())
                for branch in BRANCHES
            },
        },
    }

    history_path = output / "history.json"
    if args.resume:
        checkpoint = torch.load(output / "latest.pth", map_location=device,
                                weights_only=False)
        previous = checkpoint["metadata"]
        for key in ("format", "general_sha256", "split_sha256", "cache_sha256",
                    "seed", "batch_size", "train_references", "epochs_requested"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}")
        metadata = previous
        restore_candidate(candidate, checkpoint["candidate"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        best, stale = checkpoint["best"], checkpoint["stale"]
        start_epoch = checkpoint["epoch"] + 1
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        (output / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        save_checkpoint(output / "initial.pth", candidate, optimizer, scaler,
                        metadata, 0, float("-inf"), 0, False)
        best, stale, start_epoch, history = float("-inf"), 0, 1, []

    for epoch in range(start_epoch, args.epochs + 1):
        order = list(train_indices)
        random.Random(args.seed + epoch).shuffle(order)
        loader = DataLoader(Subset(train, order), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
        random.seed(args.seed + epoch * 100003)
        np.random.seed((args.seed + epoch * 100003) % (2**32))
        torch.manual_seed(args.seed + epoch * 100003)
        torch.cuda.manual_seed_all(args.seed + epoch * 100003)
        candidate.train()
        candidate.general.eval()
        for module in candidate.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        total_loss = 0.0
        seen = 0
        routes = {branch: 0 for branch in BRANCHES}
        trace = hashlib.sha256()
        for step, batch in enumerate(loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            presence = batch["presence"].to(device, non_blocking=True)
            pooled = pooled_for_mixed_batch(cache, batch, device)
            branches, context = routed_batch(general, router, images, pooled,
                                             batch["text"], "shared", candidate)
            for ref_id, sent_id, branch in zip(batch["ref_id"].tolist(),
                                               batch["sent_id"].tolist(), branches):
                trace.update(f"{ref_id}:{sent_id}:{branch}\n".encode("ascii"))
            optimizer.zero_grad(set_to_none=True)
            weighted_losses = []
            for branch in BRANCHES:
                ids = [idx for idx, selected in enumerate(branches) if selected == branch]
                if not ids:
                    continue
                routes[branch] += len(ids)
                selected_context = type(context)(
                    tuple(feature[ids] for feature in context.image),
                    context.pooled_text[ids], context.input_size)
                with torch.autocast("cuda"):
                    logits = candidate.forward_expert(selected_context, branch)
                    loss, _, _ = calculate_loss(logits, masks[ids], presence[ids])
                weighted_losses.append(loss * len(ids))
            loss = sum(weighted_losses) / len(images)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
            seen += len(images)
            if args.smoke and step >= args.smoke_steps:
                break
            if step % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step,
                                  "steps": len(loader), "loss": total_loss / seen,
                                  "routes": routes}), flush=True)

        metrics = evaluate(candidate, "shared", general, router, dev_loader,
                           cache, device)
        rows = metrics.pop("rows")
        row = {"epoch": epoch, "train_loss": total_loss / max(1, seen),
               "train_references_seen": seen, "train_routes": routes,
               "train_trace_sha256": trace.hexdigest(), "dev": metrics}
        history.append(row)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        snapshot = {"format": "clean_deep_experts_v1", "epoch": epoch,
                    "candidate": candidate_state(candidate),
                    "metadata": metadata, "dev": metrics}
        save(output / f"epoch_{epoch:02d}.pth", snapshot)
        if metrics["mean_iou"] > best:
            best, stale = metrics["mean_iou"], 0
            snapshot["best"] = best
            save(output / "best.pth", snapshot)
            (output / "best_dev_rows.json").write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        else:
            stale += 1
        save_checkpoint(output / "latest.pth", candidate, optimizer, scaler,
                        metadata, epoch, best, stale, True)
        print(json.dumps({"epoch": epoch, "train_loss": row["train_loss"],
                          "dev_gIoU": metrics["mean_iou"],
                          "dev_cIoU": metrics["ciou"],
                          "by_bucket": metrics["by_bucket"],
                          "train_trace_sha256": row["train_trace_sha256"]}), flush=True)
        if args.smoke:
            break
        if stale >= args.patience:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-refs", type=int, default=16)
    parser.add_argument("--smoke-dev-refs", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.smoke_steps) < 1:
        parser.error("epochs, batch size, patience, and smoke steps must be positive")
    run(args)
