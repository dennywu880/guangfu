"""Evaluate decoder-sharing candidates from the selected shared-encoder checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dataset import RefELDataset
from multilevel_selection import build_model
from shared_moe import BRANCH_FILES, SharedFeatures
from train_detail_adapter import (
    ROOT, build_disjoint_train_pool, evaluate, load_or_create_split,
    load_router, pooled_for, segmentation_loss, size_bucket, stack_items,
)
from train_multilevel_head_joint import load as load_head_joint
from train_shared_layer4_joint import (
    DEVELOPMENT_INDICES, load as load_layer4_checkpoint,
)
from train_text import FormalTextCache


SHARE_CANDIDATES = {
    "private": (),
    "output": ("final_conv", "segmentation_head"),
    "late": ("decoder1", "final_conv", "segmentation_head"),
    "decoder4": ("decoder4",),
    "deep3": ("decoder4", "decoder3"),
    "full_decoder": (
        "decoder4", "decoder3", "decoder2", "decoder1",
        "final_conv", "segmentation_head",
    ),
}
HEAD_JOINT_CHECKPOINT = ROOT / "results" / "head_joint_e3_e4_20260925" / "best.pth"
LAYER4_CHECKPOINT = ROOT / "results" / "layer4_joint_20260926" / "epoch_02.pth"
ENCODER_SHARED_DIR = ROOT / "results" / "split_ratio_existing_best_20261001"
ENCODER_SHARED_CHECKPOINT = ENCODER_SHARED_DIR / "shared" / "epoch_02.pth"
ENCODER_SHARED_MANIFEST = ENCODER_SHARED_DIR / "manifest.json"
ENCODER_SHARED_METRICS = ENCODER_SHARED_DIR / "shared" / "epoch_02_metrics.json"
HEAD_COMPONENTS = (
    "text_gate_e4", "text_gate_e3", "text_gate_e2",
    "decoder4", "decoder3", "decoder2", "decoder1",
    "final_conv", "segmentation_head",
)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected_shared_encoder(model):
    """Restore the ancestor chain and exact selected shared-encoder state."""
    required = (
        HEAD_JOINT_CHECKPOINT, LAYER4_CHECKPOINT, ENCODER_SHARED_CHECKPOINT,
        ENCODER_SHARED_MANIFEST, ENCODER_SHARED_METRICS,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("A checkpoint or manifest in the selected source chain is missing")

    manifest = json.loads(ENCODER_SHARED_MANIFEST.read_text(encoding="utf-8"))
    if Path(manifest["source_checkpoint"]) != LAYER4_CHECKPOINT:
        raise ValueError("Encoder split manifest points to an unexpected layer4 source")
    if manifest["source_sha256"] != sha256(LAYER4_CHECKPOINT):
        raise ValueError("Layer4 source hash does not match the encoder split manifest")
    if manifest["base_checkpoint"] != str(HEAD_JOINT_CHECKPOINT):
        raise ValueError("Encoder split manifest points to an unexpected base checkpoint")
    if manifest["base_sha256"] != sha256(HEAD_JOINT_CHECKPOINT):
        raise ValueError("Head-joint base hash does not match the encoder split manifest")

    # The selected encoder checkpoint is incremental: restore its layer4-joint
    # parent first, then apply this candidate's layer4, heads, and selectors.
    load_layer4_checkpoint(LAYER4_CHECKPOINT, model)
    state = torch.load(ENCODER_SHARED_CHECKPOINT, map_location="cpu", weights_only=False)
    if state.get("format") != "split_ratio_from_existing_v1":
        raise ValueError("Unsupported selected encoder checkpoint format")
    if state.get("split") != "shared" or state.get("epoch") != 2:
        raise ValueError("Expected the epoch-2 fully shared encoder candidate")
    if state.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("Selected encoder checkpoint has a different parent hash")
    if state.get("branches"):
        raise ValueError("Selected shared-encoder checkpoint unexpectedly has private encoder branches")
    if set(state["heads"]) != set(model.base.heads):
        raise ValueError("Selected encoder checkpoint has an incomplete expert-head set")
    if set(state["selections"]) != set(model.selections):
        raise ValueError("Selected encoder checkpoint has an incomplete selector set")

    model.base.general.layer4.load_state_dict(state["layer4"], strict=True)
    for branch, head in model.base.heads.items():
        head.load_state_dict(state["heads"][branch], strict=True)
        if set(state["selections"][branch]) != set(model.selections[branch]):
            raise ValueError(f"Selected encoder checkpoint is missing selectors for {branch}")
        for level, selector in model.selections[branch].items():
            selector.load_state_dict(state["selections"][branch][level], strict=True)
    return manifest, state


def averaged_module_state(heads, module_name):
    states = [getattr(head, module_name).state_dict() for head in heads]
    result = {}
    for key in states[0]:
        values = [state[key] for state in states]
        if values[0].is_floating_point() or values[0].is_complex():
            result[key] = torch.stack(values).mean(0).to(values[0].dtype)
        else:
            result[key] = torch.stack(values).max(0).values
    return result


class DecoderSharingCandidate(nn.Module):
    """Keep the General path fixed and tie selected expert decoder modules."""

    def __init__(self, source_model, name, shared_names):
        super().__init__()
        self.base = source_model.base
        self.selections = source_model.selections
        self.selection_levels = source_model.levels
        self.name = name
        self.shared_names = tuple(shared_names)
        self.shared_modules = nn.ModuleDict()
        branches = tuple(BRANCH_FILES)
        heads = [self.base.heads[branch] for branch in branches]
        for module_name in self.shared_names:
            module = deepcopy(getattr(heads[0], module_name))
            module.load_state_dict(averaged_module_state(heads, module_name), strict=True)
            self.shared_modules[module_name] = module
            for head in heads:
                setattr(head, module_name, None)

    def forward_general(self, *args, **kwargs):
        return self.base.forward_general(*args, **kwargs)

    def forward_expert(self, context, branch):
        head = self.base.heads[branch]
        features = list(self.base.detail_features(context, branch))
        for level in self.selection_levels:
            features[int(level[1:])] = self.selections[branch][level](
                features[int(level[1:])]
            )
        e0, e1, e2, e3, e4 = features

        def module(name):
            shared = self.shared_modules[name] if name in self.shared_modules else None
            return shared if shared is not None else getattr(head, name)

        x = module("decoder4")(
            head.text_gate_e4(e4, context.pooled_text),
            head.text_gate_e3(e3, context.pooled_text),
        )
        x = module("decoder3")(x, head.text_gate_e2(e2, context.pooled_text))
        x = module("decoder2")(x, e1)
        x = module("decoder1")(x, e0)
        x = F.interpolate(x, size=context.input_size,
                           mode="bilinear", align_corners=False)
        return module("segmentation_head")(module("final_conv")(x))

    def trainable_groups(self):
        selection = list(self.selections.parameters())
        expert = list(self.base.heads.parameters())
        shared = list(self.shared_modules.parameters())
        return selection, expert + shared

    def parameter_counts(self):
        general = sum(p.numel() for p in self.base.general.parameters())
        selectors = sum(p.numel() for p in self.selections.parameters())
        all_heads = sum(p.numel() for p in self.base.heads.parameters())
        shared = sum(p.numel() for p in self.shared_modules.parameters())
        per_head = sum(p.numel() for p in self.base.heads[tuple(BRANCH_FILES)[0]].parameters())
        per_selector = sum(p.numel() for p in self.selections[tuple(BRANCH_FILES)[0]].parameters())
        return {
            "total_image_model_parameters_cached_text_excluded": general + selectors + all_heads + shared,
            "active_one_route_image_model_parameters_cached_text_excluded": general + per_head + per_selector + shared,
            "branch_specific_head_parameters": all_heads,
            "shared_decoder_parameters": shared,
            "external_text_encoder_counted_once_in_deployment": True,
        }

    def save_checkpoint(self, path, metadata, epoch):
        torch.save({
            "format": "decoder_sharing_joint_v1",
            "source_sha256": metadata["source_sha256"],
            "candidate": self.name,
            "shared_names": self.shared_names,
            "epoch": epoch,
            "shared_modules": {key: module.state_dict()
                               for key, module in self.shared_modules.items()},
            "heads": {key: head.state_dict()
                      for key, head in self.base.heads.items()},
            "selections": {
                branch: {level: module.state_dict()
                         for level, module in levels.items()}
                for branch, levels in self.selections.items()
            },
        }, path)


def make_candidate(device, name):
    source_model = build_model(device, ("e3", "e4"))
    load_selected_shared_encoder(source_model)
    model = DecoderSharingCandidate(
        source_model, name, SHARE_CANDIDATES[name]
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.selections.parameters():
        parameter.requires_grad_(True)
    for parameter in model.base.heads.parameters():
        parameter.requires_grad_(True)
    for parameter in model.shared_modules.parameters():
        parameter.requires_grad_(True)
    return model


def context_select(context, indices):
    return SharedFeatures(
        tuple(feature.index_select(0, indices) for feature in context.image),
        context.pooled_text.index_select(0, indices),
        context.input_size,
        context.source_image.index_select(0, indices)
        if context.source_image is not None else None,
    )


def train_epoch(model, optimizer, dataset, indices, router, cache,
                device, args, epoch):
    selection_params, expert_params = model.trainable_groups()
    shuffled = list(indices)
    random.Random(args.seed + epoch).shuffle(shuffled)
    model.eval()
    model.base.eval()
    losses = []
    for start in range(0, len(shuffled), args.batch_size):
        batch_ids = shuffled[start:start + args.batch_size]
        items, images, masks, texts = stack_items(dataset, batch_ids, device)
        pooled = pooled_for(cache, items, "train", device)
        with torch.no_grad():
            general, context = model.forward_general(images, pooled_text=pooled)
            areas = (general.sigmoid() >= 0.5).float().flatten(1).mean(1).tolist()
        routes = [router.route_sample(text, float(area))[0]
                  for text, area in zip(texts, areas)]
        branch_ids = {
            name: torch.tensor([i for i, route in enumerate(routes) if route == name],
                               dtype=torch.long, device=device)
            for name in BRANCH_FILES
        }
        optimizer.zero_grad(set_to_none=True)
        combined = None
        used = 0
        for branch, offsets in branch_ids.items():
            if offsets.numel() == 0:
                continue
            selected_context = context_select(context, offsets)
            logits = model.forward_expert(selected_context, branch)
            loss = segmentation_loss(logits, masks.index_select(0, offsets))
            weighted = loss * (offsets.numel() / len(batch_ids))
            combined = weighted if combined is None else combined + weighted
            used += int(offsets.numel())
        if combined is None or used != len(batch_ids):
            raise RuntimeError("A training batch was not assigned to exactly one expert")
        combined.backward()
        torch.nn.utils.clip_grad_norm_(selection_params + expert_params, 5.0)
        optimizer.step()
        losses.append(float(combined.detach()))
    return sum(losses) / max(1, len(losses))


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA unavailable; pass --cpu for explicit CPU execution")
    device = torch.device("cpu" if args.cpu else "cuda")
    output = ROOT / "results" / args.output
    output.mkdir(parents=True, exist_ok=False)
    if not DEVELOPMENT_INDICES.is_file():
        raise FileNotFoundError("Required fixed development indices are missing")

    train = RefELDataset(ROOT, "train", 384, False)
    val = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    dev = json.loads(DEVELOPMENT_INDICES.read_text(encoding="utf-8"))
    old_dev, holdout = load_or_create_split(val, args.seed)
    pool, excluded, split_audit = build_disjoint_train_pool(train, val, old_dev, holdout)
    dev_images = {val.refs[i]["image_id"] for i in dev}
    excluded |= dev_images
    pool = [i for i in pool if train.refs[i]["image_id"] not in excluded]
    if len(dev) != 1200 or len(dev_images) != 1200:
        raise ValueError("Expected the fixed 1,200-reference, image-distinct development set")
    if args.smoke_samples:
        dev = dev[:args.smoke_samples]
    if len(pool) < args.train_samples:
        raise ValueError("Eligible existing-dataset training pool is too small")

    router = load_router()
    encoder_manifest = json.loads(ENCODER_SHARED_MANIFEST.read_text(encoding="utf-8"))
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(ENCODER_SHARED_CHECKPOINT),
        "source_sha256": sha256(ENCODER_SHARED_CHECKPOINT),
        "source_choice_reason": "selected fully shared encoder candidate from the existing split-ratio study; decoder-sharing results are conditional on this encoder allocation",
        "source_ancestors": {
            "layer4_joint_checkpoint": str(LAYER4_CHECKPOINT),
            "layer4_joint_sha256": sha256(LAYER4_CHECKPOINT),
            "head_joint_checkpoint": str(HEAD_JOINT_CHECKPOINT),
            "head_joint_sha256": sha256(HEAD_JOINT_CHECKPOINT),
        },
        "encoder_split_manifest_sha256": sha256(ENCODER_SHARED_MANIFEST),
        "encoder_split_manifest_source_sha256": encoder_manifest["source_sha256"],
        "candidate_shared_modules": {name: SHARE_CANDIDATES[name]
                                     for name in args.candidates},
        "shared_weight_initialization": "arithmetic mean of the three existing expert module state dicts; integer buffers use maximum",
        "training": "joint optimizer updates shared modules and all expert-private heads/adapters; branch losses accumulate before each optimizer step",
        "fixed_components": ["image encoder", "text encoder/cache", "General path", "text/area router", "Presence Gate omitted in this evaluator"],
        "input_size": 384,
        "threshold": 0.5,
        "loss": "BCE plus Dice, unweighted",
        "epochs": args.epochs,
        "train_samples_per_epoch": args.train_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "selector_lr": args.selector_lr,
        "head_lr": args.head_lr,
        "development_indices_sha256": hashlib.sha256(DEVELOPMENT_INDICES.read_bytes()).hexdigest(),
        "development_references": len(dev),
        "eligible_train_references": len(pool),
        "split_audit": split_audit,
        "historical_checkpoint_exposure": "possible; development results are comparative diagnostics, not blind generalization claims",
        "testB_used": False,
        "other_dataset_used": False,
        "web_changed": False,
    }
    write_json(output / "manifest.json", metadata)
    (output / "parameter_audit.json").write_text("{}", encoding="utf-8")

    audit_model = make_candidate(device, "private")
    audit_params = {}
    for name in args.candidates:
        model = make_candidate(device, name)
        audit_params[name] = model.parameter_counts()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(output / "parameter_audit.json", audit_params)
    del audit_model

    # Low-savings output-only candidates are recorded at conversion time. Full
    # training is reserved for structures that can materially reduce storage.
    baseline_total = audit_params["private"]["total_image_model_parameters_cached_text_excluded"]
    trained = ["private"]
    for name in args.candidates:
        if name == "private":
            continue
        total = audit_params[name]["total_image_model_parameters_cached_text_excluded"]
        saved_fraction = (baseline_total - total) / baseline_total
        if saved_fraction >= args.minimum_saving_fraction:
            trained.append(name)
    metadata["trained_candidates"] = trained
    metadata["minimum_parameter_saving_fraction_for_joint_training"] = args.minimum_saving_fraction
    write_json(output / "manifest.json", metadata)

    history = {}
    for name in args.candidates:
        print(f"\n=== Candidate {name} ===", flush=True)
        model = make_candidate(device, name)
        count = model.parameter_counts()
        model.eval()
        initial_full = evaluate(
            model, val, dev, router, device, cache, "val", args.eval_batch_size
        )
        initial_rows = initial_full["rows"]
        initial_metrics = {k: v for k, v in initial_full.items() if k != "rows"}
        if name == "private" and not args.smoke_samples:
            source_metrics = json.loads(
                ENCODER_SHARED_METRICS.read_text(encoding="utf-8")
            )["metrics"]
            for key in ("mean_IoU", "cIoU"):
                if abs(initial_metrics[key] - source_metrics[key]) > 1e-5:
                    raise RuntimeError(
                        f"Loaded source checkpoint does not reproduce saved {key}: "
                        f"{initial_metrics[key]:.8f} vs {source_metrics[key]:.8f}"
                    )
        model.save_checkpoint(output / f"{name}_epoch_00.pth", metadata, 0)
        history[name] = [{"epoch": 0, "metrics": initial_metrics,
                          "parameter_counts": count,
                          "joint_training_scheduled": name in trained}]
        write_json(output / f"{name}_epoch_00_metrics.json", {
            "metrics": initial_metrics, "rows": initial_rows
        })
        print({"candidate": name, "epoch": 0,
               "gIoU": initial_metrics["mean_IoU"],
               "cIoU": initial_metrics["cIoU"],
               "false_activation": initial_metrics["false_activation"],
               "parameters": count}, flush=True)

        if name in trained:
            selection_params, expert_params = model.trainable_groups()
            optimizer = torch.optim.AdamW([
                {"params": selection_params, "lr": args.selector_lr},
                {"params": expert_params, "lr": args.head_lr},
            ], weight_decay=1e-4)
            for epoch in range(1, args.epochs + 1):
                epoch_ids = list(pool)
                random.Random(args.seed + epoch).shuffle(epoch_ids)
                epoch_ids = epoch_ids[:args.train_samples]
                loss = train_epoch(model, optimizer, train, epoch_ids,
                                   router, cache, device, args, epoch)
                full_metrics = evaluate(
                    model, val, dev, router, device, cache, "val", args.eval_batch_size
                )
                rows = full_metrics["rows"]
                metrics = {k: v for k, v in full_metrics.items() if k != "rows"}
                row = {"epoch": epoch, "mean_train_loss": loss,
                       "metrics": metrics, "parameter_counts": count}
                history[name].append(row)
                model.save_checkpoint(output / f"{name}_epoch_{epoch:02d}.pth",
                                      metadata, epoch)
                write_json(output / f"{name}_epoch_{epoch:02d}_metrics.json", {
                    "metrics": metrics, "rows": rows
                })
                write_json(output / "history.json", history)
                print({"candidate": name, "epoch": epoch,
                       "train_loss": loss,
                       "gIoU": metrics["mean_IoU"], "cIoU": metrics["cIoU"],
                       "ultra_tiny": metrics["by_bucket"]["ultra_tiny"]["mean_IoU"],
                       "tiny": metrics["by_bucket"]["tiny"]["mean_IoU"],
                       "false_activation": metrics["false_activation"]}, flush=True)
        write_json(output / "history.json", history)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(output / "completion.json", {
        "completed": True,
        "trained_candidates": trained,
        "source_unchanged": sha256(ENCODER_SHARED_CHECKPOINT) == metadata["source_sha256"],
        "source_ancestors_unchanged": (
            sha256(LAYER4_CHECKPOINT) == metadata["source_ancestors"]["layer4_joint_sha256"]
            and sha256(HEAD_JOINT_CHECKPOINT) == metadata["source_ancestors"]["head_joint_sha256"]
        ),
        "web_changed": False,
        "testB_used": False,
        "promotion": "none; compare Pareto tradeoffs and confirm finalists before any deployment",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="decoder_sharing_from_shared_encoder_20261008")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--selector-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--minimum-saving-fraction", type=float, default=0.01)
    parser.add_argument("--smoke-samples", type=int, default=0,
                        help="Use a small development slice for pipeline smoke tests")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--candidates", nargs="+", choices=tuple(SHARE_CANDIDATES),
                        default=list(SHARE_CANDIDATES),
                        help="Candidate structures to evaluate/train")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.train_samples, args.batch_size,
           args.eval_batch_size) < 1:
        parser.error("epochs, sample counts, and batch sizes must be positive")
    if not 0 <= args.minimum_saving_fraction < 1:
        parser.error("minimum-saving-fraction must be in [0, 1)")
    if "private" not in args.candidates:
        parser.error("candidate list must include private as the matched control")
    if args.smoke_samples < 0:
        parser.error("smoke-samples cannot be negative")
    run(args)
