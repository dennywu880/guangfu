"""Train a leakage-free general TG-UNet baseline on the fixed image split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "pretrained_cache"))
if "HF_HOME" not in os.environ and Path("D:/thunet新/hf-cache").is_dir():
    os.environ["HF_HOME"] = "D:/thunet新/hf-cache"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RefELDataset
from model_text import TextGuidedUNet
from train_detail_adapter import size_bucket
from train_text import FormalTextCache, calculate_loss, create_optimizer, text_sha256


SPLIT = ROOT / "results" / "source_image_split_20260925" / "split.json"
GREFS = ROOT / "datasets" / "grefcoco" / "grefs(unc).json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split() -> dict:
    manifest = json.loads(SPLIT.read_text(encoding="utf-8"))
    if manifest.get("format") != "source_image_disjoint_refel_v1":
        raise ValueError("Unexpected source-image split format")
    if manifest["source_sha256"] != sha256(GREFS):
        raise ValueError("Annotations changed after the split was made")
    refs = json.loads(GREFS.read_text(encoding="utf-8"))
    partitions = ("train", "dev", "test")
    selected = [index for part in partitions for index in manifest["indices"][part]]
    if sorted(selected) != list(range(len(refs))):
        raise ValueError("Split must assign each reference exactly once")
    image_sets = []
    for part in partitions:
        indices = manifest["indices"][part]
        images = {int(refs[index]["image_id"]) for index in indices}
        if images != set(manifest["image_ids"][part]):
            raise ValueError(f"Image IDs do not match the {part} manifest")
        image_sets.append(images)
    if any(image_sets[i] & image_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Source-image leakage across partitions")
    return manifest


class CleanRefDataset(RefELDataset):
    def __init__(self, partition: str, indices: list[int]):
        super().__init__(ROOT, "train" if partition == "train" else "val",
                         384, partition == "train")
        all_refs = json.loads(GREFS.read_text(encoding="utf-8"))
        self.refs = [all_refs[index] for index in indices]
        self.partition = partition

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["original_split"] = self.refs[index]["split"]
        return item


def pooled_for_mixed_batch(cache: FormalTextCache, batch: dict,
                           device: torch.device) -> torch.Tensor:
    ids = batch["sent_id"].long().view(-1)
    for row, sent_id in enumerate(ids.tolist()):
        if not 0 <= sent_id < len(cache.features):
            raise ValueError(f"Invalid sentence ID {sent_id}")
        if int(cache.ref_ids[sent_id]) != int(batch["ref_id"][row]):
            raise ValueError(f"Reference ID mismatch for sentence {sent_id}")
        expected = cache.split_to_code[batch["original_split"][row]]
        if int(cache.split_codes[sent_id]) != expected:
            raise ValueError(f"Original split mismatch for sentence {sent_id}")
        if cache.text_hashes[sent_id] != text_sha256(batch["text"][row]):
            raise ValueError(f"Text mismatch for sentence {sent_id}")
    return cache.features[ids].contiguous().to(device, non_blocking=True)


def summarize(rows: list[dict]) -> dict:
    def group(items):
        if not items:
            return {"count": 0, "mean_iou": None, "ciou": None}
        union = sum(item["union"] for item in items)
        return {
            "count": len(items),
            "mean_iou": sum(item["iou"] for item in items) / len(items),
            "ciou": sum(item["intersection"] for item in items) / union if union else 1.0,
        }

    buckets = {name: group([row for row in rows if row["bucket"] == name])
               for name in ("no_target", "ultra_tiny", "tiny", "larger")}
    positive = group([row for row in rows if row["bucket"] != "no_target"])
    negative = buckets["no_target"]
    result = group(rows)
    result.update({
        "positive_mean_iou": positive["mean_iou"],
        "no_target_accuracy": negative["mean_iou"],
        "false_activations": sum(row["bucket"] == "no_target" and row["union"] > 0
                                 for row in rows),
        "by_bucket": buckets,
    })
    return result


@torch.inference_mode()
def evaluate(model, loader, cache, device, limit: int | None = None):
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True) > 0.5
        pooled = pooled_for_mixed_batch(cache, batch, device)
        prediction = model(images, pooled_text=pooled).sigmoid() >= 0.5
        for pred, gt in zip(prediction, masks):
            inter = int((pred & gt).sum().item())
            union = int((pred | gt).sum().item())
            rows.append({
                "bucket": size_bucket(gt), "intersection": inter, "union": union,
                "iou": inter / union if union else 1.0,
            })
            if limit is not None and len(rows) >= limit:
                return summarize(rows)
    return summarize(rows)


def vision_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not key.startswith("text_encoder.")}


def save(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_vision_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("text_encoder.") for key in missing):
        raise ValueError("Checkpoint does not match the TG-UNet vision model")


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def run(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RTX 5060 CUDA is required for this training run")
    split = load_split()
    output_name = args.output or ("clean_general_smoke" if args.smoke
                                  else "clean_general_source_disjoint")
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
    train_loader = DataLoader(torch.utils.data.Subset(train, train_indices),
                              batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    metadata = {
        "format": "clean_general_v1", "split_sha256": sha256(SPLIT),
        "cache_sha256": cache.manifest["feature_file_sha256"],
        "source_image_disjoint": True, "train_references": len(train_indices),
        "dev_references": len(dev), "test_used": False,
        "image_size": 384, "batch_size": args.batch_size,
        "epochs_requested": args.epochs, "seed": args.seed,
        "initialization": "torchvision ResNet34 ImageNet; frozen bert-base-uncased",
        "checkpoint_policy": "initial, latest, best; original weights untouched",
    }
    model = TextGuidedUNet(pretrained=True).to(device)
    model.text_encoder.cpu()
    model.text_encoder.eval()
    optimizer = create_optimizer(model)
    scaler = torch.amp.GradScaler("cuda")
    if args.resume:
        checkpoint = torch.load(output / "latest.pth", map_location=device,
                                weights_only=False)
        previous = checkpoint["metadata"]
        for key in ("format", "split_sha256", "cache_sha256", "seed",
                    "batch_size", "train_references"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}")
        load_vision_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        best = checkpoint["best"]
        stale = checkpoint["stale"]
        start_epoch = checkpoint["epoch"] + 1
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    else:
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        initial = {"format": "clean_general_v1", "epoch": 0,
                   "model": vision_state(model), "metadata": metadata}
        save(output / "initial.pth", initial)
        best = float("-inf")
        stale = 0
        start_epoch = 1
        history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        model.text_encoder.eval()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            presence = batch["presence"].to(device, non_blocking=True)
            pooled = pooled_for_mixed_batch(cache, batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=True):
                logits = model(images, pooled_text=pooled)
                loss, _, _ = calculate_loss(logits, masks, presence)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]], 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            if step % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step,
                                  "steps": len(train_loader), "loss": running / step}), flush=True)
        metrics = evaluate(model, dev_loader, cache, device,
                           args.smoke_dev_refs if args.smoke else None)
        row = {"epoch": epoch, "train_loss": running / len(train_loader),
               "dev": metrics}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = metrics["mean_iou"]
        if score > best:
            best = score
            stale = 0
            save(output / "best.pth", {"format": "clean_general_v1", "epoch": epoch,
                                        "model": vision_state(model), "metadata": metadata,
                                        "dev": metrics})
        else:
            stale += 1
        save(output / "latest.pth", {
            "format": "clean_general_v1", "epoch": epoch,
            "model": vision_state(model), "metadata": metadata,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng_state(), "best": best, "stale": stale,
        })
        epoch_snapshot = output / f"epoch_{epoch:02d}.pth"
        if epoch_snapshot.exists():
            raise FileExistsError(f"Refusing to overwrite an epoch snapshot: {epoch_snapshot}")
        save(epoch_snapshot, {"format": "clean_general_v1", "epoch": epoch,
                              "model": vision_state(model), "metadata": metadata,
                              "dev": metrics})
        print(json.dumps({"epoch": epoch, "dev_giou": score,
                          "dev_ciou": metrics["ciou"],
                          "positive_mean_iou": metrics["positive_mean_iou"],
                          "no_target_accuracy": metrics["no_target_accuracy"]}), flush=True)
        if not args.smoke and stale >= args.patience:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--smoke-train-refs", type=int, default=16)
    parser.add_argument("--smoke-dev-refs", type=int, default=8)
    options = parser.parse_args()
    if min(options.epochs, options.batch_size, options.patience,
           options.smoke_train_refs, options.smoke_dev_refs) < 1:
        parser.error("All counts must be positive")
    run(options)
