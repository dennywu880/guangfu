"""Experimental one-encoder training; never changes the deployed checkpoint."""

import argparse
import copy
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import RefELDataset
from inference_engine import BASELINE, DefectInferenceEngine, _PooledTextOnlyEncoder
from shared_moe import BRANCH_FILES, SharedMoE


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "single_encoder_joint_20260924"
ENCODER_PREFIXES = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")


def sample_unique_val(dataset, dev_count, holdout_count, seed):
    ids = list(range(len(dataset)))
    random.Random(seed).shuffle(ids)
    selected = []
    images = set()
    for index in ids:
        image_id = dataset.refs[index]["image_id"]
        if image_id not in images:
            selected.append(index)
            images.add(image_id)
        if len(selected) == dev_count + holdout_count:
            break
    if len(selected) != dev_count + holdout_count:
        raise ValueError("Insufficient distinct validation images")
    dev = selected[:dev_count]
    holdout = selected[dev_count:]
    excluded_train_images = {dataset.refs[i]["image_id"] for i in selected}
    return dev, holdout, excluded_train_images


def sample_train(dataset, excluded_images, count, seed):
    eligible = [i for i, ref in enumerate(dataset.refs)
                if ref["image_id"] not in excluded_images]
    if count > len(eligible):
        raise ValueError("Requested more training references than eligible")
    return random.Random(seed).sample(eligible, count)


def save_model(model, path, metadata):
    state = model.general.state_dict()
    encoder = {key: value.detach().cpu() for key, value in state.items()
               if key.startswith(ENCODER_PREFIXES)}
    torch.save({
        "format": "single_encoder_v1", "metadata": metadata,
        "encoder": encoder,
        "heads": {name: {key: value.detach().cpu() for key, value in head.state_dict().items()}
                  for name, head in model.heads.items()},
    }, path)


def load_model(model, path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format") != "single_encoder_v1":
        raise ValueError("Not a single-encoder checkpoint")
    model.general.load_state_dict(data["encoder"], strict=False)
    for name in BRANCH_FILES:
        model.heads[name].load_state_dict(data["heads"][name], strict=True)


def gate_and_route(model, engine, image, text):
    engine.hook.value = None
    general, context = model.forward_general(image, texts=[text])
    if engine.hook.value is None:
        raise RuntimeError("Presence Gate hook did not fire")
    deep = engine._image_feature_from_hook(engine.hook.value).float()
    stats, values = engine._general_stats(general.sigmoid())
    branch, _, size = BASELINE.route_sample(text, values["area050"])
    gate_input = torch.cat([
        deep, context.pooled_text.float(), stats.float(),
        torch.tensor([BASELINE.route_onehot(branch)], device=image.device),
        torch.tensor([BASELINE.text_onehot(size)], device=image.device),
    ], dim=1)
    reject = bool((engine.gate((gate_input - engine.gate_mean) / engine.gate_std)
                   .sigmoid() >= engine.gate_threshold).item())
    return general, context, branch, reject


def metrics(rows):
    if not rows:
        return {"n": 0}
    no_target = [r for r in rows if r["gt_empty"]]
    return {
        "n": len(rows),
        "mean_IoU": sum(r["iou_regular"] for r in rows) / len(rows),
        "cIoU": sum(r["intersection"] for r in rows) /
                max(1, sum(r["union"] for r in rows)),
        "no_target_n": len(no_target),
        "no_target_false_activation": sum(not r["pred_empty"] for r in no_target),
        "gate_rejections": sum(r["gate_reject"] for r in rows),
    }


def mask_size(mask):
    pixels = int(mask.sum().item())
    if pixels == 0:
        return "no_target"
    ratio = pixels / mask.numel()
    if ratio < 0.001:
        return "ultra_tiny"
    if ratio < 0.006:
        return "tiny"
    return "larger"


@torch.no_grad()
def evaluate(model, engine, dataset, indices, original=False):
    model.eval()
    rows = []
    for step, index in enumerate(indices, 1):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(engine.device)
        gt = item["mask"].unsqueeze(0).to(engine.device) > 0.5
        _, context, branch, reject = gate_and_route(model, engine, image, item["text"])
        if reject:
            pred = torch.zeros_like(gt)
        elif original:
            pred = engine._activate_expert(branch)(
                image, pooled_text=context.pooled_text).sigmoid() >= 0.5
        else:
            pred = model.forward_expert(context, branch).sigmoid() >= 0.5
        result = BASELINE.binary_stats(pred, gt)
        rows.append({"index": index, "branch": branch, "size": mask_size(gt),
                     "gate_reject": reject,
                     "gt_empty": not bool(gt.any().item()), **result})
        if step % 100 == 0:
            print(f"Evaluate {'original' if original else 'single'} {step}/{len(indices)}",
                  flush=True)
    result = metrics(rows)
    result["by_branch"] = {branch: metrics([r for r in rows if r["branch"] == branch])
                           for branch in BRANCH_FILES}
    result["by_gt_size"] = {size: metrics([r for r in rows if r["size"] == size])
                            for size in ("no_target", "ultra_tiny", "tiny", "larger")}
    result["rows"] = rows
    return result


@torch.no_grad()
def routed_train_items(model, engine, dataset, indices):
    model.eval()
    groups = defaultdict(list)
    for step, index in enumerate(indices, 1):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(engine.device)
        _, _, branch, _ = gate_and_route(model, engine, image, item["text"])
        groups[branch].append((index, item["text"]))
        if step % 200 == 0:
            print(f"Route training items {step}/{len(indices)}", flush=True)
    return groups


def supervised_loss(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits.float(), mask)
    probs = logits.float().sigmoid()
    overlap = (probs * mask).sum()
    dice = 1 - (2 * overlap + 1) / (probs.sum() + mask.sum() + 1)
    return bce + dice


def set_training_phase(model, joint):
    for param in model.general.parameters():
        param.requires_grad_(False)
    for name, param in model.general.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            param.requires_grad_(joint)
    for param in model.heads.parameters():
        param.requires_grad_(True)
    model.eval()
    # Batch size 1 makes BatchNorm running estimates unstable; retain the
    # checkpoint statistics while training affine/decoder weights.
    model.heads.eval()


def train_phase(model, engine, train, indices, phase, teacher_general, args, history):
    joint = phase == "joint"
    set_training_phase(model, joint)
    groups = routed_train_items(model, engine, train, indices)
    set_training_phase(model, joint)
    optimizer = torch.optim.AdamW([
        {"params": list(model.heads.parameters()), "lr": args.head_lr},
        {"params": [p for p in model.general.parameters() if p.requires_grad],
         "lr": args.encoder_lr},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=engine.device.type == "cuda")
    total_loss = steps = 0
    started = time.perf_counter()
    training_items = [(branch, index, text) for branch, members in groups.items()
                      for index, text in members]
    random.Random(args.seed + (11 if joint else 10)).shuffle(training_items)
    teachers = {branch: engine.experts[branch].to(engine.device).eval()
                for branch in BRANCH_FILES}
    for branch, index, text in training_items:
        item = train[index]
        image = item["image"].unsqueeze(0).to(engine.device)
        mask = item["mask"].unsqueeze(0).to(engine.device).float()
        with torch.no_grad():
            _, pooled, _ = model.general.text_encoder([text], engine.device)
            teacher_logits = teachers[branch](image, pooled_text=pooled).detach()
            teacher_general_logits = (teacher_general(image, pooled_text=pooled).detach()
                                      if joint else None)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=engine.device.type == "cuda"):
            general, context = model.forward_general(image, pooled_text=pooled)
            student_logits = model.forward_expert(context, branch)
            loss = supervised_loss(student_logits, mask)
            if bool(mask.any().item()):
                target = teacher_logits.float().sigmoid()
                soft = F.binary_cross_entropy_with_logits(
                    student_logits.float(), target, reduction="none")
                loss = loss + args.distill_weight * (soft * (1 + 4 * target)).mean()
            if joint:
                loss = loss + args.anchor_weight * F.smooth_l1_loss(
                    general.float(), teacher_general_logits.float())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {phase} loss at index {index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], 1.0)
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        total_loss += float(loss.detach())
        if steps % 100 == 0:
            print(f"{phase} {steps}/{len(indices)} loss={total_loss/steps:.4f} "
                  f"elapsed={time.perf_counter()-started:.0f}s", flush=True)
        if steps % args.checkpoint_every == 0:
            save_model(model, OUT / "latest.pth", {"phase": phase, "steps": steps})
    row = {"phase": phase, "steps": steps,
           "train_loss": total_loss / max(1, steps),
           "elapsed_seconds": time.perf_counter() - started,
           "routed_counts": {branch: len(groups[branch]) for branch in BRANCH_FILES}}
    history.append(row)
    save_model(model, OUT / f"{phase}.pth", row)
    save_model(model, OUT / "latest.pth", row)
    return row


def concise(result):
    return {key: value for key, value in result.items() if key != "rows"}


def run(args):
    global OUT
    OUT = args.output
    os.environ["TG_SHARED_MOE"] = "0"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    engine = DefectInferenceEngine()
    model = SharedMoE.from_checkpoints(ROOT / "checkpoints", engine.device,
                                       general=engine.general_model)
    train = RefELDataset(ROOT, split="train", image_size=384, random_text=True)
    val = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    dev_ids, holdout_ids, excluded_images = sample_unique_val(
        val, args.dev, args.holdout, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    split = {"seed": args.seed, "dev_indices": dev_ids,
             "holdout_indices": holdout_ids,
             "excluded_image_ids_from_training": sorted(excluded_images),
             "note": "Original pretrained checkpoints may have previously seen these images."}
    (OUT / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    history = []
    with torch.no_grad():
        original_dev = evaluate(model, engine, val, dev_ids, original=True)
        single_dev = evaluate(model, engine, val, dev_ids)
        original_holdout = evaluate(model, engine, val, holdout_ids, original=True)
    baseline = {"original": concise(original_dev), "single": concise(single_dev)}
    (OUT / "dev_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    save_model(model, OUT / "initial.pth", {"phase": "initial", "dev": concise(single_dev)})
    print(f"Baseline dev: {baseline}", flush=True)

    teacher_general = copy.deepcopy(model.general)
    teacher_general.text_encoder = _PooledTextOnlyEncoder()
    teacher_general.eval()
    for param in teacher_general.parameters():
        param.requires_grad_(False)

    for phase, count in (("heads", args.head_samples), ("joint", args.joint_samples)):
        if count == 0:
            continue
        indices = sample_train(train, excluded_images, count,
                               args.seed + (1 if phase == "heads" else 2))
        result = train_phase(model, engine, train, indices, phase,
                             teacher_general, args, history)
        with torch.no_grad():
            score = evaluate(model, engine, val, dev_ids)
        result["dev"] = concise(score)
        (OUT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Completed {phase}: {result}", flush=True)

    candidates = [("initial", concise(single_dev))]
    candidates.extend((row["phase"], row["dev"]) for row in history)
    original_summary = concise(original_dev)
    eligible = [
        (name, score) for name, score in candidates
        if score["cIoU"] >= original_summary["cIoU"] - 0.02
        and score["no_target_false_activation"] <=
        original_summary["no_target_false_activation"] + 2
    ]
    chosen_name, chosen_dev = max(
        eligible or candidates,
        key=lambda item: (item[1]["mean_IoU"], item[1]["cIoU"]),
    )
    load_model(model, OUT / f"{chosen_name}.pth")
    selection = {"chosen_on_dev": chosen_name, "dev": chosen_dev,
                 "satisfied_guards": bool(eligible),
                 "candidate_names": [name for name, _ in candidates]}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Selected on dev: {selection}", flush=True)

    # The holdout is not used for selection. Original predictions were frozen
    # before joint tuning, because tuning changes the General encoder.
    with torch.no_grad():
        holdout = {
            "original": concise(original_holdout),
            "single_selected": concise(evaluate(model, engine, val, holdout_ids)),
        }
    holdout["selection"] = selection
    (OUT / "holdout.json").write_text(json.dumps(holdout, indent=2), encoding="utf-8")
    print(f"Holdout: {holdout}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-samples", type=int, default=600)
    parser.add_argument("--joint-samples", type=int, default=900)
    parser.add_argument("--dev", type=int, default=160)
    parser.add_argument("--holdout", type=int, default=160)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path, default=OUT)
    arguments = parser.parse_args()
    if min(arguments.head_samples, arguments.joint_samples, arguments.dev,
           arguments.holdout) < 0 or arguments.dev == 0 or arguments.holdout == 0:
        parser.error("Counts must be nonnegative; dev and holdout must be positive")
    run(arguments)
