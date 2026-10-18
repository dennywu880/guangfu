import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import RefELDataset
from model_text import TextGuidedUNet
from train_text import FormalTextCache


# ============================================================
# STEP 43
# FINAL FROZEN TESTB:
#   Presence Gate v2
#   + newest Ultra / Tiny / Larger Experts
#   + M2SegNet-style evaluation metrics
#
# IMPORTANT:
# - NO training
# - NO threshold search
# - NO checkpoint selection
# - NO TestB-based tuning
# - all configuration is frozen from Train/Val
# ============================================================

ROOT = Path(__file__).resolve().parent

IMAGE_SIZE = 384
BATCH_SIZE = 4
NUM_WORKERS = 0

GENERAL_MASK_THRESHOLD = 0.5
EXPERT_MASK_THRESHOLD = 0.5

ROUTER_ULTRA_TINY = 0.002550
ROUTER_TINY_LARGER = 0.016290

GROUPS = ["ultra_tiny", "tiny", "larger"]

# Only for diagnostic size breakdown; does NOT affect routing.
GT_ULTRA_THRESHOLD = 0.001
GT_TINY_THRESHOLD = 0.006


# ============================================================
# Frozen checkpoints
# ============================================================

CHECKPOINT_DIR = ROOT / "checkpoints"

GENERAL_CHECKPOINT = (
    CHECKPOINT_DIR / "text_gate_epoch13_profile_fixed.pth"
)

EXPERT_CHECKPOINTS = {
    "ultra_tiny":
        CHECKPOINT_DIR / "ultra_expert_continued_best.pth",

    "tiny":
        CHECKPOINT_DIR / "tiny_expert_continued_best.pth",

    "larger":
        CHECKPOINT_DIR / "larger_expert_continued_best.pth",

}

PRESENCE_GATE_CHECKPOINT = (
    CHECKPOINT_DIR / "presence_gate_v2_best.pth"
)

PRESENCE_GATE_RESULT_JSON = (
    ROOT
    / "runtime_data"
    / "presence_gate_v2"
    / "presence_gate_v2_result.json"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "final_testB_gate_v2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = (
    OUTPUT_DIR / "final_testB_gate_v2_results.json"
)

OUTPUT_CSV = (
    OUTPUT_DIR / "final_testB_gate_v2_samples.csv"
)


# ============================================================
# Published M2SegNet numbers from the paper table
# (for convenient display only; not used in any calculation)
# ============================================================

M2SEGNET_PAPER = {
    "cIoU": 0.7557,
    "gIoU": 0.5921,
    "Recall": 0.6751,
    "Precision": 0.7281,
    "F1": 0.6946,
    "mPr": 0.5835,
}


# ============================================================
# Generic helpers
# ============================================================

def get_ref_id(ref):
    if "ref_id" in ref:
        return int(ref["ref_id"])
    if "id" in ref:
        return int(ref["id"])
    raise RuntimeError("Reference has no ref_id / id.")


def load_seg_model(path, device):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint:\n{path}"
        )

    model = TextGuidedUNet(
        pretrained=False
    ).to(device)

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict)
        and "model_state_dict" in ckpt
        else ckpt
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    if hasattr(model, "text_encoder"):
        model.text_encoder.eval()

        if hasattr(model.text_encoder, "bert"):
            model.text_encoder.bert.eval()

    return model


def build_ref_to_index(dataset):
    return {
        get_ref_id(ref): i
        for i, ref in enumerate(dataset.refs)
    }


# ============================================================
# Text cues / frozen Router
# ============================================================

def contains_word(text, word):
    return (
        re.search(
            r"\b" + re.escape(word) + r"\b",
            str(text).lower(),
        )
        is not None
    )


def parse_text_size(text):
    text = str(text).lower()

    for word in ["tiny", "miniature", "little", "small"]:
        if contains_word(text, word):
            return "small_text"

    for word in ["medium", "moderate", "average"]:
        if contains_word(text, word):
            return "medium_text"

    for word in ["large", "big"]:
        if contains_word(text, word):
            return "large_text"

    return "unspecified"


TEXT_FAMILY = {
    "small_text": "tiny_family",
    "medium_text": "tiny_family",
    "large_text": "larger_family",
}


def route_sample(text, pred_ratio):
    text_size = parse_text_size(text)

    if text_size in TEXT_FAMILY:
        family = TEXT_FAMILY[text_size]
        route_source = "text"
    else:
        family = (
            "tiny_family"
            if pred_ratio <= ROUTER_TINY_LARGER
            else "larger_family"
        )
        route_source = "pred_area_fallback"

    if family == "larger_family":
        branch = "larger"
    else:
        branch = (
            "ultra_tiny"
            if pred_ratio <= ROUTER_ULTRA_TINY
            else "tiny"
        )

    return branch, route_source, text_size


def text_onehot(text_size):
    mapping = {
        "small_text": [1.0, 0.0, 0.0, 0.0],
        "medium_text": [0.0, 1.0, 0.0, 0.0],
        "large_text": [0.0, 0.0, 1.0, 0.0],
        "unspecified": [0.0, 0.0, 0.0, 1.0],
    }
    return mapping[text_size]


def route_onehot(branch):
    mapping = {
        "ultra_tiny": [1.0, 0.0, 0.0],
        "tiny": [0.0, 1.0, 0.0],
        "larger": [0.0, 0.0, 1.0],
    }
    return mapping[branch]


# ============================================================
# Presence Gate v2
# ============================================================

class PresenceGateV2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def extract_tensor(obj):
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (list, tuple)):
        for value in obj:
            result = extract_tensor(value)
            if result is not None:
                return result

    if isinstance(obj, dict):
        for value in obj.values():
            result = extract_tensor(value)
            if result is not None:
                return result

    return None


def find_deep_image_module(model, preferred_name=None):
    # Prefer the exact hook path used during Gate training.
    if preferred_name:
        modules = dict(model.named_modules())

        if preferred_name in modules:
            return preferred_name, modules[preferred_name]

    candidates = []

    for name, module in model.named_modules():
        lname = name.lower()

        if (
            lname == "layer4"
            or lname.endswith(".layer4")
            or lname.endswith("_layer4")
        ):
            candidates.append((name, module))

    if not candidates:
        for name, module in model.named_modules():
            lname = name.lower()

            if (
                lname == "layer3"
                or lname.endswith(".layer3")
                or lname.endswith("_layer3")
            ):
                candidates.append((name, module))

    if not candidates:
        raise RuntimeError(
            "Could not detect ResNet layer4/layer3."
        )

    candidates.sort(
        key=lambda item: (
            item[0].count("."),
            len(item[0]),
        ),
        reverse=True,
    )

    return candidates[0]


class FeatureHook:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(
            self._hook
        )

    def _hook(self, module, inputs, output):
        tensor = extract_tensor(output)

        if tensor is None:
            raise RuntimeError(
                "Presence Gate image feature hook returned no Tensor."
            )

        self.value = tensor

    def close(self):
        self.handle.remove()


def load_presence_gate(device):
    if not PRESENCE_GATE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing Presence Gate checkpoint:\n"
            f"{PRESENCE_GATE_CHECKPOINT}"
        )

    ckpt = torch.load(
        PRESENCE_GATE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    input_dim = int(ckpt["input_dim"])

    gate = PresenceGateV2(
        input_dim=input_dim
    ).to(device)

    gate.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    gate.eval()

    feature_mean = np.asarray(
        ckpt["feature_mean"],
        dtype=np.float32,
    )

    feature_std = np.asarray(
        ckpt["feature_std"],
        dtype=np.float32,
    )

    hook_name = str(
        ckpt.get(
            "image_feature_hook",
            "layer4",
        )
    )

    # Official final threshold is the Val-selected threshold.
    threshold = float(
        ckpt.get(
            "selected_threshold",
            0.91,
        )
    )

    if PRESENCE_GATE_RESULT_JSON.exists():
        with open(
            PRESENCE_GATE_RESULT_JSON,
            "r",
            encoding="utf-8",
        ) as f:
            result_json = json.load(f)

        official = result_json.get(
            "official_choice",
            {}
        )

        if official.get("threshold") is not None:
            threshold = float(
                official["threshold"]
            )

    return (
        gate,
        feature_mean,
        feature_std,
        hook_name,
        threshold,
    )


# ============================================================
# Mask metric helpers
# ============================================================

def binary_stats(pred, gt):
    pred = pred.bool()
    gt = gt.bool()

    tp = int(
        (pred & gt).sum().item()
    )

    fp = int(
        (pred & ~gt).sum().item()
    )

    fn = int(
        (~pred & gt).sum().item()
    )

    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn

    dice = (
        1.0
        if (pred_sum + gt_sum) == 0
        else (2.0 * tp) / (pred_sum + gt_sum)
    )

    iou_regular = (
        1.0
        if union == 0
        else tp / union
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "intersection": tp,
        "union": union,
        "dice": float(dice),
        "iou_regular": float(iou_regular),
        "pred_empty": bool(pred_sum == 0),
        "gt_empty": bool(gt_sum == 0),
        "pred_pixels": int(pred_sum),
        "gt_pixels": int(gt_sum),
    }


def gt_group_from_mask(gt):
    area_ratio = float(
        gt.float().mean().item()
    )

    if area_ratio < GT_ULTRA_THRESHOLD:
        group = "ultra_tiny"
    elif area_ratio < GT_TINY_THRESHOLD:
        group = "tiny"
    else:
        group = "larger"

    return group, area_ratio


# ============================================================
# PHASE 1
# General + Gate + Router over TestB
# ============================================================

@torch.no_grad()
def phase1_general_gate_router(
    general_model,
    gate,
    feature_mean,
    feature_std,
    hook_name,
    gate_threshold,
    loader,
    device,
    text_cache,
    expected_split="testB",
):
    detected_hook_name, hook_module = (
        find_deep_image_module(
            general_model,
            preferred_name=hook_name,
        )
    )

    hook = FeatureHook(
        hook_module
    )

    print(
        f"Presence image feature hook: "
        f"{detected_hook_name}"
    )

    records = {}
    route_counts = Counter()
    route_source_counts = Counter()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        ).float()

        presence = (
            batch["presence"]
            .detach()
            .cpu()
            .view(-1)
            .numpy()
        )

        pooled_text = text_cache.get_batch(
            batch=batch,
            expected_split=expected_split,
            device=device,
        )

        hook.value = None

        logits = general_model(
            images,
            pooled_text=pooled_text,
        )

        deep = hook.value

        if deep is None:
            raise RuntimeError(
                "Presence feature hook was not triggered."
            )

        if deep.ndim == 4:
            image_feature = deep.mean(
                dim=(2, 3)
            )
        elif deep.ndim == 2:
            image_feature = deep
        else:
            image_feature = deep.flatten(1)

        probs = torch.sigmoid(logits)
        flat = probs.flatten(1)

        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False)
        maxv = flat.max(dim=1).values
        q95 = torch.quantile(flat, 0.95, dim=1)
        q99 = torch.quantile(flat, 0.99, dim=1)

        eps = 1e-6
        p = probs.clamp(eps, 1.0 - eps)

        entropy = -(
            p * p.log()
            + (1.0 - p)
            * (1.0 - p).log()
        ).flatten(1).mean(dim=1)

        area010 = (
            (flat >= 0.10)
            .float()
            .mean(dim=1)
        )

        area030 = (
            (flat >= 0.30)
            .float()
            .mean(dim=1)
        )

        area050 = (
            (flat >= 0.50)
            .float()
            .mean(dim=1)
        )

        area070 = (
            (flat >= 0.70)
            .float()
            .mean(dim=1)
        )

        area090 = (
            (flat >= 0.90)
            .float()
            .mean(dim=1)
        )

        fg050 = (
            (flat >= 0.50)
            .float()
        )

        fg_count = fg050.sum(dim=1)

        fg_mean050 = torch.where(
            fg_count > 0,
            (flat * fg050)
            .sum(dim=1)
            / fg_count.clamp_min(1.0),
            torch.zeros_like(fg_count),
        )

        stats = torch.stack([
            mean,
            std,
            maxv,
            q95,
            q99,
            entropy,
            area010,
            area030,
            area050,
            area070,
            area090,
            fg_mean050,
        ], dim=1)

        texts = list(
            batch["text"]
        )

        route_features = []
        text_cues = []
        branches = []
        sources = []
        text_sizes = []

        for i, text in enumerate(texts):
            branch, source, text_size = route_sample(
                text=text,
                pred_ratio=float(
                    area050[i].item()
                ),
            )

            route_features.append(
                route_onehot(branch)
            )

            text_cues.append(
                text_onehot(text_size)
            )

            branches.append(branch)
            sources.append(source)
            text_sizes.append(text_size)

        gate_x = torch.cat([
            image_feature.float(),
            pooled_text.float(),
            stats.float(),
            torch.tensor(
                route_features,
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                text_cues,
                dtype=torch.float32,
                device=device,
            ),
        ], dim=1)

        gate_mean = torch.as_tensor(
            feature_mean,
            dtype=torch.float32,
            device=device,
        ).view(1, -1)

        gate_std = torch.as_tensor(
            feature_std,
            dtype=torch.float32,
            device=device,
        ).view(1, -1)

        gate_z = (
            gate_x - gate_mean
        ) / gate_std

        p_no_target = torch.sigmoid(
            gate(gate_z)
        )

        gate_reject = (
            p_no_target
            >= gate_threshold
        )

        general_pred = (
            probs >= GENERAL_MASK_THRESHOLD
        )

        ref_ids = (
            batch["ref_id"]
            .detach()
            .cpu()
            .view(-1)
            .tolist()
        )

        masks_cpu = (
            masks.detach().cpu()
        )

        general_pred_cpu = (
            general_pred.detach().cpu()
        )

        p_nt_cpu = (
            p_no_target
            .detach()
            .cpu()
            .view(-1)
            .numpy()
        )

        reject_cpu = (
            gate_reject
            .detach()
            .cpu()
            .view(-1)
            .numpy()
        )

        for i, ref_id in enumerate(ref_ids):
            ref_id = int(ref_id)
            is_positive = bool(
                float(presence[i]) > 0.5
            )

            gt = (
                masks_cpu[i] > 0.5
            )

            general_stats = binary_stats(
                general_pred_cpu[i],
                gt,
            )

            if is_positive:
                gt_3way, gt_area_ratio = (
                    gt_group_from_mask(gt)
                )
            else:
                gt_3way = ""
                gt_area_ratio = 0.0

            branch = branches[i]
            source = sources[i]

            route_counts[branch] += 1
            route_source_counts[source] += 1

            records[ref_id] = {
                "ref_id": ref_id,
                "dataset_index": None,

                "is_positive": is_positive,

                "gt_3way":
                    gt_3way,

                "gt_area_ratio":
                    float(gt_area_ratio),

                "route_branch":
                    branch,

                "route_source":
                    source,

                "text_size":
                    text_sizes[i],

                "general_pred_ratio":
                    float(
                        area050[i].item()
                    ),

                "p_no_target":
                    float(
                        p_nt_cpu[i]
                    ),

                "gate_rejects_as_no_target":
                    bool(
                        reject_cpu[i]
                    ),

                "general_dice":
                    general_stats["dice"],

                "general_iou":
                    general_stats["iou_regular"],

                "general_pred_empty":
                    general_stats["pred_empty"],

                "final_tp": None,
                "final_fp": None,
                "final_fn": None,
                "final_intersection": None,
                "final_union": None,
                "final_dice": None,
                "final_iou": None,
                "final_pred_empty": None,
            }

            # If Gate rejects, final prediction is zero immediately.
            if bool(reject_cpu[i]):
                zero_pred = torch.zeros_like(
                    gt,
                    dtype=torch.bool,
                )

                final_stats = binary_stats(
                    zero_pred,
                    gt,
                )

                for key, dst in [
                    ("tp", "final_tp"),
                    ("fp", "final_fp"),
                    ("fn", "final_fn"),
                    ("intersection", "final_intersection"),
                    ("union", "final_union"),
                    ("dice", "final_dice"),
                    ("iou_regular", "final_iou"),
                    ("pred_empty", "final_pred_empty"),
                ]:
                    records[ref_id][dst] = (
                        final_stats[key]
                    )

        if (
            batch_index == 1
            or batch_index % 250 == 0
            or batch_index == len(loader)
        ):
            print(
                f"  General/Gate batches: "
                f"{batch_index:04d}/{len(loader):04d}"
            )

    hook.close()

    return (
        records,
        route_counts,
        route_source_counts,
    )


# ============================================================
# PHASE 2
# Experts only for samples that survived Gate
# ============================================================

@torch.no_grad()
def run_expert_branch(
    model,
    loader,
    device,
    text_cache,
):
    out = {}

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        ).float()

        pooled_text = text_cache.get_batch(
            batch=batch,
            expected_split="testB",
            device=device,
        )

        logits = model(
            images,
            pooled_text=pooled_text,
        )

        probs = torch.sigmoid(logits)

        pred = (
            probs >= EXPERT_MASK_THRESHOLD
        )

        pred_cpu = pred.detach().cpu()
        masks_cpu = masks.detach().cpu()

        ref_ids = (
            batch["ref_id"]
            .detach()
            .cpu()
            .view(-1)
            .tolist()
        )

        for i, ref_id in enumerate(ref_ids):
            gt = (
                masks_cpu[i] > 0.5
            )

            stats = binary_stats(
                pred_cpu[i],
                gt,
            )

            out[int(ref_id)] = stats

        if (
            batch_index == 1
            or batch_index % 250 == 0
            or batch_index == len(loader)
        ):
            print(
                f"    Expert batches: "
                f"{batch_index:04d}/{len(loader):04d}"
            )

    return out


# ============================================================
# Summaries
# ============================================================

def mean_or_zero(values):
    return (
        float(np.mean(values))
        if values
        else 0.0
    )


def summarize_system(records):
    all_records = list(records.values())

    positive = [
        r for r in all_records
        if r["is_positive"]
    ]

    no_target = [
        r for r in all_records
        if not r["is_positive"]
    ]

    positive_dice = [
        r["final_dice"]
        for r in positive
    ]

    all_dice = [
        r["final_dice"]
        for r in all_records
    ]

    no_target_acc = [
        1.0
        if bool(r["final_pred_empty"])
        else 0.0
        for r in no_target
    ]

    return {
        "N": len(all_records),
        "positive_N": len(positive),
        "no_target_N": len(no_target),

        "positive_mean_dice":
            mean_or_zero(positive_dice),

        "positive_dice_ge_0_8":
            mean_or_zero([
                1.0
                if x >= 0.8
                else 0.0
                for x in positive_dice
            ]),

        "positive_dice_lt_0_5":
            mean_or_zero([
                1.0
                if x < 0.5
                else 0.0
                for x in positive_dice
            ]),

        "no_target_accuracy":
            mean_or_zero(no_target_acc),

        "full_all_sample_dice":
            mean_or_zero(all_dice),
    }


def summarize_gate(records):
    all_records = list(records.values())

    positive = [
        r for r in all_records
        if r["is_positive"]
    ]

    no_target = [
        r for r in all_records
        if not r["is_positive"]
    ]

    pos_survival = mean_or_zero([
        0.0
        if r["gate_rejects_as_no_target"]
        else 1.0
        for r in positive
    ])

    nt_recall = mean_or_zero([
        1.0
        if r["gate_rejects_as_no_target"]
        else 0.0
        for r in no_target
    ])

    return {
        "positive_survival":
            pos_survival,
        "false_reject_positive_rate":
            1.0 - pos_survival,
        "no_target_recall":
            nt_recall,
    }


def summarize_by_true_size(records):
    out = {}

    for group in GROUPS:
        rows = [
            r for r in records.values()
            if (
                r["is_positive"]
                and r["gt_3way"] == group
            )
        ]

        out[group] = {
            "N":
                len(rows),

            "mean_dice":
                mean_or_zero([
                    r["final_dice"]
                    for r in rows
                ]),
        }

    return out


def summarize_by_route(records):
    out = {}

    for branch in GROUPS:
        pos = [
            r for r in records.values()
            if (
                r["is_positive"]
                and r["route_branch"] == branch
            )
        ]

        nt = [
            r for r in records.values()
            if (
                (not r["is_positive"])
                and r["route_branch"] == branch
            )
        ]

        out[branch] = {
            "positive_N":
                len(pos),

            "positive_mean_dice":
                mean_or_zero([
                    r["final_dice"]
                    for r in pos
                ]),

            "no_target_N":
                len(nt),

            "no_target_accuracy":
                mean_or_zero([
                    1.0
                    if r["final_pred_empty"]
                    else 0.0
                    for r in nt
                ]),
        }

    return out


# ============================================================
# M2SegNet-style metrics
# ============================================================

def compute_m2segnet_style_metrics(records):
    rows = list(records.values())

    # cIoU: cumulative/global intersection over cumulative/global union.
    total_intersection = sum(
        int(r["final_intersection"])
        for r in rows
    )

    total_union = sum(
        int(r["final_union"])
        for r in rows
    )

    ciou = (
        total_intersection / total_union
        if total_union > 0
        else 1.0
    )

    # gIoU: positive samples use ordinary IoU;
    # no-target samples are 1 if predicted empty, otherwise 0.
    generalized_iou = []

    for r in rows:
        if r["is_positive"]:
            generalized_iou.append(
                float(r["final_iou"])
            )
        else:
            generalized_iou.append(
                1.0
                if bool(r["final_pred_empty"])
                else 0.0
            )

    giou = float(
        np.mean(generalized_iou)
    )

    # Global pixel TP / FP / FN.
    tp = sum(
        int(r["final_tp"])
        for r in rows
    )

    fp = sum(
        int(r["final_fp"])
        for r in rows
    )

    fn = sum(
        int(r["final_fn"])
        for r in rows
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 1.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Pr@X: proportion of samples whose generalized per-image IoU
    # reaches threshold X. mPr = mean Pr@0.1 ... Pr@0.9.
    pr = {}

    for x in range(1, 10):
        threshold = x / 10.0

        value = float(
            np.mean([
                iou >= threshold
                for iou
                in generalized_iou
            ])
        )

        pr[f"Pr@{threshold:.1f}"] = value

    mpr = float(
        np.mean(
            list(pr.values())
        )
    )

    return {
        "cIoU": float(ciou),
        "gIoU": float(giou),
        "Recall": float(recall),
        "Precision": float(precision),
        "F1": float(f1),
        **pr,
        "mPr": float(mpr),
        "pixel_counts": {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "intersection_sum":
                int(total_intersection),
            "union_sum":
                int(total_union),
        },
    }


# ============================================================
# Save CSV
# ============================================================

def save_sample_csv(records):
    fields = [
        "ref_id",
        "is_positive",
        "gt_3way",
        "gt_area_ratio",
        "route_branch",
        "route_source",
        "text_size",
        "general_pred_ratio",
        "p_no_target",
        "gate_rejects_as_no_target",
        "general_dice",
        "general_iou",
        "general_pred_empty",
        "final_dice",
        "final_iou",
        "final_pred_empty",
        "final_tp",
        "final_fp",
        "final_fn",
        "final_intersection",
        "final_union",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in sorted(
            records.values(),
            key=lambda x: x["ref_id"],
        ):
            writer.writerow({
                key: row.get(key, "")
                for key in fields
            })


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("=" * 112)
    print(
        "STEP 43: FINAL FROZEN TESTB — PRESENCE GATE v2 + NEW 3-EXPERT SYSTEM"
    )
    print("=" * 112)

    print()
    print("NO training.")
    print("NO threshold tuning.")
    print("NO checkpoint selection.")
    print("This run uses the Train/Val-frozen configuration.")

    for path in [
        GENERAL_CHECKPOINT,
        PRESENCE_GATE_CHECKPOINT,
        *EXPERT_CHECKPOINTS.values(),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required checkpoint:\n{path}"
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    (
        gate,
        gate_mean,
        gate_std,
        hook_name,
        gate_threshold,
    ) = load_presence_gate(device)

    print()
    print("=" * 112)
    print("FROZEN CONFIGURATION")
    print("=" * 112)
    print(
        f"Presence Gate threshold : "
        f"{gate_threshold:.4f}"
    )
    print(
        f"Presence image hook      : "
        f"{hook_name}"
    )
    print(
        f"Router Ultra/Tiny        : "
        f"{ROUTER_ULTRA_TINY:.6f}"
    )
    print(
        f"Router Tiny/Larger       : "
        f"{ROUTER_TINY_LARGER:.6f}"
    )
    print(
        f"Expert mask threshold    : "
        f"{EXPERT_MASK_THRESHOLD:.3f}"
    )

    print()
    print("Expert checkpoints:")
    for group, path in EXPERT_CHECKPOINTS.items():
        print(
            f"  {group:12s}: {path.name}"
        )

    text_cache = FormalTextCache(ROOT)

    test_dataset = RefELDataset(
        root_dir=ROOT,
        split="testB",
        image_size=IMAGE_SIZE,
        random_text=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    ref_to_index = build_ref_to_index(
        test_dataset
    )

    print()
    print(
        f"TestB total refs: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Phase 1
    # --------------------------------------------------------

    print()
    print("=" * 112)
    print("PHASE 1: GENERAL + PRESENCE GATE v2 + FROZEN ROUTER")
    print("=" * 112)

    general_model = load_seg_model(
        GENERAL_CHECKPOINT,
        device,
    )

    (
        records,
        route_counts,
        route_source_counts,
    ) = phase1_general_gate_router(
        general_model=general_model,
        gate=gate,
        feature_mean=gate_mean,
        feature_std=gate_std,
        hook_name=hook_name,
        gate_threshold=gate_threshold,
        loader=test_loader,
        device=device,
        text_cache=text_cache,
    )

    del general_model
    del gate

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ref_id in records:
        records[ref_id]["dataset_index"] = (
            ref_to_index[ref_id]
        )

    gate_stats = summarize_gate(
        records
    )

    print()
    print(
        f"Gate Positive Survival : "
        f"{gate_stats['positive_survival']:.2%}"
    )
    print(
        f"Gate No-target Recall  : "
        f"{gate_stats['no_target_recall']:.2%}"
    )

    # --------------------------------------------------------
    # Phase 2
    # --------------------------------------------------------

    print()
    print("=" * 112)
    print("PHASE 2: NEW BEST EXPERTS FOR GATE-SURVIVING SAMPLES")
    print("=" * 112)

    for branch in GROUPS:
        ref_ids = [
            ref_id
            for ref_id, row
            in records.items()
            if (
                row["route_branch"] == branch
                and not row[
                    "gate_rejects_as_no_target"
                ]
            )
        ]

        indices = [
            records[ref_id][
                "dataset_index"
            ]
            for ref_id in ref_ids
        ]

        print()
        print(
            f"{branch}: surviving N={len(indices)}"
        )
        print(
            f"checkpoint: "
            f"{EXPERT_CHECKPOINTS[branch].name}"
        )

        if not indices:
            continue

        loader = DataLoader(
            Subset(
                test_dataset,
                indices,
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        model = load_seg_model(
            EXPERT_CHECKPOINTS[branch],
            device,
        )

        branch_output = run_expert_branch(
            model=model,
            loader=loader,
            device=device,
            text_cache=text_cache,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for ref_id in ref_ids:
            stats = branch_output[ref_id]

            mapping = {
                "tp": "final_tp",
                "fp": "final_fp",
                "fn": "final_fn",
                "intersection":
                    "final_intersection",
                "union":
                    "final_union",
                "dice":
                    "final_dice",
                "iou_regular":
                    "final_iou",
                "pred_empty":
                    "final_pred_empty",
            }

            for src, dst in mapping.items():
                records[ref_id][dst] = (
                    stats[src]
                )

    missing = [
        ref_id
        for ref_id, row
        in records.items()
        if row["final_dice"] is None
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} TestB samples "
            f"have no final prediction. "
            f"First: {missing[:10]}"
        )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    system = summarize_system(
        records
    )

    by_true_size = summarize_by_true_size(
        records
    )

    by_route = summarize_by_route(
        records
    )

    paper_metrics = (
        compute_m2segnet_style_metrics(
            records
        )
    )

    print()
    print("=" * 112)
    print("FINAL TESTB: GATE v2 + NEW 3-EXPERT SYSTEM")
    print("=" * 112)

    print()
    print(
        f"TestB N                     : "
        f"{system['N']}"
    )
    print(
        f"Positive N                  : "
        f"{system['positive_N']}"
    )
    print(
        f"No-target N                 : "
        f"{system['no_target_N']}"
    )

    print()
    print("POSITIVE SEGMENTATION")
    print(
        f"Positive Mean Dice          : "
        f"{system['positive_mean_dice']:.4f}"
    )
    print(
        f"Dice >= 0.8                 : "
        f"{system['positive_dice_ge_0_8']:.2%}"
    )
    print(
        f"Dice < 0.5                  : "
        f"{system['positive_dice_lt_0_5']:.2%}"
    )

    print()
    print("NO-TARGET")
    print(
        f"No-target Accuracy          : "
        f"{system['no_target_accuracy']:.4f}"
    )
    print(
        f"Gate Positive Survival      : "
        f"{gate_stats['positive_survival']:.2%}"
    )
    print(
        f"Gate No-target Recall       : "
        f"{gate_stats['no_target_recall']:.2%}"
    )

    print()
    print("FULL TESTB")
    print(
        f"Full TestB All-sample Dice  : "
        f"{system['full_all_sample_dice']:.4f}"
    )

    print()
    print("=" * 112)
    print("FINAL TESTB BY TRUE SIZE")
    print("=" * 112)

    for group in GROUPS:
        item = by_true_size[group]

        print(
            f"{group:12s} | "
            f"N={item['N']:4d} | "
            f"Dice={item['mean_dice']:.4f}"
        )

    print()
    print("=" * 112)
    print("FINAL TESTB BY ROUTED BRANCH")
    print("=" * 112)

    for group in GROUPS:
        item = by_route[group]

        print()
        print(
            f"{group:12s} POS | "
            f"N={item['positive_N']:4d} | "
            f"Dice={item['positive_mean_dice']:.4f}"
        )

        print(
            f"{group:12s} NT  | "
            f"N={item['no_target_N']:4d} | "
            f"Acc={item['no_target_accuracy']:.4f}"
        )

    print()
    print("=" * 112)
    print("M2SEGNET-STYLE METRICS")
    print("=" * 112)

    for key in [
        "cIoU",
        "gIoU",
        "Recall",
        "Precision",
        "F1",
        "mPr",
    ]:
        ours = paper_metrics[key]
        m2 = M2SEGNET_PAPER[key]

        print(
            f"{key:10s} | "
            f"Ours={ours * 100:6.2f}% | "
            f"M2SegNet={m2 * 100:6.2f}% | "
            f"Delta={(ours - m2) * 100:+6.2f} pp"
        )

    print()
    print("Pr@X:")
    for x in range(1, 10):
        key = f"Pr@{x/10:.1f}"

        print(
            f"  {key:6s}: "
            f"{paper_metrics[key] * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_sample_csv(
        records
    )

    output = {
        "step": 43,
        "testB_used": True,
        "frozen_after_val": True,

        "configuration": {
            "general_checkpoint":
                str(GENERAL_CHECKPOINT),

            "presence_gate_checkpoint":
                str(PRESENCE_GATE_CHECKPOINT),

            "presence_gate_threshold":
                float(gate_threshold),

            "presence_image_hook":
                hook_name,

            "router_ultra_tiny":
                ROUTER_ULTRA_TINY,

            "router_tiny_larger":
                ROUTER_TINY_LARGER,

            "expert_mask_threshold":
                EXPERT_MASK_THRESHOLD,

            "expert_checkpoints": {
                k: str(v)
                for k, v
                in EXPERT_CHECKPOINTS.items()
            },
        },

        "system": system,
        "gate": gate_stats,
        "by_true_size": by_true_size,
        "by_route": by_route,

        "m2segnet_style_metrics": {
            k: v
            for k, v
            in paper_metrics.items()
            if k != "pixel_counts"
        },

        "pixel_counts":
            paper_metrics["pixel_counts"],

        "m2segnet_paper_reference_numbers":
            M2SEGNET_PAPER,

        "outputs": {
            "summary_json":
                str(OUTPUT_JSON),

            "sample_csv":
                str(OUTPUT_CSV),
        },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print()
    print("=" * 112)
    print("STEP 43 FINISHED — FINAL FROZEN TESTB")
    print("=" * 112)

    print()
    print(
        f"Summary JSON:\n{OUTPUT_JSON}"
    )
    print()
    print(
        f"Sample CSV:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
