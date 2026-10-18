import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import RefELDataset
from model_text import TextGuidedUNet
from train_text import FormalTextCache


# ============================================================
# STEP 42
# Presence Gate v2 — FEATURE-LEVEL, ROUTE-AWARE
#
# Stronger than Step 41:
#   - frozen General deep IMAGE feature (auto-hooked from ResNet layer4)
#   - frozen BERT pooled TEXT feature
#   - full General probability statistics
#   - frozen Router one-hot
#   - text size one-hot
#
# The segmentation Experts are NOT retrained.
# TestB is NOT used.
#
# Gate logic:
#   if P(no-target) >= selected threshold:
#       output all-zero mask
#   else:
#       keep the current Step-40 3-Expert result
#
# Threshold is chosen on Val for the highest Full-Val Dice,
# subject to positive survival >= 98.5%.
# For transparency the script also reports best solutions under
# 99.5%, 99.0%, and 98.5% survival constraints.
# ============================================================

ROOT = Path(__file__).resolve().parent

SEED = 20260915
IMAGE_SIZE = 384
EXTRACT_BATCH_SIZE = 4
NUM_WORKERS = 0

GATE_BATCH_SIZE = 512
MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 15
LR = 8e-4
WEIGHT_DECAY = 2e-4
NO_TARGET_POS_WEIGHT_CAP = 4.0

SURVIVAL_CONSTRAINTS = [0.995, 0.990, 0.985]

THRESHOLD_GRID = np.unique(
    np.concatenate([
        np.arange(0.50, 0.951, 0.01),
        np.arange(0.955, 0.991, 0.005),
        np.arange(0.992, 0.9991, 0.001),
        np.asarray([0.9992, 0.9994, 0.9996, 0.9998]),
    ])
).astype(np.float64)

CHECKPOINT_DIR = ROOT / "checkpoints"
GENERAL_CHECKPOINT = (
    CHECKPOINT_DIR / "text_gate_epoch13_profile_fixed.pth"
)

OUTPUT_DIR = ROOT / "runtime_data" / "presence_gate_v2"
STEP40_SAMPLE_CSV = (
    OUTPUT_DIR / "new_best_experts_full_val_samples.csv"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FEATURE_CACHE = OUTPUT_DIR / "train_features_v2.npz"
VAL_FEATURE_CACHE = OUTPUT_DIR / "val_features_v2.npz"

BEST_CHECKPOINT = CHECKPOINT_DIR / "presence_gate_v2_best.pth"
HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
VAL_PREDICTIONS_CSV = OUTPUT_DIR / "val_predictions.csv"
RESULT_JSON = OUTPUT_DIR / "presence_gate_v2_result.json"


# Frozen router settings already used by the system.
ROUTER_ULTRA_TINY = 0.002550
ROUTER_TINY_LARGER = 0.016290

GROUPS = ["ultra_tiny", "tiny", "larger"]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Router / text cues
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
    else:
        family = (
            "tiny_family"
            if pred_ratio <= ROUTER_TINY_LARGER
            else "larger_family"
        )

    if family == "larger_family":
        branch = "larger"
    else:
        branch = (
            "ultra_tiny"
            if pred_ratio <= ROUTER_ULTRA_TINY
            else "tiny"
        )

    return branch, text_size


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
# General model + automatic deep-image feature hook
# ============================================================

def load_general(device):
    if not GENERAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing General checkpoint:\n{GENERAL_CHECKPOINT}"
        )

    model = TextGuidedUNet(pretrained=False).to(device)

    ckpt = torch.load(
        GENERAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict)
        and "model_state_dict" in ckpt
        else ckpt
    )

    model.load_state_dict(state, strict=True)
    model.eval()
    model.text_encoder.eval()
    model.text_encoder.bert.eval()

    return model


def find_deep_image_module(model):
    """
    Prefer a ResNet layer4 module. This keeps the script independent
    of the exact wrapper name (encoder.layer4, backbone.layer4, etc.).
    """
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
        # Defensive fallback: layer3 is still a deep image feature.
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
            "Could not auto-detect ResNet layer4/layer3 in TextGuidedUNet.\n"
            "Run: python -c \"from model_text import TextGuidedUNet; "
            "m=TextGuidedUNet(pretrained=False); "
            "[print(n) for n,_ in m.named_modules()]\"\n"
            "and send the output around the encoder names."
        )

    # Prefer the deepest / most-qualified name.
    candidates.sort(
        key=lambda item: (
            item[0].count("."),
            len(item[0]),
        ),
        reverse=True,
    )

    return candidates[0]


def extract_tensor(obj):
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (list, tuple)):
        for value in obj:
            tensor = extract_tensor(value)
            if tensor is not None:
                return tensor

    if isinstance(obj, dict):
        for value in obj.values():
            tensor = extract_tensor(value)
            if tensor is not None:
                return tensor

    return None


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
                "Deep image hook did not return a Tensor."
            )

        self.value = tensor

    def close(self):
        self.handle.remove()


# ============================================================
# Feature extraction
# ============================================================

STAT_NAMES = [
    "prob_mean",
    "prob_std",
    "prob_max",
    "prob_q95",
    "prob_q99",
    "entropy_mean",
    "area_ge_010",
    "area_ge_030",
    "area_ge_050",
    "area_ge_070",
    "area_ge_090",
    "fg_mean_ge_050",
]


@torch.no_grad()
def extract_split_features(split, cache_path, device):
    if cache_path.exists():
        print()
        print(f"Loading cached {split} v2 features:")
        print(cache_path)

        data = np.load(
            cache_path,
            allow_pickle=False,
        )

        return {
            "ref_ids":
                data["ref_ids"].astype(np.int64),
            "image_features":
                data["image_features"].astype(np.float32),
            "text_features":
                data["text_features"].astype(np.float32),
            "stats":
                data["stats"].astype(np.float32),
            "route_features":
                data["route_features"].astype(np.float32),
            "text_cues":
                data["text_cues"].astype(np.float32),
            "labels_no_target":
                data["labels_no_target"].astype(np.float32),
            "hook_name":
                str(data["hook_name"].item()),
        }

    print()
    print("=" * 108)
    print(f"EXTRACT FEATURE-LEVEL PRESENCE DATA: {split.upper()}")
    print("=" * 108)

    dataset = RefELDataset(
        root_dir=ROOT,
        split=split,
        image_size=IMAGE_SIZE,
        random_text=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=EXTRACT_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    text_cache = FormalTextCache(ROOT)
    model = load_general(device)

    hook_name, hook_module = find_deep_image_module(model)

    print()
    print(f"Deep image feature module: {hook_name}")

    hook = FeatureHook(hook_module)

    ref_id_list = []
    image_feature_list = []
    text_feature_list = []
    stat_list = []
    route_feature_list = []
    text_cue_list = []
    label_list = []

    start = time.perf_counter()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        pooled_text = text_cache.get_batch(
            batch=batch,
            expected_split=split,
            device=device,
        )

        hook.value = None

        logits = model(
            images,
            pooled_text=pooled_text,
        )

        deep = hook.value

        if deep is None:
            raise RuntimeError(
                "Image feature hook was not triggered."
            )

        # Deep ResNet feature -> Global Average Pool.
        if deep.ndim == 4:
            image_feature = deep.mean(
                dim=(2, 3)
            )
        elif deep.ndim == 2:
            image_feature = deep
        else:
            image_feature = deep.flatten(
                1
            )

        probs = torch.sigmoid(logits)
        flat = probs.flatten(1)

        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False)
        maxv = flat.max(dim=1).values

        q95 = torch.quantile(
            flat,
            0.95,
            dim=1,
        )
        q99 = torch.quantile(
            flat,
            0.99,
            dim=1,
        )

        eps = 1e-6
        p = probs.clamp(eps, 1.0 - eps)
        entropy = -(
            p * p.log()
            + (1.0 - p) * (1.0 - p).log()
        ).flatten(1).mean(dim=1)

        area010 = (flat >= 0.10).float().mean(dim=1)
        area030 = (flat >= 0.30).float().mean(dim=1)
        area050 = (flat >= 0.50).float().mean(dim=1)
        area070 = (flat >= 0.70).float().mean(dim=1)
        area090 = (flat >= 0.90).float().mean(dim=1)

        fg050 = (flat >= 0.50).float()
        fg_count = fg050.sum(dim=1)
        fg_mean050 = torch.where(
            fg_count > 0,
            (flat * fg050).sum(dim=1)
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

        texts = list(batch["text"])

        routes = []
        text_cues = []

        for i, text in enumerate(texts):
            branch, text_size = route_sample(
                text=text,
                pred_ratio=float(
                    area050[i].item()
                ),
            )

            routes.append(
                route_onehot(branch)
            )

            text_cues.append(
                text_onehot(text_size)
            )

        presence = (
            batch["presence"]
            .detach()
            .cpu()
            .view(-1)
            .numpy()
        )

        labels_nt = (
            presence <= 0.5
        ).astype(np.float32)

        ref_ids = (
            batch["ref_id"]
            .detach()
            .cpu()
            .view(-1)
            .numpy()
            .astype(np.int64)
        )

        ref_id_list.append(ref_ids)

        image_feature_list.append(
            image_feature
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        text_feature_list.append(
            pooled_text
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        stat_list.append(
            stats
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        route_feature_list.append(
            np.asarray(
                routes,
                dtype=np.float32,
            )
        )

        text_cue_list.append(
            np.asarray(
                text_cues,
                dtype=np.float32,
            )
        )

        label_list.append(labels_nt)

        if (
            batch_index == 1
            or batch_index % 500 == 0
            or batch_index == len(loader)
        ):
            elapsed = (
                time.perf_counter()
                - start
            ) / 60.0

            print(
                f"{split}: "
                f"{batch_index:05d}/{len(loader):05d}"
                f" | {elapsed:.1f} min"
            )

    hook.close()
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "ref_ids":
            np.concatenate(ref_id_list).astype(np.int64),
        "image_features":
            np.concatenate(image_feature_list).astype(np.float32),
        "text_features":
            np.concatenate(text_feature_list).astype(np.float32),
        "stats":
            np.concatenate(stat_list).astype(np.float32),
        "route_features":
            np.concatenate(route_feature_list).astype(np.float32),
        "text_cues":
            np.concatenate(text_cue_list).astype(np.float32),
        "labels_no_target":
            np.concatenate(label_list).astype(np.float32),
        "hook_name":
            hook_name,
    }

    # float16 saves a lot of disk without affecting the later MLP materially.
    np.savez_compressed(
        cache_path,
        ref_ids=result["ref_ids"],
        image_features=result["image_features"].astype(np.float16),
        text_features=result["text_features"].astype(np.float16),
        stats=result["stats"].astype(np.float32),
        route_features=result["route_features"].astype(np.float32),
        text_cues=result["text_cues"].astype(np.float32),
        labels_no_target=result["labels_no_target"],
        hook_name=np.asarray(result["hook_name"]),
    )

    print()
    print(f"Saved:\n{cache_path}")
    print(
        "Shapes: "
        f"image={result['image_features'].shape}, "
        f"text={result['text_features'].shape}, "
        f"stats={result['stats'].shape}"
    )
    print(
        f"No-target: "
        f"{int(result['labels_no_target'].sum())}/"
        f"{len(result['labels_no_target'])}"
    )

    return result


# ============================================================
# Step-40 system results
# ============================================================

def parse_bool(value):
    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def load_step40_rows():
    if not STEP40_SAMPLE_CSV.exists():
        raise FileNotFoundError(
            "Step 40 sample result is missing:\n"
            f"{STEP40_SAMPLE_CSV}"
        )

    rows = {}

    with open(
        STEP40_SAMPLE_CSV,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            ref_id = int(row["ref_id"])

            rows[ref_id] = {
                "is_positive":
                    parse_bool(row["is_positive"]),
                "expert_hard_dice":
                    float(row["expert_hard_dice"]),
                "route_branch":
                    row["route_branch"],
            }

    return rows


# ============================================================
# Feature normalization / concatenation
# ============================================================

def build_feature_matrix(data):
    return np.concatenate([
        data["image_features"],
        data["text_features"],
        data["stats"],
        data["route_features"],
        data["text_cues"],
    ], axis=1).astype(np.float32)


# ============================================================
# Gate network
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


# ============================================================
# Full-system gate evaluation
# ============================================================

def evaluate_threshold(
    p_no_target,
    labels_nt,
    baseline_dice,
    threshold,
):
    labels_nt_bool = labels_nt >= 0.5
    labels_pos_bool = ~labels_nt_bool

    reject = (
        p_no_target >= threshold
    )

    final_dice = baseline_dice.copy()

    # Correctly rejected no-target -> perfect zero-mask Dice.
    final_dice[
        reject & labels_nt_bool
    ] = 1.0

    # Incorrectly rejected positive -> zero-mask vs positive GT.
    final_dice[
        reject & labels_pos_bool
    ] = 0.0

    positive_survival = float(
        np.mean(
            ~reject[labels_pos_bool]
        )
    )

    return {
        "threshold":
            float(threshold),
        "positive_survival_recall":
            positive_survival,
        "false_reject_positive_rate":
            1.0 - positive_survival,
        "positive_dice":
            float(
                final_dice[
                    labels_pos_bool
                ].mean()
            ),
        "no_target_accuracy":
            float(
                final_dice[
                    labels_nt_bool
                ].mean()
            ),
        "gate_no_target_recall":
            float(
                reject[
                    labels_nt_bool
                ].mean()
            ),
        "full_val_dice":
            float(
                final_dice.mean()
            ),
    }


def best_for_constraint(
    p_no_target,
    labels_nt,
    baseline_dice,
    min_survival,
):
    rows = [
        evaluate_threshold(
            p_no_target,
            labels_nt,
            baseline_dice,
            float(t),
        )
        for t in THRESHOLD_GRID
    ]

    feasible = [
        row
        for row in rows
        if (
            row["positive_survival_recall"]
            >= min_survival
        )
    ]

    if not feasible:
        return None

    return max(
        feasible,
        key=lambda row: (
            row["full_val_dice"],
            row["no_target_accuracy"],
            row["positive_survival_recall"],
        )
    )


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    print()
    print("=" * 112)
    print(
        "STEP 42: FEATURE-LEVEL ROUTE-AWARE PRESENCE GATE v2"
    )
    print("=" * 112)
    print()
    print("No Expert retraining.")
    print("No TestB.")
    print(
        "Inputs: General deep image feature + BERT text feature "
        "+ General probability stats + route/text cues."
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

    train_data = extract_split_features(
        "train",
        TRAIN_FEATURE_CACHE,
        device,
    )

    val_data = extract_split_features(
        "val",
        VAL_FEATURE_CACHE,
        device,
    )

    train_x = build_feature_matrix(
        train_data
    )

    val_x = build_feature_matrix(
        val_data
    )

    train_y = train_data[
        "labels_no_target"
    ].astype(np.float32)

    val_y = val_data[
        "labels_no_target"
    ].astype(np.float32)

    # --------------------------------------------------------
    # Align Step-40 final 3-Expert Dice with Val feature rows
    # --------------------------------------------------------

    step40_rows = load_step40_rows()

    missing = [
        int(ref_id)
        for ref_id in val_data["ref_ids"]
        if int(ref_id) not in step40_rows
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} refs missing from Step-40 CSV. "
            f"First: {missing[:10]}"
        )

    baseline_dice = np.asarray([
        step40_rows[
            int(ref_id)
        ][
            "expert_hard_dice"
        ]
        for ref_id
        in val_data["ref_ids"]
    ], dtype=np.float64)

    labels_nt_bool = val_y >= 0.5
    labels_pos_bool = ~labels_nt_bool

    baseline_positive = float(
        baseline_dice[
            labels_pos_bool
        ].mean()
    )

    baseline_nt = float(
        baseline_dice[
            labels_nt_bool
        ].mean()
    )

    baseline_full = float(
        baseline_dice.mean()
    )

    print()
    print("=" * 112)
    print("STEP-40 BASELINE")
    print("=" * 112)
    print(
        f"Positive Dice      : {baseline_positive:.4f}"
    )
    print(
        f"No-target Accuracy : {baseline_nt:.4f}"
    )
    print(
        f"Full-Val Dice      : {baseline_full:.4f}"
    )

    # --------------------------------------------------------
    # Normalize with Train stats only.
    # --------------------------------------------------------

    mean = train_x.mean(
        axis=0
    ).astype(np.float32)

    std = train_x.std(
        axis=0
    ).astype(np.float32)

    std[
        std < 1e-6
    ] = 1.0

    train_z = (
        (train_x - mean) / std
    ).astype(np.float32)

    val_z = (
        (val_x - mean) / std
    ).astype(np.float32)

    print()
    print(
        f"Gate input dimension: {train_z.shape[1]}"
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_z),
        torch.from_numpy(train_y),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=GATE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_tensor = torch.from_numpy(
        val_z
    ).to(device)

    gate = PresenceGateV2(
        input_dim=train_z.shape[1]
    ).to(device)

    n_nt = float(
        train_y.sum()
    )
    n_pos = float(
        len(train_y) - n_nt
    )

    pos_weight = min(
        n_pos / max(n_nt, 1.0),
        NO_TARGET_POS_WEIGHT_CAP,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        )
    )

    print(
        f"Train class pos_weight(no-target): "
        f"{pos_weight:.3f}"
    )

    # --------------------------------------------------------
    # Train, selecting on Full-Val Dice under 98.5% survival.
    # --------------------------------------------------------

    global_min_survival = min(
        SURVIVAL_CONSTRAINTS
    )

    best_full = baseline_full
    best_epoch = 0
    best_row = None
    best_state = None

    no_improve = 0
    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        gate.train()

        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(
                device,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = gate(x_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                gate.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += float(
                loss.item()
            )
            n_batches += 1

        gate.eval()

        with torch.no_grad():
            p_nt = torch.sigmoid(
                gate(val_tensor)
            ).cpu().numpy()

        selected = best_for_constraint(
            p_nt,
            val_y,
            baseline_dice,
            min_survival=
                global_min_survival,
        )

        if selected is None:
            raise RuntimeError(
                "No threshold satisfies the 98.5% positive "
                "survival constraint."
            )

        current_full = float(
            selected["full_val_dice"]
        )

        scheduler.step(
            current_full
        )

        history.append({
            "epoch":
                epoch,
            "train_loss":
                total_loss / max(n_batches, 1),
            "threshold":
                selected["threshold"],
            "positive_survival":
                selected[
                    "positive_survival_recall"
                ],
            "positive_dice":
                selected["positive_dice"],
            "no_target_accuracy":
                selected[
                    "no_target_accuracy"
                ],
            "gate_no_target_recall":
                selected[
                    "gate_no_target_recall"
                ],
            "full_val_dice":
                current_full,
            "lr":
                optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={history[-1]['train_loss']:.4f} | "
            f"Thr={selected['threshold']:.4f} | "
            f"PosSurvive={selected['positive_survival_recall']:.2%} | "
            f"PosDice={selected['positive_dice']:.4f} | "
            f"NTAcc={selected['no_target_accuracy']:.4f} | "
            f"NTRecall={selected['gate_no_target_recall']:.2%} | "
            f"Full={selected['full_val_dice']:.4f}"
        )

        if (
            current_full
            >
            best_full + 1e-6
        ):
            best_full = current_full
            best_epoch = epoch
            best_row = dict(selected)

            best_state = {
                key:
                    value.detach().cpu().clone()
                for key, value
                in gate.state_dict().items()
            }

            no_improve = 0

            torch.save(
                {
                    "stage":
                        "presence_gate_v2_feature_level",
                    "epoch":
                        epoch,
                    "model_state_dict":
                        best_state,
                    "feature_mean":
                        mean,
                    "feature_std":
                        std,
                    "image_feature_hook":
                        train_data["hook_name"],
                    "input_dim":
                        int(train_z.shape[1]),
                    "selected_threshold":
                        float(
                            selected["threshold"]
                        ),
                    "selection":
                        best_row,
                    "survival_constraint":
                        global_min_survival,
                    "testB_used":
                        False,
                },
                BEST_CHECKPOINT,
            )

            print(
                f"  NEW v2 GATE BEST: "
                f"Full-Val={best_full:.4f}"
            )
        else:
            no_improve += 1

        if (
            no_improve
            >= EARLY_STOPPING_PATIENCE
        ):
            print()
            print(
                "Early stopping: "
                f"{EARLY_STOPPING_PATIENCE} "
                "epochs without improving Full-Val Dice."
            )
            break

    # --------------------------------------------------------
    # Load best and report several survival policies.
    # --------------------------------------------------------

    if best_state is None:
        print()
        print(
            "Presence Gate v2 did not beat the no-gate baseline."
        )

        best_epoch = 0
        p_nt_best = np.zeros(
            len(val_y),
            dtype=np.float32,
        )

    else:
        gate.load_state_dict(
            best_state
        )

        gate.eval()

        with torch.no_grad():
            p_nt_best = torch.sigmoid(
                gate(val_tensor)
            ).cpu().numpy()

    policy_results = {}

    print()
    print("=" * 112)
    print("BEST v2 GATE — SURVIVAL POLICY COMPARISON")
    print("=" * 112)

    for survival in SURVIVAL_CONSTRAINTS:
        row = best_for_constraint(
            p_nt_best,
            val_y,
            baseline_dice,
            min_survival=survival,
        )

        if row is None:
            continue

        policy_results[
            f"{survival:.3f}"
        ] = row

        print()
        print(
            f"Positive survival >= {survival:.1%}"
        )
        print(
            f"  threshold          : "
            f"{row['threshold']:.4f}"
        )
        print(
            f"  positive survival  : "
            f"{row['positive_survival_recall']:.2%}"
        )
        print(
            f"  positive Dice      : "
            f"{row['positive_dice']:.4f}"
        )
        print(
            f"  no-target Accuracy : "
            f"{row['no_target_accuracy']:.4f}"
        )
        print(
            f"  gate NT Recall     : "
            f"{row['gate_no_target_recall']:.2%}"
        )
        print(
            f"  FULL-VAL Dice      : "
            f"{row['full_val_dice']:.4f}"
        )

    # Official choice is highest Full-Val among the listed policies.
    valid_rows = list(
        policy_results.values()
    )

    if valid_rows:
        official = max(
            valid_rows,
            key=lambda row: (
                row["full_val_dice"],
                row["positive_survival_recall"],
            )
        )
    else:
        official = {
            "threshold": None,
            "positive_survival_recall": 1.0,
            "positive_dice": baseline_positive,
            "no_target_accuracy": baseline_nt,
            "gate_no_target_recall": 0.0,
            "false_reject_positive_rate": 0.0,
            "full_val_dice": baseline_full,
        }

    # --------------------------------------------------------
    # Save sample predictions.
    # --------------------------------------------------------

    threshold = official["threshold"]

    if threshold is None:
        reject = np.zeros(
            len(val_y),
            dtype=bool,
        )
    else:
        reject = (
            p_nt_best >= threshold
        )

    final_dice = baseline_dice.copy()

    final_dice[
        reject & labels_nt_bool
    ] = 1.0

    final_dice[
        reject & labels_pos_bool
    ] = 0.0

    with open(
        VAL_PREDICTIONS_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ref_id",
                "is_no_target",
                "p_no_target",
                "gate_rejects_as_no_target",
                "baseline_expert_dice",
                "final_dice",
                "route_branch",
            ],
        )

        writer.writeheader()

        for i, ref_id in enumerate(
            val_data["ref_ids"]
        ):
            writer.writerow({
                "ref_id":
                    int(ref_id),
                "is_no_target":
                    bool(labels_nt_bool[i]),
                "p_no_target":
                    float(p_nt_best[i]),
                "gate_rejects_as_no_target":
                    bool(reject[i]),
                "baseline_expert_dice":
                    float(baseline_dice[i]),
                "final_dice":
                    float(final_dice[i]),
                "route_branch":
                    step40_rows[
                        int(ref_id)
                    ][
                        "route_branch"
                    ],
            })

    if history:
        with open(
            HISTORY_CSV,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    history[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(history)

    result = {
        "step":
            42,
        "testB_used":
            False,
        "feature_level_gate":
            True,
        "image_feature_hook":
            train_data["hook_name"],
        "input_dim":
            int(train_z.shape[1]),
        "baseline": {
            "positive_dice":
                baseline_positive,
            "no_target_accuracy":
                baseline_nt,
            "full_val_dice":
                baseline_full,
        },
        "best_training_epoch":
            best_epoch,
        "policy_results":
            policy_results,
        "official_choice":
            official,
        "gain": {
            "positive_dice":
                official["positive_dice"]
                -
                baseline_positive,
            "no_target_accuracy":
                official["no_target_accuracy"]
                -
                baseline_nt,
            "full_val_dice":
                official["full_val_dice"]
                -
                baseline_full,
        },
        "outputs": {
            "checkpoint":
                (
                    str(BEST_CHECKPOINT)
                    if BEST_CHECKPOINT.exists()
                    else None
                ),
            "history":
                str(HISTORY_CSV),
            "val_predictions":
                str(VAL_PREDICTIONS_CSV),
            "train_feature_cache":
                str(TRAIN_FEATURE_CACHE),
            "val_feature_cache":
                str(VAL_FEATURE_CACHE),
        },
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print()
    print("=" * 112)
    print("STEP 42 FINAL RESULT")
    print("=" * 112)

    print()
    print(
        f"Baseline Positive Dice : "
        f"{baseline_positive:.4f}"
    )
    print(
        f"Gate Positive Dice     : "
        f"{official['positive_dice']:.4f}"
    )
    print(
        f"Positive Survival      : "
        f"{official['positive_survival_recall']:.2%}"
    )

    print()
    print(
        f"Baseline NT Accuracy   : "
        f"{baseline_nt:.4f}"
    )
    print(
        f"Gate NT Accuracy       : "
        f"{official['no_target_accuracy']:.4f}"
    )
    print(
        f"Gate NT Recall         : "
        f"{official['gate_no_target_recall']:.2%}"
    )

    print()
    print(
        f"Baseline Full-Val Dice : "
        f"{baseline_full:.4f}"
    )
    print(
        f"Gate Full-Val Dice     : "
        f"{official['full_val_dice']:.4f}"
    )
    print(
        f"Full-Val Gain          : "
        f"{official['full_val_dice'] - baseline_full:+.4f}"
    )

    print()
    print(
        f"Selected threshold     : "
        f"{official['threshold']}"
    )
    print(
        f"Best training epoch    : "
        f"{best_epoch}"
    )
    print(
        f"Image feature hook     : "
        f"{train_data['hook_name']}"
    )

    print()
    print(f"Result JSON:\n{RESULT_JSON}")
    print()
    print(f"Gate checkpoint:\n{BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()
