import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import RefELDataset

from model_text import (
    TextGuidedUNet,
    BERT_NAME,
    TEXT_MAX_LENGTH,
    TEXT_DIM
)


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

IMAGE_SIZE = 384

BATCH_SIZE = 4

# Keep unchanged for the first cache-accelerated experiment.
NUM_WORKERS = 0

SEED = 42


# ============================================================
# Formal full training
# ============================================================

MAX_EPOCHS = 20

EARLY_STOP_PATIENCE = 5


# ============================================================
# Learning rates
# ============================================================

ENCODER_LR = 3e-5

HEAD_LR = 3e-4

WEIGHT_DECAY = 1e-4


# ============================================================
# Loss
# ============================================================

POS_WEIGHT = 8.0

BCE_WEIGHT = 0.4

DICE_WEIGHT = 0.6


# ============================================================
# Evaluation threshold
# ============================================================

MASK_THRESHOLD = 0.5


# ============================================================
# Logging
# ============================================================

LOG_EVERY = 250


# ============================================================
# Cache
# ============================================================

CACHE_DIR = (
    ROOT
    / "text_cache"
    / "bert_pool_v1_b4_fp32_maxlen32"
)

CACHE_FEATURE_FILE = (
    CACHE_DIR
    / "text_features.pt"
)

CACHE_METADATA_FILE = (
    CACHE_DIR
    / "text_metadata.pt"
)

CACHE_MANIFEST_FILE = (
    CACHE_DIR
    / "manifest.json"
)

CACHE_BUILDING_FILE = (
    CACHE_DIR
    / "BUILDING.json"
)

GREFS_FILE = (
    ROOT
    / "datasets"
    / "grefcoco"
    / "grefs(unc).json"
)


# ============================================================
# Checkpoints
# ============================================================

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Protected Epoch-3 source.
#
# NEVER overwrite this file.
# ------------------------------------------------------------

SOURCE_EPOCH3_CHECKPOINT = (
    CHECKPOINT_DIR
    / "text_gate_full_latest_epoch3_backup.pth"
)

# ------------------------------------------------------------
# Cache-training checkpoints.
#
# These use NEW filenames so the previous ONLINE training
# checkpoints remain untouched.
# ------------------------------------------------------------

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "text_gate_cache_full_latest.pth"
)

BEST_DICE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "text_gate_cache_full_best_dice.pth"
)

BEST_SCORE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "text_gate_cache_full_best_score.pth"
)


# ============================================================
# Hash helpers
# ============================================================

def text_sha256(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def file_sha256(path):

    hasher = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            hasher.update(
                block
            )

    return hasher.hexdigest()


# ============================================================
# Formal Text Cache
#
# Fail-closed cache reader.
# ============================================================

class FormalTextCache:

    def __init__(
        self,
        root_dir
    ):

        self.root_dir = Path(
            root_dir
        )

        self.cache_dir = (
            self.root_dir
            / "text_cache"
            / "bert_pool_v1_b4_fp32_maxlen32"
        )

        self.feature_file = (
            self.cache_dir
            / "text_features.pt"
        )

        self.metadata_file = (
            self.cache_dir
            / "text_metadata.pt"
        )

        self.manifest_file = (
            self.cache_dir
            / "manifest.json"
        )

        self.building_file = (
            self.cache_dir
            / "BUILDING.json"
        )

        self.grefs_file = (
            self.root_dir
            / "datasets"
            / "grefcoco"
            / "grefs(unc).json"
        )

        print("\n")
        print("=" * 80)
        print(
            "TG_Unet FORMAL TEXT CACHE CHECK"
        )
        print("=" * 80)

        # ====================================================
        # Basic file existence
        # ====================================================

        if self.building_file.exists():

            raise RuntimeError(
                "BUILDING.json still exists.\n"
                "Formal cache is not allowed for training."
            )

        for path in [
            self.feature_file,
            self.metadata_file,
            self.manifest_file
        ]:

            if not path.exists():

                raise FileNotFoundError(
                    f"Missing formal cache file:\n"
                    f"{path}"
                )

        # ====================================================
        # Manifest
        # ====================================================

        with open(
            self.manifest_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.manifest = json.load(
                f
            )

        self._verify_manifest()

        # ====================================================
        # Verify real file hashes
        # ====================================================

        print(
            "\nVerifying cache file SHA256..."
        )

        actual_feature_hash = (
            file_sha256(
                self.feature_file
            )
        )

        actual_metadata_hash = (
            file_sha256(
                self.metadata_file
            )
        )

        if (
            actual_feature_hash
            !=
            self.manifest[
                "feature_file_sha256"
            ]
        ):

            raise RuntimeError(
                "text_features.pt SHA256 "
                "does not match manifest."
            )

        if (
            actual_metadata_hash
            !=
            self.manifest[
                "metadata_file_sha256"
            ]
        ):

            raise RuntimeError(
                "text_metadata.pt SHA256 "
                "does not match manifest."
            )

        # ====================================================
        # Verify current source JSON
        # ====================================================

        print(
            "Verifying current grefs SHA256..."
        )

        current_grefs_hash = (
            file_sha256(
                self.grefs_file
            )
        )

        if (
            current_grefs_hash
            !=
            self.manifest[
                "source_grefs_sha256"
            ]
        ):

            raise RuntimeError(
                "Current grefs(unc).json differs "
                "from the cache source."
            )

        print(
            "Cache SHA256 verification PASSED."
        )

        # ====================================================
        # Load cache
        # ====================================================

        print(
            "\nLoading formal text features..."
        )

        self.features = torch.load(
            self.feature_file,
            map_location="cpu",
            weights_only=False
        )

        print(
            "Loading formal text metadata..."
        )

        self.metadata = torch.load(
            self.metadata_file,
            map_location="cpu",
            weights_only=False
        )

        # ====================================================
        # Structure verification
        # ====================================================

        self._verify_structure()

        self.ref_ids = (
            self.metadata[
                "ref_ids"
            ]
        )

        self.split_codes = (
            self.metadata[
                "split_codes"
            ]
        )

        self.text_hashes = (
            self.metadata[
                "text_sha256"
            ]
        )

        self.split_to_code = {
            "train": 0,
            "val": 1,
            "testB": 2
        }

        print("\n")
        print(
            "FORMAL TEXT CACHE READY."
        )

        print(
            f"Feature shape : "
            f"{tuple(self.features.shape)}"
        )

        print(
            f"Feature dtype : "
            f"{self.features.dtype}"
        )

        print(
            f"Cache version : "
            f"{self.manifest['cache_version']}"
        )

        print("=" * 80)

    # ========================================================
    # Manifest validation
    # ========================================================

    def _verify_manifest(
        self
    ):

        m = self.manifest

        required = {

            "project":
                "TG_Unet",

            "status":
                "COMPLETE",

            "cache_version":
                "bert_pool_v1_b4_fp32_maxlen32",

            "cache_type":
                "pure_frozen_bert_pooled_text",

            "total_sentences":
                195000,

            "sent_id_min":
                0,

            "sent_id_max":
                194999,

            "sent_id_contiguous":
                True,

            "bert_name":
                BERT_NAME,

            "bert_frozen":
                True,

            "bert_eval":
                True,

            "dtype":
                "float32",

            "feature_dim":
                TEXT_DIM,

            "max_length":
                TEXT_MAX_LENGTH,

            "padding":
                "max_length",

            "cache_batch_size":
                4
        }

        for key, expected in required.items():

            actual = m.get(
                key
            )

            if actual != expected:

                raise RuntimeError(
                    "Formal cache manifest mismatch.\n"
                    f"key      = {key}\n"
                    f"expected = {expected}\n"
                    f"actual   = {actual}"
                )

        if (
            m.get(
                "strict_full_reverify"
            )
            is not True
        ):

            raise RuntimeError(
                "Cache was not fully reverified."
            )

        if (
            m.get(
                "unequal_sentences"
            )
            !=
            0
        ):

            raise RuntimeError(
                "Formal cache contains unequal sentences."
            )

        if float(
            m.get(
                "max_abs_error",
                -1
            )
        ) != 0.0:

            raise RuntimeError(
                "Formal cache max_abs_error is not zero."
            )

        if (
            m.get(
                "feature_file"
            )
            !=
            self.feature_file.name
        ):

            raise RuntimeError(
                "Manifest feature filename mismatch."
            )

        if (
            m.get(
                "metadata_file"
            )
            !=
            self.metadata_file.name
        ):

            raise RuntimeError(
                "Manifest metadata filename mismatch."
            )

        print(
            "\nManifest check PASSED."
        )

    # ========================================================
    # Cache tensor validation
    # ========================================================

    def _verify_structure(
        self
    ):

        if (
            self.features.shape
            !=
            torch.Size(
                [
                    195000,
                    TEXT_DIM
                ]
            )
        ):

            raise RuntimeError(
                f"Wrong feature shape: "
                f"{self.features.shape}"
            )

        if (
            self.features.dtype
            !=
            torch.float32
        ):

            raise RuntimeError(
                "Formal text cache must be FP32."
            )

        # ----------------------------------------------------
        # Chunked finite check to avoid a huge temporary tensor.
        # ----------------------------------------------------

        for start in range(
            0,
            self.features.shape[0],
            10000
        ):

            chunk = self.features[
                start:
                start + 10000
            ]

            if not torch.isfinite(
                chunk
            ).all():

                raise RuntimeError(
                    "NaN/Inf detected in "
                    "formal text cache."
                )

        if (
            self.metadata[
                "ref_ids"
            ].shape
            !=
            torch.Size(
                [195000]
            )
        ):

            raise RuntimeError(
                "Metadata ref_ids shape mismatch."
            )

        if (
            self.metadata[
                "split_codes"
            ].shape
            !=
            torch.Size(
                [195000]
            )
        ):

            raise RuntimeError(
                "Metadata split_codes shape mismatch."
            )

        if (
            len(
                self.metadata[
                    "text_sha256"
                ]
            )
            !=
            195000
        ):

            raise RuntimeError(
                "Metadata text hash count mismatch."
            )

        print(
            "Cache structure check PASSED."
        )

    # ========================================================
    # Batch lookup
    #
    # This function is used for EVERY train / val batch.
    #
    # It does not trust sent_id alone.
    #
    # It cross-checks:
    #
    # sent_id
    # ref_id
    # text SHA256
    # split
    # ========================================================

    def get_batch(
        self,
        batch,
        expected_split,
        device
    ):

        if (
            expected_split
            not in
            self.split_to_code
        ):

            raise ValueError(
                f"Unknown split: "
                f"{expected_split}"
            )

        # ----------------------------------------------------
        # sent_id
        # ----------------------------------------------------

        sent_ids = batch[
            "sent_id"
        ]

        if not torch.is_tensor(
            sent_ids
        ):

            sent_ids = torch.tensor(
                sent_ids,
                dtype=torch.long
            )

        sent_ids = (
            sent_ids
            .detach()
            .cpu()
            .long()
            .view(-1)
        )

        # ----------------------------------------------------
        # ref_id
        # ----------------------------------------------------

        ref_ids = batch[
            "ref_id"
        ]

        if torch.is_tensor(
            ref_ids
        ):

            ref_ids = (
                ref_ids
                .detach()
                .cpu()
                .long()
                .view(-1)
                .tolist()
            )

        else:

            ref_ids = [
                int(x)
                for x in ref_ids
            ]

        texts = list(
            batch[
                "text"
            ]
        )

        batch_size = (
            sent_ids.numel()
        )

        if not (
            batch_size
            ==
            len(ref_ids)
            ==
            len(texts)
        ):

            raise RuntimeError(
                "Cache lookup batch size mismatch."
            )

        expected_split_code = (
            self.split_to_code[
                expected_split
            ]
        )

        # ----------------------------------------------------
        # Semantic identity checks
        # ----------------------------------------------------

        for local_index in range(
            batch_size
        ):

            sent_id = int(
                sent_ids[
                    local_index
                ].item()
            )

            ref_id = int(
                ref_ids[
                    local_index
                ]
            )

            text = texts[
                local_index
            ]

            if not (
                0
                <=
                sent_id
                <
                195000
            ):

                raise RuntimeError(
                    f"Invalid sent_id={sent_id}"
                )

            cached_ref_id = int(
                self.ref_ids[
                    sent_id
                ].item()
            )

            if (
                cached_ref_id
                !=
                ref_id
            ):

                raise RuntimeError(
                    "Formal cache ref_id mismatch.\n"
                    f"sent_id={sent_id}\n"
                    f"dataset ref_id={ref_id}\n"
                    f"cache ref_id={cached_ref_id}"
                )

            cached_split = int(
                self.split_codes[
                    sent_id
                ].item()
            )

            if (
                cached_split
                !=
                expected_split_code
            ):

                raise RuntimeError(
                    "Formal cache split mismatch.\n"
                    f"sent_id={sent_id}\n"
                    f"expected split={expected_split}\n"
                    f"cached code={cached_split}"
                )

            current_hash = (
                text_sha256(
                    text
                )
            )

            cached_hash = (
                self.text_hashes[
                    sent_id
                ]
            )

            if (
                current_hash
                !=
                cached_hash
            ):

                raise RuntimeError(
                    "Formal cache text SHA256 mismatch.\n"
                    f"sent_id={sent_id}\n"
                    f"text={text}"
                )

        # ----------------------------------------------------
        # sent_id is the semantic row index.
        # ----------------------------------------------------

        pooled_text = (
            self.features[
                sent_ids
            ]
            .contiguous()
        )

        if (
            pooled_text.shape
            !=
            torch.Size(
                [
                    batch_size,
                    TEXT_DIM
                ]
            )
        ):

            raise RuntimeError(
                "Cached batch feature shape mismatch."
            )

        if (
            pooled_text.dtype
            !=
            torch.float32
        ):

            raise RuntimeError(
                "Cached batch must be FP32."
            )

        pooled_text = pooled_text.to(
            device,
            non_blocking=True
        )

        if pooled_text.requires_grad:

            raise RuntimeError(
                "Cached pooled_text "
                "must not require gradients."
            )

        return pooled_text


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# RNG checkpoint state
# ============================================================

def capture_rng_state():

    state = {

        "python":
            random.getstate(),

        "numpy":
            np.random.get_state(),

        "torch":
            torch.get_rng_state()
    }

    if torch.cuda.is_available():

        state[
            "cuda"
        ] = (
            torch.cuda.get_rng_state_all()
        )

    else:

        state[
            "cuda"
        ] = None

    return state


def restore_rng_state(
    state
):

    if state is None:

        return False

    try:

        random.setstate(
            state[
                "python"
            ]
        )

        np.random.set_state(
            state[
                "numpy"
            ]
        )

        torch.set_rng_state(
            state[
                "torch"
            ]
        )

        if (
            torch.cuda.is_available()
            and
            state.get(
                "cuda"
            )
            is not None
        ):

            torch.cuda.set_rng_state_all(
                state[
                    "cuda"
                ]
            )

        return True

    except Exception as exc:

        raise RuntimeError(
            "Failed to restore RNG state."
        ) from exc


# ============================================================
# Loss
# ============================================================

def calculate_loss(
    logits,
    masks,
    presence
):

    masks = masks.float()

    presence = (
        presence
        .float()
        .view(-1)
    )

    positive_weight = torch.tensor(
        [POS_WEIGHT],
        dtype=logits.dtype,
        device=logits.device
    )

    bce_map = (
        F.binary_cross_entropy_with_logits(
            logits,
            masks,
            pos_weight=positive_weight,
            reduction="none"
        )
    )

    bce_per_sample = (
        bce_map.mean(
            dim=(
                1,
                2,
                3
            )
        )
    )

    probabilities = torch.sigmoid(
        logits
    )

    dims = (
        1,
        2,
        3
    )

    intersection = (
        probabilities
        *
        masks
    ).sum(
        dim=dims
    )

    probability_sum = (
        probabilities
    ).sum(
        dim=dims
    )

    target_sum = (
        masks
    ).sum(
        dim=dims
    )

    dice = (
        2.0
        *
        intersection
        +
        1.0
    ) / (
        probability_sum
        +
        target_sum
        +
        1.0
    )

    dice_loss_per_sample = (
        1.0
        -
        dice
    )

    positive_loss = (
        BCE_WEIGHT
        *
        bce_per_sample
        +
        DICE_WEIGHT
        *
        dice_loss_per_sample
    )

    negative_loss = (
        bce_per_sample
    )

    sample_loss = torch.where(
        presence > 0.5,
        positive_loss,
        negative_loss
    )

    total_loss = (
        sample_loss.mean()
    )

    return (
        total_loss,
        bce_per_sample.mean().detach(),
        dice_loss_per_sample.mean().detach()
    )


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def calculate_metrics(
    logits,
    masks,
    presence,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities
        >
        threshold
    ).float()

    masks = masks.float()

    dims = (
        1,
        2,
        3
    )

    intersection = (
        predictions
        *
        masks
    ).sum(
        dim=dims
    )

    prediction_sum = (
        predictions
    ).sum(
        dim=dims
    )

    target_sum = (
        masks
    ).sum(
        dim=dims
    )

    hard_dice = (
        2.0
        *
        intersection
        +
        1e-6
    ) / (
        prediction_sum
        +
        target_sum
        +
        1e-6
    )

    soft_intersection = (
        probabilities
        *
        masks
    ).sum(
        dim=dims
    )

    probability_sum = (
        probabilities
    ).sum(
        dim=dims
    )

    soft_dice = (
        2.0
        *
        soft_intersection
        +
        1e-6
    ) / (
        probability_sum
        +
        target_sum
        +
        1e-6
    )

    positive_selector = (
        presence.view(-1)
        >
        0.5
    )

    negative_selector = (
        ~positive_selector
    )

    predicted_empty = (
        prediction_sum == 0
    )

    prediction_ratio = (
        predictions.mean(
            dim=dims
        )
    )

    return {

        "hard_dice":
            hard_dice,

        "soft_dice":
            soft_dice,

        "positive_selector":
            positive_selector,

        "negative_selector":
            negative_selector,

        "predicted_empty":
            predicted_empty,

        "prediction_ratio":
            prediction_ratio
    }


# ============================================================
# Validation using FORMAL CACHE
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    text_cache
):

    model.eval()

    positive_hard_sum = 0.0
    positive_soft_sum = 0.0
    positive_count = 0

    no_target_correct = 0
    no_target_count = 0
    no_target_pred_ratio_sum = 0.0

    all_hard_sum = 0.0
    all_pred_ratio_sum = 0.0
    all_count = 0

    for batch in loader:

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True
        )

        masks = batch[
            "mask"
        ].to(
            device,
            non_blocking=True
        )

        presence = batch[
            "presence"
        ].to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # val uses fixed first sentence from Dataset.
        # sent_id points to exactly that sentence.
        # ----------------------------------------------------

        pooled_text = (
            text_cache.get_batch(
                batch=batch,
                expected_split="val",
                device=device
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Frozen BERT is completely bypassed.
        # ----------------------------------------------------

        logits = model(
            images,
            pooled_text=pooled_text
        )

        metrics = calculate_metrics(
            logits,
            masks,
            presence,
            threshold=MASK_THRESHOLD
        )

        hard_dice = (
            metrics[
                "hard_dice"
            ]
        )

        soft_dice = (
            metrics[
                "soft_dice"
            ]
        )

        positive_selector = (
            metrics[
                "positive_selector"
            ]
        )

        negative_selector = (
            metrics[
                "negative_selector"
            ]
        )

        predicted_empty = (
            metrics[
                "predicted_empty"
            ]
        )

        prediction_ratio = (
            metrics[
                "prediction_ratio"
            ]
        )

        all_hard_sum += (
            hard_dice.sum().item()
        )

        all_pred_ratio_sum += (
            prediction_ratio.sum().item()
        )

        all_count += (
            images.shape[0]
        )

        if positive_selector.any():

            positive_hard_sum += (
                hard_dice[
                    positive_selector
                ].sum().item()
            )

            positive_soft_sum += (
                soft_dice[
                    positive_selector
                ].sum().item()
            )

            positive_count += int(
                positive_selector.sum().item()
            )

        if negative_selector.any():

            no_target_correct += int(
                predicted_empty[
                    negative_selector
                ].sum().item()
            )

            no_target_count += int(
                negative_selector.sum().item()
            )

            no_target_pred_ratio_sum += (
                prediction_ratio[
                    negative_selector
                ].sum().item()
            )

    positive_hard_dice = (
        positive_hard_sum
        /
        max(
            positive_count,
            1
        )
    )

    positive_soft_dice = (
        positive_soft_sum
        /
        max(
            positive_count,
            1
        )
    )

    no_target_accuracy = (
        no_target_correct
        /
        max(
            no_target_count,
            1
        )
    )

    no_target_pred_ratio = (
        no_target_pred_ratio_sum
        /
        max(
            no_target_count,
            1
        )
    )

    all_hard_dice = (
        all_hard_sum
        /
        max(
            all_count,
            1
        )
    )

    all_pred_ratio = (
        all_pred_ratio_sum
        /
        max(
            all_count,
            1
        )
    )

    selection_score = (
        0.85
        *
        positive_hard_dice
        +
        0.15
        *
        no_target_accuracy
    )

    return {

        "positive_hard_dice":
            positive_hard_dice,

        "positive_soft_dice":
            positive_soft_dice,

        "no_target_accuracy":
            no_target_accuracy,

        "no_target_pred_ratio":
            no_target_pred_ratio,

        "all_hard_dice":
            all_hard_dice,

        "all_pred_ratio":
            all_pred_ratio,

        "selection_score":
            selection_score,

        "positive_count":
            positive_count,

        "no_target_count":
            no_target_count
    }


# ============================================================
# Optimizer
#
# EXACT same parameter grouping as previous formal training.
# ============================================================

def create_optimizer(
    model
):

    encoder_parameters = (

        list(
            model.stem.parameters()
        )
        +
        list(
            model.layer1.parameters()
        )
        +
        list(
            model.layer2.parameters()
        )
        +
        list(
            model.layer3.parameters()
        )
        +
        list(
            model.layer4.parameters()
        )
    )

    head_parameters = (

        list(
            model.text_gate_e2.parameters()
        )
        +
        list(
            model.text_gate_e3.parameters()
        )
        +
        list(
            model.text_gate_e4.parameters()
        )
        +
        list(
            model.decoder4.parameters()
        )
        +
        list(
            model.decoder3.parameters()
        )
        +
        list(
            model.decoder2.parameters()
        )
        +
        list(
            model.decoder1.parameters()
        )
        +
        list(
            model.final_conv.parameters()
        )
        +
        list(
            model.segmentation_head.parameters()
        )
    )

    optimizer = torch.optim.AdamW(

        [

            {
                "params":
                    encoder_parameters,

                "lr":
                    ENCODER_LR
            },

            {
                "params":
                    head_parameters,

                "lr":
                    HEAD_LR
            }

        ],

        weight_decay=WEIGHT_DECAY
    )

    return optimizer


# ============================================================
# Atomic checkpoint save
# ============================================================

def atomic_torch_save(
    data,
    path
):

    temporary_path = Path(
        str(path) + ".tmp"
    )

    torch.save(
        data,
        temporary_path
    )

    temporary_path.replace(
        path
    )


# ============================================================
# Checkpoint creation
# ============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scaler,
    best_dice,
    best_score,
    best_dice_epoch,
    best_score_epoch,
    epochs_without_dice_improvement,
    val_result,
    text_cache
):

    checkpoint_data = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "rng_state":
            capture_rng_state(),

        "best_positive_dice":
            best_dice,

        "best_selection_score":
            best_score,

        "best_dice_epoch":
            best_dice_epoch,

        "best_score_epoch":
            best_score_epoch,

        "epochs_without_dice_improvement":
            epochs_without_dice_improvement,

        "positive_hard_dice":
            val_result[
                "positive_hard_dice"
            ],

        "positive_soft_dice":
            val_result[
                "positive_soft_dice"
            ],

        "no_target_accuracy":
            val_result[
                "no_target_accuracy"
            ],

        "selection_score":
            val_result[
                "selection_score"
            ],

        "image_size":
            IMAGE_SIZE,

        "mask_threshold":
            MASK_THRESHOLD,

        "encoder_lr":
            ENCODER_LR,

        "head_lr":
            HEAD_LR,

        "weight_decay":
            WEIGHT_DECAY,

        "cache_training":
            True,

        "cache_version":
            text_cache.manifest[
                "cache_version"
            ],

        "cache_feature_sha256":
            text_cache.manifest[
                "feature_file_sha256"
            ],

        "cache_metadata_sha256":
            text_cache.manifest[
                "metadata_file_sha256"
            ]
    }

    atomic_torch_save(
        checkpoint_data,
        path
    )


# ============================================================
# Source checkpoint -> baseline val information
# ============================================================

def checkpoint_val_result(
    checkpoint
):

    return {

        "positive_hard_dice":
            float(
                checkpoint.get(
                    "positive_hard_dice",
                    -1.0
                )
            ),

        "positive_soft_dice":
            float(
                checkpoint.get(
                    "positive_soft_dice",
                    -1.0
                )
            ),

        "no_target_accuracy":
            float(
                checkpoint.get(
                    "no_target_accuracy",
                    -1.0
                )
            ),

        "selection_score":
            float(
                checkpoint.get(
                    "selection_score",
                    -1.0
                )
            )
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )

    print("\n")
    print("=" * 80)
    print(
        "TG_Unet CACHE-ACCELERATED FULL TRAINING"
    )
    print("=" * 80)

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    # ========================================================
    # Load and validate formal cache
    # ========================================================

    text_cache = FormalTextCache(
        ROOT
    )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "\nLoading full training dataset..."
    )

    train_dataset = RefELDataset(
        root_dir=ROOT,
        split="train",
        image_size=IMAGE_SIZE,
        random_text=True
    )

    print(
        "\nLoading full validation dataset..."
    )

    val_dataset = RefELDataset(
        root_dir=ROOT,
        split="val",
        image_size=IMAGE_SIZE,
        random_text=False
    )

    if len(
        train_dataset
    ) != 45000:

        raise RuntimeError(
            "Expected 45,000 train refs."
        )

    if len(
        val_dataset
    ) != 8000:

        raise RuntimeError(
            "Expected 8,000 val refs."
        )

    print(
        "\nDataset sizes:"
    )

    print(
        f"Train = "
        f"{len(train_dataset)}"
    )

    print(
        f"Val   = "
        f"{len(val_dataset)}"
    )

    # ========================================================
    # DataLoaders
    #
    # Deliberately unchanged:
    #
    # batch = 4
    # workers = 0
    # ========================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        ),

        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type == "cuda"
        ),

        drop_last=False
    )

    print(
        f"\nTrain batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Val batches  : "
        f"{len(val_loader)}"
    )

    # ========================================================
    # Model
    #
    # pretrained=False:
    #
    # We immediately load a checkpoint.
    # ========================================================

    print(
        "\nCreating TextGuidedUNet..."
    )

    model = TextGuidedUNet(
        pretrained=False
    ).to(
        device
    )

    # --------------------------------------------------------
    # BERT must remain frozen even though cache bypasses it.
    # --------------------------------------------------------

    bert_trainable = sum(

        parameter.numel()

        for parameter
        in model.text_encoder.bert.parameters()

        if parameter.requires_grad
    )

    if bert_trainable != 0:

        raise RuntimeError(
            "BERT is unexpectedly trainable."
        )

    model.text_encoder.bert.eval()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = create_optimizer(
        model
    )

    # ========================================================
    # AMP scaler
    # ========================================================

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # ========================================================
    # Select resume source automatically
    #
    # First cache run:
    #   Epoch-3 protected backup
    #
    # Later interrupted run:
    #   cache full latest
    # ========================================================

    if LATEST_CHECKPOINT.exists():

        resume_checkpoint_path = (
            LATEST_CHECKPOINT
        )

        resume_kind = (
            "CACHE LATEST"
        )

    else:

        resume_checkpoint_path = (
            SOURCE_EPOCH3_CHECKPOINT
        )

        resume_kind = (
            "PROTECTED EPOCH-3 SOURCE"
        )

    if not resume_checkpoint_path.exists():

        raise FileNotFoundError(
            f"Resume checkpoint not found:\n"
            f"{resume_checkpoint_path}"
        )

    print("\n")
    print("=" * 80)
    print(
        f"RESUME SOURCE: {resume_kind}"
    )
    print("=" * 80)

    print(
        resume_checkpoint_path
    )

    # ========================================================
    # If first run from Epoch 3:
    # verify checkpoint is exactly the one used to build cache.
    # ========================================================

    if (
        resume_checkpoint_path
        ==
        SOURCE_EPOCH3_CHECKPOINT
    ):

        source_hash = file_sha256(
            SOURCE_EPOCH3_CHECKPOINT
        )

        expected_source_hash = (
            text_cache.manifest[
                "source_checkpoint_sha256"
            ]
        )

        if (
            source_hash
            !=
            expected_source_hash
        ):

            raise RuntimeError(
                "Protected Epoch-3 checkpoint "
                "does not match cache manifest."
            )

        print(
            "Epoch-3 source SHA256 PASSED."
        )

    # ========================================================
    # Load checkpoint
    # ========================================================

    checkpoint = torch.load(
        resume_checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    checkpoint_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    # ========================================================
    # Resume validity
    # ========================================================

    if (
        resume_checkpoint_path
        ==
        SOURCE_EPOCH3_CHECKPOINT
    ):

        if checkpoint_epoch != 3:

            raise RuntimeError(
                "Protected source must be Epoch 3."
            )

    else:

        if (
            checkpoint.get(
                "cache_training"
            )
            is not True
        ):

            raise RuntimeError(
                "Cache latest checkpoint is not "
                "marked cache_training=True."
            )

        if (
            checkpoint.get(
                "cache_version"
            )
            !=
            text_cache.manifest[
                "cache_version"
            ]
        ):

            raise RuntimeError(
                "Resume checkpoint cache version mismatch."
            )

        if (
            checkpoint.get(
                "cache_feature_sha256"
            )
            !=
            text_cache.manifest[
                "feature_file_sha256"
            ]
        ):

            raise RuntimeError(
                "Resume checkpoint feature cache mismatch."
            )

        if checkpoint_epoch < 4:

            raise RuntimeError(
                "Cache latest checkpoint "
                "must be Epoch 4 or later."
            )

    start_epoch = (
        checkpoint_epoch
        +
        1
    )

    print(
        "\nCheckpoint loaded with strict=True."
    )

    print(
        f"Checkpoint epoch : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Start epoch      : "
        f"{start_epoch}"
    )

    print(
        "Optimizer state  : restored"
    )

    # ========================================================
    # GradScaler
    #
    # Epoch-3 old checkpoint did not save it.
    # Future cache checkpoints do.
    # ========================================================

    if (
        "scaler_state_dict"
        in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

        print(
            "AMP scaler state : restored"
        )

    else:

        print(
            "AMP scaler state : fresh "
            "(old Epoch-3 checkpoint had no scaler state)"
        )

    # ========================================================
    # Best values
    #
    # IMPORTANT:
    # Epoch-3 old latest checkpoint saved best metadata
    # BEFORE updating that epoch.
    #
    # Therefore use max(stored best, current epoch result).
    # ========================================================

    current_checkpoint_dice = float(
        checkpoint.get(
            "positive_hard_dice",
            -1.0
        )
    )

    stored_best_dice = float(
        checkpoint.get(
            "best_positive_dice",
            -1.0
        )
    )

    best_positive_dice = max(
        current_checkpoint_dice,
        stored_best_dice
    )

    current_checkpoint_score = float(
        checkpoint.get(
            "selection_score",
            -1.0
        )
    )

    stored_best_score = float(
        checkpoint.get(
            "best_selection_score",
            -1.0
        )
    )

    best_selection_score = max(
        current_checkpoint_score,
        stored_best_score
    )

    if (
        resume_checkpoint_path
        ==
        SOURCE_EPOCH3_CHECKPOINT
    ):

        # Epoch 3 was the current best at the moment
        # training was stopped.
        best_dice_epoch = 3
        best_score_epoch = 3

        epochs_without_dice_improvement = 0

    else:

        best_dice_epoch = int(
            checkpoint.get(
                "best_dice_epoch",
                checkpoint_epoch
            )
        )

        best_score_epoch = int(
            checkpoint.get(
                "best_score_epoch",
                checkpoint_epoch
            )
        )

        epochs_without_dice_improvement = int(
            checkpoint.get(
                "epochs_without_dice_improvement",
                0
            )
        )

    print("\n")
    print(
        f"Best Positive Dice so far : "
        f"{best_positive_dice:.4f}"
    )

    print(
        f"Best Dice epoch           : "
        f"{best_dice_epoch}"
    )

    print(
        f"Best Selection Score      : "
        f"{best_selection_score:.4f}"
    )

    print(
        f"Best Score epoch          : "
        f"{best_score_epoch}"
    )

    # ========================================================
    # Restore RNG for future cache checkpoints
    #
    # Epoch-3 old checkpoint cannot restore RNG because those
    # states were not saved in the original training script.
    # ========================================================

    if (
        "rng_state"
        in checkpoint
    ):

        restore_rng_state(
            checkpoint[
                "rng_state"
            ]
        )

        print(
            "RNG state                 : restored"
        )

    else:

        print(
            "RNG state                 : unavailable "
            "for old Epoch-3 checkpoint"
        )

    # ========================================================
    # Create Epoch-3 baseline best files on first cache run.
    #
    # This guarantees that even if Epoch 4+ never beats
    # Epoch 3, we still retain a cache-series best checkpoint.
    # ========================================================

    if (
        resume_checkpoint_path
        ==
        SOURCE_EPOCH3_CHECKPOINT
    ):

        baseline_result = (
            checkpoint_val_result(
                checkpoint
            )
        )

        if not BEST_DICE_CHECKPOINT.exists():

            save_checkpoint(
                BEST_DICE_CHECKPOINT,
                checkpoint_epoch,
                model,
                optimizer,
                scaler,
                best_positive_dice,
                best_selection_score,
                best_dice_epoch,
                best_score_epoch,
                epochs_without_dice_improvement,
                baseline_result,
                text_cache
            )

            print(
                "\nCreated cache-series "
                "Epoch-3 best-dice baseline."
            )

        if not BEST_SCORE_CHECKPOINT.exists():

            save_checkpoint(
                BEST_SCORE_CHECKPOINT,
                checkpoint_epoch,
                model,
                optimizer,
                scaler,
                best_positive_dice,
                best_selection_score,
                best_dice_epoch,
                best_score_epoch,
                epochs_without_dice_improvement,
                baseline_result,
                text_cache
            )

            print(
                "Created cache-series "
                "Epoch-3 best-score baseline."
            )

    # ========================================================
    # Already complete?
    # ========================================================

    if start_epoch > MAX_EPOCHS:

        print(
            "\nTraining is already beyond "
            f"MAX_EPOCHS={MAX_EPOCHS}."
        )

        return

    # ========================================================
    # Training
    # ========================================================

    epoch_times = []

    print("\n")
    print("=" * 80)
    print(
        "START CACHE-ACCELERATED FULL TRAINING"
    )
    print("=" * 80)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Frozen BERT forward is now BYPASSED."
    )

    print(
        "Formal verified pooled_text cache is used instead."
    )

    print(
        f"Start from Epoch {start_epoch}."
    )

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1
    ):

        epoch_start = (
            time.perf_counter()
        )

        model.train()

        # BERT remains frozen/eval even though cache bypasses it.
        model.text_encoder.eval()
        model.text_encoder.bert.eval()

        running_loss = 0.0
        running_bce = 0.0
        running_dice = 0.0

        num_batches = 0

        # ====================================================
        # Train
        # ====================================================

        for batch_index, batch in enumerate(
            train_loader,
            start=1
        ):

            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True
            )

            masks = batch[
                "mask"
            ].to(
                device,
                non_blocking=True
            )

            presence = batch[
                "presence"
            ].to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # random_text=True is still active.
            #
            # Dataset randomly selected text + sent_id.
            # Cache verifies identity and returns feature.
            # ------------------------------------------------

            pooled_text = (
                text_cache.get_batch(
                    batch=batch,
                    expected_split="train",
                    device=device
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):

                # --------------------------------------------
                # No tokenizer.
                # No BERT forward.
                #
                # Starts directly at verified pooled_text.
                # --------------------------------------------

                logits = model(
                    images,
                    pooled_text=pooled_text
                )

                (
                    loss,
                    bce_value,
                    dice_value

                ) = calculate_loss(
                    logits,
                    masks,
                    presence
                )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite loss detected."
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter
                    in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
            )

            running_bce += (
                bce_value.item()
            )

            running_dice += (
                dice_value.item()
            )

            num_batches += 1

            if (
                batch_index == 1
                or
                batch_index
                %
                LOG_EVERY
                ==
                0
                or
                batch_index
                ==
                len(train_loader)
            ):

                print(

                    f"Epoch "
                    f"{epoch:02d}/{MAX_EPOCHS} | "

                    f"Batch "
                    f"{batch_index:05d}/"
                    f"{len(train_loader):05d} | "

                    f"Loss "
                    f"{loss.item():.4f}"
                )

        # ====================================================
        # Train averages
        # ====================================================

        train_loss = (
            running_loss
            /
            num_batches
        )

        train_bce = (
            running_bce
            /
            num_batches
        )

        train_dice_loss = (
            running_dice
            /
            num_batches
        )

        # ====================================================
        # Validation
        # ====================================================

        val_result = validate(
            model,
            val_loader,
            device,
            text_cache
        )

        epoch_time = (
            time.perf_counter()
            -
            epoch_start
        )

        epoch_times.append(
            epoch_time
        )

        average_epoch_time = (
            sum(
                epoch_times
            )
            /
            len(
                epoch_times
            )
        )

        remaining_epochs = (
            MAX_EPOCHS
            -
            epoch
        )

        estimated_remaining = (
            average_epoch_time
            *
            remaining_epochs
        )

        # ====================================================
        # Summary
        # ====================================================

        print("\n")
        print("-" * 80)
        print(
            f"EPOCH {epoch:02d} SUMMARY"
        )
        print("-" * 80)

        print(
            f"Train Loss          : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train BCE           : "
            f"{train_bce:.4f}"
        )

        print(
            f"Train DiceLoss      : "
            f"{train_dice_loss:.4f}"
        )

        print(
            f"Val Positive Dice   : "
            f"{val_result['positive_hard_dice']:.4f}"
        )

        print(
            f"Val Positive Soft   : "
            f"{val_result['positive_soft_dice']:.4f}"
        )

        print(
            f"Val All-sample Dice : "
            f"{val_result['all_hard_dice']:.4f}"
        )

        print(
            f"No-target Accuracy   : "
            f"{val_result['no_target_accuracy']:.4f}"
        )

        print(
            f"No-target PredRatio  : "
            f"{val_result['no_target_pred_ratio']:.6f}"
        )

        print(
            f"All PredRatio        : "
            f"{val_result['all_pred_ratio']:.6f}"
        )

        print(
            f"Selection Score      : "
            f"{val_result['selection_score']:.4f}"
        )

        print(
            f"Epoch time           : "
            f"{epoch_time / 60:.1f} min"
        )

        print(
            f"Estimated remaining  : "
            f"{estimated_remaining / 3600:.2f} h"
        )

        print("-" * 80)

        # ====================================================
        # Determine new best values FIRST.
        #
        # This fixes the old latest-checkpoint metadata bug.
        # ====================================================

        current_positive_dice = (
            val_result[
                "positive_hard_dice"
            ]
        )

        current_score = (
            val_result[
                "selection_score"
            ]
        )

        dice_improved = (
            current_positive_dice
            >
            best_positive_dice
        )

        score_improved = (
            current_score
            >
            best_selection_score
        )

        if dice_improved:

            best_positive_dice = (
                current_positive_dice
            )

            best_dice_epoch = (
                epoch
            )

            epochs_without_dice_improvement = 0

        else:

            epochs_without_dice_improvement += 1

        if score_improved:

            best_selection_score = (
                current_score
            )

            best_score_epoch = (
                epoch
            )

        # ====================================================
        # Save LATEST after best values were updated.
        #
        # This fixes the previous stale-best metadata issue.
        # ====================================================

        save_checkpoint(
            LATEST_CHECKPOINT,
            epoch,
            model,
            optimizer,
            scaler,
            best_positive_dice,
            best_selection_score,
            best_dice_epoch,
            best_score_epoch,
            epochs_without_dice_improvement,
            val_result,
            text_cache
        )

        print(
            "\nSaved CACHE latest checkpoint."
        )

        # ====================================================
        # Best Dice
        # ====================================================

        if dice_improved:

            save_checkpoint(
                BEST_DICE_CHECKPOINT,
                epoch,
                model,
                optimizer,
                scaler,
                best_positive_dice,
                best_selection_score,
                best_dice_epoch,
                best_score_epoch,
                epochs_without_dice_improvement,
                val_result,
                text_cache
            )

            print(
                f">>> NEW BEST DICE: "
                f"{best_positive_dice:.4f}"
            )

        # ====================================================
        # Best Score
        # ====================================================

        if score_improved:

            save_checkpoint(
                BEST_SCORE_CHECKPOINT,
                epoch,
                model,
                optimizer,
                scaler,
                best_positive_dice,
                best_selection_score,
                best_dice_epoch,
                best_score_epoch,
                epochs_without_dice_improvement,
                val_result,
                text_cache
            )

            print(
                f">>> NEW BEST SCORE: "
                f"{best_selection_score:.4f}"
            )

        # ====================================================
        # Early stopping
        # ====================================================

        if (
            epochs_without_dice_improvement
            >=
            EARLY_STOP_PATIENCE
        ):

            print("\n")
            print("=" * 80)
            print(
                "EARLY STOPPING"
            )

            print(
                f"Val Positive Dice did not "
                f"improve for "
                f"{EARLY_STOP_PATIENCE} epochs."
            )

            print("=" * 80)

            break

    # ========================================================
    # Finish
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "CACHE-ACCELERATED FULL TRAINING FINISHED"
    )
    print("=" * 80)

    print(
        f"Best Positive Dice : "
        f"{best_positive_dice:.4f}"
    )

    print(
        f"Best Dice epoch    : "
        f"{best_dice_epoch}"
    )

    print(
        f"Best Score         : "
        f"{best_selection_score:.4f}"
    )

    print(
        f"Best Score epoch   : "
        f"{best_score_epoch}"
    )

    print(
        "\nBest Dice checkpoint:"
    )

    print(
        BEST_DICE_CHECKPOINT
    )

    print(
        "\nBest Score checkpoint:"
    )

    print(
        BEST_SCORE_CHECKPOINT
    )

    print(
        "\nLatest checkpoint:"
    )

    print(
        LATEST_CHECKPOINT
    )


if __name__ == "__main__":

    main()
