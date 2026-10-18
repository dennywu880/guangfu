import json
import random
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# ImageNet normalization
# ResNet34 pretrained model will use these values later.
# ============================================================

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# Load JSON only once
# ============================================================

@lru_cache(maxsize=2)
def load_database(
    instance_json_path,
    gref_json_path
):
    """
    Load Ref-EL-Defect annotations.

    The cache prevents train/val datasets from repeatedly
    loading the huge JSON files in the same Python process.
    """

    print("\nLoading dataset annotations...")

    with open(
        instance_json_path,
        "r",
        encoding="utf-8"
    ) as f:

        instances = json.load(f)

    with open(
        gref_json_path,
        "r",
        encoding="utf-8"
    ) as f:

        refs = json.load(f)

    # Fast lookup:
    # image_id -> image information
    images_by_id = {
        image["id"]: image
        for image in instances["images"]
    }

    # Fast lookup:
    # annotation_id -> annotation
    anns_by_id = {
        ann["id"]: ann
        for ann in instances["annotations"]
    }

    print("Dataset annotations loaded.")

    return (
        refs,
        images_by_id,
        anns_by_id
    )


# ============================================================
# Polygon -> mask
# ============================================================

def build_mask(
    ref,
    images_by_id,
    anns_by_id
):
    """
    Convert polygon annotations to a pixel-level binary mask.

    Output:
        [H, W]
        uint8
        values: 0 / 1
    """

    image_id = ref["image_id"]

    image_info = images_by_id[
        image_id
    ]

    height = int(
        image_info["height"]
    )

    width = int(
        image_info["width"]
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # No-target expression:
    # ground truth is an empty mask
    # --------------------------------------------------------

    if ref.get(
        "no_target",
        False
    ):
        return mask

    ann_ids = ref.get(
        "ann_id",
        []
    )

    if not isinstance(
        ann_ids,
        list
    ):
        ann_ids = [ann_ids]

    # --------------------------------------------------------
    # Merge all matching defect regions
    # --------------------------------------------------------

    for ann_id in ann_ids:

        if ann_id == -1:
            continue

        if ann_id not in anns_by_id:

            raise KeyError(
                f"Annotation {ann_id} not found."
            )

        ann = anns_by_id[
            ann_id
        ]

        segmentation = ann.get(
            "segmentation",
            []
        )

        if not isinstance(
            segmentation,
            list
        ):
            continue

        for polygon in segmentation:

            if not isinstance(
                polygon,
                list
            ):
                continue

            # At least three points
            if len(polygon) < 6:
                continue

            points = np.asarray(
                polygon,
                dtype=np.float32
            )

            points = points.reshape(
                -1,
                2
            )

            points = np.rint(
                points
            ).astype(
                np.int32
            )

            cv2.fillPoly(
                mask,
                [points],
                color=1
            )

    mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    return mask


# ============================================================
# Dataset
# ============================================================

class RefELDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=384,
        random_text=True
    ):
        """
        Args:
            root_dir:
                project root directory

            split:
                train / val / testB

            image_size:
                final image size

            random_text:
                True:
                    randomly select one sentence from each ref

                False:
                    always use the first sentence
        """

        super().__init__()

        self.root_dir = Path(
            root_dir
        )

        self.split = split

        self.image_size = int(
            image_size
        )

        self.random_text = (
            random_text
        )

        # ----------------------------------------------------
        # Dataset paths
        # ----------------------------------------------------

        self.instance_json = (
            self.root_dir
            / "datasets"
            / "grefcoco"
            / "instances.json"
        )

        self.gref_json = (
            self.root_dir
            / "datasets"
            / "grefcoco"
            / "grefs(unc).json"
        )

        self.image_dir = (
            self.root_dir
            / "datasets"
            / "images"
            / "train2014_el"
        )

        # ----------------------------------------------------
        # Load annotations
        # ----------------------------------------------------

        (
            all_refs,
            self.images_by_id,
            self.anns_by_id

        ) = load_database(
            str(self.instance_json),
            str(self.gref_json)
        )

        # ----------------------------------------------------
        # Keep only requested split
        # ----------------------------------------------------

        self.refs = [

            ref

            for ref in all_refs

            if ref.get(
                "split"
            ) == split
        ]

        if len(self.refs) == 0:

            raise RuntimeError(
                f"No samples found "
                f"for split='{split}'"
            )

        print(
            f"\nRefELDataset:"
        )

        print(
            f"  split       = {split}"
        )

        print(
            f"  samples     = {len(self.refs)}"
        )

        print(
            f"  image_size  = {image_size}"
        )

        print(
            f"  random_text = {random_text}"
        )

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(
            self.refs
        )

    # ========================================================
    # Choose referring sentence
    #
    # IMPORTANT:
    # text and sent_id must always come from the SAME
    # sentence item.
    # ========================================================

    def get_text(
            self,
            ref
    ):

        sentences = ref.get(
            "sentences",
            []
        )

        if len(sentences) == 0:
            raise RuntimeError(
                f"ref_id={ref.get('ref_id', -1)} "
                f"contains no sentences."
            )

        # ----------------------------------------------------
        # Train:
        # randomly choose one paraphrase
        #
        # Val / Test:
        # always choose the first sentence
        # ----------------------------------------------------

        if (
                self.random_text
                and
                self.split == "train"
        ):

            sentence_item = random.choice(
                sentences
            )

        else:

            sentence_item = sentences[0]

        # ----------------------------------------------------
        # sent_id is the PRIMARY cache identity.
        # ----------------------------------------------------

        if "sent_id" not in sentence_item:
            raise RuntimeError(
                f"Missing sent_id in "
                f"ref_id={ref.get('ref_id', -1)}"
            )

        sent_id = int(
            sentence_item[
                "sent_id"
            ]
        )

        # ----------------------------------------------------
        # Formal cache was built from sentence["sent"].
        #
        # During full-cache preflight we already proved
        # raw == sent for all 195,000 sentences.
        #
        # Keep checking here so a changed JSON cannot silently
        # produce a wrong text/cache pairing.
        # ----------------------------------------------------

        raw_text = sentence_item.get(
            "raw"
        )

        sent_text = sentence_item.get(
            "sent"
        )

        if raw_text is None:
            raise RuntimeError(
                f"Missing raw text for sent_id={sent_id}"
            )

        if sent_text is None:
            raise RuntimeError(
                f"Missing sent text for sent_id={sent_id}"
            )

        if raw_text != sent_text:
            raise RuntimeError(
                f"raw != sent for sent_id={sent_id}\n"
                f"raw : {raw_text}\n"
                f"sent: {sent_text}"
            )

        # IMPORTANT:
        # Use exactly the same text field used to build
        # the formal cache.
        text = sent_text

        return (
            text,
            sent_id
        )

    # ========================================================
    # Read one sample
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        ref = self.refs[
            index
        ]

        image_id = ref[
            "image_id"
        ]

        image_info = self.images_by_id[
            image_id
        ]

        filename = image_info[
            "file_name"
        ]

        image_path = (
            self.image_dir
            / filename
        )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image_bgr is None:

            raise FileNotFoundError(
                f"Cannot read image:\n"
                f"{image_path}"
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        original_height = (
            image_rgb.shape[0]
        )

        original_width = (
            image_rgb.shape[1]
        )

        # ----------------------------------------------------
        # Build GT pixel mask
        # ----------------------------------------------------

        mask = build_mask(
            ref,
            self.images_by_id,
            self.anns_by_id
        )

        # ----------------------------------------------------
        # Safety check before resize
        # ----------------------------------------------------

        if (
            image_rgb.shape[:2]
            !=
            mask.shape
        ):

            raise RuntimeError(

                f"Image/mask shape mismatch:\n"

                f"file = {filename}\n"

                f"image = "
                f"{image_rgb.shape[:2]}\n"

                f"mask = "
                f"{mask.shape}"
            )

        # ----------------------------------------------------
        # Resize IMAGE
        #
        # Bilinear interpolation
        # ----------------------------------------------------

        image_rgb = cv2.resize(
            image_rgb,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Resize MASK
        #
        # IMPORTANT:
        # always nearest-neighbor for segmentation mask.
        # ----------------------------------------------------

        mask = cv2.resize(
            mask,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # Force mask back to exact 0/1
        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Image:
        # uint8 [0,255]
        # ->
        # float [0,1]
        # ----------------------------------------------------

        image = image_rgb.astype(
            np.float32
        )

        image = (
            image / 255.0
        )

        # ----------------------------------------------------
        # ImageNet normalization
        #
        # Needed by pretrained ResNet34.
        # ----------------------------------------------------

        image = (
            image
            -
            IMAGENET_MEAN
        ) / IMAGENET_STD

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # numpy -> torch
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).float()

        # [H,W] -> [1,H,W]
        mask = mask.unsqueeze(
            0
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        (
            text,
            sent_id

        ) = self.get_text(
            ref
        )

        # ----------------------------------------------------
        # Presence label
        #
        # target exists    -> 1
        # no-target        -> 0
        # ----------------------------------------------------

        no_target = bool(
            ref.get(
                "no_target",
                False
            )
        )

        presence = torch.tensor(

            0.0
            if no_target
            else 1.0,

            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        sample = {

            "image":
                image,

            "mask":
                mask,

            "text":
                text,

            "sent_id":
                sent_id,

            "presence":
                presence,

            "no_target":
                no_target,

            "image_id":
                image_id,

            "ref_id":
                ref.get(
                    "ref_id",
                    -1
                ),

            "file_name":
                filename,

            "original_height":
                original_height,

            "original_width":
                original_width
        }

        return sample
