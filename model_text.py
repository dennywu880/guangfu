import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet34,
    ResNet34_Weights
)

from transformers import (
    AutoTokenizer,
    AutoModel
)


# ============================================================
# Configuration
# ============================================================

BERT_NAME = "bert-base-uncased"
TEXT_MAX_LENGTH = 32
TEXT_DIM = 768


# ============================================================
# Basic U-Net convolution block
#
# IMPORTANT:
# Keep BatchNorm.
# This is the version already proven to train normally.
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-Net decoder block
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [
                x,
                skip
            ],
            dim=1
        )

        x = self.conv(
            x
        )

        return x


# ============================================================
# Frozen BERT text encoder
#
# Output:
#
# token_features:
# [B, L, 768]
#
# pooled_text:
# [B, 768]
#
# attention_mask:
# [B, L]
#
# We use MASKED MEAN POOLING instead of only [CLS].
# ============================================================

class FrozenBERTEncoder(nn.Module):

    def __init__(
        self,
        model_name=BERT_NAME,
        max_length=TEXT_MAX_LENGTH
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length

        print(
            "\nLoading BERT tokenizer..."
        )

        # ----------------------------------------------------
        # Strictly use local BERT files.
        #
        # Training should not depend on network connection.
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )

        print(
            "Loading frozen BERT encoder..."
        )

        self.bert = AutoModel.from_pretrained(
            model_name,
            local_files_only=True
        )

        # ----------------------------------------------------
        # Freeze every BERT parameter
        # ----------------------------------------------------

        for parameter in self.bert.parameters():

            parameter.requires_grad = False

        # ----------------------------------------------------
        # Frozen BERT must never use Dropout
        # ----------------------------------------------------

        self.bert.eval()

        print(
            "BERT loaded and frozen."
        )

    # ========================================================
    # Keep BERT in eval mode even when whole model calls
    # model.train().
    # ========================================================

    def train(
        self,
        mode=True
    ):

        super().train(
            mode
        )

        # BERT itself must always stay in eval mode.
        self.bert.eval()

        return self

    # ========================================================
    # Original ONLINE text forward
    #
    # IMPORTANT:
    # This is the path used in Epoch 1-3.
    # Do not change its mathematical logic.
    # ========================================================

    def forward(
            self,
            texts,
            device
    ):

        # ----------------------------------------------------
        # Tokenize batch of strings
        #
        # IMPORTANT:
        # Use fixed-length padding so the text feature does NOT
        # depend on what other sentences happen to be in batch.
        # ----------------------------------------------------

        encoded = self.tokenizer(
            list(texts),

            padding="max_length",

            truncation=True,

            max_length=self.max_length,

            return_tensors="pt"
        )

        input_ids = (
            encoded[
                "input_ids"
            ].to(
                device
            )
        )

        attention_mask = (
            encoded[
                "attention_mask"
            ].to(
                device
            )
        )

        # ----------------------------------------------------
        # Frozen BERT forward
        #
        # IMPORTANT:
        # Force FP32 even when outer training uses AMP.
        # This keeps:
        #
        # online train
        # online eval
        # cached feature generation
        #
        # all numerically aligned.
        # ----------------------------------------------------

        self.bert.eval()

        if device.type == "cuda":

            with torch.no_grad():

                with torch.amp.autocast(
                        device_type="cuda",
                        enabled=False
                ):
                    outputs = self.bert(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    token_features = (
                        outputs.last_hidden_state
                    )

                    # ----------------------------------------
                    # Masked mean pooling
                    # ----------------------------------------

                    mask = (
                        attention_mask
                        .unsqueeze(-1)
                        .float()
                    )

                    summed_features = (
                            token_features
                            *
                            mask
                    ).sum(
                        dim=1
                    )

                    valid_token_count = (
                        mask.sum(
                            dim=1
                        )
                        .clamp(
                            min=1.0
                        )
                    )

                    pooled_text = (
                            summed_features
                            /
                            valid_token_count
                    )

        else:

            with torch.no_grad():

                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                token_features = (
                    outputs.last_hidden_state
                )

                mask = (
                    attention_mask
                    .unsqueeze(-1)
                    .float()
                )

                summed_features = (
                        token_features
                        *
                        mask
                ).sum(
                    dim=1
                )

                valid_token_count = (
                    mask.sum(
                        dim=1
                    )
                    .clamp(
                        min=1.0
                    )
                )

                pooled_text = (
                        summed_features
                        /
                        valid_token_count
                )

        return (
            token_features,
            pooled_text,
            attention_mask
        )


# ============================================================
# Text Channel Gate
#
# Text:
# [B, 768]
#
# Visual:
# [B, C, H, W]
#
# Gate:
# [B, C, 1, 1]
#
# Residual multiplicative modulation:
#
# output = visual * (1 + delta)
#
# delta starts near zero.
# ============================================================

class TextGate(nn.Module):

    def __init__(
        self,
        text_dim,
        channels
    ):
        super().__init__()

        self.projection = nn.Linear(
            text_dim,
            channels
        )

        # ----------------------------------------------------
        # Small initialization
        #
        # Keeps initial network close to identity while still
        # allowing text gradients to flow into the Text Gate.
        # ----------------------------------------------------

        nn.init.normal_(
            self.projection.weight,
            mean=0.0,
            std=1e-4
        )

        nn.init.zeros_(
            self.projection.bias
        )

    def forward(
        self,
        visual_feature,
        text_feature
    ):

        delta = self.projection(
            text_feature
        )

        delta = torch.tanh(
            delta
        )

        delta = (
            delta
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        gated_feature = (
            visual_feature
            *
            (
                1.0
                +
                delta
            )
        )

        return gated_feature


# ============================================================
# TG_Unet Stage 1
#
# ResNet34
# +
# Frozen BERT
# +
# Text Gates
# +
# U-Net Decoder
#
# No Cross Attention yet.
# No Presence Head yet.
# ============================================================

class TextGuidedUNet(nn.Module):

    def __init__(
        self,
        pretrained=True,
        bert_name=BERT_NAME
    ):
        super().__init__()

        # ====================================================
        # Text encoder
        # ====================================================

        self.text_encoder = FrozenBERTEncoder(
            model_name=bert_name,
            max_length=TEXT_MAX_LENGTH
        )

        # ====================================================
        # Image encoder
        # ====================================================

        if pretrained:

            weights = (
                ResNet34_Weights.DEFAULT
            )

        else:

            weights = None

        backbone = resnet34(
            weights=weights
        )

        # ----------------------------------------------------
        # H / 2
        # ----------------------------------------------------

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.maxpool = (
            backbone.maxpool
        )

        # ----------------------------------------------------
        # H / 4
        # ----------------------------------------------------

        self.layer1 = (
            backbone.layer1
        )

        # ----------------------------------------------------
        # H / 8
        # ----------------------------------------------------

        self.layer2 = (
            backbone.layer2
        )

        # ----------------------------------------------------
        # H / 16
        # ----------------------------------------------------

        self.layer3 = (
            backbone.layer3
        )

        # ----------------------------------------------------
        # H / 32
        # ----------------------------------------------------

        self.layer4 = (
            backbone.layer4
        )

        # ====================================================
        # Text Gates
        #
        # Only gate e2 / e3 / e4.
        #
        # e0 and e1 remain purely visual.
        # ====================================================

        self.text_gate_e2 = TextGate(
            text_dim=TEXT_DIM,
            channels=128
        )

        self.text_gate_e3 = TextGate(
            text_dim=TEXT_DIM,
            channels=256
        )

        self.text_gate_e4 = TextGate(
            text_dim=TEXT_DIM,
            channels=512
        )

        # ====================================================
        # U-Net decoder
        # ====================================================

        self.decoder4 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=64
        )

        self.final_conv = DoubleConv(
            64,
            32
        )

        self.segmentation_head = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    # ========================================================
    # Keep BERT frozen and in eval mode
    # ========================================================

    def train(
        self,
        mode=True
    ):

        super().train(
            mode
        )

        # ----------------------------------------------------
        # Image Encoder / Decoder / Text Gates follow mode.
        #
        # Frozen BERT must always remain eval.
        # ----------------------------------------------------

        self.text_encoder.bert.eval()

        return self

    # ========================================================
    # Forward
    #
    # Two mutually exclusive text input modes:
    #
    # ONLINE:
    #
    # model(images, texts)
    #
    # CACHED:
    #
    # model(
    #     images,
    #     pooled_text=cached_feature
    # )
    #
    # Everything after pooled_text is identical.
    # ========================================================

    def forward(
        self,
        images,
        texts=None,
        return_debug=False,
        pooled_text=None
    ):

        input_size = (
            images.shape[-2:]
        )

        device = (
            images.device
        )

        # ====================================================
        # Validate text input source
        # ====================================================

        if (
            texts is None
            and
            pooled_text is None
        ):

            raise ValueError(
                "TextGuidedUNet requires either "
                "'texts' or 'pooled_text'."
            )

        if (
            texts is not None
            and
            pooled_text is not None
        ):

            raise ValueError(
                "Provide only ONE text source: "
                "'texts' OR 'pooled_text', not both."
            )

        # ====================================================
        # ONLINE text path
        #
        # This is the original Epoch 1-3 behavior.
        # ====================================================

        if pooled_text is None:

            (
                token_features,
                pooled_text,
                attention_mask

            ) = self.text_encoder(
                texts,
                device
            )

            text_source = "online"

        # ====================================================
        # CACHED text path
        #
        # Only pure pooled BERT features enter here.
        #
        # Fail closed:
        # do not silently repair invalid cache.
        # ====================================================

        else:

            text_source = "cache"

            token_features = None
            attention_mask = None

            # ------------------------------------------------
            # Type check
            # ------------------------------------------------

            if not torch.is_tensor(
                pooled_text
            ):

                raise TypeError(
                    "Cached pooled_text must be "
                    "a torch.Tensor."
                )

            # ------------------------------------------------
            # Shape check
            # ------------------------------------------------

            if pooled_text.ndim != 2:

                raise ValueError(
                    "Cached pooled_text must have "
                    "shape [B, 768]. "
                    f"Got {tuple(pooled_text.shape)}"
                )

            if (
                pooled_text.shape[0]
                !=
                images.shape[0]
            ):

                raise ValueError(
                    "Cached pooled_text batch size "
                    "does not match image batch size. "
                    f"Images={images.shape[0]}, "
                    f"Text={pooled_text.shape[0]}"
                )

            if (
                pooled_text.shape[1]
                !=
                TEXT_DIM
            ):

                raise ValueError(
                    "Cached pooled_text feature "
                    f"dimension must be {TEXT_DIM}. "
                    f"Got {pooled_text.shape[1]}"
                )

            # ------------------------------------------------
            # Device check
            #
            # We deliberately do NOT silently call .to(device).
            # ------------------------------------------------

            if (
                pooled_text.device
                !=
                device
            ):

                raise ValueError(
                    "Cached pooled_text must already "
                    "be on the same device as images. "
                    f"Image device={device}, "
                    f"Text device={pooled_text.device}"
                )

            # ------------------------------------------------
            # Dtype check
            #
            # Current cache V1 is strictly FP32.
            # ------------------------------------------------

            if (
                pooled_text.dtype
                !=
                torch.float32
            ):

                raise ValueError(
                    "Current TG_Unet cache requires "
                    "FP32 pooled_text. "
                    f"Got dtype={pooled_text.dtype}"
                )

            # ------------------------------------------------
            # Cached encoder output must not carry gradients.
            # ------------------------------------------------

            if pooled_text.requires_grad:

                raise ValueError(
                    "Cached pooled_text must not "
                    "require gradients."
                )

            # ------------------------------------------------
            # Numerical validity
            # ------------------------------------------------

            if not torch.isfinite(
                pooled_text
            ).all():

                raise ValueError(
                    "Cached pooled_text contains "
                    "NaN or Inf."
                )

        # ====================================================
        # Image encoder
        #
        # IMPORTANT:
        # This is NEVER cached at the current stage because
        # the ResNet image encoder is still trainable.
        # ====================================================

        # ----------------------------------------------------
        # H / 2
        # ----------------------------------------------------

        e0 = self.stem(
            images
        )

        # ----------------------------------------------------
        # H / 4
        # ----------------------------------------------------

        x = self.maxpool(
            e0
        )

        e1 = self.layer1(
            x
        )

        # ----------------------------------------------------
        # H / 8
        # ----------------------------------------------------

        e2 = self.layer2(
            e1
        )

        # ----------------------------------------------------
        # H / 16
        # ----------------------------------------------------

        e3 = self.layer3(
            e2
        )

        # ----------------------------------------------------
        # H / 32
        # ----------------------------------------------------

        e4 = self.layer4(
            e3
        )

        # ====================================================
        # Text modulation
        #
        # ONLINE and CACHE paths join here.
        #
        # Everything below this point is shared.
        # ====================================================

        e2_text = self.text_gate_e2(
            e2,
            pooled_text
        )

        e3_text = self.text_gate_e3(
            e3,
            pooled_text
        )

        e4_text = self.text_gate_e4(
            e4,
            pooled_text
        )

        # ====================================================
        # Decoder
        # ====================================================

        d4 = self.decoder4(
            e4_text,
            e3_text
        )

        d3 = self.decoder3(
            d4,
            e2_text
        )

        d2 = self.decoder2(
            d3,
            e1
        )

        d1 = self.decoder1(
            d2,
            e0
        )

        # ====================================================
        # Restore original resolution
        # ====================================================

        x = F.interpolate(
            d1,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        x = self.final_conv(
            x
        )

        logits = (
            self.segmentation_head(
                x
            )
        )

        # ====================================================
        # Debug output
        # ====================================================

        if return_debug:

            return {

                "logits":
                    logits,

                "pooled_text":
                    pooled_text,

                "token_features":
                    token_features,

                "attention_mask":
                    attention_mask,

                "text_source":
                    text_source
            }

        return logits


# ============================================================
# Simple forward test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n")
    print("=" * 70)
    print("TG_Unet TEXT-GUIDED U-NET FORWARD CHECK")
    print("=" * 70)

    print(
        "\nDevice:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    # ========================================================
    # Model
    # ========================================================

    model = TextGuidedUNet(
        pretrained=True
    ).to(
        device
    )

    model.eval()

    # ========================================================
    # Fake images
    # ========================================================

    images = torch.randn(
        2,
        3,
        384,
        384,
        device=device
    )

    texts = [

        (
            "The Finger Interruption type Contact defect "
            "not connected to busbar with little size "
            "in EL image"
        ),

        (
            "All Contact defects in the EL image"
        )
    ]

    # ========================================================
    # ONLINE forward
    # ========================================================

    with torch.no_grad():

        online_outputs = model(
            images,
            texts,
            return_debug=True
        )

    online_logits = (
        online_outputs[
            "logits"
        ]
    )

    pooled_text = (
        online_outputs[
            "pooled_text"
        ]
    )

    token_features = (
        online_outputs[
            "token_features"
        ]
    )

    attention_mask = (
        online_outputs[
            "attention_mask"
        ]
    )

    # ========================================================
    # CACHED-input forward
    #
    # Here we reuse the ONLINE pooled feature directly.
    # This is only a structural smoke test.
    #
    # Strict disk-cache verification comes later.
    # ========================================================

    with torch.no_grad():

        cached_outputs = model(
            images,
            pooled_text=(
                pooled_text
                .detach()
                .float()
            ),
            return_debug=True
        )

    cached_logits = (
        cached_outputs[
            "logits"
        ]
    )

    # ========================================================
    # Print shapes
    # ========================================================

    print(
        "\nImage shape:"
    )

    print(
        images.shape
    )

    print(
        "\nMask logits shape:"
    )

    print(
        online_logits.shape
    )

    print(
        "\nPooled text shape:"
    )

    print(
        pooled_text.shape
    )

    print(
        "\nToken features shape:"
    )

    print(
        token_features.shape
    )

    print(
        "\nAttention mask shape:"
    )

    print(
        attention_mask.shape
    )

    print(
        "\nONLINE text source:"
    )

    print(
        online_outputs[
            "text_source"
        ]
    )

    print(
        "\nCACHE text source:"
    )

    print(
        cached_outputs[
            "text_source"
        ]
    )

    # ========================================================
    # Parameter statistics
    # ========================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    bert_total = sum(
        p.numel()
        for p in model.text_encoder.bert.parameters()
    )

    bert_trainable = sum(
        p.numel()
        for p in model.text_encoder.bert.parameters()
        if p.requires_grad
    )

    print(
        "\nParameters:"
    )

    print(
        f"Total model     : "
        f"{total_params / 1e6:.2f} M"
    )

    print(
        f"Trainable model : "
        f"{trainable_params / 1e6:.2f} M"
    )

    print(
        f"BERT total      : "
        f"{bert_total / 1e6:.2f} M"
    )

    print(
        f"BERT trainable  : "
        f"{bert_trainable / 1e6:.2f} M"
    )

    # ========================================================
    # NaN / Inf check
    # ========================================================

    print(
        "\nOutput diagnostics:"
    )

    print(
        "ONLINE NaN:",
        torch.isnan(
            online_logits
        ).any().item()
    )

    print(
        "ONLINE Inf:",
        torch.isinf(
            online_logits
        ).any().item()
    )

    print(
        "CACHE NaN:",
        torch.isnan(
            cached_logits
        ).any().item()
    )

    print(
        "CACHE Inf:",
        torch.isinf(
            cached_logits
        ).any().item()
    )

    print(
        "Logit min:",
        online_logits.min().item()
    )

    print(
        "Logit max:",
        online_logits.max().item()
    )

    # ========================================================
    # ONLINE vs direct cached-input structural comparison
    #
    # Same pooled tensor -> logits should be exactly equal
    # or extremely close.
    # ========================================================

    logit_difference = (
        online_logits
        -
        cached_logits
    ).abs()

    max_logit_error = (
        logit_difference
        .max()
        .item()
    )

    mean_logit_error = (
        logit_difference
        .mean()
        .item()
    )

    online_probability = (
        torch.sigmoid(
            online_logits
        )
    )

    cached_probability = (
        torch.sigmoid(
            cached_logits
        )
    )

    probability_difference = (
        online_probability
        -
        cached_probability
    ).abs()

    max_probability_error = (
        probability_difference
        .max()
        .item()
    )

    online_binary = (
        online_probability
        >
        0.5
    )

    cached_binary = (
        cached_probability
        >
        0.5
    )

    binary_equal = (
        torch.equal(
            online_binary,
            cached_binary
        )
    )

    print(
        "\nONLINE vs CACHE structural check:"
    )

    print(
        f"Max logit error : "
        f"{max_logit_error:.12e}"
    )

    print(
        f"Mean logit error: "
        f"{mean_logit_error:.12e}"
    )

    print(
        f"Max prob error  : "
        f"{max_probability_error:.12e}"
    )

    print(
        f"Binary masks equal: "
        f"{binary_equal}"
    )

    # ========================================================
    # Final check
    # ========================================================

    print("\n")
    print("=" * 70)

    passed = (

        online_logits.shape
        ==
        torch.Size(
            [
                2,
                1,
                384,
                384
            ]
        )

        and

        cached_logits.shape
        ==
        online_logits.shape

        and

        not torch.isnan(
            online_logits
        ).any()

        and

        not torch.isinf(
            online_logits
        ).any()

        and

        not torch.isnan(
            cached_logits
        ).any()

        and

        not torch.isinf(
            cached_logits
        ).any()

        and

        bert_trainable
        ==
        0

        and

        binary_equal
    )

    if passed:

        print(
            "TEXT MODEL CHECK PASSED"
        )

    else:

        print(
            "TEXT MODEL CHECK FAILED"
        )

    print("=" * 70)
