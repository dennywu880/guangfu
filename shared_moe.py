"""Shared-image-encoder TG-UNet with three independently initialized heads.

The general checkpoint supplies the sole ResNet34 and frozen BERT. Expert
checkpoints supply only text gates and decoders. This is an initialization
for subsequent distillation, not an accuracy-preserving weight conversion.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34

from model_text import DecoderBlock, DoubleConv, TextGate, TextGuidedUNet


BRANCH_FILES = {
    "ultra_tiny": "ultra_expert_continued_best.pth",
    "tiny": "tiny_expert_continued_best.pth",
    "larger": "larger_expert_continued_best.pth",
}


class FeatureAdapter(nn.Module):
    """Small residual transforms between the shared encoder and one head."""

    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1)
            for channels in (64, 64, 128, 256, 512)
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, image_features):
        return tuple(
            feature + projection(feature)
            for feature, projection in zip(image_features, self.projections)
        )


class SharedImageEncoder(nn.Module):
    """One ResNet34 image encoder reused by all three specialist heads."""

    def __init__(self):
        super().__init__()
        backbone = resnet34(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def load_from_full_checkpoint(self, path):
        state = _state(path)
        expected = self.state_dict()
        missing = expected.keys() - state.keys()
        if missing:
            raise RuntimeError(f"Expert checkpoint is missing encoder keys: {sorted(missing)[:5]}")
        self.load_state_dict({key: state[key] for key in expected}, strict=True)

    def forward(self, image):
        e0 = self.stem(image)
        e1 = self.layer1(self.maxpool(e0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return (e0, e1, e2, e3, e4)


def _state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("model_state_dict", checkpoint)


class ExpertHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_gate_e2 = TextGate(text_dim=768, channels=128)
        self.text_gate_e3 = TextGate(text_dim=768, channels=256)
        self.text_gate_e4 = TextGate(text_dim=768, channels=512)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.final_conv = DoubleConv(64, 32)
        self.segmentation_head = nn.Conv2d(32, 1, kernel_size=1)

    def load_from_full_checkpoint(self, path):
        state = _state(path)
        expected = self.state_dict()
        missing = expected.keys() - state.keys()
        if missing:
            raise RuntimeError(f"Expert checkpoint is missing head keys: {sorted(missing)[:5]}")
        self.load_state_dict({key: state[key] for key in expected}, strict=True)


@dataclass
class SharedFeatures:
    image: tuple
    pooled_text: torch.Tensor
    input_size: tuple
    source_image: torch.Tensor | None = None


class SharedMoE(nn.Module):
    def __init__(self, general, heads):
        super().__init__()
        self.general = general
        self.heads = nn.ModuleDict(heads)
        self.adapters = nn.ModuleDict()
        self.expert_encoder = None

    @classmethod
    def from_checkpoints(cls, checkpoint_dir, device="cpu", general=None,
                         expert_backbone=None):
        checkpoint_dir = Path(checkpoint_dir)
        if general is None:
            general = TextGuidedUNet(pretrained=False)
            general.load_state_dict(
                _state(checkpoint_dir / "text_gate_epoch13_profile_fixed.pth"),
                strict=True,
            )
        heads = {}
        for branch, filename in BRANCH_FILES.items():
            head = ExpertHead()
            head.load_from_full_checkpoint(checkpoint_dir / filename)
            heads[branch] = head
        model = cls(general, heads)
        if expert_backbone is not None:
            model.enable_expert_encoder(
                checkpoint_dir / BRANCH_FILES[expert_backbone]
            )
        return model.to(device).eval()

    def train(self, mode=True):
        super().train(mode)
        self.general.text_encoder.eval()
        return self

    @staticmethod
    def _decode(head, features, image_features=None):
        e0, e1, e2, e3, e4 = features.image if image_features is None else image_features
        t = features.pooled_text
        x = head.decoder4(head.text_gate_e4(e4, t), head.text_gate_e3(e3, t))
        x = head.decoder3(x, head.text_gate_e2(e2, t))
        x = head.decoder2(x, e1)
        x = head.decoder1(x, e0)
        x = F.interpolate(x, size=features.input_size, mode="bilinear", align_corners=False)
        return head.segmentation_head(head.final_conv(x))

    def forward_general(self, images, texts=None, pooled_text=None):
        if (texts is None) == (pooled_text is None):
            raise ValueError("Provide exactly one of texts or pooled_text")
        if pooled_text is None:
            _, pooled_text, _ = self.general.text_encoder(texts, images.device)
        else:
            if (pooled_text.shape != (images.shape[0], 768)
                    or pooled_text.device != images.device
                    or pooled_text.dtype != torch.float32):
                raise ValueError("pooled_text must be float32 [batch, 768] on the image device")
        general = self.general
        e0 = general.stem(images)
        e1 = general.layer1(general.maxpool(e0))
        e2 = general.layer2(e1)
        e3 = general.layer3(e2)
        e4 = general.layer4(e3)
        context = SharedFeatures((e0, e1, e2, e3, e4), pooled_text,
                                 images.shape[-2:], images)
        return self._decode(general, context), context

    def forward_expert(self, context, branch):
        image_features = self.expert_features(context, branch)
        return self._decode(self.heads[branch], context, image_features)

    def enable_adapters(self):
        if self.adapters:
            return
        device = next(self.general.parameters()).device
        self.adapters = nn.ModuleDict({
            branch: FeatureAdapter().to(device) for branch in self.heads
        })

    def enable_expert_encoder(self, path):
        if self.expert_encoder is not None:
            raise RuntimeError("Expert encoder is already initialized")
        encoder = SharedImageEncoder()
        encoder.load_from_full_checkpoint(path)
        self.expert_encoder = encoder.to(next(self.general.parameters()).device)

    def expert_features(self, context, branch):
        image_features = context.image
        if self.expert_encoder is not None:
            if context.source_image is None:
                raise ValueError("The shared expert encoder requires the source image")
            image_features = self.expert_encoder(context.source_image)
        if branch in self.adapters:
            return self.adapters[branch](image_features)
        return image_features

    def forward(self, images, branch, texts=None, pooled_text=None):
        general_logits, context = self.forward_general(images, texts, pooled_text)
        return general_logits, self.forward_expert(context, branch)

    def save_heads(self, path):
        torch.save({name: head.state_dict() for name, head in self.heads.items()}, path)

    def load_heads(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if set(state) != set(self.heads):
            raise ValueError(f"Expected heads {list(self.heads)}, got {list(state)}")
        for name, head in self.heads.items():
            head.load_state_dict(state[name], strict=True)

    def save_repair(self, path):
        if set(self.adapters) != set(self.heads):
            raise RuntimeError("All adapters must be enabled before saving")
        torch.save({
            "format": "shared_moe_repair_v1",
            "heads": {name: head.state_dict() for name, head in self.heads.items()},
            "adapters": {name: adapter.state_dict() for name, adapter in self.adapters.items()},
        }, path)

    def load_repair(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "shared_moe_repair_v1":
            raise ValueError("Unrecognized shared MoE repair checkpoint")
        if set(state["heads"]) != set(self.heads) or set(state["adapters"]) != set(self.heads):
            raise ValueError("Repair checkpoint has incomplete expert branches")
        self.enable_adapters()
        for name in self.heads:
            self.heads[name].load_state_dict(state["heads"][name], strict=True)
            self.adapters[name].load_state_dict(state["adapters"][name], strict=True)
