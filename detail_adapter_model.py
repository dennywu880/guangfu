"""Shared TG-UNet with branch-specific high-resolution detail adapters.

This module is intentionally separate from ``shared_moe.py`` and is not
imported by the web application.  It keeps one image encoder, then applies a
small residual adapter to the e0/e1 features for the selected expert.  The
last pointwise convolution is zero-initialized, so the initial prediction is
the unmodified shared-encoder prediction and the experiment can be rolled
back by deleting its result directory.
"""

from pathlib import Path

import torch
from torch import nn

from model_text import TextGuidedUNet
from shared_moe import BRANCH_FILES, ExpertHead, SharedMoE, _state


class PooledTextOnlyEncoder(nn.Module):
    """Offline placeholder; all callers must pass the formal pooled cache."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bert = nn.Identity()

    def forward(self, *args, **kwargs):
        raise RuntimeError("This isolated experiment requires pooled_text cache")


def build_textless_unet(state=None, device="cpu"):
    """Construct TG-UNet without touching the unavailable tokenizer cache."""
    import model_text
    original = model_text.FrozenBERTEncoder
    model_text.FrozenBERTEncoder = PooledTextOnlyEncoder
    try:
        model = TextGuidedUNet(pretrained=False)
    finally:
        model_text.FrozenBERTEncoder = original
    if state is not None:
        image_state = {
            key: value for key, value in state.items()
            if not key.startswith("text_encoder.")
        }
        model.load_state_dict(image_state, strict=False)
    return model.to(device)


class DepthwiseDetailAdapter(nn.Module):
    """Residual high-resolution feature bridge for one expert.

    A depthwise 3x3 convolution captures local crack/edge patterns while the
    pointwise projection mixes channels.  Zeroing the final projection makes
    the adapter an identity at initialization instead of changing the
    baseline before it has learned anything.
    """

    def __init__(self, channels=64, groups=8):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups:
            groups -= 1
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False,
        )
        self.norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, feature):
        return feature + self.pointwise(
            self.activation(self.norm(self.depthwise(feature)))
        )


class DetailAdapterBranch(nn.Module):
    """Adapters for e0 (H/2) and e1 (H/4) of a single expert branch."""

    def __init__(self):
        super().__init__()
        self.e0 = DepthwiseDetailAdapter(64)
        self.e1 = DepthwiseDetailAdapter(64)

    def forward(self, features):
        e0, e1, e2, e3, e4 = features
        return self.e0(e0), self.e1(e1), e2, e3, e4


class DetailAdapterMoE(SharedMoE):
    """One shared encoder plus three lightweight shallow-detail branches."""

    def __init__(self, general, heads):
        super().__init__(general, heads)
        self.detail_adapters = nn.ModuleDict()

    @classmethod
    def from_checkpoints(cls, checkpoint_dir, device="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        general_state = _state(checkpoint_dir / "text_gate_epoch13_profile_fixed.pth")
        general = build_textless_unet(general_state, device="cpu")
        heads = {}
        for branch, filename in BRANCH_FILES.items():
            head = ExpertHead()
            head.load_from_full_checkpoint(checkpoint_dir / filename)
            heads[branch] = head
        model = cls(general, heads).to(device).eval()
        return model

    def enable_detail_adapters(self):
        if self.detail_adapters:
            return
        device = next(self.general.parameters()).device
        self.detail_adapters = nn.ModuleDict({
            branch: DetailAdapterBranch().to(device)
            for branch in BRANCH_FILES
        })

    def detail_features(self, context, branch):
        if branch not in BRANCH_FILES:
            raise KeyError(f"Unknown expert branch: {branch}")
        if branch not in self.detail_adapters:
            return context.image
        return self.detail_adapters[branch](context.image)

    def forward_expert(self, context, branch):
        image_features = self.detail_features(context, branch)
        return self._decode(self.heads[branch], context, image_features)

    def save_checkpoint(self, path, stage, base_metadata=None):
        """Save only mutable state; immutable base checkpoints stay untouched."""
        if set(self.detail_adapters) != set(BRANCH_FILES):
            raise RuntimeError("Enable all detail adapters before saving")
        shallow_prefixes = ("stem.", "layer1.", "layer2.")
        shallow = {
            key: value.detach().cpu()
            for key, value in self.general.state_dict().items()
            if key.startswith(shallow_prefixes)
        }
        data = {
            "format": "detail_adapter_moe_v1",
            "stage": stage,
            "base": base_metadata or {},
            "general_shallow": shallow,
            "heads": {
                name: {key: value.detach().cpu()
                       for key, value in head.state_dict().items()}
                for name, head in self.heads.items()
            },
            "detail_adapters": {
                name: {key: value.detach().cpu()
                       for key, value in adapter.state_dict().items()}
                for name, adapter in self.detail_adapters.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)

    def load_checkpoint(self, path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("format") != "detail_adapter_moe_v1":
            raise ValueError(f"Unsupported detail adapter checkpoint: {path}")
        self.enable_detail_adapters()
        self.general.load_state_dict(data["general_shallow"], strict=False)
        for name in self.heads:
            self.heads[name].load_state_dict(data["heads"][name], strict=True)
            self.detail_adapters[name].load_state_dict(
                data["detail_adapters"][name], strict=True
            )
        return data


def build_detail_model(checkpoint_dir, device="cpu"):
    """Build the isolated experiment from immutable original checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    model = DetailAdapterMoE.from_checkpoints(checkpoint_dir, device=device)
    model.enable_detail_adapters()
    return model
