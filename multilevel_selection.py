"""Isolated, gated expert feature selection on one shared image encoder.

This is an MTAN-inspired ablation, not a reproduction of MTAN or PLE. The
production application never imports this module.
"""

import hashlib
from pathlib import Path

import torch
from torch import nn

from detail_adapter_model import BRANCH_FILES, build_detail_model
from feature_bridge_experiment import BASE_CHECKPOINT
from train_detail_adapter import ROOT


LEVEL_CHANNELS = {"e1": 64, "e2": 128, "e3": 256, "e4": 512}


class GatedExpertFeature(nn.Module):
    """Spatial gate selects a low-cost expert residual from a shared feature."""

    def __init__(self, channels):
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)

    def forward(self, feature):
        return feature + self.gate(feature) * self.detail(feature)


class MultiLevelSelection(nn.Module):
    def __init__(self, base, levels):
        super().__init__()
        self.base = base
        self.levels = tuple(levels)
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("Levels must be distinct and nonempty")
        if any(level not in LEVEL_CHANNELS for level in self.levels):
            raise ValueError(f"Unsupported levels: {self.levels}")
        self.selections = nn.ModuleDict({
            branch: nn.ModuleDict({
                level: GatedExpertFeature(LEVEL_CHANNELS[level])
                for level in self.levels
            })
            for branch in BRANCH_FILES
        })

    def forward_general(self, *args, **kwargs):
        return self.base.forward_general(*args, **kwargs)

    def forward_expert(self, context, branch):
        features = list(self.base.detail_features(context, branch))
        for level in self.levels:
            features[int(level[1:])] = self.selections[branch][level](
                features[int(level[1:])])
        return self.base._decode(self.base.heads[branch], context, features)

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        for parameter in self.selections.parameters():
            parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def save_checkpoint(self, path, metadata):
        torch.save({
            "format": "multilevel_selection_v1",
            "levels": self.levels,
            "base_checkpoint": str(BASE_CHECKPOINT),
            "base_sha256": hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest(),
            "metadata": metadata,
            "selections": {
                branch: {level: module.state_dict()
                         for level, module in level_modules.items()}
                for branch, level_modules in self.selections.items()
            },
        }, path)

    def load_checkpoint(self, path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("format") != "multilevel_selection_v1":
            raise ValueError("Unknown multilevel selection checkpoint")
        if tuple(data["levels"]) != self.levels:
            raise ValueError("Checkpoint levels differ from model levels")
        saved_hash = data.get("base_sha256", data["metadata"].get("base_sha256"))
        if saved_hash != hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest():
            raise ValueError("Base checkpoint changed; refusing an incompatible rollback")
        for branch, levels in self.selections.items():
            for level, module in levels.items():
                module.load_state_dict(data["selections"][branch][level])
        return data["metadata"]


def build_model(device, levels):
    if not Path(BASE_CHECKPOINT).is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)
    base = build_detail_model(ROOT / "checkpoints", device)
    base.load_checkpoint(BASE_CHECKPOINT)
    model = MultiLevelSelection(base, levels).to(device)
    model.freeze_base()
    return model
