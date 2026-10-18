"""One shared shallow encoder with trainable expert-specific e3/e4 blocks."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from multilevel_selection import build_model
from train_multilevel_head_joint import load as load_head_joint
from train_e3e4_budget import SOURCE


class DeepBranchSelection(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        general = shared.base.general
        self.branches = nn.ModuleDict({
            branch: nn.ModuleDict({
                "e3": deepcopy(general.layer3),
                "e4": deepcopy(general.layer4),
            })
            for branch in shared.base.heads
        })

    def forward_general(self, *args, **kwargs):
        return self.shared.forward_general(*args, **kwargs)

    def forward_expert(self, context, branch):
        e0, e1, e2, _, _ = self.shared.base.detail_features(context, branch)
        e3 = self.branches[branch]["e3"](e2)
        e4 = self.branches[branch]["e4"](e3)
        e3 = self.shared.selections[branch]["e3"](e3)
        e4 = self.shared.selections[branch]["e4"](e4)
        return self.shared.base._decode(
            self.shared.base.heads[branch], context, (e0, e1, e2, e3, e4)
        )


def build_deep_branch(device):
    shared = build_model(device, ("e3", "e4"))
    load_head_joint(SOURCE, shared)
    return DeepBranchSelection(shared).to(device)
