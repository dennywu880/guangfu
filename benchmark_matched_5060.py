"""Matched cached-text inference audit for Final V3 and shared candidates.

Run each mode in a fresh Python process to keep model-resident CUDA memory
separate. BERT parameters are counted once, but online BERT compute is not
included: both modes receive the same verified formal pooled-text feature.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from torch import nn
from thop import profile

from dataset import RefELDataset
from detail_adapter_model import build_textless_unet
from multilevel_selection import build_model
from train_multilevel_head_joint import load as load_joint
from train_text import FormalTextCache


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "shared_testB_comparison_20260925"
CANDIDATES = {
    "head_joint_best": ROOT / "results/head_joint_e3_e4_20260925/best.pth",
    "distill_dev_ciou_best": ROOT / "results/selective_distill_e3_e4_20260925/best.pth",
}


def original_protocol():
    spec = importlib.util.spec_from_file_location(
        "frozen_testb_protocol", ROOT / "43_final_testB_gate_v2_and_m2_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


class OriginalCachedPath(nn.Module):
    def __init__(self, general, expert):
        super().__init__()
        self.general = general
        self.expert = expert

    def forward(self, image, pooled):
        return self.general(image, pooled_text=pooled), self.expert(image, pooled_text=pooled)


class SharedCachedPath(nn.Module):
    def __init__(self, model, branch):
        super().__init__()
        self.model = model
        self.branch = branch

    def forward(self, image, pooled):
        general_logits, context = self.model.forward_general(image, pooled_text=pooled)
        return general_logits, self.model.forward_expert(context, self.branch)


def general_stats(logits):
    probs = logits.sigmoid()
    flat = probs.flatten(1)
    mean = flat.mean(1)
    std = flat.std(1, unbiased=False)
    maxv = flat.max(1).values
    q95 = torch.quantile(flat, 0.95, dim=1)
    q99 = torch.quantile(flat, 0.99, dim=1)
    p = probs.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log()).flatten(1).mean(1)
    area = [(flat >= threshold).float().mean(1) for threshold in (.1, .3, .5, .7, .9)]
    fg = (flat >= .5).float()
    fg_count = fg.sum(1)
    fg_mean = torch.where(fg_count > 0, (flat * fg).sum(1) / fg_count.clamp_min(1),
                          torch.zeros_like(fg_count))
    return torch.stack([mean, std, maxv, q95, q99, entropy, *area, fg_mean], dim=1), area[2]


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("RTX 5060 CUDA is required for this audit")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    protocol = original_protocol()
    dataset = RefELDataset(ROOT, split="testB", image_size=384, random_text=False)
    sample = dataset[args.sample_index]
    image = sample["image"].unsqueeze(0).to(device)
    cache = FormalTextCache(ROOT)
    batch = {
        "sent_id": torch.tensor([sample["sent_id"]]),
        "ref_id": torch.tensor([sample["ref_id"]]),
        "text": [sample["text"]],
    }
    pooled = cache.get_batch(batch, expected_split="testB", device=device)
    gate, gate_mean, gate_std, hook_name, threshold = protocol.load_presence_gate(device)
    branch = args.branch

    if args.mode == "original":
        general_state = torch.load(protocol.GENERAL_CHECKPOINT, map_location="cpu", weights_only=False)
        general = build_textless_unet(general_state.get("model_state_dict", general_state), device)
        expert_state = torch.load(protocol.EXPERT_CHECKPOINTS[branch], map_location="cpu", weights_only=False)
        expert = build_textless_unet(expert_state.get("model_state_dict", expert_state), device)
        path = OriginalCachedPath(general, expert).eval()
        active_without_bert_gate = parameter_count(general) + parameter_count(expert)
        deployed_parameters = 211140101
    else:
        model = build_model(device, ("e3", "e4"))
        load_joint(CANDIDATES[args.mode], model)
        model.eval()
        path = SharedCachedPath(model, branch).eval()
        general = model.base.general
        active_without_bert_gate = (parameter_count(general)
                                    + parameter_count(model.base.heads[branch])
                                    + parameter_count(model.base.detail_adapters[branch])
                                    + parameter_count(model.selections[branch]))
        deployed_parameters = parameter_count(model) + 109482240 + parameter_count(gate)

    hook = protocol.FeatureHook(dict(general.named_modules())[hook_name])
    mean = torch.as_tensor(gate_mean, device=device).view(1, -1)
    std = torch.as_tensor(gate_std, device=device).view(1, -1)
    branch_onehot = torch.tensor([protocol.route_onehot(branch)], device=device)
    _, _, text_size = protocol.route_sample(sample["text"], 0.0)
    cue_onehot = torch.tensor([protocol.text_onehot(text_size)], device=device)

    def forward_full():
        hook.value = None
        general_logits, expert_logits = path(image, pooled)
        deep = hook.value
        image_feature = deep.mean((2, 3)) if deep.ndim == 4 else deep.flatten(1)
        stats, area050 = general_stats(general_logits)
        route, _, text_cue = protocol.route_sample(sample["text"], float(area050[0].item()))
        if route != branch or text_cue != text_size:
            raise RuntimeError(f"Sample routed to {route}, expected {branch}")
        gate_input = torch.cat([image_feature, pooled, stats, branch_onehot, cue_onehot], dim=1)
        no_target_probability = gate((gate_input - mean) / std).sigmoid()
        return expert_logits, no_target_probability

    # THOP covers the image path and the Presence Gate; online BERT is excluded.
    image_flops, _ = profile(path, inputs=(image, pooled), verbose=False)
    gate_flops, _ = profile(gate, inputs=(torch.zeros(1, gate.net[0].in_features, device=device),),
                            verbose=False)

    forward_full()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        forward_full()
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    start = time.perf_counter()
    for _ in range(args.iterations):
        forward_full()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    hook.close()
    result = {
        "mode": args.mode,
        "checkpoint": str(CANDIDATES[args.mode]) if args.mode != "original" else str(protocol.EXPERT_CHECKPOINTS[branch]),
        "gpu": torch.cuda.get_device_name(0),
        "sample_index": args.sample_index,
        "ref_id": int(sample["ref_id"]),
        "text": sample["text"],
        "branch": branch,
        "input": "384x384 FP32 batch=1",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "deployed_parameters_with_one_bert": deployed_parameters,
        "active_parameters_with_one_bert_and_gate": active_without_bert_gate + 109482240 + parameter_count(gate),
        "FLOPs_G_cached_text_path": (image_flops + gate_flops) / 1e9,
        "peak_allocated_gpu_MiB": peak_mib,
        "latency_ms_cached_text_path": elapsed * 1000 / args.iterations,
        "FPS_cached_text_path": args.iterations / elapsed,
        "exclusions": ["online BERT/tokenizer", "image file reading", "UI packaging"],
        "memory_residency": "general + gate + active expert only" if args.mode == "original"
                           else "general + gate + all three expert heads/adapters",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT / f"matched_5060_{args.mode}.json"
    filename.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", *CANDIDATES], required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--branch", choices=["ultra_tiny", "tiny", "larger"], default="ultra_tiny")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    run(parser.parse_args())
