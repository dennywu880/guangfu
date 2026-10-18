from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from dataset import RefELDataset
from model_text import TextGuidedUNet
from evaluate_decoder_sharing_testB import load_candidate
from train_text import FormalTextCache


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "metric_recheck_20261014"
BRANCH = "tiny"
WARMUP = 20
ITERATIONS = 100
REPEATS = 5


def load_protocol():
    spec = importlib.util.spec_from_file_location(
        "metric_recheck_protocol", ROOT / "43_final_testB_gate_v2_and_m2_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def general_stats(logits: torch.Tensor):
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
    fg_mean = torch.where(
        fg_count > 0,
        (flat * fg).sum(1) / fg_count.clamp_min(1),
        torch.zeros_like(fg_count),
    )
    values = torch.stack([mean, std, maxv, q95, q99, entropy, *area, fg_mean], dim=1)
    return values, area[2]


class InferencePath(nn.Module):
    def __init__(self, general, expert_forward, gate, mean, std, hook, protocol):
        super().__init__()
        self.general = general
        self.expert_forward = expert_forward
        self.gate = gate
        self.register_buffer("gate_mean", torch.as_tensor(mean).view(1, -1))
        self.register_buffer("gate_std", torch.as_tensor(std).view(1, -1))
        self.hook = hook
        self.protocol = protocol
        self.branch = BRANCH

    def general_expert(self, image, pooled):
        self.hook.value = None
        if hasattr(self.general, "forward_general"):
            general_logits, context = self.general.forward_general(image, pooled_text=pooled)
        else:
            general_logits, context = self.general(image, pooled_text=pooled)
        expert_logits = self.expert_forward(context, self.branch)
        return general_logits, expert_logits

    def forward(self, image, pooled, text):
        self.hook.value = None
        if hasattr(self.general, "forward_general"):
            general_logits, context = self.general.forward_general(image, pooled_text=pooled)
        else:
            general_logits, context = self.general(image, pooled_text=pooled)
        deep = self.hook.value
        image_feature = deep.mean((2, 3)) if deep.ndim == 4 else deep.flatten(1)
        stats, area050 = general_stats(general_logits)
        branch, _, text_size = self.protocol.route_sample(text, float(area050[0].item()))
        if branch != self.branch:
            raise RuntimeError(f"Selected fixed test branch {self.branch}, got {branch}")
        route = torch.tensor([self.protocol.route_onehot(branch)], device=image.device)
        cue = torch.tensor([self.protocol.text_onehot(text_size)], device=image.device)
        gate_x = torch.cat([image_feature, pooled, stats, route, cue], dim=1)
        gate_probability = torch.sigmoid(
            self.gate((gate_x - self.gate_mean) / self.gate_std)
        )
        expert_logits = self.expert_forward(context, branch)
        return general_logits, expert_logits, gate_probability


def make_path(mode: str, device, protocol):
    gate, mean, std, hook_name, _ = protocol.load_presence_gate(device)
    if mode in {"private", "decoder4"}:
        candidate, checkpoint = load_candidate(mode, device)
        general = candidate
        expert_forward = candidate.forward_expert
        general_module = candidate.base.general
        total_parameters = candidate.parameter_counts()[
            "total_image_model_parameters_cached_text_excluded"
        ]
        active_parameters = candidate.parameter_counts()[
            "active_one_route_image_model_parameters_cached_text_excluded"
        ]
        checkpoint_name = str(checkpoint)
    else:
        from transformers import BertConfig, BertModel
        import model_text

        class OfflineBertShell(nn.Module):
            def __init__(self, model_name=model_text.BERT_NAME,
                         max_length=model_text.TEXT_MAX_LENGTH):
                super().__init__()
                self.model_name = model_name
                self.max_length = max_length
                config = BertConfig(
                    vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072,
                    max_position_embeddings=512,
                )
                self.bert = BertModel(config)

            def forward(self, *args, **kwargs):
                raise RuntimeError("Cached pooled text is required in this benchmark")

        original_encoder = model_text.FrozenBERTEncoder
        model_text.FrozenBERTEncoder = OfflineBertShell
        try:
            general_module = protocol.load_seg_model(protocol.GENERAL_CHECKPOINT, device)
            experts = {}
            for branch, path in protocol.EXPERT_CHECKPOINTS.items():
                expert = protocol.load_seg_model(path, "cpu")
                expert.text_encoder.bert = nn.Identity()
                expert.eval()
                experts[branch] = expert
        finally:
            model_text.FrozenBERTEncoder = original_encoder
        active_expert = experts[BRANCH].to(device).eval()

        class OriginalPath(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.general_model = model

            def forward(self, image, pooled_text):
                debug = self.general_model(
                    image, pooled_text=pooled_text, return_debug=True
                )
                return debug["logits"], SimpleNamespace(
                    source_image=image, pooled_text=debug["pooled_text"]
                )

        general = OriginalPath(general_module).eval()

        expert_forward = lambda context, branch: active_expert(
            context.source_image, pooled_text=context.pooled_text
        )
        general = OriginalPath(general_module).eval()
        total_parameters = parameter_count(general_module) + sum(
            parameter_count(expert) for expert in experts.values()
        )
        active_parameters = parameter_count(general_module) + parameter_count(active_expert)
        checkpoint_name = str(protocol.GENERAL_CHECKPOINT)

    hook_module = dict(general_module.named_modules())[hook_name]
    hook = protocol.FeatureHook(hook_module)
    path = InferencePath(general, expert_forward, gate, mean, std, hook, protocol).to(device).eval()
    if mode in {"private", "decoder4"}:
        deployed_parameters = total_parameters + 109_482_240 + parameter_count(gate)
        active_parameters = active_parameters + 109_482_240 + parameter_count(gate)
    else:
        deployed_parameters = total_parameters + parameter_count(gate)
        active_parameters = active_parameters + parameter_count(gate)
    return path, deployed_parameters, active_parameters, checkpoint_name


@torch.inference_mode()
def run(mode: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this recheck")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    protocol = load_protocol()
    dataset = RefELDataset(ROOT, "testB", protocol.IMAGE_SIZE, False)
    item = dataset[0]
    image = item["image"].unsqueeze(0).to(device)
    cache = FormalTextCache(ROOT)
    batch = {
        "sent_id": torch.tensor([item["sent_id"]]),
        "ref_id": torch.tensor([item["ref_id"]]),
        "text": [item["text"]],
    }
    pooled = cache.get_batch(batch, expected_split="testB", device=device)
    path, deployed_parameters, active_parameters, checkpoint = make_path(mode, device, protocol)

    # Select one stable route from the frozen rule for this same reference.
    path.hook.value = None
    if hasattr(path.general, "forward_general"):
        general_logits, _ = path.general.forward_general(image, pooled_text=pooled)
    else:
        general_logits, _ = path.general(image, pooled_text=pooled)
    _, area050 = general_stats(general_logits)
    selected, _, _ = protocol.route_sample(item["text"], float(area050[0].item()))
    if selected != BRANCH:
        path.branch = selected
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        profile_memory=False,
    ) as profiler:
        path.general_expert(image, pooled)
    torch.cuda.synchronize()
    image_flops = sum(event.flops for event in profiler.key_averages())

    # Warm the exact cached-text inference path before synchronized wall timing.
    for _ in range(WARMUP):
        path(image, pooled, item["text"])
    torch.cuda.synchronize()
    run_latencies_ms = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(ITERATIONS):
            path(image, pooled, item["text"])
        torch.cuda.synchronize()
        run_latencies_ms.append((time.perf_counter() - started) * 1000 / ITERATIONS)
    path.hook.close()
    mean_latency = statistics.median(run_latencies_ms)
    result = {
        "mode": mode,
        "checkpoint": checkpoint,
        "gpu": torch.cuda.get_device_name(0),
        "input": "384x384 FP32 batch=1",
        "sample_index": 0,
        "ref_id": int(item["ref_id"]),
        "branch": path.branch,
        "warmup_iterations": WARMUP,
        "timed_iterations": ITERATIONS,
        "timing_repeats": REPEATS,
        "repeat_latency_ms": run_latencies_ms,
        "total_deployed_parameters_including_all_experts_bert_once_gate_once": deployed_parameters,
        "active_path_parameters_including_one_expert_bert_once_gate_once": active_parameters,
        "general_plus_one_expert_profiler_flops": int(image_flops),
        "general_plus_one_expert_profiler_flops_G": image_flops / 1e9,
        "cached_text_pipeline_latency_ms_median_of_repeat_means": mean_latency,
        "cached_text_pipeline_latency_ms_mean_of_repeat_means": statistics.mean(run_latencies_ms),
        "cached_text_pipeline_FPS_from_median_latency": 1000 / mean_latency,
        "scope": "General + frozen Gate + frozen routing + one expert; cached BERT pooled text; excludes file I/O and GUI",
        "flops_scope": "General + one expert only; same PyTorch profiler convention for every model; excludes Gate and cached BERT encoder compute",
        "testB_is_historically_exposed": True,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / f"{mode}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("private", "decoder4", "final_v3"), required=True)
    run(parser.parse_args().mode)
