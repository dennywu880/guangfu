"""Measure original and shared full paths under the same local GPU protocol."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from thop import profile

from dataset import RefELDataset
from inference_engine import BASELINE, DefectInferenceEngine
from shared_moe import BRANCH_FILES, SharedFeatures, SharedMoE


ROOT = Path(__file__).resolve().parent
CASES = {"ultra_tiny": 5245, "tiny": 3918, "larger": 185}


def params(module):
    return sum(p.numel() for p in module.parameters())


class OriginalExpert(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, text):
        return self.model(image, pooled_text=text)


class SharedExpert(nn.Module):
    def __init__(self, model, branch):
        super().__init__()
        self.encoder = model.expert_encoder
        self.head = model.heads[branch]

    def forward(self, image, text):
        features = SharedFeatures(self.encoder(image), text, image.shape[-2:])
        return SharedMoE._decode(self.head, features)


class SingleEncoderHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, e0, e1, e2, e3, e4, text):
        context = SharedFeatures((e0, e1, e2, e3, e4), text, (384, 384))
        return SharedMoE._decode(self.head, context)


@torch.inference_mode()
def run(mode, warmup, iterations):
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires CUDA")
    os.environ["TG_SHARED_MOE"] = "0"
    torch.set_num_threads(4)
    engine = DefectInferenceEngine()
    shared = None
    if mode in ("shared", "single"):
        shared = SharedMoE.from_checkpoints(
            ROOT / "checkpoints", engine.device,
            general=engine.general_model,
            expert_backbone="larger" if mode == "shared" else None)
    dataset = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
    results = {}
    count_general = params(engine.general_model)
    count_gate = params(engine.gate)
    original_heads = {}
    original_encoders = {}
    for branch, expert in engine.experts.items():
        weights = dict(expert.named_parameters())
        original_encoders[branch] = sum(
            value.numel() for key, value in weights.items()
            if key.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4."))
        )
        original_heads[branch] = sum(
            value.numel() for key, value in weights.items()
            if key.startswith(("text_gate_", "decoder", "final_conv.", "segmentation_head."))
        )
    expert_encoder_count = params(shared.expert_encoder) if shared and shared.expert_encoder else None
    if shared:
        head_counts = {branch: params(head) for branch, head in shared.heads.items()}
        deployed = count_general + count_gate + (expert_encoder_count or 0) + sum(head_counts.values())
    else:
        head_counts = original_heads
        deployed = count_general + count_gate + sum(
            original_encoders[b] + original_heads[b] for b in BRANCH_FILES)

    for branch, index in CASES.items():
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(engine.device)
        text = sample["text"]
        teacher = engine._activate_expert(branch) if mode == "original" else None

        def whole_path():
            engine.hook.value = None
            if shared:
                general, context = shared.forward_general(image, texts=[text])
                pooled = context.pooled_text
            else:
                debug = engine.general_model(image, texts=[text], return_debug=True)
                general = debug["logits"]
                pooled = debug["pooled_text"].detach().float()
            image_feature = engine._image_feature_from_hook(engine.hook.value).float()
            stats, values = engine._general_stats(general.sigmoid())
            route, _, size = BASELINE.route_sample(text, values["area050"])
            features = torch.cat([
                image_feature, pooled, stats.float(),
                torch.tensor([BASELINE.route_onehot(route)], device=engine.device),
                torch.tensor([BASELINE.text_onehot(size)], device=engine.device),
            ], dim=1)
            no_target = engine.gate((features - engine.gate_mean) / engine.gate_std).sigmoid()
            if route != branch:
                raise RuntimeError(f"Case {index} routed to {route}, expected {branch}")
            if shared:
                logits = shared.forward_expert(context, branch)
            else:
                logits = teacher(image, pooled_text=pooled)
            return logits, pooled, features, no_target

        logits, pooled, gate_features, no_target = whole_path()
        torch.cuda.synchronize()
        if mode == "single":
            expert_wrapper = SingleEncoderHead(shared.heads[branch])
        else:
            expert_wrapper = SharedExpert(shared, branch) if shared else OriginalExpert(teacher)
        if branch == "ultra_tiny":
            general_flops, _ = profile(engine.general_model, inputs=(image, [text]), verbose=False)
            if mode == "single":
                _, context = shared.forward_general(image, pooled_text=pooled)
                expert_inputs = (*context.image, pooled)
            else:
                expert_inputs = (image, pooled)
            expert_flops, _ = profile(expert_wrapper, inputs=expert_inputs, verbose=False)
            gate_flops, _ = profile(engine.gate,
                                    inputs=((gate_features - engine.gate_mean) / engine.gate_std,),
                                    verbose=False)
            flop_counts = {"general": general_flops, "expert": expert_flops,
                           "gate": gate_flops}

        resident_mib = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        whole_path()
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        for _ in range(warmup):
            whole_path()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            whole_path()
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000 / iterations
        results[branch] = {
            "dataset_index": index, "text": text,
            "gate_rejected": bool((no_target >= engine.gate_threshold).item()),
            "active_parameters": count_general + count_gate +
                                 ((expert_encoder_count or 0) + head_counts[branch] if shared else
                                  original_encoders[branch] + original_heads[branch]),
            "latency_ms": latency_ms,
            "resident_gpu_mib": resident_mib,
            "peak_gpu_mib": peak_mib,
            "peak_extra_mib": peak_mib - resident_mib,
        }

    report = {
        "mode": mode, "gpu": torch.cuda.get_device_name(0),
        "input": "384x384 FP32 batch=1", "warmup": warmup, "iterations": iterations,
        "deployed_parameters_excluding_duplicate_bert": deployed,
        "general_parameters_including_bert": count_general,
        "presence_gate_parameters": count_gate,
        "expert_encoder_parameters_per_branch": original_encoders,
        "expert_head_parameters_per_branch": head_counts,
        "shared_expert_encoder_parameters": expert_encoder_count,
        "ultra_tiny_flops_thop": flop_counts,
        "cases": results,
        "note": "Full path computes expert even for a gate-rejected case to compare fixed work. "
                "GPU memory is process-local; resident and peak values include loaded weights.",
    }
    output = ROOT / "results" / f"runtime_audit_{mode}_5060.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("original", "shared", "single"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    run(args.mode, args.warmup, args.iterations)
