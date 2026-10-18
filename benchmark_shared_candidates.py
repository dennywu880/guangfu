"""Uniform RTX 5060 complexity audit for shared testB candidates.

Text is supplied from the frozen formal cache, matching the testB evaluator;
therefore this measures image/segmentation runtime and explicitly excludes
the one-time BERT text encoding cost.
"""
import json
import time
from pathlib import Path

import torch
from torch import nn
from thop import profile

from dataset import RefELDataset
from multilevel_selection import build_model
from train_multilevel_head_joint import load as load_joint
from train_text import FormalTextCache
from evaluate_shared_testB import original_protocol

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "shared_testB_comparison_20260925"
CANDIDATES = {
    "multilevel_best": ROOT / "results/multilevel_e3_e4_20260925/best.pth",
    "head_joint_best": ROOT / "results/head_joint_e3_e4_20260925/best.pth",
    "distill_dev_ciou_best": ROOT / "results/selective_distill_e3_e4_20260925/best.pth",
}


class RoutedPath(nn.Module):
    def __init__(self, model, branch):
        super().__init__()
        self.model = model
        self.branch = branch

    def forward(self, image, pooled_text):
        _, context = self.model.forward_general(image, pooled_text=pooled_text)
        return self.model.forward_expert(context, self.branch)


def count(module):
    return sum(p.numel() for p in module.parameters())


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    device = torch.device("cuda")
    dataset = RefELDataset(ROOT, split="testB", image_size=384, random_text=False)
    cache = FormalTextCache(ROOT)
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    batch = {
        "text": [sample["text"]],
        "ref_id": torch.tensor([sample["ref_id"]]),
        "sent_id": torch.tensor([sample["sent_id"]]),
    }
    pooled = cache.get_batch(batch=batch, expected_split="testB", device=device)
    protocol = original_protocol()
    gate, _, _, _, _ = protocol.load_presence_gate(device)
    gate_input_dim = gate.net[0].in_features
    gate_flops, _ = profile(gate, inputs=(torch.zeros(1, gate_input_dim, device=device),), verbose=False)
    out = {}
    for name, checkpoint in CANDIDATES.items():
        model = build_model(device, ("e3", "e4"))
        if name == "multilevel_best":
            model.load_checkpoint(checkpoint)
        else:
            load_joint(checkpoint, model)
        model.eval()
        branch = "larger"
        path = RoutedPath(model, branch).eval()
        flops, _ = profile(path, inputs=(image, pooled), verbose=False)
        # Count a gate separately; this is the fixed Presence Gate architecture.
        start_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            path(image, pooled)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(40):
            path(image, pooled)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000 / 40
        peak = torch.cuda.max_memory_allocated() / 1024**2
        textless = count(model)
        # One BERT is shared by all branches in deployment.  The exact BERT
        # count is obtained from the original frozen model audit.
        bert_params = 109482240
        gate_params = count(gate)
        active_textless = (count(model.base.general) + count(model.base.heads[branch])
                           + count(model.base.detail_adapters[branch])
                           + count(model.selections[branch]))
        out[name] = {
            "gpu": torch.cuda.get_device_name(0),
            "input": "384x384 FP32 batch=1",
            "text_protocol": "formal pooled-text cache; BERT encoding excluded",
            "textless_candidate_parameters": textless,
            "estimated_deployed_parameters_with_one_shared_bert_and_gate": textless + bert_params + gate_params,
            "active_parameters_with_one_shared_bert_gate_and_one_branch": active_textless + bert_params + gate_params,
            "FLOPs_G_image_expert_and_gate": (flops + gate_flops) / 1e9,
            "latency_ms_image_and_expert": latency_ms,
            "FPS_image_and_expert": 1000.0 / latency_ms,
            "peak_gpu_memory_MiB": peak,
            "note": "Candidate has one shared ResNet encoder; timing excludes BERT, Gate, router and UI/file packaging. THOP FLOPs include the gate but not BERT; this is not a complete-chain number.",
        }
        del path, model
        torch.cuda.empty_cache()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "complexity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
