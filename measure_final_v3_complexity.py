"""FP32 CUDA complexity benchmark for the frozen Final V3 inference system."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader

from dataset import RefELDataset
from inference_engine import BASELINE, DefectInferenceEngine


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
WARMUP = 20
ITERATIONS = 100


class PooledExpert(nn.Module):
    """Expose the real expert cached-text forward to THOP."""

    def __init__(self, expert: nn.Module):
        super().__init__()
        self.expert = expert

    def forward(self, image: torch.Tensor, pooled_text: torch.Tensor):
        return self.expert(image, pooled_text=pooled_text)


def first_real_batch() -> tuple[torch.Tensor, str, str]:
    """Get an actual Ref-EL-Defect batch and its training sentence."""
    dataset = RefELDataset(ROOT, split="train", image_size=384, random_text=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        text = str(batch["text"][0])
        # The known positive Final V3 reference prevents a no-target gate path.
        if "tiny" in text.lower():
            return batch["image"].cuda(non_blocking=False), text, str(batch["file_name"][0])
    raise RuntimeError("No real Ref-EL-Defect text sample was found.")


@torch.inference_mode()
def forward_path(engine: DefectInferenceEngine, image: torch.Tensor, text: str):
    """Actual Final V3 model path, excluding GT/file/UI packaging."""
    engine.hook.value = None
    debug = engine.general_model(image, texts=[text], return_debug=True)
    pooled = debug["pooled_text"].detach().float()
    general_probs = torch.sigmoid(debug["logits"])
    stats, scalars = engine._general_stats(general_probs)
    deep = engine.hook.value
    image_feature = engine._image_feature_from_hook(deep).float()
    branch, _, text_size = BASELINE.route_sample(text, scalars["area050"])
    route_features = torch.tensor([BASELINE.route_onehot(branch)], dtype=torch.float32, device=image.device)
    text_cues = torch.tensor([BASELINE.text_onehot(text_size)], dtype=torch.float32, device=image.device)
    gate_x = torch.cat([image_feature, pooled, stats.float(), route_features, text_cues], dim=1)
    gate_z = (gate_x - engine.gate_mean) / engine.gate_std
    gate_probability = torch.sigmoid(engine.gate(gate_z))
    # Complexity uses one actual routed expert forward. The gate is still run.
    expert = engine._activate_expert(branch)
    logits = expert(image, pooled_text=pooled)
    return logits, branch, pooled, gate_z, gate_probability


def million(module: nn.Module) -> float:
    return sum(parameter.numel() for parameter in module.parameters()) / 1e6


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.backends.cudnn.benchmark = True
    engine = DefectInferenceEngine()
    image, text, filename = first_real_batch()

    # Determine the real route before profiling/counting the active path.
    _, branch, pooled, gate_z, _ = forward_path(engine, image, text)
    torch.cuda.synchronize()
    expert = engine._activate_expert(branch)

    # Final V3 package has General + Gate + all three stored experts. Runtime
    # path uses General + Gate + exactly one selected expert at a time.
    deployed_params = million(engine.general_model) + million(engine.gate) + sum(million(item) for item in engine.experts.values())
    active_params = million(engine.general_model) + million(engine.gate) + million(expert)

    general_flops, _ = profile(engine.general_model, inputs=(image, [text]), verbose=False)
    expert_flops, _ = profile(PooledExpert(expert), inputs=(image, pooled), verbose=False)
    gate_flops, _ = profile(engine.gate, inputs=(gate_z,), verbose=False)
    active_flops_g = (general_flops + expert_flops + gate_flops) / 1e9

    # Peak allocated GPU memory for the real whole-image Final V3 path.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    forward_path(engine, image, text)
    torch.cuda.synchronize()
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    for _ in range(WARMUP):
        forward_path(engine, image, text)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        forward_path(engine, image, text)
    torch.cuda.synchronize()
    fps = ITERATIONS / (time.perf_counter() - start)

    metrics_path = RESULTS_DIR / "final_testB_gate_v2" / "final_testB_gate_v2_results.json"
    paper = json.loads(metrics_path.read_text(encoding="utf-8"))
    v3 = paper["m2segnet_style_metrics"]
    m2 = paper["m2segnet_paper_reference_numbers"]

    report = f"""================================
Model Complexity Evaluation
===========================

Model:
TG-UNet Final V3 (General + Presence Gate v2 + 3 Experts)

Input:
384x384, batch size 1, FP32, evaluation mode

GPU:
{torch.cuda.get_device_name(0)}

Real Ref-EL-Defect sample:
{filename}
Text: {text}
Routed expert: {branch}

---

Total Parameters (deployed 3-expert package):
{deployed_params:.2f} M

Active-path Parameters (General + Gate + routed expert):
{active_params:.2f} M

FLOPs (active routed path, THOP):
{active_flops_g:.2f} G

GPU Memory (peak allocated, active routed path):
{memory_mb:.0f} MB

Inference FPS (whole-image Final V3 path, BERT included):
{fps:.2f}

---

Protocol: 20 warm-up iterations, 100 timed iterations, CUDA synchronized.
FLOPs covers General + Presence Gate v2 + the one expert selected by the frozen router.
Parameters includes all three deployed experts; only one expert is resident on CUDA per inference.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "complexity_results.txt").write_text(report, encoding="utf-8")

    table = f"""| Method | gIoU | cIoU | Recall | Precision | F1-Score | mPr | Params (M) | FLOPs (G) | Memory (MB) | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2SegNet (paper reference) | {m2['gIoU']:.4f} | {m2['cIoU']:.4f} | {m2['Recall']:.4f} | {m2['Precision']:.4f} | {m2['F1']:.4f} | {m2['mPr']:.4f} | N/R | N/R | N/R | N/R |
| TG-UNet Final V3 | {v3['gIoU']:.4f} | {v3['cIoU']:.4f} | {v3['Recall']:.4f} | {v3['Precision']:.4f} | {v3['F1']:.4f} | {v3['mPr']:.4f} | {deployed_params:.2f} | {active_flops_g:.2f} | {memory_mb:.0f} | {fps:.2f} |

Complexity protocol: RTX 4060 Laptop GPU 8GB, FP32, batch size 1, 384x384, evaluation mode. M2SegNet complexity is N/R because this project only stores the paper's segmentation metrics, not an executable M2SegNet checkpoint under the same protocol.
"""
    (RESULTS_DIR / "table_row.txt").write_text(table, encoding="utf-8")
    print(report)
    print(table)


if __name__ == "__main__":
    main()
