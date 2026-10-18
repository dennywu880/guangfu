"""Measure cached-text inference cost of the isolated multilevel candidates."""

import gc
import json
import statistics

import torch

from detail_adapter_model import build_detail_model
from multilevel_selection import build_model
from train_detail_adapter import ROOT
from feature_bridge_experiment import BASE_CHECKPOINT


RUNS = {
    "e2": ("e2", "multilevel_e2_20260925"),
    "e1_e2": ("e1", "e2", "multilevel_e1_e2_20260925"),
    "e2_e3": ("e2", "e3", "multilevel_e2_e3_20260925"),
}


@torch.inference_mode()
def measure(model, image, pooled):
    model.eval()

    def step():
        _, context = model.forward_general(image, pooled_text=pooled)
        model.forward_expert(context, "tiny")

    for _ in range(20):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 50)
    return {
        "median_gpu_ms": statistics.median(times),
        "repeat_gpu_ms": times,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "parameters_excluding_BERT": sum(p.numel() for p in model.parameters()),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    device = torch.device("cuda")
    image = torch.randn(1, 3, 384, 384, device=device)
    pooled = torch.randn(1, 768, device=device)
    results = {}
    base = build_detail_model(ROOT / "checkpoints", device)
    base.load_checkpoint(BASE_CHECKPOINT)
    results["base"] = measure(base, image, pooled)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    for name, parts in RUNS.items():
        *levels, directory = parts
        model = build_model(device, levels)
        model.load_checkpoint(ROOT / "results" / directory / "completed.pth")
        results[name] = measure(model, image, pooled)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    output = ROOT / "results" / "multilevel_route_20260925" / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "protocol": "Batch 1, 384x384, cached pooled text, general encoder + tiny head; excludes I/O, BERT, router and presence Gate. Five repeats of 50 passes, 20 warmup passes.",
        "results": results,
    }, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
