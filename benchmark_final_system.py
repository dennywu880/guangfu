# benchmark_final_system.py
# Formal efficiency benchmark for the final TG-UNet system.
#
# Measures the ACTUAL inference path exposed by inference_engine.py:
#   General -> Presence Gate v2 -> Router -> one Expert (target)
#   General -> Presence Gate v2 -> early exit          (no-target)
#
# Metrics:
#   - Unique resident parameter count
#   - Executed-path FLOPs (PyTorch FlopCounterMode; profiler fallback)
#   - Peak CUDA allocated/reserved memory
#   - End-to-end latency (batch=1, file I/O excluded)
#   - FPS
#
# No checkpoint, threshold, dataset, or model parameter is modified.

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image


ROOT = Path(__file__).resolve().parent

GREF_CANDIDATES = [
    ROOT / "datasets" / "grefcoco" / "grefs(unc).json",
    ROOT / "datasets" / "grefcoco" / "grefs_unc.json",
]

IMAGE_DIR_CANDIDATES = [
    ROOT / "datasets" / "images" / "train2014_el",
    ROOT / "datasets" / "images" / "train2014",
    ROOT / "datasets" / "images",
]


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def format_millions(n: int) -> str:
    return f"{n / 1e6:.3f} M"


def format_gflops(n: float) -> str:
    return f"{n / 1e9:.3f} GFLOPs"


def safe_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): safe_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_jsonable(v) for v in x]
    return str(x)


# ---------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------

def import_engine_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        return importlib.import_module("inference_engine")
    except Exception as exc:
        raise RuntimeError(
            "Cannot import inference_engine.py from the project root.\n"
            f"Project root: {ROOT}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def instantiate_engine(module):
    # Prefer an already-created engine object if the module exposes one.
    for name in ("ENGINE", "engine", "INFERENCE_ENGINE"):
        obj = getattr(module, name, None)
        if obj is not None and hasattr(obj, "infer"):
            return obj, f"inference_engine.{name}"

    cls = getattr(module, "DefectInferenceEngine", None)
    if cls is None:
        # Conservative fallback: find a class with an infer() method.
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if hasattr(candidate, "infer"):
                cls = candidate
                break

    if cls is None:
        raise RuntimeError(
            "No DefectInferenceEngine (or compatible class with infer()) "
            "was found in inference_engine.py."
        )

    sig = inspect.signature(cls)
    kwargs = {}

    for name, p in sig.parameters.items():
        if name == "self":
            continue
        lname = name.lower()

        if p.default is not inspect._empty:
            continue

        if lname in {
            "root", "root_dir", "project_root", "project_dir",
            "base_dir", "base_path"
        }:
            kwargs[name] = ROOT
        elif lname in {"device"}:
            kwargs[name] = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            raise RuntimeError(
                "The engine constructor has a required parameter that this "
                "benchmark cannot safely guess:\n"
                f"  {name}\n"
                f"Constructor signature: {sig}\n"
                "Pass a default value in inference_engine.py or adapt "
                "instantiate_engine() explicitly."
            )

    try:
        return cls(**kwargs), f"{cls.__module__}.{cls.__name__}"
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate {cls.__name__}{sig}\n"
            f"kwargs={kwargs}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# Dataset sample discovery
# ---------------------------------------------------------------------

def find_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def read_grefs() -> Tuple[Path, list]:
    gref_path = find_existing_path(GREF_CANDIDATES)
    if gref_path is None:
        raise FileNotFoundError(
            "Cannot find grefs JSON. Checked:\n"
            + "\n".join(str(p) for p in GREF_CANDIDATES)
        )

    with gref_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Some exports wrap the list.
        for key in ("refs", "references", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected grefs JSON structure in {gref_path}"
        )

    return gref_path, data


def extract_text(ref: dict) -> Optional[str]:
    sentences = ref.get("sentences")
    if isinstance(sentences, list) and sentences:
        s = sentences[0]
        if isinstance(s, dict):
            for key in ("raw", "sent", "text", "expression"):
                value = s.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(s, str) and s.strip():
            return s.strip()

    for key in ("text", "expression", "sentence", "raw"):
        value = ref.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def locate_image(file_name: str) -> Optional[Path]:
    for d in IMAGE_DIR_CANDIDATES:
        p = d / file_name
        if p.exists():
            return p

    # Last-resort exact-name search under datasets/images only.
    root = ROOT / "datasets" / "images"
    if root.exists():
        matches = list(root.rglob(file_name))
        if matches:
            return matches[0]

    return None


def normalize_split(x: Any) -> str:
    return str(x or "").strip().lower().replace("_", "").replace("-", "")


def choose_sample(
    refs: list,
    requested_split: str,
    want_no_target: bool,
) -> Tuple[Path, str, dict]:
    requested = normalize_split(requested_split)

    # First pass: requested split only.
    passes = [requested, None]

    for split_filter in passes:
        for ref in refs:
            if split_filter is not None:
                if normalize_split(ref.get("split")) != split_filter:
                    continue

            no_target = bool(ref.get("no_target", False))
            ann_id = ref.get("ann_id")
            if ann_id == -1 or ann_id == [-1]:
                no_target = True

            if no_target != want_no_target:
                continue

            file_name = ref.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                continue

            text = extract_text(ref)
            if not text:
                continue

            image_path = locate_image(file_name)
            if image_path is None:
                continue

            return image_path, text, ref

    kind = "no-target" if want_no_target else "positive"
    raise RuntimeError(
        f"Could not find a usable {kind} sample "
        f"(requested split={requested_split})."
    )


# ---------------------------------------------------------------------
# infer() argument adapter
# ---------------------------------------------------------------------

IMAGE_NAMES = {
    "image", "img", "input_image", "image_input", "el_image",
    "source_image", "pil_image", "image_file"
}
IMAGE_PATH_NAMES = {
    "image_path", "img_path", "path", "file_path", "filepath"
}
TEXT_NAMES = {
    "text", "expression", "query", "final_query", "free_text",
    "prompt", "sentence", "description"
}
GT_NAMES = {
    "gt", "gt_mask", "ground_truth", "ground_truth_mask",
    "optional_gt", "mask_gt"
}
DEFECT_NAMES = {"defect_type", "defect", "category", "class_name"}
POSITION_NAMES = {"position", "region", "location"}
SIZE_NAMES = {"size", "defect_size"}
RELATION_NAMES = {
    "relation", "busbar_relation", "busbar", "connection_relation"
}


def build_infer_call(engine, image_path: Path, text: str):
    method = engine.infer
    sig = inspect.signature(method)

    # Keep file I/O outside timed region.
    pil = Image.open(image_path).convert("RGB")
    try:
        pil.filename = str(image_path)
    except Exception:
        pass

    args = []
    kwargs = {}

    for name, p in sig.parameters.items():
        if name == "self":
            continue

        lname = name.lower()

        if lname in IMAGE_PATH_NAMES:
            value = str(image_path)
        elif lname in IMAGE_NAMES or "image" in lname and "path" not in lname:
            value = pil
        elif lname in TEXT_NAMES or any(
            token in lname for token in ("expression", "query", "text")
        ):
            value = text
        elif lname in GT_NAMES or "ground_truth" in lname or lname.startswith("gt"):
            value = None
        elif lname in DEFECT_NAMES:
            value = None
        elif lname in POSITION_NAMES:
            value = None
        elif lname in SIZE_NAMES:
            value = None
        elif lname in RELATION_NAMES or "busbar" in lname:
            value = None
        elif p.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(
                "Cannot safely map a required infer() parameter:\n"
                f"  {name}\n"
                f"infer signature: {sig}\n"
                "Please adapt build_infer_call() for this parameter."
            )

        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            # kwargs are safer for readable signatures.
            kwargs[name] = value
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            kwargs[name] = value

    def call():
        return method(*args, **kwargs)

    return call, sig, kwargs, pil


# ---------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------

def collect_modules(obj: Any) -> List[Tuple[str, nn.Module]]:
    found: List[Tuple[str, nn.Module]] = []
    seen_obj = set()

    def rec(x: Any, path: str, depth: int):
        oid = id(x)
        if oid in seen_obj:
            return
        seen_obj.add(oid)

        if isinstance(x, nn.Module):
            found.append((path, x))
            return

        if depth > 4:
            return

        if isinstance(x, dict):
            for k, v in x.items():
                rec(v, f"{path}[{k!r}]", depth + 1)
            return

        if isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                rec(v, f"{path}[{i}]", depth + 1)
            return

        if hasattr(x, "__dict__"):
            # Recurse through engine-owned objects, but do not walk huge
            # third-party internals such as tokenizers.
            module_name = getattr(type(x), "__module__", "")
            if depth == 0 or module_name.startswith(
                ("inference_engine", "__main__")
            ):
                for k, v in vars(x).items():
                    if k.startswith("__"):
                        continue
                    rec(v, f"{path}.{k}", depth + 1)

    rec(obj, "engine", 0)
    return found


def count_unique_parameters(engine) -> dict:
    modules = collect_modules(engine)

    unique = {}
    by_device = {}
    trainable = 0

    for path, module in modules:
        for name, p in module.named_parameters(recurse=True):
            pid = id(p)
            if pid in unique:
                continue
            unique[pid] = (path, name, p)
            dev = str(p.device)
            by_device[dev] = by_device.get(dev, 0) + p.numel()
            if p.requires_grad:
                trainable += p.numel()

    total = sum(item[2].numel() for item in unique.values())

    return {
        "module_roots_found": [path for path, _ in modules],
        "unique_total_params": int(total),
        "unique_trainable_params": int(trainable),
        "params_by_device": {k: int(v) for k, v in by_device.items()},
    }


# ---------------------------------------------------------------------
# Benchmark primitives
# ---------------------------------------------------------------------

def warmup(call, n: int) -> None:
    with torch.inference_mode():
        for _ in range(n):
            _ = call()
            sync_cuda()


def measure_latency(call, repeats: int) -> dict:
    times_ms = []

    gc.collect()
    sync_cuda()

    with torch.inference_mode():
        for _ in range(repeats):
            sync_cuda()
            t0 = time.perf_counter()
            _ = call()
            sync_cuda()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    return {
        "repeats": repeats,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "p90_ms": percentile(times_ms, 0.90),
        "p95_ms": percentile(times_ms, 0.95),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "fps_from_mean": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
        "all_ms": times_ms,
    }


def measure_cuda_memory(call) -> dict:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "CUDA is not available."
        }

    gc.collect()
    torch.cuda.empty_cache()
    sync_cuda()

    baseline_alloc = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        _ = call()
        sync_cuda()

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    return {
        "available": True,
        "baseline_allocated_bytes": int(baseline_alloc),
        "baseline_reserved_bytes": int(baseline_reserved),
        "peak_allocated_bytes": int(peak_alloc),
        "peak_reserved_bytes": int(peak_reserved),
        "incremental_peak_allocated_bytes": int(
            max(0, peak_alloc - baseline_alloc)
        ),
        "baseline_allocated_gib": baseline_alloc / (1024 ** 3),
        "peak_allocated_gib": peak_alloc / (1024 ** 3),
        "peak_reserved_gib": peak_reserved / (1024 ** 3),
        "incremental_peak_allocated_gib": max(
            0, peak_alloc - baseline_alloc
        ) / (1024 ** 3),
    }


def measure_flops(call) -> dict:
    # Preferred path in recent PyTorch versions.
    try:
        from torch.utils.flop_counter import FlopCounterMode

        sync_cuda()
        with torch.inference_mode():
            with FlopCounterMode(display=False) as flop_counter:
                _ = call()
                sync_cuda()

        total = float(flop_counter.get_total_flops())
        if total > 0:
            return {
                "success": True,
                "method": "torch.utils.flop_counter.FlopCounterMode",
                "flops": total,
                "gflops": total / 1e9,
                "macs_equivalent_if_1MAC_2FLOPs": total / 2.0,
                "note": (
                    "Executed-path operator FLOPs. "
                    "Only operations supported by PyTorch's counter "
                    "are included."
                ),
            }
    except Exception as exc:
        primary_error = f"{type(exc).__name__}: {exc}"
    else:
        primary_error = "FlopCounterMode returned zero FLOPs."

    # Fallback to torch.profiler's supported-op FLOP estimates.
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        sync_cuda()
        with torch.inference_mode():
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                _ = call()
                sync_cuda()

        total = 0.0
        for evt in prof.key_averages():
            value = getattr(evt, "flops", 0) or 0
            total += float(value)

        return {
            "success": total > 0,
            "method": "torch.profiler.profile(with_flops=True)",
            "flops": total,
            "gflops": total / 1e9,
            "macs_equivalent_if_1MAC_2FLOPs": total / 2.0,
            "note": (
                "Executed-path supported-op FLOPs; this may be a lower "
                "bound because not every PyTorch operator exposes FLOPs."
            ),
            "primary_counter_error": primary_error,
        }

    except Exception as exc:
        return {
            "success": False,
            "method": "unavailable",
            "flops": None,
            "gflops": None,
            "primary_counter_error": primary_error,
            "profiler_error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------
# One path benchmark
# ---------------------------------------------------------------------

def benchmark_case(
    name: str,
    call,
    image_path: Path,
    text: str,
    warmup_n: int,
    repeats: int,
) -> dict:
    print()
    print("=" * 88)
    print(f"BENCHMARK CASE: {name}")
    print("=" * 88)
    print("Image:", image_path)
    print("Text :", text)

    print(f"\nWarmup: {warmup_n} runs")
    warmup(call, warmup_n)

    print("\nMeasuring peak CUDA memory...")
    memory = measure_cuda_memory(call)

    print("\nMeasuring executed-path FLOPs...")
    flops = measure_flops(call)

    print(f"\nMeasuring latency: {repeats} runs")
    latency = measure_latency(call, repeats)

    print("\nResult:")
    if flops.get("success"):
        print(f"  FLOPs            : {flops['gflops']:.3f} GFLOPs")
    else:
        print("  FLOPs            : unavailable")

    if memory.get("available"):
        print(
            f"  Peak VRAM alloc  : "
            f"{memory['peak_allocated_gib']:.3f} GiB"
        )
        print(
            f"  Peak VRAM reserve: "
            f"{memory['peak_reserved_gib']:.3f} GiB"
        )
        print(
            f"  Activation delta : "
            f"{memory['incremental_peak_allocated_gib']:.3f} GiB"
        )

    print(f"  Mean latency     : {latency['mean_ms']:.3f} ms/image")
    print(f"  Median latency   : {latency['median_ms']:.3f} ms/image")
    print(f"  P95 latency      : {latency['p95_ms']:.3f} ms/image")
    print(f"  FPS              : {latency['fps_from_mean']:.3f}")

    return {
        "case": name,
        "image": str(image_path),
        "text": text,
        "flops": flops,
        "memory": memory,
        "latency": latency,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Formal benchmark for the final TG-UNet inference system."
    )
    parser.add_argument(
        "--split",
        default="testB",
        help="Dataset split used only to select representative samples."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup runs per case."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Timed runs per case."
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Benchmark only the full target path."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional explicit image path for the positive-path benchmark."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional explicit text expression used with --image."
    )
    args = parser.parse_args()

    print()
    print("=" * 88)
    print("TG-UNET FINAL SYSTEM - FORMAL EFFICIENCY BENCHMARK")
    print("=" * 88)
    print("Project root :", ROOT)
    print("PyTorch      :", torch.__version__)
    print("CUDA avail.  :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA runtime :", torch.version.cuda)
        print("GPU          :", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(
            "GPU memory   :",
            f"{props.total_memory / (1024 ** 3):.2f} GiB"
        )

    # Make execution more deterministic/stable for timing.
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("\nLoading the REAL inference engine...")
    module = import_engine_module()
    engine, engine_source = instantiate_engine(module)
    print("Engine       :", engine_source)

    # Count the actual modules/parameters the engine has loaded.
    param_stats = count_unique_parameters(engine)
    print("\nResident unique parameters:")
    print(
        "  Total      :",
        format_millions(param_stats["unique_total_params"])
    )
    print(
        "  Trainable  :",
        format_millions(param_stats["unique_trainable_params"])
    )
    if param_stats["params_by_device"]:
        for dev, n in param_stats["params_by_device"].items():
            print(f"  {dev:10s}: {format_millions(n)}")

    gref_path = None
    refs = None

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not args.text:
            raise ValueError("--image requires --text.")
        positive_image = image_path
        positive_text = args.text
        positive_ref = {"source": "explicit CLI input"}
    else:
        gref_path, refs = read_grefs()
        positive_image, positive_text, positive_ref = choose_sample(
            refs=refs,
            requested_split=args.split,
            want_no_target=False,
        )

    pos_call, pos_sig, pos_kwargs, pos_pil = build_infer_call(
        engine, positive_image, positive_text
    )

    print("\ninfer() signature:", pos_sig)
    print("Benchmark input mapping:")
    for k, v in pos_kwargs.items():
        if isinstance(v, Image.Image):
            shown = f"PIL.Image(size={v.size}, mode={v.mode})"
        else:
            shown = repr(v)
        print(f"  {k} = {shown}")

    # One untimed smoke call before formal runs, so incompatible API errors
    # happen before the benchmark.
    print("\nRunning one smoke inference...")
    with torch.inference_mode():
        smoke_output = pos_call()
        sync_cuda()
    print("Smoke inference: OK")

    positive = benchmark_case(
        name="TARGET / FULL PATH (General + Gate + Router + 1 Expert)",
        call=pos_call,
        image_path=positive_image,
        text=positive_text,
        warmup_n=args.warmup,
        repeats=args.repeats,
    )

    no_target = None
    nt_pil = None

    if not args.positive_only:
        if refs is None:
            try:
                gref_path, refs = read_grefs()
            except Exception:
                refs = None

        if refs is not None:
            try:
                nt_image, nt_text, nt_ref = choose_sample(
                    refs=refs,
                    requested_split=args.split,
                    want_no_target=True,
                )
                nt_call, nt_sig, nt_kwargs, nt_pil = build_infer_call(
                    engine, nt_image, nt_text
                )

                with torch.inference_mode():
                    _ = nt_call()
                    sync_cuda()

                no_target = benchmark_case(
                    name="NO-TARGET / EARLY-EXIT PATH (General + Gate)",
                    call=nt_call,
                    image_path=nt_image,
                    text=nt_text,
                    warmup_n=args.warmup,
                    repeats=args.repeats,
                )
            except Exception as exc:
                print(
                    "\n[WARNING] No-target benchmark was skipped:",
                    f"{type(exc).__name__}: {exc}"
                )

    report = {
        "benchmark_name": "TG-UNet Final System Formal Efficiency Benchmark",
        "project_root": str(ROOT),
        "engine_source": engine_source,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "gpu_total_memory_gib": (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if torch.cuda.is_available()
                else None
            ),
        },
        "protocol": {
            "batch_size": 1,
            "input_resolution": "engine default / final project setting (expected 384x384)",
            "warmup_runs_per_case": args.warmup,
            "timed_runs_per_case": args.repeats,
            "latency_scope": (
                "DefectInferenceEngine.infer(), image file I/O excluded; "
                "model preprocessing/tokenization/postprocessing inside infer() included."
            ),
            "memory_scope": (
                "CUDA peak allocated/reserved memory with already-loaded "
                "final engine; includes resident model tensors and inference activations."
            ),
            "flops_scope": (
                "actual executed target/no-target inference path; "
                "PyTorch supported-op FLOP counting."
            ),
        },
        "parameter_stats": param_stats,
        "positive_case": positive,
        "no_target_case": no_target,
        "sample_source_grefs": str(gref_path) if gref_path else None,
    }

    out_json = ROOT / "benchmark_final_system_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(safe_jsonable(report), f, indent=2, ensure_ascii=False)

    # Compact TXT summary for easy copy into a report/table.
    out_txt = ROOT / "benchmark_final_system_summary.txt"
    lines = []
    lines.append("TG-UNet Final System - Formal Efficiency Benchmark")
    lines.append("=" * 72)
    lines.append(f"GPU: {report['environment']['gpu']}")
    lines.append(f"PyTorch: {torch.__version__}")
    lines.append("Batch size: 1")
    lines.append(f"Warmup: {args.warmup}")
    lines.append(f"Repeats: {args.repeats}")
    lines.append("")
    lines.append(
        f"Resident Params: "
        f"{param_stats['unique_total_params'] / 1e6:.3f} M"
    )
    lines.append("")

    def add_case(label: str, case: Optional[dict]):
        if not case:
            return
        lines.append(label)
        lines.append("-" * 72)
        fdata = case["flops"]
        mdata = case["memory"]
        ldata = case["latency"]

        if fdata.get("success"):
            lines.append(f"FLOPs: {fdata['gflops']:.3f} GFLOPs")
            lines.append(f"FLOPs method: {fdata['method']}")
        else:
            lines.append("FLOPs: unavailable")

        if mdata.get("available"):
            lines.append(
                f"Peak GPU allocated: "
                f"{mdata['peak_allocated_gib']:.3f} GiB"
            )
            lines.append(
                f"Peak GPU reserved: "
                f"{mdata['peak_reserved_gib']:.3f} GiB"
            )
            lines.append(
                f"Incremental inference allocation: "
                f"{mdata['incremental_peak_allocated_gib']:.3f} GiB"
            )

        lines.append(f"Mean latency: {ldata['mean_ms']:.3f} ms/image")
        lines.append(f"Median latency: {ldata['median_ms']:.3f} ms/image")
        lines.append(f"P95 latency: {ldata['p95_ms']:.3f} ms/image")
        lines.append(f"FPS: {ldata['fps_from_mean']:.3f}")
        lines.append("")

    add_case("TARGET FULL PATH", positive)
    add_case("NO-TARGET EARLY-EXIT PATH", no_target)

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 88)
    print("BENCHMARK FINISHED")
    print("=" * 88)
    print("JSON :", out_json)
    print("TXT  :", out_txt)
    print()
    print("For the model-comparison table, use the TARGET FULL PATH values")
    print("unless you explicitly label a dataset-average or no-target result.")
    print("Do not mix 'peak allocated' and 'peak reserved' memory across papers.")

    # Keep PIL references alive until all CUDA work is complete.
    _ = (pos_pil, nt_pil, smoke_output)


if __name__ == "__main__":
    main()
