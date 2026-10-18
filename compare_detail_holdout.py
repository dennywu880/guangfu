"""One-shot same-protocol holdout comparison of dev-selected checkpoints."""

import json
import os
from pathlib import Path

import torch

from dataset import RefELDataset
from detail_adapter_model import build_detail_model
from train_detail_adapter import evaluate, load_router
from train_text import FormalTextCache


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASE = RESULTS / "detail_adapter_disjoint_20260924"
WEIGHTED = RESULTS / "detail_adapter_small_weight_20260924"
CHECKPOINT = Path("head_detail") / "best.pth"


def concise(result):
    return {key: value for key, value in result.items() if key != "rows"}


@torch.no_grad()
def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    split_a = json.loads((BASE / "split_used.json").read_text(encoding="utf-8"))
    split_b = json.loads((WEIGHTED / "split_used.json").read_text(encoding="utf-8"))
    for field in ("dev_indices", "holdout_indices", "train_pool_indices"):
        if split_a[field] != split_b[field]:
            raise RuntimeError(f"Experiments have different {field}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = RefELDataset(ROOT, "val", 384, False)
    cache = FormalTextCache(ROOT)
    router = load_router()
    results = {}
    for name, directory in (("unweighted", BASE), ("small_weighted", WEIGHTED)):
        model = build_detail_model(ROOT / "checkpoints", device)
        model.load_checkpoint(directory / CHECKPOINT)
        result = evaluate(model, data, split_a["holdout_indices"], router,
                          device, cache, "val", 2)
        results[name] = concise(result)
        (directory / "holdout_selected_head_detail.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = WEIGHTED / "same_protocol_holdout_comparison.json"
    output.write_text(json.dumps({
        "protocol": "same 160 holdout references, same routing, no presence Gate",
        "checkpoint": str(CHECKPOINT),
        "note": "Holdout is diagnostic; original pretrained weights may have seen Val images.",
        "results": results,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
