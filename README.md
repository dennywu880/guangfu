# Candidate B: TG-UNet with a Shared Encoder and Shared Decoder4

This repository contains the source code, checkpoint ancestry, and frozen evaluation records for Candidate B of the Ref-EL-Defect photovoltaic defect referring-segmentation project.

The repository is private. The dataset is intentionally not included. The recorded TestB results are a frozen re-evaluation of a historically used test split; they are not an untouched blind-test claim.

## Candidate B results

Recorded on 7,000 TestB referring expressions at 384 x 384 input size, with the frozen General path, Presence Gate v2, historical text/area routing rule, and mask threshold 0.5:

| Metric | Result |
|---|---:|
| gIoU | 0.614152 |
| cIoU | 0.709068 |
| Recall | 0.820154 |
| Precision | 0.839617 |
| F1 | 0.829772 |
| mPr | 0.623762 |

Candidate B shares decoder4 across the three expert branches. Relative to the all-private decoder candidate, the measured image-model parameter count falls from 38.023M to 33.302M (12.42%). This is a mixed accuracy tradeoff, not a uniform improvement. Definitions, resource-measurement limits, per-sample records, protocol, and file hashes are in `results/decoder_sharing_testB_frozen_20261009/REVIEW.md` and its JSON/CSV records.

## Repository contents

- Python source files are at the repository root, preserving the project's root-relative path assumptions.
- `checkpoints/` contains the General, three expert, and Presence Gate v2 source weights.
- `results/` contains the incremental ancestors needed to restore and load Candidate B, the Candidate B checkpoint, and its frozen evaluation records.
- `.gitattributes` configures PyTorch checkpoints for Git LFS.
- No images, annotation JSON, BERT feature cache, runtime prediction cache, or user-interface outputs are included.

Candidate B is stored as an incremental checkpoint, not a standalone `.pth` file. Keep the source checkpoint tree and filenames intact. `train_decoder_sharing.py` reconstructs the candidate from that tree; `evaluate_decoder_sharing_testB.py` performs the frozen evaluation and requires the dataset and text cache described below.

## Environment

The original environment snapshot is in `environment.yml` (Python 3.10, PyTorch 2.12.0 with CUDA 12.6, and the recorded Python dependencies). GPU driver compatibility and package availability can vary by operating system and hardware. Install a compatible PyTorch build from the official PyTorch instructions if the exact CUDA build is unavailable; do not treat the environment snapshot as proof that another machine has been validated.

Create the environment with:

```powershell
conda env create -f environment.yml
conda activate defect_ris
```

The evaluation scripts also import the local project's Transformers and PyTorch dependencies. For inference from arbitrary text, obtain the `bert-base-uncased` model/tokenizer through Hugging Face or place a compatible local copy where `model_text.py` expects it. The checked-in cached text embeddings are intentionally omitted: they are tied to the original dataset/reference IDs and are not a substitute for the tokenizer when evaluating new text.

## Data required for evaluation

Obtain Ref-EL-Defect separately and place the files in the following layout:

```text
datasets/
├── grefcoco/
│   ├── instances.json
│   └── grefs(unc).json
└── images/
    └── train2014_el/
```

The frozen TestB evaluator expects the matching formal text-feature cache under `text_cache/bert_pool_v1_b4_fp32_maxlen32/`. The cache builder and manifest checks are in `train_text.py`. Dataset access and any redistribution must follow the dataset's license and original terms.

## Reconstruct and evaluate Candidate B

From the repository root, with the required dataset and text cache installed, the training script can retrain only Candidate B into a new result directory:

```powershell
python train_decoder_sharing.py --candidates decoder4 --output candidateB_retrain
```

The frozen evaluator currently evaluates all four decoder layouts (all-private, decoder4, decoder4+decoder3, and full decoder); it does not expose a single-candidate filter:

```powershell
python evaluate_decoder_sharing_testB.py --output candidateB_testB_recheck
```

The checked-in `decoder4_epoch_02.pth` is already-trained Candidate B. Do not retrain or rerun TestB just to read the stored report. The dataset is excluded, and this repository has not been smoke-tested on a clean second computer. The JSON records preserve the original results without implying independent reproduction.

## Reproducibility and limitations

- Evaluation configuration and checkpoint SHA-256 values are recorded in the result JSON files.
- The TestB split was previously used in this project. The frozen run did not tune thresholds or select the checkpoint using TestB, but it is still not an independent blind test.
- The reported 128.3 ms/reference throughput is the original full evaluation pipeline measurement on an RTX 5060 Laptop GPU with cached text features. It is not an application FPS result and excludes BERT transformer execution and GUI rendering.
- The reported FLOPs are a profiler estimate with documented exclusions; see the review report before comparing to other papers.
- Large `.pth` files are tracked with Git LFS. A clone must have Git LFS installed and pull LFS objects.

## License

No license is provided in this repository yet. Do not assume that the code or weights may be redistributed outside the authorized collaborators until the project owner adds an appropriate license and verifies the source dataset and model-weight redistribution terms.
