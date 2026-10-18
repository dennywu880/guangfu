# Shared-encoder prototype

The untouched Final V3 is at `D:\thunet新\TG_UNet\TG_UNet`.
The complete backup is in the sibling `tgunet_final_v3_backup_20260923` directory.
This directory is the isolated experimental copy.

`shared_moe.py` loads the frozen general model's ResNet34 and BERT once.
Only each expert's text gates, decoder and mask head are copied from its
existing checkpoint. The general model's mask, presence gate, and fixed
text/area router remain in the inference flow. The general image features
are computed once and reused by the selected expert.

Run `python check_shared_moe.py --samples 12` with the project's Python
environment to compare expert-only IoU on identical validation examples.
The presence gate is deliberately omitted from that probe; the result is
not the official complete-system TestB metric. Do not select hyperparameters
against TestB. The initialized shared backbone is *not* guaranteed to match
the separate expert backbones. Fine-tune/distill on Train, select on Val,
then compare both full systems once on untouched TestB.

`TG_SHARED_MOE=1` selects the experimental shared architecture in
`inference_engine.py`. Unset or `TG_SHARED_MOE=0` keeps the higher-scoring
original independent experts as the default. The D: installation is unchanged.
In the current same-sample 48-reference Val probe (without Presence Gate),
original mean IoU is 0.6553 versus 0.6126 for the shared initialization.
This small exploratory sample does not establish full-system performance.
Using the `larger` expert's ResNet34 as the single shared image encoder
improved the 120-reference probe to cIoU **0.7140** versus **0.7159** for
the independent experts, with **167,755,780** versus **210,325,124** runtime
parameters. A separate 500-reference holdout gave cIoU **0.7503** versus
**0.7497**, while mean per-image IoU was lower (**0.6079** versus **0.6428**).
The holdout also showed a small increase in false activation on no-target
examples. These results justify this version as a compression candidate,
not as a claimed accuracy improvement.
After three epochs of head-only tuning (360 Train samples per epoch), a
fixed 120-sample validation comparison using exactly the same references
and the same router gave original cIoU **0.7159** vs shared cIoU **0.6502**.
These are expert-only metrics without Presence Gate, so neither should be
reported as full-system TestB performance. The shared initialization's
runtime parameters fell from 210,325,124 to 146,471,108, both counts
excluding the separately loaded Presence Gate and duplicate BERT copies.
Run `python train_shared_heads.py --epochs 3 --train-samples 1200 --val-samples 240`
to tune only the expert heads on Train, saving a checkpoint each epoch and
selecting `results/shared_moe_training/best_heads.pth` on Val. Set
`TG_SHARED_HEADS` to that file to evaluate the tuned heads. The script uses
fixed router decisions and never trains on the TestB split.
The old launcher executable has a hard-coded Python path and is not a
launcher for this copy; start `app.py` with a configured project environment.

For the installed Python 3.14 environment without Gradio, `local_ui.py`
offers a small localhost page using the same `DefectInferenceEngine`.
Set `HF_HOME=D:\thunet新\hf-cache` and offline Hugging Face mode.
Run baseline with `TG_SHARED_MOE=0`, `TG_LOCAL_PORT=8790`; run experimental
with `TG_SHARED_MOE=1`, `TG_SHARED_HEADS` set to the best heads checkpoint,
and `TG_LOCAL_PORT=8791`. Run `smoke_local_ui.py` to check both paths.

The pre-repair snapshot is `tgunet_shared_moe_pre_repair_20260923`.

## Small-defect failure check (2026-09-23)

The paired 500-reference Val audit is saved in
`results/full_system_larger_val_comparison.json`; its per-case analysis is in
`results/shared_failure_analysis.json`. The original and shared systems have
nearly equal cIoU (0.778752 vs 0.778967), but mean per-image IoU falls from
0.672127 to 0.644251. Of 500 cases, 83 improve, 172 regress and 245 tie.
The largest regressions concentrate in tiny and ultra-tiny annotated defects.
`results/shared_failure_gallery.png` shows missed or displaced thin targets,
as well as some cases where the shared model improves. This pattern is
consistent with the small heads receiving features from a different image
encoder than the one used to train them; it is not proof of that mechanism.

`probe_small_defect_thresholds.py` tested a reversible inference-only change
on fresh Val examples, grouped by source image. It excluded the earlier 120
and 500 reference images, used 160 image-disjoint examples to select the two
small-branch thresholds, and evaluated them on another 160 distinct images.
The selected thresholds were 0.25 for both small branches, versus the
unchanged 0.5. On the independent holdout, original mean IoU was 0.622574;
shared at 0.5 was 0.578785; shared with selected thresholds was 0.580802.
Shared cIoU changed from 0.749381 to 0.748593. The full record is in
`results/small_defect_threshold_probe.json`. This minor gain does not repair
the regression, so the threshold change is **not enabled** in the UI or model.
The original system remains the default. These are Val diagnostics, not
final TestB results or a claim of model improvement.

## One-encoder prototype (2026-09-24)

The isolated `train_single_encoder.py` experiment reuses General ResNet34
features for the chosen expert head and runs one image encoder rather than
General plus a second expert encoder. See
`results/single_encoder_experiment_20260924.md` for the protocol, same-GPU
complexity audit, training checkpoints, and negative validation findings.
It reduces structural operation count but currently loses segmentation IoU;
therefore it is **not deployed** and the original model remains the default.
