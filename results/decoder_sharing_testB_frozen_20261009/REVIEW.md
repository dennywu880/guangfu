# Decoder Sharing Recheck and Frozen TestB Evaluation

## Scope

The decoder comparison is conditional on the existing fully shared encoder candidate (`split_ratio_existing_best_20261001/shared/epoch_02.pth`). The four tested decoder layouts are all-private, share decoder4, share decoder4+decoder3, and share the full decoder/output. All use the same fixed General path, route rule, 384x384 input and 0.5 mask threshold.

TestB was evaluated after the structures and checkpoints were fixed. Its historical project exposure is documented in `protocol.json`; this is a frozen re-evaluation, not an untouched blind test. No threshold or checkpoint selection used these TestB results.

## Repeated Development Check

Each run used the same 1,200-reference development set, two training epochs of 1,200 references each, and the same optimizer recipe. The first run is the original seed; two additional random seeds were run for the all-private and decoder4 candidates.

The seed 20261009 job was stopped after both requested candidates completed epoch 2 because the original launcher also began evaluating unrelated layouts. Their saved epoch-2 results are complete, but that job has no overall completion marker. Seed 20261010 reran only the two requested candidates and completed successfully; both jobs kept separate output directories.

| Seed | All-private gIoU | Decoder4 gIoU | Difference | All-private cIoU | Decoder4 cIoU | Difference |
|---|---:|---:|---:|---:|---:|---:|
| 20261008 | 0.586957 | 0.585907 | -0.001050 | 0.703824 | 0.703574 | -0.000250 |
| 20261009 | 0.585246 | 0.584504 | -0.000741 | 0.703775 | 0.702047 | -0.001728 |
| 20261010 | 0.586027 | 0.584621 | -0.001405 | 0.707132 | 0.706621 | -0.000511 |

The decoder4 candidate had slightly lower development gIoU and cIoU in all three runs. The observed gIoU losses were 0.0007-0.0014 and cIoU losses were 0.0003-0.0017. This is a small, directionally consistent difference under this short training recipe; it does not establish that decoder4 sharing is unstable or that longer training would behave the same way.

## TestB Segmentation Metrics

Each row contains 7,000 references (5,834 positive and 1,166 no-target). gIoU, cIoU, Recall, Precision, F1 and mPr use the frozen step-43 definitions. mPr is the mean of Pr@0.1 through Pr@0.9; the individual Pr values are in each candidate JSON.

| Decoder layout | gIoU | cIoU | Recall | Precision | F1 | mPr |
|---|---:|---:|---:|---:|---:|---:|
| All-private | 0.612241 | 0.709546 | 0.818898 | 0.841610 | 0.830099 | 0.621254 |
| Share decoder4 | 0.614152 | 0.709068 | 0.820154 | 0.839617 | 0.829772 | 0.623762 |
| Share decoder4+decoder3 | 0.610589 | 0.707016 | 0.820550 | 0.836331 | 0.828365 | 0.619730 |
| Share full decoder/output | 0.609891 | 0.707006 | 0.817015 | 0.840020 | 0.828358 | 0.618778 |

Relative to all-private, decoder4 sharing changes gIoU by +0.001911, cIoU by -0.000478, Recall by +0.001256, Precision by -0.001993, F1 by -0.000327 and mPr by +0.002508. It is a mixed tradeoff, not a uniform accuracy improvement. Sharing more decoder stages lowers both gIoU and cIoU compared with all-private on this TestB run.

Positive mean IoU by annotated defect size:

| Decoder layout | Ultra-tiny | Tiny | Larger |
|---|---:|---:|---:|
| All-private | 0.299360 | 0.489026 | 0.598958 |
| Share decoder4 | 0.304188 | 0.491612 | 0.600902 |
| Share decoder4+decoder3 | 0.294297 | 0.482364 | 0.599422 |
| Share full decoder/output | 0.283982 | 0.483117 | 0.598368 |

Decoder4 sharing has slightly higher per-reference IoU in these three size groups, while its pooled cIoU is marginally lower overall. This reflects the different weighting of mean-per-reference IoU and pooled pixel IoU; neither metric should be substituted for the other.

## Model Size and Runtime Measurements

| Decoder layout | Image-model parameters | Image + one BERT-base | Active route + one BERT-base | Image-model parameter reduction | Profiled FLOPs/image | Measured GPU peak* | Evaluation pipeline / reference* |
|---|---:|---:|---:|---:|---:|---:|---:|
| All-private | 38.023M | 147.505M | 138.964M | 0.00% | 71.069G | 0.719 GiB | 128.3 ms |
| Share decoder4 | 33.302M | 142.784M | 138.964M | 12.42% | 71.069G | 0.701 GiB | 128.3 ms |
| Share decoder4+decoder3 | 32.121M | 141.604M | 138.964M | 15.52% | 71.069G | 0.696 GiB | 128.3 ms |
| Share full decoder/output | 31.549M | 141.031M | 138.964M | 17.03% | 71.069G | 0.694 GiB | 128.3 ms |

The BERT-base count (109,482,240 parameters) is added once to the cached-text image-model counts. Sharing cuts stored image-model parameters, but every single routed inference still activates the same number of parameters. It therefore does not reduce the measured FLOPs in this design. The small peak-memory differences follow the reduced resident image weights; they do not demonstrate a speedup.

The FLOP figure is a PyTorch profiler estimate for the General path plus one routed expert at 384x384. It excludes the external BERT transformer, Presence Gate, GUI, and unsupported profiler operators. Peak CUDA memory is the peak recorded during each candidate's routed expert pass after resetting CUDA counters; it does not include the earlier shared General/Gate pass. The 128.3 ms/reference figure is the full evaluation pipeline measured on an RTX 5060 Laptop GPU, including image loading, cached text-feature lookup, General/Gate/Router and routed expert evaluation. It excludes GUI rendering and BERT transformer execution. Treat it as an evaluation-throughput measurement, not a standalone application FPS benchmark.

Checkpoint byte sizes in the JSON are incremental candidate files, not self-contained deployment packages. Each depends on the shared encoder and its recorded ancestor checkpoints.

## Conclusion

Among these four candidates, sharing decoder4 gives the largest tested parameter reduction with the smallest measured accuracy tradeoff. TestB shows slight gains in mean-per-reference measures and defect-size IoUs, alongside small declines in pooled cIoU, Precision and F1. The extra-seed development check consistently favors all-private by a small margin. These results support keeping decoder4 sharing as a compression candidate, but do not establish a uniquely best deployment choice or a generalization result independent of prior TestB use. Deeper decoder sharing has no demonstrated accuracy advantage here despite greater compression.

Full per-reference records, all threshold-precision values, resource fields and hashes are in `comparison.json`, the four candidate JSON files and their sample CSV files. The original training candidates and source checkpoints were preserved.
