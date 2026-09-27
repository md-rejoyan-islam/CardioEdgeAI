# 📊 Consolidated Results

All evaluation on **MIT-BIH DS2 (inter-patient)**: 50,026 beats
(44,742 N / 1,837 S / 3,447 V) from 22 recordings whose patients were
never seen in training. Primary metric: macro-F1 (classes are ~90/3/7 %
imbalanced). Machine-readable numbers: `docs/results.json`.

## Model comparison

| # | Model | Params | Acc | Macro-F1 | N F1 | S F1 | V F1 |
|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression (RR+morph.) | — | 84.3 % | 0.603 | 0.914 | 0.240 | 0.655 |
| 2 | Random Forest (RR+morph.) | — | 94.3 % | 0.650 | 0.971 | 0.100 | 0.880 |
| 3 | 1D CNN, morphology only (rejected) | 11,443 | 81.1 % | 0.522 | 0.888 | 0.042 | 0.634 |
| 4 | 1D CNN + RR (float32) | 11,571 | 94.4 % | 0.706 | 0.971 | 0.295 | 0.851 |
| 5 | v2 float (S-oversampling, not deployed) | 11,571 | 93.7 % | **0.718** | 0.966 | **0.389** | 0.800 |
| 6 | **v1, TFLite INT8 (deployed)** | 11,571 | **94.3 %** | 0.712 | 0.970 | 0.320 | 0.847 |

Row 3 documents the negative result that motivates the RR branch.
Row 5 is the best *float* model but is not post-training-quantization
robust (below); the deployed artifact remains v1 INT8.

## Training ablations (DS2 macro-F1)

| Recipe | Macro-F1 | Note |
|---|---|---|
| v1: weighted CE {N:0.6, S:8, V:2} | 0.706 | production recipe |
| v2a: wide (24/48/96) + focal(γ=2) + oversamp S15%/V12% | 0.643 | S collapsed |
| v2b: wide + oversamp S8% + v1 weights | 0.691 | S up, V down |
| **v2: v1 width + oversamp S8% + v1 weights** | **0.718** | best float |
| morphology-only CNN (any weighting) | ≤ 0.522 | S undetectable |

## Quantization robustness study (why v2 is not deployed)

Post-training full-integer quantization of the v2 model under three
calibration strategies:

| Calibration | v2 INT8 accuracy |
|---|---|
| random 2,000 beats (~99 % N) | 86.1 % |
| all S+V beats + N fill | 33.0 % |
| minorities capped at 25 % | 82.8 % |

The oversampled model's decision margins are too tight for PTQ under
every strategy, while v1 quantizes with *zero* loss (94.3 % INT8 vs
94.4 % float). Findings: (1) oversampling improves float S-detection but
manufactures quantization fragility; (2) deployed edge models must be
re-validated after every training change, not assumed stable; (3) QAT is
the natural future fix for v2. The representative dataset is now
stratified (minorities capped at 25 %) on principle, and v1 keeps its
accuracy under it.

## R-peak detector selection (inference-time path)

Scored against gold annotations on all 48 records, greedy one-to-one
matching within 50 ms (`src/evaluation/r_detector_eval.py`,
`docs/r_detector_eval.json`):

| Method | Sensitivity | Precision | F1 |
|---|---|---|---|
| pantompkins1985 (original choice) | 0.658 | 0.603 | 0.629 |
| hamilton2002 | 0.963 | 0.884 | 0.922 |
| engzeemod2012 | 0.930 | 0.919 | 0.925 |
| **neurokit (now default)** | **0.974** | 0.896 | **0.934** |

Pan-Tompkins — the textbook default — missed one third of beats on the
filtered MLII signals; switching the default detector is the single
largest end-to-end accuracy fix in the deployed pipeline.

## Compression path (v1, deployed)

| Stage | Size | Reduction |
|---|---|---|
| keras float32 (training artifact) | 211.2 KB | — |
| TFLite float32 | 50.3 KB | 4.2x |
| **TFLite full-integer INT8** | **23.8 KB** | **8.9x** |

## Efficiency (deploy/tflite/benchmark.py, laptop CPU, 2 threads)

| Metric | Value |
|---|---|
| Mean latency / beat | 0.02 ms |
| p95 latency | 0.04 ms |
| Throughput | ~54,600 beats/s |
| Real-time requirement | 1-2 beats/s |
| ESP32-S3 memory budget | 23.8 KB model + 48 KB arena + 14.4 KB buffer ≪ 512 KB SRAM |

## Dataset summary (data/processed/mitbih_nsv.npz)

| Set | Beats | N | S | V |
|---|---|---|---|---|
| DS1 fit (training) | 40,320 | 36,397 | 773 | 3,150 |
| DS1 val (early stopping: 215/220/223/230) | 10,250 | 9,442 | 170 | 638 |
| DS2 (test) | 50,026 | 44,742 | 1,837 | 3,447 |
| Total | 100,596 | 90,581 | 2,780 | 7,235 |

## Reproducing everything

```bash
py -m pip install -r requirements.txt
py src/data/download_mitbih.py
py src/data/visualize.py                # Phase 1 figures
py src/data/preprocess.py               # Phase 2 dataset
py src/features/classical.py            # Phase 3
py src/models/cnn.py                    # Phase 4 v1 (seed 42)
py src/models/cnn_v2.py                 # Phase 4b ablation model
py src/evaluation/r_detector_eval.py    # detector comparison
py src/models/quantize.py               # Phase 5 (v1 -> INT8)
py deploy/tflite/export_model_header.py # Phase 6 model.h
py deploy/tflite/benchmark.py
py tests/test_live_pipeline.py          # end-to-end live-path test
```

Environment: Python 3.14.2, Windows 11; versions pinned in
requirements.txt.
