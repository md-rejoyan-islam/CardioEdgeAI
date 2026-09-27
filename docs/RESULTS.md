# Consolidated Results

All evaluation on **MIT-BIH DS2 (inter-patient)**: 50,026 beats
(44,742 N / 1,837 S / 3,447 V) from 22 recordings whose patients were
never seen in training. Primary metric: macro-F1 (classes are ~90/3/7 %
imbalanced). Machine-readable numbers: `docs/results.json`.

## Model comparison

| # | Model | Params | Size | Acc | Macro-F1 | N F1 | S F1 | V F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression (RR+morph.) | — | — | 84.3 % | 0.603 | 0.914 | 0.240 | 0.655 |
| 2 | Random Forest (RR+morph.) | — | — | 94.3 % | 0.650 | 0.971 | 0.100 | 0.880 |
| 3 | 1D CNN, morphology only (rejected) | 11,443 | — | 81.1 % | 0.522 | 0.888 | 0.042 | 0.634 |
| 4 | 1D CNN + RR branch (float32) | 11,571 | 211 KB | 94.4 % | 0.706 | 0.971 | 0.295 | 0.851 |
| 5 | **Same, TFLite INT8 (deployed)** | 11,571 | **23.8 KB** | **94.5 %** | **0.713** | 0.971 | 0.313 | 0.856 |

Row 3 is kept deliberately: the morphology-only CNN documents the
negative result that motivates the RR branch (S beats are defined by
timing, not shape).

## Compression path (model 4 → 5)

| Stage | Size | Reduction |
|---|---|---|
| keras float32 (training artifact) | 211.2 KB | — |
| TFLite float32 | 50.3 KB | 4.2x |
| **TFLite full-integer INT8** | **23.8 KB** | **8.9x** |

Accuracy **improved** slightly after INT8 (+0.08 pp accuracy, +0.007
macro-F1) — no quantization damage to recover, so QAT was not needed.

## Efficiency (deploy/tflite/benchmark.py, laptop CPU, 2 threads)

| Metric | Value |
|---|---|
| Mean latency / beat | 0.02 ms |
| p95 latency | 0.04 ms |
| Throughput | ~54,600 beats/s |
| Real-time requirement | 1-2 beats/s |
| ESP32-S3 memory budget | 23.8 KB model + 48 KB arena + 14.4 KB buffer ≪ 512 KB SRAM |

On-board ESP32 latency: to be measured at hardware bring-up
(see docs/PHASE6.md open items).

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
py src/data/visualize.py          # Phase 1 figures
py src/data/preprocess.py         # Phase 2 dataset
py src/features/classical.py      # Phase 3
py src/models/cnn.py              # Phase 4 (seed 42, deterministic)
py src/models/quantize.py         # Phase 5
py deploy/tflite/export_model_header.py  # Phase 6 model.h
py deploy/tflite/benchmark.py
```

Environment: Python 3.14.2, Windows 11; versions pinned in
requirements.txt.
