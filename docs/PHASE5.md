# Phase 5 — Lightweight Optimization (Full-Integer INT8)

**Status:** complete · **Entry point:** `py src/models/quantize.py`

## What was done

Post-training full-integer quantization (weights **and** activations, int8
input/output tensors) of the Phase 4 model, calibrated on 2,000 DS1 beats:

```
phase4_cnn.keras (float32)  --TFLiteConverter-->  float32 .tflite
                             --+Optimize.DEFAULT+representative dataset
                             --ops: TFLITE_BUILTINS_INT8-->  INT8 .tflite
```

## Engineering note (thesis-worthy finding)

TFLite's integer quantizer **cannot calibrate graphs whose first op is
Conv1D** (`conv.cc: input->dims->size != 4 (3 != 4)`, reproduced with both
the new and legacy quantizers). The Phase 4 architecture was therefore
rebuilt with `Conv2D((1, k))` kernels on a `(batch, 1, time, 1)` tensor —
mathematically identical (verified: retrained model reproduced the exact
Phase 4 test metrics), but quantizable. Building 1D-CNN ECG models in this
shape from the start avoids the issue entirely on any toolchain.

## Results (DS2, inter-patient, TFLite interpreter inference)

| Artifact | Size | Accuracy | Macro-F1 | N F1 | S F1 | V F1 |
|---|---|---|---|---|---|---|
| keras float32 (training format) | 211.2 KB | 94.39 % | 0.706 | 0.971 | 0.295 | 0.851 |
| TFLite float32 (converted) | 50.3 KB | 94.39 % | 0.706 | 0.971 | 0.295 | 0.851 |
| **TFLite INT8 (quantized)** | **23.8 KB** | **94.63 %** | **0.724** | 0.972 | **0.343** | 0.857 |

- **8.9x smaller** than the training artifact, **4.3x smaller** than the
  float TFLite — with **zero accuracy cost** (INT8 is marginally better;
  quantization noise acted as a mild regularizer, a commonly reported
  effect on small models).
- The INT8 result was produced with the actual TFLite interpreter and
  int8 input/output tensors — the same inference path an ESP32 takes in
  Phase 6, so no train/deploy mismatch.

## Deployment budget

23.8 KB model + ~40 KB tensor arena fits comfortably in ESP32-S3 SRAM
(512 KB) and even in Cortex-M0+ class MCUs with 64+ KB RAM.

## Future work (documented, not blocking)

- Quantization-aware training (QAT) — likely unnecessary here, but a
  natural ablation for the thesis.
- Structured pruning and knowledge distillation — the accuracy budget is
  already strong; these are additional comparison rows.

## Next

Phase 6 — TFLite Micro deployment on ESP32-S3 (firmware + model header).
