# Phase 4 — 1D CNN Baseline (Morphology + RR Branch)

**Status:** complete · **Entry point:** `py src/models/cnn.py`

## Architecture (11,571 parameters, 209 KB keras file)

```
beat window (360 x 1)
  -> Conv1D(16, k=7) -> BN -> ReLU -> MaxPool(2)
  -> Conv1D(32, k=5) -> BN -> ReLU -> MaxPool(2)
  -> Conv1D(64, k=3) -> BN -> ReLU -> GlobalAvgPool   (64 features)
                                                        \
RR timing (4 scalars) --------------------------------> concat -> Dense(32)
                                                       -> Dropout(0.3)
                                                       -> Dense(3, softmax)
```

## The key finding of this phase

A **morphology-only CNN cannot detect S beats** inter-patient. Three
tuning rounds (all patient-separated, early-stopped on a held-out record
group):

| Round | Input | Class weights | Acc | Macro-F1 | S recall |
|---|---|---|---|---|---|
| 1 | morphology only | balanced | 71.9 % | 0.517 | 0.31 (N recall 0.72) |
| 2 | morphology only | sqrt-balanced | 81.1 % | 0.522 | 0.07 |
| 3 | **+ RR branch** | {N:0.6, S:8, V:2} | **94.4 %** | **0.706** | **0.22** |

Supraventricular ectopic beats (S) are defined by *premature timing*,
not shape — the RR branch (pre-RR, post-RR, local 5-beat mean RR,
pre/local ratio) supplies exactly that information. This mirrors the
classical-feature finding of Phase 3 and is a central thesis point:
**cheap timing features + small CNN beat the alternatives at this budget.**

RR features are normalized with *fixed physiological constants*
(mean 0.8 s, std 0.3 s), not dataset statistics, so the identical
computation runs on the ESP32 firmware (Phase 6).

## Final results (DS2, inter-patient)

| Model | Acc | Macro-F1 | N F1 | S F1 | V F1 |
|---|---|---|---|---|---|
| Random Forest (Phase 3) | 94.3 % | 0.650 | 0.971 | 0.100 | 0.880 |
| **1D CNN + RR (this)** | **94.4 %** | **0.706** | 0.971 | **0.295** | 0.851 |

- Beats the RF on macro-F1 with equal accuracy and every per-class F1.
- S recall 0.22 at precision 0.44 — still the hard class (consistent
  with the inter-patient literature); future work: focal loss,
  per-patient calibration, larger S sampling.
- Training: 25 epochs max, early stopping (patience 8) on records
  215/220/223/230 held out from DS1; seed 42, fully reproducible
  (re-run matched the tuning run exactly).

## Artifacts

- Model: `models/phase4_cnn.keras` (git-ignored; regenerate by running the script)
- Confusion matrix: `figures/1d-cnn-+-rr-(14k-params)_confusion.png`
- Metrics: `docs/results.json` → key `1D CNN + RR (14k params)`

## Next

Phase 5 — INT8 quantization of this model (target: ~4x smaller, edge-ready).
