# 2️⃣ Phase 2 — Preprocessing Pipeline

**Status:** complete · **Entry point:** `py src/data/preprocess.py`

## Pipeline

```
raw MLII signal (360 Hz)
   -> zero-phase Butterworth bandpass 0.5–40 Hz (order 4, sosfiltfilt)
   -> per-record z-score normalization (full-record mean/std)
   -> R-peak-centered windows: 0.4 s before / 0.6 s after = 360 samples
   -> AAMI grouping, keep N / S / V (F and Q excluded in v1)
   -> inter-patient split DS1 (train) / DS2 (test)
   -> data/processed/mitbih_nsv.npz
```

## Design decisions

1. **Expert annotations for training data.** Beat windows are cut on the
   cardiologist-annotated R peaks (`.atr`), the gold standard. A
   Pan-Tompkins detector (`preprocess.detect_r_peaks`, via NeuroKit2) is
   provided for *inference-time* use only — never mixed into label
   generation, so detection errors cannot contaminate training labels.
2. **Zero-phase filtering** (`sosfiltfilt`) — forward-backward filtering
   introduces no phase shift, so QRS timing and morphology are preserved.
3. **Per-record normalization** (mean/std of the whole filtered record)
   rather than per-beat — avoids amplitude info loss at the beat level and
   matches what a deployed device would do over a streaming buffer.
4. **Boundary beats dropped** — a window that crosses the start/end of a
   record is discarded (a few beats per record at most).
5. **Inter-patient split (DS1/DS2)** — the standard de Chazal-style
   partition; no patient's data appears in both train and test, so
   reported accuracy reflects generalization to unseen patients.

## Built dataset

`data/processed/mitbih_nsv.npz` — `X` (beats × 360), `y` (N/S/V),
`record`, `r` (R-peak sample), `train_mask`/`test_mask`.

| Set | Beats | N | S | V |
|---|---|---|---|---|
| DS1 (train) | 50,570 | 45,839 | 943 | 3,788 |
| DS2 (test) | 50,026 | 44,742 | 1,837 | 3,447 |
| **All** | **100,596** | **90,581** | **2,780** | **7,235** |

Counts match the published AAMI distribution for MIT-BIH within rounding,
confirming the mapping is correct.

## Class imbalance (must handle in training)

- N : S : V ≈ 90.6 k : 2.8 k : 7.2 k → S is only **2.8 %** of beats.
- Phase 3/4 models use `class_weight='balanced'`; evaluation reports
  macro-F1 and per-class recall, not just accuracy.

## Next

Phase 3 — classical baseline on RR-interval + morphology features.
