# 3️⃣ Phase 3 — Classical Baseline (RR + Morphology Features)

**Status:** complete · **Entry point:** `py src/features/classical.py`

## Models

Two interpretable models trained on DS1 (50,570 beats), evaluated on DS2
(50,026 beats), both with `class_weight="balanced"`:

1. **Logistic Regression** (standardized features, max_iter 2000)
2. **Random Forest** (300 trees)

## Features (28 per beat)

- **RR features (5)** — pre-RR, post-RR, local mean of the 5 preceding RR
  intervals, pre-RR / local-RR ratio, post-RR − pre-RR. Computed *within
  each record* so no patient leakage. Post-RR delays a streaming decision
  by one beat (one heartbeat of latency) — noted for the thesis discussion.
- **Morphology features (23)** — amplitudes/positions/std/signed area of
  the full window, the QRS region (R ± 0.1 s), the ST/T region, beat
  amplitude, half-max width, rising/falling slopes at R, and an 8-point
  downsample of the 360-sample window.

## Results (DS2, inter-patient)

| Model | Accuracy | Macro-F1 | N recall | S recall | V recall |
|---|---|---|---|---|---|
| Logistic Regression | 84.3 % | 0.603 | 0.862 | 0.477 | 0.800 |
| Random Forest | **94.3 %** | **0.650** | 0.984 | **0.066** | 0.887 |

Confusion matrices: `figures/logistic-regression-(rr+morphology)_confusion.png`,
`figures/random-forest-(rr+morphology)_confusion.png`.
Full metrics: `docs/results.json`. Models: `models/phase3_*.joblib`.

## Interpretation

- RR intervals alone carry strong V information (wide/premature beats),
  which both models pick up (V recall ≥ 0.80).
- **S beats remain the hard class**: the RF trades almost all S recall for
  N precision (S recall 6.6 %). This is consistent with the literature —
  inter-patient S detection from hand-crafted features is the classic
  weak spot (de Chazal & Camps-style pipelines report ≈ 35–45 % S
  sensitivity). Fine morphology learned by a CNN is the expected remedy,
  which motivates Phase 4.
- Plain accuracy (94.3 %) is misleading under this imbalance — macro-F1
  is the primary comparison metric for the rest of the thesis.

## Next

Phase 4 — 1D CNN baseline on raw beat windows.
