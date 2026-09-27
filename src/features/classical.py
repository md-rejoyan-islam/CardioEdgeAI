"""Phase 3 — classical baseline: RR-interval + morphology features.

Two interpretable models (logistic regression, random forest) trained on
DS1 and evaluated on DS2. Purpose: a floor for the deep models and a
sanity check that the features carry real signal (especially RR intervals
for S/V discrimination).

Feature groups (28 total):
- RR features (5): pre-RR, post-RR, local mean RR (5 preceding beats),
  ratio pre-RR / local RR, RR difference (post - pre).
  NOTE: post-RR delays a streaming decision by one beat — acceptable for
  a monitor, flagged here for the thesis discussion.
- Morphology features (23): amplitudes and timings of the window
  (max/min/argmax/argmin in QRS and T regions), width above half-max,
  signed areas, slopes, and an 8-point coarse downsample of the window.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MODELS_DIR, SEED  # noqa: E402
from src.evaluation.evaluate import evaluate, load_processed_dataset  # noqa: E402

BEFORE = 144  # samples before R peak (0.4 s at 360 Hz)


def rr_features(r: np.ndarray, record: np.ndarray) -> np.ndarray:
    """RR features computed per record (no cross-patient leakage)."""
    feats = np.zeros((len(r), 5), dtype=np.float64)
    for rec in np.unique(record):
        idx = np.where(record == rec)[0]
        rr = np.diff(r[idx])  # RR intervals between consecutive beats
        # rr[i] connects beat i and beat i+1 within this record
        for k, i in enumerate(idx):
            pre = rr[k - 1] if k >= 1 else np.nan
            post = rr[k] if k < len(rr) else np.nan
            local = np.nanmean(rr[max(0, k - 5):k]) if k >= 1 else np.nan
            feats[i] = (
                pre, post, local,
                pre / local if local and local == local else np.nan,
                post - pre,
            )
    # replace NaN (first beat of a record) with that record's median RR
    col_median = np.nanmedian(feats, axis=0)
    nan_mask = ~np.isfinite(feats)
    feats[nan_mask] = np.take(col_median, np.where(nan_mask)[1])
    return feats


def morphology_features(X: np.ndarray) -> np.ndarray:
    """Amplitude/shape features per beat window (z-scored units)."""
    qrs = X[:, :BEFORE + 36]  # R peak +-0.1 s
    t_reg = X[:, BEFORE + 36:]  # remaining ST/T part of the window
    f = []
    for arr, prefix in ((X, "w"), (qrs, "q"), (t_reg, "t")):
        f += [arr.max(1), arr.min(1), arr.argmax(1), arr.argmin(1),
              arr.std(1), np.trapezoid(arr, axis=1)]
    # window-specific extras
    amp = X.max(1) - X.min(1)
    above = np.abs(X) > (np.abs(X).max(1, keepdims=True) * 0.5)
    width = above.sum(1)
    rising = (X[:, BEFORE] - X[:, BEFORE - 72]) / 72.0
    falling = (X[:, BEFORE + 72] - X[:, BEFORE]) / 72.0
    coarse = X[:, ::45]  # 8-point downsample of the 360-sample window
    feats = np.column_stack(
        [f[0], f[1], f[2], f[3], f[4], f[5],
         f[6], f[7], f[8], f[9], f[10], f[11],
         f[12], f[13], f[14], f[15], f[16], f[17],
         amp, width, rising, falling] + [coarse[:, i] for i in range(8)])
    return feats


def main() -> None:
    X, y, r, record, train_mask, test_mask = load_processed_dataset()
    print("building features ...")
    F = np.column_stack([rr_features(r, record),
                         morphology_features(X.astype(np.float64))])
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(F[train_mask])
    Ftr, Fte = scaler.transform(F[train_mask]), scaler.transform(F[test_mask])
    ytr, yte = y[train_mask], y[test_mask]

    MODELS_DIR.mkdir(exist_ok=True)
    models = {
        "Logistic Regression (RR+morphology)":
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED),
        "Random Forest (RR+morphology)":
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1),
    }
    for name, model in models.items():
        print(f"training {name} ...")
        model.fit(Ftr, ytr)
        evaluate(name, yte, model.predict(Fte))
        slug = name.split(" ")[0].lower()
        joblib.dump({"model": model, "scaler": scaler},
                    MODELS_DIR / f"phase3_{slug}.joblib")
    print("models saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
