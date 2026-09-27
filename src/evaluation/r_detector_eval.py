"""Measure R-peak detector quality against MIT-BIH gold annotations.

The deployed pipeline classifies beats cut on the R detector (no
annotations exist at inference time), so detector quality directly caps
end-to-end accuracy. This script scores NeuroKit2's detector variants on
all 48 records and writes docs/r_detector_eval.json.

Matching: one-to-one greedy within 50 ms of a gold R peak (standard R
detector tolerance). Gold beats = annotated QRS events in the v1 scope
groups N/S/V (non-beat annotations like rhythm changes are excluded).
"""
import json
import sys
from pathlib import Path

import neurokit2 as nk
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MITDB_RECORDS  # noqa: E402
from src.data.preprocess import SYMBOL_TO_GROUP, load_record  # noqa: E402

TOLERANCE_S = 0.05
METHODS = ("pantompkins1985", "hamilton2002", "engzeemod2012", "neurokit")
OUT = Path(__file__).resolve().parents[2] / "docs" / "r_detector_eval.json"


def match(detected: np.ndarray, gold: np.ndarray, tol_samples: int):
    """Greedy one-to-one matching; returns (tp, fp, fn)."""
    used = np.zeros(len(gold), dtype=bool)
    tp = 0
    for d in detected:
        if len(gold) == 0:
            break
        dist = np.abs(gold - d)
        i = int(np.argmin(dist))
        if dist[i] <= tol_samples and not used[i]:
            used[i] = True
            tp += 1
    fp = len(detected) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def main() -> None:
    tol = int(TOLERANCE_S * 360)
    results = {}
    for method in METHODS:
        tp = fp = fn = 0
        for record_id in MITDB_RECORDS:
            signal, samples, symbols = load_record(record_id)
            gold = np.array([s for s, sym in zip(samples, symbols)
                             if SYMBOL_TO_GROUP.get(sym) in ("N", "S", "V")])
            try:
                info = nk.ecg_peaks(signal, sampling_rate=360,
                                    method=method)[1]
            except Exception as exc:  # noqa: BLE001
                print(f"{method} failed on {record_id}: {exc}")
                continue
            detected = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
            d_tp, d_fp, d_fn = match(detected, gold, tol)
            tp += d_tp; fp += d_fp; fn += d_fn
        se = tp / (tp + fn) if tp + fn else 0.0
        ppv = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * se * ppv / (se + ppv) if se + ppv else 0.0
        results[method] = {"sensitivity": round(se, 4),
                           "precision": round(ppv, 4),
                           "f1": round(f1, 4),
                           "tp": tp, "fp": fp, "fn": fn}
        print(f"{method:16s} Se={se:.4f}  +P={ppv:.4f}  F1={f1:.4f}  "
              f"(tp={tp} fp={fp} fn={fn})")
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best = max(results, key=lambda m: results[m]["f1"])
    print(f"\nbest: {best} -> results saved to {OUT}")


if __name__ == "__main__":
    main()
