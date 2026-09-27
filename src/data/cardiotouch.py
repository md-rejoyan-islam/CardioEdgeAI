"""Phase 7 — external-validation adapter for Bionet CardioTouch 3000 exports.

Runs the deployed INT8 model on ECG recorded by the lab machine, through
the *inference-time* path (Pan-Tompkins R detection — no gold-standard
annotations exist for own recordings). This measures real-world
generalization: different amplifier, different electrodes, rest-ECG
12-lead context, exported lead II.

Supported export formats (auto-detected):
  - CSV / text with two columns: sample index or seconds, millivolts
    (comma, semicolon or whitespace separated; header row tolerated)
  - WAV (16-bit PCM, single channel, any rate)

The CardioTouch 3000 can also export SCP-ECG / vendor binary / PDF; those
need the actual files before a parser can be written honestly — the
protocol below assumes the lab exports plain CSV or WAV (BMS-Plus
workstation supports it). If only SCP-ECD is available, convert first
(e.g. with an SCP-ECG viewer) to CSV.

Usage:
    py src/data/cardiotouch.py --file exported_ecg.csv

Ethics (from the thesis protocol):
  - recordings taken by a trained ECG technician / under clinician
    supervision, with informed consent
  - files anonymized before entering this pipeline (no names/IDs in the
    file content; keep the linking log offline)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sps

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MODELS_DIR, SAMPLING_RATE_HZ  # noqa: E402
from src.data.preprocess import (bandpass, detect_r_peaks,  # noqa: E402
                                 segment_beats)
from src.models.cnn import RR_MEAN_S, RR_STD_S, rr_feature_matrix  # noqa: E402


def load_ecg_file(path: Path):
    """Return (signal_mv, fs_hz) from CSV/text or WAV, auto-detected."""
    if path.suffix.lower() == ".wav":
        from scipy.io import wavfile
        fs, data = wavfile.read(path)
        signal = data.astype(np.float64)
        if signal.ndim > 1:
            signal = signal[:, 0]
        # scale 16-bit PCM to an approximate mV range
        peak = np.abs(signal).max() or 1
        return signal / peak * 2.0, fs

    # plain CSV/text
    for delim in (",", ";", None):
        try:
            arr = np.loadtxt(path, delimiter=delim, ndmin=2)
            break
        except ValueError:
            arr = None
    if arr is None or arr.shape[1] < 2:
        raise ValueError(
            f"could not parse {path} as two-column (index/s, mV) data")

    x, y = arr[:, 0], arr[:, 1]
    dt = np.median(np.diff(x))
    if dt <= 0:
        raise ValueError("first column is not monotonic (index/seconds)")
    fs = 1.0 / dt
    if fs < 10:  # column was sample index, not time — rate not inferable
        raise ValueError(
            "first column looks like sample indices; export with a time "
            "column or pass --fs explicitly")
    return y, fs


def resample(signal: np.ndarray, fs: float) -> np.ndarray:
    if abs(fs - SAMPLING_RATE_HZ) < 0.5:
        return signal
    n = int(len(signal) * SAMPLING_RATE_HZ / fs)
    return sps.resample_poly(signal, SAMPLING_RATE_HZ, int(round(fs)))


def classify_recording(path: Path, fs_override: float | None = None) -> None:
    import tensorflow as tf

    raw, fs = load_ecg_file(path)
    if fs_override:
        fs = fs_override
    print(f"loaded {path.name}: {len(raw)} samples @ {fs:.1f} Hz")
    signal = bandpass(resample(raw, fs))

    r_peaks = detect_r_peaks(signal)
    print(f"Pan-Tompkins: {len(r_peaks)} R peaks detected "
          f"({len(r_peaks) / (len(signal) / SAMPLING_RATE_HZ):.1f} bpm mean)")

    beats, _, _, _ = segment_beats(signal, r_peaks, ["N"] * len(r_peaks), "ct")
    if not beats:
        print("no complete beat windows — recording too short")
        return
    X = np.asarray(beats, dtype=np.float32)
    RR = rr_feature_matrix(r_peaks, np.array(["ct"] * len(r_peaks)))

    blob = (MODELS_DIR / "phase5_cnn_int8.tflite").read_bytes()
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    sig_in, rr_in = interpreter.get_input_details()[:2]
    out = interpreter.get_output_details()[0]

    def q(details, value):
        scale, zero = details["quantization"]
        return (np.clip(np.round(value / (scale + 1e-12)) + zero,
                        -128, 127)).astype(np.int8)

    preds = []
    for i in range(len(X)):
        interpreter.set_tensor(sig_in["index"], q(sig_in, X[i:i + 1, :, None]))
        interpreter.set_tensor(rr_in["index"], q(rr_in, RR[i:i + 1]))
        interpreter.invoke()
        preds.append(np.argmax(interpreter.get_tensor(out["index"])[0]))

    vals, counts = np.unique(preds, return_counts=True)
    dist = dict(zip([["N", "S", "V"][v] for v in vals], counts.tolist()))
    print(f"classified beats: {dist}")
    print("NOTE: no labels exist for own recordings — a clinician must "
          "review the ECG before these predictions mean anything.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True,
                        help="CardioTouch export (CSV/txt/WAV)")
    parser.add_argument("--fs", type=float, default=None,
                        help="sampling rate override (Hz) if the file has "
                             "no time column")
    args = parser.parse_args()
    classify_recording(args.file, args.fs)
