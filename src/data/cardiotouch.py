"""Phase 7 — external-validation adapter for Bionet CardioTouch 3000 exports.

Runs the deployed INT8 model on ECG recorded by the lab machine, through
the *inference-time* path (Pan-Tompkins R detection — no gold-standard
annotations exist for own recordings). This measures real-world
generalization: different amplifier, different electrodes, rest-ECG
12-lead context, exported lead II.

Supported export formats (auto-detected by `load_ecg_file`):
  - CSV / text with two columns: seconds and millivolts
    (comma, semicolon or whitespace separated; a non-numeric header row
    is skipped automatically)
  - WAV (16-bit PCM, single channel, any rate)
  - SCP-ECG (.scp): detected and identified (magic bytes + section
    table), but waveform decoding is NOT implemented yet — the decoder
    must be written against a real sample file first. Clear guidance is
    returned instead of silently wrong output.

Usage:
    py src/data/cardiotouch.py --file exported_ecg.csv
    py src/data/cardiotouch.py --file export.csv --fs 1000

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
from src.models.cnn import rr_feature_matrix  # noqa: E402

SCP_MAGIC = b"SCPECG"


class ScpNotDecodedError(RuntimeError):
    """Raised when an SCP-ECG file is detected but not yet decodable."""


def read_two_column_text(path: Path):
    """Return (seconds, mV) arrays from CSV/txt, tolerating header rows."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for delim in (",", ";", None):
                parts = line.split(delim) if delim else line.split()
                if len(parts) >= 2:
                    try:
                        rows.append((float(parts[0].strip()),
                                     float(parts[1].strip())))
                        break
                    except ValueError:
                        continue  # header or junk line -> skipped
    if not rows:
        raise ValueError(f"no two-column numeric data found in {path}")
    arr = np.asarray(rows)
    return arr[:, 0], arr[:, 1]


def read_scp_info(path: Path) -> dict:
    """Parse the SCP-ECG header/section table (no waveform decode)."""
    blob = path.read_bytes()
    if not blob.startswith(SCP_MAGIC):
        raise ValueError("not an SCP-ECG file (missing SCPECG magic)")
    n_sections = int.from_bytes(blob[10:12], "little")
    info = {"sections": n_sections, "ids": []}
    for i in range(1, min(n_sections, 12)):  # pointer table starts at 16
        ptr = blob[16 + (i - 1) * 16: 16 + i * 16]
        if len(ptr) < 16:
            break
        info["ids"].append(int.from_bytes(ptr[12:14], "little"))
    info["has_rhythm_section"] = 6 in info["ids"]
    return info


def load_ecg_file(path: Path):
    """Return (signal_mv, fs_hz) from CSV/text/WAV; detect SCP files."""
    suffix = path.suffix.lower()
    if suffix in (".scp", ".ecg") or path.read_bytes()[:6] == SCP_MAGIC:
        info = read_scp_info(path)
        raise ScpNotDecodedError(
            f"{path.name} is an SCP-ECG file "
            f"({info['sections']} sections, ids {info['ids']}). "
            "Waveform decoding is not implemented yet. Either (a) export "
            "CSV from BMS-Plus / EKG Viewer and point --file at it, or "
            "(b) keep this .scp file — once a real sample exists the "
            "decoder can be written and verified against it.")
    if suffix == ".wav":
        from scipy.io import wavfile
        fs, data = wavfile.read(path)
        signal = data.astype(np.float64)
        if signal.ndim > 1:
            signal = signal[:, 0]
        peak = np.abs(signal).max() or 1
        return signal / peak * 2.0, float(fs)

    t, y = read_two_column_text(path)
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("first column is not monotonic (seconds)")
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


def classify_recording(path: Path, fs_override: float | None = None,
                       verbose: bool = True) -> dict:
    """Classify one exported recording; returns a result dict.

    Returned dict: file, n_samples, fs, bpm, n_beats, distribution
    ({'N': .., 'S': .., 'V': ..}), per_beat class list.
    """
    import tensorflow as tf

    raw, fs = load_ecg_file(path)
    if fs_override:
        fs = fs_override
    signal = bandpass(resample(raw, fs))

    r_peaks = detect_r_peaks(signal)
    duration_s = len(signal) / SAMPLING_RATE_HZ
    bpm = len(r_peaks) / duration_s * 60 if duration_s else 0.0

    beats, _, _, _ = segment_beats(signal, r_peaks,
                                   ["N"] * len(r_peaks), "ct")
    result = {"file": path.name, "n_samples": len(raw), "fs": round(fs, 2),
              "bpm": round(bpm, 1), "n_beats": 0,
              "distribution": {"N": 0, "S": 0, "V": 0}, "per_beat": []}
    if not beats:
        if verbose:
            print(f"[{path.name}] no complete beat windows — too short")
        return result

    X = np.asarray(beats, dtype=np.float32)
    RR = rr_feature_matrix(np.asarray(r_peaks, dtype=np.int64),
                           np.array(["ct"] * len(r_peaks)))

    blob = (MODELS_DIR / "phase5_cnn_int8.tflite").read_bytes()
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    sig_in, rr_in = interpreter.get_input_details()[:2]
    out = interpreter.get_output_details()[0]

    def q(details, value):
        scale, zero = details["quantization"]
        return (np.clip(np.round(value / (scale + 1e-12)) + zero,
                        -128, 127)).astype(np.int8)

    classes = ["N", "S", "V"]
    per_beat = []
    for i in range(len(X)):
        interpreter.set_tensor(sig_in["index"], q(sig_in, X[i:i + 1, :, None]))
        interpreter.set_tensor(rr_in["index"], q(rr_in, RR[i:i + 1]))
        interpreter.invoke()
        per_beat.append(classes[
            int(np.argmax(interpreter.get_tensor(out["index"])[0]))])

    vals, counts = np.unique(per_beat, return_counts=True)
    result.update({"n_beats": len(per_beat),
                   "distribution": dict(zip(vals.tolist(),
                                            counts.tolist())),
                   "per_beat": per_beat})
    if verbose:
        print(f"[{path.name}] {len(raw)} samples @ {fs:.1f} Hz | "
              f"{len(r_peaks)} R peaks ({bpm:.0f} bpm)")
        print(f"[{path.name}] classified {len(per_beat)} beats: "
              f"{result['distribution']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True,
                        help="CardioTouch export (CSV/txt/WAV)")
    parser.add_argument("--fs", type=float, default=None,
                        help="sampling rate override (Hz) if the file has "
                             "no time column")
    args = parser.parse_args()
    classify_recording(args.file, args.fs)
    print("NOTE: no labels exist for own recordings — a clinician must "
          "review the ECG before these predictions mean anything.")
