"""Phase 1 sanity checks and figures: raw vs filtered ECG, beat classes.

Writes PNG figures to docs/figures/ (tracked in git as documentation).
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import DATA_RAW_DIR, SAMPLING_RATE_HZ  # noqa: E402
from src.data.preprocess import load_record, segment_beats  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"
SAMPLE_RECORD = "100"  # MLII, clean normal-sinus record for the demo plot


def fig_signal_quality():
    """First 5 s of record 100: raw vs bandpass-filtered."""
    import wfdb
    rec = wfdb.rdrecord(str(DATA_RAW_DIR / SAMPLE_RECORD))
    raw = rec.p_signal[:, 0]
    filtered, _, _ = load_record(SAMPLE_RECORD)

    t = np.arange(int(5 * SAMPLING_RATE_HZ)) / SAMPLING_RATE_HZ
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, raw[:len(t)], lw=0.7, color="tab:red")
    axes[0].set_ylabel("mV (raw)")
    axes[0].set_title(f"MIT-BIH record {SAMPLE_RECORD} (MLII), first 5 s")
    axes[1].plot(t, filtered[:len(t)], lw=0.7, color="tab:blue")
    axes[1].set_ylabel("mV (filtered)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase1_signal_quality.png", dpi=150)
    plt.close(fig)


def fig_class_examples():
    """One example beat window per class (N / S / V) from record 100/234."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for ax, (rec_id, group) in zip(axes, [("100", "N"), ("100", "S"),
                                          ("234", "V")]):
        signal, samples, symbols = load_record(rec_id)
        beats, groups, _, _ = segment_beats(signal, samples, symbols, rec_id)
        idx = groups.index(group) if group in groups else 0
        t = (np.arange(len(beats[idx])) - 144) / SAMPLING_RATE_HZ
        ax.plot(t, beats[idx], lw=1.0, color="tab:blue")
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"class {group} (record {rec_id})")
        ax.set_xlabel("time from R peak (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("z-scored mV")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase1_beat_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_signal_quality()
    fig_class_examples()
    print("figures saved to", FIG_DIR)

    # Full-database class counts come from the preprocessing build step
    # (src/data/preprocess.py); here we just show symbol variety per record.
    import wfdb
    c = Counter()
    for rid in ("100", "234"):
        c.update(wfdb.rdann(str(DATA_RAW_DIR / rid), "atr").symbol)
    print("annotation symbols in demo records:", dict(c))
