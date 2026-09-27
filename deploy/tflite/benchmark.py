"""Laptop-replay benchmark: INT8 model latency and throughput.

Times the Phase 5 INT8 TFLite model over the DS2 test set with the same
interpreter configuration an ESP32 would use (2 threads here; MCU is
single-threaded). Reports mean/median/p95 latency per beat and beats/s.

Usage: py deploy/tflite/benchmark.py
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MODELS_DIR  # noqa: E402
from src.evaluation.evaluate import load_processed_dataset  # noqa: E402
from src.models.cnn import rr_feature_matrix  # noqa: E402
from src.models.quantize import tflite_predict  # noqa: E402

N_TIME = 2000  # beats timed for the latency statistics


def main() -> None:
    blob = (MODELS_DIR / "phase5_cnn_int8.tflite").read_bytes()
    X, y, r, record, _, test_mask = load_processed_dataset()
    RR = rr_feature_matrix(r, record)
    Xte, RRte = X[test_mask][:N_TIME].astype(np.float32), RR[test_mask][:N_TIME]

    # correctness smoke check first (same predictions as Phase 5 eval)
    classes = np.array(["N", "S", "V"])
    pred = classes[tflite_predict(blob, Xte, RRte)]
    print(f"smoke check on {len(Xte)} beats: "
          f"{(pred == y[test_mask][:N_TIME]).mean():.4%} correct")

    # timed loop — one invoke per beat, interpreter reused (device-like)
    interpreter = tf.lite.Interpreter(model_content=blob, num_threads=2)
    interpreter.allocate_tensors()
    sig_in = interpreter.get_input_details()[0]
    rr_in = interpreter.get_input_details()[1]
    out = interpreter.get_output_details()[0]

    def q(details, value):
        scale, zero = details["quantization"]
        return (np.clip(value / (scale + 1e-12), -128, 127)
                + zero).astype(np.int8)

    times = []
    for i in range(len(Xte)):
        t0 = time.perf_counter()
        interpreter.set_tensor(
            sig_in["index"], q(sig_in, Xte[i:i + 1, :, None]))
        interpreter.set_tensor(rr_in["index"], q(rr_in, RRte[i:i + 1]))
        interpreter.invoke()
        interpreter.get_tensor(out["index"])
        times.append((time.perf_counter() - t0) * 1e3)

    times_ms = sorted(times)
    print(f"latency per beat (laptop CPU, 2 threads): "
          f"mean {statistics.mean(times_ms):.2f} ms | "
          f"median {statistics.median(times_ms):.2f} ms | "
          f"p95 {times_ms[int(0.95 * len(times_ms))]:.2f} ms")
    print(f"throughput: {1000 / statistics.median(times_ms):.0f} beats/s "
          f"(real-time needs only {360 / 360:.0f} beat/s at 60-100 bpm)")


if __name__ == "__main__":
    main()
