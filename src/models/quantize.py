"""Phase 5 — lightweight optimization: full-integer INT8 quantization.

Takes the Phase 4 CNN and produces:
1. `models/phase5_cnn_float.tflite` — float32 TFLite (conversion baseline)
2. `models/phase5_cnn_int8.tflite`  — full-integer INT8 (weights + activations)

Both are evaluated on DS2 with the TFLite interpreter so the accuracy cost
of quantization is measured on-device-format inference, not a Keras
approximation. Size / accuracy deltas are appended to docs/results.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MODELS_DIR  # noqa: E402
from src.evaluation.evaluate import evaluate, load_processed_dataset  # noqa: E402

REPRESENTATIVE_BEATS = 2000


def representative_dataset(Xtr):
    rng = np.random.default_rng(42)
    for i in rng.choice(len(Xtr), REPRESENTATIVE_BEATS, replace=False):
        yield [Xtr[i:i + 1].astype(np.float32)]


def convert(model, Xtr):
    # float32 TFLite
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite = conv.convert()

    # full-integer INT8 with input/output as int8 (edge-friendly)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: representative_dataset(Xtr)
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    int8_tflite = conv.convert()
    return float_tflite, int8_tflite


def tflite_predict(tflite_bytes: bytes, Xte) -> np.ndarray:
    """Run the interpreter over all test beats; handles int8 quantized IO."""
    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes, num_threads=2)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zero = inp["quantization"]
    preds = np.empty(len(Xte), dtype=np.int64)
    for i, x in enumerate(Xte):
        data = x[None, :, None]
        if inp["dtype"] == np.int8:
            data = (np.clip(data / (scale + 1e-12), -128, 127)
                    + zero).astype(np.int8)
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()
        logits = interpreter.get_tensor(out["index"])[0]
        preds[i] = np.argmax(logits)
    return preds


def main() -> None:
    X, y, _, _, train_mask, test_mask = load_processed_dataset()
    classes = np.array(["N", "S", "V"])
    y_int = np.searchsorted(classes, y)
    Xtr, Xte = X[train_mask], X[test_mask]

    model = tf.keras.models.load_model(MODELS_DIR / "phase4_cnn.keras")
    print("converting ...")
    float_tflite, int8_tflite = convert(model, Xtr.astype(np.float32))

    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "phase5_cnn_float.tflite").write_bytes(float_tflite)
    (MODELS_DIR / "phase5_cnn_int8.tflite").write_bytes(int8_tflite)
    keras_kb = (MODELS_DIR / "phase4_cnn.keras").stat().st_size / 1024
    float_kb = len(float_tflite) / 1024
    int8_kb = len(int8_tflite) / 1024
    print(f"keras float32 : {keras_kb:8.1f} KB")
    print(f"tflite float32: {float_kb:8.1f} KB")
    print(f"tflite int8   : {int8_kb:8.1f} KB")

    for name, blob in (("TFLite float32 (converted)", float_tflite),
                       ("TFLite INT8 (quantized)", int8_tflite)):
        pred = classes[tflite_predict(blob, Xte.astype(np.float32))]
        metrics = evaluate(name, y[test_mask], pred)

    results_path = Path(__file__).resolve().parents[2] / "docs" / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["TFLite float32 (converted)"]["model_size_kb"] = round(float_kb)
    results["TFLite INT8 (quantized)"]["model_size_kb"] = round(int8_kb)
    results["TFLite INT8 (quantized)"]["compression_vs_keras"] = round(
        keras_kb / int8_kb, 1)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved results ->", results_path)


if __name__ == "__main__":
    main()
