# ESP32-S3 Deployment (TFLite Micro)

## What is here

```
CardioEdgeAI/
├── CardioEdgeAI.ino   Arduino sketch: ADC -> filter -> R-peak -> classify
└── model.h            INT8 model bytes (auto-generated — do not edit by hand)
```

## Bill of materials

| Part | Purpose |
|---|---|
| ESP32-S3 DevKitC | MCU, 512 KB SRAM, WiFi/BLE |
| AD8232 breakout | single-lead ECG analog front end (3-electrode) |
| Ag/AgCl electrodes + leads | RA / LA / RL placement |
| Jumper wires, breadboard | OUT -> GPIO4, 3V3, GND |

## Software setup

1. Arduino IDE 2.x → Boards Manager → install **esp32 by Espressif**.
2. Library Manager → install **TensorFlowLite_ESP32**.
3. Open `CardioEdgeAI.ino`, select the ESP32S3 Dev Module board, Upload.
4. Serial Monitor at 115200 — each detected beat prints
   `BEAT <N|S|V> conf=0.xx rr_pre=0.72s rr_post=0.98s`.

## Regenerating the model

After any retraining/quantization run:

```bash
py src/models/cnn.py                  # trains models/phase4_cnn.keras
py src/models/quantize.py            # makes models/phase5_cnn_int8.tflite
py deploy/tflite/export_model_header.py   # writes model.h from the .tflite
```

## Memory budget (from Phase 5 artifacts)

| Item | Size |
|---|---|
| INT8 model (`model.h`) | 23.8 KB |
| Tensor arena (runtime) | 48 KB (set in sketch; ~40 KB suffices) |
| Rolling z-score buffer | 14.4 KB (10 s × 360 Hz × 4 B) |
| Total | « 512 KB ESP32-S3 SRAM |

## Pipeline parity with training

Every transform on the device mirrors `src/` exactly:
0.5–40 Hz bandpass (`src/data/preprocess.py` biquad coefficients),
z-score over a rolling window, 0.4 s / 0.6 s window around the R peak,
RR features with the fixed (0.8 s, 0.3 s) normalization
(`src/models/cnn.py`), int8 input quantization using the tensor's own
scale/zero-point (same path as `src/models/quantize.py`).

One deliberate difference: the on-device R detector is a derivative +
adaptive threshold (training used gold-standard annotations). Its quality
must be reported separately — see `docs/PHASE6.md`.

## Safety

This firmware is a research prototype. Never use it as a medical device.
Electrode contact with skin must go through a properly isolated ECG front
end (the AD8232 is patient-isolated); do not power from mains-referenced
supplies while connected to a person.
