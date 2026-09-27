# Phase 6 — Edge Deployment (TFLite Micro on ESP32-S3)

**Status:** code complete — **hardware bring-up pending** (board not yet
connected in this session; firmware written and documented for the lab
session).

## Deliverables

| Artifact | What it is |
|---|---|
| `deploy/tflite/export_model_header.py` | `.tflite` → `model.h` C byte array |
| `deploy/esp32/CardioEdgeAI/CardioEdgeAI.ino` | full streaming firmware (see below) |
| `deploy/esp32/CardioEdgeAI/model.h` | 24,352 generated model bytes (23.8 KB INT8) |
| `deploy/tflite/benchmark.py` | laptop-replay latency benchmark |
| `deploy/esp32/README.md` | wiring, Arduino setup, memory budget, safety |

## Firmware pipeline (mirrors training exactly)

```
ADC @ 360 Hz
  -> 0.5-40 Hz biquad bandpass            (same design as preprocess.py)
  -> rolling 10 s z-score buffer
  -> derivative R-peak detector           (adaptive threshold + 250 ms refractory)
  -> on each new R peak:
       classify the PREVIOUS beat         (post-RR now known -> 1-beat latency)
       window: R-0.4 s .. R+0.6 s
       RR features, fixed (0.8 s, 0.3 s) normalization  (same as cnn.py)
       int8 quantize with tensor scale/zero-point        (same as quantize.py)
       TFLite Micro invoke -> "BEAT <N|S|V> conf=... rr_pre=... rr_post=..."
```

The one-beat latency is a deliberate, documented consequence of using
post-RR (the strongest S-class feature, Phase 4). A real-time cardiac
monitor classifies ~1-2 beats/s, so one beat of latency is negligible.

## Laptop-replay benchmark (`py deploy/tflite/benchmark.py`)

INT8 interpreter, one invoke per beat, interpreter reused (device-like):

| Metric | Value |
|---|---|
| Mean latency per beat | 0.02 ms |
| Median / p95 | 0.02 / 0.04 ms |
| Throughput | ~54,600 beats/s |
| Real-time requirement | ~1-2 beats/s (60-100 bpm) |

Headroom vs real time is roughly 4 orders of magnitude on a laptop CPU;
on the ESP32-S3 (240 MHz, no XNNPACK-by-default build) expect single-digit
ms per beat — still 2-3 orders of magnitude of headroom. On-board numbers
to be measured during bring-up and recorded here.

## Memory budget

| Item | Size |
|---|---|
| INT8 model in flash (`model.h`) | 23.8 KB |
| Tensor arena (SRAM) | 48 KB allocated, ~40 KB needed |
| Rolling z-score buffer (10 s) | 14.4 KB |
| Total SRAM committed | ≪ 512 KB (ESP32-S3) |

## Open items for the lab session

1. Flash + serial session on the actual ESP32-S3 board.
2. Measure on-board latency (`micros()` around Invoke) and append here.
3. AD8232 electrode connection test (RA/LA/RL).
4. Compare on-device classifications against laptop replay of the same
   recording (parity check).
5. The on-device R detector is not the gold standard used in training —
   quantify its sensitivity/PPV against annotations on MIT-BIH replay and
   report it separately from model accuracy.

## Safety

Research prototype only — not a medical device. The AD8232 front end is
patient-isolated; never power the rig from a mains-referenced supply while
electrodes are on a person.
