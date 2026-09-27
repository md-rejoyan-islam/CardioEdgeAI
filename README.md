# CardioEdgeAI

**Lightweight deep learning for real-time ECG arrhythmia detection on edge and wearable devices.**

CardioEdgeAI trains a compact arrhythmia classifier on ECG signals, compresses it with full-integer quantization, and deploys it as a 23.8 KB TensorFlow Lite Micro model that runs in real time on microcontrollers (ESP32-S3, ARM Cortex-M) — no cloud, no PC.

- Repository: <https://github.com/md-rejoyan-islam/CardioEdgeAI>
- Thesis: *Lightweight Machine Learning Model for Real-Time ECG Arrhythmia Detection on Edge/Wearable Devices*
- Full experiment log: [`docs/RESULTS.md`](docs/RESULTS.md) · per-phase reports: [`docs/`](docs/)

## Table of contents

1. [Results at a glance](#results-at-a-glance)
2. [Problem](#problem)
3. [How it works](#how-it-works)
4. [Model architecture](#model-architecture)
5. [Dataset & evaluation protocol](#dataset--evaluation-protocol)
6. [Key research findings](#key-research-findings)
7. [Repository structure](#repository-structure)
8. [Full reproduction guide](#full-reproduction-guide)
9. [Deployment](#deployment-esp32-s3--tflite-micro)
10. [Live data from the CardioTouch 3000](#live-data-from-the-cardiotouch-3000)
11. [Limitations & future work](#limitations--future-work)
12. [Ethics & safety](#ethics--safety)
13. [References](#references)

## Results at a glance

Evaluation on **MIT-BIH DS2 (inter-patient)** — 50,026 beats from 22
recordings whose patients were **never seen in training**. Primary
metric: macro-F1 (classes are ~90 / 3 / 7 % imbalanced).

| # | Model | Params | Size | Acc | Macro-F1 | N F1 | S F1 | V F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression (RR + morphology) | — | — | 84.3 % | 0.603 | 0.914 | 0.240 | 0.655 |
| 2 | Random Forest (RR + morphology) | — | — | 94.3 % | 0.650 | 0.971 | 0.100 | 0.880 |
| 3 | 1D CNN, morphology only *(rejected)* | 11,443 | — | 81.1 % | 0.522 | 0.888 | 0.042 | 0.634 |
| 4 | 1D CNN + RR branch (float32) | 11,571 | 211 KB | 94.4 % | 0.706 | 0.971 | 0.295 | 0.851 |
| 5 | v2 float (S-oversampling, not deployed) | 11,571 | 211 KB | 93.7 % | **0.718** | 0.966 | **0.389** | 0.800 |
| 6 | **v1, TFLite full-INT8 — deployed** | 11,571 | **23.8 KB** | **94.3 %** | 0.712 | 0.970 | 0.320 | 0.847 |

- INT8 quantization: **8.9x smaller** than the training artifact, **4.2x** than float TFLite — with **zero accuracy loss** on the deployed v1 model.
- Laptop-replay latency: **0.02 ms/beat** (~54,600 beats/s; real time needs 1–2).
- Inference-time R-peak detector (NeuroKit): **F1 0.934** vs gold annotations.

## Problem

Cardiac arrhythmias are often intermittent, so reliable screening needs **continuous, long-term ECG monitoring**. Cloud-based deep learning is too large and power-hungry for wearable and edge hardware, which has only a few hundred KB of RAM and a battery that must last days. The question this project answers:

> How small and fast can an accurate arrhythmia detector become, and what accuracy survives aggressive compression on real hardware?

## How it works

```
ECG signal (MIT-BIH, MLII lead, native 360 Hz)
        |
        v
Preprocessing -- zero-phase 0.5-40 Hz Butterworth bandpass
                 per-record z-score normalization
        |
        v
R-peak centered windows -- 0.4 s before / 0.6 s after = 360 samples
        |
        v
Two-branch model -- morphology CNN  +  4 RR-timing features
        |
        v
Full-integer INT8 quantization (stratified calibration)
        |
        v
Edge deploy -- TFLite Micro on ESP32-S3 - watch-folder live demo
        |
        v
Profiling -- accuracy - latency - RAM - flash - energy
```

## Model architecture

11,571 parameters (v1, deployed). The morphology branch uses `Conv2D`
with `(1, k)` kernels on a `(batch, 1, time, 1)` tensor — mathematically
identical to Conv1D, but required because TFLite's integer quantizer
cannot calibrate a graph whose first op is Conv1D.

```
beat window (360 x 1)
  -> Reshape(1, 360, 1)
  -> Conv2D(16, (1,7)) -> BN -> ReLU -> MaxPool(1,2)
  -> Conv2D(32, (1,5)) -> BN -> ReLU -> MaxPool(1,2)
  -> Conv2D(64, (1,3)) -> BN -> ReLU -> GlobalAvgPool   (64 features)
                                                        \
RR timing (4 scalars) --------------------------------> concat -> Dense(32)
                                                       -> Dropout(0.3)
                                                       -> Dense(3, softmax)
```

RR features (computed identically on the ESP32, fixed physiological
normalization mean 0.8 s / std 0.3 s): pre-RR, post-RR, local mean of the
5 preceding RRs, pre/local ratio. Post-RR costs one beat of latency —
documented trade-off for the strongest S-class feature.

Training: Adam 1e-3, batch 256, up to 25 epochs, early stopping
(patience 8) on four patient-separated validation records
(215/220/223/230), class weights {N:0.6, S:8, V:2}, seed 42 (re-runs are
bit-identical).

## Dataset & evaluation protocol

| Set | Beats | N | S | V |
|---|---|---|---|---|
| DS1 fit (training) | 40,320 | 36,397 | 773 | 3,150 |
| DS1 val (early stopping) | 10,250 | 9,442 | 170 | 638 |
| DS2 (test) | 50,026 | 44,742 | 1,837 | 3,447 |
| **Total** | **100,596** | **90,581** | **2,780** | **7,235** |

- **Task**: beat classification into AAMI groups N / S / V (F and Q
  beats excluded in v1 and reported as out of scope).
- **Inter-patient splits** (standard DS1/DS2): no patient appears in both
  train and test — reported accuracy reflects generalization to unseen
  patients, the honest (and much harder) protocol.
- **Metrics**: accuracy, macro-F1 (primary), per-class precision/recall/F1,
  confusion matrix (`docs/results.json` holds every number).
- **Efficiency metrics**: parameters, model size, interpreter latency,
  tensor-arena RAM.
- Source: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/),
  48 half-hour records, 360 Hz, cardiologist beat annotations.

## Key research findings

1. **Morphology alone cannot detect S beats inter-patient.** Three tuning
   rounds of a morphology-only CNN never exceeded S recall 0.31 without
   destroying N. Supraventricular ectopy is defined by *premature
   timing* — adding 4 cheap RR scalars tripled S-F1 (0.10 → 0.30). This
   is the project's central architectural claim.
2. **The textbook R-detector default is a trap.** Pan-Tompkins scored
   F1 = 0.63 against gold annotations on the filtered signals (missed
   ~1/3 of beats); NeuroKit's detector scored F1 = 0.93. Switching the
   default was the single largest end-to-end accuracy fix.
   (`src/evaluation/r_detector_eval.py`)
3. **Oversampling buys float accuracy and sells quantization robustness.**
   S-oversampling lifted float macro-F1 to 0.718 but the model collapsed
   under INT8 PTQ under *every* calibration strategy (86 / 33 / 83 %).
   v1 quantizes losslessly and stays deployed; QAT is the route to
   v2-class training on-device.
4. **TFLite cannot integer-quantize Conv1D-first graphs**
   (`conv.cc input->dims->size != 4`), new and legacy quantizers alike —
   the `Conv2D((1,k))` pattern is the fix and costs nothing.
5. **Quantization acted as a mild regularizer** on v1: INT8 marginally
   outperformed float (macro-F1 0.712 vs 0.706 in the final run).

## Repository structure

```
CardioEdgeAI/
├── data/
│   ├── raw/             # PhysioNet records (downloaded, git-ignored)
│   └── processed/       # beat windows + splits npz (git-ignored)
├── src/
│   ├── config.py        # all constants: rates, windows, AAMI map, splits, seed
│   ├── data/
│   │   ├── download_mitbih.py   # dataset fetcher
│   │   ├── preprocess.py        # bandpass, R-peak (train + inference), segmentation
│   │   ├── visualize.py         # sanity figures -> docs/figures/
│   │   └── cardiotouch.py       # external-validation adapter (CSV/WAV/SCP-detect)
│   ├── features/classical.py    # Phase 3 RR+morphology baselines
│   ├── models/
│   │   ├── cnn.py               # v1 production model (also: RR features)
│   │   ├── cnn_v2.py            # oversampling ablation model
│   │   └── quantize.py          # full-INT8 conversion + interpreter eval
│   └── evaluation/
│       ├── evaluate.py          # reports, confusion matrices, results.json
│       └── r_detector_eval.py   # detector comparison vs gold annotations
├── deploy/
│   ├── tflite/          # model.h exporter + latency benchmark
│   └── esp32/CardioEdgeAI/      # TFLite Micro firmware + generated model.h
├── machine/             # the lab ECG machine: photos, manuals, electrode
│   ├── images/          #   and data-collection guides (Bionet CardioTouch 3000)
│   ├── manuals/         #   practical guide (Bengali docx) + official manuals
│   ├── electrodes/      #   how electrodes work, 12-lead placement
│   └── data-collection/ #   recording & export protocol, step by step
├── tools/
│   ├── watch_folder.py  # live: auto-classify new BMS-Plus exports
│   └── serial_capture.py# live: RS-232 byte logger (protocol discovery)
├── tests/test_live_pipeline.py  # end-to-end test on synthetic export
├── docs/                # PHASE1-8 reports, RESULTS, LIVE_DEMO, figures, results.json
├── requirements.txt     # pinned (Python 3.14, TF 2.22.0rc0)
├── CITATION.cff
└── LICENSE              # MIT
```

## Full reproduction guide

Environment: Python 3.14+, ~2 GB disk for packages + data, CPU-only
(training the CNN takes ~3 minutes on a modern laptop).

```bash
git clone https://github.com/md-rejoyan-islam/CardioEdgeAI.git
cd CardioEdgeAI
py -m venv .venv && .venv\Scripts\activate   # or your usual python
pip install -r requirements.txt
```

Run everything in order (each step prints its own results):

```bash
py src/data/download_mitbih.py            # ~85 MB PhysioNet fetch
py src/data/visualize.py                  # signal-quality figures
py src/data/preprocess.py                 # builds data/processed/mitbih_nsv.npz
py src/features/classical.py              # LR + RF baselines
py src/models/cnn.py                      # v1 model -> models/phase4_cnn.keras
py src/models/cnn_v2.py                   # ablation model (0.718 macro-F1, float)
py src/evaluation/r_detector_eval.py      # detector comparison (~10 min)
py src/models/quantize.py                 # v1 -> 23.8 KB INT8 + interpreter eval
py deploy/tflite/export_model_header.py   # regenerates deploy/esp32 model.h
py deploy/tflite/benchmark.py             # latency benchmark
py tests/test_live_pipeline.py            # full live-path test (must pass)
```

Single-file classification of any exported recording:

```bash
py src/data/cardiotouch.py --file recording.csv          # or .txt / .wav
py src/data/cardiotouch.py --file x.csv --fs 1000        # no time column
```

## Deployment (ESP32-S3 + TFLite Micro)

Hardware: ESP32-S3 DevKitC + AD8232 single-lead front end (OUT → GPIO4).
Software: Arduino IDE + **esp32** board package + **TensorFlowLite_ESP32**
library. Open `deploy/esp32/CardioEdgeAI/CardioEdgeAI.ino`, upload,
open Serial Monitor at 115200 — every detected beat prints
`BEAT <N|S|V> conf=… rr_pre=… rr_post=…`.

Firmware pipeline mirrors training exactly (same biquads, same 10 s
rolling z-score, same fixed RR normalization, round-to-nearest int8
quantization — verified equal to the Python eval path). The previous
beat is classified on each new R peak because post-RR needs the next
beat: one-beat latency, ~1–2 classifications/s at rest.

Memory budget: 23.8 KB model (flash) + 48 KB tensor arena + 14.4 KB
rolling buffer ≪ 512 KB ESP32-S3 SRAM.

Details, bring-up checklist and open items: [`docs/PHASE6.md`](docs/PHASE6.md)
and [`deploy/esp32/README.md`](deploy/esp32/README.md).

## The lab machine

The external-validation data source — photos of the actual unit, its
signal chain, electrode physics/placement, and the full
record-and-export protocol live in [`machine/`](machine/README.md)
(includes a Bengali practical guide).

## Live data from the CardioTouch 3000

Full guide: [`docs/LIVE_DEMO.md`](docs/LIVE_DEMO.md). Three paths:

| Path | What | Status |
|---|---|---|
| **A — watch folder** | LAN → BMS-Plus → export CSV → auto-classified in ~1 s | **tested end-to-end** |
| **B — RS-232 capture** | `tools/serial_capture.py` logs raw bytes for protocol discovery | logger ready |
| **C — continuous wearable** | AD8232 + ESP32 firmware | firmware written, board pending |

```bash
py tools/watch_folder.py --folder "C:/BMS-Plus/Exports"
```

Accepted exports: two-column CSV/text (seconds, mV — headers, `,`/`;`/space
all tolerated) and WAV. SCP-ECG files are detected and explained (no
pip-installable parser exists; decode waits for a real sample file).
Every processed export also becomes Phase 7 validation data.

## Limitations & future work

- **S remains the hard class** (deployed F1 0.32 at precision 0.43) —
  consistent with the inter-patient literature. Routes: QAT for the
  oversampled v2 (float F1 0.39), focal tuning, per-patient calibration.
- On-board ESP32 latency/energy **not yet measured** (no board in this
  session); laptop numbers give 3–4 orders of magnitude headroom.
- CardioTouch external validation has **zero recordings yet** — the
  tooling and protocol are ready; the lab session produces the data.
- The ESP32 firmware's simple derivative R-detector is not the calibrated
  NeuroKit detector used in Python; quantify its sensitivity separately
  (open item in `docs/PHASE6.md`).
- F/Q AAMI classes excluded in v1; PTB-XL generalization untested.

## Ethics & safety

Any human ECG recordings used for external validation are collected under
the supervision of a trained ECG technician or clinician, with informed
consent, and anonymized before touching this repository (consent log kept
offline). The AD8232 rig must run on battery/isolated power near a
person. This system is a research prototype and is **not** a medical
device.

## References

A curated index of 85 related papers (46 full-text PDFs) supports this
project — including LiteNet, ECG-TCN, TinyML arrhythmia classifiers,
quantization/pruning/distillation literature (Deep Compression, QAT,
MCUNet, TFLite Micro). Key methodological anchors:

- Moody & Mark (2001), *The MIT-BIH Arrhythmia Database*.
- de Chazal et al. (2004), *Automatic classification of heartbeats using
  ECG morphology and heartbeat interval features* (DS1/DS2 protocol,
  AAMI grouping).
- Jacob et al. (2018), *Quantization and training of neural networks for
  efficient integer-arithmetic-only inference*.
- David et al. (2021), *TensorFlow Lite Micro*.

## License

Released under the [MIT License](LICENSE). If you use this work in
academic research, please cite it (see `CITATION.cff`).
