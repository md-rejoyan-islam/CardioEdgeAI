# CardioEdgeAI

**Lightweight deep learning for real-time ECG arrhythmia detection on edge and wearable devices.**

CardioEdgeAI trains a compact arrhythmia classifier on ECG signals, compresses it with
integer quantization, and deploys it as an INT8 TensorFlow Lite Micro model that runs in
real time on microcontrollers (ESP32-S3, ARM Cortex-M) — no cloud, no PC.

## Results at a glance (MIT-BIH, inter-patient DS2 test)

| Model | Size | Accuracy | Macro-F1 |
|---|---|---|---|
| Random Forest (RR + morphology features) | — | 94.3 % | 0.650 |
| 1D CNN + RR-timing branch (float32) | 211 KB | 94.4 % | 0.706 |
| **Same, TFLite full-INT8 (deployed)** | **23.8 KB** | **94.5 %** | **0.713** |

INT8 quantization shrank the model **8.9x with zero accuracy loss**; laptop-replay
latency is 0.02 ms/beat (~54k beats/s — real time needs 1-2). Full tables and the
reproduction recipe: [docs/RESULTS.md](docs/RESULTS.md).

## Problem

Cardiac arrhythmias are often intermittent, so reliable screening needs **continuous,
long-term ECG monitoring**. Cloud-based deep learning models are too large and power-hungry
for wearable and edge hardware, which has only a few hundred KB of RAM and a battery that
must last days. The challenge this project addresses:

> How small and fast can an accurate arrhythmia detector become, and what accuracy
> survives after aggressive compression on real hardware?

## Approach

```
ECG signal (MIT-BIH, MLII lead, 360 Hz)
        |
        v
Preprocessing -- bandpass filter - baseline wander removal - R-peak detection
        |
        v
Beat windows (0.4 s before / 0.6 s after each R peak)
        |
        v
Models -------- RR + morphology baseline -> 1D CNN -> compressed student model
        |
        v
Compression --- INT8 quantization - structured pruning - knowledge distillation
        |
        v
Edge deploy ---- TFLite Micro on ESP32-S3 / Cortex-M - laptop signal replay
        |
        v
Profiling ------ accuracy - latency - peak RAM - flash size - energy per inference
```

## Key design decisions (v1)

| Decision | Choice |
|---|---|
| Task | Beat classification into AAMI groups **N / S / V** |
| Training data | MIT-BIH Arrhythmia Database (MLII lead, native 360 Hz) |
| Beat window | 0.4 s before R peak, 0.6 s after (360 samples per beat) |
| Baseline | Classical model on RR-interval + morphology features, then 1D CNN **with an RR-timing branch** (morphology alone cannot detect S beats) |
| Compression | Full-integer INT8 post-training quantization (QAT/pruning/distillation: future work) |
| Deployment | Laptop replay first, then ESP32-S3 (TFLite Micro) |
| Evaluation | Patient-separated splits, per-class F1, confusion matrix |
| External test | Lead-II ECG exported from a lab Bionet CardioTouch 3000 |

## Repository structure

```
CardioEdgeAI/
├── data/
│   ├── raw/          # PhysioNet records (downloaded, git-ignored)
│   └── processed/    # filtered signals, beat windows, splits (git-ignored)
├── notebooks/        # exploration and experiment notebooks
├── src/
│   ├── config.py     # all fixed constants (rates, windows, AAMI mapping)
│   ├── data/         # download + preprocessing pipeline
│   ├── models/       # baseline, 1D CNN, compression
│   └── evaluation/   # metrics, plots, comparison tables
├── models/           # trained artifacts (.h5 / .tflite, git-ignored)
├── deploy/
│   ├── tflite/       # conversion + benchmark scripts
│   └── esp32/        # ESP32-S3 firmware using TFLite Micro
├── tools/            # live-demo tools: watch-folder classifier, RS-232 capture
├── tests/            # end-to-end pipeline tests (synthetic exports)
├── docs/             # thesis chapters, figures, literature notes
├── requirements.txt
├── CITATION.cff
└── README.md
```

## Getting started

```bash
git clone https://github.com/md-rejoyan-islam/CardioEdgeAI.git
cd CardioEdgeAI

python -m venv .venv
source .venv/bin/activate        # Windows (py launcher): py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Download the MIT-BIH Arrhythmia Database (~100 MB) into data/raw/mitdb
python src/data/download_mitbih.py --out data/raw/mitdb
```

Then reproduce the full pipeline with `docs/RESULTS.md` ("Reproducing
everything").

## Dataset

- **MIT-BIH Arrhythmia Database** — 48 half-hour two-lead ambulatory ECG records,
  360 Hz, beat-level annotations by cardiologists.
  https://physionet.org/content/mitdb/1.0.0/
- Later phases: PTB-XL and PhysioNet Challenge sets for generalization checks,
  plus lab-recorded CardioTouch 3000 lead-II exports for external validation.

## Evaluation protocol

- **Inter-patient splits** (e.g., DS1/DS2): no patient appears in both train and test.
- **Accuracy metrics**: overall accuracy, per-class precision/recall/F1, macro F1,
  confusion matrix.
- **Efficiency metrics**: parameter count, model size (KB), inference latency (ms)
  measured on the actual board, peak RAM, and energy per inference.

## Roadmap

- [x] Phase 1 — Environment setup, MIT-BIH download, signal visualization ([docs](docs/PHASE1.md))
- [x] Phase 2 — Preprocessing pipeline (filtering, baseline removal, R-peak, segmentation) ([docs](docs/PHASE2.md))
- [x] Phase 3 — Classical baseline (RR-interval + morphology features) ([docs](docs/PHASE3.md))
- [x] Phase 4 — 1D CNN baseline with patient-separated evaluation ([docs](docs/PHASE4.md))
- [x] Phase 5 — Lightweight optimization (INT8 quantization) ([docs](docs/PHASE5.md))
- [x] Phase 6 — Edge deployment code (TFLite Micro firmware + benchmark; board bring-up pending) ([docs](docs/PHASE6.md))
- [x] Phase 7 — CardioTouch validation adapter + protocol (data collection pending) ([docs](docs/PHASE7.md))
- [x] Phase 8 — Consolidated results + documentation ([docs](docs/RESULTS.md); thesis chapters continue offline)

## Live data from the CardioTouch 3000

Three supported paths — watch-folder auto-classification (tested), RS-232
serial capture, and the continuous AD8232+ESP32 demo:
see [docs/LIVE_DEMO.md](docs/LIVE_DEMO.md).

```bash
py tools/watch_folder.py --folder "C:/BMS-Plus/Exports"   # classify each new export
py tests/test_live_pipeline.py                            # verify the whole chain
```

## Ethics note

Any human ECG recordings used for external validation are collected under the
supervision of a trained ECG technician or clinician and anonymized before storage.
This system is a research prototype and is **not** a medical device.

## References

A curated index of 85 related papers (46 full PDFs) supports this project, including
LiteNet, ECG-TCN, TinyML arrhythmia classifiers, and the model-compression literature
(Deep Compression, QAT, MCUNet, TFLite Micro). See the thesis literature matrix.

## License

Released under the [MIT License](LICENSE).
