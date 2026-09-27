# CardioEdgeAI

**Lightweight deep learning for real-time ECG arrhythmia detection on edge and wearable devices.**

CardioEdgeAI trains a compact arrhythmia classifier on ECG signals, compresses it with
quantization / pruning / knowledge distillation, and deploys it as an INT8 TensorFlow Lite
Micro model that runs in real time on microcontrollers (ESP32-S3, ARM Cortex-M) —
no cloud, no PC.

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
| Baseline | Classical model on RR-interval + morphology features, then 1D CNN |
| Compression | INT8 quantization, pruning, knowledge distillation |
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
├── docs/             # thesis chapters, figures, literature notes
├── requirements.txt
├── CITATION.cff
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/CardioEdgeAI.git
cd CardioEdgeAI

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download the MIT-BIH Arrhythmia Database (~100 MB) into data/raw/mitdb
python src/data/download_mitbih.py --out data/raw/mitdb
```

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
- [ ] Phase 6 — On-device deployment (TFLite Micro on ESP32-S3) + profiling
- [ ] Phase 7 — External validation on lab-recorded ECG (CardioTouch 3000)
- [ ] Phase 8 — Thesis writing and final comparison tables

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
