# 7️⃣ Phase 7 — External Validation on CardioTouch 3000 Recordings

**Status:** tooling complete — **data collection pending** (needs the lab
machine, a technician, and consented volunteers; not runnable from this
session).

## Why this phase matters

Everything so far is internal validation on MIT-BIH (1975-1989 Holter
recordings, ambulatory patients). The thesis claim "works in the real
world" needs the model to survive a *domain shift*: a modern rest-ECG
12-lead machine, different amplifier and filter chain, different electrode
placement, and local subjects. The lab's **Bionet CardioTouch 3000**
provides exactly that: export lead II, replay it through the deployed
INT8 model with the inference-time R detector (no gold annotations exist
for own recordings — that is the point).

## Tool delivered: `src/data/cardiotouch.py`

```
py src/data/cardiotouch.py --file export.csv          # or .txt / .wav
py src/data/cardiotouch.py --file export.csv --fs 1000 # no time column
```

Pipeline: parse (CSV/text/WAV auto-detected) → resample to 360 Hz →
same bandpass as training → **Pan-Tompkins** R detection → same windows,
z-score and RR features → INT8 TFLite model → beat-class distribution
report. Every step is the deployed pipeline, so the run measures the
system, not just the network.

## Data collection protocol (agreed plan)

1. Technician/clinician supervised 10 s rest-ECG recordings on the
   CardioTouch 3000; standard 12-lead placement; export lead II.
2. **Anonymize before it touches this repository** — no names/IDs inside
   files; consent log kept offline and separate.
3. Export formats: BMS-Plus workstation plain CSV preferred (adapter
   also reads WAV). If the machine only exports SCP-ECG, convert to CSV
   first and record the conversion step in the thesis methods section.
4. Suggested minimum for a first analysis chapter table: 20-30 subjects
   (mostly normal sinus rhythm; any arrhythmia subjects only if a
   cardiologist labels the beats).

## Analysis to report (when data exists)

| Metric | Meaning |
|---|---|
| R-detector sensitivity / PPV vs manual R marks | deployment-pipeline quality |
| N-agreement with clinician-reviewed normal beats | domain-shift robustness |
| False-S / false-V rates per hour on normal subjects | alarm-worthiness |
| Beat-class distribution vs expected rhythm | sanity check |

## Limitations to state honestly in the thesis

- Own recordings will be (initially) normal-rhythm only → S/V performance
  on CardioTouch data cannot be validated until labeled arrhythmia
  recordings exist.
- Class distribution shifts (rest vs ambulatory, modern electrodes) may
  change calibration; the confidence threshold may need re-tuning
  (documented as future work).
