# Phase 1 — Environment, Dataset, Exploration

**Status:** complete · **Commit:** see git history

## What was done

1. **Environment** (Windows 11, Git Bash):
   - Python 3.14.2 (`py` launcher)
   - TensorFlow 2.22.0-rc0 (first TF line with Python 3.14 wheels)
   - numpy 2.5.2, scipy, scikit-learn 1.9.1, matplotlib, wfdb 4.3.1,
     neurokit2 0.2.13
   - Install: `py -m pip install -r requirements.txt`

2. **Dataset download** — `py src/data/download_mitbih.py`
   - MIT-BIH Arrhythmia Database, all 48 records (`.dat`/`.hea`/`.atr`),
     ~85 MB into `data/raw/mitdb/` (git-ignored; re-download anytime).

3. **Sanity checks & figures** — `py src/data/visualize.py`

   | Figure | Content |
   |---|---|
   | `figures/phase1_signal_quality.png` | Record 100 (MLII), first 5 s, raw vs 0.5–40 Hz bandpass-filtered |
   | `figures/phase1_beat_examples.png` | One R-centered window per class: N (100), S (100, A-type), V (234) |

   Record-level annotation counts confirmed the expected beat types, e.g.
   record 100: `N:2239, A:33, V:1`; record 234: `N:2700, J:50, V:3`.

## Observations

- The bandpass filter visibly removes baseline wander without distorting
  QRS morphology — safe to apply before segmentation.
- Class imbalance is severe (see Phase 2 stats): S beats are ~2.8 % of the
  database. Training will need class weighting or focal loss; plain
  accuracy will be misleading.

## Next

Phase 2 — preprocessing pipeline and the N/S/V beat dataset build.
