# Recording & Exporting ECG Data from the CardioTouch 3000

The complete lab-session protocol, from subject to classified file.
Follow it top-to-bottom every time so every recording is usable.

## 0. Before any subject (one-time setup)

- [ ] LAN cable machine ↔ PC; machine **Device Setup → Network Setup**
      with a fixed IP in the PC's subnet
- [ ] BMS-Plus / EKG Viewer installed on the PC, device added & reachable
- [ ] Export folder fixed, e.g. `C:\BMS-Plus\Exports` — this is what
      `tools/watch_folder.py` monitors
- [ ] Fresh electrodes, alcohol swabs, marker, consent forms
- [ ] Watch-folder running on the PC:
      `py tools/watch_folder.py --folder "C:/BMS-Plus/Exports"`

## 1. Record (per subject, ~5 minutes)

1. Consent + explain; subject lies supine, relaxed, phone off/body still.
2. Skin prep + 10 electrodes per
   [`../electrodes/README.md`](../electrodes/README.md) checklist.
3. On the machine confirm **12 clean traces, no Lead Fault**.
4. Press **AUTO** → ~10 s 12-lead snapshot is acquired and printed.
5. Let the machine's on-screen interpretation print *for the clinician's
   reference only* — it is never used as a label for the model.

## 2. Export the lead-II data

1. In BMS-Plus/EKG Viewer: open the just-transferred study.
2. Export **lead II as CSV** (two columns: time and mV preferred; a
   plain WAV export also works) into the watched folder.
3. The watch-folder tool auto-classifies it within ~1 s — result prints
   on console, lands in `live_results.log` and a `.result.json` next to
   the export.
4. If the machine only produces SCP-ECG: keep the `.scp` file (the
   pipeline detects and explains it) and convert it in the Viewer to CSV.

## 3. Anonymize — non-negotiable

- File name must contain **no name, no ID, no date of birth**:
  use `ct_<seq>_<session>.csv` (e.g. `ct_007_s03.csv`).
- Open the CSV and strip any header lines carrying patient metadata.
- The consent ↔ subject-code linking sheet stays **offline**, never in
  this repository.
- Record age group / sex / rhythm in a separate `subjects.csv` keyed by
  the anonymous code only.

## 4. After the session

- Copy the whole export folder into `data/external/` (git-ignored) for
  analysis; keep the raw originals on a backup drive.
- Any arrhythmia findings a clinician labels beat-by-beat become the
  only ground truth for external S/V performance (see
  [`../../docs/PHASE7.md`](../../docs/PHASE7.md)).
- Update `docs/PHASE7.md` results tables as the collection grows.

## Quick command reference

```bash
# live classification of every new export
py tools/watch_folder.py --folder "C:/BMS-Plus/Exports"

# classify a single file explicitly
py src/data/cardiotouch.py --file ct_007_s03.csv

# no time column in the CSV? force the sampling rate
py src/data/cardiotouch.py --file ct_007_s03.csv --fs 1000

# serial capture instead of LAN (protocol discovery)
py tools/serial_capture.py --list
py tools/serial_capture.py --port COM3 --baud 9600
```

## Ethics rules (short form)

Supervised recordings only (technician/clinician present) · informed
consent before electrodes touch skin · anonymize before anything enters
a computer folder this repository touches · research prototype, not a
medical device — no clinical decisions from the model's output.
