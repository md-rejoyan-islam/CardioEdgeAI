# 🔴 Live Data from the CardioTouch 3000 — Guide

How ECG data gets out of the lab machine and into CardioEdgeAI in real
time. Three paths, pick by need:

| Path | What | Latency | Status |
|---|---|---|---|
| **A — watch-folder (recommended)** | LAN → BMS-Plus on PC → export CSV to a folder → auto-classified | ~1 s per recording | **tested end-to-end** (synthetic export, `tests/test_live_pipeline.py`) |
| **B — RS-232 serial stream** | null-modem cable → `tools/serial_capture.py` logs raw bytes | true stream | logger ready; protocol must be identified from a capture |
| **C — continuous wearable demo** | AD8232 + ESP32-S3 firmware (Phase 6) | per beat, true continuous | firmware written; board bring-up pending |

## Path A setup (do this in the lab)

1. Connect CardioTouch 3000 ↔ PC with a LAN cable. On the machine:
   **Device Setup → Network Setup** — set an IP in the same subnet as
   the PC. Install **BMS-Plus / EKG Viewer** on the PC and add the device.
2. Take a recording; export it (CSV preferred) into one fixed folder,
   e.g. `C:\BMS-Plus\Exports`.
3. On the same PC run:

   ```
   py tools/watch_folder.py --folder "C:/BMS-Plus/Exports"
   ```

   Every new export is classified within ~a second of finishing and:
   - results print to the console (`74 bpm, N=72 S=0 V=0`),
   - append to `Exports/live_results.log`,
   - a full per-beat JSON is written next to the file
     (`ct_export_100.result.json`).

Accepted formats: `.csv` / `.txt` (two columns: seconds, mV; header rows
tolerated; comma/semicolon/whitespace) and `.wav`. Files are only
processed once their size stops changing (export finished writing).

**Exports as SCP-ECG?** `src/data/cardiotouch.py` detects SCP files
(magic bytes + section table) and explains what to do instead of failing
silently. No pip-installable SCP parser exists (checked PyPI), and the
decoder should be written against a real sample file — keep the first
`.scp` export and finish it then, or export CSV from the EKG Viewer.

## Path B — serial capture (if the cable/port is available)

1. USB-serial adapter + null-modem cable to the machine's RS-232C port.
2. Find the port: `py tools/serial_capture.py --list`
3. Record baud 9600 first, then 115200 if that looks like noise:

   ```
   py tools/serial_capture.py --port COM3 --baud 9600
   ```

4. Start a recording on the machine; Ctrl+C to stop. You get
   `captures/ct_*.bin` (raw bytes) + `captures/ct_*.log` (timestamped hex
   + ASCII preview). Repeat for several recordings.
5. Inspect the `.log` for structure (file magic, lead labels, sample
   rates). With a capture in hand the stream format can be identified
   and a live serial → classify bridge added to this repo.

Note: Bionet's serial protocol is not publicly documented — asking
Bionet support for the CardioTouch 3000 communication protocol
(research/thesis use) is the fastest legitimate route.

## Path C — continuous demo

The thesis' continuous real-time pipeline is the ESP32 firmware in
`deploy/esp32/` (AD8232 front end, beat-by-beat output on serial). The
CardioTouch is the *reference/validation* device; the AD8232 rig is the
*wearable* being demonstrated. See `docs/PHASE6.md`.

## What was verified without hardware

`tests/test_live_pipeline.py` builds a synthetic export (60 s of raw
MIT-BIH record 100 MLII written as `time_s,mV` CSV with header) and
verifies the whole chain: parse → resample → bandpass → Pan-Tompkins →
windows → INT8 model → result files, both directly and through the
watch-folder tool. Result: 74 bpm, 72/72 beats classified N — consistent
with record 100 being ~99 % normal. The SCP detector was verified
against a minimal synthetic SCP header.
