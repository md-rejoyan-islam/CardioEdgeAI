# The Lab ECG Machine — Bionet CardioTouch 3000

Everything about the machine this project records its external-validation
ECG with: what it is, how it works, its connections, and how data leaves
it. The photos in [`images/`](images/) are the actual lab unit.

> বাংলায় বিস্তারিত হাতে-কলমে গাইড:
> [`manuals/CardioTouch_3000_Practical_Guide_BN.docx`](manuals/CardioTouch_3000_Practical_Guide_BN.docx)

| Folder | Contents |
|---|---|
| [`images/`](images/) | Photos of the lab machine (converted HEIC → JPG) |
| [`manuals/`](manuals/) | Practical guide (Bengali docx); add the official user manual PDF here when obtained |
| [`electrodes/`](electrodes/) | How ECG electrodes work + 12-lead placement |
| [`data-collection/`](data-collection/) | Step-by-step recording & export protocol for this thesis |

## What the machine is

The **Bionet CardioTouch 3000** is a digital **12-channel resting ECG
electrocardiograph** (Bionet Co., Ltd., Korea):

- 4.3-inch touchscreen, one-touch AUTO operation
- Simultaneous 12-lead acquisition and printout (thermal A4)
- Interpretation analysis, regular/negative-dose grid printing
- Resting **snapshot** device: each recording is a ~10 s 12-lead ECG —
  it is **not** a continuous ambulatory monitor (that role belongs to
  the project's AD8232 + ESP32 rig)

## How it works (signal chain)

```
heart's electrical depolarization (mV-level)
   -> 10 electrodes on the body (4 limb + 6 chest)
   -> protected analog front end (isolation, defib protection)
   -> amplification + filtering (bandpass, notch)
   -> ADC (digital samples)
   -> 12 simultaneous leads computed (I, II, III, aVR/aVL/aVF, V1-V6)
   -> on-screen waveform, automatic measurement (HR, intervals, axis)
   -> printout / transfer to PC
```

Each lead is a different *view* of the same electrical event — the same
heart activity appears with different amplitude/shape per lead. This
project uses **lead II** (RA → LL), the classic monitoring lead with the
tallest R waves, matching how the MLII signal of MIT-BIH was chosen.

## Back-panel connections (what leaves the machine)

| Port | Use in this project |
|---|---|
| **LAN (Ethernet)** | Main data path: machine ↔ PC running BMS-Plus / EKG Viewer (set IPs via Device Setup → Network Setup on the machine). Exports land on the PC → classified by `tools/watch_folder.py`. |
| **RS-232C serial** | Byte-level stream for protocol discovery: `tools/serial_capture.py` logs everything the machine sends. Protocol not publicly documented (Bionet support request pending). |
| **USB / printer** | Printouts and file export; not used by the pipeline. |

Live-data paths in detail: [`../docs/LIVE_DEMO.md`](../docs/LIVE_DEMO.md).

## Data formats out of the machine

| Format | Pipeline support |
|---|---|
| CSV (time, mV) via BMS-Plus export | ✅ direct input to the classifier |
| WAV | ✅ direct input |
| SCP-ECG | Detected + identified; decoder waits for a real sample file |
| PDF printout / paper | Human reading only |

## Photos

| File | Shows |
|---|---|
| `images/machine_quick_guide_card.jpg` | The lab unit with its quick-guide card: limb-electrode placement (RA/LA/RL/LL), chest V1–V6 diagram, AUTO recording flow — screen showing a **Lead Fault** warning (electrode contact issue, see troubleshooting below) |
| `images/machine_photo_2.jpg` | Lab machine — update this caption after reviewing |
| `images/machine_photo_3.jpg` | Lab machine — update this caption after reviewing |

## Everyday troubleshooting (from the quick-guide card)

- **"Lead Fault" on screen** — an electrode is off/dry/badly placed.
  Re-attach, clean the skin site, use fresh gel; check the cable snaps.
- Flat/noisy trace — check gel contact, patient movement, cable tension;
  re-run AUTO after fixing.
- Network transfer failing — confirm machine IP and PC IP share a
  subnet, and BMS-Plus device list points at the machine.

## Where this machine fits in the thesis

1. **External validation (Phase 7):** exported lead-II recordings test
   the deployed model on *its* amplifier/electrodes/patients — the
   domain-shift evidence.
2. **Gold-standard reference:** a clinician-read CardioTouch ECG beside
   the wearable rig's output for the same subject.
3. **Never the deployed hardware:** the model itself runs on ESP32; the
   CardioTouch is the data source and reference, not the target.
