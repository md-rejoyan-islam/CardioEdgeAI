# How ECG Electrodes Work & 12-Lead Placement

The practical, physics-first version. For the hands-on Bengali walkthrough
see [`../manuals/CardioTouch_3000_Practical_Guide_BN.docx`](../manuals/CardioTouch_3000_Practical_Guide_BN.docx).

## What an electrode actually does

The heart's depolarization wave creates tiny potential differences on the
skin (~0.5–4 mV). An electrode is a **transducer between ionic currents
in tissue and electron currents in wire**:

```
skin <-> conductive gel <-> Ag/AgCl pellet <-> metal snap <-> ECG cable
```

- **Ag/AgCl (silver/silver-chloride)** is the standard sensing element:
  it is *non-polarizable*, meaning it exchanges charge with the gel
  without creating its own unstable voltage.
- **Gel** (salt bridge) lowers skin impedance — dead skin cells are the
  main enemy; light abrasion/alcohol prep cuts artifacts dramatically.
- The **right-leg (RL) electrode is not a sensor** — it carries an
  inverted common-mode signal back to the body to cancel mains
  interference (driven-right-leg circuit). A bad RL electrode = noisy
  everything.
- Electrodes are single-patient-use: gel dries, adhesive loosens,
  infection control.

## The 10 electrodes → 12 leads

A "lead" is not an electrode — it is a *voltage difference the machine
computes* between electrodes:

**Limb electrodes (4):**

| Electrode | Position |
|---|---|
| RA | right arm (wrist/inner forearm) |
| LA | left arm |
| RL | right leg (reference/drive-back, ankle) |
| LL | left leg |

Limb leads computed from them:

- **I** = LA − RA · **II** = LL − RA · **III** = LL − LA
- Augmented: aVR, aVL, aVF (each vs. the average of the other two limbs)

**Chest (precordial) electrodes (6):**

| Lead | Position on the chest |
|---|---|
| V1 | 4th intercostal space, right sternal border |
| V2 | 4th intercostal space, left sternal border |
| V3 | midway between V2 and V4 |
| V4 | 5th intercostal space, left mid-clavicular line |
| V5 | same level as V4, anterior axillary line |
| V6 | same level as V4/V5, mid-axillary line |

V1–V6 look at the heart's horizontal plane (best for R/S progression,
bundle blocks, anterior ischemia); the limb leads look at the frontal
plane.

## Why this project uses lead II

Lead II (LL − RA) runs roughly parallel to the heart's normal
depolarization axis, so P waves and R waves are tallest and cleanest —
it is the classic monitoring lead and the same choice MIT-BIH made with
its MLII channel. All training windows, RR timing, and the ESP32 rig's
single AD8232 channel correspond to lead II.

## Practical placement checklist (rest ECG)

1. Explain the procedure, get consent; supine/sitting, relaxed, still.
2. Prep skin: light alcohol swab, let dry; shave tiny spots if needed.
3. Limb electrodes on inner forearms/ankles-inside, gel pad flat.
4. Chest V1–V6 by palpation (sternal angle → 2nd intercostal, count down
   to 4th/5th); symmetric, not over bone edges.
5. Cable off the chest (hanging weight pulls electrodes loose).
6. On the machine: check all 12 traces — any flat/noisy trace or a
   **Lead Fault** warning means fix that electrode before recording.
7. Remove gently, mark the paper/export, dispose electrodes.

## Artifact quick table

| Symptom | Usual cause |
|---|---|
| Wandering baseline | loose electrode, gel drying, breathing deeply, cable tension |
| 50/60 Hz fuzz | RL electrode bad, cable near power lines, unshielded environment |
| Sudden spikes | patient movement, electrode snap contacting metal |
| Flat one lead | that electrode off / dried / wrong spot |
