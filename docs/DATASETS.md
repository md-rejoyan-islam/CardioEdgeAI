# 📦 Dataset Inventory

Datasets used or planned for CardioEdgeAI, all stored locally under
`data/raw/<name>/` (git-ignored — re-download with the commands below).

| Dataset | Folder | Size | Status | Purpose |
|---|---|---|---|---|
| MIT-BIH Arrhythmia | `data/raw/mitdb/` | 90 MB | ✅ complete | Beat-level N/S/V training + testing (the main dataset) |
| PTB-XL 1.0.3 | `data/raw/ptbxl/` | ~7 GB | ✅ complete | Modern 12-lead generalization testing |
| MIT-BIH Atrial Fibrillation | `data/raw/afdb/` | ~900 MB | ⏳ pending (PhysioNet throttled) | AF rhythm-level future work |
| Kaggle ECG Heartbeat | `data/raw/kaggle_ecg/` | ~700 MB | 🔑 needs Kaggle account | Beginner-friendly CSV beats (derived from MIT-BIH/PTB) |
| Chapman-Shaoxing | `data/raw/chapman/` | ~10 GB | 🔑 needs credentialed PhysioNet | Large-scale 12-lead future work |

## MIT-BIH Arrhythmia Database (main dataset)

- **Source**: <https://physionet.org/content/mitdb/1.0.0/>
- **Contents**: 48 half-hour 2-lead records, 47 patients, 360 Hz,
  cardiologist beat annotations (`.dat/.hea/.atr`)
- **License**: PhysioNet open data — cite Moody & Mark (2001)
- **Downloaded with**:
  ```bash
  py src/data/download_mitbih.py
  ```

## PTB-XL 1.0.3

- **Source (canonical)**: <https://physionet.org/content/ptb-xl/1.0.3/>
- **Downloaded from**: Hugging Face mirror
  [`roisincrtai/ptb-xl-1.0.3`](https://huggingface.co/datasets/roisincrtai/ptb-xl-1.0.3)
  — byte-identical layout (87,205 files incl. `SHA256SUMS.txt` for
  verification). Reason: PhysioNet route was throttled to ~60 KB/s from
  this network; the HF CDN delivered the same files much faster.
- **Contents**: 21,799 × 10 s 12-lead ECGs, 18,869 patients, 500 Hz
  (`records500/`) + 100 Hz (`records100/`), `ptbxl_database.csv` with
  SNOMED labels and ready-made `strat_fold` splits
- **License**: Open — cite Wagner et al., *Scientific Data* (2020), CC-BY-4.0
- **Downloaded with**:
  ```bash
  py -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='roisincrtai/ptb-xl-1.0.3', repo_type='dataset', \
  local_dir='data/raw/ptbxl', max_workers=16)"
  ```
- **Integrity**: run `SHA256SUMS.txt` verification after download (see below)

## MIT-BIH Atrial Fibrillation Database (afdb)

- **Source**: <https://physionet.org/content/afdb/1.0.0/>
- **Contents**: 25 long-term (~10 h) 2-lead records, 250 Hz, AF episode
  annotations — the dataset for rhythm-level AF detection future work
- **Download** (run when PhysioNet is responsive):
  ```bash
  py -c "import wfdb; wfdb.dl_database('afdb', dl_dir='data/raw/afdb')"
  ```

## Kaggle ECG Heartbeat Categorization

- **Source**: <https://www.kaggle.com/datasets/shayanfazeli/heartbeat>
- Derived (pre-segmented beat CSVs) from MIT-BIH + PTB — useful for
  quick Pandas experiments; **not** a source of new information for this
  project (our pipeline already builds beat windows from the raw MIT-BIH)
- **Requires a Kaggle account + API token** (`kaggle.json`):
  ```bash
  pip install kaggle
  # place kaggle.json in %USERPROFILE%\.kaggle\ first
  kaggle datasets download -d shayanfazeli/heartbeat -p data/raw/kaggle_ecg --unzip
  ```

## Chapman-Shaoxing 12-lead

- **Source**: <https://physionet.org/content/ecg-arrhythmia/1.0.0/>
- **Credentialed access required**: PhysioNet account → *credentialing*
  (DUA acceptance) → then `wfdb` downloads work. Apply once, approval
  usually takes a day.
- 45,152 patients × 10 s 12-lead — production-scale future work only.

## Verify a downloaded dataset (PTB-XL example)

```bash
cd data/raw/ptbxl
py -c "
import hashlib, pathlib
ok = bad = 0
for line in open('SHA256SUMS.txt', encoding='utf-8'):
    h, name = line.split()
    p = pathlib.Path(name)
    if not p.exists(): bad += 1; continue
    if hashlib.sha256(p.read_bytes()).hexdigest() == h: ok += 1
    else: bad += 1
print(f'verified {ok} files, {bad} missing/corrupt')
"
```
