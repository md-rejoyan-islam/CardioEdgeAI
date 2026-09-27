"""Live demo tool 1/3 — watch-folder auto-classifier.

Watches a folder (e.g. the BMS-Plus export directory) and classifies every
new ECG file with the deployed INT8 model the moment it lands — the
"near-live" CardioTouch pipeline (Path A in docs/LIVE_DEMO.md).

Usage:
    py tools/watch_folder.py --folder "C:/BMS-Plus/Exports"
    py tools/watch_folder.py --folder ./exports --interval 2

Accepted: .csv .txt .wav (see src/data/cardiotouch.py). Files are
classified only after their size stops changing (export fully written).
Results print to console and append to <folder>/live_results.log;
a per-file .result.json is written next to each export.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.cardiotouch import classify_recording  # noqa: E402

WATCHED_SUFFIXES = {".csv", ".txt", ".wav"}
SETTLE_POLLS = 2  # file must keep the same size this many polls in a row


def process(path: Path, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        result = classify_recording(path, verbose=True)
    except Exception as exc:  # noqa: BLE001 — demo tool: report and continue
        print(f"[{stamp}] FAILED {path.name}: {exc}")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{stamp}\tFAIL\t{path.name}\t{exc}\n")
        return
    path.with_suffix(".result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    dist = result["distribution"]
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp}\tOK\t{path.name}\t{result['bpm']} bpm\t"
                 f"N={dist.get('N', 0)} S={dist.get('S', 0)} "
                 f"V={dist.get('V', 0)}\n")
    print(f"[{stamp}] logged -> {log_path.name} + "
          f"{path.with_suffix('.result.json').name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True,
                        help="folder to watch (BMS-Plus export directory)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="polling interval in seconds (default 2)")
    args = parser.parse_args()
    folder = args.folder
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "live_results.log"

    print(f"watching {folder} (interval {args.interval}s) — Ctrl+C to stop")
    known = set()      # fully processed files
    pending = {}       # file -> (last_size, stable_polls)
    try:
        while True:
            seen = set()
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in WATCHED_SUFFIXES:
                    continue
                if path.name.startswith("~$"):  # office temp files
                    continue
                seen.add(path)
                if path not in known and path not in pending:
                    pending[path] = [path.stat().st_size, 0]

            # wait for exports to finish writing before classifying
            for path in list(pending):
                if path not in seen:
                    del pending[path]  # disappeared mid-write
                    continue
                size = path.stat().st_size
                if size > 0 and size == pending[path][0]:
                    pending[path][1] += 1
                else:
                    pending[path] = [size, 0]
                if pending[path][1] >= SETTLE_POLLS:
                    del pending[path]
                    known.add(path)
                    process(path, log_path)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
