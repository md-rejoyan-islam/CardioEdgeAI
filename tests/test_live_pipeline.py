"""End-to-end test of the live pipeline using synthetic exports.

Builds a CSV that mimics a CardioTouch/BMS-Plus export (time_s, mV at
360 Hz, with a header row) from 60 s of MIT-BIH record 100's raw MLII
signal, then:

1. classifies it directly via cardiotouch.classify_recording
2. drops it into a watched folder and verifies the watch-folder tool
   picks it up and writes the result files

Run: py tests/test_live_pipeline.py   (needs data/raw/mitdb + models/)
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

import numpy as np  # noqa: E402
import wfdb  # noqa: E402

from src.data.cardiotouch import classify_recording  # noqa: E402

DURATION_S = 60
RECORD = "100"


def make_export_csv(out_dir: Path) -> Path:
    """Write a CardioTouch-style CSV: header + 'time_s,mV' rows."""
    rec = wfdb.rdrecord(str(REPO / "data" / "raw" / "mitdb" / RECORD))
    n = DURATION_S * 360
    raw = rec.p_signal[:n, 0]  # unfiltered MLII in mV, like an export
    path = out_dir / f"ct_export_{RECORD}.csv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Time(s),Lead II (mV)\n")  # header row must be tolerated
        for i, v in enumerate(raw):
            fh.write(f"{i / 360:.5f},{v:.6f}\n")
    return path


def main() -> None:
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = make_export_csv(Path(tmp))
        print(f"synthetic export: {csv_path.name} "
              f"({csv_path.stat().st_size / 1024:.0f} KB)")

        # --- direct classification -------------------------------------
        result = classify_recording(csv_path)
        print("direct result:", {k: result[k] for k in
                                 ("fs", "bpm", "n_beats", "distribution")})
        if result["n_beats"] < 50:
            failures.append(f"too few beats classified: {result['n_beats']}")
        if result["distribution"].get("N", 0) < 0.9 * result["n_beats"]:
            failures.append(
                f"record 100 is ~99% normal; got {result['distribution']}")
        if not 55 <= result["bpm"] <= 90:
            failures.append(f"implausible bpm for record 100: {result['bpm']}")

        # --- watch-folder pipeline -------------------------------------
        watch_dir = Path(tmp) / "watched"
        watch_dir.mkdir()
        log_path = watch_dir / "live_results.log"
        watcher = subprocess.Popen(
            [sys.executable, str(REPO / "tools" / "watch_folder.py"),
             "--folder", str(watch_dir), "--interval", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            time.sleep(3)                      # let the watcher start
            (watch_dir / csv_path.name).write_text(
                csv_path.read_text(encoding="utf-8"), encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                if log_path.exists():
                    break
                time.sleep(2)
            time.sleep(2)                      # let result.json flush
        finally:
            watcher.terminate()
            out, _ = watcher.communicate(timeout=10)
        print("--- watcher output (tail) ---")
        print("\n".join(out.splitlines()[-6:]))

        if not log_path.exists():
            failures.append("watch-folder never processed the dropped file")
        else:
            line = log_path.read_text(encoding="utf-8").strip()
            print("live_results.log:", line)
            if "\tOK\t" not in line:
                failures.append(f"watch-folder logged failure: {line}")
            result_json = watch_dir / csv_path.with_suffix(
                ".result.json").name
            if not result_json.exists():
                failures.append("no .result.json written")
            else:
                saved = json.loads(result_json.read_text(encoding="utf-8"))
                if saved["n_beats"] != result["n_beats"]:
                    failures.append("watcher result differs from direct run")

    print("\n" + ("ALL LIVE-PIPELINE TESTS PASSED" if not failures
                  else "FAILURES:\n- " + "\n- ".join(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
