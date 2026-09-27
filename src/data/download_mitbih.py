"""Download the MIT-BIH Arrhythmia Database into the local data folder.

Usage (from the repository root):
    python src/data/download_mitbih.py --out data/raw/mitdb

Reference: https://physionet.org/content/mitdb/1.0.0/
The database is ~100 MB; each record arrives as a signal (.dat),
header (.hea) and annotation (.atr) file trio.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import wfdb  # noqa: E402

from src.config import MITDB_RECORDS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="data/raw/mitdb",
        help="output directory (default: data/raw/mitdb)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(MITDB_RECORDS)} MIT-BIH records to {out_dir} ...")
    wfdb.dl_database("mitdb", dl_dir=str(out_dir), records=list(MITDB_RECORDS))

    n = len(list(out_dir.glob("*.hea")))
    print(f"Done. {n} header files present in {out_dir}.")
    print("Next step: preprocessing pipeline (filtering -> R-peak -> segmentation).")


if __name__ == "__main__":
    main()
