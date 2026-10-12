#!/bin/bash
# Retry the afdb download whenever PhysioNet's route recovers (>100 KB/s).
# Exits when afdb is complete (marker: 04015.atr exists) or after 12 rounds.
cd "$(dirname "$0")/.."
for i in $(seq 1 12); do
  [ -f data/raw/afdb/04015.atr ] && echo "AFDB_COMPLETE" && exit 0
  speed=$(curl -s -o /dev/null -w "%{speed_download}" --max-time 15 \
          "https://physionet.org/files/afdb/1.0.0/RECORDS")
  echo "round $i: physionet ${speed} B/s"
  # awk instead of bc (bc is unavailable in Git Bash on Windows)
  if awk -v s="$speed" 'BEGIN{exit !(s > 100000)}'; then
    echo "route recovered - downloading afdb"
    py -c "import wfdb; wfdb.dl_database('afdb', dl_dir='data/raw/afdb')"
    [ -f data/raw/afdb/04015.atr ] && echo "AFDB_COMPLETE" && exit 0
  fi
  sleep 600
done
echo "AFDB_RETRY_EXHAUSTED"
