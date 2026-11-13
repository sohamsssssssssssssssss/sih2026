#!/usr/bin/env bash
# CPU-only Kaggle cell (Accelerator: None, Internet: on): fetch the third-party DIOR-VOC mirror,
# hash it, extract it to /kaggle/tmp and write only small reports to /kaggle/working.
# Large output in /kaggle/working is what stalled the G2 commit save for ~9 h, so images stay out of it.
# Paste into a %%bash cell, or run: bash kaggle/fetch_dior_cpu.sh
set -euo pipefail
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND"' ERR

URL="https://zenodo.org/records/11213149/files/DIOR-VOC.zip"
# Empty: record the hash only. After the first run, paste the recorded hash here so later runs enforce it.
EXPECTED_SHA256=""
T=/kaggle/tmp/dior
W=/kaggle/working
mkdir -p "$T" "$W"

curl -L --fail --retry 3 -o "$T/DIOR-VOC.zip" "$URL"
ls -l "$T/DIOR-VOC.zip"
ACTUAL=$(sha256sum "$T/DIOR-VOC.zip" | cut -d' ' -f1)
echo "$ACTUAL  DIOR-VOC.zip  $URL" | tee "$W/DIOR-VOC.zip.sha256"
if [ -n "$EXPECTED_SHA256" ] && [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
  echo "sha256 mismatch: expected $EXPECTED_SHA256"; exit 1
fi

python - "$T/DIOR-VOC.zip" "$T/extracted" <<'EOF'
import os, sys, zipfile
archive, dest = sys.argv[1], os.path.realpath(sys.argv[2])
with zipfile.ZipFile(archive) as z:
    for m in z.infolist():
        p = os.path.realpath(os.path.join(dest, m.filename))
        if p != dest and not p.startswith(dest + os.sep):
            raise SystemExit(f"unsafe path in zip: {m.filename}")
    z.extractall(dest)
    print("extracted", len(z.infolist()), "entries")
EOF
rm -f "$T/DIOR-VOC.zip"

cd "$T/extracted"
{
  echo "source: $URL"
  echo "sha256: $ACTUAL"
  echo "--- layout"; find . -maxdepth 3 -type d | sort | head -30
  echo "--- split files"; find . -maxdepth 5 -path "*ImageSets/Main*" -name "*.txt" -print0 | xargs -0 -r wc -l | tail -8
  echo "--- annotations"; find . -name "*.xml" | wc -l
  echo "--- images"; find . \( -name "*.jpg" -o -name "*.png" \) | wc -l
} | tee "$W/dior-layout.txt"
find . -type f | sort > "$W/dior-files.txt"
du -sh "$T/extracted" "$W"
echo "CELL DONE (data at $T/extracted; /kaggle/tmp is not saved with the notebook output)"
