#!/usr/bin/env bash
# CPU-only Kaggle cell (Accelerator: None, Internet: on): fetch the third-party DIOR-VOC mirror,
# hash it, extract it to /kaggle/tmp and write only small reports to /kaggle/working.
# Large output in /kaggle/working is what stalled the G2 commit save for ~9 h, so images stay out of it.
# Paste into a %%bash cell, or run: bash kaggle/fetch_dior_cpu.sh
set -euo pipefail
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND"' ERR

URL="https://zenodo.org/records/11213149/files/DIOR-VOC.zip"
# Pinned from the verified Kaggle CPU run (13 Nov 2026); kaggle/eval_count_detection.py pins the same bytes.
EXPECTED_SHA256="7a386a56ba815333627d89d3b8b372d1af2a07b78955c017530ec62b7f1e0540"
EXPECTED_BYTES=7423852627
T=/kaggle/tmp/dior
W=/kaggle/working
mkdir -p "$T" "$W"

curl -L --fail --retry 3 -o "$T/DIOR-VOC.zip" "$URL"
ls -l "$T/DIOR-VOC.zip"
ACTUAL=$(sha256sum "$T/DIOR-VOC.zip" | cut -d' ' -f1)
echo "$ACTUAL  DIOR-VOC.zip  $URL" | tee "$W/DIOR-VOC.zip.sha256"
if [ "$ACTUAL" != "$EXPECTED_SHA256" ] || [ "$(stat -c %s "$T/DIOR-VOC.zip")" != "$EXPECTED_BYTES" ]; then
  echo "HARD STOP: mirror bytes differ from the pinned sha256/size ($EXPECTED_SHA256, $EXPECTED_BYTES)"; exit 1
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
