"""Resolve every image referenced by the CDVQA manifests; check size/mode; hash used images.

Run: python3.11 data/cdvqa/verify_images.py
Writes results/cdvqa-g0/image-check.json (committed) and ~/data/cdvqa/SHA256SUMS-used-images.txt (not in git).
"""
import hashlib
import json
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "data/manifests/cdvqa"
IMAGE_ROOT = Path.home() / "data/cdvqa/images/SECOND_train_set"
SUMS = Path.home() / "data/cdvqa/SHA256SUMS-used-images.txt"
RESULT = REPO / "results/cdvqa-g0/image-check.json"
MANIFEST_NAMES = ("train-8k.v2.jsonl", "eval-test1-400.v1.jsonl", "eval-test2-400.v1.jsonl")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def used_images():
    """Every relative image path referenced by the three manifests, with its manifest names."""
    used, report = {}, {}
    for name in MANIFEST_NAMES:
        rows = [json.loads(line) for line in (MANIFESTS / name).read_text().splitlines() if line.strip()]
        pairs = {(r["before_image"], r["after_image"]) for r in rows}
        report[name] = {"qa": len(rows), "pairs": len(pairs)}
        for before, after in pairs:
            for rel in (before, after):
                used.setdefault(rel, set()).add(name)
    return used, report


ARCHIVES = Path.home() / "data/cdvqa/images/second_dataset"
ZIP = Path.home() / "data/cdvqa/cdvqa-images.zip"


def update_provenance(result):
    """Record archive, checksum-list and zip hashes in provenance.json (the builder preserves this block only if
    it is re-run before this script; run this script last)."""
    provenance_path = MANIFESTS / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    archives = {}
    for name, expected in (("SECOND_train_set.rar", "5ee2a82b5824b3f5e3c5bfaf018835862e7623993ff9ac5ce57fd11fcfad2b4e"),
                           ("SECOND_total_test.zip", "bf347e52bb802a17878faccee455319a38360dab94c202d88ee6746e9d7253c9")):
        actual = sha256(ARCHIVES / name)
        if actual != expected:
            raise ValueError(f"archive hash mismatch: {name}")
        archives[name] = {"sha256": actual, "source": "Google Drive folder linked from captain-whu.github.io/SCD (second_dataset)",
                          "note": "downloaded through the browser; the browser auto-unpacked the outer second_dataset.zip, so no outer-archive hash exists"}
    images = {"archives": archives, "image_root": "~/data/cdvqa/images/SECOND_train_set (im1, im2, label1, label2)",
              "layout": "CDVQA file names all live in SECOND_train_set; test/ is unused",
              "sha256sum_list": {"path": "~/data/cdvqa/SHA256SUMS-used-images.txt",
                                 "sha256": hashlib.sha256(SUMS.read_bytes()).hexdigest(),
                                 "entries": result["images_ok"]},
              "zip": {"path": "~/data/cdvqa/cdvqa-images.zip", "entries": result["images_ok"],
                      "sha256": sha256(ZIP) if ZIP.exists() else None},
              "verified": {"distinct_images": result["distinct_images_used"], "failures": result["failure_count"],
                           "all_512x512_rgb": result["failure_count"] == 0}}
    provenance["images"] = images
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def main():
    used, per_manifest = used_images()
    failures, hashes = [], {}
    for rel in sorted(used):
        path = IMAGE_ROOT / rel
        if not path.is_file():
            failures.append({"path": rel, "error": "missing"})
            continue
        with Image.open(path) as image:
            if image.size != (512, 512) or image.mode != "RGB":
                failures.append({"path": rel, "error": f"size {image.size} mode {image.mode}"})
                continue
        hashes[rel] = sha256(path)
    for name, info in per_manifest.items():
        resolved = sum(1 for before, after in {(r["before_image"], r["after_image"]) for r in
                       (json.loads(line) for line in (MANIFESTS / name).read_text().splitlines() if line.strip())}
                       if before not in {f["path"] for f in failures} and after not in {f["path"] for f in failures})
        info["pairs_resolved"] = resolved
        info["pairs_resolved_pct"] = round(100.0 * resolved / info["pairs"], 4)
    SUMS.write_text("".join(f"{hashes[rel]}  {rel}\n" for rel in sorted(hashes)))
    result = {
        "image_root": "~/data/cdvqa/images/SECOND_train_set (folders im1, im2, label1, label2)",
        "requirement": "512x512 RGB, both before (im1) and after (im2)",
        "distinct_images_used": len(used),
        "images_ok": len(hashes),
        "failures": failures[:50],
        "failure_count": len(failures),
        "per_manifest": per_manifest,
        "used_images_sha256_list_sha256": hashlib.sha256(SUMS.read_bytes()).hexdigest(),
        "used_images_sha256_list": SUMS.name,
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    update_provenance(result)
    print(json.dumps({k: result[k] for k in ("distinct_images_used", "images_ok", "failure_count", "per_manifest")}, indent=2))


if __name__ == "__main__":
    main()
