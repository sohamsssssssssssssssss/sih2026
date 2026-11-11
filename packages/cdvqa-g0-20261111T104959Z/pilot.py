"""CDVQA G0 package: offline CPU preflight only. No weights, no images, no GPU work."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the package inventory equal to package-manifest.json
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eval.cdvqa_pairs import assert_no_pair_leak, load_vocabulary, normalise  # noqa: E402

MANIFESTS = ROOT / "data/manifests/cdvqa"
EXPECTED_ROWS = {"train-8k.v1.jsonl": 8000, "eval-test1-400.v1.jsonl": 400, "eval-test2-400.v1.jsonl": 400}
FORBIDDEN_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".safetensors", ".bin", ".pt", ".pth"}


def sha256_file(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    package = json.loads((ROOT / "package-manifest.json").read_text())
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "package-manifest.json")
    if {str(p.relative_to(ROOT)) for p in files} != set(package["payload_sha256"]):
        raise ValueError("Package file inventory differs from package-manifest.json")
    for name, expected in package["payload_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Package payload mismatch: {name}")
    heavy = [str(p.relative_to(ROOT)) for p in files if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    if heavy:
        raise ValueError(f"Package must not contain images or weights: {heavy[:3]}")
    provenance = json.loads((MANIFESTS / "provenance.json").read_text())
    for name, expected in provenance["manifest_sha256"].items():
        if sha256_file(MANIFESTS / name) != expected:
            raise ValueError(f"Manifest hash mismatch: {name}")
    vocab = load_vocabulary(MANIFESTS / "provenance.json")
    counts = {}
    for name, expected in EXPECTED_ROWS.items():
        rows = [json.loads(line) for line in (MANIFESTS / name).read_text().splitlines() if line.strip()]
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} rows, found {len(rows)}")
        bad = [r["id"] for r in rows if normalise(r["answer"]) not in vocab[r["question_type"]]]
        if bad:
            raise ValueError(f"{name}: answers outside vocabulary, e.g. {bad[:3]}")
        counts[name] = {"rows": len(rows), "pairs": len({r["pair_id"] for r in rows})}
    assert_no_pair_leak(MANIFESTS / "train-8k.v1.jsonl", MANIFESTS / "eval-test1-400.v1.jsonl")
    assert_no_pair_leak(MANIFESTS / "train-8k.v1.jsonl", MANIFESTS / "eval-test2-400.v1.jsonl")
    return {"payload": "PASS", "manifests": "PASS", "vocabulary": "PASS", "leakage": "PASS",
            "images_in_package": False, "counts": counts}


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2))
