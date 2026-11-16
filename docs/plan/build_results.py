"""Generate docs/plan/RESULTS.md from results files. Never edit RESULTS.md by hand.

Usage (from the repo root): python3.11 docs/plan/build_results.py [--transfer-dir ~/ben-work/transfer-run-1]
Inputs: the 16 Nov Kaggle transfer outputs (outside git; sha256 of each is printed into RESULTS.md) and
committed files read with `git show <commit>:<path>`. Output is deterministic (no timestamps).
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "RESULTS.md"
TRANSFER_COMMIT = "4707d4c"  # feature/ben-landcover; code hashes in ben-transfer.json are checked against it
BEN_COMMIT, BEN_ANALYSIS = "99f27ea", "results/ben-landcover/analysis.json"
G3_COMMIT = "89fc5bf"  # training/cdvqa-g0
MODES = ("s1", "s2", "s1s2")
CONDS = ("base_real", "adapter_real", "adapter_identical", "adapter_swapped")
DIFFS = ("europe-ben_test", "india-europe", "india-ben_test")

# Copied exactly from SATQUERY_CONTEXT.md (16 Nov 2026), sections 6, 7b and 7c.
EARLIER = [
    ("Confirmed baseline: Qwen2.5-VL-3B zero-shot RSVQA-LR = 0.565 acc, 200 samples, commit bbbde2c (`eval/eval.py`, results_v2.json).",
     "**VERIFY**: no `results_v2.json` exists anywhere in git history; commit bbbde2c changes eval code only."),
    ("RSVQA-LR val (389 Q): adapter_real macro 0.696 vs base_real 0.468.",
     "commit 20596be (`training/rgb-pilot-b12-b13`), `results/rgb-pilot-b13/SOURCE.md`: adapter_real 0.6957, base_real 0.4683."),
    ("Counting 0.24 with real and blank image; adapter likely always answers \"0\" (24/100 count answers are \"0\").",
     "commit 20596be, `results/rgb-pilot-b13/SOURCE.md`: count 0.2400 for adapter_real and adapter_constant."),
    ("G1 base zero-shot test1-400: macro 0.395 vs majority 0.44 (change_or_not 0.74, smallest_change 0.16, change_ratio 0.20).",
     "commit 89fc5bf, `results/cdvqa-g3/test1-result.json`: base_real and majority_baseline (values checked below)."),
]


def git_bytes(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], check=True, capture_output=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def ci(c, d=4):
    return "—" if c is None else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def sd(x, d=4):
    return f"{x:+.{d}f}"


def transfer_section(tdir):
    files = {n: (tdir / n).read_bytes() for n in ["ben-transfer.json"] + [f"ben-landcover-{m}.json" for m in MODES]}
    hashes = {n: sha(b) for n, b in files.items()}
    t = json.loads(files["ben-transfer.json"])
    committed = json.loads(git_bytes(BEN_COMMIT, BEN_ANALYSIS))["macro_map"]
    code_ok = {k: t["code"][f"{k}_py_sha256"] == sha(git_bytes(TRANSFER_COMMIT, f"ben/{k}.py")) for k in ("transfer", "landcover")}
    assert all(code_ok.values()), f"ben-transfer.json code hashes do not match {TRANSFER_COMMIT}: {code_ok}"
    src = lambda n: f"`{n}` ({hashes[n][:12]})"
    L = ["## 1. BEN land cover → WorldCover transfer (Kaggle run, 16 Nov 2026)", "",
         f"Code commit {TRANSFER_COMMIT} (`feature/ben-landcover`). The sha256 values of `ben/transfer.py` and `ben/landcover.py` "
         f"recorded in ben-transfer.json match {TRANSFER_COMMIT} (checked by this script). "
         f"WorldCover pack MANIFEST sha256 `{t['wc_manifest_sha256']}`. Device: {t['device']}, torch {t['torch_version']}.", "",
         "Source files (outside git, in the transfer-run directory):", "", "| File | sha256 |", "|---|---|"]
    L += [f"| `{n}` | `{h}` |" for n, h in hashes.items()]
    L += ["", "### 1a. BEN test macro mAP, 19 classes (retrain vs committed run)", "",
          "| Mode | Retrain mAP | CI95 | Prior baseline | Shuffled control | Committed (99f27ea) | Match | Source |",
          "|---|---|---|---|---|---|---|---|"]
    for m in MODES:
        n = f"ben-landcover-{m}.json"
        r = json.loads(files[n])
        assert r["macro_map"] == t["modes"][m]["ben_test_macro_map_19"]
        L.append(f"| {m} | {f(r['macro_map'])} | {ci(r['bootstrap']['macro_map_ci95'])} | {f(r['prior_baseline']['macro_map'])} | "
                 f"{f(r['control']['macro_map'])} | {f(committed[m])} | {'exact' if r['macro_map'] == committed[m] else 'DIFFERS'} | {src(n)} |")
    for m in MODES:
        res = t["modes"][m]
        common, coarse = res["common_classes"], res["coarse_classes"]
        L += ["", f"### 1b. Mode {m}: coarse-class mAP on WorldCover", "",
              f"Groups ben_test / india / europe use the common class set ({', '.join(common)}). Each region row uses its own "
              f"class set (classes with ≥10 positives in that region), so region rows are not directly comparable with each other.", "",
              "| Group | n | Classes | mAP | CI95 | Prevalence | Shuffled | Source |", "|---|---|---|---|---|---|---|---|"]
        for g in ("ben_test", "india", "europe"):
            x = res["groups_common"][g]
            L.append(f"| **{g}** (common set) | {x['n']} | {len(x['classes_used'])} | {f(x['macro_map'])} | {ci(x['macro_map_ci95'])} | "
                     f"{f(x['prevalence_macro_map'])} | {f(x['shuffled_macro_map'])} | {src('ben-transfer.json')} |")
        for g in t["regions"]:
            x = res["groups"][g]
            names = ", ".join(coarse[i] for i in x["classes_used"])
            L.append(f"| {g} | {x['n']} | {names} | {f(x['macro_map'])} | {ci(x['macro_map_ci95'])} | "
                     f"{f(x['prevalence_macro_map'])} | {f(x['shuffled_macro_map'])} | {src('ben-transfer.json')} |")
        L += ["", "| Difference (common set, unpaired bootstrap) | Value | CI95 | Source |", "|---|---|---|---|"]
        for d in DIFFS:
            x = res["differences"][d]
            assert x["classes"] == common
            L.append(f"| {d} | {sd(x['diff'])} | {ci(x['ci95'])} | {src('ben-transfer.json')} |")
    L += ["", "Reading: europe − ben_test = format / label-source shift; india − europe = geography shift; "
          "india − ben_test = total drop. Labels are WorldCover map fractions, not ground truth.", ""]
    return L


def g3_section():
    L = ["## 2. CDVQA G3 change adapter (Kaggle cdvqa-g3-eval V1)", "",
         f"Commit {G3_COMMIT} (`training/cdvqa-g0`), files under `results/cdvqa-g3/`. Controls are scored against the real pair's "
         "ground truth, so a high control score means the answer did not depend on the image pair.", ""]
    for split in ("test1", "test2"):
        path = f"results/cdvqa-g3/{split}-result.json"
        raw = git_bytes(G3_COMMIT, path)
        r = json.loads(raw)
        s = {c: r["conditions"][c]["summary"] for c in CONDS}
        src = f"`{path}` ({sha(raw)[:12]}) @ {G3_COMMIT}"
        L += [f"### 2{'a' if split == 'test1' else 'b'}. {split} (manifest sha256 `{r['eval_manifest_sha256']}`)", "",
              "| Condition | Macro acc | n | Malformed | Source |", "|---|---|---|---|---|",
              f"| majority answer | {f(r['majority_baseline']['macro_accuracy'], 3)} | — | — | {src} |"]
        L += [f"| {c} | {f(s[c]['macro_accuracy'], 3)} | {s[c]['n']} | {s[c]['malformed_n']} | {src} |" for c in CONDS]
        L += ["", "| Paired difference (macro, bootstrap over image pairs) | Value | CI95 | Source |", "|---|---|---|---|"]
        for name, x in r["paired_differences"]["comparisons"].items():
            L.append(f"| {name} | {sd(x['macro']['difference'])} | {ci(x['macro']['ci95'], 3)} | {src} |")
        ident = [x for x in r["conditions"]["adapter_identical"]["records"] if x["question_type"] == "change_or_not"]
        real = {x["sample_id"]: x for x in r["conditions"]["adapter_real"]["records"]}
        swap = [(real[x["sample_id"]], x) for x in r["conditions"]["adapter_swapped"]["records"]]
        dec = [(a, b) for a, b in swap if a["question_type"] == "decrease_or_not"]
        same = sum(a["prediction"] == b["prediction"] for a, b in swap)
        no = sum(x["prediction"] == "no" for x in ident)
        L += ["", "| Control behaviour | Value | Expected if the adapter used the pair | Source |", "|---|---|---|---|",
              f"| identical pair, change_or_not: answers \"no\" | {no}/{len(ident)} = {no / len(ident):.3f} | 1.000 | {src} |",
              f"| swapped pair: answer unchanged, all types | {same}/{len(swap)} = {same / len(swap):.3f} | low | {src} |",
              f"| swapped pair, decrease_or_not: answer flipped | {sum(a['prediction'] != b['prediction'] for a, b in dec)}/{len(dec)} | high | {src} |", ""]
        if split == "test1":
            g1 = (s["base_real"]["macro_accuracy"], r["majority_baseline"]["macro_accuracy"])
            assert g1 == (0.395, 0.44), g1  # the context file's "G1" numbers are these
    L += ["Claim limits (from `results/cdvqa-g3/SOURCE.md`): not allowed — \"uses temporal order\", \"detects change without text "
          "priors\", any comparison with the paper's 73–75 %. `cdvqa_validated` stays false.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer-dir", type=Path, default=Path.home() / "ben-work" / "transfer-run-1")
    a = ap.parse_args()
    L = ["# RESULTS", "", "Generated by `docs/plan/build_results.py`. Do not edit by hand; rerun the script.", ""]
    L += transfer_section(a.transfer_dir) + g3_section()
    L += ["## 3. Recorded earlier, evidence in repo", "", "| As written in SATQUERY_CONTEXT.md | Evidence |", "|---|---|"]
    L += [f"| {q} | {e} |" for q, e in EARLIER]
    OUT.write_text("\n".join(L) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
