"""The eval CLI. OWNER: Lead.

No number enters a deck, a report, or a conversation unless it came out of
this script. A result from someone's personal notebook does not exist.

Usage:
    python scripts/eval.py --config s0_baseline_qwen3b
    python scripts/eval.py --config s0_baseline_qwen3b --suite rsvqa --limit 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from sih.config import load_config, dump_config  # noqa: E402
from sih.eval.metrics import normalise, summarise, degenerate  # noqa: E402

SYSTEM = (
    "You are analysing a satellite remote-sensing image. "
    "Answer with the shortest possible span: a single word, a number, or yes/no. "
    "Do not explain. Do not add punctuation."
)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


@torch.inference_mode()
def run_suite(model, processor, samples, max_new_tokens=16):
    from qwen_vl_utils import process_vision_info

    records, t0 = [], time.time()
    for i, s in enumerate(samples):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": s["image"]},
                {"type": "text", "text": s["question"]},
            ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, videos=vids,
                           padding=True, return_tensors="pt").to(model.device)

        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        records.append({
            "question": s["question"],
            "gold": normalise(s["answer"]),
            "pred_raw": raw,
            "pred": normalise(raw),
            "correct": normalise(raw) == normalise(s["answer"]),
        })

        if (i + 1) % 50 == 0:
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"  {i+1}/{len(samples)}  acc={acc:.3f}  "
                  f"{(time.time()-t0)/(i+1):.2f}s/sample")
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    suite = args.suite or cfg["eval"]["suites"][0]
    limit = args.limit or cfg["eval"]["limit"]

    print(f"\n=== EVAL: {cfg['name']} / {suite} / limit={limit} ===\n")

    from sih.models.registry import build_model
    model, processor = build_model(cfg)

    from sih.data.suites import load_suite
    samples = load_suite(suite, limit=limit)
    print(f"[data] {len(samples)} samples")
    print(f"[data] example: {samples[0]['question']!r} -> {samples[0]['answer']!r}\n")

    records = run_suite(model, processor, samples,
                        max_new_tokens=cfg["eval"]["max_new_tokens"])
    summary = summarise(records)

    print("\n=== RESULT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    warn = degenerate(summary)
    if warn:
        print(f"\n  WARNING: {warn}\n")

    out = Path(args.out or f"results/{cfg['name']}__{suite}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config_name": cfg["name"],
        "config": dump_config(cfg),
        "suite": suite,
        "git_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "summary": summary,
        "warning": warn,
        "records": records,
    }, indent=2))
    print(f"\n[saved] {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
