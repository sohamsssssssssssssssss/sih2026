"""50-step smoke test. NOTHING launches on the RTX Pro 6000 without passing this.

A shape mismatch that crashes at step 3, repeated across a week, costs 30 of
your 150 allocated hours. This runs in ~3 minutes on a Kaggle T4.

Usage:
    python scripts/smoke.py --config s0_baseline_qwen3b
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from sih.config import load_config  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    print(f"\n=== SMOKE TEST: {args.config} ===\n")

    # --- 1. config ---
    try:
        cfg = load_config(args.config)
        check("config loads", True, cfg["name"])
    except Exception as e:
        check("config loads", False, str(e))
        return 1

    # --- 2. hardware ---
    if not check("CUDA available", torch.cuda.is_available()):
        return 1

    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    check("GPU detected", True, f"{dev} sm_{cap[0]}{cap[1]} {total_gb:.1f}GB")

    # Precision sanity: bf16 needs Ampere (sm_80+). T4 is sm_75.
    want = cfg["hardware"]["precision"]
    bf16_ok = torch.cuda.is_bf16_supported()
    if want == "bf16" and not bf16_ok:
        check("precision valid for this GPU", False,
              "config asks bf16 but GPU lacks support — use fp16")
        return 1
    check("precision valid for this GPU", True, want)

    # FlashAttention-2 needs Ampere+ too.
    attn = cfg["hardware"]["attn_implementation"]
    if attn == "flash_attention_2" and cap[0] < 8:
        check("attn impl valid", False, "flash_attention_2 needs sm_80+; use sdpa")
        return 1
    check("attn impl valid", True, attn)

    # --- 3. model ---
    t0 = time.time()
    try:
        from sih.models.registry import build_model
        model, processor = build_model(cfg)
        check("model loads", True, f"{time.time()-t0:.0f}s")
    except Exception as e:
        check("model loads", False, f"{type(e).__name__}: {e}")
        return 1

    alloc = torch.cuda.memory_allocated() / 1e9
    check("weights fit", alloc < total_gb * 0.9, f"{alloc:.1f}GB / {total_gb:.1f}GB")

    # --- 4. data ---
    try:
        from sih.data.loader import build_loader
        loader = build_loader(cfg, split="train", limit=args.steps * 2)
        batch = next(iter(loader))
        shapes = {k: tuple(v.shape) for k, v in batch.items()
                  if hasattr(v, "shape")}
        check("dataloader yields batch", True, str(shapes))
    except NotImplementedError:
        check("dataloader yields batch", False,
              "NOT IMPLEMENTED — B1 has not shipped the loader yet")
        print("\n  (expected before 25 Sept; stop here for Stage 0 eval-only runs)\n")
        return 0
    except Exception as e:
        check("dataloader yields batch", False, f"{type(e).__name__}: {e}")
        return 1

    # --- 5. train loop ---
    if not cfg.get("lora", {}).get("enabled", False):
        print("\n  (no LoRA — eval-only config, skipping train steps)\n")
        return 0

    try:
        model.train()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["train"]["lr"],
        )
        scaler = torch.amp.GradScaler("cuda", enabled=want == "fp16")

        losses = []
        for i, batch in enumerate(loader):
            if i >= args.steps:
                break
            batch = {k: v.to(model.device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16 if want == "fp16"
                                else torch.bfloat16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg["train"]["max_grad_norm"],
            )
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())

        check("forward+backward runs", True, f"{len(losses)} steps")
        finite = all(l == l and abs(l) != float("inf") for l in losses)
        check("loss finite", finite, f"first={losses[0]:.4f} last={losses[-1]:.4f}")
        if not finite:
            return 1

        # Not a hard gate — 50 steps is too few to guarantee descent — but a
        # loss that has INCREASED over 50 steps usually means LR is too high.
        trend = sum(losses[-10:]) / 10 - sum(losses[:10]) / 10
        check("loss trending down", trend < 0.0,
              f"delta={trend:+.4f}" + ("" if trend < 0 else "  <-- check LR"))

        peak = torch.cuda.max_memory_allocated() / 1e9
        check("peak VRAM within budget", peak < total_gb * 0.95,
              f"{peak:.1f}GB / {total_gb:.1f}GB")

    except torch.cuda.OutOfMemoryError:
        check("forward+backward runs", False,
              "OOM — lower per_device_batch_size or image_size")
        return 1
    except Exception as e:
        check("forward+backward runs", False, f"{type(e).__name__}: {e}")
        return 1

    # --- 6. checkpoint round-trip ---
    try:
        out = Path("results/_smoke_ckpt")
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        check("checkpoint saves", any(out.iterdir()))
    except Exception as e:
        check("checkpoint saves", False, f"{type(e).__name__}: {e}")
        return 1

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n=== {'ALL PASSED' if not failed else f'{len(failed)} FAILED'} ===")
    print("Cleared for GPU allocation.\n" if not failed else "DO NOT run on the 6000 Pro.\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
