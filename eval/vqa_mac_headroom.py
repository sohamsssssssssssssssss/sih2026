"""Attempt-2 memory headroom check: B13 soak questions while sampling system available memory."""
import argparse
import json
from pathlib import Path
import random
import re
import resource
import subprocess
import threading
import time
import urllib.request

from eval.vqa_mac_parity import KINDS, make_model, predict

HEADROOM_GIB = 3.0


def available_bytes(text):
    """Return (free + inactive + speculative) pages x page size from vm_stat output."""
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)\.", text)}
    return page * (pages["free"] + pages["inactive"] + pages["speculative"])


def app_up():
    for url in ("http://127.0.0.1:8000/api/health", "http://127.0.0.1:3000/"):
        try:
            urllib.request.urlopen(url, timeout=5).close()
        except OSError:
            return False
    return True


def soak_rows(rows):
    rng = random.Random(17)
    selected = []
    for kind, count in zip(KINDS, (8, 8, 7, 7)):
        group = [r for r in rows if r["question_type"] == kind]
        rng.shuffle(group)
        selected.extend(group[:count])
    rng.shuffle(selected)
    return selected


def main(args):
    import mlx.core as mx

    up = app_up()
    if args.app and not up:
        raise SystemExit("Demo app is not answering on ports 8000 and 3000")
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            samples.append(available_bytes(subprocess.check_output(["vm_stat"], text=True)))
            stop.wait(0.25)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    instruction = json.loads(args.cuda.read_text())["prompt"]["answer_instruction"]
    mx.reset_peak_memory()
    model, processor = make_model(mx, args.base, args.adapter)
    answers = [predict(mx, model, processor, args.images / r["image"], r["question"], instruction)
               for r in soak_rows(rows)]
    stop.set()
    thread.join()
    minimum = min(samples) / 2**30
    result = dict(app_running=up, n=len(answers), samples=len(samples), min_available_gib=minimum,
                  headroom_pass=minimum >= HEADROOM_GIB, peak_mlx_bytes=mx.get_peak_memory(),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--app", action="store_true", help="require the demo app to be running")
    for name in ("base", "adapter", "manifest", "cuda", "images", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
