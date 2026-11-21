"""Load 8-bit B13 once and serve requests over the inherited private socket."""
import json
from pathlib import Path
import socket
import sys

from eval.vqa_mac_fresh_test import CONVERSION_SHA, INSTRUCTION
from eval.vqa_mac_parity import ADAPTER_SHA, make_model, predict, sha


def main(channel):
    import mlx.core as mx

    models = Path.home() / "models"
    base = models / "qwen2.5-vl-3b-mlx-8bit"
    adapter = models / "satquery-b13-adapter/adapter_model.safetensors"
    manifest = base / "MANIFEST"
    if sha(manifest) != CONVERSION_SHA or sha(adapter) != ADAPTER_SHA:
        raise ValueError("MLX conversion or B13 adapter hash mismatch")
    for line in manifest.read_text().split("SHA-256 (conversion payload):\n", 1)[1].splitlines():
        digest, filename = line.split(maxsplit=1)
        if sha(base / filename) != digest:
            raise ValueError(f"MLX conversion member hash mismatch: {filename}")
    model, processor = make_model(mx, base, adapter)
    warmup = Path("data/demo/resolution/loveda_LoveDA_images_png_0_gsd10.0.png").resolve()
    predict(mx, model, processor, warmup, "Is there a road?", INSTRUCTION)

    def send(payload):
        channel.sendall(json.dumps(payload).encode() + b"\n")

    send({"ready": True})
    with channel.makefile("rb") as requests:
        for line in requests:
            request = json.loads(line)
            answer = predict(mx, model, processor, Path(request["path"]), request["question"], INSTRUCTION)
            send({"answer": answer})


if __name__ == "__main__":
    with socket.socket(fileno=int(sys.argv[1])) as socket_channel:
        main(socket_channel)
