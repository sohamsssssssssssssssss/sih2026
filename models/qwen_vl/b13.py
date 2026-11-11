"""Frozen B13 identity and inference contract copied from the RGB pilot runner."""

import hashlib
import os
import re
from pathlib import Path

ADAPTER_ENV = "SATQUERY_VQA_ADAPTER"
BASE_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
ANSWER_INSTRUCTION = "Answer with a single word or number only. No explanation."
ADAPTER_HASHES = {
    "adapter_config.json": "716251f9cfbb73ed5fb408e41f2eb2b30281b8f98fb7c6a92a1dd0b9204e3644",
    "adapter_model.safetensors": "7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a",
}


def adapter_path():
    value = os.environ.get(ADAPTER_ENV)
    return Path(value).expanduser().resolve() if value else None


def adapter_identity(path):
    hashes = {}
    for name, expected in ADAPTER_HASHES.items():
        with (path / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        if hashes[name] != expected:
            raise ValueError(f"B13 adapter SHA256 mismatch: {name}")
    return {"name": "B13", "base_revision": BASE_REVISION, "files_sha256": hashes}


def messages(image_path, question):
    # Source: training/rgb-pilot-b12-b13:training/remote_sensing.py::messages.
    return [{"role": "user", "content": [
        {"type": "image", "image": Path(image_path).resolve().as_uri(),
         "resized_height": 392, "resized_width": 392},
        {"type": "text", "text": f"{question} {ANSWER_INSTRUCTION}"},
    ]}]


def count_target(question):
    # shortcut: English count templates only; extend with labeled routing examples.
    query = question.strip().rstrip("?.!")
    match = re.match(r"^(?:how many |what is the (?:number|amount) of |count (?:the )?)(.+)$", query, re.I)
    if match is None:
        return None
    return re.sub(r"\s+(?:are|is)\s+.*$", "", match[1], flags=re.I).strip()
