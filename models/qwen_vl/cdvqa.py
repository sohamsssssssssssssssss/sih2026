"""CDVQA change-VQA contract: config, question typing, prompt (from the frozen package) and answer checks."""

import hashlib
import importlib.util
import json
import os
import re
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "change_vqa.json"
FLAG_ENV = "SATQUERY_CHANGE_VQA"
ADAPTER_ENV = "SATQUERY_CDVQA_ADAPTER"
IMAGE_SIZE = 392  # package pilot-config image_size
CAVEAT = ("Trained on SECOND aerial 0.5-3 m RGB pairs; not validated on Cartosat or RISAT imagery. "
          "Accuracy is unmeasured in this app until the G3 evaluation result is committed.")


def config():
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if os.environ.get(FLAG_ENV) is not None:
        value["enabled"] = os.environ[FLAG_ENV] == "1"
    if os.environ.get(ADAPTER_ENV):
        value["adapter_path"] = os.environ[ADAPTER_ENV]
    return value


def enabled():
    return config()["enabled"] is True


def adapter_path():
    value = config().get("adapter_path")
    return Path(value).expanduser().resolve() if value else None


def _load(relative, name):
    path = ROOT / config()["package"] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@cache
def package():
    """The frozen package's remote_sensing module, loaded by file path (unique name: no clash with training/)."""
    return _load("training/remote_sensing.py", "cdvqa_package_remote_sensing")


@cache
def provenance():
    return json.loads((ROOT / config()["package"] / "data/manifests/cdvqa/provenance.json").read_text())


def choices(question_type):
    """Allowed answers exactly as the eval prompt shows them (original case)."""
    return tuple(provenance()["answer_choices"][question_type])


def normalise(text):
    # Same rule as the package's eval/cdvqa_pairs.normalise (strip + casefold).
    return text.strip().casefold()


def validate(raw, question_type):
    vocabulary = provenance()["answer_vocabulary"][question_type]
    return "ok" if normalise(raw) in vocabulary else "malformed"


def messages(before, after, question, question_type):
    """Eval prompt [before, after, question] built by the package's own messages()."""
    module = package()
    example = module.TrainingExample(
        image_path=Path(before).resolve(), instruction=question, response="", dataset="cdvqa",
        source="app", split="test", sample_id="app", after_image_path=Path(after).resolve(),
        choices=choices(question_type))
    return module.messages(example, IMAGE_SIZE, include_target=False)


_CLASS = r"(?:water|buildings|low[ _]vegetation|trees|playgrounds|nvg[ _]surface|non-vegetated ground surface)"
_AREA = rf"(?:the )?(?:areas|regions) of {_CLASS}"
_STAGE = r"(?: in the (?:first|second|pre-change|post-change|pre-event|post-event) image)?"
# shortcut: matches the CDVQA English templates only; paraphrases get "unsupported". Widen with labelled examples.
_TYPES = (
    ("change_to_what", rf"what have {_AREA}{_STAGE} mainly changed to"),
    ("smallest_change", rf"what (?:type of change is the smallest|is the smallest change){_STAGE}"),
    ("largest_change", rf"what (?:type of change is the largest|is the largest change){_STAGE}"),
    ("change_ratio_types", rf"(?:what is the change (?:ratio|percentage|proportion) of {_CLASS}"
                           rf"|how much area of {_CLASS} has changed){_STAGE}"),
    ("change_ratio", r"what is the (?:non-change ratio|change ratio) of the imagery"
                     r"|how much of the area has (?:not )?changed"
                     r"|what is the percentage of (?:changed|unchanged|non-change) (?:areas|regions)"),
    ("increase_or_not", rf"(?:did {_AREA} increase|have {_AREA} increased)"),
    ("decrease_or_not", rf"(?:did {_AREA} decrease|have {_AREA} decreased)"),
    ("change_or_not", rf"(?:did {_AREA} change|have {_AREA} changed){_STAGE}"),
)


def question_type(question):
    text = " ".join(question.strip().rstrip("?.!").split())
    for name, pattern in _TYPES:
        if re.fullmatch(pattern, text, re.I):
            return name
    return None


def adapter_identity(path):
    hashes = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        with (Path(path) / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"id": config()["adapter_id"], "files_sha256": hashes,
            "adapter_model_sha256": hashes["adapter_model.safetensors"]}


def pair_refusal(image_paths, gsds=(None, None)):
    """Reason to refuse the pair, or None. Sizes come from the image headers."""
    if len(image_paths) != 2:
        return f"Change questions need exactly two images (before, after); got {len(image_paths)}."
    from PIL import Image
    sizes = []
    for path in image_paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    limit = config()["max_ratio"]
    for axis in (0, 1):
        small, large = sorted((sizes[0][axis], sizes[1][axis]))
        if large > limit * small:
            return f"Image sizes differ by more than {limit:g}x ({sizes[0]} vs {sizes[1]})."
    if None not in gsds:
        small, large = sorted(gsds)
        if large > limit * small:
            return f"Ground sample distances differ by more than {limit:g}x ({gsds[0]:g} vs {gsds[1]:g} m/px)."
    return None


def _shared_runner(before, after, question, question_type):
    from orchestrator.registry import get
    return get("qwen2.5vl-3b").infer_pair(before, after, question, question_type)


from models.base import Model, ModelReadiness  # noqa: E402


class CdvqaChangeModel(Model):
    """Free-form change questions on a before/after optical pair, via the CDVQA adapter on the shared base."""

    name = "qwen2.5vl-3b-cdvqa"
    version = "Qwen/Qwen2.5-VL-3B-Instruct+cdvqa"
    isolated = True

    def __init__(self, runner=None):
        # Injectable so CPU tests use a stub; default shares the registry's Qwen instance (one base in memory).
        self.runner = runner or _shared_runner

    def readiness(self):
        if not enabled():
            return ModelReadiness(False, "FEATURE_DISABLED", f"Change VQA is off; set {FLAG_ENV}=1.")
        path = adapter_path()
        if path is None:
            return ModelReadiness(False, "ADAPTER_MISSING", f"Set {ADAPTER_ENV} to the CDVQA adapter directory.")
        try:
            adapter_identity(path)
        except OSError as exc:
            return ModelReadiness(False, "ADAPTER_INVALID", str(exc))
        if self.runner is _shared_runner:
            from orchestrator.registry import get
            return get("qwen2.5vl-3b").readiness()
        return ModelReadiness(True)

    def infer(self, image_paths, question):
        qtype = question_type(question)
        if qtype is None:
            raise ValueError("unsupported_change_question")
        if len(image_paths) != 2:
            raise ValueError("Change VQA needs exactly two images")
        identity = adapter_identity(adapter_path())
        raw = self.runner(image_paths[0], image_paths[1], question, qtype).strip()
        status = validate(raw, qtype)
        return {
            "answer": raw if status == "ok" else f"malformed: {raw or '(empty)'}",
            "answer_source": "cdvqa_adapter",
            "confidence": None,
            "evidence": [],
            "change_vqa": {
                "answer_status": status,
                "question_type": qtype,
                "allowed_answers": list(choices(qtype)),
                "adapter": identity,
                "validation_status": "validated" if config().get("cdvqa_validated") is True else "experimental_unvalidated",
                "caveat": CAVEAT,
            },
        }


RULE = "cdvqa_change_vqa"
UNSUPPORTED_RULE = "cdvqa_unsupported"
GATE = "change-vqa-gate"
UNSUPPORTED_MESSAGE = ("unsupported_change_question: the change adapter only answers CDVQA-style questions "
                       "(change or not, increase/decrease, change to what, largest/smallest change, change ratio, "
                       "class change ratio for water, buildings, low vegetation, trees, playgrounds, NVG surface).")
