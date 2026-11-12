"""Conservative count-resolution policy; physical sizes are configurable assumptions."""
import json
import math
import re
from pathlib import Path
from models.grounding_dino.count import CONFIG


def metadata_gsd(value):
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:m/px)?\s*", value)
        value = float(match[1]) if match else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def filename_gsd(value):
    match = re.search(r"(?:^|[_-])gsd(\d+(?:\.\d+)?)(?=$|[_ .-])", value, re.I)
    return metadata_gsd(match[1]) if match else None


def resolution_gate(noun, gsd_m, source="unknown", config: Path = CONFIG):
    policy = json.loads(config.read_text())["resolution"]
    pixels, sizes = policy["min_pixels"], policy["object_size_m"]
    if not sizes or "default" not in sizes or any(isinstance(v, bool) or not isinstance(v, (int, float)) or metadata_gsd(v) is None for v in [pixels, *sizes.values()]):
        raise ValueError("Resolution sizes and min_pixels must be positive finite numbers")
    phrase = re.sub(r"[_-]+", " ", noun.lower())
    aliases = {"car": "vehicle", "truck": "vehicle", "plane": "airplane", "aircraft": "airplane", "storagetank": "storage tank"}
    names = {**{name: name for name in sizes if name != "default"}, **aliases}
    category = next((names[name] for name in sorted(names, key=len, reverse=True)
                     if re.search(r"\b" + re.escape(name) + r"s?\b", phrase)), "default")
    size = sizes.get(category, sizes["default"])
    required = size / pixels
    if gsd_m is not None and (not isinstance(gsd_m, (int, float)) or metadata_gsd(gsd_m) is None):
        raise ValueError("GSD must be positive finite metres per pixel or unknown")
    status = "gsd_unknown" if gsd_m is None else "not_resolvable" if size / gsd_m < pixels else "resolvable"
    return {"count_status": status, "gsd_m": gsd_m, "gsd_source": source,
            "object_class": category, "object_size_m": size, "required_gsd_m": required,
            "min_pixels": pixels, **({"unverified_gsd": True} if gsd_m is None else {})}


def refusal(gate):
    return {**gate, "answer": (
        f"The {gate['object_class']} count cannot be determined at {gate['gsd_m']:g} m/px. "
        f"Imagery at {gate['required_gsd_m']:g} m/px or finer is required by the resolution policy."),
        "answer_source": "resolution_gate", "evidence": [{"type": "count_resolution", **gate}]}
