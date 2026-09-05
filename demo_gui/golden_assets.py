"""Resolve local assets used by the verified Streamlit golden path.

The golden demo scene/question are frozen here so the judge-facing GUI
always shows the same result regardless of how the committed ladder
artifact happens to be ordered.
"""

from pathlib import Path
from typing import Any

GOLDEN_SCENE_ID = "loveda_LoveDA_images_png_0_gsd0.3"
GOLDEN_QUESTION = "Is there a building in this image?"
GOLDEN_GSD = 0.3


def golden_result(report: dict[str, Any]) -> dict[str, Any]:
    """Return the single frozen golden-path row from a ladder artifact.

    Matches on scene ID, question, and GSD together — a scene ID alone is
    not unique, since the ladder artifact asks more than one question per
    tile. Raises a clear error if the pinned result is missing, ambiguous,
    or marked incorrect, rather than silently falling back to a different
    row.
    """
    matches = [
        row
        for row in report["results"]
        if row["tile_id"] == GOLDEN_SCENE_ID
        and row["question"] == GOLDEN_QUESTION
        and float(row["gsd"]) == GOLDEN_GSD
    ]
    if not matches:
        raise ValueError(
            "Pinned golden result not found in the committed ladder artifact: "
            f"tile_id={GOLDEN_SCENE_ID!r}, question={GOLDEN_QUESTION!r}, "
            f"gsd={GOLDEN_GSD}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Pinned golden result is ambiguous: {len(matches)} rows match "
            f"tile_id={GOLDEN_SCENE_ID!r}, question={GOLDEN_QUESTION!r}, "
            f"gsd={GOLDEN_GSD}"
        )
    row = matches[0]
    if row.get("correct") is not True:
        raise ValueError(
            "Pinned golden result exists but is marked incorrect in the "
            f"committed artifact: tile_id={GOLDEN_SCENE_ID!r}, "
            f"question={GOLDEN_QUESTION!r}"
        )
    return row


def local_golden_image(scene_id: str, root: Path) -> Path | None:
    """Return the matching local ladder image when it is available."""
    local_id = scene_id.replace(
        "loveda_LoveDA_images_png_", "loveda_Train_Rural_images_png_"
    )
    gsd = scene_id.rsplit("_gsd", 1)[-1]
    candidate = root / "data" / "ladder" / gsd / f"{local_id}.png"
    return candidate if candidate.is_file() else None
