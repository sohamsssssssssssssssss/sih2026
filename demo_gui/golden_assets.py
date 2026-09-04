"""Resolve local assets used by the verified Streamlit golden path."""

from pathlib import Path


def local_golden_image(scene_id: str, root: Path) -> Path | None:
    """Return the matching local ladder image when it is available."""
    local_id = scene_id.replace(
        "loveda_LoveDA_images_png_", "loveda_Train_Rural_images_png_"
    )
    gsd = scene_id.rsplit("_gsd", 1)[-1]
    candidate = root / "data" / "ladder" / gsd / f"{local_id}.png"
    return candidate if candidate.is_file() else None
