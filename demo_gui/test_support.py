"""Small compatibility helpers for Streamlit's version-dependent test tree."""


def image_elements(block):
    """Use the singular image key, retaining Streamlit 1.52's legacy elements."""
    images = block.get("image")
    if images:
        return images
    return [element for element in block if element.type == "imgs"]
