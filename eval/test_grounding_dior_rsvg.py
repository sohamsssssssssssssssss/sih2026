import xml.etree.ElementTree as ET
from pathlib import Path

from eval.suites.grounding_dior_rsvg import (
    iou_xyxy,
    load_test_records,
    normalize_xyxy,
)


def test_official_index_loader_and_coordinate_metrics(tmp_path: Path) -> None:
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "JPEGImages").mkdir()
    (tmp_path / "test.txt").write_text("1\n", encoding="utf-8")
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = "scene.jpg"
    for category, phrase, coordinates in (
        ("airplane", "ignored train expression", (1, 1, 2, 2)),
        ("bridge", "the bridge", (20, 10, 60, 30)),
    ):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = category
        ET.SubElement(obj, "pose").text = "unspecified"
        box = ET.SubElement(obj, "bndbox")
        for name, value in zip(("xmin", "ymin", "xmax", "ymax"), coordinates):
            ET.SubElement(box, name).text = str(value)
        ET.SubElement(obj, "description").text = phrase
    ET.ElementTree(root).write(tmp_path / "Annotations" / "00001.xml")

    _, records = load_test_records(tmp_path, expected_split_size=1)

    assert records[0]["expression"] == "the bridge"
    assert records[0]["category"] == "bridge"
    normalized = normalize_xyxy(records[0]["ground_truth_pixel_xyxy"], 100, 50)
    assert normalized == [0.2, 0.2, 0.6, 0.6]
    assert iou_xyxy(normalized, normalized) == 1.0
    assert iou_xyxy(normalized, [0.7, 0.7, 0.9, 0.9]) == 0.0

    (tmp_path / "val.txt").write_text("0\n", encoding="utf-8")
    _, validation = load_test_records(
        tmp_path, expected_split_size=1, split_file="val.txt"
    )
    assert [row["category"] for row in validation] == ["airplane"]
