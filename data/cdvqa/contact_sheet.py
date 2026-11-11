"""Visual sanity sheet: 8 random eval pairs, before | after, with question and answer underneath.

Run: python3.11 data/cdvqa/contact_sheet.py -> results/cdvqa-g0/sanity_pairs.png
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
IMAGE_ROOT = Path.home() / "data/cdvqa/images/SECOND_train_set"
ROWS = REPO / "data/manifests/cdvqa/eval-test1-400.v1.jsonl"
OUT = REPO / "results/cdvqa-g0/sanity_pairs.png"
SIDE, CAPTION, GAP = 256, 46, 12


def main():
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    chosen = random.Random(17).sample(rows, 8)
    sheet = Image.new("RGB", (2 * SIDE + 3 * GAP, 8 * (SIDE + CAPTION + GAP) + GAP), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(chosen):
        top = GAP + i * (SIDE + CAPTION + GAP)
        for j, rel in enumerate((row["before_image"], row["after_image"])):
            with Image.open(IMAGE_ROOT / rel) as im:
                sheet.paste(im.convert("RGB").resize((SIDE, SIDE)), (GAP + j * (SIDE + GAP), top))
        draw.text((GAP, top + SIDE + 4), f"BEFORE {row['before_image']}   |   AFTER {row['after_image']}", fill=(0, 0, 0))
        draw.text((GAP, top + SIDE + 18), f"Q: {row['question']}", fill=(0, 0, 0))
        draw.text((GAP, top + SIDE + 32), f"A: {row['answer']}   [{row['question_type']}, pair {row['pair_id']}]", fill=(60, 60, 60))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(OUT, [r["id"] for r in chosen])


if __name__ == "__main__":
    main()
