"""Official RSVQA-LR test-split acquisition and loading."""

import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data" / "raw" / "rsvqa_lr"
ZENODO_RECORD = "6344334"
FILES = {
    "LR_split_train_images.json": (585112, "7d1e7c099c65e39b3e773578cdabe79a"),
    "LR_split_train_questions.json": (11940731, "935d59a05d126496fe61c541b4ab2d55"),
    "LR_split_train_answers.json": (7428162, "a5ff787f9977b0050b9bbf4e32bbb533"),
    "LR_split_val_images.json": (123258, "7a9f267d2cd106025c45a2b68dce5351"),
    "LR_split_val_questions.json": (3483748, "67b99979ebd468330355d656bf4d6d29"),
    "LR_split_val_answers.json": (2690962, "61ba49ece26c989f81a9a1e2fe0d475b"),
    "LR_split_test_images.json": (123273, "4a5ae90a5686bbcffd1d7ec06ddbb692"),
    "LR_split_test_questions.json": (2717368, "9bddc53d7399a43378f743ec0ff1f95f"),
    "LR_split_test_answers.json": (1922393, "f925d70eb74bb4094966670cb4c2f840"),
    "all_questions.json": (15270117, "fed409776edd11790c596ea0848984c7"),
    "all_answers.json": (9169791, "9042f398d25413a10ea530b7b2a94dd2"),
    "Images_LR.zip": (95008155, "2329258d74d54600628b8652a0e42672"),
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream integrity checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(filename: str, destination: Path) -> None:
    expected_size, expected_md5 = FILES[filename]
    if destination.is_file():
        if destination.stat().st_size == expected_size and _md5(destination) == expected_md5:
            print(f"Already downloaded: {destination}")
            return
        raise RuntimeError(f"Existing file failed integrity validation: {destination}")
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{filename}/content"
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "sih26167-rsvqa/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    except (OSError, urllib.error.URLError) as exc:
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError(f"Download failed and curl is unavailable: {exc}") from exc
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
    if partial.stat().st_size != expected_size or _md5(partial) != expected_md5:
        raise RuntimeError(f"Downloaded file failed integrity validation: {filename}")
    partial.replace(destination)


def _extract_images(root: Path) -> None:
    archive = root / "Images_LR.zip"
    marker = root / ".Images_LR.extracted"
    if marker.exists() and len(list(root.rglob("*.tif"))) >= 772:
        print(f"Already extracted: {archive.name}")
        return
    resolved_root = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (root / member.filename).resolve().is_relative_to(resolved_root):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(root)
    marker.write_text("ok\n", encoding="utf-8")


def download_rsvqa_lr(root: Path = DEFAULT_ROOT) -> Path:
    """Download the complete official release and extract its shared image archive."""
    root.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        _download(filename, root / filename)
    _extract_images(root)
    return root


def load_rsvqa_lr(
    limit: int = 200,
    full: bool = False,
    root: Path = DEFAULT_ROOT,
) -> list[dict]:
    """Return official test samples in the common evaluation contract."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    download_rsvqa_lr(root)
    questions = json.loads((root / "LR_split_test_questions.json").read_text())["questions"]
    answers = json.loads((root / "LR_split_test_answers.json").read_text())["answers"]
    images = json.loads((root / "LR_split_test_images.json").read_text())["images"]
    answer_by_question = {
        int(answer["question_id"]): str(answer["answer"])
        for answer in answers
        if answer.get("active")
    }
    image_by_id = {int(image["id"]): image for image in images if image.get("active")}
    image_path_by_name = {path.name: path.resolve() for path in root.rglob("*.tif")}
    active_questions = [
        question
        for question in questions
        if question.get("active")
        and int(question["id"]) in answer_by_question
        and int(question["img_id"]) in image_by_id
    ]
    if full or limit >= len(active_questions):
        selected = active_questions
    else:
        # Deterministically cover the whole official test split instead of taking
        # 200 adjacent questions from only the first two images.
        selected = [
            active_questions[(index * len(active_questions)) // limit]
            for index in range(limit)
        ]
    samples = []
    for question in selected:
        image = image_by_id[int(question["img_id"])]
        # The archive stores converted images by database ID (for example 232.tif),
        # while original_name records the much longer Sentinel-2 source filename.
        image_name = f"{int(image['id'])}.tif"
        try:
            image_path = image_path_by_name[image_name]
        except KeyError as exc:
            raise FileNotFoundError(f"RSVQA image is missing: {image_name}") from exc
        samples.append(
            {
                "image_paths": [str(image_path)],
                "question": str(question["question"]),
                "expected_answer": answer_by_question[int(question["id"])],
            }
        )
    return samples
