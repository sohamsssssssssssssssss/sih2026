import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"(?<![\w.])(?:\d+\.\d+%?|\d+%)(?!\w|\.\d)")


@pytest.mark.parametrize("name", ["docs/judge/JUDGE-QA.md", "docs/STATUS.md"])
def test_numeric_claims_are_indexed(name):
    index = (ROOT / "docs/evidence/INDEX.md").read_text()
    indexed = set()
    for line in index.splitlines():
        if line.startswith("|"):
            _, claim, number, *_ = line.split("|")
            indexed.update(NUMBER.findall(claim + " " + number))
    document = (ROOT / name).read_text()
    document = re.sub(r"\]\([^\n)]*\)", "]", document)
    missing = set(NUMBER.findall(document)) - indexed
    assert not missing, f"{name} has unindexed numbers: {sorted(missing)}"
