"""Extract only explicitly listed S2 archive members from a gzip tar on stdin."""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path


def extract(stream, members: set[str], output: Path) -> int:
    found = set()
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if member.name not in members:
                continue
            if member.name in found or not member.isfile():
                raise ValueError(f"Duplicate or non-file selected member: {member.name}")
            found.add(member.name)
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if len(found) % 6000 == 0:
                print(f"extracted {len(found)}/{len(members)}", file=sys.stderr, flush=True)
            if len(found) == len(members):
                break
    missing = members - found
    if missing:
        raise ValueError(f"Selected archive members missing: {len(missing)}; first: {min(missing)}")
    return len(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    members = set(args.members.read_text().splitlines())
    if not members or any(not item.startswith("BigEarthNet-S2/") or not item.endswith(".tif") or ".." in Path(item).parts for item in members):
        raise ValueError("Member list must contain only S2 TIFF paths")
    print(f"extracted {extract(sys.stdin.buffer, members, args.output)} selected TIFFs")


if __name__ == "__main__":
    main()
