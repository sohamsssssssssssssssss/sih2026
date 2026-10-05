"""Client-safe rendering of local file paths.

Evidence and execution traces are served to clients, so they must never carry
absolute server paths. Files under the repository root are shown by their
repo-relative POSIX path; anything else is reduced to its bare file name.
"""

import os
import posixpath
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]


def _relative_to(candidate: Path, root: Path) -> str | None:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    return relative.as_posix() if relative.parts else None


def public_path(path: str | os.PathLike[str], root: Path | None = None) -> str:
    """Return ``path`` relative to the repo root, else only its file name.

    Relative inputs are taken as already repo-relative; any that escape upward
    or carry a drive prefix (foreign absolute paths) are reduced to a name.
    """
    # Treat both separators as separators so foreign (e.g. Windows) paths reduce too.
    raw = os.fspath(path).replace("\\", "/")
    name = PurePosixPath(raw).name
    if not raw.startswith("/"):
        normalized = posixpath.normpath(raw)
        first = normalized.split("/", 1)[0]
        if normalized == "." or first == ".." or ":" in first:
            return name
        return normalized
    base = Path(root if root is not None else REPO_ROOT)
    candidate = Path(posixpath.normpath(raw))
    for option, option_root in (
        (candidate, Path(os.path.abspath(base))),
        (candidate.resolve(), base.resolve()),
    ):
        relative = _relative_to(option, option_root)
        if relative is not None:
            return relative
    return name
