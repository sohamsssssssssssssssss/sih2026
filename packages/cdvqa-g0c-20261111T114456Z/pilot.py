"""CDVQA two-image pilot entry point. See KAGGLE.md for the commands and their order."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the package inventory equal to package-manifest.json
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.cdvqa_pilot import main  # noqa: E402

if __name__ == "__main__":
    main()
