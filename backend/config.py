"""T09 runtime settings: environment overrides and defaults in one place."""
import math
import os
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AskConfig:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('SATQUERY_DATA_DIR', ROOT/'data/runtime/ask')).expanduser())
    trace_dir: Path = field(default_factory=lambda: Path(os.getenv('SATQUERY_TRACE_DIR', ROOT/'data/runtime/trace')).expanduser())
    thresholds_file: Path = field(default_factory=lambda: Path(os.getenv('SATQUERY_THRESHOLDS_FILE', ROOT/'configs/answerability.json')).expanduser())
    timeout: float = field(default_factory=lambda: float(os.getenv('SATQUERY_TIMEOUT', '60')))
    worldcover_dir: Path = field(default_factory=lambda: Path(os.getenv('SATQUERY_WORLDCOVER_DIR', Path.home()/'ben-work/wc/pack')).expanduser())
    flood_dir: Path = field(default_factory=lambda: Path(os.getenv('SATQUERY_FLOOD_DIR', Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1')).expanduser())
    upload_limit: int = 200 * 1024 * 1024
    decoded_limit: int = 512 * 1024 * 1024

    def __post_init__(self):
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('SATQUERY_TIMEOUT must be finite and positive')
