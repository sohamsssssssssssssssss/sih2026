"""One canonical JSON form for every hash in the repository.

``json.dumps(value, sort_keys=True)`` is frozen on purpose: it is exactly the
serialization the hash-chained trace and ``Scene.scene_id`` used before T07, so
changing separators, escaping, or NaN handling would change existing digests.
"""

import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize `value` deterministically with sorted keys (stable across runs)."""
    return json.dumps(value, sort_keys=True)
