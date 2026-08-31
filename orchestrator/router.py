"""Route requests through the public Model interface only."""

from datetime import datetime, timezone
from typing import Any

from orchestrator.registry import get
from orchestrator.trace import append_record


def route(
    model_name: str,
    image_paths: list[str],
    question: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = get(model_name)
    result = model.infer(image_paths=image_paths, question=question)
    trace_record = append_record(
        {
            "model_name": model_name,
            "model_version": getattr(model, "version", "unknown"),
            "params": params or {},
            "input_summary": {
                "image_paths": image_paths,
                "question": question,
                "n_images": len(image_paths),
            },
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {**result, "trace": trace_record}
