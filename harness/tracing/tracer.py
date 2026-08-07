import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from harness.core.types import Trace, TraceStep


class Tracer:
    """Records the steps of a run into a Trace and persists it to disk."""

    def __init__(self, goal: str, model_id: str, runs_dir: str = "runs"):
        self._trace = Trace(
            run_id=str(uuid.uuid4()),          # unique id for this run
            goal=goal,
            model_id=model_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._runs_dir = Path(runs_dir)

    def record_step(self, step: TraceStep) -> None:
        # append the step to the trace's steps list.
        # self._trace.steps is a list; add `step` to it.
        self._trace.steps.append(step)

    def finalize(self, status: str, input_tokens: int, output_tokens: int, latency_ms: float) -> str:
        # Record the run's final status and totals.
        self._trace.final_status = status
        self._trace.total_input_tokens = input_tokens
        self._trace.total_output_tokens = output_tokens
        self._trace.total_latency_ms = latency_ms
        return self._save()

    def _save(self) -> str:
        # Make sure the runs directory exists.
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        path = self._runs_dir / f"{self._trace.run_id}.json"

        # write the trace to `path` as JSON.
        # Pydantic models have .model_dump_json() which produces a JSON string.
        with open(path, "w") as f:
            f.write(self._trace.model_dump_json(indent=2))
        return str(path)

    @property
    def run_id(self) -> str:
        return self._trace.run_id