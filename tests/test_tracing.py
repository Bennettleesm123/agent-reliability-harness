import json
from pathlib import Path
from harness.core.types import Trace
from harness.tracing.tracer import Tracer, TraceStep


def test_trace_saves_and_loads(tmp_path):
    # tmp_path is a pytest fixture — a temporary directory, auto-cleaned.
    tracer = Tracer(goal="test goal", model_id="test:model", runs_dir=str(tmp_path))
    tracer.record_step(TraceStep(step_number=1, kind="model_call",
                                 input_tokens=10, output_tokens=20, latency_ms=100.0))
    path = tracer.finalize("success", input_tokens=10, output_tokens=20, latency_ms=100.0)

    # The file should exist.
    assert Path(path).exists()

    # Load it back and check the structure survived the round-trip.
    with open(path) as f:
        data = json.load(f)
    loaded = Trace(**data)   # re-parse into a Trace — proves it's valid

    assert loaded.goal == "test goal"
    assert loaded.model_id == "test:model"
    assert loaded.final_status == "success"
    assert len(loaded.steps) == 1
    assert loaded.steps[0].input_tokens == 10
    assert loaded.total_output_tokens == 20