from harness.core.types import Trace, TraceStep
from harness.eval.replay import ReplayModel, load_trace
from harness.core.loop import run_agent, RunConfig


def make_trace():
    """A hand-built trace with one recorded model-call step."""
    return Trace(
        run_id="test-run",
        goal="test goal",
        model_id="ollama:qwen3:8b",
        started_at="2026-01-01T00:00:00Z",
        steps=[
            TraceStep(step_number=1, kind="model_call",
                      text="the recorded answer",
                      input_tokens=10, output_tokens=5, latency_ms=50.0),
        ],
    )


def test_replay_returns_recorded_response():
    trace = make_trace()
    model = ReplayModel(trace)
    # Running the agent with the replay model should produce the recorded text.
    result = run_agent("test goal", model, RunConfig())
    assert result.status.value == "success"
    assert result.answer == "the recorded answer"       # exact recorded text
    assert result.total_output_tokens == 5              # recorded metrics


def test_replay_is_deterministic():
    trace = make_trace()
    # Two separate replays of the same trace give identical results.
    r1 = run_agent("test goal", ReplayModel(trace), RunConfig())
    r2 = run_agent("test goal", ReplayModel(trace), RunConfig())
    assert r1.answer == r2.answer                       # same every time


def test_replay_exhaustion_raises():
    trace = make_trace()   # only 1 recorded step
    model = ReplayModel(trace)
    model.complete([])     # consume the one step
    try:
        model.complete([])  # asking for a 2nd -> should raise
        assert False, "should have raised on exhausted replay"
    except RuntimeError:
        pass