from harness.core.types import Trace
from harness.dashboard.summary import summarize_traces


def make_trace(status):
    return Trace(
        run_id=f"run-{status}", goal="g", model_id="m",
        started_at="2026-01-01T00:00:00Z",
        final_status=status,
        total_input_tokens=10, total_output_tokens=20, total_latency_ms=100.0,
    )


def test_summarize_empty():
    summary = summarize_traces([])
    assert summary["total_runs"] == 0


def test_summarize_mixed():
    traces = [
        make_trace("success"),
        make_trace("success"),
        make_trace("error"),
        make_trace("timeout"),
    ]
    summary = summarize_traces(traces)
    assert summary["total_runs"] == 4
    assert summary["success_rate"] == 50.0          # 2 of 4
    assert summary["status_breakdown"]["success"] == 2
    assert summary["status_breakdown"]["error"] == 1
    assert summary["status_breakdown"]["timeout"] == 1
    assert summary["total_input_tokens"] == 40      # 4 × 10
    assert summary["avg_latency_ms"] == 100.0