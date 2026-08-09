import json
from pathlib import Path
from harness.core.types import Trace


def load_all_traces(runs_dir: str = "runs") -> list[Trace]:
    """Load every saved trace from the runs directory."""
    traces = []
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return traces
    for file in runs_path.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
        # turn the loaded dict into a Trace and add it to the list.
        # Trace(**data), then append to traces.
        traces.append(Trace(**data))
    return traces


def summarize_traces(traces: list[Trace]) -> dict:
    """Compute aggregate metrics across all traces."""
    total = len(traces)
    if total == 0:
        return {"total_runs": 0}

    # Count runs by their final status.
    status_counts = {}
    for t in traces:
        status = t.final_status or "unknown"
        # increment the count for this status in status_counts.
        # use .get(status, 0) + 1 so missing keys start at 0.
        status_counts[status] = status_counts.get(status,0)+1

    # Aggregate token and latency totals.
    total_input = sum(t.total_input_tokens for t in traces)
    total_output = sum(t.total_output_tokens for t in traces)
    total_latency = sum(t.total_latency_ms for t in traces)

    # Success rate.
    successes = status_counts.get("success", 0)
    success_rate = round(successes / total * 100, 1)

    return {
        "total_runs": total,
        "success_rate": success_rate,
        "status_breakdown": status_counts,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "avg_latency_ms": round(total_latency / total, 1),
    }


def print_summary(summary: dict) -> None:
    """Pretty-print the summary to the terminal."""
    print("\n=== Harness Run Summary ===")
    print(f"Total runs:      {summary.get('total_runs', 0)}")
    if summary.get("total_runs", 0) == 0:
        print("(no runs found)")
        return
    print(f"Success rate:    {summary['success_rate']}%")
    print(f"Avg latency:     {summary['avg_latency_ms']} ms")
    print(f"Total tokens:    {summary['total_input_tokens']} in / {summary['total_output_tokens']} out")
    print("\nStatus breakdown:")
    for status, count in summary["status_breakdown"].items():
        print(f"  {status:20s} {count}")