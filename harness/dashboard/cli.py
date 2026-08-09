import sys
from harness.dashboard.summary import load_all_traces, summarize_traces, print_summary


def main():
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    traces = load_all_traces(runs_dir)
    summary = summarize_traces(traces)
    print_summary(summary)


if __name__ == "__main__":
    main()