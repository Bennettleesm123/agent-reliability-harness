from harness.core.loop import run_agent, RunConfig
from harness.core.types import RunStatus
from harness.eval.failure_injection import FailingModel, SlowModel, FlakyModel


def test_api_error_handled_cleanly():
    # A model that always fails -> after retries, returns ERROR (no crash).
    model = FailingModel("boom")
    config = RunConfig(max_retries=2, backoff_base=0.01)  # tiny backoff for fast test
    result = run_agent("do something", model, config)
    assert result.status == RunStatus.ERROR
    assert "boom" in result.error


def test_timeout_fires_on_slow_model():
    # A model slower than the timeout -> run should TIMEOUT.
    model = SlowModel(delay_seconds=2.0)
    config = RunConfig(timeout_seconds=0.5, max_retries=0)
    result = run_agent("do something", model, config)
    assert result.status == RunStatus.TIMEOUT


def test_flaky_model_recovers_via_retries():
    # Fails twice, succeeds on the 3rd call -> retries should recover it.
    model = FlakyModel(fail_times=2)
    config = RunConfig(max_retries=3, backoff_base=0.01)
    result = run_agent("do something", model, config)
    assert result.status == RunStatus.SUCCESS               # recovered!
    assert result.answer == "succeeded after retries"