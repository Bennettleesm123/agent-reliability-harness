import time
from harness.core.types import Message, ModelResponse, FinishReason
from harness.models.base import ModelInterface


class FailingModel(ModelInterface):
    """A model that raises an exception on command — simulates an API error."""

    def __init__(self, error_message: str = "simulated API failure"):
        self._error_message = error_message

    @property
    def model_id(self) -> str:
        return "failing:model"

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> ModelResponse:
        # raise an exception with self._error_message to simulate
        # an API call failing. Use `raise RuntimeError(...)`.
        raise RuntimeError(self._error_message)


class SlowModel(ModelInterface):
    """A model that sleeps longer than the timeout — simulates a hang."""

    def __init__(self, delay_seconds: float = 5.0):
        self._delay = delay_seconds

    @property
    def model_id(self) -> str:
        return "slow:model"

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> ModelResponse:
        # Sleep to simulate a slow/hanging model call.
        time.sleep(self._delay)
        return ModelResponse(
            text="finally responded",
            finish_reason=FinishReason.STOP,
            input_tokens=1, output_tokens=1, latency_ms=self._delay * 1000,
        )


class FlakyModel(ModelInterface):
    """Fails the first N times, then succeeds — simulates transient errors
    that retry logic should recover from."""

    def __init__(self, fail_times: int = 2):
        self._fail_times = fail_times
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return "flaky:model"

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> ModelResponse:
        self._call_count += 1
        # if we haven't exceeded fail_times yet, raise a RuntimeError
        # (simulating a transient failure). Otherwise, return a successful
        # ModelResponse. Compare self._call_count to self._fail_times.
        if self._call_count <= self._fail_times:
            raise RuntimeError(f"transient failure #{self._call_count}")
        return ModelResponse(
            text="succeeded after retries",
            finish_reason=FinishReason.STOP,
            input_tokens=1, output_tokens=1, latency_ms=1.0,
        )