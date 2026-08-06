import time
from harness.core.types import Message, Role, RunResult, RunStatus
from harness.models.base import ModelInterface


# Configuration for the loop's hard limits. Dataclass-style via Pydantic
# so limits are validated (e.g. max_steps can't be negative).
from pydantic import BaseModel, Field

class RunConfig(BaseModel):
    max_steps: int = Field(default=10, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)


def run_agent(goal: str, model: ModelInterface, config: RunConfig | None = None) -> RunResult:
    # Use default limits if none provided.
    config = config or RunConfig()

    # Accumulators for metrics across all steps.
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0.0

    # The conversation starts with the user's goal.
    messages: list[Message] = [Message(role=Role.USER, content=goal)]

    start_time = time.monotonic()

    for step in range(config.max_steps):
        # enforce the timeout.
        # If (current monotonic time - start_time) exceeds config.timeout_seconds,
        # return a RunResult with status=RunStatus.TIMEOUT, steps_taken=step,
        # and the accumulated token/latency totals.
        if (time.monotonic()-start_time > config.timeout_seconds):
            return RunResult(
                status=RunStatus.TIMEOUT,
                steps_taken=step,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_latency_ms=total_latency_ms,
            )

        # Call the model
        try:
            response = model.complete(messages)
        except Exception as e:
            # BLANK 2: the model call failed. Return a RunResult with
            # status=RunStatus.ERROR, steps_taken=step, the totals so far,
            # and error=str(e).
            return RunResult(
                status=RunStatus.ERROR,
                steps_taken=step,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_latency_ms=total_latency_ms,
                error=str(e),
            )

        # Accumulate this step's metrics.
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens
        total_latency_ms += response.latency_ms

        # For this minimal version (no tools yet), any response is the final
        # answer. Return success.
        # return a RunResult with status=RunStatus.SUCCESS, answer=response.text, steps_taken=step + 1, and the totals.
        return RunResult(status=RunStatus.SUCCESS,
        answer=response.text,
        steps_taken=step + 1,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms)


    # Loop finished without returning -> hit the step limit.
    return RunResult(
        status=RunStatus.MAX_STEPS,
        steps_taken=config.max_steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
    )