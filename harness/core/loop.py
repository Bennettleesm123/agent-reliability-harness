import time
from harness.core.types import Message, Role, RunResult, RunStatus, TraceStep
from harness.models.base import ModelInterface
from harness.tracing.tracer import Tracer
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    max_steps: int = Field(default=10, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)


def run_agent(goal: str, model: ModelInterface, config: RunConfig | None = None) -> RunResult:
    config = config or RunConfig()

    # Start a tracer for this run.
    tracer = Tracer(goal=goal, model_id=model.model_id)

    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0.0

    messages: list[Message] = [Message(role=Role.USER, content=goal)]
    start_time = time.monotonic()

    for step in range(config.max_steps):
        if time.monotonic() - start_time > config.timeout_seconds:
            tracer.finalize("timeout", total_input_tokens, total_output_tokens, total_latency_ms)
            return RunResult(
                status=RunStatus.TIMEOUT, steps_taken=step,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_latency_ms=total_latency_ms,
            )

        try:
            response = model.complete(messages)
        except Exception as e:
            tracer.finalize("error", total_input_tokens, total_output_tokens, total_latency_ms)
            return RunResult(
                status=RunStatus.ERROR, steps_taken=step,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_latency_ms=total_latency_ms, error=str(e),
            )

        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens
        total_latency_ms += response.latency_ms

        # Record this model call as a trace step.
        tracer.record_step(TraceStep(
            step_number=step + 1,
            kind="model_call",
            text=response.text,          # NEW: capture the actual response
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        ))

        # Minimal version: any response is the final answer.
        tracer.finalize("success", total_input_tokens, total_output_tokens, total_latency_ms)
        return RunResult(
            status=RunStatus.SUCCESS, answer=response.text, steps_taken=step + 1,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_latency_ms=total_latency_ms,
        )

    tracer.finalize("max_steps_exceeded", total_input_tokens, total_output_tokens, total_latency_ms)
    return RunResult(
        status=RunStatus.MAX_STEPS, steps_taken=config.max_steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
    )