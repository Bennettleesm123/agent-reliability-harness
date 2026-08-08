import json
from pathlib import Path
from harness.core.types import Message, ModelResponse, FinishReason, Trace
from harness.models.base import ModelInterface


def load_trace(path: str) -> Trace:
    """Load a saved trace from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    # turn the loaded dict into a Trace object.
    # Trace is a Pydantic model, construct it from the dict with **data.
    return Trace(**data)


class ReplayModel(ModelInterface):
    """A model that replays recorded responses from a trace instead of
    calling a real LLM. Implements ModelInterface, so the loop can use it
    exactly like a real model — but it's deterministic."""

    def __init__(self, trace: Trace):
        self._trace = trace
        # Pull out the recorded model-call steps, in order.
        self._model_steps = [s for s in trace.steps if s.kind == "model_call"]
        self._index = 0   # which recorded step we're up to

    @property
    def model_id(self) -> str:
        # return an id showing this is a replay of the original model.
        # combine "replay:" with the trace's original model_id
        # (self._trace.model_id).
        return f"replay:{self._trace.model_id}"

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> ModelResponse:
        # If we've run out of recorded steps, that's an error — the replay
        # is being asked for more than was recorded.
        if self._index >= len(self._model_steps):
            raise RuntimeError("Replay exhausted: no more recorded responses")

        step = self._model_steps[self._index]
        self._index += 1

        # build a ModelResponse from the recorded step.
        # Note: the trace step recorded tokens and latency but NOT the text
        # (our minimal trace didn't store response text yet). For now,
        # return a placeholder text and the recorded token/latency values.
        # - text: "(replayed step {step.step_number})"
        # - finish_reason: FinishReason.STOP
        # - input_tokens, output_tokens, latency_ms: from the recorded step
        return ModelResponse(
            text=step.text if step.text is not None else f"(replayed step {step.step_number})",
            finish_reason=FinishReason.STOP,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
            latency_ms=step.latency_ms,
        )