import time
import ollama
from harness.core.types import Message, ModelResponse, FinishReason
from harness.models.base import ModelInterface


class LocalModel(ModelInterface):
    def __init__(self, model_name: str = "qwen3:14b"):
        self._model_name = model_name

    @property
    def model_id(self) -> str:
        # It should combine the provider and the model name, e.g. "ollama:qwen3:14b".
        return f"ollama {self._model_name}"

    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        # Convert our typed Message objects into the plain dicts Ollama expects.
        ollama_messages = [
            {"role": m.role.value, "content": m.content} for m in messages
        ]

        # Time the call so we can record latency (the harness needs this).
        start = time.monotonic()
        response = ollama.chat(model=self._model_name, messages=ollama_messages)
        # compute elapsed time in MILLISECONDS.
        # time.monotonic() returns seconds as a float. Convert the difference to ms.
        latency_ms = (time.monotonic() - start)*1000

        # Ollama returns token counts in the response metadata.
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
 
        # build and return a ModelResponse.
        # - text comes from response["message"]["content"]
        # - finish_reason: for now, always FinishReason.STOP (we'll handle
        #   tool calls in a later step)
        # - fill in input_tokens, output_tokens, latency_ms from above
        return ModelResponse(
            text=response["message"]["content"],
            finish_reason=FinishReason.STOP,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )