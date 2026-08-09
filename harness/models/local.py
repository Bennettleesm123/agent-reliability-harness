import os
import time
import ollama
from harness.core.types import Message, ModelResponse, FinishReason
from harness.models.base import ModelInterface


class LocalModel(ModelInterface):
    def __init__(self, model_name: str = "qwen3:14b"):
        self._model_name = model_name
        # Ollama host: localhost normally, host.docker.internal inside Docker.
        # Configurable via env var so the same code works both ways.
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    @property
    def model_id(self) -> str:
        return f"ollama:{self._model_name}"

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> ModelResponse:
        ollama_messages = [
            {"role": m.role.value, "content": m.content} for m in messages
        ]
        start = time.monotonic()
        # Use the configured client instead of the module-level ollama.chat.
        response = self._client.chat(model=self._model_name, messages=ollama_messages)
        latency_ms = (time.monotonic() - start) * 1000

        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        return ModelResponse(
            text=response["message"]["content"],
            finish_reason=FinishReason.STOP,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )