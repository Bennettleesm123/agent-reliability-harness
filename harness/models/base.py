from abc import ABC, abstractmethod
from harness.core.types import Message, ModelResponse


class ModelInterface(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        """A stable identifier for this model, e.g. 'ollama:qwen3:14b'."""
        ...

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        ...
