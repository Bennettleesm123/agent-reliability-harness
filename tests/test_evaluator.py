from harness.core.types import (
    EvalCase, Message, ModelResponse, FinishReason,
)
from harness.models.base import ModelInterface
from harness.eval.evaluator import evaluate_case, evaluate_suite, summarize


# A fake model that always returns a fixed answer — deterministic, no LLM.
class FakeModel(ModelInterface):
    def __init__(self, answer: str):
        self._answer = answer

    @property
    def model_id(self) -> str:
        return "fake:model"

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> ModelResponse:
        return ModelResponse(
            text=self._answer,
            finish_reason=FinishReason.STOP,
            input_tokens=1, output_tokens=1, latency_ms=1.0,
        )


def test_case_passes_when_answer_contains_expected():
    model = FakeModel("The capital of France is Paris.")
    case = EvalCase(name="capital", goal="What is the capital of France?",
                    expected_contains="Paris")
    result = evaluate_case(case, model)
    assert result.passed is True


def test_case_fails_when_answer_missing_expected():
    model = FakeModel("The capital of France is Paris.")
    case = EvalCase(name="wrong", goal="What is the capital of France?",
                    expected_contains="London")   # not in the answer
    result = evaluate_case(case, model)
    assert result.passed is False
    assert len(result.reasons) > 0                 # should explain why


def test_suite_summary():
    model = FakeModel("Paris")
    cases = [
        EvalCase(name="pass1", goal="x", expected_contains="Paris"),   # passes
        EvalCase(name="pass2", goal="y", expected_contains="Paris"),   # passes
        EvalCase(name="fail1", goal="z", expected_contains="Berlin"),  # fails
    ]
    results = evaluate_suite(cases, model)
    summary = summarize(results)
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["success_rate"] == 66.7