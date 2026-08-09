from harness.core.types import EvalCase, EvalResult, RunStatus
from harness.core.loop import run_agent, RunConfig
from harness.models.base import ModelInterface


def evaluate_case(case: EvalCase, model: ModelInterface, config: RunConfig | None = None) -> EvalResult:
    """Run one eval case and score it."""
    result = run_agent(case.goal, model, config)

    reasons = []
    passed = True

    # Check 1: did the run succeed (if it was supposed to)?
    if case.should_succeed:
        # if the run's status is NOT success, the case fails.
        # Compare result.status to RunStatus.SUCCESS. If they differ, set
        # passed = False and append a reason explaining it.
        if result.status != RunStatus.SUCCESS:
            passed = False
            reasons.append(f"expected success but got {result.status.value}")

    # Check 2: does the answer contain the expected text?
    if case.expected_contains is not None:
        answer = result.answer or ""
        # if case.expected_contains is NOT in the answer, fail.
        if case.expected_contains not in answer :
            passed = False
            reasons.append(f"answer missing expected text: '{case.expected_contains}'")

    # (Tool-call accuracy check would go here — we'll add it once the loop
    #  actually records tool calls, in a later phase. Noted as a limitation.)

    return EvalResult(
        case_name=case.name,
        passed=passed,
        reasons=reasons,
        actual_answer=result.answer,
        actual_status=result.status.value,
    )


def evaluate_suite(cases: list[EvalCase], model: ModelInterface, config: RunConfig | None = None) -> list[EvalResult]:
    """Run a whole suite of eval cases."""
    # run evaluate_case on each case and collect the results.
    # Return a list of EvalResult. (A list comprehension works well here.)
    return [evaluate_case (case, model, config) for case in cases]


def summarize(results: list[EvalResult]) -> dict:
    """Compute summary metrics over a suite's results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": round(passed / total * 100, 1) if total else 0.0,
    }