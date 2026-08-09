from harness.core.loop import run_agent, RunConfig
from harness.models.local import LocalModel
from harness.eval.failure_injection import FailingModel, SlowModel

# One success (real model)
run_agent("Say hi", LocalModel(model_name="qwen3:8b"), RunConfig())

# One error (failing model, retries exhausted)
run_agent("This will fail", FailingModel("boom"), RunConfig(max_retries=1, backoff_base=0.01))

# One timeout (slow model)
run_agent("This is slow", SlowModel(delay_seconds=2.0), RunConfig(timeout_seconds=0.5, max_retries=0))

print("Generated a mix of runs.")