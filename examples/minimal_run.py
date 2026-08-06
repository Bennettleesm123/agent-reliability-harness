from harness.models.local import LocalModel
from harness.core.loop import run_agent

model = LocalModel(model_name="qwen3:8b")   # 8b for speed
result = run_agent("Say hello in one sentence.", model)

print("Status:", result.status)
print("Answer:", result.answer)
print("Steps:", result.steps_taken)
print("Tokens:", result.total_input_tokens, "in /", result.total_output_tokens, "out")
print("Latency:", round(result.total_latency_ms), "ms")