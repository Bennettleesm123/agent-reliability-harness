from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


# The role of a message in a conversation. An Enum (not a raw string) so
# invalid roles are impossible — you can't typo "assistan" and have it
# silently accepted. This is the harness philosophy: make bad states
# unrepresentable.
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


# A single message. Making this a Pydantic model (rather than a loose dict)
# means every message is validated: role must be a valid Role, content must
# be a string. No malformed messages can enter the system.
class Message(BaseModel):
    role: Role
    content: str
    # Optional: if this message is a tool result, which call it answers.
    tool_call_id: str | None = None


# Why the model finished generating. An Enum because the harness branches
# on this (a "tool_call" finish means execute a tool; "length" means the
# output was truncated and may be malformed; "error" means recovery logic).
class FinishReason(str, Enum):
    STOP = "stop"              # model finished normally
    TOOL_CALL = "tool_call"    # model wants to call a tool
    LENGTH = "length"          # hit token limit (output may be cut off)
    ERROR = "error"            # something went wrong


# The normalized result of a model call. EVERY model (local or hosted)
# returns this exact shape, so the rest of the harness never sees
# provider-specific formats. This is the "normalize at the boundary"
# principle — provider quirks stop here.
class ModelResponse(BaseModel):
    text: str
    finish_reason: FinishReason
    input_tokens: int = Field(ge=0)   # ge=0: can't be negative (validation)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    # Raw tool call the model requested, if any (parsed in a later step).
    raw_tool_call: dict | None = None

class RunStatus(str, Enum):
    SUCCESS = "success"
    MAX_STEPS = "max_steps_exceeded"
    TIMEOUT = "timeout"
    ERROR = "error"


class RunResult(BaseModel):
    status: RunStatus
    answer: str | None = None          # the final answer, if successful
    steps_taken: int
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    error: str | None = None           # error detail, if status is ERROR

class PolicyDecision(str, Enum):
    ALLOW = "allow"           # execute automatically
    REQUIRE_APPROVAL = "require_approval"  # ask a human first
    DENY = "deny"             # refuse outright

# One recorded step within a run — a model call or a tool call.
class TraceStep(BaseModel):
    step_number: int
    kind: str
    text: str | None = None          # the model's response text (for replay)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    policy_decision: str | None = None
    error: str | None = None


# The full trace of one run.
class Trace(BaseModel):
    run_id: str
    goal: str
    model_id: str
    started_at: str                    # ISO timestamp
    steps: list[TraceStep] = Field(default_factory=list)
    final_status: str | None = None
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0, ge=0)

# One evaluation test case: a task plus how to judge success.
class EvalCase(BaseModel):
    name: str
    goal: str
    # Scoring criteria (all optional — use whichever apply):
    expected_contains: str | None = None      # answer should contain this text
    expected_tools: list[str] | None = None   # these tools should be called
    should_succeed: bool = True               # should the run succeed at all?


# The result of running one eval case.
class EvalResult(BaseModel):
    case_name: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)  # why it passed/failed
    actual_answer: str | None = None
    actual_status: str | None = None