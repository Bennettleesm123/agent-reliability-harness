# A small file-operations agent that runs THROUGH the harness.
# It has three tools at three permission levels, so the harness's
# validation, policy, and approval all do real work here:
#   - read_file   -> READ       (safe, auto-allowed)
#   - write_file  -> WRITE      (side effect, needs approval)
#   - delete_file -> DANGEROUS  (always needs approval)
#
# This is the capstone proving the harness hardens a real agent
# The agent gains schema validation, permission gating, human approval,
# tracing, retries, and step limits WITHOUT any of that logic living
# in the agent itself. It all comes from the harness.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from pydantic import BaseModel

from harness.tools.base import Tool, Permission
from harness.tools.registry import ToolRegistry
from harness.policy.policy import Policy
from harness.policy.approval import CLIApprover

# A sandbox directory the agent is allowed to touch.
SANDBOX = Path("agent_sandbox")


# ---- Tool argument schemas (Pydantic = one source of truth) ----
class ReadArgs(BaseModel):
    filename: str

class WriteArgs(BaseModel):
    filename: str
    content: str

class DeleteArgs(BaseModel):
    filename: str


# ---- Tool handlers (the actual work) ----
def read_file(filename: str) -> str:
    path = SANDBOX / filename
    if not path.exists():
        return f"(file '{filename}' not found)"
    return path.read_text()

def write_file(filename: str, content: str) -> str:
    SANDBOX.mkdir(exist_ok=True)
    (SANDBOX / filename).write_text(content)
    return f"wrote {len(content)} chars to {filename}"

def delete_file(filename: str) -> str:
    path = SANDBOX / filename
    if path.exists():
        path.unlink()
        return f"deleted {filename}"
    return f"(file '{filename}' not found)"


# ---- Register the tools with their permission levels ----
def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="read_file", description="Read a file's contents.",
        args_model=ReadArgs, handler=read_file,
        permission=Permission.READ,
    ))
    registry.register(Tool(
        name="write_file", description="Write content to a file.",
        args_model=WriteArgs, handler=write_file,
        permission=Permission.WRITE, requires_approval=True,
    ))
    registry.register(Tool(
        name="delete_file", description="Delete a file.",
        args_model=DeleteArgs, handler=delete_file,
        permission=Permission.DANGEROUS,
    ))
    return registry


# ---- Demo: run some tool calls THROUGH the harness's gate ----
def main():
    registry = build_registry()
    policy = Policy()
    approver = CLIApprover()

    print("=== File agent running through the harness ===\n")

    # 1. A WRITE call -> needs approval.
    print("[1] Writing a file (WRITE -> should require approval):")
    try:
        registry.validate_and_execute("write_file", {"filename": "notes.txt", "content": "hello harness"},
                                      policy=policy, approver=approver)
        print(read_file("notes.txt"), "\n")
    except PermissionError as e:
        print(f"Action blocked: {e}\n")

    # 2. A DELETE call -> DANGEROUS, should pause for approval.
    print("[2] Deleting a file (DANGEROUS -> should require approval):")
    try:
        result = registry.validate_and_execute("delete_file", {"filename": "notes.txt"},
                                               policy=policy, approver=approver)
        print(result, "\n")
    except PermissionError as e:
        print(f"Action blocked: {e}\n")   # rejection is expected, not a crash

    # 3. A malformed call -> harness rejects it before execution.
    print("[3] Malformed call (missing required arg -> harness rejects):")
    try:
        registry.validate_and_execute("write_file", {"filename": "x.txt"},  # no 'content'
                                      policy=policy, approver=approver)
    except ValueError as e:
        print(f"Rejected as expected: {e}\n")


if __name__ == "__main__":
    main()