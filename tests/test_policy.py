from pydantic import BaseModel
from harness.tools.base import Tool, Permission
from harness.tools.registry import ToolRegistry
from harness.policy.policy import Policy
from harness.policy.approval import AutoApprover
from harness.core.types import PolicyDecision


class NoArgs(BaseModel):
    pass


def dummy_handler() -> str:
    return "executed"


def make_tool(name, permission, requires_approval=False):
    return Tool(
        name=name, description="test", args_model=NoArgs,
        handler=dummy_handler, permission=permission,
        requires_approval=requires_approval,
    )


# --- Policy decision tests (no execution, just the decision) ---

def test_read_is_allowed():
    policy = Policy()
    tool = make_tool("read_tool", Permission.READ)
    assert policy.decide(tool) == PolicyDecision.ALLOW


def test_dangerous_requires_approval():
    policy = Policy()
    tool = make_tool("danger_tool", Permission.DANGEROUS)
    assert policy.decide(tool) == PolicyDecision.REQUIRE_APPROVAL


def test_write_without_flag_is_allowed():
    policy = Policy()
    tool = make_tool("write_tool", Permission.WRITE, requires_approval=False)
    assert policy.decide(tool) == PolicyDecision.ALLOW


def test_write_with_flag_requires_approval():
    policy = Policy()
    tool = make_tool("write_tool", Permission.WRITE, requires_approval=True)
    assert policy.decide(tool) == PolicyDecision.REQUIRE_APPROVAL


# --- Execution + approval tests ---

def test_read_executes_without_approval():
    registry = ToolRegistry()
    registry.register(make_tool("read_tool", Permission.READ))
    policy = Policy()
    # AutoApprover would approve, but READ shouldn't even need it.
    result = registry.validate_and_execute("read_tool", {}, policy=policy, approver=AutoApprover())
    assert result == "executed"


def test_dangerous_executes_when_approved():
    registry = ToolRegistry()
    registry.register(make_tool("danger_tool", Permission.DANGEROUS))
    policy = Policy()
    # AutoApprover says yes -> should execute.
    result = registry.validate_and_execute("danger_tool", {}, policy=policy, approver=AutoApprover())
    assert result == "executed"


def test_dangerous_blocked_when_rejected():
    from harness.policy.approval import Approver

    class RejectingApprover(Approver):
        def approve(self, tool, args) -> bool:
            return False   # always reject

    registry = ToolRegistry()
    registry.register(make_tool("danger_tool", Permission.DANGEROUS))
    policy = Policy()
    try:
        registry.validate_and_execute("danger_tool", {}, policy=policy, approver=RejectingApprover())
        assert False, "should have raised — approval was rejected"
    except PermissionError:
        pass   # correct — rejected approval prevented execution