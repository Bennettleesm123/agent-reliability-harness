from pydantic import BaseModel
from harness.tools.base import Tool, Permission
from harness.tools.registry import ToolRegistry


# A toy tool's arguments: it adds two integers.
class AddArgs(BaseModel):
    a: int
    b: int


# The toy tool's handler — the actual function that runs.
def add_handler(a: int, b: int) -> int:
    return a + b


def make_registry():
    registry = ToolRegistry()
    registry.register(Tool(
        name="add",
        description="Add two integers.",
        args_model=AddArgs,
        handler=add_handler,
        permission=Permission.READ,
    ))
    return registry


# Test 1: valid arguments should execute and return the right answer.
def test_valid_args_execute():
    registry = make_registry()
    result = registry.validate_and_execute("add", {"a": 3, "b": 4})
    assert result == 7


# Test 2: invalid arguments (wrong type) should be REJECTED, not executed.
def test_invalid_args_rejected():
    registry = make_registry()
    try:
        registry.validate_and_execute("add", {"a": "banana", "b": 4})
        assert False, "should have raised on invalid args"
    except ValueError:
        pass  # correct — the harness rejected the bad call


# Test 3: an unknown tool name should raise.
def test_unknown_tool():
    registry = make_registry()
    try:
        registry.validate_and_execute("nonexistent", {})
        assert False, "should have raised on unknown tool"
    except KeyError:
        pass


# Test 4: registering the same tool name twice should raise.
def test_duplicate_registration():
    registry = make_registry()
    try:
        registry.register(Tool(
            name="add", description="dup", args_model=AddArgs, handler=add_handler,
        ))
        assert False, "should have raised on duplicate"
    except ValueError:
        pass