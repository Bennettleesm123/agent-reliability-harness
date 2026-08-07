from pydantic import ValidationError
from harness.tools.base import Tool
from harness.core.types import PolicyDecision

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        # prevent registering two tools with the same name.
        # If tool.name is already a key in self._tools, raise a ValueError
        # with a helpful message. Otherwise, store the tool under its name.
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas_for_model(self) -> list[dict]:
        # What the LLM sees: name, description, and argument schema for each tool.
        return [
            {"name": t.name, "description": t.description, "parameters": t.json_schema()}
            for t in self._tools.values()
        ]

    def validate_and_execute(self, name: str, raw_args: dict, policy=None, approver=None):
        tool = self.get(name)

        # Validate arguments (same as before).
        try:
            validated = tool.args_model(**raw_args)
        except ValidationError as e:
            raise ValueError(f"Invalid arguments for tool '{name}': {e}")

        # NEW: consult the policy before executing, if one was provided.
        if policy is not None:
            decision = policy.decide(tool)

            if decision == PolicyDecision.DENY:
                raise PermissionError(f"Tool '{name}' denied by policy")

            if decision == PolicyDecision.REQUIRE_APPROVAL:
                # Need a human (or approver) to say yes.
                if approver is None:
                    raise PermissionError(f"Tool '{name}' needs approval but no approver provided")
                approved = approver.approve(tool, validated.model_dump())
                if not approved:
                    raise PermissionError(f"Tool '{name}' rejected by approver")

        # Passed validation and policy — execute.
        return tool.handler(**validated.model_dump())