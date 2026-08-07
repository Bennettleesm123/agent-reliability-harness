from abc import ABC, abstractmethod
from harness.tools.base import Tool


class Approver(ABC):
    """Interface for getting human approval of a tool call."""

    @abstractmethod
    def approve(self, tool: Tool, args: dict) -> bool:
        """Return True if the action is approved, False if rejected."""
        ...


class CLIApprover(Approver):
    """Asks for approval via the command line."""

    def approve(self, tool: Tool, args: dict) -> bool:
        print(f"\n⚠️  Approval needed for '{tool.name}' ({tool.permission.value})")
        print(f"   Arguments: {args}")
        response = input("   Approve? [y/N]: ").strip().lower()
        # return True only if the response is "y" or "yes".
        return response in ("y", "yes")


class AutoApprover(Approver):
    """Auto-approves everything — for tests only, never production."""

    def approve(self, tool: Tool, args: dict) -> bool:
        return True