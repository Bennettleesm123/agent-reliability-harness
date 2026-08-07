from harness.core.types import PolicyDecision
from harness.tools.base import Tool, Permission


class Policy:
    #Decides whether a tool call is allowed, needs approval, or denied. Separates the decision (policy) from the execution (mechanism in the
    #registry), so approval rules can change without touching execution.
    

    def decide(self, tool: Tool) -> PolicyDecision:
        # DANGEROUS tools always need a human.
        if tool.permission == Permission.DANGEROUS:
            # return the decision that requires human approval.
            return PolicyDecision.REQUIRE_APPROVAL

        # WRITE tools need approval only if explicitly flagged.
        if tool.permission == Permission.WRITE:
            # if the tool requires_approval, return REQUIRE_APPROVAL;
            # otherwise return ALLOW.
            if tool.requires_approval:
                return PolicyDecision.REQUIRE_APPROVAL
            return PolicyDecision.ALLOW

        # READ tools (and anything else) are allowed automatically.
        # return the decision that allows automatic execution.
        return PolicyDecision.ALLOW