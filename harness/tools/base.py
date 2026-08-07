from pydantic import BaseModel
from typing import Callable, Any, Type
from enum import Enum


# Permission level for a tool. Read = safe/no side effects; Write = has
# side effects (may need approval); Dangerous = always needs approval.
class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


# A registered tool. The args_model is a Pydantic model class describing
# the tool's arguments — it's the single source of truth used BOTH to
# generate the schema shown to the LLM AND to validate the LLM's arguments.
class Tool(BaseModel):
    name: str
    description: str
    args_model: Type[BaseModel]        # a Pydantic model CLASS (not instance)
    handler: Callable[..., Any]
    permission: Permission = Permission.READ
    requires_approval: bool = False

    # Pydantic needs this to allow arbitrary types (Callable, model classes).
    model_config = {"arbitrary_types_allowed": True}

    def json_schema(self) -> dict:
        # return the JSON schema for this tool's arguments.
        # Pydantic models have a built-in method that produces a JSON schema.
        # self.args_model has a method called model_json_schema()
        return self.args.model_json_schema()