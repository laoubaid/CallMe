from typing import Any
from pydantic import BaseModel, ConfigDict

class TypeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str

class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str

class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, TypeInfo]
    returns: TypeInfo

class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
