# pyrefly: ignore [missing-import]
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict
    returns: dict