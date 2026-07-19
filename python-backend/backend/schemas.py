from typing import Literal
from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=10, max_length=5_000_000)
    source: str = Field(default="manual", max_length=300)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    provider: Literal["ai-infra"] | None = None
    model: str | None = Field(default=None, max_length=200)
    model_id: str | None = Field(default=None, max_length=100)
    mode: Literal["chat", "knowledge", "auto"] = "chat"
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    use_ai_infra: bool = True
    route_mode: Literal["direct", "baseline", "infra"] | None = None


class CodeSaveRequest(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    content: str = Field(max_length=500_000)


class CodeEditRequest(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    instruction: str = Field(min_length=2, max_length=4000)
    content: str = Field(max_length=500_000)


class CodeWorkspaceRequest(BaseModel):
    root: str = Field(min_length=1, max_length=1000)
