from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str


class LLMProvider(ABC):
    id: str
    name: str
    model: str

    @property
    @abstractmethod
    def configured(self) -> bool: ...

    @abstractmethod
    async def generate(self, question: str, contexts: list[dict], history: list[dict] | None = None, mode: str = "knowledge") -> GenerationResult: ...


def build_messages(question: str, contexts: list[dict], history: list[dict] | None = None, mode: str = "knowledge") -> list[dict[str, str]]:
    history = history or []
    if mode == "chat":
        system = "你是知问，一个专业、友好、清晰的通用 AI 助手。请直接回答用户问题；复杂问题给出有条理的解释，不确定时如实说明。使用与用户相同的语言。"
        return [{"role": "system", "content": system}, *history[-12:], {"role": "user", "content": question}]
    evidence = "\n\n".join(f'{item["citation"]} 来源：{item["title"]}\n{item["text"]}' for item in contexts)
    return [
        {"role": "system", "content": "你是严谨的知识库问答助手。只能根据提供的资料回答；资料不足时必须明确说明，不能编造。使用中文回答，并在相关结论后标注引用编号，例如 [1]。"},
        *history[-8:],
        {"role": "user", "content": f"资料：\n{evidence or '无可用资料'}\n\n问题：{question}"}
    ]
