import httpx
from .base import GenerationResult, LLMProvider, build_messages


class AIInfraProvider(LLMProvider):
    id, name = "ai-infra", "C++ AI Infra"

    def __init__(self, base_url: str, model: str):
        self.base_url, self.model = base_url.rstrip("/"), model

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    async def generate(self, question: str, contexts: list[dict], history: list[dict] | None = None, mode: str = "knowledge") -> GenerationResult:
        messages = build_messages(question, contexts, history, mode)
        prompt = "\n\n".join(f'{item["role"]}: {item["content"]}' for item in messages)
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.base_url}/v1/infer", params={"model": self.model, "max_tokens": 1000}, json={"prompt": prompt, "max_tokens": 1000})
            response.raise_for_status()
            data = response.json()
        return GenerationResult(data["text"], self.name, self.model)
