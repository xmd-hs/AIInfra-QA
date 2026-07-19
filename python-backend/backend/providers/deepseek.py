import httpx
from .base import GenerationResult, LLMProvider, build_messages


class DeepSeekProvider(LLMProvider):
    id, name, model = "deepseek-direct", "DeepSeek Direct", "deepseek-chat"

    def __init__(self, base_url: str, api_key: str, model: str = "deepseek-chat"):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def generate(self, question, contexts, history=None, mode="chat"):
        if not self.configured:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        payload = {"model": self.model, "messages": build_messages(question, contexts, history, mode), "temperature": 0.5 if mode == "chat" else 0.2, "max_tokens": 1400}
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers={"authorization": f"Bearer {self.api_key}"}, json=payload)
            response.raise_for_status()
            data = response.json()
        return GenerationResult(data["choices"][0]["message"]["content"], self.name, self.model)

    async def complete(self, prompt: str, model: str = "deepseek-chat") -> str:
        result = await self.generate(prompt, [], [], "chat")
        return result.text


class PythonBaselineProvider(DeepSeekProvider):
    """Match the AI Infra request shape while bypassing its C++ scheduler."""

    id, name = "python-baseline", "Python 基线（无 Infra）"

    async def generate(self, question, contexts, history=None, mode="chat"):
        messages = build_messages(question, contexts, history, mode)
        prompt = "\n\n".join(f'{item["role"]}: {item["content"]}' for item in messages)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5 if mode == "chat" else 0.2,
            "max_tokens": 1000,
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return GenerationResult(data["choices"][0]["message"]["content"], self.name, self.model)
