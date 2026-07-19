from dataclasses import dataclass
from .ai_infra import AIInfraProvider
from .deepseek import DeepSeekProvider, PythonBaselineProvider
from ..config import Settings


@dataclass(frozen=True)
class InfraModel:
    id: str
    name: str
    model: str
    description: str
    free: bool = True
    local: bool = True
    recommended: bool = False


class ProviderRegistry:
    """Model catalog whose only inference transport is C++ AI Infra."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.default_provider = "ai-infra"
        self.default_model_id = "deepseek-chat"
        self.catalog = [
            InfraModel("deepseek-chat", "DeepSeek Chat", "deepseek-chat", "可选择直接调用或经过 C++ AI Infra 调度", free=False, local=False, recommended=True),
        ]

    def resolve(self, model_id=None, provider_id=None, model=None, use_ai_infra=True, route_mode=None):
        route = route_mode or ("infra" if use_ai_infra else "direct")
        if route == "direct":
            return DeepSeekProvider(self.settings.deepseek_base_url, self.settings.deepseek_api_key, model or "deepseek-chat")
        if route == "baseline":
            return PythonBaselineProvider(self.settings.deepseek_base_url, self.settings.deepseek_api_key, model or "deepseek-chat")
        if route != "infra":
            raise ValueError("不支持的推理模式")
        if provider_id and provider_id != "ai-infra": raise ValueError("不支持的推理入口")
        selected = next((item for item in self.catalog if item.id == (model_id or self.default_model_id)), None)
        model_name = model or (selected.model if selected else model_id)
        if not model_name:
            raise ValueError("不支持的 AI Infra 模型")
        return AIInfraProvider(self.settings.ai_infra_base_url, model_name)

    def get(self, provider_id: str | None, model: str | None = None):
        return self.resolve(None, provider_id, model)

    def list(self) -> list[dict]:
        return [{
            "id": item.id, "name": item.name, "provider": "ai-infra", "model": item.model,
            "description": item.description, "free": item.free, "local": item.local,
            "recommended": item.recommended, "configured": True,
            "default": item.id == self.default_model_id,
        } for item in self.catalog]
