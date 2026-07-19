from pathlib import Path
import asyncio
from backend.config import Settings
from backend.providers.base import GenerationResult
from backend.providers.registry import ProviderRegistry
from backend.service import QAService
from backend.storage import Storage


def test_qa_service_returns_grounded_citation(tmp_path: Path):
    storage = Storage(tmp_path / "test.db")
    storage.add_document("AI Infra", "连续批处理可以提高 GPU 利用率以及系统整体吞吐量。", "test")
    class FakeRegistry:
        def resolve(self, *args):
            class Provider:
                id, name, model = "test", "Test", "test"
                async def generate(self, question, contexts, history=None, mode="knowledge"):
                    return GenerationResult(f'{contexts[0]["text"]} [1]', self.name, self.model)
            return Provider()
    result = asyncio.run(QAService(storage, FakeRegistry()).ask("怎样提高推理吞吐量？", None, None, "knowledge"))
    assert result["citations"][0]["title"] == "AI Infra"
    assert "吞吐量" in result["answer"]


def test_provider_registry_defaults_to_deepseek_chat():
    registry = ProviderRegistry(Settings(_env_file=None))
    assert registry.default_provider == "ai-infra"
    assert registry.resolve(registry.default_model_id).model == "deepseek-chat"


def test_registry_rejects_direct_provider():
    registry = ProviderRegistry(Settings(_env_file=None))
    try:
        registry.resolve(None, "ollama", "deepseek-chat")
        assert False, "direct provider should be rejected"
    except ValueError as error:
        assert "不支持" in str(error)


def test_registry_can_select_direct_deepseek():
    registry = ProviderRegistry(Settings(_env_file=None, deepseek_api_key="test-key"))
    provider = registry.resolve("deepseek-chat", use_ai_infra=False)
    assert provider.id == "deepseek-direct"
    assert provider.configured
