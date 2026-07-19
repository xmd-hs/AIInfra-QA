import asyncio
from pathlib import Path
from backend.config import Settings
from backend.providers.base import GenerationResult
from backend.service import QAService
from backend.storage import Storage


def test_chat_mode_does_not_require_knowledge_documents(tmp_path: Path):
    class FakeRegistry:
        def resolve(self, *args):
            class Provider:
                id, name, model = "test", "Test", "test"
                async def generate(self, question, contexts, history=None, mode="chat"):
                    return GenerationResult("测试动态回答", self.name, self.model)
            return Provider()
    service = QAService(Storage(tmp_path / "chat.db"), FakeRegistry())
    result = asyncio.run(service.ask("你好，请介绍一下自己", None, None, "chat", []))
    assert result["mode"] == "chat"
    assert result["citations"] == []
    assert result["answer"] == "测试动态回答"
