import time
from .providers.registry import ProviderRegistry
from .retrieval import BM25Retriever
from .storage import Storage


class QAService:
    def __init__(self, storage: Storage, registry: ProviderRegistry, max_chunks: int = 6, min_score: float = 0.05):
        self.storage, self.registry = storage, registry
        self.retriever, self.max_chunks, self.min_score = BM25Retriever(), max_chunks, min_score

    async def ask(self, question, provider_id, model, mode="chat", history=None, model_id=None, use_ai_infra=True, route_mode=None):
        started = time.perf_counter()
        search_started = time.perf_counter()
        use_knowledge = mode == "knowledge" or (mode == "auto" and bool(self.storage.chunks()))
        effective_mode = "knowledge" if use_knowledge else "chat"
        hits = [item for item in self.retriever.search(question, self.storage.chunks(), self.max_chunks) if item["score"] >= self.min_score] if use_knowledge else []
        search_ms = (time.perf_counter() - search_started) * 1000
        contexts = [{**item, "citation": f"[{index + 1}]"} for index, item in enumerate(hits)]
        provider = self.registry.resolve(model_id, provider_id, model, use_ai_infra, route_mode)
        generation_started = time.perf_counter()
        degraded, warning = False, None
        generated = await provider.generate(question, contexts, history, effective_mode)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        result = {
            "question": question, "answer": generated.text, "provider": generated.provider, "model": generated.model, "mode": effective_mode,
            "degraded": degraded, "warning": warning,
            "citations": [{"id": item["citation"], "documentId": item["documentId"], "title": item["title"], "chunk": item["index"] + 1, "score": round(item["score"], 4), "excerpt": item["text"][:240]} for item in contexts],
            "timing": {"searchMs": round(search_ms, 2), "generationMs": round(generation_ms, 2), "totalMs": round((time.perf_counter() - started) * 1000, 2)}
        }
        self.storage.add_conversation(result)
        return result
