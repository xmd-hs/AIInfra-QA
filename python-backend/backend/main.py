import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .providers.registry import ProviderRegistry
from .providers.deepseek import DeepSeekProvider
from .schemas import AskRequest, DocumentCreate, CodeEditRequest, CodeSaveRequest, CodeWorkspaceRequest
from .service import QAService
from .storage import Storage
from .code_workspace import CodeWorkspace, strip_code_fence

app = FastAPI(title="知问 API", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:3000", "http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173", "http://localhost:4173"], allow_methods=["*"], allow_headers=["*"])
storage = Storage(settings.database_path)
registry = ProviderRegistry(settings)
service = QAService(storage, registry, settings.max_context_chunks, settings.min_search_score)
code_workspace = CodeWorkspace(settings.code_workspace_path)


@app.get("/api/health")
async def health():
    provider = registry.resolve(registry.default_model_id)
    available = False
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{settings.ai_infra_base_url.rstrip('/')}/health")
            available = response.is_success
    except httpx.HTTPError:
        pass
    return {"status": "ok" if available else "unavailable", "provider": provider.name, "model": provider.model, "configured": True, "infraAvailable": available, "infraUrl": settings.ai_infra_base_url}


@app.get("/api/providers")
def providers():
    return {"providers": registry.list(), "default": registry.default_model_id}


@app.get("/api/code/files")
def code_files():
    return {"root": str(code_workspace.root), "files": code_workspace.files()}


@app.put("/api/code/workspace")
def select_code_workspace(body: CodeWorkspaceRequest):
    try:
        code_workspace.set_root(Path(body.root))
        return {"root": str(code_workspace.root), "files": code_workspace.files()}
    except (OSError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/code/file")
def code_file(path: str):
    try:
        return {"path": path, "content": code_workspace.read(path)}
    except (OSError, UnicodeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


@app.put("/api/code/file")
def save_code_file(body: CodeSaveRequest):
    try:
        code_workspace.write(body.path, body.content)
        return {"ok": True, "path": body.path}
    except (OSError, UnicodeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/code/propose")
async def propose_code_edit(body: CodeEditRequest):
    prompt = f"""You are editing the file {body.path}.
Apply only the requested change. Preserve unrelated behavior and formatting.
Return only the complete updated file content, without Markdown fences or explanation.

REQUEST:
{body.instruction}

CURRENT FILE:
{body.content}"""
    try:
        provider = registry.resolve(model_id="deepseek-chat", route_mode="infra")
        result = await provider.generate(prompt, [], [], "chat")
        return {"path": body.path, "content": strip_code_fence(result.text), "provider": result.provider, "model": result.model}
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        raise HTTPException(503, f"代码修改生成失败：{error}") from error


@app.get("/api/stats")
def stats(): return storage.stats()


@app.get("/api/documents")
def documents(): return {"documents": storage.documents()}


@app.post("/api/documents", status_code=201)
def add_document(body: DocumentCreate): return {"document": storage.add_document(body.title, body.content, body.source)}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    if not storage.delete_document(document_id):
        raise HTTPException(404, "文档不存在")
    return {"ok": True}


@app.post("/api/ask")
async def ask(body: AskRequest):
    history = [{"role": item.role, "content": item.content} for item in body.history]
    try:
        return await service.ask(body.question, body.provider, body.model, body.mode, history, body.model_id, body.use_ai_infra, body.route_mode)
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        route = body.route_mode or ("AI Infra" if body.use_ai_infra else "DeepSeek")
        raise HTTPException(503, f"{route} 推理失败：{error}") from error


@app.post("/internal/infer")
async def internal_infer(body: dict, request: Request):
    if request.headers.get("x-aiinfra-secret", "") != settings.ai_infra_shared_secret:
        raise HTTPException(401, "invalid AI Infra secret")
    provider = DeepSeekProvider(settings.deepseek_base_url, settings.deepseek_api_key, body.get("model", "deepseek-chat"))
    try:
        return {"text": await provider.complete(body.get("prompt", ""), body.get("model", "deepseek-chat"))}
    except Exception as error:
        raise HTTPException(503, f"DeepSeek 调用失败：{error}") from error


@app.get("/api/conversations")
def conversations(): return {"conversations": storage.conversations()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.qa_host, port=settings.qa_port, reload=False)
