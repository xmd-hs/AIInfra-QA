from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    qa_host: str = "0.0.0.0"
    qa_port: int = 8000
    qa_database: str = "data/knowledge.db"
    max_context_chunks: int = 6
    min_search_score: float = 0.05

    ai_infra_base_url: str = "http://127.0.0.1:8080"
    ai_infra_shared_secret: str = "development-only"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    ai_infra_model: str = "demo"
    code_workspace_root: str = ".."

    @property
    def database_path(self) -> Path:
        path = Path(self.qa_database)
        return path if path.is_absolute() else ROOT / path

    @property
    def code_workspace_path(self) -> Path:
        path = Path(self.code_workspace_root)
        return (path if path.is_absolute() else ROOT / path).resolve()


settings = Settings()
