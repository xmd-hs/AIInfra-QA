from pathlib import Path


ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".html", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".env", ".txt", ".sql", ".c", ".cc", ".cpp", ".h", ".hpp",
}
IGNORED_PARTS = {".git", "node_modules", "dist", "build", "build-windows", ".venv", "__pycache__", ".pytest_cache"}


class CodeWorkspace:
    def __init__(self, root: Path):
        self.set_root(root)

    def set_root(self, root: Path) -> None:
        candidate = root.expanduser().resolve()
        if not candidate.is_dir():
            raise ValueError("工作目录不存在或不是文件夹")
        self.root = candidate

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative.replace("\\", "/")).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("路径超出代码工作区")
        if candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError("不支持编辑这种文件类型")
        return candidate

    def files(self, limit: int = 800) -> list[str]:
        result = []
        for path in self.root.rglob("*"):
            if len(result) >= limit:
                break
            relative = path.relative_to(self.root)
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS and not any(part in IGNORED_PARTS for part in relative.parts):
                result.append(relative.as_posix())
        return sorted(result)

    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise ValueError("文件不存在")
        if path.stat().st_size > 500_000:
            raise ValueError("文件超过 500KB")
        return path.read_text(encoding="utf-8")

    def write(self, relative: str, content: str) -> None:
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        first_newline = value.find("\n")
        if first_newline >= 0:
            value = value[first_newline + 1:]
        if value.endswith("```"):
            value = value[:-3]
    return value.rstrip() + "\n"
