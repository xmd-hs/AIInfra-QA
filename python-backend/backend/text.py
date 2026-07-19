import re
from collections.abc import Iterable

STOP_WORDS = {"的", "了", "和", "是", "在", "我", "有", "就", "不", "都", "一个", "这", "那", "与", "及", "或", "为", "中", "the", "a", "an", "is", "are", "of", "to", "and", "in"}


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n")
    value = re.sub(r"[\t ]+", " ", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def tokenize(value: str) -> list[str]:
    text = normalize_text(value).lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9_+.#-]*", text)
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    return [token for token in tokens if token not in STOP_WORDS]


def chunk_text(value: str, size: int = 650, overlap: int = 100) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?；;])\s*|\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(buffer) + len(sentence) > size and len(buffer) >= size * 0.45:
            chunks.append(buffer.strip())
            buffer = buffer[-overlap:]
        if len(sentence) > size:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            step = max(1, size - overlap)
            chunks.extend(sentence[start:start + size] for start in range(0, len(sentence), step))
        else:
            buffer += (" " if buffer else "") + sentence
    if buffer.strip():
        chunks.append(buffer.strip())
    return list(dict.fromkeys(chunk for chunk in chunks if len(chunk) >= 10))


def split_sentences(value: str) -> Iterable[str]:
    return (part.strip() for part in re.split(r"(?<=[。！？.!?])\s*", normalize_text(value)) if part.strip())
