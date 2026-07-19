import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .text import chunk_text, normalize_text


class Storage:
    def __init__(self, database: Path):
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self):
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
                    characters INTEGER NOT NULL, chunk_count INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, title TEXT NOT NULL,
                    source TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
                    provider TEXT NOT NULL, model TEXT NOT NULL, citations TEXT NOT NULL,
                    timing TEXT NOT NULL, degraded INTEGER NOT NULL, warning TEXT, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            """)

    def add_document(self, title: str, content: str, source: str) -> dict[str, Any]:
        import json
        clean = normalize_text(content)
        pieces = chunk_text(clean)
        document_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        document = {"id": document_id, "title": title.strip(), "source": source, "characters": len(clean), "chunkCount": len(pieces), "createdAt": created_at}
        with self.connect() as db:
            db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)", (document_id, document["title"], source, len(clean), len(pieces), created_at))
            db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", [(str(uuid.uuid4()), document_id, document["title"], source, index, text) for index, text in enumerate(pieces)])
        return document

    def documents(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [{"id": row["id"], "title": row["title"], "source": row["source"], "characters": row["characters"], "chunkCount": row["chunk_count"], "createdAt": row["created_at"]} for row in rows]

    def chunks(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM chunks").fetchall()
        return [{"id": row["id"], "documentId": row["document_id"], "title": row["title"], "source": row["source"], "index": row["chunk_index"], "text": row["text"]} for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self.connect() as db:
            db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cursor = db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cursor.rowcount > 0

    def add_conversation(self, entry: dict[str, Any]):
        import json
        with self.connect() as db:
            db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                str(uuid.uuid4()), entry["question"], entry["answer"], entry["provider"], entry["model"],
                json.dumps(entry["citations"], ensure_ascii=False), json.dumps(entry["timing"]),
                int(entry.get("degraded", False)), entry.get("warning"), datetime.now(timezone.utc).isoformat()
            ))

    def conversations(self, limit: int = 30) -> list[dict[str, Any]]:
        import json
        with self.connect() as db:
            rows = db.execute("SELECT * FROM conversations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": row["id"], "question": row["question"], "answer": row["answer"], "provider": row["provider"], "model": row["model"], "citations": json.loads(row["citations"]), "timing": json.loads(row["timing"]), "degraded": bool(row["degraded"]), "warning": row["warning"], "createdAt": row["created_at"]} for row in rows]

    def stats(self) -> dict[str, int]:
        with self.connect() as db:
            row = db.execute("SELECT COUNT(*) documents, COALESCE(SUM(characters), 0) characters, COALESCE(SUM(chunk_count), 0) chunks FROM documents").fetchone()
            conversations = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        return {"documents": row["documents"], "characters": row["characters"], "chunks": row["chunks"], "conversations": conversations}
