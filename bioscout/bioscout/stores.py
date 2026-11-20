"""The two data stores the agents read from.

- VectorStore: ChromaDB collection of text chunks + embeddings  -> semantic search (unstructured data)
- ProteinTable: SQLite table built from the CSV                 -> exact, read-only SQL (structured data)
"""
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .config import settings
from .embeddings import Embedder, get_embedder


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    kind: str  # "document" or "table_row"


class VectorStore:
    def __init__(self, path: Path = settings.chroma_path, embedder: Embedder | None = None):
        path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedder()
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection("bioscout", metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[{"source": c.source, "kind": c.kind} for c in chunks],
        )

    def search(self, query: str, k: int = settings.top_k) -> list[dict]:
        qvec = self.embedder.embed([query], task_type="RETRIEVAL_QUERY")[0]
        res = self.collection.query(query_embeddings=[qvec], n_results=k)
        return [
            {"text": doc, "source": meta["source"], "kind": meta["kind"], "score": round(1 - dist, 3)}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def count(self) -> int:
        return self.collection.count()


class UnsafeQueryError(ValueError):
    pass


class ProteinTable:
    MAX_ROWS = 50
    _FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b", re.I)

    def __init__(self, path: Path = settings.sqlite_path):
        self.path = path

    def run_readonly(self, sql: str) -> list[dict]:
        """Guard rails: one SELECT statement, no write keywords, DB opened read-only, row cap."""
        stmt = sql.strip().rstrip(";")
        if ";" in stmt:
            raise UnsafeQueryError("Only one statement is allowed.")
        if not re.match(r"^(select|with)\b", stmt, re.I) or self._FORBIDDEN.search(stmt):
            raise UnsafeQueryError("Only read-only SELECT queries are allowed.")
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(stmt).fetchmany(self.MAX_ROWS)
            return [dict(r) for r in rows]
        finally:
            conn.close()
