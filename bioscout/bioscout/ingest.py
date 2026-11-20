"""Ingestion pipeline: data/ -> SQLite (structured) + ChromaDB (unstructured and structured).

Run:  python -m bioscout.ingest

1. Structured:   data/structured/*.csv  -> SQLite table `proteins` (for exact SQL lookups)
                                         -> each row also written as a sentence and embedded,
                                            so semantic search can find table facts too
2. Unstructured: data/unstructured/*.md|.txt|.pdf -> clean -> chunk -> embed -> ChromaDB

Chunk ids are deterministic (file name + position), so re-running updates instead of duplicating.
"""
import csv
import re
import sqlite3
import time
from pathlib import Path

from .config import settings
from .stores import Chunk, VectorStore

COLUMNS = ["accession", "name", "gene", "organism", "length", "function", "location"]


def chunk_text(text: str, size: int = settings.chunk_words, overlap: int = settings.chunk_overlap) -> list[str]:
    """Overlapping word windows, so a sentence cut at a boundary still appears whole in one chunk."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, step)]


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"[#*_`>]+", " ", text)  # strip markdown symbols
    return re.sub(r"\s+", " ", text).strip()


def load_structured(csv_dir: Path, db_path: Path) -> list[Chunk]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows += [{c: r.get(c, "") for c in COLUMNS} for r in csv.DictReader(f)]

    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("DROP TABLE IF EXISTS proteins")
        conn.execute(
            "CREATE TABLE proteins (accession TEXT PRIMARY KEY, name TEXT, gene TEXT, organism TEXT, "
            "length INTEGER, function TEXT, location TEXT)"
        )
        conn.executemany(
            f"INSERT OR REPLACE INTO proteins VALUES ({','.join('?' * len(COLUMNS))})",
            [[int(r[c]) if c == "length" and r[c] else r[c] for c in COLUMNS] for r in rows],
        )
    conn.close()

    return [
        Chunk(
            id=f"row:{r['accession']}",
            text=(
                f"{r['name']} ({r['gene']}, UniProt {r['accession']}) from {r['organism']} is {r['length']} "
                f"residues long. Function: {r['function']}. Location: {r['location']}."
            ),
            source=f"proteins.csv:{r['accession']}",
            kind="table_row",
        )
        for r in rows
    ]


def load_unstructured(doc_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(doc_dir.iterdir()):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        for i, piece in enumerate(chunk_text(read_document(path))):
            chunks.append(Chunk(id=f"doc:{path.name}:{i}", text=piece, source=path.name, kind="document"))
    return chunks


def run(data_dir: Path = settings.data_dir, store: VectorStore | None = None) -> dict:
    t0 = time.perf_counter()
    row_chunks = load_structured(data_dir / "structured", settings.sqlite_path)
    doc_chunks = load_unstructured(data_dir / "unstructured")
    store = store or VectorStore()
    store.upsert(row_chunks + doc_chunks)
    stats = {
        "table_rows": len(row_chunks),
        "document_chunks": len(doc_chunks),
        "vectors_in_store": store.count(),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    print(stats)
    return stats


if __name__ == "__main__":
    run()
