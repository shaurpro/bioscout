"""Tools the agents can call. ADK reads each function's name, type hints and docstring and shows them
to Gemini, so the docstring is effectively the tool's instructions to the model."""
from functools import lru_cache

from .stores import ProteinTable, UnsafeQueryError, VectorStore


@lru_cache(maxsize=1)
def _vector_store() -> VectorStore:
    return VectorStore()


def search_literature(query: str) -> dict:
    """Semantic search over the ingested research notes/papers and protein records.

    Use for explanations: mechanisms, functions, diseases, methods, background.
    Args:
        query: a focused natural-language search query.
    Returns:
        dict with `passages`, each having text, source (cite this) and similarity score.
    """
    hits = _vector_store().search(query)
    return {"status": "ok", "passages": hits} if hits else {"status": "no_results", "passages": []}


def query_protein_table(sql: str) -> dict:
    """Run ONE read-only SQLite SELECT on the table `proteins`.

    Columns: accession, name, gene, organism, length (integer, residues), function, location.
    Use for exact facts, filters, counts, sorting and comparisons, e.g.
    SELECT name, length FROM proteins WHERE organism = 'Homo sapiens' ORDER BY length DESC LIMIT 3
    Args:
        sql: the SELECT statement.
    Returns:
        dict with `rows` (max 50) or an `error` message you can use to fix the query.
    """
    try:
        rows = ProteinTable().run_readonly(sql)
        return {"status": "ok", "rows": rows, "source": "proteins.csv"}
    except UnsafeQueryError as e:
        return {"status": "rejected", "error": str(e)}
    except Exception as e:  # e.g. a SQL syntax error: return it so the agent can retry (ReAct)
        return {"status": "error", "error": str(e)}
