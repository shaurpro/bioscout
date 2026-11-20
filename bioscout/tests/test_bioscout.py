import asyncio

import pytest

from bioscout import ingest
from bioscout.agent import build_root_agent
from bioscout.service import BioScout
from bioscout.stores import ProteinTable, UnsafeQueryError
from bioscout.tools import query_protein_table, search_literature
from fake_llm import FakeLlm


@pytest.fixture(scope="session", autouse=True)
def ingested():
    return ingest.run()


def test_chunking_overlaps():
    words = " ".join(str(i) for i in range(500))
    chunks = ingest.chunk_text(words, size=100, overlap=20)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_ingest_counts(ingested):
    assert ingested["table_rows"] == 10
    assert ingested["document_chunks"] >= 5


def test_sql_tool_reads_structured_data():
    out = query_protein_table("SELECT name, length FROM proteins ORDER BY length DESC LIMIT 1")
    assert out["rows"][0]["length"] == 1863


@pytest.mark.parametrize("bad", ["DROP TABLE proteins", "SELECT 1; DELETE FROM proteins", "UPDATE proteins SET length=1"])
def test_sql_guard_rails(bad):
    with pytest.raises(UnsafeQueryError):
        ProteinTable().run_readonly(bad)
    assert query_protein_table(bad)["status"] == "rejected"


def test_vector_search_finds_relevant_document():
    hits = search_literature("sickle cell valine beta chain")["passages"]
    # top hit should be about hemoglobin beta (the table row or the document); both kinds are searchable
    assert hits[0]["source"] in {"proteins.csv:P68871", "hemoglobin_cooperativity.md"}
    assert "hemoglobin_cooperativity.md" in {h["source"] for h in hits}


def test_research_route_end_to_end():
    bot = BioScout(build_root_agent(FakeLlm(route="research_pipeline")))
    r = asyncio.run(bot.ask("How does MDM2 control p53?"))
    assert "MDM2" in r["answer"] and "[p53_overview.md]" in r["answer"]
    m = r["metrics"]
    assert m["agents_used"][:3] == ["coordinator", "researcher", "researcher"] or "critic" in m["agents_used"]
    assert m["tool_calls"] >= 2 and m["cost_usd"] > 0 and m["tokens_out"] > 0
    assert any(s["type"] == "tool" and s["tool"] == "search_literature" for s in r["trace"])


def test_data_route_end_to_end():
    bot = BioScout(build_root_agent(FakeLlm(route="data_analyst")))
    r = asyncio.run(bot.ask("How long is HBB?"))
    assert "147" in r["answer"]
    assert "data_analyst" in r["metrics"]["agents_used"]
    assert bot.metrics()["requests"] == 1
