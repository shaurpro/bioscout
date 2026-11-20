"""Small evaluation set: checks routing, answer keywords, citations, latency and cost.
Usage:  python scripts/eval.py      (writes .store/eval_report.json)
Add your own cases to CASES as you add data."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bioscout.config import settings  # noqa: E402
from bioscout.service import BioScout  # noqa: E402

CASES = [
    {"q": "How many residues long is human hemoglobin subunit beta?", "route": "data_analyst", "expect": ["147"]},
    {"q": "Which is the longest protein in the table and how long is it?", "route": "data_analyst", "expect": ["BRCA1", "1863"]},
    {"q": "List the proteins from organisms other than humans.", "route": "data_analyst", "expect": ["Albumin", "Green fluorescent"]},
    {"q": "How does MDM2 keep p53 levels low, and what changes after DNA damage?", "route": "research_pipeline", "expect": ["MDM2", "phosphorylat"]},
    {"q": "What mutation causes sickle-cell disease and why does it make cells sickle?", "route": "research_pipeline", "expect": ["Glu6Val", "valine"]},
    {"q": "How does GFP form its chromophore without a cofactor?", "route": "research_pipeline", "expect": ["Ser65", "Gly67"]},
    {"q": "Compare cryo-EM and X-ray crystallography for membrane proteins.", "route": "research_pipeline", "expect": ["crystal", "membrane"]},
    {"q": "Which drugs target EGFR in lung cancer and which one covers T790M?", "route": "research_pipeline", "expect": ["osimertinib"]},
]


async def main():
    bot, rows = BioScout(), []
    for case in CASES:
        r = await bot.ask(case["q"])
        text = r["answer"].lower()
        rows.append({
            "question": case["q"],
            "routed_correctly": case["route"] in r["metrics"]["agents_used"]
                or (case["route"] == "research_pipeline" and "researcher" in r["metrics"]["agents_used"]),
            "keyword_hit": all(k.lower() in text for k in case["expect"]),
            "has_citation": "[" in r["answer"],
            **{k: r["metrics"][k] for k in ("wall_time_s", "cost_usd", "tokens_per_sec", "llm_calls")},
        })
        print(rows[-1])
    n = len(rows)
    report = {
        "routing_accuracy": sum(r["routed_correctly"] for r in rows) / n,
        "answer_keyword_accuracy": sum(r["keyword_hit"] for r in rows) / n,
        "citation_rate": sum(r["has_citation"] for r in rows) / n,
        **bot.metrics(),
        "cases": rows,
    }
    out = settings.store_dir / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), f"\nSaved to {out}")


asyncio.run(main())
