# BioScout — Multi-Agent RAG Assistant for Protein Research

BioScout answers questions about proteins. It searches two kinds of data: a **table** of protein facts
(structured) and a folder of **research notes/papers** (unstructured). Several AI agents built with Google's
**Agent Development Kit (ADK)** and **Gemini** work together. A coordinator decides who handles the question,
a researcher gathers evidence step by step, and a critic checks the answer before it is returned.
Every answer comes with citations and with metrics: tokens, tokens/sec, latency and cost per request.
It runs as a REST API that can be deployed to **Google Cloud Run**.

> **One-line pitch:** "A small, end-to-end version of what an enterprise AI assistant does: ingest company
> data, retrieve the right evidence, let specialised agents reason over it, check the answer, and measure
> what every request costs."

---

## 1. What it does (example)

```
POST /ask  {"question": "How does MDM2 keep p53 levels low, and what changes after DNA damage?"}

{
  "answer": "MDM2 is an E3 ubiquitin ligase that tags p53 for degradation by the proteasome [p53_overview.md].
             After DNA damage, ATM and ATR phosphorylate p53, which blocks MDM2 binding, so p53 builds up ... ",
  "metrics": {
    "agents_used": ["coordinator", "researcher", "critic"],
    "llm_calls": 5, "tool_calls": 2,
    "tokens_in": 6120, "tokens_out": 410, "tokens_per_sec": 95.3,
    "cost_usd": 0.0029, "wall_time_s": 6.8
  }
}
```
*(Numbers above are only an illustration of the format. Run `scripts/eval.py` to get your real ones.)*

---

## 2. Architecture

```
                 ┌──────────── INGESTION  (python -m bioscout.ingest) ─────────────┐
 data/structured/proteins.csv ──► SQLite table `proteins` ──────────────┐          │
            │                                                           │          │
            └──► each row → sentence ─┐                                 │          │
 data/unstructured/*.md|pdf ─► clean ─► chunk (180 words, 40 overlap)   │          │
                                      └─► Gemini embeddings ─► ChromaDB │          │
                 └──────────────────────────────────────────────────────┼──────────┘
                                                                        │
 QUESTION ─► FastAPI /ask ─► ADK Runner                                 │
                               │                                        │
                        ┌──────▼──────┐   hierarchical delegation       │
                        │ coordinator │──────────────┐                  │
                        └──────┬──────┘              │                  │
               exact facts     │                     │ explanations     │
                        ┌──────▼───────┐   ┌─────────▼──────────┐       │
                        │ data_analyst │   │ research_pipeline  │       │
                        │  SQL tool ───┼───┤ 1. researcher      │ ReAct │
                        └──────────────┘   │    (both tools)    │◄──────┘
                                           │ 2. self_review loop│
                                           │    critic ⇄ reviser│ self-reflection
                                           └────────────────────┘
        Tracer callbacks on every model + tool call ─► traces.jsonl, /metrics
```

### The three agent patterns, in plain words
| Pattern | Where | What it means here |
|---|---|---|
| **Hierarchical delegation** | `coordinator` | A manager agent doesn't answer; it hands the question to the right specialist (ADK `sub_agents` + automatic `transfer_to_agent`). |
| **ReAct** (Reason + Act) | `researcher`, `data_analyst` | The agent thinks about what is missing, calls a tool, reads the result and repeats. If a SQL query fails, the error goes back to the agent so it can fix the query. |
| **Self-reflection** | `self_review` loop | A `critic` agent checks the draft: does it answer everything, does each sentence have a source, is anything made up? If it passes, the critic calls `exit_loop`. If not, a `reviser` fixes it. At most 2 rounds, to cap cost. |

### State management
Agents share data through ADK **session state**: the researcher saves its answer to `state["draft"]`,
the critic reads `{draft}` and writes `state["critique"]`, and the reviser overwrites `draft`.
Each API request gets a fresh session, so requests never leak into each other.

---

## 3. Code structure

```
bioscout/
├── bioscout/
│   ├── config.py       All settings (models, chunk size, prices) from environment variables
│   ├── embeddings.py   Text → vectors with Gemini embeddings (+ an offline fake for tests)
│   ├── stores.py       VectorStore (ChromaDB) and ProteinTable (read-only SQLite with guard rails)
│   ├── ingest.py       Data pipeline: CSV → SQLite + row sentences; docs → clean → chunk → embed
│   ├── tools.py        The 2 tools agents can call: search_literature, query_protein_table
│   ├── agent.py        The 5 agents and how they are connected (the core of the project)
│   ├── tracing.py      Callbacks that log tokens, latency, tokens/sec and cost for every step
│   ├── service.py      Runs one question through the agents, returns answer + metrics + trace
│   └── api.py          FastAPI: POST /ask, GET /metrics, GET /healthz
├── data/
│   ├── structured/proteins.csv   10 proteins (accession, gene, organism, length, function, location)
│   └── unstructured/*.md         5 short notes (p53, hemoglobin, GFP, EGFR, structure methods)
├── scripts/
│   ├── ask.py          Ask from the terminal
│   └── eval.py         8-question evaluation: routing, accuracy, citations, latency, cost
├── tests/              9 offline tests (fake LLM + fake embeddings, no API key needed)
├── docs/discovery_template.md   Template for a requirements/discovery session with a user
├── Dockerfile          For Cloud Run
└── requirements.txt
```

**Read the code in this order:** `tools.py` → `agent.py` → `service.py` → `tracing.py` → `ingest.py`.
`agent.py` is about 120 lines and holds the whole multi-agent design.

---

## 4. Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # paste a Gemini API key from https://aistudio.google.com/apikey

python -m pytest -q           # 9 offline tests, no key needed
python -m bioscout.ingest     # build SQLite + ChromaDB from data/
python scripts/ask.py "Why is TP53 called the guardian of the genome?" --trace
python scripts/eval.py        # metrics for your resume, saved to .store/eval_report.json

uvicorn bioscout.api:app --reload     # then open http://127.0.0.1:8000/docs
adk web                               # optional: ADK's chat UI, shows every agent step and event
```

### Deploy to Google Cloud Run
```bash
gcloud auth login && gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com
gcloud run deploy bioscout --source . --region asia-south1 --allow-unauthenticated \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1
```
Notes: with Vertex AI, give the Cloud Run service account the **Vertex AI User** IAM role (your
"Service Accounts and IAM Roles" skill badge covers this). The demo builds its vector store inside the image.
For real data, build it once and keep it in Cloud Storage, or use a managed vector DB (Vertex AI Vector Search,
or AlloyDB/Cloud SQL with pgvector). Delete the service when you're done to avoid charges.

### Use your own data
- Real protein table: download a TSV/CSV from UniProt (uniprot.org → search → Download, choose the columns
  above) and save it as `data/structured/proteins.csv` with the same column names.
- Real papers: put open-access PDFs (e.g. from PubMed Central) in `data/unstructured/`. Then run ingest again.

---

## 5. Design decisions 

- **Why two stores?** Numbers like "length = 1863" should come from SQL, not from fuzzy vector search.
  Explanations live in text and need semantic search. Table rows are *also* embedded as sentences, so a
  vague question can still find them.
- **Why chunk overlap?** A sentence cut at a chunk boundary still appears whole in the next chunk.
- **Why RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY task types?** Gemini embeddings are tuned differently for
  stored passages and for short search questions, which improves matching.
- **Why a read-only SQL guard?** The model writes SQL, so it must not be able to change data. The guard allows
  one SELECT only, blocks write keywords, opens SQLite in read-only mode and caps rows. Errors go back to the
  agent so it can fix its query.
- **Why cap the critic loop at 2?** Each round costs tokens. The trace shows the cost of every extra round.
- **Why log tokens/sec and cost per request?** LLM apps fail in new ways: slow generation, prompts growing
  across agent hand-offs, runaway loops. Per-step traces show which agent is slow or expensive.
- **Known limitation / next step:** every agent transfer re-sends the prompt. ADK's context caching
  (`context_cache_config`) would cut input-token cost. ADK is replacing `SequentialAgent`/`LoopAgent` with a
  newer `Workflow` API, so check the ADK docs when upgrading.

## 6. Ideas to extend
Hybrid search (keyword + vector) · re-ranking · Vertex AI Vector Search · streaming answers ·
OpenTelemetry/Cloud Trace export · an LLM-as-judge in `eval.py` · a third agent that queries the live UniProt API.
