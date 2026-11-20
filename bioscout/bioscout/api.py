"""REST API (FastAPI). Run locally:  uvicorn bioscout.api:app --reload
Endpoints: POST /ask   GET /metrics   GET /healthz"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .service import BioScout

app = FastAPI(title="BioScout", description="Multi-agent RAG assistant for protein research")
bot = BioScout()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    include_trace: bool = False


@app.post("/ask")
async def ask(req: AskRequest):
    result = await bot.ask(req.question)
    if not req.include_trace:
        result.pop("trace")
    return result


@app.get("/metrics")
def metrics():
    return bot.metrics()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
