"""Per-call tracing and LLM metrics.

ADK lets us attach callbacks that run before/after every model call and every tool call.
We use them to record, for each step of each request:
  - which agent made the call, latency, input/output tokens
  - tokens/sec (output tokens / model latency)
  - cost in USD (tokens x price per token)
Records are appended to .store/traces.jsonl (one JSON object per line) and kept in memory,
so one request's full trace can be returned with its answer and /metrics can aggregate them.
"""
import json
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path

from .config import settings


class Tracer:
    def __init__(self, path: Path = settings.traces_path):
        self.path = path
        self._starts: dict[tuple, float] = {}
        self._records: list[dict] = []
        self._lock = threading.Lock()

    # ---------- ADK callbacks (returning None means "don't change anything") ----------
    def before_model(self, callback_context, llm_request):
        self._starts[("llm", callback_context.invocation_id, callback_context.agent_name)] = time.perf_counter()
        return None

    def after_model(self, callback_context, llm_response):
        key = ("llm", callback_context.invocation_id, callback_context.agent_name)
        latency = time.perf_counter() - self._starts.pop(key, time.perf_counter())
        usage = llm_response.usage_metadata
        tokens_in = (usage.prompt_token_count or 0) if usage else 0
        tokens_out = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
        cost = tokens_in / 1e6 * settings.price_in_per_m + tokens_out / 1e6 * settings.price_out_per_m
        self._record(
            type="llm",
            invocation_id=callback_context.invocation_id,
            agent=callback_context.agent_name,
            latency_s=round(latency, 3),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_per_sec=round(tokens_out / latency, 1) if latency > 0 else None,
            cost_usd=round(cost, 6),
        )
        return None

    def before_tool(self, tool, args, tool_context):
        self._starts[("tool", tool_context.invocation_id, tool_context.function_call_id)] = time.perf_counter()
        return None

    def after_tool(self, tool, args, tool_context, tool_response):
        key = ("tool", tool_context.invocation_id, tool_context.function_call_id)
        latency = time.perf_counter() - self._starts.pop(key, time.perf_counter())
        self._record(
            type="tool",
            invocation_id=tool_context.invocation_id,
            agent=tool_context.agent_name,
            tool=tool.name,
            args=args,
            latency_s=round(latency, 3),
            status=tool_response.get("status") if isinstance(tool_response, dict) else None,
        )
        return None

    def callbacks(self) -> dict:
        return dict(
            before_model_callback=self.before_model,
            after_model_callback=self.after_model,
            before_tool_callback=self.before_tool,
            after_tool_callback=self.after_tool,
        )

    # ---------- storage and summaries ----------
    def _record(self, **rec):
        rec["ts"] = time.time()
        with self._lock:
            self._records.append(rec)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def for_request(self, invocation_ids: set[str]) -> list[dict]:
        return [r for r in self._records if r["invocation_id"] in invocation_ids]

    @staticmethod
    def summarize(records: list[dict], wall_time_s: float | None = None) -> dict:
        llm = [r for r in records if r["type"] == "llm"]
        tokens_out = sum(r["tokens_out"] for r in llm)
        llm_time = sum(r["latency_s"] for r in llm)
        return {
            "agents_used": list(dict.fromkeys(r["agent"] for r in llm)),
            "llm_calls": len(llm),
            "tool_calls": sum(r["type"] == "tool" for r in records),
            "tokens_in": sum(r["tokens_in"] for r in llm),
            "tokens_out": tokens_out,
            "tokens_per_sec": round(tokens_out / llm_time, 1) if llm_time else None,
            "cost_usd": round(sum(r["cost_usd"] for r in llm), 6),
            "wall_time_s": round(wall_time_s, 3) if wall_time_s is not None else None,
        }

    @staticmethod
    def aggregate(requests: list[dict]) -> dict:
        """Roll-up across requests for the /metrics endpoint."""
        if not requests:
            return {"requests": 0}
        lat = sorted(r["wall_time_s"] for r in requests)
        per_agent = defaultdict(int)
        for r in requests:
            for a in r["agents_used"]:
                per_agent[a] += 1
        return {
            "requests": len(requests),
            "latency_p50_s": round(statistics.median(lat), 3),
            "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
            "avg_cost_per_request_usd": round(statistics.mean(r["cost_usd"] for r in requests), 6),
            "avg_tokens_per_sec": round(statistics.mean(r["tokens_per_sec"] or 0 for r in requests), 1),
            "avg_llm_calls_per_request": round(statistics.mean(r["llm_calls"] for r in requests), 2),
            "agent_usage": dict(per_agent),
        }


tracer = Tracer()
