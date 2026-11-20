"""Runs one question through the agents and returns the answer together with its trace."""
import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .tracing import Tracer, tracer

APP_NAME = "bioscout"


class BioScout:
    def __init__(self, agent=None):
        if agent is None:
            from .agent import root_agent as agent
        self.sessions = InMemorySessionService()
        self.runner = Runner(app_name=APP_NAME, agent=agent, session_service=self.sessions)
        self.history: list[dict] = []  # per-request summaries, for /metrics

    async def ask(self, question: str, user_id: str = "demo") -> dict:
        # A fresh session per question keeps requests independent (stateless API).
        session = await self.sessions.create_session(app_name=APP_NAME, user_id=user_id)
        message = types.Content(role="user", parts=[types.Part(text=question)])

        t0 = time.perf_counter()
        invocation_ids, last_text = set(), ""
        async for event in self.runner.run_async(user_id=user_id, session_id=session.id, new_message=message):
            invocation_ids.add(event.invocation_id)
            if event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts if not getattr(p, "thought", False))
                if text.strip():
                    last_text = text
        wall = time.perf_counter() - t0

        # The research path leaves its (self-reviewed) answer in state["draft"].
        session = await self.sessions.get_session(app_name=APP_NAME, user_id=user_id, session_id=session.id)
        answer = session.state.get("draft") or last_text

        steps = tracer.for_request(invocation_ids)
        summary = Tracer.summarize(steps, wall)
        self.history.append(summary)
        return {"request_id": str(uuid.uuid4()), "answer": answer.strip(), "metrics": summary, "trace": steps}

    def metrics(self) -> dict:
        return Tracer.aggregate(self.history)
