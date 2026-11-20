"""The multi-agent system, built with Google ADK.

    coordinator  (LlmAgent, routes the question: hierarchical delegation)
    |-- data_analyst        (LlmAgent + SQL tool: exact facts from the structured table)
    `-- research_pipeline   (SequentialAgent)
        |-- researcher      (LlmAgent, ReAct loop over both tools, writes state["draft"])
        `-- self_review     (LoopAgent, max 2 rounds: self-reflection)
            |-- critic      (checks the draft; calls exit_loop if it is good)
            `-- reviser     (fixes the draft using the critique, may search again)

`root_agent` is the name ADK looks for, so `adk web` (run from the project root) can also open it.
"""
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import exit_loop

from .config import settings
from .tools import query_protein_table, search_literature
from .tracing import tracer

COORDINATOR = """You are the coordinator of BioScout, a protein research assistant.
Do not answer domain questions yourself. Delegate:
- data_analyst: exact facts or numbers about specific proteins (length, organism, gene, counts, rankings).
- research_pipeline: explanations, mechanisms, diseases, methods, or anything needing documents.
For greetings or questions outside protein biology, reply briefly yourself."""

DATA_ANALYST = """Answer using the query_protein_table tool. Write one SQLite SELECT for the question.
If the tool returns an error, fix the SQL and try again (at most 3 attempts).
Answer in 1-3 sentences and end with [source: proteins.csv]. Never guess values that are not in the rows."""

RESEARCHER = """Answer the user's question using ONLY evidence from tools. Work in a loop:
Thought (what is missing?) -> Action (call a tool) -> Observation (read the result) -> repeat.
- search_literature for explanations; query_protein_table for exact numbers.
- Use at most 4 tool calls, then write the answer.
- Cite every factual sentence with its source in square brackets, e.g. [p53_overview.md].
- If the evidence does not cover something, say so instead of guessing.
Output only the final answer (under 200 words)."""

CRITIC = """You review a draft answer written for the user's question in this conversation.

Draft:
{draft}

Check: (1) does it answer every part of the question, (2) does every factual sentence have a [source],
(3) does it claim anything that is not supported by the tool results above.
If all three pass, call the exit_loop tool and output nothing else.
Otherwise output a short numbered list of concrete fixes."""

REVISER = """Improve the draft using the reviewer's fixes.

Draft:
{draft}

Fixes:
{critique}

You may call search_literature if a fix needs more evidence. Keep the [source] citations.
Output only the improved answer."""


def build_root_agent(model=settings.llm_model) -> LlmAgent:
    """`model` is a model name, or any ADK BaseLlm (the tests pass in a fake one)."""
    cb = tracer.callbacks()

    data_analyst = LlmAgent(
        name="data_analyst",
        model=model,
        description="Exact facts and numbers from the structured protein table via SQL.",
        instruction=DATA_ANALYST,
        tools=[query_protein_table],
        **cb,
    )
    researcher = LlmAgent(
        name="researcher",
        model=model,
        description="Gathers evidence with tools and writes a cited draft.",
        instruction=RESEARCHER,
        tools=[search_literature, query_protein_table],
        output_key="draft",  # final text is saved to session state["draft"]
        **cb,
    )
    critic = LlmAgent(
        name="critic",
        model=model,
        description="Reviews the draft for completeness, citations and grounding.",
        instruction=CRITIC,
        tools=[exit_loop],
        output_key="critique",
        **cb,
    )
    reviser = LlmAgent(
        name="reviser",
        model=model,
        description="Rewrites the draft to address the critique.",
        instruction=REVISER,
        tools=[search_literature],
        output_key="draft",  # overwrites the draft with the improved version
        **cb,
    )
    research_pipeline = SequentialAgent(
        name="research_pipeline",
        description="Explanations and mechanisms: researches the documents, then self-reviews the answer.",
        sub_agents=[researcher, LoopAgent(name="self_review", max_iterations=2, sub_agents=[critic, reviser])],
    )
    return LlmAgent(
        name="coordinator",
        model=model,
        description="Routes each question to the right specialist agent.",
        instruction=COORDINATOR,
        sub_agents=[data_analyst, research_pipeline],
        **cb,
    )


root_agent = build_root_agent()
