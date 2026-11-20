"""A scripted stand-in for Gemini so the agent wiring can be tested without an API key.
It looks at which agent is calling (ADK puts the agent's name in the system instruction)
and returns a fixed function call or text, with fake token counts."""
import re

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def _call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


class FakeLlm(BaseLlm):
    model: str = "fake-llm"
    route: str = "research_pipeline"

    async def generate_content_async(self, llm_request, stream=False):
        si = str(llm_request.config.system_instruction or "")
        agent = re.search(r'internal name is "([^"]+)"', si).group(1)
        last = llm_request.contents[-1].parts[0] if llm_request.contents else None
        after_tool = last is not None and last.function_response is not None
        own_turn_done = after_tool and last.function_response.name != "transfer_to_agent"

        if agent == "coordinator":
            part = _call("transfer_to_agent", agent_name=self.route)
        elif agent == "data_analyst":
            part = (types.Part(text="Hemoglobin subunit beta is 147 residues long. [source: proteins.csv]")
                    if own_turn_done else _call("query_protein_table", sql="SELECT length FROM proteins WHERE gene='HBB'"))
        elif agent == "researcher":
            part = (types.Part(text="MDM2 tags p53 for degradation [p53_overview.md].")
                    if own_turn_done else _call("search_literature", query="MDM2 p53 degradation"))
        elif agent == "critic":
            part = _call("exit_loop")
        else:
            part = types.Part(text="ok")

        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1000, candidates_token_count=50, total_token_count=1050),
        )
