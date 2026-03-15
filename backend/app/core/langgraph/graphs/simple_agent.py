from langgraph.graph.state import CompiledStateGraph as CompiledGraph
from langgraph.prebuilt import create_react_agent

from app.core.llm.provider import get_chat_model


def build_simple_agent(
    model_name: str | None = None,
    tools: list | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> CompiledGraph:
    """Build a simple ReAct agent with optional tools."""
    model = get_chat_model(model=model_name)
    return create_react_agent(
        model=model,
        tools=tools or [],
        prompt=system_prompt,
    )
