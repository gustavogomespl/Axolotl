from langchain_core.tools import BaseTool

from app.config import settings
from deepagents import create_deep_agent


def build_deep_agent(
    model: str | None,
    system_prompt: str,
    subagents: list[dict] | None = None,
    name: str = "main-agent",
    tools: list[BaseTool] | None = None,
    checkpointer=None,
):
    """Build a deep agent with sub-agents, filesystem memory, and tools.

    Note: Axolotl skills (RAG, prompt) are resolved into LangChain tools by
    resolve_agent_tools() before being passed here. DeepAgents' own skill system
    (filesystem-backed) is not used because it requires state seeding that the
    current invocation flow does not provide.

    Args:
        model: LLM model identifier (e.g. "openai:gpt-4.1-mini").
        system_prompt: System prompt for the main agent.
        subagents: List of sub-agent configs, each with:
            - name: str
            - description: str
            - system_prompt: str
            - tools: list[BaseTool]
            - model: str (optional)
        name: Name for the main agent (used in tracing).
        tools: Tools for the main agent itself (includes resolved RAG skills).
        checkpointer: LangGraph checkpointer for persisting state across turns.

    Returns:
        A compiled deep agent graph (supports invoke, ainvoke, stream, astream_events).
    """
    kwargs: dict = {
        "model": model or settings.default_model,
        "system_prompt": system_prompt,
        "subagents": subagents or [],
        "name": name,
        "tools": tools or [],
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    agent = create_deep_agent(**kwargs)
    return agent
