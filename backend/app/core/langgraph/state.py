from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Base state schema for all Axolotl agents."""

    messages: Annotated[list, add_messages]
    active_agent: str
    context: dict[str, Any]  # metadata, user_id, session_id, skill_id
