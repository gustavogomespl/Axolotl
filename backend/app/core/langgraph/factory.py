from typing import Any, Literal

from pydantic import BaseModel

from app.core.langgraph.graphs.simple_agent import build_simple_agent
from app.core.langgraph.graphs.supervisor import build_supervisor_graph
from app.core.langgraph.graphs.swarm import build_swarm_graph
from app.core.langgraph.subgraphs.registry import SubgraphRegistry
from app.core.langgraph.tools.registry import ToolRegistry


class AgentConfig(BaseModel):
    """Configuration for an individual agent within a graph."""

    name: str
    prompt: str
    model: str | None = None
    tool_names: list[str] = []


class GraphConfig(BaseModel):
    """Configuration for building a LangGraph dynamically."""

    name: str
    pattern: Literal["supervisor", "swarm", "single"]
    model: str | None = None
    system_prompt: str = "You are a helpful assistant."
    agents: list[AgentConfig] = []
    subgraph_names: list[str] = []
    tool_names: list[str] = []


class GraphFactory:
    """Build LangGraph graphs dynamically from configuration."""

    @staticmethod
    def build(config: GraphConfig) -> Any:
        """Build a compiled graph from a GraphConfig.

        Args:
            config: Graph configuration specifying pattern, agents, tools, etc.

        Returns:
            Compiled LangGraph graph.
        """
        match config.pattern:
            case "supervisor":
                agents_with_tools = []
                for agent_cfg in config.agents:
                    tools = ToolRegistry.get_tools(agent_cfg.tool_names or None)
                    agents_with_tools.append({
                        "name": agent_cfg.name,
                        "prompt": agent_cfg.prompt,
                        "model": agent_cfg.model,
                        "tools": tools,
                    })
                return build_supervisor_graph(
                    agents=agents_with_tools,
                    model=config.model,
                    system_prompt=config.system_prompt,
                )

            case "swarm":
                agents_with_tools = []
                for agent_cfg in config.agents:
                    tools = ToolRegistry.get_tools(agent_cfg.tool_names or None)
                    agents_with_tools.append({
                        "name": agent_cfg.name,
                        "prompt": agent_cfg.prompt,
                        "model": agent_cfg.model,
                        "tools": tools,
                    })
                return build_swarm_graph(
                    agents=agents_with_tools,
                    model=config.model,
                )

            case "single":
                tools = ToolRegistry.get_tools(config.tool_names or None)
                return build_simple_agent(
                    model_name=config.model,
                    tools=tools,
                    system_prompt=config.system_prompt,
                )

            case _:
                raise ValueError(f"Unknown pattern: {config.pattern}")
