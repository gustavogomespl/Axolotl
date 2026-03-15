from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool

from app.core.llm.provider import get_chat_model


def build_swarm_graph(
    agents: list[dict],
    model: str | None = None,
) -> object:
    """Build a decentralized swarm graph with peer-to-peer handoffs.

    Args:
        agents: List of agent configs, each with:
            - name: str
            - tools: list[BaseTool]
            - prompt: str
            - model: str | None (optional)
        model: Default model for all agents.

    Returns:
        Compiled swarm graph.
    """
    agent_instances = []

    for agent_config in agents:
        # Create handoff tools to all other agents
        handoff_tools = [
            create_handoff_tool(agent_name=other["name"])
            for other in agents
            if other["name"] != agent_config["name"]
        ]

        agent_model = get_chat_model(model=agent_config.get("model", model))
        agent = create_react_agent(
            model=agent_model,
            tools=agent_config.get("tools", []) + handoff_tools,
            prompt=agent_config["prompt"],
            name=agent_config["name"],
        )
        agent_instances.append(agent)

    swarm = create_swarm(agents=agent_instances)
    return swarm.compile()
