from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from app.core.llm.provider import get_chat_model


def build_supervisor_graph(
    agents: list[dict],
    model: str | None = None,
    system_prompt: str = "You are a supervisor that routes tasks to specialized workers.",
) -> object:
    """Build a supervisor graph with hierarchical delegation.

    Args:
        agents: List of agent configs, each with:
            - name: str
            - tools: list[BaseTool]
            - prompt: str
            - model: str | None (optional, overrides default)
        model: Default model for supervisor and workers.
        system_prompt: System prompt for the supervisor.

    Returns:
        Compiled supervisor graph.
    """
    workers = []
    for agent_config in agents:
        worker_model = get_chat_model(model=agent_config.get("model", model))
        worker = create_react_agent(
            model=worker_model,
            tools=agent_config.get("tools", []),
            prompt=agent_config["prompt"],
            name=agent_config["name"],
        )
        workers.append(worker)

    supervisor_model = get_chat_model(model=model)
    supervisor = create_supervisor(
        agents=workers,
        model=supervisor_model,
        prompt=system_prompt,
    )

    return supervisor.compile()
