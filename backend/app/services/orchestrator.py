from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from app.core.llm.provider import get_chat_model
from app.core.vector_store.client import VectorStoreManager
from app.models.agent import Agent
from app.models.project import Project
from app.services.agent_resolver import resolve_agent_tools


async def orchestrate_project_chat(
    project: Project,
    agents: list[Agent],
    message: str,
    thread_id: str,
    model_override: str | None = None,
    checkpointer=None,
) -> str:
    """Orchestrate a chat using the project's planner/worker pattern.

    If the project has a planner agent and workers, builds a supervisor graph.
    Otherwise falls back to a single agent.
    """
    vector_store = _get_vector_store()

    planner = next((a for a in agents if a.is_planner), None)
    workers = [a for a in agents if not a.is_planner]

    # Fallback: single agent or no agents
    if not workers:
        from app.core.langgraph.graphs.simple_agent import build_simple_agent

        prompt = (
            planner.prompt if planner else project.planner_prompt or "You are a helpful assistant."
        )
        model = planner.model if planner else model_override or project.model
        agent = build_simple_agent(model_name=model, system_prompt=prompt)
        if checkpointer:
            agent = agent  # simple_agent doesn't support checkpointer directly
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content

    # Build worker react agents with their resolved tools
    worker_graphs = []
    for worker in workers:
        worker_tools = await resolve_agent_tools(worker, vector_store)
        worker_model = get_chat_model(model=worker.model or project.model)
        worker_graph = create_react_agent(
            model=worker_model,
            tools=worker_tools,
            name=worker.name,
            prompt=worker.prompt,
        )
        worker_graphs.append(worker_graph)

    # Build supervisor with planner prompt
    planner_prompt = (
        planner.prompt if planner else project.planner_prompt or "You are a helpful assistant."
    )
    planner_model_name = planner.model if planner else model_override or project.model
    supervisor_model = get_chat_model(model=planner_model_name)

    supervisor = create_supervisor(
        agents=worker_graphs,
        model=supervisor_model,
        prompt=planner_prompt,
    )

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    graph = supervisor.compile(**compile_kwargs)

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    return result["messages"][-1].content


_vector_store: VectorStoreManager | None = None


def _get_vector_store() -> VectorStoreManager | None:
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = VectorStoreManager()
        except Exception:
            return None
    return _vector_store
