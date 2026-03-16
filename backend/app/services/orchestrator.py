from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from typing_extensions import TypedDict

from app.core.llm.provider import get_chat_model
from app.core.vector_store.client import VectorStoreManager
from app.models.agent import Agent
from app.models.project import Project
from app.services.agent_resolver import resolve_agent_tools


class ProjectState(TypedDict):
    """Extended LangGraph state with project-level tracking.

    Persisted via Redis checkpointer across conversation turns.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    todo_list: list[dict]  # [{"id": "1", "task": "...", "status": "pending|done", "agent": "..."}]
    completed_tasks: list[dict]  # Finished tasks (history)
    files: list[dict]  # [{"name": "report.csv", "content": "...", "agent": "..."}]
    context: dict  # Free-form metadata


TODO_INSTRUCTIONS = """

## Task Management

You have access to a todo_list in the conversation state. Use it to:
1. BEFORE starting work, review the current todo_list and completed_tasks
2. Break complex requests into sub-tasks and add them to todo_list
3. When delegating to a worker, note which agent is handling each task
4. After a worker completes, mark the task as done by moving it to completed_tasks
5. Use completed_tasks as context - don't repeat work that's already done

When updating state, return the updated todo_list and completed_tasks in your response.
Format each task as: {"id": "<unique>", "task": "<description>", "status": "pending|done", "agent": "<agent_name>"}

If agents produce files (code, reports, data), add them to the files list.
Format each file as: {"name": "<filename>", "content": "<content>", "agent": "<agent_name>"}
"""


class OrchestratorResult:
    """Result from orchestrate_project_chat with state fields."""

    def __init__(self, content: str, state: dict[str, Any]):
        self.content = content
        self.todo_list = state.get("todo_list", [])
        self.completed_tasks = state.get("completed_tasks", [])
        self.files = state.get("files", [])


async def orchestrate_project_chat(
    project: Project,
    agents: list[Agent],
    message: str,
    thread_id: str,
    model_override: str | None = None,
    checkpointer=None,
) -> OrchestratorResult:
    """Orchestrate a chat using the project's planner/worker pattern.

    Returns OrchestratorResult with response content and state (todo_list, files, etc.).
    State is persisted via Redis checkpointer across turns.
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
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return OrchestratorResult(content=result["messages"][-1].content, state={})

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

    # Build supervisor with enriched planner prompt + ProjectState
    base_prompt = (
        planner.prompt if planner else project.planner_prompt or "You are a helpful assistant."
    )
    enriched_prompt = base_prompt + TODO_INSTRUCTIONS

    planner_model_name = planner.model if planner else model_override or project.model
    supervisor_model = get_chat_model(model=planner_model_name)

    supervisor = create_supervisor(
        agents=worker_graphs,
        model=supervisor_model,
        prompt=enriched_prompt,
        state_schema=ProjectState,
    )

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    graph = supervisor.compile(**compile_kwargs)

    result = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": message}],
            "todo_list": [],
            "completed_tasks": [],
            "files": [],
            "context": {},
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    return OrchestratorResult(content=result["messages"][-1].content, state=result)


_vector_store: VectorStoreManager | None = None


def _get_vector_store() -> VectorStoreManager | None:
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = VectorStoreManager()
        except Exception:
            return None
    return _vector_store
