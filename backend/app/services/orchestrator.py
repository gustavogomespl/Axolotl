from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from typing_extensions import TypedDict

from app.core.langgraph.graphs.deep_agent import build_deep_agent
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


DEEP_AGENT_TODO_INSTRUCTIONS = """

## Task Management

You have access to a todos list via the write_todos tool. Use it to:
1. BEFORE starting work, review current todos
2. Break complex requests into sub-tasks
3. When delegating to a sub-agent, note which agent handles each task
4. After a sub-agent completes, mark the task as completed
5. Use completed tasks as context - don't repeat work already done

Format each task as: {"content": "<description>", "status": "pending|in_progress|completed"}

If you need to produce files, use the filesystem tools to write them.
"""


def _normalize_deep_agent_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Convert DeepAgents state shape to ProjectState-compatible shape.

    DeepAgents uses different keys/shapes:
    - "todos": list of {"content": "...", "status": "pending"|"in_progress"|"done"}
              or plain strings
    - "files": dict {path: content} instead of list[dict]
    """
    todos = raw_state.get("todos", [])
    todo_list: list[dict] = []
    completed_tasks: list[dict] = []

    for i, item in enumerate(todos):
        if isinstance(item, dict):
            # DeepAgents todo middleware uses "content" key; Axolotl uses "task"
            task_entry = {
                "id": item.get("id", str(i)),
                "task": item.get("content", item.get("task", str(item))),
                "status": item.get("status", "done"),
                "agent": item.get("agent", "deep-agent"),
            }
        else:
            task_entry = {
                "id": str(i),
                "task": str(item),
                "status": "done",
                "agent": "deep-agent",
            }

        if task_entry["status"] in ("done", "completed"):
            task_entry["status"] = "done"  # normalize to Axolotl convention
            completed_tasks.append(task_entry)
        else:
            todo_list.append(task_entry)

    # Normalize files: dict {path: FileData|str} -> list[dict]
    # DeepAgents FileData is {"content": [lines...], "created_at": ..., "updated_at": ...}
    raw_files = raw_state.get("files", {})
    files: list[dict] = []
    if isinstance(raw_files, dict):
        for name, value in raw_files.items():
            if isinstance(value, dict) and "content" in value:
                # FileData: join content lines into text
                content_lines = value["content"]
                text = "\n".join(content_lines) if isinstance(content_lines, list) else str(content_lines)
            else:
                text = str(value)
            files.append({"name": name, "content": text, "agent": "deep-agent"})
    elif isinstance(raw_files, list):
        files = raw_files

    return {
        "todo_list": todo_list,
        "completed_tasks": completed_tasks,
        "files": files,
    }


def _extract_text_content(content: Any) -> str:
    """Extract text from message content that may be a string or list of content blocks.

    Some LLM providers (via DeepAgents) return content as a list of blocks like:
    [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)
    return str(content)


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

    orchestration_mode = getattr(project, "orchestration_mode", "supervisor")

    # Fallback: single agent, no agents, or explicit simple mode
    if not workers or orchestration_mode == "simple":
        from app.core.langgraph.graphs.simple_agent import build_simple_agent

        prompt = (
            planner.prompt if planner else project.planner_prompt or "You are a helpful assistant."
        )
        model = (
            (planner.model if planner and planner.model else None)
            or model_override
            or project.model
        )
        agent = build_simple_agent(model_name=model, system_prompt=prompt)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return OrchestratorResult(content=_extract_text_content(result["messages"][-1].content), state={})

    # Common: resolve planner prompt and model
    base_prompt = (
        planner.prompt if planner else project.planner_prompt or "You are a helpful assistant."
    )
    planner_model_name = (
        (planner.model if planner and planner.model else None)
        or model_override
        or project.model
    )

    # --- Deep Agent mode ---
    if orchestration_mode == "deep_agent":
        enriched_prompt = base_prompt + DEEP_AGENT_TODO_INSTRUCTIONS
        subagents = []
        for worker in workers:
            # resolve_agent_tools handles RAG skills as retriever tools
            worker_tools = await resolve_agent_tools(worker, vector_store)
            subagent_config: dict[str, Any] = {
                "name": worker.name,
                "description": worker.description or worker.prompt[:200],
                "system_prompt": worker.prompt,
                "tools": worker_tools,
            }
            if worker.model or project.model:
                subagent_config["model"] = worker.model or project.model
            subagents.append(subagent_config)

        # Resolve planner's own tools (includes RAG skills as retriever tools)
        planner_tools = await resolve_agent_tools(planner, vector_store) if planner else []

        agent = build_deep_agent(
            model=planner_model_name,
            system_prompt=enriched_prompt,
            subagents=subagents,
            tools=planner_tools,
            checkpointer=checkpointer,
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        normalized = _normalize_deep_agent_state(result)
        return OrchestratorResult(content=_extract_text_content(result["messages"][-1].content), state=normalized)

    # --- Supervisor mode (default) ---
    enriched_prompt = base_prompt + TODO_INSTRUCTIONS
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

    return OrchestratorResult(content=_extract_text_content(result["messages"][-1].content), state=result)


_vector_store: VectorStoreManager | None = None


def _get_vector_store() -> VectorStoreManager | None:
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = VectorStoreManager()
        except Exception:
            return None
    return _vector_store
