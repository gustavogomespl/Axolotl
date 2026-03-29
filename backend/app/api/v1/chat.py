import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.redis import redis_manager
from app.models.agent import Agent
from app.models.conversation import Conversation, Message
from app.models.database import get_db
from app.models.project import Project
from app.services.orchestrator import orchestrate_project_chat

router = APIRouter(prefix="/projects/{project_id}/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    phone_number: str | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    todo_list: list[dict] = []
    completed_tasks: list[dict] = []
    files: list[dict] = []


@router.post("")
async def chat(project_id: str, request: ChatRequest, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    thread_id = request.thread_id or str(uuid.uuid4())

    # Load all agents with their resources
    result = await db.execute(
        select(Agent)
        .where(Agent.project_id == project_id)
        .options(
            selectinload(Agent.tools),
            selectinload(Agent.skills),
            selectinload(Agent.mcp_servers),
        )
    )
    agents = list(result.scalars().all())

    # Get Redis checkpointer
    try:
        checkpointer = await redis_manager.get_saver()
    except Exception:
        checkpointer = None

    # Orchestrate via planner/worker pattern
    orch_result = await orchestrate_project_chat(
        project=project,
        agents=agents,
        message=request.message,
        thread_id=thread_id,
        model_override=request.model,
        checkpointer=checkpointer,
    )

    # Persist conversation if phone_number is provided
    if request.phone_number:
        conversation = Conversation(project_id=project_id, phone_number=request.phone_number)
        db.add(conversation)
        await db.flush()

        db.add(Message(conversation_id=conversation.id, role="user", content=request.message))
        db.add(
            Message(conversation_id=conversation.id, role="assistant", content=orch_result.content)
        )
        await db.commit()

    return ChatResponse(
        thread_id=thread_id,
        response=orch_result.content,
        todo_list=orch_result.todo_list,
        completed_tasks=orch_result.completed_tasks,
        files=orch_result.files,
    )


@router.post("/stream")
async def chat_stream(project_id: str, request: ChatRequest, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    thread_id = request.thread_id or str(uuid.uuid4())

    orchestration_mode = getattr(project, "orchestration_mode", "supervisor")

    use_deep_agent = False

    # Always load agents so planner config is available for all modes
    result = await db.execute(
        select(Agent)
        .where(Agent.project_id == project_id)
        .options(
            selectinload(Agent.tools),
            selectinload(Agent.skills),
            selectinload(Agent.mcp_servers),
        )
    )
    agents = list(result.scalars().all())
    planner = next((a for a in agents if a.is_planner), None)
    workers = [a for a in agents if not a.is_planner]

    if orchestration_mode == "deep_agent" and workers:
        use_deep_agent = True

    if use_deep_agent:
        from app.core.langgraph.graphs.deep_agent import build_deep_agent
        from app.services.agent_resolver import resolve_agent_tools
        from app.services.orchestrator import _get_vector_store

        vector_store = _get_vector_store()

        subagents = []
        for worker in workers:
            worker_tools = await resolve_agent_tools(worker, vector_store)
            subagent_config = {
                "name": worker.name,
                "description": worker.description or worker.prompt[:200],
                "system_prompt": worker.prompt,
                "tools": worker_tools,
            }
            if worker.model or project.model:
                subagent_config["model"] = worker.model or project.model
            subagents.append(subagent_config)

        # Resolve planner tools (API/MCP/RAG skills as retriever tools)
        planner_tools = await resolve_agent_tools(planner, vector_store) if planner else []

        # Model: planner.model -> request.model -> project.model
        model_name = (
            (planner.model if planner and planner.model else None)
            or request.model
            or project.model
        )
        from app.services.orchestrator import DEEP_AGENT_TODO_INSTRUCTIONS

        base_prompt = (
            planner.prompt
            if planner
            else project.planner_prompt or "You are a helpful assistant."
        )
        system_prompt = base_prompt + DEEP_AGENT_TODO_INSTRUCTIONS

        # Get Redis checkpointer for multi-turn state persistence
        try:
            stream_checkpointer = await redis_manager.get_saver()
        except Exception:
            stream_checkpointer = None

        agent = build_deep_agent(
            model=model_name,
            system_prompt=system_prompt,
            subagents=subagents,
            tools=planner_tools,
            checkpointer=stream_checkpointer,
        )
    else:
        # Fallback to simple agent for streaming — mirror /chat fallback using planner config
        from app.core.langgraph.graphs.simple_agent import build_simple_agent

        prompt = (
            planner.prompt if planner
            else project.planner_prompt or "You are a helpful assistant."
        )
        model = (
            (planner.model if planner and planner.model else None)
            or request.model
            or project.model
        )
        agent = build_simple_agent(model_name=model, system_prompt=prompt)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'metadata', 'thread_id': thread_id})}\n\n"

        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": request.message}]},
            config={"configurable": {"thread_id": thread_id}},
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name']})}\n\n"
            elif kind == "on_tool_end":
                yield f"data: {json.dumps({'type': 'tool_end', 'name': event['name'], 'output': str(event['data'].get('output', ''))[:500]})}\n\n"
            elif kind == "on_chain_start" and "lc_agent_name" in event.get("metadata", {}):
                yield f"data: {json.dumps({'type': 'subagent_start', 'name': event['metadata']['lc_agent_name']})}\n\n"
            elif kind == "on_chain_end" and "lc_agent_name" in event.get("metadata", {}):
                yield f"data: {json.dumps({'type': 'subagent_end', 'name': event['metadata']['lc_agent_name']})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/conversations")
async def list_conversations(
    project_id: str, phone_number: str | None = None, db: AsyncSession = Depends(get_db)
):
    query = select(Conversation).where(Conversation.project_id == project_id)
    if phone_number:
        query = query.where(Conversation.phone_number == phone_number)
    query = query.order_by(Conversation.created_at.desc())
    result = await db.execute(query)
    conversations = result.scalars().all()
    return [
        {
            "id": c.id,
            "project_id": c.project_id,
            "phone_number": c.phone_number,
            "created_at": c.created_at.isoformat() if c.created_at else "",
        }
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    project_id: str, conversation_id: str, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.project_id == project_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalars().first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else "",
        }
        for m in conversation.messages
    ]
