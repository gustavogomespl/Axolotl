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
    response_content = await orchestrate_project_chat(
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
        db.add(Message(conversation_id=conversation.id, role="assistant", content=response_content))
        await db.commit()

    return ChatResponse(thread_id=thread_id, response=response_content)


@router.post("/stream")
async def chat_stream(project_id: str, request: ChatRequest, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    thread_id = request.thread_id or str(uuid.uuid4())

    # For streaming, use simple agent as supervisor streaming is more complex
    from app.core.langgraph.graphs.simple_agent import build_simple_agent

    agent = build_simple_agent(
        model_name=request.model or project.model,
        system_prompt=project.planner_prompt or "You are a helpful assistant.",
    )

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
