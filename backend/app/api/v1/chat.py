import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.langgraph.graphs.simple_agent import build_simple_agent
from app.config import settings

router = APIRouter(tags=["chat"])

# In-memory thread store (will be replaced by PostgreSQL checkpointer in production)
_graphs: dict[str, Any] = {}


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    model: str | None = None
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and receive a response."""
    thread_id = request.thread_id or str(uuid.uuid4())

    agent = build_simple_agent(
        model_name=request.model,
        system_prompt=request.system_prompt or "You are a helpful assistant.",
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    ai_message = result["messages"][-1]
    return ChatResponse(thread_id=thread_id, response=ai_message.content)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message and receive a streaming SSE response."""
    thread_id = request.thread_id or str(uuid.uuid4())

    agent = build_simple_agent(
        model_name=request.model,
        system_prompt=request.system_prompt or "You are a helpful assistant.",
    )

    async def event_generator():
        # Send thread_id first
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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
