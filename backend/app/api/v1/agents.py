import json
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.langgraph.factory import GraphConfig, GraphFactory
from app.core.langgraph.subgraphs.registry import SubgraphRegistry

router = APIRouter(prefix="/agents", tags=["agents"])

# In-memory store (will be replaced by PostgreSQL)
_agent_configs: dict[str, dict] = {}
_compiled_graphs: dict[str, object] = {}


class AgentCreateRequest(GraphConfig):
    pass


class AgentResponse(BaseModel):
    id: str
    name: str
    pattern: str
    model: str | None = None
    system_prompt: str


@router.post("", response_model=AgentResponse)
async def create_agent(request: AgentCreateRequest):
    """Create a new agent from configuration."""
    agent_id = str(uuid.uuid4())

    try:
        graph = GraphFactory.build(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    _agent_configs[agent_id] = request.model_dump()
    _compiled_graphs[agent_id] = graph

    return AgentResponse(
        id=agent_id,
        name=request.name,
        pattern=request.pattern,
        model=request.model,
        system_prompt=request.system_prompt,
    )


@router.get("")
async def list_agents():
    """List all registered agents."""
    return [
        {"id": agent_id, **config}
        for agent_id, config in _agent_configs.items()
    ]


@router.get("/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent details."""
    config = _agent_configs.get(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"id": agent_id, **config}


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    if agent_id not in _agent_configs:
        raise HTTPException(status_code=404, detail="Agent not found")
    del _agent_configs[agent_id]
    del _compiled_graphs[agent_id]
    return {"status": "deleted"}


@router.get("/subgraphs/list")
async def list_subgraphs():
    """List all registered subgraphs."""
    return SubgraphRegistry.list_all()
