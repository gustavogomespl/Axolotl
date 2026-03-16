from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.agent import Agent
from app.models.database import get_db
from app.models.mcp_server import MCPServer
from app.models.project import Project
from app.models.skill import Skill
from app.models.tool import ToolModel

router = APIRouter(prefix="/projects/{project_id}/agents", tags=["agents"])


class AgentCreate(BaseModel):
    name: str
    description: str | None = None
    prompt: str = "You are a helpful assistant."
    model: str | None = None
    is_planner: bool = False
    config: dict | None = None
    tool_ids: list[str] = []
    skill_ids: list[str] = []
    mcp_server_ids: list[str] = []


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    prompt: str | None = None
    model: str | None = None
    is_planner: bool | None = None
    config: dict | None = None
    tool_ids: list[str] | None = None
    skill_ids: list[str] | None = None
    mcp_server_ids: list[str] | None = None


class AgentResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None = None
    prompt: str
    model: str | None = None
    is_planner: bool = False
    config: dict | None = None
    tool_ids: list[str] = []
    skill_ids: list[str] = []
    mcp_server_ids: list[str] = []
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


def _agent_to_response(agent: Agent) -> dict:
    return {
        "id": agent.id,
        "project_id": agent.project_id,
        "name": agent.name,
        "description": agent.description,
        "prompt": agent.prompt,
        "model": agent.model,
        "is_planner": agent.is_planner,
        "config": agent.config,
        "tool_ids": [t.id for t in agent.tools],
        "skill_ids": [s.id for s in agent.skills],
        "mcp_server_ids": [m.id for m in agent.mcp_servers],
        "created_at": agent.created_at.isoformat() if agent.created_at else "",
        "updated_at": agent.updated_at.isoformat() if agent.updated_at else "",
    }


async def _get_project_or_404(project_id: str, db: AsyncSession) -> Project:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("")
async def create_agent(project_id: str, request: AgentCreate, db: AsyncSession = Depends(get_db)):
    await _get_project_or_404(project_id, db)

    agent = Agent(
        project_id=project_id,
        name=request.name,
        description=request.description,
        prompt=request.prompt,
        model=request.model,
        is_planner=request.is_planner,
        config=request.config,
    )

    # Load related entities
    if request.tool_ids:
        result = await db.execute(select(ToolModel).where(ToolModel.id.in_(request.tool_ids)))
        agent.tools = list(result.scalars().all())

    if request.skill_ids:
        result = await db.execute(select(Skill).where(Skill.id.in_(request.skill_ids)))
        agent.skills = list(result.scalars().all())

    if request.mcp_server_ids:
        result = await db.execute(select(MCPServer).where(MCPServer.id.in_(request.mcp_server_ids)))
        agent.mcp_servers = list(result.scalars().all())

    db.add(agent)
    await db.commit()
    await db.refresh(agent, attribute_names=["tools", "skills", "mcp_servers"])
    return _agent_to_response(agent)


@router.get("")
async def list_agents(project_id: str, db: AsyncSession = Depends(get_db)):
    await _get_project_or_404(project_id, db)
    result = await db.execute(
        select(Agent)
        .where(Agent.project_id == project_id)
        .options(
            selectinload(Agent.tools), selectinload(Agent.skills), selectinload(Agent.mcp_servers)
        )
        .order_by(Agent.created_at)
    )
    agents = result.scalars().all()
    return [_agent_to_response(a) for a in agents]


@router.get("/{agent_id}")
async def get_agent(project_id: str, agent_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Agent)
        .where(Agent.id == agent_id, Agent.project_id == project_id)
        .options(
            selectinload(Agent.tools), selectinload(Agent.skills), selectinload(Agent.mcp_servers)
        )
    )
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_to_response(agent)


@router.put("/{agent_id}")
async def update_agent(
    project_id: str, agent_id: str, request: AgentUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Agent)
        .where(Agent.id == agent_id, Agent.project_id == project_id)
        .options(
            selectinload(Agent.tools), selectinload(Agent.skills), selectinload(Agent.mcp_servers)
        )
    )
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    update_data = request.model_dump(exclude_unset=True)

    # Handle relationship updates separately
    if "tool_ids" in update_data:
        tool_ids = update_data.pop("tool_ids")
        result = await db.execute(select(ToolModel).where(ToolModel.id.in_(tool_ids)))
        agent.tools = list(result.scalars().all())

    if "skill_ids" in update_data:
        skill_ids = update_data.pop("skill_ids")
        result = await db.execute(select(Skill).where(Skill.id.in_(skill_ids)))
        agent.skills = list(result.scalars().all())

    if "mcp_server_ids" in update_data:
        mcp_server_ids = update_data.pop("mcp_server_ids")
        result = await db.execute(select(MCPServer).where(MCPServer.id.in_(mcp_server_ids)))
        agent.mcp_servers = list(result.scalars().all())

    for key, value in update_data.items():
        setattr(agent, key, value)

    await db.commit()
    await db.refresh(agent, attribute_names=["tools", "skills", "mcp_servers"])
    return _agent_to_response(agent)


@router.delete("/{agent_id}")
async def delete_agent(project_id: str, agent_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Agent).where(Agent.id == agent_id, Agent.project_id == project_id)
    )
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    await db.delete(agent)
    await db.commit()
    return {"status": "deleted"}
