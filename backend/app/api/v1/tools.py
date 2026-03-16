from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.langgraph.tools.api_tool import APIToolConfig, create_api_tool
from app.core.langgraph.tools.registry import ToolRegistry
from app.models.database import get_db
from app.models.project import Project
from app.models.tool import ToolModel

router = APIRouter(prefix="/projects/{project_id}/tools", tags=["tools"])


class ToolCreateRequest(BaseModel):
    name: str
    description: str
    type: str = "api"
    category: str = "general"
    api_config: APIToolConfig | None = None


class ToolTestRequest(BaseModel):
    input: dict


class ToolResponse(BaseModel):
    id: str
    project_id: str | None = None
    name: str
    description: str
    type: str
    category: str
    api_config: dict | None = None
    mcp_server_name: str | None = None
    created_at: str

    model_config = {"from_attributes": True}


def _tool_to_response(tool: ToolModel) -> dict:
    return {
        "id": tool.id,
        "project_id": tool.project_id,
        "name": tool.name,
        "description": tool.description,
        "type": tool.type,
        "category": tool.category,
        "api_config": tool.api_config,
        "mcp_server_name": tool.mcp_server_name,
        "created_at": tool.created_at.isoformat() if tool.created_at else "",
    }


@router.post("")
async def create_tool(
    project_id: str, request: ToolCreateRequest, db: AsyncSession = Depends(get_db)
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if request.type == "api" and request.api_config:
        tool_fn = create_api_tool(request.api_config)
        ToolRegistry.register(tool_fn, category=request.category)

    tool = ToolModel(
        project_id=project_id,
        name=request.name,
        description=request.description,
        type=request.type,
        category=request.category,
        api_config=request.api_config.model_dump() if request.api_config else None,
    )
    db.add(tool)
    await db.commit()
    await db.refresh(tool)
    return _tool_to_response(tool)


@router.get("")
async def list_tools(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ToolModel)
        .where(ToolModel.project_id == project_id)
        .order_by(ToolModel.created_at.desc())
    )
    return [_tool_to_response(t) for t in result.scalars().all()]


@router.get("/{tool_id}")
async def get_tool(project_id: str, tool_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ToolModel).where(ToolModel.id == tool_id, ToolModel.project_id == project_id)
    )
    tool = result.scalars().first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return _tool_to_response(tool)


@router.delete("/{tool_id}")
async def delete_tool(project_id: str, tool_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ToolModel).where(ToolModel.id == tool_id, ToolModel.project_id == project_id)
    )
    tool = result.scalars().first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    ToolRegistry.remove(tool.name)
    await db.delete(tool)
    await db.commit()
    return {"status": "deleted"}


@router.post("/{tool_id}/test")
async def test_tool(
    project_id: str, tool_id: str, request: ToolTestRequest, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(ToolModel).where(ToolModel.id == tool_id, ToolModel.project_id == project_id)
    )
    tool = result.scalars().first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    tools = ToolRegistry.get_tools([tool.name])
    if not tools:
        raise HTTPException(status_code=400, detail="Tool not loaded in registry")

    try:
        result = await tools[0].ainvoke(request.input)
        return {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
