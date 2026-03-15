from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.langgraph.tools.mcp_manager import mcp_manager
from app.models.database import get_db
from app.models.mcp_server import MCPServer
from app.models.project import Project

router = APIRouter(prefix="/projects/{project_id}/mcp-servers", tags=["mcp-servers"])


class MCPServerCreate(BaseModel):
    name: str
    transport: str = "streamable_http"
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict | None = None


def _mcp_to_response(mcp: MCPServer) -> dict:
    return {
        "id": mcp.id,
        "project_id": mcp.project_id,
        "name": mcp.name,
        "transport": mcp.transport,
        "url": mcp.url,
        "command": mcp.command,
        "args": mcp.args,
        "env": mcp.env,
        "status": mcp.status,
        "created_at": mcp.created_at.isoformat() if mcp.created_at else "",
    }


@router.post("")
async def create_mcp_server(
    project_id: str, request: MCPServerCreate, db: AsyncSession = Depends(get_db)
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    mcp_server = MCPServer(
        project_id=project_id,
        name=request.name,
        transport=request.transport,
        url=request.url,
        command=request.command,
        args=request.args,
        env=request.env,
    )
    db.add(mcp_server)
    await db.commit()
    await db.refresh(mcp_server)

    # Also register with the in-memory MCP manager
    await mcp_manager.add_server(
        name=request.name,
        url=request.url,
        command=request.command,
        args=request.args,
        env=request.env,
    )

    return _mcp_to_response(mcp_server)


@router.get("")
async def list_mcp_servers(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(MCPServer)
        .where(MCPServer.project_id == project_id)
        .order_by(MCPServer.created_at.desc())
    )
    return [_mcp_to_response(m) for m in result.scalars().all()]


@router.get("/{mcp_server_id}")
async def get_mcp_server(project_id: str, mcp_server_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(MCPServer).where(MCPServer.id == mcp_server_id, MCPServer.project_id == project_id)
    )
    mcp_server = result.scalars().first()
    if not mcp_server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return _mcp_to_response(mcp_server)


@router.delete("/{mcp_server_id}")
async def delete_mcp_server(
    project_id: str, mcp_server_id: str, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(MCPServer).where(MCPServer.id == mcp_server_id, MCPServer.project_id == project_id)
    )
    mcp_server = result.scalars().first()
    if not mcp_server:
        raise HTTPException(status_code=404, detail="MCP server not found")

    await mcp_manager.remove_server(mcp_server.name)
    await db.delete(mcp_server)
    await db.commit()
    return {"status": "deleted"}


@router.post("/{mcp_server_id}/refresh")
async def refresh_mcp_server(
    project_id: str, mcp_server_id: str, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(MCPServer).where(MCPServer.id == mcp_server_id, MCPServer.project_id == project_id)
    )
    mcp_server = result.scalars().first()
    if not mcp_server:
        raise HTTPException(status_code=404, detail="MCP server not found")

    try:
        tools = await mcp_manager.connect_and_load_tools(mcp_server.name)
        mcp_server.status = "connected"
        await db.commit()
        return {"tools_loaded": len(tools), "tool_names": [t.name for t in tools]}
    except Exception as e:
        mcp_server.status = "disconnected"
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))
