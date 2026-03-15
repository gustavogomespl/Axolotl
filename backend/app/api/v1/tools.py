import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.langgraph.tools.api_tool import APIToolConfig, create_api_tool
from app.core.langgraph.tools.mcp_manager import mcp_manager
from app.core.langgraph.tools.registry import ToolRegistry

router = APIRouter(prefix="/tools", tags=["tools"])


class ToolCreateRequest(BaseModel):
    name: str
    description: str
    type: str  # "api" or "native"
    category: str = "general"
    api_config: APIToolConfig | None = None


class MCPServerCreateRequest(BaseModel):
    name: str
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict | None = None


class ToolTestRequest(BaseModel):
    input: dict


# --- Tool CRUD ---

@router.post("")
async def create_tool(request: ToolCreateRequest):
    """Create a new tool (API or native)."""
    if request.type == "api":
        if not request.api_config:
            raise HTTPException(status_code=400, detail="api_config required for API tools")
        tool = create_api_tool(request.api_config)
        ToolRegistry.register(tool, category=request.category)
        return {"name": request.name, "type": "api", "status": "registered"}
    else:
        raise HTTPException(
            status_code=400,
            detail="Only 'api' tools can be created via API. Native tools are registered in code.",
        )


@router.get("")
async def list_tools():
    """List all registered tools."""
    return ToolRegistry.list_all()


@router.get("/{name}")
async def get_tool(name: str):
    """Get tool details."""
    tools = ToolRegistry.get_tools([name])
    if not tools:
        raise HTTPException(status_code=404, detail="Tool not found")
    t = tools[0]
    return {"name": t.name, "description": t.description}


@router.delete("/{name}")
async def delete_tool(name: str):
    """Remove a tool."""
    if not ToolRegistry.remove(name):
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"status": "deleted"}


@router.post("/{name}/test")
async def test_tool(name: str, request: ToolTestRequest):
    """Test a tool with sample input."""
    tools = ToolRegistry.get_tools([name])
    if not tools:
        raise HTTPException(status_code=404, detail="Tool not found")

    tool = tools[0]
    try:
        result = await tool.ainvoke(request.input)
        return {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- MCP Server Management ---

@router.post("/mcp-servers")
async def add_mcp_server(request: MCPServerCreateRequest):
    """Register an MCP server."""
    await mcp_manager.add_server(
        name=request.name,
        url=request.url,
        command=request.command,
        args=request.args,
        env=request.env,
    )
    return {"name": request.name, "status": "registered"}


@router.get("/mcp-servers")
async def list_mcp_servers():
    """List all MCP servers."""
    return mcp_manager.list_servers()


@router.delete("/mcp-servers/{name}")
async def remove_mcp_server(name: str):
    """Remove an MCP server."""
    if not await mcp_manager.remove_server(name):
        raise HTTPException(status_code=404, detail="MCP server not found")
    return {"status": "deleted"}


@router.post("/mcp-servers/{name}/refresh")
async def refresh_mcp_server(name: str):
    """Reconnect to an MCP server and reload its tools."""
    try:
        tools = await mcp_manager.connect_and_load_tools(name)
        return {"tools_loaded": len(tools), "tool_names": [t.name for t in tools]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
