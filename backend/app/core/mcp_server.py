"""Expose Axolotl as an MCP server using FastMCP."""

from fastmcp import FastMCP

mcp = FastMCP("Axolotl")


@mcp.tool()
async def search_knowledge_base(query: str, collection: str = "default") -> str:
    """Search the Axolotl knowledge base."""
    from app.core.vector_store.client import VectorStoreManager

    vs = VectorStoreManager()
    docs = vs.cross_collection_search(query=query, collections=[collection], k=5)
    if not docs:
        return "No results found."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@mcp.tool()
async def list_available_skills() -> str:
    """List all available skills in Axolotl."""
    # Import here to avoid circular imports
    from app.api.v1.skills import _skills

    if not _skills:
        return "No skills registered."
    return "\n".join(
        f"- {s['name']} ({s['type']}): {s['description']}"
        for s in _skills.values()
    )


@mcp.tool()
async def list_available_tools() -> str:
    """List all available tools in Axolotl."""
    from app.core.langgraph.tools.registry import ToolRegistry

    tools = ToolRegistry.list_all()
    if not tools:
        return "No tools registered."
    return "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
