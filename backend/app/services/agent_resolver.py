from langchain_core.tools import BaseTool

from app.core.langgraph.tools.api_tool import APIToolConfig, create_api_tool
from app.core.langgraph.tools.mcp_manager import mcp_manager
from app.core.langgraph.tools.registry import ToolRegistry
from app.core.vector_store.client import VectorStoreManager
from app.models.agent import Agent


async def resolve_agent_tools(
    agent: Agent,
    vector_store: VectorStoreManager | None = None,
) -> list[BaseTool]:
    """Resolve an agent's DB relationships into LangChain tools.

    Expects agent.tools, agent.skills, and agent.mcp_servers
    to be eagerly loaded (via selectinload).
    """
    tools: list[BaseTool] = []

    # 1. Tools from DB (API tools or tools registered in ToolRegistry)
    for tool_model in agent.tools:
        if tool_model.type == "api" and tool_model.api_config:
            tools.append(create_api_tool(APIToolConfig(**tool_model.api_config)))
        else:
            # Try to find in global ToolRegistry (native or MCP-loaded)
            registry_tools = ToolRegistry.get_tools([tool_model.name])
            tools.extend(registry_tools)

    # 2. RAG skills become retriever tools
    if vector_store:
        for skill in agent.skills:
            if skill.type == "rag" and skill.collection_name and skill.is_active:
                try:
                    from langchain.tools.retriever import create_retriever_tool

                    retriever = vector_store.get_retriever(skill.collection_name)
                    tools.append(
                        create_retriever_tool(
                            retriever,
                            name=f"search_{skill.name}",
                            description=skill.description or f"Search {skill.name} knowledge base",
                        )
                    )
                except Exception:
                    pass

    # 3. MCP server tools
    for mcp_server in agent.mcp_servers:
        try:
            server_tools = await mcp_manager.connect_and_load_tools(mcp_server.name)
            tools.extend(server_tools)
        except Exception:
            # Server may not be registered in mcp_manager yet
            pass

    return tools
