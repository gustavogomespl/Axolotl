from langchain_core.tools import BaseTool

from app.core.langgraph.tools.registry import ToolRegistry


class MCPManager:
    """Manage connections to MCP servers.

    Supports:
    - Streamable HTTP transport
    - stdio transport
    - Auto-loading of tools from connected servers
    """

    def __init__(self):
        self._servers: dict[str, dict] = {}

    async def add_server(
        self,
        name: str,
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict | None = None,
    ) -> None:
        """Register an MCP server configuration."""
        if url:
            self._servers[name] = {
                "transport": "streamable_http",
                "url": url,
            }
        elif command:
            self._servers[name] = {
                "transport": "stdio",
                "command": command,
                "args": args or [],
                "env": env or {},
            }
        else:
            raise ValueError("Either url or command must be provided")

    async def connect_and_load_tools(self, name: str) -> list[BaseTool]:
        """Connect to an MCP server and load its tools."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        server_config = self._servers.get(name)
        if not server_config:
            raise KeyError(f"MCP server '{name}' not found")

        client = MultiServerMCPClient({name: server_config})
        async with client:
            tools = await client.get_tools()
            for t in tools:
                ToolRegistry.register(t, category="mcp")
            return tools

    async def connect_all(self) -> list[BaseTool]:
        """Connect to all registered servers and load tools."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        if not self._servers:
            return []

        client = MultiServerMCPClient(self._servers)
        async with client:
            tools = await client.get_tools()
            for t in tools:
                ToolRegistry.register(t, category="mcp")
            return tools

    async def remove_server(self, name: str) -> bool:
        if name in self._servers:
            del self._servers[name]
            return True
        return False

    def list_servers(self) -> list[dict]:
        return [{"name": k, **v} for k, v in self._servers.items()]


# Global instance
mcp_manager = MCPManager()
