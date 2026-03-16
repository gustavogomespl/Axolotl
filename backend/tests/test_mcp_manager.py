"""Tests for app.core.langgraph.tools.mcp_manager.MCPManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.langgraph.tools.mcp_manager import MCPManager
from app.core.langgraph.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear ToolRegistry so MCP tests don't leak state."""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


@pytest.fixture
def manager():
    return MCPManager()


# ---------------------------------------------------------------------------
# add_server
# ---------------------------------------------------------------------------


class TestAddServer:
    @pytest.mark.asyncio
    async def test_add_http_server(self, manager):
        await manager.add_server("weather", url="http://localhost:3000")

        servers = manager.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "weather"
        assert servers[0]["transport"] == "streamable_http"
        assert servers[0]["url"] == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_add_stdio_server(self, manager):
        await manager.add_server(
            "local",
            command="python",
            args=["-m", "mcp_server"],
            env={"API_KEY": "secret"},
        )

        servers = manager.list_servers()
        assert len(servers) == 1
        assert servers[0]["transport"] == "stdio"
        assert servers[0]["command"] == "python"
        assert servers[0]["args"] == ["-m", "mcp_server"]
        assert servers[0]["env"] == {"API_KEY": "secret"}

    @pytest.mark.asyncio
    async def test_add_stdio_server_defaults(self, manager):
        await manager.add_server("basic", command="/usr/bin/server")

        servers = manager.list_servers()
        assert servers[0]["args"] == []
        assert servers[0]["env"] == {}

    @pytest.mark.asyncio
    async def test_add_server_raises_without_url_or_command(self, manager):
        with pytest.raises(ValueError, match="Either url or command must be provided"):
            await manager.add_server("broken")

    @pytest.mark.asyncio
    async def test_add_server_overwrites_existing(self, manager):
        await manager.add_server("svc", url="http://old:3000")
        await manager.add_server("svc", url="http://new:4000")

        servers = manager.list_servers()
        assert len(servers) == 1
        assert servers[0]["url"] == "http://new:4000"


# ---------------------------------------------------------------------------
# remove_server
# ---------------------------------------------------------------------------


class TestRemoveServer:
    @pytest.mark.asyncio
    async def test_remove_existing_returns_true(self, manager):
        await manager.add_server("tmp", url="http://localhost:5000")
        result = await manager.remove_server("tmp")
        assert result is True
        assert manager.list_servers() == []

    @pytest.mark.asyncio
    async def test_remove_nonexistent_returns_false(self, manager):
        result = await manager.remove_server("ghost")
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_does_not_affect_others(self, manager):
        await manager.add_server("keep", url="http://a:1000")
        await manager.add_server("drop", url="http://b:2000")

        await manager.remove_server("drop")
        servers = manager.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "keep"


# ---------------------------------------------------------------------------
# list_servers
# ---------------------------------------------------------------------------


class TestListServers:
    def test_list_empty(self, manager):
        assert manager.list_servers() == []

    @pytest.mark.asyncio
    async def test_list_multiple_servers(self, manager):
        await manager.add_server("a", url="http://a:1000")
        await manager.add_server("b", command="cmd_b")

        servers = manager.list_servers()
        assert len(servers) == 2
        names = {s["name"] for s in servers}
        assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# connect_and_load_tools
# ---------------------------------------------------------------------------


class TestConnectAndLoadTools:
    @pytest.mark.asyncio
    async def test_connect_loads_tools_into_registry(self, manager):
        await manager.add_server("srv", url="http://localhost:3000")

        mock_tool = MagicMock()
        mock_tool.name = "mcp_search"
        mock_tool.description = "MCP search"

        mock_client = AsyncMock()
        mock_client.get_tools.return_value = [mock_tool]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await manager.connect_and_load_tools("srv")

        assert len(tools) == 1
        assert tools[0].name == "mcp_search"
        # Verify registered in ToolRegistry with category "mcp"
        meta = ToolRegistry.list_all()
        assert len(meta) == 1
        assert meta[0]["category"] == "mcp"

    @pytest.mark.asyncio
    async def test_connect_unknown_server_raises(self, manager):
        with pytest.raises(KeyError, match="MCP server 'missing' not found"):
            await manager.connect_and_load_tools("missing")

    @pytest.mark.asyncio
    async def test_connect_passes_correct_config(self, manager):
        await manager.add_server("srv", url="http://localhost:3000")

        mock_client = AsyncMock()
        mock_client.get_tools.return_value = []
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ) as mock_cls:
            await manager.connect_and_load_tools("srv")

        expected_config = {"srv": {"transport": "streamable_http", "url": "http://localhost:3000"}}
        mock_cls.assert_called_once_with(expected_config)


# ---------------------------------------------------------------------------
# connect_all
# ---------------------------------------------------------------------------


class TestConnectAll:
    @pytest.mark.asyncio
    async def test_connect_all_empty_returns_empty(self, manager):
        result = await manager.connect_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_connect_all_loads_tools_from_all_servers(self, manager):
        await manager.add_server("srv1", url="http://a:1000")
        await manager.add_server("srv2", command="run_b")

        mock_tool_a = MagicMock()
        mock_tool_a.name = "tool_a"
        mock_tool_a.description = "Tool A"
        mock_tool_b = MagicMock()
        mock_tool_b.name = "tool_b"
        mock_tool_b.description = "Tool B"

        mock_client = AsyncMock()
        mock_client.get_tools.return_value = [mock_tool_a, mock_tool_b]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await manager.connect_all()

        assert len(tools) == 2
        meta = ToolRegistry.list_all()
        assert len(meta) == 2
        assert all(m["category"] == "mcp" for m in meta)

    @pytest.mark.asyncio
    async def test_connect_all_passes_all_server_configs(self, manager):
        await manager.add_server("s1", url="http://a:1000")
        await manager.add_server("s2", url="http://b:2000")

        mock_client = AsyncMock()
        mock_client.get_tools.return_value = []
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ) as mock_cls:
            await manager.connect_all()

        call_arg = mock_cls.call_args[0][0]
        assert "s1" in call_arg
        assert "s2" in call_arg
