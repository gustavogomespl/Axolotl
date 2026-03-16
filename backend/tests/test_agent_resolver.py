"""Tests for app.services.agent_resolver.resolve_agent_tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.factories import make_agent, make_mcp_server, make_skill, make_tool


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    retriever = MagicMock()
    vs.get_retriever.return_value = retriever
    return vs


@pytest.fixture
def fake_base_tool():
    """Return a minimal object that stands in for a BaseTool."""
    tool = MagicMock()
    tool.name = "fake-tool"
    return tool


# ---- API tools ----


@pytest.mark.asyncio
async def test_api_tool_creates_via_create_api_tool():
    """Agent with an API tool (type='api', api_config set) calls create_api_tool."""
    api_config = {
        "name": "weather",
        "description": "Get weather",
        "method": "GET",
        "url": "https://api.example.com/weather",
        "headers": {},
        "auth_type": "none",
        "auth_config": {},
    }
    tool_model = make_tool(name="weather", type="api", api_config=api_config)
    agent = make_agent(tools=[tool_model], skills=[], mcp_servers=[])

    sentinel = MagicMock(name="api-tool-result")

    with patch("app.services.agent_resolver.create_api_tool", return_value=sentinel) as mock_create:
        from app.services.agent_resolver import resolve_agent_tools

        result = await resolve_agent_tools(agent)

    mock_create.assert_called_once()
    # Verify APIToolConfig was constructed from the dict
    call_arg = mock_create.call_args[0][0]
    assert call_arg.name == "weather"
    assert call_arg.method == "GET"
    assert sentinel in result


# ---- Non-API tools (ToolRegistry) ----


@pytest.mark.asyncio
async def test_non_api_tool_looks_up_in_registry():
    """Agent with a non-API tool falls through to ToolRegistry.get_tools."""
    tool_model = make_tool(name="calculator", type="native", api_config=None)
    agent = make_agent(tools=[tool_model], skills=[], mcp_servers=[])

    registry_tool = MagicMock(name="registry-calc")

    with patch("app.services.agent_resolver.ToolRegistry") as MockRegistry:
        MockRegistry.get_tools.return_value = [registry_tool]

        from app.services.agent_resolver import resolve_agent_tools

        result = await resolve_agent_tools(agent)

    MockRegistry.get_tools.assert_called_once_with(["calculator"])
    assert registry_tool in result


@pytest.mark.asyncio
async def test_non_api_tool_registry_returns_empty():
    """ToolRegistry returns nothing for an unknown tool name."""
    tool_model = make_tool(name="unknown-tool", type="native", api_config=None)
    agent = make_agent(tools=[tool_model], skills=[], mcp_servers=[])

    with patch("app.services.agent_resolver.ToolRegistry") as MockRegistry:
        MockRegistry.get_tools.return_value = []

        from app.services.agent_resolver import resolve_agent_tools

        result = await resolve_agent_tools(agent)

    assert result == []


# ---- RAG skills ----


@pytest.mark.asyncio
async def test_rag_skill_creates_retriever_tool(mock_vector_store):
    """Active RAG skill with collection_name creates a retriever tool."""
    skill = make_skill(
        name="knowledge-base",
        type="rag",
        collection_name="kb-collection",
        is_active=True,
        description="Search the KB",
    )
    agent = make_agent(tools=[], skills=[skill], mcp_servers=[])

    sentinel_retriever_tool = MagicMock(name="retriever-tool")

    with patch(
        "app.services.agent_resolver.create_retriever_tool",
        create=True,
    ) as mock_crt:
        # The function imports create_retriever_tool lazily inside the loop,
        # so we patch it at the location where it will be imported.
        with patch.dict(
            "sys.modules",
            {"langchain.tools.retriever": MagicMock(create_retriever_tool=mock_crt)},
        ):
            # Re-import to pick up the patched module
            mock_crt.return_value = sentinel_retriever_tool

            from app.services.agent_resolver import resolve_agent_tools

            result = await resolve_agent_tools(agent, vector_store=mock_vector_store)

    mock_vector_store.get_retriever.assert_called_once_with("kb-collection")
    assert sentinel_retriever_tool in result


@pytest.mark.asyncio
async def test_inactive_skill_is_skipped(mock_vector_store):
    """Inactive RAG skill is not processed."""
    skill = make_skill(
        name="disabled-kb",
        type="rag",
        collection_name="disabled-coll",
        is_active=False,
    )
    agent = make_agent(tools=[], skills=[skill], mcp_servers=[])

    from app.services.agent_resolver import resolve_agent_tools

    result = await resolve_agent_tools(agent, vector_store=mock_vector_store)

    mock_vector_store.get_retriever.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_rag_skill_without_collection_name_is_skipped(mock_vector_store):
    """RAG skill with collection_name=None is skipped."""
    skill = make_skill(
        name="no-collection",
        type="rag",
        collection_name=None,
        is_active=True,
    )
    agent = make_agent(tools=[], skills=[skill], mcp_servers=[])

    from app.services.agent_resolver import resolve_agent_tools

    result = await resolve_agent_tools(agent, vector_store=mock_vector_store)

    mock_vector_store.get_retriever.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_non_rag_skill_is_skipped(mock_vector_store):
    """Skill with type != 'rag' is ignored even when active."""
    skill = make_skill(
        name="prompt-skill",
        type="prompt",
        collection_name=None,
        is_active=True,
    )
    agent = make_agent(tools=[], skills=[skill], mcp_servers=[])

    from app.services.agent_resolver import resolve_agent_tools

    result = await resolve_agent_tools(agent, vector_store=mock_vector_store)

    mock_vector_store.get_retriever.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_rag_skill_without_vector_store():
    """When vector_store is None, RAG skills are not processed."""
    skill = make_skill(type="rag", collection_name="coll", is_active=True)
    agent = make_agent(tools=[], skills=[skill], mcp_servers=[])

    from app.services.agent_resolver import resolve_agent_tools

    result = await resolve_agent_tools(agent, vector_store=None)

    assert result == []


@pytest.mark.asyncio
async def test_rag_skill_exception_is_caught(mock_vector_store):
    """Exception during retriever creation is silently caught."""
    skill = make_skill(type="rag", collection_name="bad-coll", is_active=True)
    agent = make_agent(tools=[], skills=[skill], mcp_servers=[])

    mock_vector_store.get_retriever.side_effect = RuntimeError("connection failed")

    from app.services.agent_resolver import resolve_agent_tools

    result = await resolve_agent_tools(agent, vector_store=mock_vector_store)

    # Exception is swallowed; result is empty
    assert result == []


# ---- MCP servers ----


@pytest.mark.asyncio
async def test_mcp_server_loads_tools():
    """MCP server calls mcp_manager.connect_and_load_tools."""
    mcp = make_mcp_server(name="my-mcp")
    agent = make_agent(tools=[], skills=[], mcp_servers=[mcp])

    mcp_tool = MagicMock(name="mcp-loaded-tool")

    with patch("app.services.agent_resolver.mcp_manager") as mock_mgr:
        mock_mgr.connect_and_load_tools = AsyncMock(return_value=[mcp_tool])

        from app.services.agent_resolver import resolve_agent_tools

        result = await resolve_agent_tools(agent)

    mock_mgr.connect_and_load_tools.assert_awaited_once_with("my-mcp")
    assert mcp_tool in result


@pytest.mark.asyncio
async def test_mcp_server_failure_is_caught():
    """Exception from mcp_manager is caught; other resources still resolve."""
    mcp = make_mcp_server(name="broken-mcp")
    agent = make_agent(tools=[], skills=[], mcp_servers=[mcp])

    with patch("app.services.agent_resolver.mcp_manager") as mock_mgr:
        mock_mgr.connect_and_load_tools = AsyncMock(
            side_effect=KeyError("MCP server 'broken-mcp' not found")
        )

        from app.services.agent_resolver import resolve_agent_tools

        result = await resolve_agent_tools(agent)

    # Exception swallowed; empty list
    assert result == []


@pytest.mark.asyncio
async def test_multiple_mcp_servers_one_fails():
    """When one MCP server fails, others still load."""
    mcp_ok = make_mcp_server(name="ok-mcp")
    mcp_bad = make_mcp_server(name="bad-mcp")
    agent = make_agent(tools=[], skills=[], mcp_servers=[mcp_ok, mcp_bad])

    ok_tool = MagicMock(name="ok-tool")

    async def side_effect(name):
        if name == "bad-mcp":
            raise ConnectionError("unreachable")
        return [ok_tool]

    with patch("app.services.agent_resolver.mcp_manager") as mock_mgr:
        mock_mgr.connect_and_load_tools = AsyncMock(side_effect=side_effect)

        from app.services.agent_resolver import resolve_agent_tools

        result = await resolve_agent_tools(agent)

    assert ok_tool in result
    assert len(result) == 1


# ---- No resources ----


@pytest.mark.asyncio
async def test_agent_with_no_resources():
    """Agent with empty tools, skills, mcp_servers returns []."""
    agent = make_agent(tools=[], skills=[], mcp_servers=[])

    from app.services.agent_resolver import resolve_agent_tools

    result = await resolve_agent_tools(agent)

    assert result == []


# ---- Combined ----


@pytest.mark.asyncio
async def test_combined_tools_skills_mcp(mock_vector_store):
    """Agent with tools + skills + mcp_servers returns all resolved tools."""
    # API tool
    api_config = {
        "name": "api-tool",
        "description": "An API tool",
        "method": "POST",
        "url": "https://example.com/api",
        "headers": {},
        "auth_type": "none",
        "auth_config": {},
    }
    api_tool_model = make_tool(name="api-tool", type="api", api_config=api_config)

    # Native tool
    native_tool_model = make_tool(name="native-tool", type="native", api_config=None)

    # RAG skill
    rag_skill = make_skill(
        name="docs",
        type="rag",
        collection_name="docs-coll",
        is_active=True,
        description="Search docs",
    )

    # Inactive skill (should be skipped)
    inactive_skill = make_skill(type="rag", collection_name="old", is_active=False)

    # MCP server
    mcp = make_mcp_server(name="combo-mcp")

    agent = make_agent(
        tools=[api_tool_model, native_tool_model],
        skills=[rag_skill, inactive_skill],
        mcp_servers=[mcp],
    )

    api_sentinel = MagicMock(name="api-sentinel")
    registry_sentinel = MagicMock(name="registry-sentinel")
    retriever_sentinel = MagicMock(name="retriever-sentinel")
    mcp_sentinel = MagicMock(name="mcp-sentinel")

    with (
        patch(
            "app.services.agent_resolver.create_api_tool",
            return_value=api_sentinel,
        ),
        patch("app.services.agent_resolver.ToolRegistry") as MockRegistry,
        patch("app.services.agent_resolver.mcp_manager") as mock_mgr,
        patch.dict(
            "sys.modules",
            {
                "langchain.tools.retriever": MagicMock(
                    create_retriever_tool=MagicMock(return_value=retriever_sentinel)
                )
            },
        ),
    ):
        MockRegistry.get_tools.return_value = [registry_sentinel]
        mock_mgr.connect_and_load_tools = AsyncMock(return_value=[mcp_sentinel])

        from app.services.agent_resolver import resolve_agent_tools

        result = await resolve_agent_tools(agent, vector_store=mock_vector_store)

    assert api_sentinel in result
    assert registry_sentinel in result
    assert retriever_sentinel in result
    assert mcp_sentinel in result
    # 4 tools total (inactive skill skipped)
    assert len(result) == 4
