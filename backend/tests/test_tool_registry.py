"""Tests for app.core.langgraph.tools.registry.ToolRegistry."""

from unittest.mock import MagicMock

import pytest

from app.core.langgraph.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts and ends with a clean registry."""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


def _make_mock_tool(name: str = "mock_tool", description: str = "A mock tool"):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


# --- register ---


class TestRegister:
    def test_register_adds_tool(self):
        tool = _make_mock_tool("search")
        ToolRegistry.register(tool)
        assert len(ToolRegistry.get_tools()) == 1

    def test_register_stores_metadata_with_defaults(self):
        tool = _make_mock_tool("search", description="Search the web")
        ToolRegistry.register(tool)

        meta = ToolRegistry.list_all()
        assert len(meta) == 1
        assert meta[0]["name"] == "search"
        assert meta[0]["type"] == "native"
        assert meta[0]["category"] == "general"
        assert meta[0]["description"] == "Search the web"

    def test_register_custom_category(self):
        tool = _make_mock_tool("fetch")
        ToolRegistry.register(tool, category="api")

        meta = ToolRegistry.list_all()
        assert meta[0]["category"] == "api"

    def test_register_overwrites_existing_tool(self):
        tool_v1 = _make_mock_tool("calc", description="v1")
        tool_v2 = _make_mock_tool("calc", description="v2")

        ToolRegistry.register(tool_v1)
        ToolRegistry.register(tool_v2)

        assert len(ToolRegistry.get_tools()) == 1
        meta = ToolRegistry.list_all()
        assert meta[0]["description"] == "v2"


# --- get_tools ---


class TestGetTools:
    def test_get_tools_returns_all_when_no_names(self):
        t1 = _make_mock_tool("a")
        t2 = _make_mock_tool("b")
        ToolRegistry.register(t1)
        ToolRegistry.register(t2)

        result = ToolRegistry.get_tools()
        assert len(result) == 2

    def test_get_tools_filters_by_names(self):
        t1 = _make_mock_tool("a")
        t2 = _make_mock_tool("b")
        t3 = _make_mock_tool("c")
        ToolRegistry.register(t1)
        ToolRegistry.register(t2)
        ToolRegistry.register(t3)

        result = ToolRegistry.get_tools(names=["a", "c"])
        assert len(result) == 2
        names = [t.name for t in result]
        assert "a" in names
        assert "c" in names

    def test_get_tools_ignores_unknown_names(self):
        t1 = _make_mock_tool("a")
        ToolRegistry.register(t1)

        result = ToolRegistry.get_tools(names=["a", "nonexistent"])
        assert len(result) == 1

    def test_get_tools_empty_registry(self):
        assert ToolRegistry.get_tools() == []

    def test_get_tools_with_empty_list(self):
        t1 = _make_mock_tool("a")
        ToolRegistry.register(t1)
        result = ToolRegistry.get_tools(names=[])
        assert result == []


# --- list_all ---


class TestListAll:
    def test_list_all_returns_all_metadata(self):
        t1 = _make_mock_tool("x", "desc_x")
        t2 = _make_mock_tool("y", "desc_y")
        ToolRegistry.register(t1, category="mcp")
        ToolRegistry.register(t2, category="api")

        result = ToolRegistry.list_all()
        assert len(result) == 2
        by_name = {item["name"]: item for item in result}
        assert by_name["x"]["category"] == "mcp"
        assert by_name["y"]["category"] == "api"
        assert by_name["x"]["description"] == "desc_x"

    def test_list_all_empty_registry(self):
        assert ToolRegistry.list_all() == []


# --- remove ---


class TestRemove:
    def test_remove_existing_tool_returns_true(self):
        tool = _make_mock_tool("rm_me")
        ToolRegistry.register(tool)

        assert ToolRegistry.remove("rm_me") is True
        assert ToolRegistry.get_tools() == []
        assert ToolRegistry.list_all() == []

    def test_remove_nonexistent_tool_returns_false(self):
        assert ToolRegistry.remove("ghost") is False

    def test_remove_does_not_affect_other_tools(self):
        t1 = _make_mock_tool("keep")
        t2 = _make_mock_tool("drop")
        ToolRegistry.register(t1)
        ToolRegistry.register(t2)

        ToolRegistry.remove("drop")
        remaining = ToolRegistry.get_tools()
        assert len(remaining) == 1
        assert remaining[0].name == "keep"


# --- clear ---


class TestClear:
    def test_clear_removes_everything(self):
        for i in range(5):
            ToolRegistry.register(_make_mock_tool(f"tool_{i}"))

        ToolRegistry.clear()
        assert ToolRegistry.get_tools() == []
        assert ToolRegistry.list_all() == []

    def test_clear_on_empty_registry_is_noop(self):
        ToolRegistry.clear()
        assert ToolRegistry.get_tools() == []
