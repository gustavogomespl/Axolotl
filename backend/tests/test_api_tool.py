"""Tests for app.core.langgraph.tools.api_tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.langgraph.tools.api_tool import APIToolConfig, create_api_tool

# ---------------------------------------------------------------------------
# APIToolConfig validation
# ---------------------------------------------------------------------------


class TestAPIToolConfig:
    def test_minimal_valid_config(self):
        cfg = APIToolConfig(
            name="test",
            description="A test tool",
            method="GET",
            url="https://api.example.com/data",
        )
        assert cfg.name == "test"
        assert cfg.auth_type == "none"
        assert cfg.headers == {}
        assert cfg.body_template is None
        assert cfg.query_params == {}
        assert cfg.response_parser is None
        assert cfg.timeout == 30

    def test_full_config(self):
        cfg = APIToolConfig(
            name="weather",
            description="Get weather",
            method="POST",
            url="https://api.weather.com/{city}",
            headers={"Accept": "application/json"},
            body_template={"location": "{city}"},
            query_params={"units": "metric"},
            response_parser="data.temp",
            timeout=15,
            auth_type="bearer",
            auth_config={"token": "abc123"},
        )
        assert cfg.method == "POST"
        assert cfg.auth_type == "bearer"
        assert cfg.timeout == 15

    def test_invalid_method_rejected(self):
        with pytest.raises(Exception):
            APIToolConfig(
                name="bad",
                description="x",
                method="PATCH",
                url="https://example.com",
            )

    def test_missing_required_fields(self):
        with pytest.raises(Exception):
            APIToolConfig(name="incomplete")


# ---------------------------------------------------------------------------
# create_api_tool
# ---------------------------------------------------------------------------


class TestCreateApiTool:
    """All tests mock httpx.AsyncClient so no real HTTP calls happen."""

    def _make_config(self, **overrides) -> APIToolConfig:
        defaults = {
            "name": "test_api",
            "description": "test api tool",
            "method": "GET",
            "url": "https://api.example.com/data",
        }
        defaults.update(overrides)
        return APIToolConfig(**defaults)

    def _mock_response(self, text: str = "ok", status_code: int = 200, json_data=None):
        resp = MagicMock()
        resp.text = text
        resp.status_code = status_code
        resp.json.return_value = json_data if json_data is not None else {}
        return resp

    # --- HTTP method dispatch ---

    @pytest.mark.asyncio
    async def test_get_request(self):
        config = self._make_config(method="GET")
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response(text="response body")
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            result = await tool_fn.ainvoke({})

        mock_client.request.assert_awaited_once()
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["method"] == "GET"
        assert result == "response body"

    @pytest.mark.asyncio
    async def test_post_request(self):
        config = self._make_config(method="POST")
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response(text="created")
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            result = await tool_fn.ainvoke({})

        assert mock_client.request.call_args.kwargs["method"] == "POST"
        assert result == "created"

    @pytest.mark.asyncio
    async def test_put_request(self):
        config = self._make_config(method="PUT")
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response(text="updated")
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            await tool_fn.ainvoke({})

        assert mock_client.request.call_args.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_delete_request(self):
        config = self._make_config(method="DELETE")
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response(text="deleted")
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            await tool_fn.ainvoke({})

        assert mock_client.request.call_args.kwargs["method"] == "DELETE"

    # --- URL placeholder interpolation ---

    @pytest.mark.asyncio
    async def test_url_placeholder_interpolation(self):
        config = self._make_config(url="https://api.weather.com/{city}/forecast")
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response(text="sunny")
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            await tool_fn.ainvoke({"city": "london"})

        called_url = mock_client.request.call_args.kwargs["url"]
        assert called_url == "https://api.weather.com/london/forecast"

    # --- Header injection ---

    @pytest.mark.asyncio
    async def test_bearer_auth_header(self):
        config = self._make_config(
            auth_type="bearer",
            auth_config={"token": "my-secret-token"},
        )
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            await tool_fn.ainvoke({})

        headers = mock_client.request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-secret-token"

    @pytest.mark.asyncio
    async def test_api_key_auth_header(self):
        config = self._make_config(
            auth_type="api_key",
            auth_config={"header": "X-Custom-Key", "key": "key123"},
        )
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            await tool_fn.ainvoke({})

        headers = mock_client.request.call_args.kwargs["headers"]
        assert headers["X-Custom-Key"] == "key123"

    @pytest.mark.asyncio
    async def test_api_key_default_header_name(self):
        config = self._make_config(
            auth_type="api_key",
            auth_config={"key": "key456"},
        )
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            await tool_fn.ainvoke({})

        headers = mock_client.request.call_args.kwargs["headers"]
        assert headers["X-API-Key"] == "key456"

    # --- Body template interpolation ---

    @pytest.mark.asyncio
    async def test_body_template_interpolation(self):
        config = self._make_config(
            method="POST",
            body_template={"query": "{search_term}", "limit": "10"},
        )
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            await tool_fn.ainvoke({"search_term": "python"})

        body = mock_client.request.call_args.kwargs["json"]
        assert body["query"] == "python"
        assert body["limit"] == "10"

    # --- Response truncation ---

    @pytest.mark.asyncio
    async def test_response_truncated_at_2000_chars(self):
        long_text = "x" * 5000
        config = self._make_config()
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response(text=long_text)
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            result = await tool_fn.ainvoke({})

        assert len(result) == 2000

    @pytest.mark.asyncio
    async def test_short_response_not_truncated(self):
        config = self._make_config()
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response(text="short")
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            result = await tool_fn.ainvoke({})

        assert result == "short"

    # --- JMESPath parsing ---

    @pytest.mark.asyncio
    async def test_jmespath_parsing(self):
        config = self._make_config(response_parser="data.temperature")
        tool_fn = create_api_tool(config)

        json_data = {"data": {"temperature": 25}}
        mock_resp = self._mock_response(text='{"data":{"temperature":25}}', json_data=json_data)
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_jmespath = MagicMock()
        mock_jmespath.search.return_value = 25

        with (
            patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client),
            patch.dict("sys.modules", {"jmespath": mock_jmespath}),
        ):
            result = await tool_fn.ainvoke({})

        mock_jmespath.search.assert_called_once_with("data.temperature", json_data)
        assert result == "25"

    @pytest.mark.asyncio
    async def test_jmespath_error_falls_back_to_raw_text(self):
        config = self._make_config(response_parser="bad.path")
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response(text="raw text")
        mock_resp.json.side_effect = ValueError("not json")
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            result = await tool_fn.ainvoke({})

        # Falls back to raw text (truncated to 2000)
        assert result == "raw text"

    # --- Query params interpolation ---

    @pytest.mark.asyncio
    async def test_query_params_interpolation(self):
        config = self._make_config(
            query_params={"q": "{term}", "page": "1"},
        )
        tool_fn = create_api_tool(config)

        mock_resp = self._mock_response()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.langgraph.tools.api_tool.httpx.AsyncClient", return_value=mock_client):
            await tool_fn.ainvoke({"term": "langchain"})

        params = mock_client.request.call_args.kwargs["params"]
        assert params["q"] == "langchain"
        assert params["page"] == "1"

    # --- Tool metadata ---

    def test_created_tool_has_correct_name(self):
        config = self._make_config(name="my_api_tool")
        tool_fn = create_api_tool(config)
        assert tool_fn.name == "my_api_tool"

    def test_created_tool_has_correct_description(self):
        config = self._make_config(description="Fetches data from API")
        tool_fn = create_api_tool(config)
        assert tool_fn.description == "Fetches data from API"
