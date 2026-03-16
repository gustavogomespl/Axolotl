from typing import Any, Literal

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class APIToolConfig(BaseModel):
    """Configuration for dynamically creating an API tool."""

    name: str
    description: str
    method: Literal["GET", "POST", "PUT", "DELETE"]
    url: str  # Supports {placeholders}
    headers: dict[str, str] = {}
    body_template: dict | None = None
    query_params: dict[str, str] = {}
    response_parser: str | None = None  # JMESPath expression
    timeout: int = 30
    auth_type: Literal["none", "bearer", "api_key", "basic"] = "none"
    auth_config: dict[str, str] = {}


class _DynamicAPITool(BaseTool):
    """A dynamically created API tool backed by an APIToolConfig."""

    api_config: APIToolConfig

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use ainvoke for async execution")

    async def _arun(self, **kwargs: Any) -> str:
        config = self.api_config

        # Interpolate URL placeholders
        url = config.url
        for key, value in kwargs.items():
            url = url.replace(f"{{{key}}}", str(value))

        # Build headers
        headers = dict(config.headers)
        if config.auth_type == "bearer":
            token = config.auth_config.get("token", "")
            headers["Authorization"] = f"Bearer {token}"
        elif config.auth_type == "api_key":
            header_name = config.auth_config.get("header", "X-API-Key")
            headers[header_name] = config.auth_config.get("key", "")

        # Build body
        body = None
        if config.body_template:
            body = dict(config.body_template)
            for key, value in kwargs.items():
                for bk, bv in body.items():
                    if isinstance(bv, str) and f"{{{key}}}" in bv:
                        body[bk] = bv.replace(f"{{{key}}}", str(value))

        # Build query params
        params = dict(config.query_params)
        for key, value in kwargs.items():
            for pk, pv in params.items():
                if isinstance(pv, str) and f"{{{key}}}" in pv:
                    params[pk] = pv.replace(f"{{{key}}}", str(value))

        # Make request
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.request(
                method=config.method,
                url=url,
                headers=headers,
                json=body,
                params=params,
            )

        result = response.text

        # Parse with JMESPath if configured
        if config.response_parser:
            try:
                import jmespath

                data = response.json()
                parsed = jmespath.search(config.response_parser, data)
                result = str(parsed)
            except Exception:
                pass

        return result[:2000]  # Truncate for safety


def create_api_tool(config: APIToolConfig) -> BaseTool:
    """Generate a LangChain tool dynamically from an APIToolConfig."""
    return _DynamicAPITool(
        name=config.name,
        description=config.description,
        api_config=config,
    )
