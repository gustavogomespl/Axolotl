from langchain_core.tools import BaseTool


class ToolRegistry:
    """Centralized registry for all available tools."""

    _tools: dict[str, BaseTool] = {}
    _metadata: dict[str, dict] = {}

    @classmethod
    def register(cls, tool: BaseTool, category: str = "general") -> None:
        cls._tools[tool.name] = tool
        cls._metadata[tool.name] = {
            "type": "native",
            "category": category,
            "description": tool.description,
        }

    @classmethod
    def get_tools(cls, names: list[str] | None = None) -> list[BaseTool]:
        if names is None:
            return list(cls._tools.values())
        return [cls._tools[n] for n in names if n in cls._tools]

    @classmethod
    def list_all(cls) -> list[dict]:
        return [
            {"name": name, **meta}
            for name, meta in cls._metadata.items()
        ]

    @classmethod
    def remove(cls, name: str) -> bool:
        if name in cls._tools:
            del cls._tools[name]
            del cls._metadata[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        cls._tools.clear()
        cls._metadata.clear()
