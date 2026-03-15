from typing import Any, Callable


class SubgraphRegistry:
    """Dynamic registry for reusable subgraphs.

    Subgraphs are compiled StateGraphs that can be plugged
    as nodes in any parent graph.
    """

    _registry: dict[str, dict] = {}

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable,
        description: str,
        input_schema: dict | None = None,
        output_schema: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        cls._registry[name] = {
            "builder": builder,
            "description": description,
            "input_schema": input_schema or {},
            "output_schema": output_schema or {},
            "metadata": metadata or {},
        }

    @classmethod
    def get(cls, name: str) -> dict | None:
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> list[dict]:
        return [
            {"name": k, **{k2: v2 for k2, v2 in v.items() if k2 != "builder"}}
            for k, v in cls._registry.items()
        ]

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> Any:
        entry = cls._registry.get(name)
        if entry is None:
            raise KeyError(f"Subgraph '{name}' not found in registry")
        return entry["builder"](**kwargs)

    @classmethod
    def remove(cls, name: str) -> bool:
        if name in cls._registry:
            del cls._registry[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
