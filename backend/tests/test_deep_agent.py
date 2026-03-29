"""Tests for deep agent integration in the orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.factories import make_agent, make_project, make_skill


def _make_result(content: str, extras: dict | None = None):
    """Build a mock result dict that looks like {'messages': [msg], ...}."""
    msg = MagicMock()
    msg.content = content
    result = {"messages": [msg]}
    if extras:
        result.update(extras)
    return result


# ---------------------------------------------------------------------------
# build_deep_agent
# ---------------------------------------------------------------------------


def test_build_deep_agent_creates_agent():
    """build_deep_agent calls create_deep_agent with correct params."""
    mock_agent = MagicMock()

    with patch(
        "app.core.langgraph.graphs.deep_agent.create_deep_agent",
        return_value=mock_agent,
    ) as mock_create:
        from app.core.langgraph.graphs.deep_agent import build_deep_agent

        result = build_deep_agent(
            model="openai:gpt-4.1-mini",
            system_prompt="You are a planner.",
            subagents=[{"name": "researcher", "description": "Research things"}],
            name="test-agent",
        )

    mock_create.assert_called_once_with(
        model="openai:gpt-4.1-mini",
        system_prompt="You are a planner.",
        subagents=[{"name": "researcher", "description": "Research things"}],
        name="test-agent",
        tools=[],
    )
    assert result == mock_agent


def test_build_deep_agent_defaults():
    """build_deep_agent uses sensible defaults for optional params."""
    mock_agent = MagicMock()

    with patch(
        "app.core.langgraph.graphs.deep_agent.create_deep_agent",
        return_value=mock_agent,
    ) as mock_create:
        from app.core.langgraph.graphs.deep_agent import build_deep_agent

        build_deep_agent(model="openai:gpt-4.1-mini", system_prompt="Hello")

    mock_create.assert_called_once_with(
        model="openai:gpt-4.1-mini",
        system_prompt="Hello",
        subagents=[],
        name="main-agent",
        tools=[],
    )


def test_build_deep_agent_null_model_uses_settings_default():
    """When model is None, build_deep_agent falls back to settings.default_model."""
    mock_agent = MagicMock()

    with (
        patch(
            "app.core.langgraph.graphs.deep_agent.create_deep_agent",
            return_value=mock_agent,
        ) as mock_create,
        patch(
            "app.core.langgraph.graphs.deep_agent.settings",
        ) as mock_settings,
    ):
        mock_settings.default_model = "openai:gpt-4.1-mini"

        from app.core.langgraph.graphs.deep_agent import build_deep_agent

        build_deep_agent(model=None, system_prompt="Hello")

    assert mock_create.call_args.kwargs["model"] == "openai:gpt-4.1-mini"


def test_build_deep_agent_with_checkpointer():
    """build_deep_agent forwards checkpointer to create_deep_agent."""
    mock_agent = MagicMock()
    mock_checkpointer = MagicMock(name="redis-saver")

    with patch(
        "app.core.langgraph.graphs.deep_agent.create_deep_agent",
        return_value=mock_agent,
    ) as mock_create:
        from app.core.langgraph.graphs.deep_agent import build_deep_agent

        build_deep_agent(
            model="openai:gpt-4.1-mini",
            system_prompt="Hello",
            checkpointer=mock_checkpointer,
        )

    mock_create.assert_called_once_with(
        model="openai:gpt-4.1-mini",
        system_prompt="Hello",
        subagents=[],
        name="main-agent",
        tools=[],
        checkpointer=mock_checkpointer,
    )


def test_build_deep_agent_without_checkpointer_omits_key():
    """When checkpointer is None, it should not be passed to create_deep_agent."""
    mock_agent = MagicMock()

    with patch(
        "app.core.langgraph.graphs.deep_agent.create_deep_agent",
        return_value=mock_agent,
    ) as mock_create:
        from app.core.langgraph.graphs.deep_agent import build_deep_agent

        build_deep_agent(model="openai:gpt-4.1-mini", system_prompt="Hello")

    call_kwargs = mock_create.call_args.kwargs
    assert "checkpointer" not in call_kwargs


# ---------------------------------------------------------------------------
# _normalize_deep_agent_state
# ---------------------------------------------------------------------------


def test_normalize_deep_agent_state_empty():
    """Empty state normalizes to empty lists."""
    from app.services.orchestrator import _normalize_deep_agent_state

    result = _normalize_deep_agent_state({})
    assert result == {"todo_list": [], "completed_tasks": [], "files": []}


def test_normalize_deep_agent_state_todos_as_strings():
    """DeepAgents todos (plain strings) become completed_tasks dicts."""
    from app.services.orchestrator import _normalize_deep_agent_state

    result = _normalize_deep_agent_state({"todos": ["search web", "write report"]})
    assert len(result["completed_tasks"]) == 2
    assert result["completed_tasks"][0]["task"] == "search web"
    assert result["completed_tasks"][0]["status"] == "done"
    assert result["completed_tasks"][1]["task"] == "write report"
    assert result["todo_list"] == []


def test_normalize_deep_agent_state_splits_by_status():
    """DeepAgents todos with mixed status are split into todo_list and completed_tasks."""
    from app.services.orchestrator import _normalize_deep_agent_state

    result = _normalize_deep_agent_state({"todos": [
        {"content": "search web", "status": "done"},
        {"content": "write report", "status": "pending"},
        {"content": "review code", "status": "in_progress"},
        {"content": "summarize", "status": "completed"},  # DeepAgents uses "completed"
    ]})
    # "done" and "completed" both go to completed_tasks
    assert len(result["completed_tasks"]) == 2
    assert result["completed_tasks"][0]["task"] == "search web"
    assert result["completed_tasks"][1]["task"] == "summarize"
    assert result["completed_tasks"][1]["status"] == "done"  # normalized
    assert len(result["todo_list"]) == 2
    assert result["todo_list"][0]["task"] == "write report"
    assert result["todo_list"][0]["status"] == "pending"
    assert result["todo_list"][1]["task"] == "review code"
    assert result["todo_list"][1]["status"] == "in_progress"


def test_normalize_deep_agent_state_maps_content_to_task():
    """DeepAgents 'content' key is mapped to Axolotl 'task' key."""
    from app.services.orchestrator import _normalize_deep_agent_state

    result = _normalize_deep_agent_state({"todos": [
        {"content": "do research", "status": "pending"},
    ]})
    assert result["todo_list"][0]["task"] == "do research"
    assert "content" not in result["todo_list"][0]


def test_normalize_deep_agent_state_preserves_axolotl_dicts():
    """If todos already use Axolotl 'task' key, pass them through correctly."""
    from app.services.orchestrator import _normalize_deep_agent_state

    todo = {"id": "1", "task": "test", "status": "done", "agent": "x"}
    result = _normalize_deep_agent_state({"todos": [todo]})
    assert result["completed_tasks"][0]["task"] == "test"
    assert result["completed_tasks"][0]["id"] == "1"


def test_normalize_deep_agent_state_files_as_plain_strings():
    """DeepAgents files dict {path: str} becomes list[dict]."""
    from app.services.orchestrator import _normalize_deep_agent_state

    result = _normalize_deep_agent_state({
        "files": {"report.md": "# Report\nDone.", "data.csv": "a,b\n1,2"},
    })
    assert len(result["files"]) == 2
    names = {f["name"] for f in result["files"]}
    assert names == {"report.md", "data.csv"}
    assert result["files"][0]["content"] == "# Report\nDone."
    assert result["files"][0]["agent"] == "deep-agent"


def test_normalize_deep_agent_state_files_as_filedata():
    """DeepAgents FileData entries {content: [lines], ...} are joined into text."""
    from app.services.orchestrator import _normalize_deep_agent_state

    result = _normalize_deep_agent_state({
        "files": {
            "report.md": {
                "content": ["# Report", "Line 1", "Line 2"],
                "created_at": "2026-01-01",
                "updated_at": "2026-01-02",
            },
        },
    })
    assert len(result["files"]) == 1
    assert result["files"][0]["name"] == "report.md"
    assert result["files"][0]["content"] == "# Report\nLine 1\nLine 2"


def test_normalize_deep_agent_state_files_as_list():
    """If files are already a list, pass them through."""
    from app.services.orchestrator import _normalize_deep_agent_state

    files = [{"name": "f.txt", "content": "hello", "agent": "x"}]
    result = _normalize_deep_agent_state({"files": files})
    assert result["files"] == files


# ---------------------------------------------------------------------------
# orchestrate_project_chat with deep_agent mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_agent_mode_builds_deep_agent():
    """orchestration_mode='deep_agent' uses build_deep_agent."""
    project = make_project(
        orchestration_mode="deep_agent",
        planner_prompt="Plan things",
        model="openai:gpt-4.1-mini",
    )
    planner = make_agent(
        name="planner",
        is_planner=True,
        prompt="Supervise",
        model="openai:gpt-4.1",
        skills=[],
    )
    worker = make_agent(
        name="researcher",
        is_planner=False,
        prompt="Research things deeply.",
        model=None,
        description="A deep researcher",
        skills=[],
    )
    agents = [planner, worker]

    mock_deep_agent = MagicMock()
    mock_deep_agent.ainvoke = AsyncMock(return_value=_make_result("deep result"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.build_deep_agent",
            return_value=mock_deep_agent,
        ) as mock_build,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        result = await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="research this",
            thread_id="t-deep-1",
        )

    mock_build.assert_called_once()
    build_kwargs = mock_build.call_args.kwargs
    assert build_kwargs["model"] == "openai:gpt-4.1"
    assert "Supervise" in build_kwargs["system_prompt"]
    assert len(build_kwargs["subagents"]) == 1
    assert build_kwargs["subagents"][0]["name"] == "researcher"
    assert build_kwargs["checkpointer"] is None  # no checkpointer passed
    assert result.content == "deep result"


@pytest.mark.asyncio
async def test_deep_agent_skills_resolved_as_tools():
    """Axolotl skills are resolved via resolve_agent_tools, not DeepAgents skills system."""
    project = make_project(
        orchestration_mode="deep_agent",
        model="openai:gpt-4.1-mini",
    )
    worker = make_agent(name="w", is_planner=False, prompt="Work.", skills=[])
    agents = [worker]

    mock_deep_agent = MagicMock()
    mock_deep_agent.ainvoke = AsyncMock(return_value=_make_result("ok"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.build_deep_agent",
            return_value=mock_deep_agent,
        ) as mock_build,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project, agents=agents, message="go", thread_id="t-skills",
        )

    # skills should NOT be passed to build_deep_agent (resolved as tools instead)
    build_kwargs = mock_build.call_args.kwargs
    assert "skills" not in build_kwargs


@pytest.mark.asyncio
async def test_deep_agent_mode_no_workers_falls_back_to_simple():
    """deep_agent mode with no workers falls back to simple agent."""
    project = make_project(
        orchestration_mode="deep_agent",
        planner_prompt="Plan",
        model="openai:gpt-4.1-mini",
    )

    mock_simple_agent = MagicMock()
    mock_simple_agent.ainvoke = AsyncMock(return_value=_make_result("simple fallback"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.core.langgraph.graphs.simple_agent.build_simple_agent",
            return_value=mock_simple_agent,
        ) as mock_build,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        result = await orchestrate_project_chat(
            project=project,
            agents=[],
            message="hi",
            thread_id="t-deep-4",
        )

    mock_build.assert_called_once()
    assert result.content == "simple fallback"


@pytest.mark.asyncio
async def test_deep_agent_worker_uses_project_model_fallback():
    """Worker without model uses project.model in subagent config."""
    project = make_project(
        orchestration_mode="deep_agent",
        model="openai:gpt-4.1-mini",
    )
    worker = make_agent(
        name="worker",
        is_planner=False,
        prompt="Work.",
        model=None,
        description="A worker",
        skills=[],
    )
    agents = [worker]

    mock_deep_agent = MagicMock()
    mock_deep_agent.ainvoke = AsyncMock(return_value=_make_result("ok"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.build_deep_agent",
            return_value=mock_deep_agent,
        ) as mock_build,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-deep-5",
        )

    subagents = mock_build.call_args.kwargs["subagents"]
    assert subagents[0]["model"] == "openai:gpt-4.1-mini"


@pytest.mark.asyncio
async def test_deep_agent_checkpointer_forwarded():
    """Checkpointer is passed through to build_deep_agent."""
    project = make_project(orchestration_mode="deep_agent", model="openai:gpt-4.1-mini")
    worker = make_agent(name="w", is_planner=False, prompt="Work.", skills=[])
    agents = [worker]
    mock_checkpointer = MagicMock(name="redis-checkpointer")

    mock_deep_agent = MagicMock()
    mock_deep_agent.ainvoke = AsyncMock(return_value=_make_result("checked"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.build_deep_agent",
            return_value=mock_deep_agent,
        ) as mock_build,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-deep-ck",
            checkpointer=mock_checkpointer,
        )

    assert mock_build.call_args.kwargs["checkpointer"] is mock_checkpointer


@pytest.mark.asyncio
async def test_deep_agent_state_normalized_in_result():
    """DeepAgents state (todos, files dict) is normalized to ProjectState shape."""
    project = make_project(orchestration_mode="deep_agent", model="openai:gpt-4.1-mini")
    worker = make_agent(name="w", is_planner=False, prompt="Work.", skills=[])
    agents = [worker]

    deep_state = _make_result("done", extras={
        "todos": [
            {"content": "searched the web", "status": "done"},
            {"content": "write report", "status": "pending"},
        ],
        "files": {"report.md": "# Report", "data.csv": "a,b"},
    })
    mock_deep_agent = MagicMock()
    mock_deep_agent.ainvoke = AsyncMock(return_value=deep_state)

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.build_deep_agent",
            return_value=mock_deep_agent,
        ),
    ):
        from app.services.orchestrator import orchestrate_project_chat

        result = await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-deep-norm",
        )

    # State should be normalized: split by status, content->task, files as list
    assert result.content == "done"
    assert len(result.completed_tasks) == 1
    assert result.completed_tasks[0]["task"] == "searched the web"
    assert len(result.todo_list) == 1
    assert result.todo_list[0]["task"] == "write report"
    assert result.todo_list[0]["status"] == "pending"
    assert len(result.files) == 2
    assert all(isinstance(f, dict) and "name" in f for f in result.files)


# ---------------------------------------------------------------------------
# Explicit simple mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_mode_uses_simple_agent():
    """orchestration_mode='simple' always uses build_simple_agent."""
    project = make_project(
        orchestration_mode="simple",
        planner_prompt="Simple prompt",
        model="openai:gpt-4.1-mini",
    )
    worker = make_agent(name="w", is_planner=False, prompt="Work.")
    agents = [worker]

    mock_simple_agent = MagicMock()
    mock_simple_agent.ainvoke = AsyncMock(return_value=_make_result("simple mode"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.core.langgraph.graphs.simple_agent.build_simple_agent",
            return_value=mock_simple_agent,
        ),
    ):
        from app.services.orchestrator import orchestrate_project_chat

        result = await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="hello",
            thread_id="t-simple-1",
        )

    assert result.content == "simple mode"


# ---------------------------------------------------------------------------
# Planner model fallback chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_with_null_model_falls_back_to_project_model():
    """When planner.model is None, use model_override or project.model."""
    project = make_project(
        orchestration_mode="deep_agent",
        model="openai:gpt-4.1-mini",
    )
    planner = make_agent(
        name="planner",
        is_planner=True,
        prompt="Plan.",
        model=None,  # NULL model
        skills=[],
    )
    worker = make_agent(name="w", is_planner=False, prompt="Work.", skills=[])
    agents = [planner, worker]

    mock_deep_agent = MagicMock()
    mock_deep_agent.ainvoke = AsyncMock(return_value=_make_result("ok"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.build_deep_agent",
            return_value=mock_deep_agent,
        ) as mock_build,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-model-fallback",
        )

    # Should use project.model, NOT None
    assert mock_build.call_args.kwargs["model"] == "openai:gpt-4.1-mini"


@pytest.mark.asyncio
async def test_planner_null_model_with_override_uses_override():
    """When planner.model is None but model_override is set, use the override."""
    project = make_project(
        orchestration_mode="deep_agent",
        model="openai:gpt-4.1-mini",
    )
    planner = make_agent(
        name="planner",
        is_planner=True,
        prompt="Plan.",
        model=None,
        skills=[],
    )
    worker = make_agent(name="w", is_planner=False, prompt="Work.", skills=[])
    agents = [planner, worker]

    mock_deep_agent = MagicMock()
    mock_deep_agent.ainvoke = AsyncMock(return_value=_make_result("ok"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.build_deep_agent",
            return_value=mock_deep_agent,
        ) as mock_build,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-model-override",
            model_override="anthropic:claude-sonnet-4",
        )

    assert mock_build.call_args.kwargs["model"] == "anthropic:claude-sonnet-4"
