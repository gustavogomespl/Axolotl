"""Tests for app.services.orchestrator.orchestrate_project_chat."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.factories import make_agent, make_project


def _make_result(content: str):
    """Build a mock result dict that looks like {'messages': [msg]}."""
    msg = MagicMock()
    msg.content = content
    return {"messages": [msg]}


# ---------------------------------------------------------------------------
# No workers  ->  fallback to build_simple_agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_workers_no_planner_uses_project_planner_prompt():
    """No workers and no planner -> fallback to simple_agent with project.planner_prompt."""
    project = make_project(planner_prompt="Project system prompt", model="openai:gpt-4.1-mini")
    agents: list = []  # no agents at all

    mock_simple_agent = MagicMock()
    mock_simple_agent.ainvoke = AsyncMock(return_value=_make_result("hello from simple"))

    with (
        patch(
            "app.services.orchestrator._get_vector_store",
            return_value=None,
        ),
        patch(
            "app.core.langgraph.graphs.simple_agent.build_simple_agent",
            return_value=mock_simple_agent,
        ) as mock_build,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        result = await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="hi",
            thread_id="t-1",
        )

    mock_build.assert_called_once_with(
        model_name=project.model,
        system_prompt="Project system prompt",
    )
    assert result == "hello from simple"


@pytest.mark.asyncio
async def test_no_workers_has_planner_uses_planner_prompt():
    """No workers but has planner -> fallback to simple_agent with planner.prompt."""
    project = make_project(planner_prompt="Project level prompt", model="openai:gpt-4.1-mini")
    planner = make_agent(
        name="planner",
        is_planner=True,
        prompt="Planner agent prompt",
        model="openai:gpt-4.1",
    )
    agents = [planner]  # only planner, no workers

    mock_simple_agent = MagicMock()
    mock_simple_agent.ainvoke = AsyncMock(return_value=_make_result("planner reply"))

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
            agents=agents,
            message="plan something",
            thread_id="t-2",
        )

    # planner.prompt is used, not project.planner_prompt
    mock_build.assert_called_once_with(
        model_name="openai:gpt-4.1",
        system_prompt="Planner agent prompt",
    )
    assert result == "planner reply"


@pytest.mark.asyncio
async def test_no_workers_no_planner_no_planner_prompt_falls_back():
    """No workers, no planner, planner_prompt is None -> default assistant prompt."""
    project = make_project(planner_prompt=None, model="openai:gpt-4.1-mini")

    mock_simple_agent = MagicMock()
    mock_simple_agent.ainvoke = AsyncMock(return_value=_make_result("default"))

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
            message="hello",
            thread_id="t-3",
        )

    mock_build.assert_called_once_with(
        model_name="openai:gpt-4.1-mini",
        system_prompt="You are a helpful assistant.",
    )
    assert result == "default"


@pytest.mark.asyncio
async def test_no_workers_model_override_used():
    """When no planner and model_override is given, it is used."""
    project = make_project(planner_prompt=None, model="openai:gpt-4.1-mini")

    mock_simple_agent = MagicMock()
    mock_simple_agent.ainvoke = AsyncMock(return_value=_make_result("overridden"))

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
            message="hello",
            thread_id="t-4",
            model_override="anthropic:claude-sonnet-4",
        )

    # model_override takes precedence over project.model when there is no planner
    mock_build.assert_called_once_with(
        model_name="anthropic:claude-sonnet-4",
        system_prompt="You are a helpful assistant.",
    )
    assert result == "overridden"


# ---------------------------------------------------------------------------
# Has workers + planner  ->  supervisor pattern
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workers_and_planner_builds_supervisor():
    """Has workers + planner -> builds supervisor with create_supervisor."""
    project = make_project(planner_prompt="Project system prompt", model="openai:gpt-4.1-mini")
    planner = make_agent(
        name="planner",
        is_planner=True,
        prompt="Supervise the workers.",
        model="openai:gpt-4.1",
    )
    worker = make_agent(name="researcher", is_planner=False, model=None, prompt="Research things.")
    agents = [planner, worker]

    mock_worker_graph = MagicMock(name="worker-graph")
    mock_supervisor_model = MagicMock(name="supervisor-model")
    mock_supervisor_builder = MagicMock(name="supervisor-builder")
    mock_compiled_graph = MagicMock(name="compiled-graph")
    mock_compiled_graph.ainvoke = AsyncMock(return_value=_make_result("supervisor done"))
    mock_supervisor_builder.compile.return_value = mock_compiled_graph

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.get_chat_model",
            side_effect=lambda model: mock_supervisor_model
            if model == "openai:gpt-4.1"
            else MagicMock(),
        ) as mock_gcm,
        patch(
            "app.services.orchestrator.create_react_agent",
            return_value=mock_worker_graph,
        ) as mock_react,
        patch(
            "app.services.orchestrator.create_supervisor",
            return_value=mock_supervisor_builder,
        ) as mock_sup,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        result = await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="do research",
            thread_id="t-5",
        )

    # Worker react agent was created
    mock_react.assert_called_once_with(
        model=mock_gcm.return_value,  # worker model call
        tools=[],
        name="researcher",
        prompt="Research things.",
    )

    # Supervisor was created with planner.prompt
    mock_sup.assert_called_once_with(
        agents=[mock_worker_graph],
        model=mock_supervisor_model,
        prompt="Supervise the workers.",
    )

    # Compiled without checkpointer
    mock_supervisor_builder.compile.assert_called_once_with()

    assert result == "supervisor done"


@pytest.mark.asyncio
async def test_workers_no_planner_uses_project_planner_prompt():
    """Has workers, no planner -> uses project.planner_prompt for supervisor."""
    project = make_project(planner_prompt="Coordinate tasks", model="openai:gpt-4.1-mini")
    worker = make_agent(name="coder", is_planner=False, model=None, prompt="Write code.")
    agents = [worker]

    mock_supervisor_builder = MagicMock()
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=_make_result("coded"))
    mock_supervisor_builder.compile.return_value = mock_compiled

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("app.services.orchestrator.get_chat_model", return_value=MagicMock()),
        patch("app.services.orchestrator.create_react_agent", return_value=MagicMock()),
        patch(
            "app.services.orchestrator.create_supervisor",
            return_value=mock_supervisor_builder,
        ) as mock_sup,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        result = await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="build it",
            thread_id="t-6",
        )

    # prompt comes from project.planner_prompt
    assert mock_sup.call_args.kwargs["prompt"] == "Coordinate tasks"
    assert result == "coded"


@pytest.mark.asyncio
async def test_workers_no_planner_model_override():
    """model_override used for supervisor when no planner model is set."""
    project = make_project(planner_prompt=None, model="openai:gpt-4.1-mini")
    worker = make_agent(name="w", is_planner=False, model=None, prompt="Work.")
    agents = [worker]

    mock_sup_builder = MagicMock()
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=_make_result("ok"))
    mock_sup_builder.compile.return_value = mock_compiled

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.orchestrator.get_chat_model",
            return_value=MagicMock(),
        ) as mock_gcm,
        patch("app.services.orchestrator.create_react_agent", return_value=MagicMock()),
        patch(
            "app.services.orchestrator.create_supervisor",
            return_value=mock_sup_builder,
        ),
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-7",
            model_override="anthropic:claude-sonnet-4",
        )

    # get_chat_model should be called with the override for the supervisor
    # (once for worker, once for supervisor)
    calls = mock_gcm.call_args_list
    supervisor_call = calls[-1]  # last call is for the supervisor model
    assert (
        supervisor_call.kwargs.get("model") == "anthropic:claude-sonnet-4"
        or supervisor_call[1].get("model") == "anthropic:claude-sonnet-4"
    )


@pytest.mark.asyncio
async def test_checkpointer_passed_to_compile():
    """checkpointer kwarg is forwarded to supervisor.compile."""
    project = make_project(model="openai:gpt-4.1-mini")
    worker = make_agent(name="w", is_planner=False, model=None, prompt="Go.")
    agents = [worker]

    mock_checkpointer = MagicMock(name="checkpointer")

    mock_sup_builder = MagicMock()
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=_make_result("checked"))
    mock_sup_builder.compile.return_value = mock_compiled

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("app.services.orchestrator.get_chat_model", return_value=MagicMock()),
        patch("app.services.orchestrator.create_react_agent", return_value=MagicMock()),
        patch(
            "app.services.orchestrator.create_supervisor",
            return_value=mock_sup_builder,
        ),
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-8",
            checkpointer=mock_checkpointer,
        )

    mock_sup_builder.compile.assert_called_once_with(checkpointer=mock_checkpointer)


@pytest.mark.asyncio
async def test_no_checkpointer_compile_called_without_it():
    """Without a checkpointer, compile() is called with no extra kwargs."""
    project = make_project(model="openai:gpt-4.1-mini")
    worker = make_agent(name="w", is_planner=False, prompt="Work.")
    agents = [worker]

    mock_sup_builder = MagicMock()
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=_make_result("done"))
    mock_sup_builder.compile.return_value = mock_compiled

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("app.services.orchestrator.get_chat_model", return_value=MagicMock()),
        patch("app.services.orchestrator.create_react_agent", return_value=MagicMock()),
        patch(
            "app.services.orchestrator.create_supervisor",
            return_value=mock_sup_builder,
        ),
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-9",
        )

    mock_sup_builder.compile.assert_called_once_with()


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_last_message_content():
    """The function returns result['messages'][-1].content."""
    project = make_project(model="openai:gpt-4.1-mini")

    msg1 = MagicMock()
    msg1.content = "first"
    msg2 = MagicMock()
    msg2.content = "second"
    msg3 = MagicMock()
    msg3.content = "third and final"

    mock_simple_agent = MagicMock()
    mock_simple_agent.ainvoke = AsyncMock(return_value={"messages": [msg1, msg2, msg3]})

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
            agents=[],
            message="hi",
            thread_id="t-10",
        )

    assert result == "third and final"


@pytest.mark.asyncio
async def test_thread_id_passed_in_config():
    """thread_id is passed in the config to ainvoke."""
    project = make_project(model="openai:gpt-4.1-mini")

    mock_simple_agent = MagicMock()
    mock_simple_agent.ainvoke = AsyncMock(return_value=_make_result("ok"))

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.core.langgraph.graphs.simple_agent.build_simple_agent",
            return_value=mock_simple_agent,
        ),
    ):
        from app.services.orchestrator import orchestrate_project_chat

        await orchestrate_project_chat(
            project=project,
            agents=[],
            message="hi",
            thread_id="my-thread-123",
        )

    call_kwargs = mock_simple_agent.ainvoke.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config["configurable"]["thread_id"] == "my-thread-123"


@pytest.mark.asyncio
async def test_multiple_workers_each_gets_react_agent():
    """Each worker agent gets its own react agent with resolved tools."""
    project = make_project(model="openai:gpt-4.1-mini")
    w1 = make_agent(name="worker-1", is_planner=False, model="openai:gpt-4.1", prompt="W1")
    w2 = make_agent(name="worker-2", is_planner=False, model=None, prompt="W2")
    agents = [w1, w2]

    mock_sup_builder = MagicMock()
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=_make_result("multi"))
    mock_sup_builder.compile.return_value = mock_compiled

    tool_a = MagicMock(name="tool-a")
    tool_b = MagicMock(name="tool-b")
    resolve_call_count = 0

    async def resolve_side_effect(agent, vs):
        nonlocal resolve_call_count
        resolve_call_count += 1
        if agent.name == "worker-1":
            return [tool_a]
        return [tool_b]

    with (
        patch("app.services.orchestrator._get_vector_store", return_value=None),
        patch(
            "app.services.orchestrator.resolve_agent_tools",
            side_effect=resolve_side_effect,
        ),
        patch("app.services.orchestrator.get_chat_model", return_value=MagicMock()),
        patch(
            "app.services.orchestrator.create_react_agent",
            return_value=MagicMock(),
        ) as mock_react,
        patch(
            "app.services.orchestrator.create_supervisor",
            return_value=mock_sup_builder,
        ) as mock_sup,
    ):
        from app.services.orchestrator import orchestrate_project_chat

        result = await orchestrate_project_chat(
            project=project,
            agents=agents,
            message="go",
            thread_id="t-11",
        )

    assert resolve_call_count == 2
    assert mock_react.call_count == 2
    # supervisor receives a list of 2 worker graphs
    sup_call_agents = mock_sup.call_args.kwargs["agents"]
    assert len(sup_call_agents) == 2
    assert result == "multi"
