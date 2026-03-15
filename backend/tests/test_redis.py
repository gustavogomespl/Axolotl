"""Tests for app.core.redis.RedisManager."""

from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestRedisManagerInit:
    def test_init_uses_settings_redis_url_by_default(self):
        with patch("app.core.redis.settings") as mock_settings:
            mock_settings.redis_url = "redis://default:6379/0"
            from importlib import reload

            import app.core.redis as redis_mod

            reload(redis_mod)
            manager = redis_mod.RedisManager()

        assert manager.redis_url == "redis://default:6379/0"
        assert manager._saver is None

    def test_init_accepts_custom_redis_url(self):
        with patch("app.core.redis.settings") as mock_settings:
            mock_settings.redis_url = "redis://default:6379/0"
            from importlib import reload

            import app.core.redis as redis_mod

            reload(redis_mod)
            manager = redis_mod.RedisManager(redis_url="redis://custom:6380/1")

        assert manager.redis_url == "redis://custom:6380/1"


# ---------------------------------------------------------------------------
# get_saver
# ---------------------------------------------------------------------------


class TestGetSaver:
    @pytest.mark.asyncio
    async def test_get_saver_creates_saver_on_first_call(self):
        mock_saver = AsyncMock()

        with (
            patch("app.core.redis.settings") as mock_settings,
            patch(
                "langgraph.checkpoint.redis.aio.AsyncRedisSaver.from_conn_string",
                return_value=mock_saver,
            ) as mock_from_conn,
        ):
            mock_settings.redis_url = "redis://localhost:6379/0"
            from importlib import reload

            import app.core.redis as redis_mod

            reload(redis_mod)
            manager = redis_mod.RedisManager()
            saver = await manager.get_saver()

        mock_from_conn.assert_called_once_with("redis://localhost:6379/0")
        mock_saver.asetup.assert_awaited_once()
        assert saver is mock_saver

    @pytest.mark.asyncio
    async def test_get_saver_returns_same_instance_on_second_call(self):
        mock_saver = AsyncMock()

        with (
            patch("app.core.redis.settings") as mock_settings,
            patch(
                "langgraph.checkpoint.redis.aio.AsyncRedisSaver.from_conn_string",
                return_value=mock_saver,
            ),
        ):
            mock_settings.redis_url = "redis://localhost:6379/0"
            from importlib import reload

            import app.core.redis as redis_mod

            reload(redis_mod)
            manager = redis_mod.RedisManager()

            saver1 = await manager.get_saver()
            saver2 = await manager.get_saver()

        assert saver1 is saver2
        # from_conn_string should only be called once (lazy init)
        assert mock_saver.asetup.await_count == 1


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_calls_aclose_on_saver(self):
        mock_saver = AsyncMock()

        with (
            patch("app.core.redis.settings") as mock_settings,
            patch(
                "langgraph.checkpoint.redis.aio.AsyncRedisSaver.from_conn_string",
                return_value=mock_saver,
            ),
        ):
            mock_settings.redis_url = "redis://localhost:6379/0"
            from importlib import reload

            import app.core.redis as redis_mod

            reload(redis_mod)
            manager = redis_mod.RedisManager()

            await manager.get_saver()
            await manager.close()

        mock_saver.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_resets_saver_to_none(self):
        mock_saver = AsyncMock()

        with (
            patch("app.core.redis.settings") as mock_settings,
            patch(
                "langgraph.checkpoint.redis.aio.AsyncRedisSaver.from_conn_string",
                return_value=mock_saver,
            ),
        ):
            mock_settings.redis_url = "redis://localhost:6379/0"
            from importlib import reload

            import app.core.redis as redis_mod

            reload(redis_mod)
            manager = redis_mod.RedisManager()

            await manager.get_saver()
            await manager.close()

        assert manager._saver is None

    @pytest.mark.asyncio
    async def test_close_without_saver_is_noop(self):
        with patch("app.core.redis.settings") as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379/0"
            from importlib import reload

            import app.core.redis as redis_mod

            reload(redis_mod)
            manager = redis_mod.RedisManager()

            # Should not raise
            await manager.close()

        assert manager._saver is None

    @pytest.mark.asyncio
    async def test_close_handles_aclose_exception(self):
        mock_saver = AsyncMock()
        mock_saver.aclose.side_effect = RuntimeError("connection lost")

        with (
            patch("app.core.redis.settings") as mock_settings,
            patch(
                "langgraph.checkpoint.redis.aio.AsyncRedisSaver.from_conn_string",
                return_value=mock_saver,
            ),
        ):
            mock_settings.redis_url = "redis://localhost:6379/0"
            from importlib import reload

            import app.core.redis as redis_mod

            reload(redis_mod)
            manager = redis_mod.RedisManager()

            await manager.get_saver()
            # Should swallow the exception
            await manager.close()

        assert manager._saver is None
