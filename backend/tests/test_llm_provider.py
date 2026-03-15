"""Tests for app.core.llm.provider.get_chat_model."""

from unittest.mock import MagicMock, patch

from app.core.llm.provider import get_chat_model


class TestGetChatModel:
    @patch("app.core.llm.provider.init_chat_model")
    def test_uses_default_model_when_none_passed(self, mock_init):
        mock_init.return_value = MagicMock()

        with patch("app.core.llm.provider.settings") as mock_settings:
            mock_settings.default_model = "openai:gpt-4.1-mini"
            mock_settings.default_temperature = 0.0
            get_chat_model()

        mock_init.assert_called_once_with(
            model="openai:gpt-4.1-mini",
            temperature=0.0,
        )

    @patch("app.core.llm.provider.init_chat_model")
    def test_uses_provided_model(self, mock_init):
        mock_init.return_value = MagicMock()

        with patch("app.core.llm.provider.settings") as mock_settings:
            mock_settings.default_model = "openai:gpt-4.1-mini"
            mock_settings.default_temperature = 0.0
            get_chat_model(model="anthropic:claude-sonnet-4-6")

        mock_init.assert_called_once_with(
            model="anthropic:claude-sonnet-4-6",
            temperature=0.0,
        )

    @patch("app.core.llm.provider.init_chat_model")
    def test_passes_temperature_correctly(self, mock_init):
        mock_init.return_value = MagicMock()

        with patch("app.core.llm.provider.settings") as mock_settings:
            mock_settings.default_model = "openai:gpt-4.1-mini"
            mock_settings.default_temperature = 0.0
            get_chat_model(temperature=0.9)

        mock_init.assert_called_once_with(
            model="openai:gpt-4.1-mini",
            temperature=0.9,
        )

    @patch("app.core.llm.provider.init_chat_model")
    def test_uses_default_temperature_when_none_passed(self, mock_init):
        mock_init.return_value = MagicMock()

        with patch("app.core.llm.provider.settings") as mock_settings:
            mock_settings.default_model = "openai:gpt-4.1-mini"
            mock_settings.default_temperature = 0.5
            get_chat_model()

        mock_init.assert_called_once_with(
            model="openai:gpt-4.1-mini",
            temperature=0.5,
        )

    @patch("app.core.llm.provider.init_chat_model")
    def test_returns_chat_model_instance(self, mock_init):
        sentinel = MagicMock()
        mock_init.return_value = sentinel

        with patch("app.core.llm.provider.settings") as mock_settings:
            mock_settings.default_model = "openai:gpt-4.1-mini"
            mock_settings.default_temperature = 0.0
            result = get_chat_model()

        assert result is sentinel

    @patch("app.core.llm.provider.init_chat_model")
    def test_model_and_temperature_together(self, mock_init):
        mock_init.return_value = MagicMock()

        with patch("app.core.llm.provider.settings") as mock_settings:
            mock_settings.default_model = "openai:gpt-4.1-mini"
            mock_settings.default_temperature = 0.0
            get_chat_model(model="ollama:llama3", temperature=0.3)

        mock_init.assert_called_once_with(
            model="ollama:llama3",
            temperature=0.3,
        )

    @patch("app.core.llm.provider.init_chat_model")
    def test_temperature_zero_is_not_treated_as_none(self, mock_init):
        """Ensure temperature=0.0 is passed as 0.0, not replaced by the default."""
        mock_init.return_value = MagicMock()

        with patch("app.core.llm.provider.settings") as mock_settings:
            mock_settings.default_model = "openai:gpt-4.1-mini"
            mock_settings.default_temperature = 0.7
            get_chat_model(temperature=0.0)

        mock_init.assert_called_once_with(
            model="openai:gpt-4.1-mini",
            temperature=0.0,
        )
