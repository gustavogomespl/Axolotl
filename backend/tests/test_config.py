"""Tests for app.config.Settings."""

from app.config import Settings, settings


class TestSettingsDefaults:
    """Verify that Settings provides correct defaults when no env vars override."""

    def test_database_url_default(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        s = Settings()
        assert s.database_url == "postgresql+asyncpg://axolotl:axolotl@localhost:5432/axolotl"

    def test_redis_url_default(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        s = Settings()
        assert s.redis_url == "redis://localhost:6379/0"

    def test_default_model(self):
        s = Settings()
        assert s.default_model == "openai:gpt-4.1-mini"

    def test_default_temperature(self):
        s = Settings()
        assert s.default_temperature == 0.0

    def test_chroma_host_default(self):
        s = Settings()
        assert s.chroma_host == "localhost"

    def test_chroma_port_default(self):
        s = Settings()
        assert s.chroma_port == 8001

    def test_app_env_default(self):
        s = Settings()
        assert s.app_env == "development"

    def test_api_port_default(self):
        s = Settings()
        assert s.api_port == 8000

    def test_admin_port_default(self):
        s = Settings()
        assert s.admin_port == 8080

    def test_langsmith_tracing_default_false(self):
        s = Settings()
        assert s.langsmith_tracing is False

    def test_langsmith_api_key_default_none(self):
        s = Settings()
        assert s.langsmith_api_key is None

    def test_langsmith_project_default(self):
        s = Settings()
        assert s.langsmith_project == "axolotl"


class TestSettingsSingleton:
    def test_settings_instance_exists(self):
        assert settings is not None

    def test_settings_is_settings_instance(self):
        assert isinstance(settings, Settings)


class TestSettingsOverrides:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_MODEL", "anthropic:claude-sonnet-4-6")
        monkeypatch.setenv("DEFAULT_TEMPERATURE", "0.7")
        monkeypatch.setenv("APP_ENV", "production")

        s = Settings()
        assert s.default_model == "anthropic:claude-sonnet-4-6"
        assert s.default_temperature == 0.7
        assert s.app_env == "production"
