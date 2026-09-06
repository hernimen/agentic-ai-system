from lims_agent.core.config import Settings


def test_settings_does_not_require_langsmith_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.langsmith_api_key == ""
