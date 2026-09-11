from core.config import Settings


def test_settings_does_not_require_langsmith_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.langsmith_api_key == ""


def test_settings_parses_sops_source_dirs_from_environment(monkeypatch):
    monkeypatch.setenv(
        "SOPS_SOURCE_DIRS",
        '["./data/sops", "./data/equipment_manuals"]',
    )

    settings = Settings(_env_file=None)

    assert settings.sops_source_dirs == ["./data/sops", "./data/equipment_manuals"]
