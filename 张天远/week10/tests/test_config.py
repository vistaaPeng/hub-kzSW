"""
Tests for runtime configuration helpers.
"""

from src import config


def test_load_dotenv_sets_missing_values_without_overriding(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RAG_TEST_VALUE=from-file\n"
        "RAG_KEEP_VALUE=from-file\n"
        "# comment\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_KEEP_VALUE", "from-env")
    monkeypatch.delenv("RAG_TEST_VALUE", raising=False)

    config.load_dotenv(env_file)

    assert config.env_str("RAG_TEST_VALUE", "") == "from-file"
    assert config.env_str("RAG_KEEP_VALUE", "") == "from-env"


def test_env_bool_parses_common_values(monkeypatch):
    monkeypatch.setenv("RAG_BOOL_TRUE", "yes")
    monkeypatch.setenv("RAG_BOOL_FALSE", "0")

    assert config.env_bool("RAG_BOOL_TRUE") is True
    assert config.env_bool("RAG_BOOL_FALSE", True) is False
    assert config.env_bool("RAG_BOOL_MISSING", True) is True


def test_env_str_uses_default_for_missing_or_empty(monkeypatch):
    monkeypatch.delenv("RAG_STR_MISSING", raising=False)
    monkeypatch.setenv("RAG_STR_EMPTY", "")

    assert config.env_str("RAG_STR_MISSING", "default") == "default"
    assert config.env_str("RAG_STR_EMPTY", "default") == "default"
