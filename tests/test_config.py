from __future__ import annotations

from pathlib import Path

import pytest

from ragcite.config import Config


def test_toml_then_env_then_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    toml = tmp_path / "ragcite.toml"
    toml.write_text('[ragcite]\ntop_k = 3\nmodel = "claude-opus-5"\nrerank = true\n')

    monkeypatch.setenv("RAGCITE_TOP_K", "4")
    cfg = Config.load(toml)
    assert cfg.top_k == 4  # env beats the file
    assert cfg.rerank is True  # booleans survive the round trip
    assert cfg.model == "claude-opus-5"

    cfg = Config.load(toml, top_k=5)
    assert cfg.top_k == 5  # explicit override beats both


def test_unknown_keys_are_rejected_loudly(tmp_path: Path):
    toml = tmp_path / "ragcite.toml"
    toml.write_text("[ragcite]\ntyop_k = 3\n")
    with pytest.raises(ValueError, match="unknown config key"):
        Config.load(toml)


def test_invalid_combinations_are_caught():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Config(chunk_size=128, chunk_overlap=128).validate()
    with pytest.raises(ValueError, match="retrieval_mode"):
        Config(retrieval_mode="magic").validate()
    with pytest.raises(ValueError, match="candidate_k"):
        Config(top_k=50, candidate_k=10).validate()
