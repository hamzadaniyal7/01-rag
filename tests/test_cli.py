from __future__ import annotations

from pathlib import Path

import pytest

from ragcite.cli import main

CORPUS = str(Path(__file__).resolve().parents[1] / "corpus")


def test_dry_run_reports_retrieval_without_generating(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db = str(tmp_path / "index.db")
    common = ["--db-path", db, "--embedder", "hash:512"]
    assert main(["ingest", CORPUS, *common]) == 0
    capsys.readouterr()
    assert main(["ask", "refund timing", "--llm", "fake", "--dry-run", *common]) == 0
    out = capsys.readouterr().out
    assert "retrieved" in out
    assert "prompt tokens" in out


def test_ingest_search_ask_and_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    db = str(tmp_path / "index.db")
    common = ["--db-path", db, "--embedder", "hash:512"]

    assert main(["ingest", CORPUS, *common]) == 0
    assert "chunk(s) written" in capsys.readouterr().out

    assert main(["search", "refund processing time", *common]) == 0
    assert "refund" in capsys.readouterr().out.lower()

    assert main(["ask", "How long do refunds take?", "--llm", "fake", "--quiet", *common]) == 0
    out = capsys.readouterr().out
    assert "Sources:" in out
    assert "refund-policy.md" in out

    assert main(["stats", "--documents", *common]) == 0
    assert "documents" in capsys.readouterr().out


def test_mismatched_embedder_exits_with_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db = str(tmp_path / "index.db")
    assert main(["ingest", CORPUS, "--db-path", db, "--embedder", "hash:512"]) == 0
    capsys.readouterr()
    code = main(["search", "refund", "--db-path", db, "--embedder", "hash:256"])
    assert code == 2
    err = capsys.readouterr().err
    assert "hash:512" in err and "hash:256" in err
