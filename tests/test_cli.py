import json

from kga.cli import main


def test_dry_run_loads_corpus_without_api_configuration(tmp_path, capsys, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.md").write_text("# Internal service\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KGA_MODEL", raising=False)

    result = main(
        [
            "--corpus",
            str(corpus),
            "--output",
            str(tmp_path / "unused"),
            "--max-documents",
            "1",
            "--max-api-calls",
            "1",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output == {
        "dry_run": True,
        "documents": 1,
        "passages": 1,
        "would_exceed_api_call_limit": False,
    }
    assert not (tmp_path / "unused").exists()


def test_cli_reports_missing_model_without_traceback(tmp_path, capsys, monkeypatch):
    source = tmp_path / "one.txt"
    source.write_text("content", encoding="utf-8")
    monkeypatch.delenv("KGA_MODEL", raising=False)

    result = main(["--corpus", str(source)])

    assert result == 1
    assert "A model is required" in capsys.readouterr().err
