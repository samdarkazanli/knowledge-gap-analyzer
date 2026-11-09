import json

from kga.evaluation_cli import main
from test_evaluation import make_question


def test_evaluation_dry_run_needs_no_api_configuration(tmp_path, capsys, monkeypatch):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([make_question().model_dump(mode="json")]), encoding="utf-8"
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text("one line", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KGA_MODEL", raising=False)

    result = main(
        [
            "--questions",
            str(questions),
            "--corpus",
            str(corpus),
            "--max-questions",
            "1",
            "--max-api-calls",
            "1",
            "--context-window-tokens",
            "100000",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["dry_run"] is True
    assert output["questions"] == 1
    assert output["documents"] == 1
    assert output["would_exceed_context_window"] is False
    assert output["would_exceed_api_call_limit"] is False
    assert output["maximum_api_calls_with_retries"] == 2
    assert output["could_exceed_api_call_limit_with_retries"] is True


def test_evaluation_cli_reports_missing_model_without_traceback(
    tmp_path, capsys, monkeypatch
):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([make_question().model_dump(mode="json")]), encoding="utf-8"
    )
    corpus = tmp_path / "one.txt"
    corpus.write_text("one line", encoding="utf-8")
    monkeypatch.delenv("KGA_MODEL", raising=False)

    result = main(["--questions", str(questions), "--corpus", str(corpus)])

    assert result == 1
    assert "A model is required" in capsys.readouterr().err
