import json

from kga.models import ConceptType
from kga.question_cli import main
from test_question_generators import make_concept


def test_question_cli_generates_reports_without_provider_configuration(
    tmp_path, capsys, monkeypatch
):
    concepts = [
        make_concept("Atlas", ConceptType.APPLICATION),
        make_concept("orders-primary", ConceptType.IDENTIFIER),
    ]
    inventory = tmp_path / "concepts.json"
    inventory.write_text(
        json.dumps([item.model_dump(mode="json") for item in concepts]),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "questions"

    result = main(["--concepts", str(inventory), "--output", str(output)])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "concepts_loaded": 2,
        "company_specific_concepts": 2,
        "questions_generated": 10,
    }
    assert (output / "questions.json").is_file()
    assert (output / "questions.csv").is_file()


def test_question_cli_reports_invalid_inventory_without_traceback(tmp_path, capsys):
    missing = tmp_path / "missing.json"

    result = main(["--concepts", str(missing)])

    assert result == 1
    assert "Could not read concept inventory" in capsys.readouterr().err
