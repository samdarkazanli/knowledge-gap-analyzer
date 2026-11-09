import json

from kga.model_cli import main
from kga.models import ConceptType
from kga.question_generators import TemplateQuestionGenerator, generate_questions
from test_question_generators import make_concept


def write_inventories(tmp_path):
    concept = make_concept("Atlas", ConceptType.APPLICATION)
    questions = generate_questions([concept], TemplateQuestionGenerator())
    concepts_path = tmp_path / "concepts.json"
    questions_path = tmp_path / "questions.json"
    concepts_path.write_text(
        json.dumps([concept.model_dump(mode="json")]), encoding="utf-8"
    )
    questions_path.write_text(
        json.dumps([item.model_dump(mode="json") for item in questions]),
        encoding="utf-8",
    )
    return concepts_path, questions_path, questions


def test_model_cli_migrates_legacy_inventories_without_overwriting_them(
    tmp_path, capsys, monkeypatch
):
    concepts, questions, original_questions = write_inventories(tmp_path)
    original_content = questions.read_text(encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "formal"

    result = main(
        [
            "--concepts",
            str(concepts),
            "--questions",
            str(questions),
            "--output",
            str(output),
        ]
    )

    manifest = json.loads(capsys.readouterr().out)
    enriched = json.loads((output / "questions-enriched.json").read_text())
    assert result == 0
    assert manifest["validation_passed"] is True
    assert manifest["question_count"] == len(original_questions)
    assert manifest["requirement_count"] == len(original_questions)
    assert all(item["requirement_id"] for item in enriched)
    assert [item["question_id"] for item in enriched] == [
        item.question_id for item in original_questions
    ]
    assert questions.read_text(encoding="utf-8") == original_content


def test_model_cli_outputs_are_deterministic(tmp_path, capsys):
    concepts, questions, _ = write_inventories(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert main(["--concepts", str(concepts), "--questions", str(questions), "--output", str(first)]) == 0
    capsys.readouterr()
    assert main(["--concepts", str(concepts), "--questions", str(questions), "--output", str(second)]) == 0
    capsys.readouterr()

    assert {item.name for item in first.iterdir()} == {item.name for item in second.iterdir()}
    for first_file in first.iterdir():
        assert first_file.read_bytes() == (second / first_file.name).read_bytes()
